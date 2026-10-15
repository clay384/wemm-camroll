"""Independent immutable history imports and run access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import threading

from .common import read_json, atomic_json, safe_id, sha256, now_iso, object_hash
from .metrics import summary

MINICPM_PROTOCOL = 'minicpm47-thinking-m4-off-v1'
MINICPM_SDK_PROTOCOL = 'minicpm47-sdk-defaults-m4-off-v1'
MINICPM_INTERFACE_PROTOCOL = 'minicpm47-sdk-defaults-m4-off-v2'
MINICPM_SDK_PROTOCOLS = frozenset({MINICPM_SDK_PROTOCOL, MINICPM_INTERFACE_PROTOCOL})
MINICPM_SELFHOST_PROTOCOL = 'minicpm47-q6-thinking-m4-off-v1'
MINICPM_PROTOCOLS = frozenset({MINICPM_PROTOCOL, MINICPM_SELFHOST_PROTOCOL, *MINICPM_SDK_PROTOCOLS})
MINICPM_COMPARISON = 'pinned-index-minicpm-historical-f37-v1'
MINICPM_MANIFEST = 'minicpm47-historical-comparison-v1'
MINICPM_HISTORY_ID = 'f37-off-gate-20261006-181014-8bd9a5'
MINICPM_HISTORY_SHA256 = '8ee4497f779148783241dc0c2add0574d727e8a368e96687b6aca0bdbcc91d37'
TERMINAL_STATUSES = {'completed', 'completed_with_errors', 'blocked', 'failed', 'interrupted', 'cancelled'}


class ComparisonScopeError(ValueError):
    """Existing runs cannot be compared within the requested report scope."""


class Repository:
    def __init__(self, root, *, history_roots=()):
        self.root = Path(root).resolve()
        self.roots = (self.root, *(Path(path).resolve() for path in history_roots))
        self._cache = {}
        self._cache_lock = threading.RLock()

    def path(self, run_id):
        run_id = safe_id(run_id)
        matches = [root / kind / run_id / "run.json" for root in self.roots for kind in ("runs", "origins")
                   if (root / kind / run_id / "run.json").is_file()]
        if len(matches) > 1:
            raise ValueError("ambiguous run identifier across report roots")
        if matches:
            return matches[0]
        raise FileNotFoundError("run not found")

    def load(self, run_id):
        """Legacy mutable full-record API; report views project before copying."""
        return copy.deepcopy(self._load_cached(run_id))

    def _load_cached(self, run_id):
        # Internal references are read-only. The lock avoids duplicate parsing
        # and summary work when browser polls arrive together.
        with self._cache_lock:
            return self._read_cached(run_id)

    def _read_cached(self, run_id):
        path = self.path(run_id)
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        key = str(path)
        if key in self._cache and self._cache[key][0] == stamp:
            return self._cache[key][1]
        state = read_json(path)
        if path.parent.parent.name == "origins":
            state["system_id"] = "sentrix"
            state["original_summary"] = copy.deepcopy(state.get("summary") or {})
            state["import_provenance"] = read_json(path.parent / "provenance.json")
            state["summary"] = summary(state)
            state["comparison_scope"] = "historical_system_comparison"
        if path.parent.parent.name == "runs":
            if state.get("mode") == "supplement":
                # The adapter freezes the eligible subset denominator even
                # when later questions have not started. Generic metrics use
                # observed items for nonformal runs and would shrink it.
                if (state.get("supplement_summary_version") != "http403-supplement-v1"
                        or not isinstance(state.get("summary"), dict)):
                    raise ValueError("supplement requires a versioned subset summary")
            elif state.get("mode") in {"gate_ab", "minicpm_compare"}:
                expected = state.get('protocol_label') if state['mode'] == 'minicpm_compare' else 'camroll-flash-gate-v1'
                if state['mode'] == 'minicpm_compare' and expected not in MINICPM_PROTOCOLS:
                    raise ValueError('Unknown MiniCPM report protocol')
                if (state.get("gate_summary_version") != expected
                        or not isinstance(state.get("summary"), dict)):
                    raise ValueError("gate report requires a versioned fixed-denominator summary")
            else:
                state["summary"] = summary(state)
        self._cache[key] = (stamp, state)
        return state

    @staticmethod
    def brief(state):
        keys = ("run_id", "system_id", "mode", "status", "created_at", "started_at", "finished_at", "updated_at",
                "album_id", "dataset_id", "qa_set", "qa_count", "config", "model_profile", "model_name",
                "judge_model", "qa_concurrency", "judge_concurrency", "cache_mode", "timing_comparable",
                "formal_protocol", "protocol_label", "view_kind", "parent_run_id", "repair_run_id",
                "repair_qa_ids", "repair_wall_ms", "repair_budget_before", "repair_budget_after",
                "repaired_valid_count", "attempted_turn_count_total", "budget_before", "budget_after", "phases",
                "index_version", "summary", "resource_summary", "code_revision", "fatal_error",
                'experiment', 'experiment_config', 'hardware_context', 'build_summary',
                'planned_turn_count', 'resumed_at', 'planned_eligible_qa_count',
                'supplement_summary_version', 'supplement_complete', 'build_skipped',
                'query_cache_policy', 'source_unchanged', 'pinned_snapshot_unchanged',
                'wall_ms', 'stop_reason', 'inference_config', 'index_binding', 'gate_summary_version',
                'gate_complete', 'gate_policy', 'gate_summary', 'seen_benchmark', 'suite_id',
                'inference_complete', 'comparison_scope')
        return {key: state.get(key) for key in keys if key in state} | Repository._report_fields(state)

    @staticmethod
    def _report_fields(state):
        if state.get('mode') not in {'gate_ab', 'minicpm_compare'}:
            return {}
        # Gate's frozen summary.completed includes judgment completeness.
        # Project execution separately, without changing the stored summary
        # or inferring an outcome for any unrecorded/in-flight turn.
        items = state.get('items') or []
        result = {'report_progress': {
            'finished_question_count': sum(row.get('final_turn_executed') is True for row in items),
            'executed_turn_count': sum(len(row.get('runtime_turns') or []) for row in items)}}
        # Public labels contain identities, never filesystem paths. Ancestry is
        # verified by the independent exporter; this view never follows it.
        metadata = state.get('reference') or state.get('continuation')
        if isinstance(metadata, dict):
            provenance = {'kind': 'reference' if state.get('reference') else 'continuation'}
            for key in ('parent_run_id', 'source_suite_id'):
                try:
                    provenance[key] = safe_id(metadata.get(key))
                except ValueError:
                    pass
            digest = metadata.get('parent_run_sha256')
            if isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
                provenance['parent_run_sha256'] = digest
            for key in ('inherited_turn_count', 'new_turn_count'):
                if type(metadata.get(key)) is int and 0 <= metadata[key] <= 111:
                    provenance[key] = metadata[key]
            result['report_provenance'] = provenance
        return result

    def get_brief(self, run_id):
        return copy.deepcopy(self.brief(self._load_cached(run_id)))

    def detail(self, run_id):
        state = self._load_cached(run_id)
        result = {key: value for key, value in state.items()
                  if key not in {"items", "media_mapping", "qa_ids", "reference", "continuation"}}
        result.update(self._report_fields(state))
        result["items"] = [{key: row.get(key) for key in ("qa_id", "question", "answer", "task_type", "angle", "difficulty",
                           "answerability", "execution_status", "judge_status", "retrieval_precision",
                           "retrieval_recall", "retrieval_f1", "agent_status", "wall_clock_ms",
                           "answer_origin", "execution_validity", "gate_status", "error_type")}
                           | {"judge": {key: (row.get("judge") or {}).get(key) for key in ("score", "reason", "judge_status")}}
                           for row in state.get("items", [])]
        present = {row["qa_id"] for row in result["items"]}
        result["items"].extend({**row, "judge": {"score": None, "reason": "not_started", "judge_status": None}}
                              for row in state.get("planned_status", []) if row["qa_id"] not in present)
        return copy.deepcopy(result)

    def _item_cached(self, run_id, qa_id):
        state = self._load_cached(run_id)
        value = next((row for row in state.get("items", []) if row.get("qa_id") == qa_id), None)
        if value is None:
            planned = next((row for row in state.get("planned_status", []) if row["qa_id"] == qa_id), None)
            if planned is None:
                raise FileNotFoundError("item not found")
            value = {**planned, "answer": "", "judge": {"score": None, "reason": "not_started"}, "runtime_turns": []}
        return value

    def _item_with_audit(self, run_id, qa_id):
        """Project durable Gate traces missing from a possibly interrupted run.

        Only fixed sibling files and matching run/question/turn contexts are
        read. Answers, scores, execution status and frozen summaries stay as
        recorded; this overlay is never written back to the run or its cache.
        """
        value = self._item_cached(run_id, qa_id)
        state = self._load_cached(run_id)
        if state.get('mode') not in {'gate_ab', 'minicpm_compare'}:
            return value
        plan = next((row for row in state.get('qa_plan', []) if row.get('qa_id') == qa_id), {})
        turn_count = plan.get('planned_turn_count')
        if type(turn_count) is not int or turn_count <= 0:
            return value
        result = dict(value)
        directory = self.path(run_id).parent
        files = (('gate_events.jsonl', 'gate_events'), ('model_calls.jsonl', 'model_call_metrics'),
                 ('verifier_calls.jsonl', 'model_call_metrics'))
        seen, added, issues = {}, 0, []

        def identity(record, field, turn):
            if field == 'model_call_metrics' and record.get('call_id'):
                return (turn, record['call_id'], record.get('request_id'), bool(record.get('replayed')))
            payload = {key: data for key, data in record.items()
                       if key not in {'evaluation_context', 'turn_index', 'report_audit'}}
            return (turn, json.dumps(payload, sort_keys=True, ensure_ascii=False))

        for filename, field in files:
            path = directory / filename
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                issues.append({'source': filename, 'issue': 'unsafe_trace_path'})
                continue
            if not path.is_file():
                continue
            if field not in seen:
                result[field] = list(value.get(field) or [])
                seen[field] = {identity(row, field, row.get('turn_index', row.get('conversation_turn', 0)))
                               for row in result[field]}
            try:
                # Decode one physical line at a time: a partial UTF-8 tail
                # must not hide earlier valid records in the same read buffer.
                with path.open('rb') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            issues.append({'source': filename, 'issue': 'incomplete_or_invalid_record'})
                            continue
                        if not isinstance(record, dict):
                            continue
                        context = record.get('evaluation_context')
                        if not isinstance(context, dict):
                            continue
                        turn = context.get('turn_index')
                        if (context.get('run_id') != run_id or context.get('qa_id') != qa_id
                                or type(turn) is not int or not 0 <= turn < turn_count):
                            continue
                        key = identity(record, field, turn)
                        if key in seen[field]:
                            continue
                        seen[field].add(key)
                        result[field].append({**record,
                            'turn_index' if field == 'gate_events' else 'conversation_turn': turn,
                            'report_audit': {'source': filename, 'not_in_run_snapshot': True}})
                        added += 1
            except (OSError, UnicodeError):
                issues.append({'source': filename, 'issue': 'trace_unreadable'})
        if added or issues:
            result['trace_audit'] = {'read_only': True, 'supplemented_count': added, 'issues': issues,
                'note': 'Durable traces absent from run.json; no missing answer, score or execution outcome is inferred.'}
        return result

    @staticmethod
    def _trace_brief(record, index):
        omitted = {"input", "output", "context", "args", "arguments", "result", "response", "messages"}
        return {**{key: value for key, value in record.items()
                   if key not in omitted and (not isinstance(value, (dict, list)) or key in {"error", "usage"})},
                "trace_index": index}

    def item(self, run_id, qa_id, *, light=False):
        value = self._item_with_audit(run_id, qa_id)
        if not light:
            return copy.deepcopy(value)
        omitted = {"model_call_metrics", "tool_trace", "runtime_turns", "conversation",
                   "resume_history", "resume_last_response", "conversation_history", "gate_events", "gate_attempts"}
        result = {key: data for key, data in value.items() if key not in omitted}
        result["runtime_turns"] = [{key: data for key, data in turn.items() if key not in omitted}
                                   for turn in value.get("runtime_turns") or []]
        result["conversation"] = result["runtime_turns"] if len(result["runtime_turns"]) > 1 else []
        for key in ("model_call_metrics", "tool_trace"):
            result[key] = [self._trace_brief(record, index) for index, record in enumerate(value.get(key) or [])]
        result["trace_counts"] = {"model": len(result["model_call_metrics"]), "tool": len(result["tool_trace"])}
        if "gate_events" in value:
            result["gate_events"] = [self._trace_brief(record, index) for index, record in enumerate(value["gate_events"])]
            result["trace_counts"]["gate"] = len(result["gate_events"])
        result["trace_payloads_deferred"] = True
        return copy.deepcopy(result)

    def item_trace(self, run_id, qa_id, kind, index):
        key = {"model": "model_call_metrics", "tool": "tool_trace", "gate": "gate_events"}.get(kind)
        if key is None or index < 0:
            raise FileNotFoundError("trace not found")
        records = self._item_with_audit(run_id, qa_id).get(key) or []
        if index >= len(records):
            raise FileNotFoundError("trace not found")
        return copy.deepcopy(records[index])

    def list(self):
        rows = []
        for root, kind in ((root, kind) for root in self.roots for kind in ("runs", "origins")):
            for path in (root / kind).glob("*/run.json"):
                try:
                    rows.append(self.get_brief(path.parent.name))
                except (OSError, ValueError):
                    continue
        return sorted(rows, key=lambda row: row.get("created_at") or row.get("started_at") or "", reverse=True)

    def import_history(self, directory):
        source = Path(directory).resolve()
        state = read_json(source / "run.json")
        run_id = safe_id(state["run_id"])
        if state.get("album_id") != "album3-max" or state.get("qa_set") != "100qa-full" or state.get("mode") not in {"full", "reuse"}:
            raise ValueError("historical comparison requires full/reuse album3-max / 100qa-full")
        if len(state.get("items") or []) != 100:
            raise ValueError("historical run must contain 100 QA records including failures")
        if state.get("mode") == "reuse":
            integrity = state.get("input_integrity") or {}
            if (integrity.get("qa_sha256") != "f625ae9e26b4666c277b424bd472b6790baf0d6773e9020b20eff985d4cc2eb0" or
                integrity.get("dataset_sha256") != "120e2e7273e731ed24fd24cdf477ce81e7c6b71e43554b0366aa8677b8098627"):
                raise ValueError("reuse input does not match the approved comparison dataset")
        target = self.root / "origins" / run_id
        digest = sha256(source / "run.json")
        if target.exists():
            if sha256(target / "run.json") == digest:
                return run_id
            raise ValueError("history already exists with different bytes")
        temporary = target.with_name("." + run_id + ".importing")
        if temporary.exists():
            raise ValueError("incomplete historical import exists; inspect before retry")
        temporary.mkdir(parents=True, mode=0o700)
        files = []
        try:
            for name in ("run.json", "results.jsonl", "gpu_samples.jsonl"):
                original = source / name
                if not original.is_file():
                    continue
                shutil.copyfile(original, temporary / name)
                actual = sha256(temporary / name)
                if actual != sha256(original):
                    raise ValueError("history changed during import")
                (temporary / name).chmod(0o400)
                files.append({"name": name, "sha256": actual, "bytes": original.stat().st_size})
            atomic_json(temporary / "provenance.json", {"imported_at": now_iso(), "files": files,
                "source_run_id": run_id, "source": "Sentrix PhotoBench read-only export",
                "note": "Original bytes preserved; compatibility recomputation is separate."})
            temporary.rename(target)
        except Exception:
            # Preserve interrupted local copy for diagnosis; original is untouched.
            raise
        return run_id

    def _remaining_comparison_scope(self, states):
        """One terminal, hash-bound four-run comparison; never rewrite suite IDs."""
        for index, root in enumerate(self.roots[:-1]):
            result = self._remaining_comparison_scope_at(states, root, self.roots[index + 1])
            if result is not None:
                return result
        return None

    def _remaining_comparison_scope_at(self, states, current_root, parent_root):
        path = current_root / 'remaining-comparison.json'
        if not path.is_file() or path.is_symlink() or path.resolve().parent != current_root:
            return None
        try:
            scope = read_json(path)
            expected = ('F37-Off', 'F37-On', 'F38-Off', 'F38-On')
            entries = scope['runs']
            if (scope.get('schema_version') != 'camroll-flash-gate-remaining-comparison-v1'
                    or scope.get('protocol') != 'camroll-flash-gate-v1'
                    or scope.get('comparison_complete') is not False
                    or tuple(row['group'] for row in entries) != expected
                    or len({row['run_id'] for row in entries}) != 4
                    or not {state['run_id'] for state in states} <= {row['run_id'] for row in entries}
                    or not scope.get('index_binding') or not scope.get('input_integrity')
                    or any(not scope['input_integrity'].get(key) for key in ('qa_sha256', 'dataset_sha256'))):
                return None
            safe_id(scope['suite_id'])
            safe_id(scope['source_suite_id'])
            if scope['suite_id'] == scope['source_suite_id']:
                return None
            for index, row in enumerate(entries):
                run_id = safe_id(row['run_id'])
                source = self.path(run_id)
                owning_root = parent_root if index < 3 else current_root
                expected_suite = scope['source_suite_id'] if index < 3 else scope['suite_id']
                if (source.parent.parent.parent != owning_root or source.resolve() != source.absolute()
                        or row.get('source') != ('history' if index < 3 else 'current')
                        or row.get('suite_id') != expected_suite or sha256(source) != row.get('run_sha256')):
                    return None
                state = self._load_cached(run_id)
                if (state.get('suite_id') != expected_suite or state.get('mode') != 'gate_ab'
                        or state.get('formal_protocol') is not True
                        or state.get('protocol_label') != scope['protocol']
                        or state.get('inference_config', {}).get('group') != expected[index]
                        or state.get('status') not in {'completed', 'completed_with_errors', 'blocked', 'failed', 'interrupted', 'cancelled'}
                        or state.get('index_binding') != scope['index_binding']
                        or state.get('input_integrity') != scope['input_integrity']
                        or object_hash(state.get('qa_plan')) != scope.get('qa_plan_sha256')):
                    return None
            return {'suite_id': scope['suite_id'], 'source_suite_id': scope['source_suite_id'],
                    'comparison_complete': False, 'source_hashes_verified': True}
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def _minicpm_comparison_scope(self, states, *, manifest=None):
        """Permit exactly the pinned original F37-Off and one new terminal run.

        All paths come from configured repository roots; no manifest path is
        followed. Hash both files at each request before trusting cached data.
        """
        path = self.root / 'minicpm-comparison.json'
        try:
            if len(states) != 2:
                return None
            if manifest is None:
                if not path.is_file() or path.is_symlink() or path.resolve().parent != self.root:
                    return None
                manifest = read_json(path)
            scope = manifest
            entries = scope['runs']
            if (scope.get('schema_version') != MINICPM_MANIFEST
                    or scope.get('protocol') not in MINICPM_PROTOCOLS
                    or scope.get('comparison_scope') != MINICPM_COMPARISON
                    or len(entries) != 2
                    or [row.get('source') for row in entries] != ['current', 'history']
                    or len({row['run_id'] for row in entries}) != 2
                    or {state['run_id'] for state in states} != {row['run_id'] for row in entries}
                    or entries[1]['run_id'] != MINICPM_HISTORY_ID
                    or entries[1].get('run_sha256') != MINICPM_HISTORY_SHA256):
                return None
            verified = []
            for index, entry in enumerate(entries):
                source = self.path(safe_id(entry['run_id']))
                owner = source.parent.parent.parent
                if source.resolve() != source.absolute() or (index == 0 and owner != self.root):
                    return None
                if index == 1 and (owner not in self.roots[1:] or owner.parent.name != 'photobench-flash-gate-20261006f'):
                    return None
                if sha256(source) != entry.get('run_sha256'):
                    return None
                state = self._load_cached(entry['run_id'])
                protocol = scope['protocol'] if index == 0 else 'camroll-flash-gate-v1'
                if (state.get('mode') != ('minicpm_compare' if index == 0 else 'gate_ab')
                        or state.get('protocol_label') != protocol
                        or state.get('formal_protocol') is not True
                        or state.get('status') not in TERMINAL_STATUSES
                        or state.get('qa_count') != 100 or state.get('planned_turn_count') != 111
                        or state.get('planned_eligible_qa_count') != 81
                        or state.get('summary', {}).get('retrieval_metric_count') != 81
                        or state.get('index_binding') != scope.get('index_binding')
                        or not state.get('index_binding')
                        or state.get('input_integrity') != scope.get('input_integrity')
                        or any(not state.get('input_integrity', {}).get(key) for key in ('qa_sha256', 'dataset_sha256'))
                        or len(state.get('qa_plan') or []) != 100
                        or len({row['qa_id'] for row in state['qa_plan']}) != 100
                        or sum(row['planned_turn_count'] for row in state['qa_plan']) != 111
                        or object_hash(state['qa_plan']) != scope.get('qa_plan_sha256')
                        or not state.get('judge_protocol')
                        or object_hash(state['judge_protocol']) != scope.get('judge_protocol_sha256')):
                    return None
                cfg = state.get('inference_config') or {}
                if cfg.get('gate_policy') != 'off':
                    return None
                if index == 0:
                    if cfg.get('generation_model') != ('minicpm-v4.7-35b-a3b-q6-thinking' if protocol == MINICPM_SELFHOST_PROTOCOL else 'MiniCPM-V-4.7-35B-A3B'):
                        return None
                    if protocol == MINICPM_PROTOCOL:
                        if (cfg.get('enable_thinking') is not True or cfg.get('stream') is not False
                                or cfg.get('max_output_tokens') != 8192):
                            return None
                    else:
                        from .minicpm_experiment import inference_config
                        profile = inference_config(cfg)
                        if profile.protocol_version != protocol or profile.request_profile != ('selfhost-q6-v1' if protocol == MINICPM_SELFHOST_PROTOCOL else 'sdk-defaults-v1'):
                            return None
                if index == 1 and (cfg.get('group') != 'F37-Off' or cfg.get('generation_model') != 'qwen3.7-flash'):
                    return None
                verified.append(entry['run_id'])
            return {'source_hashes_verified': True, 'historical_run_id': MINICPM_HISTORY_ID,
                    'historical_run_sha256': MINICPM_HISTORY_SHA256, 'run_ids': verified,
                    'protocol': scope['protocol'], 'contemporary': False,
                    'quality_denominator': 100, 'retrieval_denominator': 81}
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def compare(self, run_ids):
        states = [self._load_cached(run_id) for run_id in run_ids]
        if not states:
            return {"runs": [], "rows": []}
        if any(state.get("mode") == "supplement" for state in states):
            raise ValueError("supplement subsets are reported separately from formal comparisons")
        if any(state.get("formal_protocol") is False for state in states):
            raise ValueError("smoke runs are excluded from formal comparisons")
        minicpm_comparison = any(state.get('mode') == 'minicpm_compare' for state in states)
        minicpm_scope = None
        if minicpm_comparison:
            minicpm_scope = self._minicpm_comparison_scope(states)
            if minicpm_scope is None:
                raise ComparisonScopeError('MiniCPM 仅允许终态且来源哈希通过的新正式组与指定原 F37-Off 对照；'
                                           '其他历史协议、验收样本和运行中记录请单独查看。')
        gate_comparison = not minicpm_comparison and any(state.get('mode') == 'gate_ab' for state in states)
        remaining_scope = None
        if gate_comparison:
            if not all(state.get('mode') == 'gate_ab' and state.get('protocol_label') == 'camroll-flash-gate-v1'
                       for state in states):
                raise ValueError('Gate protocol is reported separately from historical protocols')
            bindings = {json.dumps(state.get('index_binding'), sort_keys=True) for state in states}
            if len(bindings) != 1 or not states[0].get('index_binding'):
                raise ValueError('Gate comparison requires the same immutable index binding')
            if len(states) > 1:
                suites = [state.get('suite_id') for state in states]
                if (any(not isinstance(suite, str) or not suite.strip() for suite in suites)
                        or len(set(suites)) != 1):
                    remaining_scope = self._remaining_comparison_scope(states)
                    if remaining_scope is None:
                        raise ComparisonScopeError('Gate 对照只允许同一批次，或终态导出且全部来源哈希通过的固定续测组合；'
                                                   '运行中的新批次与其他历史记录请单独查看。')
        qa_ids = list(dict.fromkeys(row["qa_id"] for state in states for row in (state.get("qa_plan") or state.get("items", []))))
        plans = {row['qa_id']: row for state in states for row in state.get('qa_plan', [])}
        maps = [{row["qa_id"]: row for row in state.get("items", [])} for state in states]
        rows = []
        for qa_id in qa_ids:
            observations = [mapping.get(qa_id) for mapping in maps]
            first = next((row for row in observations if row), plans.get(qa_id, {}))
            rows.append({"qa_id": qa_id, "question": first.get("question"), "results": [
                {"run_id": state["run_id"], "score": (row.get("judge") or {}).get("score") if row else None,
                 "recall": row.get("retrieval_recall") if row else None,
                 "precision": row.get("retrieval_precision") if row else None,
                 "agent_ms": (row.get("timing_breakdown") or {}).get("agent_wall_ms") if row else None,
                 "status": row.get("execution_status") if row else "missing"}
                for state, row in zip(states, observations)]})
        qa_hashes = {state.get("input_integrity", {}).get("qa_sha256") for state in states}
        dataset_hashes = {state.get("input_integrity", {}).get("dataset_sha256") for state in states}
        contemporary = all(state.get('experiment_config') for state in states)
        return copy.deepcopy({"runs": [self.brief(state) for state in states], "rows": rows,
                "same_qa_hash": len(qa_hashes) == 1 and None not in qa_hashes,
                "same_dataset_hash": len(dataset_hashes) == 1 and None not in dataset_hashes,
                "remaining_scope": remaining_scope,
                "minicpm_scope": minicpm_scope,
                "comparison_scope": MINICPM_COMPARISON if minicpm_comparison else "frozen_index_flash_gate_comparison" if gate_comparison else "contemporary_full_system_comparison" if contemporary else "historical_system_comparison",
                "warning": (("非同期历史对照：仅新 MiniCPM 与哈希锁定的原 F37-Off，共用 QA、计划、数据、冻结索引和 Judge 协议。" +
                    ("MiniCPM 的 Thinking、温度、输出上限和 stream 请求参数均未指定，使用供应商默认；实际上限未知，不标为 Thinking On 或 Off。" if minicpm_scope['protocol'] in MINICPM_SDK_PROTOCOLS else "MiniCPM 开启 thinking、输出上限8192、非流式。") +
                    "实际工具和 JSON 模式以冻结配置为准；context_limit 单轮失败后继续，历史停止规则不改。固定100题质量与81题交付指标；非流式 TTFT 未观测，不作同条件速度或单一模型因果结论。") if minicpm_comparison else ("本次仅独立执行剩余F38-On；前三组只读引用此前封存的结果，F38-Off仍已阻断，四组比较尚未完整。" if remaining_scope else "") + ("冻结 WeMM 索引上的 Flash × Gate 对照。Doubao同时担任事实Gate和独立最终评分，存在模型关联。album3为已见基准，不代表独立泛化能力。" if gate_comparison else "同期 M4 完整系统比较；不能单独归因于 embedding 或 reranker。" if contemporary else "历史运行时间、硬件和冷暖状态不同；质量与指标口径对齐，不代表同硬件速度排名。"))})
