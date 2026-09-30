"""Independent immutable history imports and run access."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import threading

from .common import read_json, atomic_json, safe_id, sha256, now_iso
from .metrics import summary


class Repository:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._cache = {}
        self._cache_lock = threading.RLock()

    def path(self, run_id):
        run_id = safe_id(run_id)
        for kind in ("runs", "origins"):
            path = self.root / kind / run_id / "run.json"
            if path.is_file():
                return path
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
                'wall_ms', 'stop_reason')
        return {key: state.get(key) for key in keys if key in state}

    def get_brief(self, run_id):
        return copy.deepcopy(self.brief(self._load_cached(run_id)))

    def detail(self, run_id):
        state = self._load_cached(run_id)
        result = {key: value for key, value in state.items() if key not in {"items", "media_mapping", "qa_ids"}}
        result["items"] = [{key: row.get(key) for key in ("qa_id", "question", "answer", "task_type", "angle", "difficulty",
                           "answerability", "execution_status", "judge_status", "retrieval_precision",
                           "retrieval_recall", "retrieval_f1", "agent_status", "wall_clock_ms")}
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

    @staticmethod
    def _trace_brief(record, index):
        omitted = {"input", "output", "context", "args", "arguments", "result", "response", "messages"}
        return {**{key: value for key, value in record.items()
                   if key not in omitted and (not isinstance(value, (dict, list)) or key in {"error", "usage"})},
                "trace_index": index}

    def item(self, run_id, qa_id, *, light=False):
        value = self._item_cached(run_id, qa_id)
        if not light:
            return copy.deepcopy(value)
        omitted = {"model_call_metrics", "tool_trace", "runtime_turns", "conversation",
                   "resume_history", "resume_last_response", "conversation_history"}
        result = {key: data for key, data in value.items() if key not in omitted}
        result["runtime_turns"] = [{key: data for key, data in turn.items() if key not in omitted}
                                   for turn in value.get("runtime_turns") or []]
        result["conversation"] = result["runtime_turns"] if len(result["runtime_turns"]) > 1 else []
        for key in ("model_call_metrics", "tool_trace"):
            result[key] = [self._trace_brief(record, index) for index, record in enumerate(value.get(key) or [])]
        result["trace_counts"] = {"model": len(result["model_call_metrics"]), "tool": len(result["tool_trace"])}
        result["trace_payloads_deferred"] = True
        return copy.deepcopy(result)

    def item_trace(self, run_id, qa_id, kind, index):
        key = {"model": "model_call_metrics", "tool": "tool_trace"}.get(kind)
        if key is None or index < 0:
            raise FileNotFoundError("trace not found")
        records = self._item_cached(run_id, qa_id).get(key) or []
        if index >= len(records):
            raise FileNotFoundError("trace not found")
        return copy.deepcopy(records[index])

    def list(self):
        rows = []
        for kind in ("runs", "origins"):
            for path in (self.root / kind).glob("*/run.json"):
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

    def compare(self, run_ids):
        states = [self._load_cached(run_id) for run_id in run_ids]
        if not states:
            return {"runs": [], "rows": []}
        if any(state.get("mode") == "supplement" for state in states):
            raise ValueError("supplement subsets are reported separately from formal comparisons")
        if any(state.get("formal_protocol") is False for state in states):
            raise ValueError("smoke runs are excluded from formal comparisons")
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
                "comparison_scope": "contemporary_full_system_comparison" if contemporary else "historical_system_comparison",
                "warning": "同期 M4 完整系统比较；不能单独归因于 embedding 或 reranker。" if contemporary else "历史运行时间、硬件和冷暖状态不同；质量与指标口径对齐，不代表同硬件速度排名。"})
