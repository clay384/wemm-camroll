"""Separate, one-shot HTTP-403 recovery measurements on a pinned W38Max snapshot.

The frozen Runner performs each Agent turn and Judge call. This adapter only
selects the authorized subset, isolates its records, and bounds submissions.
It never builds an album or composes replacement results into the first run.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import os
from pathlib import Path
import shutil
import threading
import time

from .common import atomic_json, append_jsonl, now_iso, object_hash, read_json, safe_id, sha256
from .dataset import Dataset, snapshot_media
from .metrics import _fixed_macro, retrieval_diagnostics, summary
from .runner import Runner


FROZEN_REVISION = "four-88743288eb39fb9666e7881c0b2d88dd03aaa7eb"
SUMMARY_VERSION = "http403-supplement-v1"
ADAPTER_PROTOCOL = "http403-recovery-original-asset-binding-v2"


def _error_403(record):
    if not isinstance(record, dict):
        return False
    return (record.get("error_code") == "http_403" or record.get("code") == "http_403"
            or any(type(record.get(key)) is int and record[key] == 403
                   for key in ("http_status", "status_code", "code")))


def http403_evidence(item):
    """Only actual call/error fields count; answers and prompt payloads do not."""
    found = []
    def inspect(record, path, *, tool=False):
        if not isinstance(record, dict):
            return
        if _error_403(record):
            found.append({"path": path, "call_id": record.get("call_id") or record.get("budget_call_id"),
                          "error_code": "http_403"})
        for key in ("error", "exception"):
            if isinstance(record.get(key), dict):
                inspect(record[key], path + "." + key)
        if tool and isinstance(record.get("result"), dict):
            inspect(record["result"], path + ".result")
        for key in ("model_call_metrics", "model_calls"):
            for index, call in enumerate(record.get(key) or []):
                inspect(call, f"{path}.{key}[{index}]")
    for key in ("model_call_metrics", "tool_trace"):
        for index, record in enumerate(item.get(key) or []):
            inspect(record, f"{key}[{index}]", tool=key == "tool_trace")
    return found


def select_http403_items(parent):
    if parent.get("mode") != "full" or parent.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("supplement requires a terminal original full run")
    items = parent.get("items") or []
    if len({row["qa_id"] for row in items}) != len(items):
        raise ValueError("parent has duplicate question records")
    # A valid original score is retained; it never excludes an affected question
    # from this separately authorized measurement.
    return [row for row in items if row.get("execution_status") == "error" and http403_evidence(row)]


def file_hashes(directory):
    root = Path(directory)
    files = list(root.rglob("*"))
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in files):
        raise ValueError("snapshot must be a regular directory without symlinks")
    return {str(path.relative_to(root)): sha256(path) for path in sorted(files) if path.is_file()}


def verify_frozen_core(parent, freeze_path):
    """Report/adapter files may differ; all paid behavior must match 8874328."""
    freeze = read_json(freeze_path)
    if parent.get("code_revision") != FROZEN_REVISION or freeze.get("source_revision") != FROZEN_REVISION:
        raise ValueError("supplement requires the original 8874328 paid implementation")
    package = Path(__file__).resolve().parent.parent
    release = package.parents[1]
    paths = list((package / "camroll_agent").rglob("*.py"))
    paths += [package / "camroll_eval" / name for name in (
        "__init__.py", "runner.py", "judge.py", "judge_prompts.json", "compat.py", "task_actions.json",
        "dataset.py", "metrics.py", "provenance.json", "common.py", "telemetry.py")]
    matched = {}
    for path in paths:
        name = str(path.relative_to(release))
        digest = sha256(path)
        if digest != freeze.get("source_files", {}).get(name):
            raise ValueError("frozen paid implementation changed: " + name)
        matched[name] = digest
    if parent.get("input_integrity") != freeze.get("input_integrity"):
        raise ValueError("parent inputs differ from the frozen source")
    return {"frozen_revision": FROZEN_REVISION, "freeze_sha256": sha256(freeze_path),
            "matched_files": matched, "adapter_sha256": sha256(__file__)}


def supplement_summary(state):
    """Frozen scoring calculations with explicit authorized subset denominators."""
    metrics = summary(state)
    eligible = [row for row in state.get("items", []) if row.get("answerability") != "unanswerable"]
    denominator = state["planned_eligible_qa_count"]
    final = _fixed_macro(eligible, "image_retrieval_counts", denominator)
    metrics["final_delivery_macro"] = final
    metrics["candidate_retrieval_macro"] = _fixed_macro(eligible, "candidate_retrieval_counts", denominator)
    metrics["retrieval_diagnostics"] = retrieval_diagnostics(state, denominator)
    for metric in ("precision", "recall", "f1"):
        for prefix in ("retrieval_", "media_retrieval_", "image_retrieval_"):
            metrics[prefix + metric + "_macro"] = final[metric]
    metrics.update(retrieval_metric_count=denominator, image_retrieval_metric_count=denominator,
        retrieval_observed_metric_count=final["observed_metric_count"], retrieval_recall_mean=final["recall"],
        retrieval_metric_scope="fixed_supplement_subset", benchmark_e2e_latency_excluding_judge_ms=None,
        retrieval_excluded_unanswerable_count=state["qa_count"] - denominator)
    return metrics


class _SnapshotBoundBackend:
    """Keep frozen Agent/Library behavior while guarding the original album."""
    def __init__(self, backend, memory, version, check_source, on_violation):
        self._backend, self._memory, self._version = backend, Path(memory).resolve(), version
        self._check_source = check_source
        self._on_violation = on_violation

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def agent(self, memory, config):
        try:
            if Path(memory).resolve() != self._memory:
                raise ValueError("Supplement queries must retain the original managed asset boundary")
            self._check_source()
            agent = self._backend.agent(self._memory, config)
            library = agent.library
            if library.version != self._version or library.memory_dir != self._memory:
                raise ValueError("Supplement Agent changed the pinned original album binding")
            library.image_paths(list(library.images))
            return agent
        except Exception:
            self._on_violation()
            raise


class SupplementRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._submission_stop = threading.Event()
        self._blocked_reason = None
        self._backend = None

    def _block(self, reason, *, cancel_paid=False):
        with self._persist_lock:
            # Preserve the precise accounting failure when frozen Runner has
            # already raised its generic cancellation flag.
            priority = {"cancelled": 0, "http_403": 1, "budget_exceeded": 2,
                        "budget_limit_changed": 3, "request_uncertain": 4,
                        "snapshot_integrity_changed": 5}
            if self._blocked_reason is None or priority.get(reason, 2) > priority.get(self._blocked_reason, 2):
                self._blocked_reason = reason
            self._submission_stop.set()
            if cancel_paid:
                self.cancel.set()

    def _budget_gate(self):
        if self._backend is None:
            return True
        try:
            budget = self._backend.budget()
            reason = ("budget_limit_changed" if budget.get("limit_cny") != 600 else
                      "request_uncertain" if budget.get("uncertain_requests", 0) else
                      "budget_exceeded" if budget.get("remaining_cny", 0) <= 0 else None)
        except Exception as exc:
            reason = getattr(exc, "code", None) or "budget_check_failed"
        if reason:
            self._block(reason, cancel_paid=True)
            return False
        return True

    def _cancelled(self):
        self._budget_gate()
        if super()._cancelled():
            self._block(self._blocked_reason or "cancelled", cancel_paid=True)
        return self.cancel.is_set()

    def _generation_call(self, event):
        if _error_403(event):
            self._block("http_403")

    def _judge_call(self, event):
        if _error_403(event):
            self._block("http_403")
        super()._judge_call(event)

    def _save(self):
        # Frozen metrics remain untouched. Persist explicit subset denominators
        # even when a later provider failure leaves planned questions unstarted.
        with self._persist_lock:
            state = self._state
            state["updated_at"] = now_iso()
            observed = {row["qa_id"]: row for row in state.get("items", [])}
            state["planned_status"] = [{**row,
                "execution_status": observed.get(row["qa_id"], {}).get("execution_status", "not_started"),
                "judge_status": observed.get(row["qa_id"], {}).get("judge_status", "not_started"),
                "executed_turn_count": len(observed.get(row["qa_id"], {}).get("runtime_turns") or [])}
                for row in state["qa_plan"]]
            state["summary"] = supplement_summary(state)
            state["supplement_summary_version"] = SUMMARY_VERSION
            if self._blocked_reason:
                state["stop_reason"] = self._blocked_reason
            atomic_json(self._path / "run.json", state)

    def run_supplement(self, parent_id, *, run_id, freeze_path, strict=True,
                       expected_qa_count=64, expected_turn_count=71, lock_held=False):
        if not lock_held:
            self.acquire()
        writable = False
        sampler = None
        original = None
        parent_path = self.root / "runs" / safe_id(parent_id) / "run.json"
        began = time.monotonic()
        try:
            parent_hash = sha256(parent_path)
            parent = read_json(parent_path)
            selected = select_http403_items(parent)
            if parent.get("experiment") != "W38Max" or parent.get("config") != "M4" or (
                    parent.get("qa_concurrency"), parent.get("judge_concurrency")) != (1, 4):
                raise ValueError("supplement is bound to W38Max / M4 / Agent 1 / Judge 4")
            dataset = Dataset(self.source, strict=strict)
            integrity = dataset.integrity()
            if integrity != parent.get("input_integrity"):
                raise ValueError("supplement inputs changed")
            core = verify_frozen_core(parent, freeze_path)
            chosen = {row["qa_id"]: row for row in selected}
            rows = [row for row in dataset.rows if row["qa_id"] in chosen]
            plan = {row["qa_id"]: row["planned_turn_count"] for row in parent.get("qa_plan", [])}
            turns = sum(len(dataset.turns(row)) for row in rows)
            if (len(rows) != len(selected) or len(rows) != expected_qa_count or turns != expected_turn_count
                    or any(plan.get(row["qa_id"]) != len(dataset.turns(row)) for row in rows)):
                raise ValueError("HTTP 403 subset differs from the authorized question/turn plan")
            run_id = safe_id(run_id)
            self._path = self.root / "runs" / run_id
            if self._path.exists():
                raise ValueError("supplement run already exists; automatic retry is forbidden")
            version = safe_id(parent["index_version"])
            original = self.albums / safe_id(parent["album_id"])
            snapshot = original / "snapshots" / version
            snapshot_hashes = file_hashes(snapshot)
            pointer = (original / "CURRENT").read_bytes()
            if pointer.decode().strip() != version:
                raise ValueError("parent snapshot pointer changed")
            from camroll_agent.library import Library
            from camroll_agent.retrieval import cache_key, resolve_experiment
            experiment = copy.deepcopy(parent["experiment_config"])
            profile, namespace = resolve_experiment(experiment)
            if profile.name != "W38Max" or namespace != parent_id:
                raise ValueError("parent experiment binding changed")
            binding_path = original / "experiment.json"
            binding = read_json(binding_path)
            if binding != {"profile": profile.name, "configuration_hash": cache_key([], profile, namespace),
                           "cache_namespace": namespace}:
                raise ValueError("parent album binding differs from its frozen profile")
            binding_hash = sha256(binding_path)
            library = Library(original, config="M4", experiment_config=experiment)
            if library.version != version or library.metadata.get("image_embedding_count") != len(dataset.photos):
                raise ValueError("parent vector coverage changed")
            # Published records contain absolute managed asset paths. A copied
            # snapshot is an audit artifact, not a relocatable runnable album.
            # Validate every real asset before creating any paid client.
            paths = library.image_paths(list(library.images))
            if len(paths) != len(dataset.photos):
                raise ValueError("parent managed asset coverage changed")
            parent_album_hashes = file_hashes(original)
            original = {"memory": original, "snapshot": snapshot, "snapshot_hashes": snapshot_hashes,
                        "current": pointer, "binding_sha256": binding_hash, "parent_sha256": parent_hash,
                        "album_hashes": parent_album_hashes}
            selection = {"parent_run_id": parent_id, "parent_sha256": parent_hash,
                "qa_ids": [row["qa_id"] for row in rows], "question_count": len(rows), "planned_turn_count": turns,
                "rule": "parent.execution_status=error AND structured call error=http_403; original scores retained",
                "evidence": {row["qa_id"]: http403_evidence(chosen[row["qa_id"]]) for row in rows}}
            selection_hash = object_hash(selection)
            claim = self.root / "supplement_claims" / safe_id(parent_id) / (selection_hash + ".json")
            claim.parent.mkdir(parents=True, exist_ok=True)
            with claim.open("x") as stream:
                import json
                json.dump({"run_id": run_id, "selection_sha256": selection_hash, "created_at": now_iso()}, stream)
            claim.chmod(0o600)
            self._path.mkdir(mode=0o700)
            memory = self._path / "pinned_memory"
            shutil.copytree(snapshot, memory / "snapshots" / version)
            (memory / "CURRENT").write_bytes(pointer)
            shutil.copyfile(binding_path, memory / "experiment.json")
            if file_hashes(memory / "snapshots" / version) != snapshot_hashes:
                raise ValueError("pinned snapshot copy mismatch")
            Library(memory, config="M4", experiment_config=experiment)
            media, metadata = snapshot_media(memory, version)
            journal_path = self._path / "request-journal.sqlite"
            self._state = {"run_id": run_id, "system_id": "camroll_wemm", "mode": "supplement",
                "view_kind": "http403_recovery_supplement", "formal_protocol": False, "status": "preparing",
                "protocol_label": ADAPTER_PROTOCOL, "created_at": now_iso(), "started_at": now_iso(),
                "parent_run_id": parent_id, "parent_sha256": parent_hash, "parent_formal_qa_count": parent["qa_count"],
                "parent_planned_turn_count": parent["planned_turn_count"], "index_version": version,
                "snapshot_hashes": snapshot_hashes, "snapshot_metadata": metadata, "pinned_memory": str(memory),
                "pinned_memory_purpose": "audit_only_snapshot_copy_not_query_album",
                "query_memory": str(original["memory"]), "asset_binding": "original_album_assets_read_only",
                "parent_album_manifest_sha256": object_hash(parent_album_hashes),
                "asset_binding_validation": {"validated_image_count": len(paths), "index_version": version,
                    "before_paid_calls": True, "policy": "Frozen Library.image_paths original memory_dir/assets boundary"},
                "album_id": parent["album_id"], "dataset_id": parent.get("dataset_id"), "qa_set": dataset.qa_set,
                "experiment": "W38Max", "experiment_config": experiment, "generation_request_namespace": run_id,
                "request_journal_path": str(journal_path), "core_verification": core,
                "code_revision": os.getenv("CAMROLL_RELEASE_REVISION", "unrecorded"), "paid_core_revision": FROZEN_REVISION,
                "adapter_sha256": sha256(__file__), "config": "M4", "qa_concurrency": 1, "judge_concurrency": 4,
                "qa_ids": selection["qa_ids"], "qa_count": len(rows), "qa_plan": [{"qa_id": row["qa_id"],
                    "question": row["question"], "planned_turn_count": len(dataset.turns(row))} for row in rows],
                "planned_turn_count": turns, "planned_eligible_qa_count": sum(row.get("answerability") != "unanswerable" for row in rows),
                "items": [], "input_integrity": integrity, "media_mapping": media, "timing_comparable": False,
                "metric_source": parent["metric_source"], "phases": {"qa_eval": {}}, "build_skipped": True,
                "cache_mode": "reused_parent_snapshot_fresh_queries", "query_cache_policy": "fresh_clients_no_parent_cache",
                "selection_sha256": selection_hash, "hardware_context": copy.deepcopy(parent.get("hardware_context", {})),
                "note": "Separate recovery measurement; original album is queried read-only with its managed asset boundary intact. The copied snapshot is audit-only. No first-run records or scores are replaced and no album is rebuilt. Full album hash guards add local verification time; timing is not comparable with cold runs."}
            writable = True
            atomic_json(self._path / "selection.json", selection)
            atomic_json(self._path / "questions.json", rows)
            atomic_json(self._path / "input_integrity.json", integrity)
            atomic_json(self._path / "media_mapping.json", media)
            atomic_json(self._path / "parent_album_hashes.json", parent_album_hashes)
            self._save()
            judge = self.judge_factory(cancel=self.cancel, on_call=self._judge_call, request_namespace=run_id,
                request_journal_path=journal_path, expected_limit_cny=600)
            if judge.protocol() != parent["judge_protocol"]:
                raise ValueError("supplement Judge protocol changed")
            self._state.update(judge_protocol=judge.protocol(), judge_model=parent["judge_model"])
            backend = self.backend_factory(experiment_config=experiment, run_id=run_id, journal_path=journal_path)
            def check_source():
                if (file_hashes(original["memory"]) != original["album_hashes"]
                        or sha256(parent_path) != parent_hash):
                    self._block("snapshot_integrity_changed", cancel_paid=True)
                    raise ValueError("original album or parent run changed before question submission")
            self._backend = _SnapshotBoundBackend(backend, original["memory"], version, check_source,
                lambda: self._block("snapshot_integrity_changed", cancel_paid=True))
            self._state["budget_before"] = self._backend.budget()
            if self.sample_resources:
                from .telemetry import ResourceSampler
                sampler = ResourceSampler(self._path / "resources.jsonl", lambda: self._state["status"])
                sampler.start()
            phase = self._state["phases"]["qa_eval"]
            phase.update(status="running", started_at=now_iso(), agent_phase_started_at=now_iso(),
                         agent_phase_started_at_epoch=time.time(), qa_concurrency=1, judge_concurrency=4)
            self._state["status"] = "evaluating"
            self._save()
            capture = (self._backend.qwen.capture_calls(self._generation_call)
                       if hasattr(getattr(self._backend, "qwen", None), "capture_calls") else contextlib.nullcontext())
            with capture, concurrent.futures.ThreadPoolExecutor(max_workers=4) as judges:
                pending = set()
                for row in rows:
                    if len(pending) >= 4:
                        done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            future.result()
                    if self._cancelled() or self._submission_stop.is_set():
                        break
                    # Exactly one Agent QA is in flight. No previous answer,
                    # context, score, or query cache is passed as prior.
                    item = self._evaluate(row, self._backend, original["memory"], version, "M4", media, run_id)
                    item["supplement_provenance"] = {"parent_run_id": parent_id, "supplement_run_id": run_id,
                        "original_item_sha256": object_hash(chosen[row["qa_id"]]), "snapshot_version": version}
                    self._item(item)
                    for turn in item.get("runtime_turns") or []:
                        failure = {"RequestUncertain": "request_uncertain", "BudgetExceeded": "budget_exceeded",
                                   "BudgetLimitChanged": "budget_limit_changed"}.get(turn.get("error_type"))
                        if failure:
                            self._block(failure, cancel_paid=True)
                    if http403_evidence(item):
                        self._block("http_403")
                    # A rejected 403 stops new QAs, not the paid Judge work
                    # already authorized for the QA that just finished.
                    if not self.cancel.is_set() and item.get("runtime_turns"):
                        pending.add(judges.submit(self._score, copy.deepcopy(item), row, judge))
                phase.update(agent_phase_finished_at=now_iso(), agent_phase_finished_at_epoch=time.time())
                phase["agent_phase_wall_ms"] = (phase["agent_phase_finished_at_epoch"] - phase["agent_phase_started_at_epoch"]) * 1000
                self._state["status"] = "judging"
                self._save()
                for future in concurrent.futures.as_completed(pending):
                    future.result()
            self._budget_gate()
            phase.update(status="done", finished_at=now_iso())
            observed = {item["qa_id"]: item for item in self._state["items"]}
            def valid_score(record):
                score = (record.get("judge") or {}).get("score")
                return type(score) is int and score in (0, 1, 2)
            complete = len(observed) == len(rows) and all(
                len(observed[row["qa_id"]].get("runtime_turns") or []) == len(dataset.turns(row))
                and observed[row["qa_id"]].get("judge_status") == "completed"
                and valid_score(observed[row["qa_id"]])
                and all(valid_score(turn)
                        and (turn.get("task_judge") or {}).get("actual_action") in {"answer", "clarify", "refuse", "none"}
                        and type((turn.get("task_judge") or {}).get("correct")) is bool
                        for turn in observed[row["qa_id"]].get("runtime_turns") or []) for row in rows)
            self._state["supplement_complete"] = complete and not self._submission_stop.is_set()
            self._state["status"] = ("blocked" if self._submission_stop.is_set() else "completed" if complete
                and all(item.get("execution_status") == "complete" for item in observed.values()) else "completed_with_errors")
        except Exception as exc:
            if writable:
                self._state.update(status="failed", supplement_complete=False, fatal_error={"type": type(exc).__name__})
            raise
        finally:
            try:
                if sampler:
                    try:
                        sampler.stop()
                        self._state["resource_summary"] = sampler.summary()
                    except Exception as exc:
                        if writable:
                            self._state.setdefault("finalization_errors", []).append(
                                {"phase": "resource_summary", "type": type(exc).__name__})
                if writable:
                    if self._backend:
                        try:
                            self._state["budget_after"] = self._backend.budget()
                        except Exception as exc:
                            self._state["budget_after"] = None
                            self._state.setdefault("finalization_errors", []).append(
                                {"phase": "budget_after", "type": type(exc).__name__})
                            self._block(getattr(exc, "code", None) or "budget_check_failed", cancel_paid=True)
                    try:
                        self._state["source_unchanged"] = (sha256(parent_path) == original["parent_sha256"]
                            and file_hashes(original["snapshot"]) == original["snapshot_hashes"]
                            and (original["memory"] / "CURRENT").read_bytes() == original["current"]
                            and sha256(original["memory"] / "experiment.json") == original["binding_sha256"])
                        self._state["parent_album_unchanged"] = file_hashes(original["memory"]) == original["album_hashes"]
                        self._state["source_unchanged"] = self._state["source_unchanged"] and self._state["parent_album_unchanged"]
                        self._state["pinned_snapshot_unchanged"] = file_hashes(memory / "snapshots" / version) == snapshot_hashes
                    except Exception as exc:
                        self._state["source_unchanged"] = self._state["pinned_snapshot_unchanged"] = None
                        self._state["parent_album_unchanged"] = None
                        self._state.setdefault("finalization_errors", []).append(
                            {"phase": "snapshot_integrity", "type": type(exc).__name__})
                    if (self._state.get("source_unchanged") is not True or self._state.get("pinned_snapshot_unchanged") is not True
                            or self._state.get("parent_album_unchanged") is not True):
                        self._block("snapshot_integrity_changed", cancel_paid=True)
                        self._state.update(status="failed", supplement_complete=False)
                    elif self._state.get("finalization_errors"):
                        self._state.update(status="failed", supplement_complete=False)
                    elif self._submission_stop.is_set() and self._state["status"] != "failed":
                        self._state.update(status="blocked", supplement_complete=False)
                    self._state.update(finished_at=now_iso(), wall_ms=(time.monotonic() - began) * 1000)
                    for item in self._state["items"]:
                        append_jsonl(self._path / "results.jsonl", item)
                    self._save()
            finally:
                self.release()
        return self._state
