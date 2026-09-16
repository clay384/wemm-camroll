"""Isolated full-chain runner. No Sentrix network or model-manager operations."""
from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import asdict, is_dataclass
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .common import atomic_json, append_jsonl, read_json, now_iso, sha256, safe_id, object_hash
from .dataset import Dataset, snapshot_media
from .judge import Judge
from .metrics import (initial_item, add_media_metrics, normalize_calls, normalize_tools,
                      native_stability, summary)
from .compat import BenchmarkRun


class CamrollBackend:
    """Direct Python adapter; never redirects evaluation into the old 8770 service."""
    def __init__(self):
        from camroll_agent.qwen import QwenClient
        from camroll_agent.wemm import EmbeddingClient
        self.qwen = QwenClient()
        self.embedding = EmbeddingClient()

    def build(self, spec, memory, progress, cancel):
        from camroll_agent.pipeline import build
        return build(spec, memory, self.qwen, self.embedding, concurrency=4, resume=True,
                     progress_callback=progress, cancel_callback=cancel)

    def agent(self, memory, config):
        from camroll_agent.agent import Agent
        from camroll_agent.library import Library
        library = Library(memory, self.embedding, config=config)
        return Agent(str(memory), llm=self.qwen, vlm=self.qwen, library=library, config=config)

    def budget(self):
        return self.qwen.usage_summary()


def configured_paths():
    root = Path(os.getenv("CAMROLL_EVAL_ROOT", "runtime/photobench-v2/evaluation")).expanduser().resolve()
    source = Path(os.getenv("CAMROLL_EVAL_SOURCE_ROOT", "runtime/photobench-v2/source/album3-max")).expanduser().resolve()
    albums = Path(os.getenv("CAMROLL_EVAL_ALBUM_ROOT", str(root / "albums"))).expanduser().resolve()
    return root, source, albums


class Runner:
    def __init__(self, root, source, albums=None, *, backend_factory=CamrollBackend,
                 judge_factory=Judge, sample_resources=True):
        self.root, self.source = Path(root).resolve(), Path(source).resolve()
        self.albums = Path(albums or self.root / "albums").resolve()
        self.backend_factory, self.judge_factory = backend_factory, judge_factory
        self.sample_resources = sample_resources
        self.cancel = threading.Event()
        self._persist_lock = threading.RLock()
        self._state = None
        self._path = None
        self._lock_file = None

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / ".runner.lock").open("a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError("another evaluation is active") from None

    def release(self):
        if self._lock_file:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _cancelled(self):
        if self._path and (self._path / "cancel.requested").exists():
            self.cancel.set()
        return self.cancel.is_set()

    def _save(self):
        with self._persist_lock:
            self._state["updated_at"] = now_iso()
            observed = {row["qa_id"]: row for row in self._state.get("items", [])}
            self._state["planned_status"] = [{**row,
                "execution_status": observed.get(row["qa_id"], {}).get("execution_status", "not_started"),
                "judge_status": observed.get(row["qa_id"], {}).get("judge_status", "not_started"),
                "executed_turn_count": len(observed.get(row["qa_id"], {}).get("runtime_turns") or [])}
                for row in self._state.get("qa_plan", [])]
            self._state["summary"] = summary(self._state)
            atomic_json(self._path / "run.json", self._state)

    def _progress(self, event):
        with self._persist_lock:
            append_jsonl(self._path / "build_events.jsonl", event)
            self._state["build_progress"] = event
            self._state["updated_at"] = now_iso()
            # Small progress file avoids repeatedly serializing all QA traces.
            atomic_json(self._path / "progress.json", {"status": "building", "event": event, "updated_at": now_iso()})

    def _judge_call(self, event):
        with self._persist_lock:
            append_jsonl(self._path / "judge_calls.jsonl", event)

    def _item(self, item):
        with self._persist_lock:
            items = {row["qa_id"]: row for row in self._state["items"]}
            items[item["qa_id"]] = item
            order = {qa: i for i, qa in enumerate(self._state["qa_ids"])}
            self._state["items"] = sorted(items.values(), key=lambda r: order[r["qa_id"]])
            atomic_json(self._path / "items" / (safe_id(item["qa_id"]) + ".json"), item)
            self._save()

    def run(self, *, run_id=None, config="M1", qa_concurrency=12, judge_concurrency=12,
            strict=True, resume=False, lock_held=False):
        if config not in {"M1", "M2", "M3", "M4"}:
            raise ValueError("invalid configuration")
        if not 1 <= qa_concurrency <= 32 or not 1 <= judge_concurrency <= 32:
            raise ValueError("concurrency must be between 1 and 32")
        if strict and (qa_concurrency, judge_concurrency) != (12, 12):
            raise ValueError("formal protocol is fixed at Agent 12 / Judge 12")
        if not lock_held:
            self.acquire()
        sampler = None
        writable = False
        try:
            dataset = Dataset(self.source, strict=strict)
            integrity = dataset.integrity()
            run_id = safe_id(run_id or time.strftime("%Y%m%d-%H%M%S") + "-camroll-" + uuid.uuid4().hex[:8])
            self._path = self.root / "runs" / run_id
            album_id = safe_id("eval-" + run_id)
            memory = self.albums / album_id
            judge = self.judge_factory(cancel=self.cancel, on_call=self._judge_call)
            protocol = judge.protocol()
            if resume:
                self._state = read_json(self._path / "run.json")
                if self._state.get("status") in {"completed", "completed_with_errors"}:
                    raise ValueError("completed runs are immutable; use a new run")
                if self._state["input_integrity"]["dataset_sha256"] != integrity["dataset_sha256"]:
                    raise ValueError("resume input hash changed")
                if self._state["config"] != config:
                    raise ValueError("resume configuration changed")
                if self._state.get("code_revision") != os.getenv("CAMROLL_RELEASE_REVISION", "unrecorded"):
                    raise ValueError("resume code revision changed; use a new run")
                if (self._state.get("qa_concurrency"), self._state.get("judge_concurrency")) != (qa_concurrency, judge_concurrency):
                    raise ValueError("resume concurrency changed")
                if self._state["input_integrity"].get("task_action_policy_sha256") != integrity["task_action_policy_sha256"]:
                    raise ValueError("resume task policy changed")
                if self._state.get("judge_protocol") != protocol:
                    raise ValueError("resume Judge protocol changed")
                if self._state.get("metric_source") != read_json(Path(__file__).with_name("provenance.json")):
                    raise ValueError("resume scoring source changed")
                writable = True
                self._state.setdefault("resumed_at", []).append(now_iso())
                self._state["timing_comparable"] = False
                (self._path / "cancel.requested").unlink(missing_ok=True)
                self.cancel.clear()
            else:
                if self._path.exists() or memory.exists():
                    raise ValueError("run and album must be new; use explicit resume")
                self._path.mkdir(parents=True)
                self._state = {"run_id": run_id, "system_id": "camroll_wemm", "mode": "full",
                    "status": "preparing", "created_at": now_iso(), "started_at": now_iso(),
                    "album_id": album_id, "dataset_id": dataset.manifest.get("album_id"), "qa_set": dataset.qa_set,
                    "formal_protocol": strict, "protocol_label": "photobench-355-100" if strict else "smoke-nonformal",
                    "qa_count": len(dataset.rows), "qa_ids": [r["qa_id"] for r in dataset.rows],
                    "qa_plan": [{"qa_id": r["qa_id"], "question": r["question"],
                                 "planned_turn_count": len(dataset.turns(r))} for r in dataset.rows],
                    "planned_turn_count": sum(len(dataset.turns(r)) for r in dataset.rows),
                    "config": config, "qa_concurrency": qa_concurrency, "judge_concurrency": judge_concurrency,
                    "caption_concurrency": 4, "cache_mode": "cold", "timing_comparable": bool(strict),
                    "input_integrity": integrity, "items": [], "phases": {},
                    "metric_source": read_json(Path(__file__).with_name("provenance.json")),
                    "code_revision": os.getenv("CAMROLL_RELEASE_REVISION", "unrecorded"),
                    "hardware_context": {"generation": "Qwen cloud API", "embedding": "existing WeMM GPU 1",
                        "gpu_uuid": "GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0", "cloud_gpu_memory_gib": None,
                        "comparison_scope": "historical_system_comparison"}}
                writable = True
                atomic_json(self._path / "input_integrity.json", integrity)
                atomic_json(self._path / "questions.json", dataset.rows)
                atomic_json(self._path / "build_spec.json", dataset.build_spec(album_id))
            self._save()
            self._state["judge_protocol"] = protocol
            self._state["judge_model"] = protocol["model"]
            backend = self.backend_factory()
            self._state.setdefault("budget_before", backend.budget())
            if self.sample_resources:
                from .telemetry import ResourceSampler
                sampler = ResourceSampler(self._path / "resources.jsonl", lambda: self._state.get("status"))
                sampler.start()
            build = self._state.get("build_summary")
            if not build or not build.get("published"):
                self._state["status"] = "building"
                self._state["phases"].setdefault("identity_seed", {"started_at": now_iso(), "status": "running"})
                self._save()
                build_start = time.monotonic()
                build = backend.build(self._path / "build_spec.json", memory, self._progress, self._cancelled)
                self._state["build_summary"] = build
                self._state["phases"]["pipeline"] = {"status": "done", "finished_at": now_iso(),
                    "wall_ms": round((time.monotonic() - build_start) * 1000, 3), "stages": build.get("stages", [])}
                self._state["phases"]["identity_seed"].update(status="done")
            if not build.get("published") or not build.get("index_version"):
                raise RuntimeError("build did not publish a complete snapshot")
            version = build["index_version"]
            if (memory / "CURRENT").read_text().strip() != version:
                raise ValueError("published snapshot pointer changed")
            media, metadata = snapshot_media(memory, version)
            source_hashes = {r["sha256"] for r in integrity["files"] if r["path"] in set(dataset.manifest["photos"])}
            if len(media) != len(source_hashes) or metadata.get("image_embedding_count", len(media)) != len(media):
                raise RuntimeError("incomplete image registration or image embedding coverage")
            self._state["index_version"] = version
            self._state["media_mapping"] = media
            atomic_json(self._path / "media_mapping.json", media)
            self._save()
            if self._cancelled():
                self._state["status"] = "cancelled"
                return self._state
            self._state["status"] = "evaluating"
            phase = self._state["phases"].setdefault("qa_eval", {})
            phase.update(status="running", started_at=now_iso(), agent_phase_started_at=now_iso(),
                         agent_phase_started_at_epoch=time.time(), qa_concurrency=qa_concurrency,
                         judge_concurrency=judge_concurrency)
            self._save()
            existing = {r["qa_id"]: r for r in self._state["items"]}
            remaining = [r for r in dataset.rows if r["qa_id"] not in existing or
                         existing[r["qa_id"]].get("judge_status") not in {"completed", "failed", "skipped"}]
            # Submit a bounded number at a time; cancellation stops future Agent
            # submissions. Already-running calls are allowed to finish safely.
            agent_start = time.monotonic()
            judge_futures = []
            rows_by_id = {r["qa_id"]: r for r in dataset.rows}
            with concurrent.futures.ThreadPoolExecutor(max_workers=judge_concurrency) as judges:
                with concurrent.futures.ThreadPoolExecutor(max_workers=qa_concurrency) as agents:
                    pending = {}
                    cursor = iter(remaining)
                    def submit_next():
                        if self._cancelled():
                            return False
                        row = next(cursor, None)
                        if row is None:
                            return False
                        old = existing.get(row["qa_id"])
                        if old and old.get("execution_status") in {"complete", "error"}:
                            future = judges.submit(self._score, copy.deepcopy(old), row, judge)
                            judge_futures.append(future)
                        else:
                            future = agents.submit(self._evaluate, row, backend, memory, version, config, media, run_id, old)
                            pending[future] = row
                        return True
                    for _ in range(qa_concurrency):
                        submit_next()
                    while pending:
                        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            row = pending.pop(future)
                            item = future.result()
                            self._item(item)
                            if not self._cancelled() and item.get("execution_status") != "cancelled":
                                judge_futures.append(judges.submit(self._score, copy.deepcopy(item), row, judge))
                            submit_next()
                    # If resume entries consumed the initial slots without an
                    # Agent future, schedule all remaining cached Judge work.
                    while submit_next():
                        if pending:
                            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                            for future in done:
                                row = pending.pop(future)
                                item = future.result()
                                self._item(item)
                                if not self._cancelled() and item.get("execution_status") != "cancelled":
                                    judge_futures.append(judges.submit(self._score, copy.deepcopy(item), row, judge))
                    for future, row in list(pending.items()):
                        item = future.result()
                        self._item(item)
                        if not self._cancelled() and item.get("execution_status") != "cancelled":
                            judge_futures.append(judges.submit(self._score, copy.deepcopy(item), row, judge))
                with self._persist_lock:
                    phase.update(agent_phase_finished_at=now_iso(), agent_phase_finished_at_epoch=time.time(),
                                 agent_phase_wall_ms=round((time.monotonic() - agent_start) * 1000, 3),
                                 agent_completed=len(self._state["items"]))
                    self._state["status"] = "judging" if not self._cancelled() else "cancelling"
                self._save()
                for future in concurrent.futures.as_completed(judge_futures):
                    future.result()  # _score checkpoints immediately on completion.
            phase.update(status="done", finished_at=now_iso())
            self._state["budget_after"] = backend.budget()
            self._state["status"] = "cancelled" if self._cancelled() else "completed"
            self._state["stop_reason"] = getattr(judge, "stop_reason", None)
            if any(r.get("execution_status") == "error" or r.get("judge_status") == "failed" for r in self._state["items"]):
                self._state["status"] = "completed_with_errors" if not self._cancelled() else "cancelled"
            if not self._cancelled() and len(self._state["items"]) != len(dataset.rows):
                self._state["status"] = "completed_with_errors"
            # Export once; per-item atomic checkpoints are authoritative during a run.
            if (self._path / "results.jsonl").exists():
                (self._path / "results.jsonl").unlink()
            for item in self._state["items"]:
                append_jsonl(self._path / "results.jsonl", item)
            return self._state
        except Exception as exc:
            if self._state and writable:
                self._state.update(status="failed", fatal_error={"type": type(exc).__name__,
                                   "reason": str(exc) if isinstance(exc, (ValueError, RuntimeError)) and "key" not in str(exc).lower() else type(exc).__name__})
                # Build exceptions carry recoverable structured summaries.
                if isinstance(getattr(exc, "summary", None), dict):
                    self._state["build_summary"] = exc.summary
                    if exc.summary.get("stop_reason") == "cancelled":
                        self._state.update(status="cancelled", fatal_error=None)
                self._save()
            raise
        finally:
            if sampler:
                sampler.stop()
            if self._state and writable:
                self._state["finished_at"] = now_iso()
                if sampler:
                    self._state["resource_summary"] = sampler.summary()
                self._save()
            self.release()

    def _evaluate(self, row, backend, memory, version, config, media, run_id, prior=None):
        item = copy.deepcopy(prior) if prior and prior.get("execution_status") == "cancelled" else initial_item(row)
        started = time.monotonic()
        timeline = {"started_at": now_iso(), "started_at_epoch": time.time()}
        history = item.get("resume_history") or []
        records = item.get("runtime_turns") or []
        last = item.get("resume_last_response") or {}
        prior_agent_ms = (item.get("timing_breakdown") or {}).get("agent_wall_ms", 0)
        agent = None
        failed = False
        for turn_index, definition in enumerate(Dataset.turns(row)):
            if turn_index < len(records):
                continue
            if self._cancelled():
                break
            response = {}
            if not failed:
                try:
                    if agent is None:
                        agent = backend.agent(memory, config)
                    response = agent.ask(definition["message"], history=history,
                        request_context={"run_id": run_id, "qa_id": row["qa_id"], "turn_index": turn_index},
                        structured_output=True, stream_model=True)
                    response = asdict(response) if is_dataclass(response) else dict(response)
                    if response.get("index_version") != version:
                        raise ValueError("Agent snapshot changed")
                    history = response.get("conversation_history") or []
                    if not isinstance(history, list):
                        raise ValueError("Agent conversation history is malformed")
                    if turn_index < len(Dataset.turns(row)) - 1 and not history:
                        raise ValueError("Agent did not retain multi-turn history")
                    if response.get("turn_outcome") == "error":
                        failed = True
                except Exception as exc:
                    failed = True
                    response = {"final_text": "", "turn_outcome": "error", "stopped_reason": "execution_error",
                                "error_type": type(exc).__name__, "index_version": version}
            record = {"index": turn_index, "message": definition["message"], "expected_action": definition["expected_action"],
                      "answer": response.get("answer") or response.get("final_text", ""),
                      "agent_status": "error" if response.get("turn_outcome") == "error" else "complete",
                      "termination_reason": response.get("stopped_reason", ""),
                      "turn_outcome": response.get("turn_outcome", "error"), "error_type": response.get("error_type")}
            add_media_metrics(record, response, media)
            records.append(record)
            offset = len(item["model_call_metrics"])
            calls = normalize_calls(response.get("model_call_metrics"), turn_index)
            item["model_call_metrics"].extend(calls)
            tools = normalize_tools(response.get("tool_trace"), turn_index, offset)
            call_indices = {call.get("call_id"): offset + index for index, call in enumerate(calls) if call.get("call_id")}
            for tool in tools:
                if tool.get("model_call_index") is None and tool.get("model_call_id") in call_indices:
                    tool["model_call_index"] = call_indices[tool["model_call_id"]]
            item["tool_trace"].extend(tools)
            last = response
            # Per-turn checkpoint prevents a cancellation from discarding a
            # paid, completed clarification turn. No automatic Agent retry.
            item.update(runtime_turns=records, resume_history=history, resume_last_response=last,
                        execution_status="cancelled", conversation_turn_count=len(records))
            if self._state is not None:
                self._item(copy.deepcopy(item))
            if failed:
                break
        final = records[-1] if records else {"answer": "", "message": row["question"], "agent_status": "cancelled",
                                              "turn_outcome": "error", "termination_reason": "cancelled"}
        cancelled = self._cancelled() and len(records) < len(Dataset.turns(row))
        item.update(answer=final["answer"], question=final["message"], runtime_turns=records,
                    conversation=records if len(records) > 1 else [], conversation_turn_count=len(records),
                    conversation_context_mode="shared_history" if len(records) > 1 else "single_turn",
                    agent_status=final["agent_status"], turn_outcome=final["turn_outcome"],
                    termination_reason=final["termination_reason"], index_version=version,
                    execution_status="cancelled" if cancelled else "error" if failed else "complete")
        add_media_metrics(item, last, media)
        item["agent_stability"] = native_stability(item)
        calls, tools = BenchmarkRun._annotate_agent_loop_timings(item["model_call_metrics"], item["tool_trace"])
        item["model_call_metrics"], item["tool_trace"] = calls, tools
        elapsed = round((time.monotonic() - started) * 1000 + prior_agent_ms, 3)
        timeline.update(agent_finished_at=now_iso(), agent_finished_at_epoch=time.time(), judge_queued_at_epoch=time.time())
        item["wall_clock_ms"] = elapsed
        item["timing_breakdown"] = BenchmarkRun._build_timing_breakdown(calls, tools, elapsed, elapsed, None, timeline, True)
        if not cancelled:
            item.pop("resume_history", None)
            item.pop("resume_last_response", None)
        return item

    def _score(self, item, row, judge):
        start, epoch = time.monotonic(), time.time()
        with self._persist_lock:
            phase = self._state["phases"]["qa_eval"]
            phase.setdefault("judge_phase_started_at", now_iso())
            phase.setdefault("judge_phase_started_at_epoch", epoch)
        timing = item["timing_breakdown"]
        timeline = timing["timeline"]
        timeline.update(judge_started_at=now_iso(), judge_started_at_epoch=epoch)
        queue_ms = max(0., (epoch - timeline.get("judge_queued_at_epoch", epoch)) * 1000)
        judge.score_item(item, row)
        elapsed = round((time.monotonic() - start) * 1000, 3)
        timeline.update(judge_finished_at=now_iso(), judge_finished_at_epoch=time.time(), judge_queue_wait_ms=queue_ms)
        item["wall_clock_ms"] = timing["agent_wall_ms"] + queue_ms + elapsed
        item["judge_ms"] = elapsed
        item["timing_breakdown"] = BenchmarkRun._build_timing_breakdown(item["model_call_metrics"], item["tool_trace"],
            item["wall_clock_ms"], timing["agent_wall_ms"], elapsed, timeline, True, queue_ms)
        with self._persist_lock:
            phase = self._state["phases"]["qa_eval"]
            phase.update(judge_phase_finished_at=now_iso(), judge_phase_finished_at_epoch=time.time(),
                         judge_phase_wall_ms=(time.time() - phase["judge_phase_started_at_epoch"]) * 1000)
        self._item(item)
        return item
