"""Independent Flash Gate A/B evaluation over one immutable WeMM album.

Index provenance and inference settings are deliberately separate. The original
album is opened read-only, with its original asset boundary and configuration.
No build, old-run resume, answer replay, or snapshot rewriting is supported.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType

from .common import append_jsonl, atomic_json, now_iso, object_hash, read_json, safe_id, sha256
from .compat import BenchmarkRun
from .dataset import Dataset, snapshot_media
from .judge import Judge
from .gate_metrics import gate_metrics
from .metrics import add_media_metrics, initial_item, native_stability, normalize_calls, normalize_tools
from .metrics import _fixed_macro, retrieval_diagnostics, summary
from .runner import Runner


PROTOCOL = "camroll-flash-gate-v1"
FIXED_SNAPSHOT = "bcc1cd1176ab39697c789a4f44c7426e"
GROUPS = MappingProxyType({"F37-Off": ("qwen3.7-flash", "off"),
    "F37-On": ("qwen3.7-flash", "repair-evidence-v1"),
    "F38-Off": ("qwen3.8-flash", "off"), "F38-On": ("qwen3.8-flash", "repair-evidence-v1")})


@dataclass(frozen=True)
class InferenceConfig:
    group: str
    generation_model: str
    gate_policy: str
    protocol_version: str = PROTOCOL
    temperature: float = 0.0
    enable_thinking: bool = False
    deadline_s: int = 600
    agent_concurrency: int = 1
    judge_concurrency: int = 4
    judge_model: str = "doubao-seed-2.0-lite"
    expected_budget_cny: int = 600
    mode: str = "M4"

    def __post_init__(self):
        if (type(self.enable_thinking) is not bool or type(self.temperature) not in (int, float)
                or any(type(value) is not int for value in (self.deadline_s, self.agent_concurrency,
                    self.judge_concurrency, self.expected_budget_cny))):
            raise ValueError("Flash Gate protocol settings have invalid types")
        if GROUPS.get(self.group) != (self.generation_model, self.gate_policy):
            raise ValueError("Unknown or altered Flash Gate group")
        if (self.protocol_version, self.temperature, self.enable_thinking, self.deadline_s,
                self.agent_concurrency, self.judge_concurrency, self.judge_model,
                self.expected_budget_cny, self.mode) != (PROTOCOL, 0., False, 600, 1, 4,
                "doubao-seed-2.0-lite", 600, "M4"):
            raise ValueError("Flash Gate protocol settings are immutable")

    def as_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return object_hash(self.as_dict())


def inference_config(group):
    if isinstance(group, InferenceConfig):
        return InferenceConfig(**group.as_dict())
    if isinstance(group, dict):
        return InferenceConfig(**group)
    if group not in GROUPS:
        raise ValueError("Unknown Flash Gate group")
    return InferenceConfig(group, *GROUPS[group])


def file_hashes(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Managed album must be a regular directory")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Linked album files are not permitted")
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths if path.is_file()}


@dataclass(frozen=True)
class IndexBinding:
    album_id: str
    source_run_id: str
    snapshot_version: str
    experiment_config_json: str
    snapshot_hashes: tuple[tuple[str, str], ...]
    album_hashes: tuple[tuple[str, str], ...]
    source_run_sha256: str
    image_count: int

    @property
    def experiment_config(self):
        return json.loads(self.experiment_config_json)

    def as_dict(self):
        return {"album_id": self.album_id, "source_run_id": self.source_run_id,
            "snapshot_version": self.snapshot_version, "experiment_config": self.experiment_config,
            "snapshot_hashes": dict(self.snapshot_hashes), "album_manifest_sha256": object_hash(dict(self.album_hashes)),
            "source_run_sha256": self.source_run_sha256, "image_count": self.image_count,
            "asset_policy": "original_album_assets_read_only", "build_performed": False}

    @property
    def fingerprint(self):
        return object_hash(self.as_dict())

    @classmethod
    def capture(cls, memory, parent_path, *, expected_snapshot=FIXED_SNAPSHOT, expected_images=355, strict_source=True):
        from camroll_agent.library import Library
        memory, parent_path = Path(memory).resolve(), Path(parent_path).resolve()
        parent = read_json(parent_path)
        if strict_source:
            if expected_snapshot != FIXED_SNAPSHOT or expected_images != 355 or parent.get("experiment") != "W38Max":
                raise ValueError("Formal Gate source must be the approved 355-image W38Max snapshot")
        elif (parent.get("synthetic_nonformal") is not True or parent.get("experiment") != "W37"
                or expected_snapshot == FIXED_SNAPSHOT):
            raise ValueError("Nonformal Gate source requires an explicitly labeled synthetic fixture")
        if (parent.get("mode") != "full"
                or parent.get("status") not in {"completed", "completed_with_errors"}
                or parent.get("album_id") != memory.name or parent.get("index_version") != expected_snapshot):
            raise ValueError("Index source differs from the approved original W38Max snapshot")
        library = Library(memory, config="M4", experiment_config=parent["experiment_config"])
        if (library.version != expected_snapshot or library.experiment.embedding_backend != "wemm"
                or library.experiment.embedding_dimension != 4096 or library.experiment.reranker_model is not None
                or library.metadata.get("image_embedding_count") != expected_images
                or len(library.images) != expected_images):
            raise ValueError("Approved WeMM snapshot configuration or image coverage changed")
        paths = library.image_paths(list(library.images))
        from PIL import Image
        for path in paths:
            with Image.open(path) as image:
                image.verify()
        snapshots = file_hashes(memory / "snapshots" / library.version)
        whole_album = file_hashes(memory)
        return cls(memory.name, parent["run_id"], library.version,
            json.dumps(parent["experiment_config"], sort_keys=True, separators=(",", ":")),
            tuple(sorted(snapshots.items())), tuple(sorted(whole_album.items())), sha256(parent_path), len(paths))

    def validate(self, memory, parent_path):
        if (sha256(parent_path) != self.source_run_sha256
                or file_hashes(memory) != dict(self.album_hashes)
                or (Path(memory) / "CURRENT").read_text().strip() != self.snapshot_version):
            raise ValueError("Original album, assets, configuration or parent run changed")


def valid_score(value):
    return type(value) is int and value in (0, 1, 2)


def valid_judgment(turn):
    action = turn.get("task_judge") or {}
    return (valid_score((turn.get("judge") or {}).get("score")) and type(action.get("correct")) is bool
            and action.get("actual_action") in {"answer", "clarify", "refuse", "none"})


def gate_summary(state):
    metrics = summary(state)
    items = state.get("items") or []
    turns = [turn for item in items for turn in item.get("runtime_turns") or []]
    eligible = [row for row in items if row.get("answerability") != "unanswerable"]
    denominator = state["planned_eligible_qa_count"]
    final = _fixed_macro(eligible, "image_retrieval_counts", denominator)
    metrics.update(final_delivery_macro=final,
        candidate_retrieval_macro=_fixed_macro(eligible, "candidate_retrieval_counts", denominator),
        retrieval_diagnostics=retrieval_diagnostics(state, denominator),
        retrieval_metric_count=denominator, image_retrieval_metric_count=denominator,
        retrieval_observed_metric_count=final["observed_metric_count"],
        retrieval_metric_scope="fixed_gate_ab_planned_image_delivery",
        retrieval_excluded_unanswerable_count=state["qa_count"] - denominator,
        benchmark_e2e_latency_excluding_judge_ms=None)
    for name in ("precision", "recall", "f1"):
        for prefix in ("retrieval_", "media_retrieval_", "image_retrieval_"):
            metrics[prefix + name + "_macro"] = final[name]
    scores = [(row.get("judge") or {}).get("score") for row in items if row.get("judge_eligible")]
    scores = [value for value in scores if valid_score(value)]
    expected_skip = sum(turn.get("judge_status") == "skipped_expected" for turn in turns)
    eligible_turns = [turn for turn in turns if turn.get("judge_eligible")]
    valid_turns = sum(valid_judgment(turn) for turn in eligible_turns)
    distribution = {"correct": scores.count(2), "partial": scores.count(1), "incorrect": scores.count(0),
                    "unscored": state["qa_count"] - len(scores)}
    metrics.update(quality_distribution_planned=distribution,
        not_started_count=max(0, state["qa_count"] - sum(bool(row.get("runtime_turns")) for row in items)),
        answer_quality_mean=sum(scores) / len(scores) if scores else None,
        judge_valid_count=len(scores), judge_distribution={str(i): scores.count(i) for i in (0, 1, 2)},
        judge_invalid_count=sum(row.get("judge_eligible", False) and not valid_score((row.get("judge") or {}).get("score")) for row in items),
        completed=sum(row.get("final_turn_executed", False) and all(
            valid_judgment(turn) if turn.get("judge_eligible") else turn.get("judge_status") == "skipped_expected"
            for turn in row.get("runtime_turns") or []) for row in items),
        quality_correct_planned_rate=distribution["correct"] / state["qa_count"],
        quality_unscored_planned_count=distribution["unscored"], quality_valid_final_count=len(scores),
        quality_expected_skip_final_count=sum(row.get("judge_status") == "skipped_expected" for row in items),
        quality_mean_scored=sum(scores) / len(scores) if scores else None,
        quality_mean_fixed_planned=sum(scores) / state["qa_count"],
        judge_eligible_turn_count=len(eligible_turns), judge_valid_turn_count=valid_turns,
        judge_expected_skip_turn_count=expected_skip,
        judge_missing_eligible_turn_count=len(eligible_turns) - valid_turns,
        unexecuted_turn_count=max(0, state["planned_turn_count"] - len(turns)),
        gate_exhausted_turn_count=sum(turn.get("origin") == "gate_exhausted" for turn in turns),
        model_contract_failure_turn_count=sum(turn.get("failure_category") == "model_contract_failure" for turn in turns),
        validator_failure_turn_count=sum(turn.get("failure_category") == "validator_failure" for turn in turns),
        infrastructure_failure_turn_count=sum(turn.get("failure_category") == "infrastructure_failure" for turn in turns),
        gate_attempt_count=sum(turn.get("gate_attempts", 0) for turn in turns),
        gate=gate_metrics(state),
        summary_version=PROTOCOL)
    return metrics


class GateBackend:
    """New inference transport over the unchanged original index configuration."""
    def __init__(self, *, index_binding, inference, run_id, journal_path, on_gate_event=None, on_verifier_call=None):
        from camroll_agent.qwen import QwenClient
        from camroll_agent.retrieval import make_retrieval_clients
        self.binding, self.inference = index_binding, inference_config(inference)
        self.embedding, self.reranker = make_retrieval_clients(index_binding.experiment_config)
        self.qwen = QwenClient(model=self.inference.generation_model, expected_limit_cny=600,
            request_namespace=run_id, request_journal_path=journal_path, strict_response_model=True)
        self.on_gate_event = on_gate_event
        self.verifier = None
        if self.inference.gate_policy != "off":
            from camroll_agent.evidence_gate import EvidenceGateVerifier
            self.verifier = EvidenceGateVerifier(client=self.qwen, on_call=on_verifier_call)

    def budget(self):
        return self.qwen.usage_summary()

    def agent(self, memory, config):
        from camroll_agent.agent import Agent
        from camroll_agent.library import Library
        from camroll_agent.retrieval import validate_embedding
        library = Library(memory, self.embedding, config=config, experiment_config=self.binding.experiment_config)
        if library.version != self.binding.snapshot_version:
            raise ValueError("Agent index version differs from approved snapshot")
        validate_embedding(self.embedding, library.experiment)
        if self.embedding.fingerprint != library.fingerprint:
            raise ValueError("Live embedding fingerprint differs from approved snapshot")
        library.image_paths(list(library.images))
        return Agent(str(memory), llm=self.qwen, vlm=self.qwen, library=library, config=config,
            deadline_s=self.inference.deadline_s, inference_model=self.inference.generation_model,
            gate_policy=self.inference.gate_policy, evidence_verifier=self.verifier, on_gate_event=self.on_gate_event)


FATAL_CODES = {"http_400", "http_401", "http_403", "http_404", "http_422", "context_limit", "missing_credentials",
    "request_uncertain", "transport_uncertain", "usage_missing", "configuration_changed", "asset_binding_error",
    "unhandled_execution_error",
    "response_model_mismatch", "gate_response_model_mismatch", "gate_payload_mismatch", "retrieval_backend_error",
    "budget_exceeded", "budget_exhausted", "budget_limit_changed", "asset_binding_changed", "index_binding_changed"}
FATAL_TYPES = {"BudgetExceeded", "BudgetLimitChanged", "RequestUncertain", "FileNotFoundError", "AssetBindingError",
    "WeMMError", "QwenRetrievalError", "ConnectionError"}
PROVIDER_CONFIGURATION_CODES = {"invalid_request", "invalid_image", "invalid_schema", "unknown_model", "missing_endpoint"}


def infrastructure_failure(value):
    """Inspect structured operational metadata, never answer/prompt strings."""
    if not isinstance(value, dict):
        return None
    for key in ("error_code", "provider_code", "code", "stopped_reason", "termination_reason"):
        if isinstance(value.get(key), str) and value[key] in FATAL_CODES:
            return value[key]
    if value.get("exception_type") == "ProviderError" or value.get("error_type") == "ProviderError":
        for key in ("error_code", "provider_code"):
            if isinstance(value.get(key), str) and value[key] in PROVIDER_CONFIGURATION_CODES:
                return value[key]
    if value.get("http_status") in (400, 401, 403, 404, 422) or value.get("status_code") in (400, 401, 403, 404, 422):
        return "http_" + str(value.get("http_status") or value["status_code"])
    # An Off transport may complete its existing schema capability fallback.
    # A blocked Gate output has no such fallback and is a configuration fault.
    if value.get("state") == "blocked" and value.get("error_code") == "unsupported_schema":
        return "unsupported_schema"
    for key in ("error_type", "exception_type"):
        if isinstance(value.get(key), str) and value[key] in FATAL_TYPES:
            return value[key]
    for key in ("error", "result"):
        result = infrastructure_failure(value.get(key))
        if result:
            return result
    for key in ("model_call_metrics", "model_calls", "tool_trace"):
        for row in value.get(key) or []:
            result = infrastructure_failure(row)
            if result:
                return result
    for event in value.get("gate_events") or []:
        if isinstance(event, dict) and event.get("state") == "blocked":
            result = infrastructure_failure(event)
            if result:
                return result
    return None


def validator_failure(value):
    """Separate the independent checker's failed output from Flash's output.

    Successful format repair remains a normal verifier call. The verifier's
    exhausted-format code lives in the blocked Gate event; its final physical
    request records parse_status/format_attempt, so both pieces are required.
    """
    if not isinstance(value, dict):
        return None
    calls = (value.get("model_call_metrics") or []) + (value.get("model_calls") or [])
    checker_calls = [row for row in calls if isinstance(row, dict) and any(
        row.get(field) == "evidence_gate" for field in ("role", "call_type", "original_call_type"))]
    if not checker_calls:
        return None
    latest = checker_calls[-1]
    code = latest.get("error_code")
    if isinstance(code, str) and code in {"gate_invalid_verdict", "refusal"}:
        return code
    exhausted = any(isinstance(event, dict) and event.get("state") == "blocked"
        and event.get("error_code") == "gate_invalid_verdict" for event in value.get("gate_events") or [])
    if (exhausted and latest.get("parse_status") == "invalid"
            and type(latest.get("format_attempt")) is int and latest["format_attempt"] == 1):
        return "gate_invalid_verdict"
    return None


class GateRunner(Runner):
    def __init__(self, *args, backend_factory=GateBackend, **kwargs):
        super().__init__(*args, backend_factory=backend_factory, **kwargs)
        self._backend = None
        self._binding = None
        self._memory = self._parent_path = None
        self._stop_reason = None
        self._call_context = threading.local()

    def _stop(self, reason):
        self._stop_reason = self._stop_reason or reason
        self.cancel.set()

    def _cancelled(self):
        if self._path and (self._path / "cancel.requested").exists():
            self._stop("cancelled")
        if self._backend is not None and not self.cancel.is_set():
            try:
                status = self._backend.budget()
                if status.get("limit_cny") != 600:
                    self._stop("budget_limit_changed")
                elif status.get("uncertain_requests", 0):
                    self._stop("request_uncertain")
                elif status.get("remaining_cny", 0) <= 0:
                    self._stop("budget_exceeded")
            except Exception as exc:
                self._stop(getattr(exc, "code", None) or "budget_check_failed")
        return self.cancel.is_set()

    def _event(self, filename, event):
        event = {**copy.deepcopy(event), "evaluation_context": copy.deepcopy(getattr(self._call_context, "value", {}))}
        with self._persist_lock:
            append_jsonl(self._path / filename, event)
        fatal = infrastructure_failure(event)
        if fatal:
            self._stop(fatal)

    def _judge_call(self, event):
        self._event("judge_calls.jsonl", event)

    def _evaluate(self, row, backend, memory, version, config, media, run_id):
        """One fresh conversation, including failed model turns, without retries."""
        item = initial_item(row)
        started = time.monotonic()
        timeline = {"started_at": now_iso(), "started_at_epoch": time.time()}
        records, history, last = [], [], {}
        definitions = Dataset.turns(row)
        item["gate_events"] = []
        try:
            self._binding.validate(memory, self._parent_path)
            agent = backend.agent(memory, config)
            if (agent.library.memory_dir.resolve() != Path(memory).resolve()
                    or agent.library.version != version):
                raise ValueError("Agent original album binding changed")
            agent.library.image_paths(list(agent.library.images))
        except Exception as exc:
            self._stop("index_binding_changed")
            item["infrastructure_error"] = {"error_type": type(exc).__name__, "error_code": "index_binding_changed"}
            agent = None
        for index, definition in enumerate(definitions):
            if self._cancelled():
                break
            self._call_context.value = {"run_id": run_id, "qa_id": row["qa_id"], "turn_index": index, "role": "agent"}
            response = {}
            try:
                response = agent.ask(definition["message"], history=history,
                    request_context={"run_id": run_id, "qa_id": row["qa_id"], "turn_index": index},
                    structured_output=True, stream_model=True)
                response = asdict(response) if is_dataclass(response) else dict(response)
                if response.get("index_version") != version:
                    response.update(origin="system_failure", error_code="index_binding_changed")
                returned_history = response.get("conversation_history")
                if not isinstance(returned_history, list) or not returned_history:
                    response.update(origin="system_failure", error_code="configuration_changed")
                else:
                    history = returned_history
            except Exception as exc:
                response = {"final_text": "", "turn_outcome": "error", "stopped_reason": "execution_error",
                    "error_type": type(exc).__name__, "error_code": getattr(exc, "code", None) or "unhandled_execution_error",
                    "origin": "system_failure", "index_version": version}
            raw_path = self._path / "turn_attempts" / safe_id(row["qa_id"]) / (str(index) + ".json")
            atomic_json(raw_path, {"context": self._call_context.value, "response": response})
            fatal = infrastructure_failure(response)
            origin = response.get("origin")
            if origin not in {"model", "system_failure", "gate_exhausted"}:
                origin, fatal = "system_failure", "configuration_changed"
            outcome = response.get("turn_outcome", "error")
            if origin == "model" and outcome not in {"answer", "clarify", "refuse"}:
                origin = "system_failure"
            checker_failure = validator_failure(response) if origin == "system_failure" and not fatal else None
            category = ("infrastructure_failure" if fatal else "validator_failure" if checker_failure else
                        "model_contract_failure" if origin == "system_failure" else
                        "gate_exhausted" if origin == "gate_exhausted" else None)
            eligible = origin == "model" and not fatal
            if fatal:
                origin = "system_failure"
                self._stop(fatal)
            record = {"index": index, "message": definition["message"], "expected_action": definition["expected_action"],
                "answer": response.get("answer") or response.get("final_text", ""), "origin": origin, "answer_origin": origin,
                "failure_category": category, "judge_eligible": eligible,
                "validator_failure_code": checker_failure,
                "agent_status": "complete" if eligible else "error", "turn_outcome": outcome,
                "termination_reason": response.get("stopped_reason", ""), "error_type": response.get("error_type"),
                "error_code": response.get("error_code"), "gate_status": response.get("gate_status", "off"),
                "gate_attempts": response.get("gate_attempts", 0), "gate_events": response.get("gate_events") or [],
                "raw_attempt_path": str(raw_path.relative_to(self._path)),
                "judge_status": "pending" if eligible else "skipped_expected",
                "judge": {"score": None, "reason": "pending" if eligible else category},
                "task_judge": {"correct": None, "actual_action": None, "reason": "pending" if eligible else category},
                "evidence_judge": {"score": None, "reason": "disabled"}}
            metric_response = response if eligible else {**response, "selected_image_ids": [], "selected_event_ids": []}
            add_media_metrics(record, metric_response, media)
            records.append(record)
            item["gate_events"].extend({**event, "turn_index": index} for event in record["gate_events"])
            offset = len(item["model_call_metrics"])
            calls = normalize_calls(response.get("model_call_metrics"), index)
            item["model_call_metrics"].extend(calls)
            tools = normalize_tools(response.get("tool_trace"), index, offset)
            call_indices = {call.get("call_id"): offset + i for i, call in enumerate(calls) if call.get("call_id")}
            for tool in tools:
                if tool.get("model_call_index") is None and tool.get("model_call_id") in call_indices:
                    tool["model_call_index"] = call_indices[tool["model_call_id"]]
            item["tool_trace"].extend(tools)
            last = metric_response
            item.update(runtime_turns=records, conversation_history=history, execution_status="running",
                        conversation_turn_count=len(records))
            self._item(copy.deepcopy(item))
        final_executed = len(records) == len(definitions)
        final = records[-1] if records else {}
        final_eligible = final_executed and final.get("judge_eligible", False)
        status = "pending" if any(turn["judge_eligible"] for turn in records) else "skipped_expected" if final_executed else "not_started"
        item.update(answer=final.get("answer", ""), question=final.get("message", row["question"]), runtime_turns=records,
            conversation=records if len(definitions) > 1 else [], conversation_turn_count=len(records),
            conversation_context_mode="shared_history" if len(definitions) > 1 else "single_turn",
            agent_status=final.get("agent_status", "not_started"), turn_outcome=final.get("turn_outcome", "error"),
            origin=final.get("origin"), answer_origin=final.get("origin"), failure_category=final.get("failure_category"),
            validator_failure_code=final.get("validator_failure_code"),
            gate_status=final.get("gate_status"), gate_attempts=sum(turn["gate_attempts"] for turn in records),
            termination_reason=final.get("termination_reason", self._stop_reason), index_version=version,
            final_turn_executed=final_executed, judge_eligible=final_eligible, judge_status=status,
            judge=copy.deepcopy(final.get("judge")) if final_executed else {"score": None, "reason": "final_turn_not_executed"},
            task_judge=copy.deepcopy(final.get("task_judge")) if final_executed else {"correct": None},
            task_judges=[turn["task_judge"] for turn in records],
            execution_status="not_started" if not records else "incomplete" if not final_executed else
                "error" if any(not turn["judge_eligible"] for turn in records) else "complete")
        add_media_metrics(item, last if final_executed else {}, media)
        item["agent_stability"] = native_stability(item)
        calls, tools = BenchmarkRun._annotate_agent_loop_timings(item["model_call_metrics"], item["tool_trace"])
        item["model_call_metrics"], item["tool_trace"] = calls, tools
        elapsed = round((time.monotonic() - started) * 1000, 3)
        timeline.update(agent_finished_at=now_iso(), agent_finished_at_epoch=time.time(), judge_queued_at_epoch=time.time())
        item["wall_clock_ms"] = elapsed
        item["timing_breakdown"] = BenchmarkRun._build_timing_breakdown(calls, tools, elapsed, elapsed, None, timeline, True)
        return item

    def _score(self, item, row, judge):
        """Frozen quality/action Judges, excluding failures but retaining context."""
        start, epoch = time.monotonic(), time.time()
        with self._persist_lock:
            phase = self._state["phases"]["qa_eval"]
            phase.setdefault("judge_phase_started_at", now_iso())
            phase.setdefault("judge_phase_started_at_epoch", epoch)
        timing = item["timing_breakdown"]
        timeline = timing["timeline"]
        timeline.update(judge_started_at=now_iso(), judge_started_at_epoch=epoch)
        queue_ms = max(0., (epoch - timeline.get("judge_queued_at_epoch", epoch)) * 1000)
        definitions, records = Dataset.turns(row), item["runtime_turns"]
        for index, record in enumerate(records):
            if not record["judge_eligible"]:
                continue
            if self._cancelled():
                record["judge_status"] = "cancelled"
                continue
            self._call_context.value = {"run_id": self._state["run_id"], "qa_id": row["qa_id"], "turn_index": index, "role": "judge"}
            namespace = self._state["run_id"] + ":" + row["qa_id"] + ":turn:" + str(index) + ":judge"
            context = judge.request_context(namespace) if hasattr(judge, "request_context") else contextlib.nullcontext()
            reference = definitions[index].get("reference_answer") or (
                "应要求用户提供至少一个具体、可用于后续检索的锚点；询问形式不限。"
                if record["expected_action"] == "clarify" else row.get("answer", ""))
            shared = dict(task_type=row.get("task_type"), question_type=row.get("question_type"),
                answerability=row.get("answerability"), conversation=records[:index + 1])
            try:
                with context:
                    record["task_judge"] = judge._judge_task_action(record["message"], record["answer"], record["expected_action"],
                        record["agent_status"], record["termination_reason"], reference=reference, **shared)
                    if not self._cancelled():
                        record["judge"] = judge._judge(record["message"], reference, record["answer"],
                            expected_action=record["expected_action"], **shared)
                record["judge_status"] = "completed" if valid_judgment(record) else "failed"
            except Exception as exc:
                record.update(judge_status="failed", judge_error_type=type(exc).__name__)
                fatal = infrastructure_failure({"error_type": type(exc).__name__, "error_code": getattr(exc, "code", None)})
                if fatal:
                    self._stop(fatal)
            self._item(copy.deepcopy(item))
        final = records[-1] if records else {}
        final_executed = len(records) == len(definitions)
        item["task_judges"] = [record["task_judge"] for record in records]
        item["task_judge"] = copy.deepcopy(final.get("task_judge")) if final_executed else {"correct": None}
        item["judge"] = copy.deepcopy(final.get("judge")) if final_executed else {"score": None, "reason": "final_turn_not_executed"}
        item["judge_status"] = "incomplete" if not final_executed else "failed" if any(
            turn["judge_eligible"] and not valid_judgment(turn) for turn in records) else (
            "completed" if final.get("judge_eligible") else "skipped_expected")
        elapsed = round((time.monotonic() - start) * 1000, 3)
        timeline.update(judge_finished_at=now_iso(), judge_finished_at_epoch=time.time(), judge_queue_wait_ms=queue_ms)
        item["wall_clock_ms"] = timing["agent_wall_ms"] + queue_ms + elapsed
        item["judge_ms"] = elapsed
        item["timing_breakdown"] = BenchmarkRun._build_timing_breakdown(item["model_call_metrics"], item["tool_trace"],
            item["wall_clock_ms"], timing["agent_wall_ms"], elapsed, timeline, True, queue_ms)
        with self._persist_lock:
            phase.update(judge_phase_finished_at=now_iso(), judge_phase_finished_at_epoch=time.time(),
                judge_phase_wall_ms=(time.time() - phase["judge_phase_started_at_epoch"]) * 1000)
        self._item(item)
        return item

    def _save(self):
        with self._persist_lock:
            state = self._state
            state["updated_at"] = now_iso()
            observed = {row["qa_id"]: row for row in state.get("items", [])}
            state["planned_status"] = [{**plan,
                "execution_status": observed.get(plan["qa_id"], {}).get("execution_status", "not_started"),
                "judge_status": observed.get(plan["qa_id"], {}).get("judge_status", "not_started"),
                "executed_turn_count": len(observed.get(plan["qa_id"], {}).get("runtime_turns") or []),
                "turns": [{"index": index, "execution_status": turn.get("agent_status", "not_started"),
                    "judge_status": turn.get("judge_status", "not_started"), "answer_origin": turn.get("origin")}
                    for index in range(plan["planned_turn_count"])
                    for turn in [next((value for value in observed.get(plan["qa_id"], {}).get("runtime_turns") or []
                        if value.get("index") == index), {})]]}
                for plan in state["qa_plan"]]
            state["summary"] = gate_summary(state)
            if self._stop_reason:
                state["stop_reason"] = self._stop_reason
            atomic_json(self._path / "run.json", state)

    def run_gate(self, *, run_id, suite_id, group, source_album_id, source_run_id,
                 source_evaluation_root=None, strict=True, expected_snapshot=FIXED_SNAPSHOT, lock_held=False):
        if not lock_held:
            self.acquire()
        writable, sampler = False, None
        began = time.monotonic()
        try:
            profile = inference_config(group)
            dataset = Dataset(self.source, strict=strict)
            parent_root = Path(source_evaluation_root or self.root)
            self._parent_path = parent_root / "runs" / safe_id(source_run_id) / "run.json"
            self._memory = self.albums / safe_id(source_album_id)
            self._path = self.root / "runs" / safe_id(run_id)
            if self._path.exists():
                raise ValueError("Gate runs are one-shot; existing run cannot be resumed or overwritten")
            claim = self.root / "gate_claims" / safe_id(suite_id) / (profile.group + ".json")
            claim.parent.mkdir(parents=True, exist_ok=True)
            with claim.open("x") as stream:
                json.dump({"run_id": run_id, "inference_fingerprint": profile.fingerprint, "created_at": now_iso()}, stream)
            claim.chmod(0o600)
            self._path.mkdir(mode=0o700, parents=True)
            integrity = dataset.integrity()
            self._state = {"run_id": run_id, "suite_id": suite_id, "system_id": "camroll_wemm", "mode": "gate_ab",
                "protocol_label": PROTOCOL, "gate_summary_version": PROTOCOL, "formal_protocol": strict, "status": "preparing", "config": "M4",
                "experiment": profile.group, "inference_config": profile.as_dict(), "inference_fingerprint": profile.fingerprint,
                "qa_concurrency": 1, "judge_concurrency": 4, "created_at": now_iso(), "started_at": now_iso(),
                "source_run_id": source_run_id, "album_id": source_album_id, "query_memory": str(self._memory),
                "index_version": expected_snapshot, "qa_set": dataset.qa_set, "dataset_id": dataset.manifest["album_id"],
                "qa_count": len(dataset.rows), "qa_ids": [row["qa_id"] for row in dataset.rows],
                "qa_plan": [{"qa_id": row["qa_id"], "question": row["question"],
                    "planned_turn_count": len(dataset.turns(row))} for row in dataset.rows],
                "planned_turn_count": sum(len(dataset.turns(row)) for row in dataset.rows),
                "planned_eligible_qa_count": sum(row.get("answerability") != "unanswerable" for row in dataset.rows),
                "items": [], "phases": {"qa_eval": {}}, "input_integrity": integrity,
                "build_skipped": True, "timing_comparable": False, "build_cost_cny": 0.,
                "generation_request_namespace": run_id, "query_cache_namespace": run_id,
                "query_cache_policy": "fresh_run_client_no_persistent_query_cache_original_index_namespace_retained",
                "code_revision": os.getenv("CAMROLL_RELEASE_REVISION", "unrecorded"),
                "adapter_sha256": sha256(__file__), "gate_complete": False,
                "metric_source": {"protocol": PROTOCOL, "quality_denominator": len(dataset.rows)}}
            writable = True
            atomic_json(self._path / "questions.json", dataset.rows)
            atomic_json(self._path / "input_integrity.json", integrity)
            self._save()
            parent = read_json(self._parent_path)
            if parent.get("input_integrity") != integrity:
                raise ValueError("Gate input dataset differs from the original frozen album")
            self._binding = IndexBinding.capture(self._memory, self._parent_path,
                expected_snapshot=expected_snapshot, expected_images=len(dataset.photos), strict_source=strict)
            if self._binding.source_run_id != source_run_id:
                raise ValueError("Source run identity does not match its requested directory")
            media, metadata = snapshot_media(self._memory, self._binding.snapshot_version)
            self._state.update(index_binding=self._binding.as_dict(), index_binding_fingerprint=self._binding.fingerprint,
                snapshot_metadata=metadata, media_mapping=media, experiment_config=self._binding.experiment_config,
                asset_binding_validation={"validated_images": self._binding.image_count, "decoded_images": self._binding.image_count,
                    "before_paid_client_creation": True, "asset_policy": "original_album_assets_read_only"})
            atomic_json(self._path / "index-binding.json", self._binding.as_dict())
            atomic_json(self._path / "source-album-hashes.json", dict(self._binding.album_hashes))
            atomic_json(self._path / "media_mapping.json", media)
            journal = self._path / "request-journal.sqlite"
            self._state["request_journal_path"] = str(journal)
            self._save()
            self._backend = self.backend_factory(index_binding=self._binding, inference=profile, run_id=run_id,
                journal_path=journal, on_gate_event=lambda event: self._event("gate_events.jsonl", event),
                on_verifier_call=lambda event: self._event("verifier_calls.jsonl", event))
            judge = self.judge_factory(cancel=self.cancel, on_call=self._judge_call, model=profile.judge_model,
                request_namespace=run_id, request_journal_path=journal, expected_limit_cny=600)
            if judge.protocol() != parent["judge_protocol"]:
                raise ValueError("Final Judge protocol differs from original frozen evaluation")
            self._state.update(judge_protocol=judge.protocol(), judge_model=profile.judge_model, budget_before=self._backend.budget())
            if self.sample_resources:
                from .telemetry import ResourceSampler
                sampler = ResourceSampler(self._path / "resources.jsonl", lambda: self._state["status"])
                sampler.start()
            phase = self._state["phases"]["qa_eval"]
            phase.update(status="running", started_at=now_iso(), agent_phase_started_at=now_iso(),
                         agent_phase_started_at_epoch=time.time(), qa_concurrency=1, judge_concurrency=4)
            self._state["status"] = "evaluating"
            self._save()
            client = getattr(self._backend, "qwen", None)
            capture = client.capture_calls(lambda event: self._event("model_calls.jsonl", event)) if hasattr(client, "capture_calls") else contextlib.nullcontext()
            with capture, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending = set()
                for row in dataset.rows:
                    if len(pending) >= 4:
                        done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            future.result()
                    if self._cancelled():
                        break
                    item = self._evaluate(row, self._backend, self._memory, self._binding.snapshot_version, "M4", media, run_id)
                    self._item(item)
                    if not self.cancel.is_set() and any(turn.get("judge_eligible") for turn in item["runtime_turns"]):
                        pending.add(pool.submit(self._score, copy.deepcopy(item), row, judge))
                phase.update(agent_phase_finished_at=now_iso(), agent_phase_finished_at_epoch=time.time())
                phase["agent_phase_wall_ms"] = (phase["agent_phase_finished_at_epoch"] - phase["agent_phase_started_at_epoch"]) * 1000
                self._state["status"] = "judging"
                self._save()
                for future in concurrent.futures.as_completed(pending):
                    future.result()
            self._cancelled()
            phase.update(status="done", finished_at=now_iso())
            self._state["status"] = "blocked" if self.cancel.is_set() else "completed_with_errors" if any(
                row["execution_status"] != "complete" or any(turn.get("judge_eligible") and not valid_judgment(turn)
                    for turn in row.get("runtime_turns") or []) for row in self._state["items"]) else "completed"
        except Exception as exc:
            if writable:
                self._state.update(status="blocked", fatal_error={"type": type(exc).__name__}, gate_complete=False)
                self._stop(getattr(exc, "code", None) or "configuration_or_asset_preflight_failed")
            raise
        finally:
            try:
                if writable:
                    if sampler:
                        try:
                            sampler.stop(); self._state["resource_summary"] = sampler.summary()
                        except Exception as exc:
                            self._state.setdefault("finalization_errors", []).append({"phase": "resources", "type": type(exc).__name__})
                    if self._backend:
                        try:
                            self._state["budget_after"] = self._backend.budget()
                            self._cancelled()
                        except Exception as exc:
                            self._stop("budget_check_failed")
                            self._state["budget_after"] = None
                    if self._binding:
                        try:
                            self._binding.validate(self._memory, self._parent_path)
                            self._state["source_unchanged"] = True
                        except Exception:
                            self._state["source_unchanged"] = False
                            self._stop("index_binding_changed")
                    else:
                        self._state["source_unchanged"] = None
                    if self.cancel.is_set():
                        self._state["status"] = "blocked"
                    self._state.update(finished_at=now_iso(), wall_ms=(time.monotonic() - began) * 1000)
                    self._state["gate_complete"] = self._is_complete()
                    for item in self._state["items"]:
                        append_jsonl(self._path / "results.jsonl", item)
                    self._save()
            finally:
                self.release()
        return self._state

    def _is_complete(self):
        state = self._state
        observed = {item["qa_id"]: item for item in state["items"]}
        if (self.cancel.is_set() or state.get("source_unchanged") is not True or state.get("finalization_errors")
                or len(observed) != state["qa_count"]):
            return False
        for plan in state["qa_plan"]:
            item = observed[plan["qa_id"]]
            turns = item.get("runtime_turns") or []
            if len(turns) != plan["planned_turn_count"]:
                return False
            for turn in turns:
                if turn.get("judge_eligible"):
                    if not valid_judgment(turn):
                        return False
                elif turn.get("judge_status") != "skipped_expected" or (turn.get("judge") or {}).get("score") is not None:
                    return False
            if item.get("judge_eligible") and not valid_judgment(item):
                return False
        return True
