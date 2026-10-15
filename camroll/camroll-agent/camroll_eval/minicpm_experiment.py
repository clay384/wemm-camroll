"""Versioned MiniCPM inference over the unchanged original WeMM album.

This is a new protocol, not a continuation or alteration of Flash Gate v1.
The only per-turn infrastructure exception is an identified context limit; it
remains an unscored failure while the remaining planned turns execute.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

from .common import object_hash, read_json, safe_id, sha256
from .dataset import Dataset
from .gate_experiment import (FIXED_SNAPSHOT, GateBackend, GateRunner, IndexBinding,
                              gate_summary, infrastructure_failure)

PROTOCOL = "minicpm47-thinking-m4-off-v1"
GROUP = "MiniCPM47-Thinking-On-Off"
SDK_DEFAULTS_PROTOCOL = "minicpm47-sdk-defaults-m4-off-v1"
SDK_INTERFACE_PROTOCOL = "minicpm47-sdk-defaults-m4-off-v2"
SDK_DEFAULTS_GROUP = "MiniCPM47-SDK-Defaults-Off"
SELFHOST_PROTOCOL = "minicpm47-q6-thinking-m4-off-v1"
SELFHOST_MODEL = "minicpm-v4.7-35b-a3b-q6-thinking"
SUPPORTED_PROTOCOLS = frozenset({PROTOCOL, SDK_DEFAULTS_PROTOCOL, SDK_INTERFACE_PROTOCOL, SELFHOST_PROTOCOL})
MODEL = "MiniCPM-V-4.7-35B-A3B"


@dataclass(frozen=True)
class MiniCPMInferenceConfig:
    group: str = GROUP
    provider: str = "modelbest"
    generation_model: str = MODEL
    gate_policy: str = "off"
    protocol_version: str = PROTOCOL
    temperature: float | None = 0.0
    enable_thinking: bool | None = True
    stream: bool | None = False
    max_output_tokens: int | None = 8192
    context_limit: int = 256000
    deadline_s: int = 600
    agent_concurrency: int = 1
    judge_concurrency: int = 4
    judge_model: str = "doubao-seed-2.0-lite"
    expected_budget_cny: int = 600
    mode: str = "M4"
    decision_mode: str = "native_tools"
    json_mode: str = "json_schema"
    request_profile: str = "fixed-v1"
    thinking_mode: str = "on"
    acceptance_policy: str = "facts-and-interface-v1"
    adapter_version: str = "legacy-v1"

    def __post_init__(self):
        fixed = {"group": GROUP, "provider": "modelbest", "generation_model": MODEL, "gate_policy": "off",
            "protocol_version": PROTOCOL, "temperature": 0.0, "enable_thinking": True,
            "stream": False, "max_output_tokens": 8192, "context_limit": 256000,
            "deadline_s": 600, "agent_concurrency": 1, "judge_concurrency": 4,
            "judge_model": "doubao-seed-2.0-lite", "expected_budget_cny": 600, "mode": "M4",
            "request_profile": "fixed-v1", "thinking_mode": "on",
            "acceptance_policy": "facts-and-interface-v1", "adapter_version": "legacy-v1"}
        if self.request_profile == "sdk-defaults-v1":
            fixed.update(group=SDK_DEFAULTS_GROUP, protocol_version=SDK_DEFAULTS_PROTOCOL,
                         temperature=None, enable_thinking=None, stream=None, max_output_tokens=None,
                         request_profile="sdk-defaults-v1", thinking_mode="unspecified")
            if self.acceptance_policy == "interface-only-v2":
                fixed.update(protocol_version=SDK_INTERFACE_PROTOCOL,
                             acceptance_policy="interface-only-v2", adapter_version="modelbest-json-v2")
        if self.request_profile == "selfhost-q6-v1":
            fixed.update(group="MiniCPM47-Q6-Thinking-Off", provider="llamacpp-selfhost",
                generation_model=SELFHOST_MODEL, protocol_version=SELFHOST_PROTOCOL,
                context_limit=262144, deadline_s=1200, request_profile="selfhost-q6-v1",
                thinking_mode="server-on", acceptance_policy="interface-only-v2", adapter_version="selfhost-q6-v1")
        for name, expected in fixed.items():
            value = getattr(self, name)
            valid_type = (type(value) in (int, float) if name == "temperature" and expected is not None
                          else type(value) is type(expected))
            if not valid_type or value != expected:
                raise ValueError("MiniCPM protocol setting differs: " + name)
        if self.decision_mode not in {"native_tools", "json_decision"}:
            raise ValueError("Unknown MiniCPM decision capability")
        if self.json_mode not in {"json_schema", "json_object", "prompt"}:
            raise ValueError("Unknown MiniCPM JSON capability")

    def as_dict(self):
        value = asdict(self)
        if self.acceptance_policy == "facts-and-interface-v1":
            value.pop("acceptance_policy")
            value.pop("adapter_version")
        if self.request_profile == "fixed-v1":
            # Preserve already frozen v1 files and their exact fingerprints.
            value.pop("request_profile")
            value.pop("thinking_mode")
        return value

    @classmethod
    def sdk_defaults(cls, *, decision_mode="native_tools", json_mode="json_schema",
                     acceptance_policy="facts-and-interface-v1"):
        return cls(group=SDK_DEFAULTS_GROUP,
                   protocol_version=SDK_INTERFACE_PROTOCOL if acceptance_policy == "interface-only-v2" else SDK_DEFAULTS_PROTOCOL,
                   temperature=None, enable_thinking=None, stream=None, max_output_tokens=None,
                   request_profile="sdk-defaults-v1", thinking_mode="unspecified",
                   decision_mode=decision_mode, json_mode=json_mode, acceptance_policy=acceptance_policy,
                   adapter_version="modelbest-json-v2" if acceptance_policy == "interface-only-v2" else "legacy-v1")

    @classmethod
    def selfhost(cls, **modes):
        return cls(group="MiniCPM47-Q6-Thinking-Off", provider="llamacpp-selfhost",
            generation_model=SELFHOST_MODEL, protocol_version=SELFHOST_PROTOCOL,
            context_limit=262144, deadline_s=1200, request_profile="selfhost-q6-v1",
            thinking_mode="server-on", acceptance_policy="interface-only-v2", adapter_version="selfhost-q6-v1", **modes)

    def client_options(self):
        options = {key: getattr(self, key) for key in
                   ("decision_mode", "json_mode", "request_profile", "thinking_mode")}
        if self.adapter_version == "modelbest-json-v2":
            options["compatibility_mode"] = "modelbest-v2"
        if self.request_profile == "selfhost-q6-v1":
            options["compatibility_mode"] = "selfhost-q6-v1"
        return options

    def matches_request(self, body):
        """Check the frozen generation contract in SDK kwargs or logged wire JSON."""
        if not isinstance(body, dict) or body.get("model") != self.generation_model or not isinstance(body.get("messages"), list):
            return False
        common = {"model", "messages", "tools", "tool_choice", "response_format"}
        if self.request_profile == "sdk-defaults-v1":
            # Absence matters: null or an explicit stream=False also changes the
            # minimal request validated against the provider's SDK example.
            return not set(body) - common
        if self.request_profile == "selfhost-q6-v1":
            return (not set(body) - common - {"max_tokens", "temperature", "stream"}
                and type(body.get("max_tokens")) is int and body["max_tokens"] == 8192
                and type(body.get("temperature")) in (int, float) and body["temperature"] == 0
                and body.get("stream") is False)
        extra = {"max_tokens", "temperature", "stream", "extra_body", "chat_template_kwargs"}
        if set(body) - common - extra or ("extra_body" in body and "chat_template_kwargs" in body):
            return False
        thinking = body.get("extra_body", {"chat_template_kwargs": body.get("chat_template_kwargs")})
        if (not isinstance(thinking, dict) or set(thinking) != {"chat_template_kwargs"}
                or not isinstance(thinking["chat_template_kwargs"], dict)
                or set(thinking["chat_template_kwargs"]) != {"enable_thinking"}
                or thinking["chat_template_kwargs"]["enable_thinking"] is not True):
            return False
        return (type(body.get("max_tokens")) is int and body["max_tokens"] == 8192
                and type(body.get("temperature")) in (int, float) and body["temperature"] == 0
                and body.get("stream") is False)

    @property
    def fingerprint(self):
        return object_hash(self.as_dict())


def inference_config(value=None):
    if value is None or value == GROUP:
        return MiniCPMInferenceConfig()
    if value == SDK_DEFAULTS_GROUP:
        return MiniCPMInferenceConfig.sdk_defaults()
    if isinstance(value, MiniCPMInferenceConfig):
        return MiniCPMInferenceConfig(**value.as_dict())
    if isinstance(value, dict):
        return MiniCPMInferenceConfig(**value)
    raise ValueError("Unknown MiniCPM inference profile")


class MiniCPMBackend(GateBackend):
    """Original Library/assets and retrieval, with one explicit cloud provider."""
    def __init__(self, *, index_binding, inference, run_id, journal_path,
                 on_gate_event=None, on_verifier_call=None, ledger=None,
                 pricing_evidence=None, api_key=None, client_factory=None):
        from camroll_agent.retrieval import make_retrieval_clients
        self.binding, self.inference = index_binding, inference_config(inference)
        if client_factory is None:
            client_factory = client_class(self.inference)
        self.embedding, self.reranker = make_retrieval_clients(index_binding.experiment_config)
        if pricing_evidence is None:
            pricing_path = os.getenv("CAMROLL_MINICPM_PRICING_EVIDENCE")
            if not pricing_path:
                raise ValueError("Explicit MiniCPM pricing evidence is required")
            pricing_evidence = read_json(Path(pricing_path))
        options = {"ledger": ledger} if ledger is not None else {}
        self.qwen = client_factory(pricing_evidence=copy.deepcopy(pricing_evidence),
            api_key=api_key or os.getenv("MODELBEST_API_KEY"),
            base_url=os.getenv("CAMROLL_MINICPM_BASE_URL") if self.inference.provider == "llamacpp-selfhost" else "https://api.modelbest.cn/v1", request_namespace=run_id,
            request_journal_path=journal_path, **self.inference.client_options(),
            timeout_s=600 if self.inference.provider == "llamacpp-selfhost" else 90, strict_response_model=True, **options)
        self.on_gate_event, self.verifier = on_gate_event, None


_CODE_FIELDS = frozenset({"error_code", "provider_code", "code", "stopped_reason", "termination_reason"})


def _has_context_limit(value):
    if not isinstance(value, dict):
        return False
    if any(value.get(key) == "context_limit" for key in _CODE_FIELDS):
        return True
    return any(_has_context_limit(value.get(key)) for key in ("error", "result")) or any(
        _has_context_limit(row) for key in ("model_call_metrics", "model_calls", "tool_trace")
        for row in value.get(key) or [])


def minicpm_infrastructure_failure(value):
    def exhausted_retry(node):
        if not isinstance(node, dict):
            return None
        for key in _CODE_FIELDS:
            code = node.get(key)
            if code in {"http_404_empty", "http_429"} and ("attempt" not in node or node.get("attempt", 0) >= 2):
                return "http_404" if code == "http_404_empty" else code
        for key in ("error", "result"):
            failed = exhausted_retry(node.get(key))
            if failed:
                return failed
        for key in ("model_call_metrics", "model_calls", "tool_trace", "gate_events"):
            for row in node.get(key) or []:
                failed = exhausted_retry(row)
                if failed:
                    return failed
        return None
    # Retain independent fatal evidence (for example usage_missing or HTTP 400)
    # even when another trace also contains an identified capacity failure.
    def project(node):
        if isinstance(node, dict):
            codes = [node.get(key) for key in _CODE_FIELDS]
            permitted_status = set()
            if "context_limit" in codes:
                permitted_status.update((400, 413, 422))
            if type(node.get("attempt")) is int and node["attempt"] == 1:
                if "http_404_empty" in codes:
                    permitted_status.add(404)
                if "http_429" in codes:
                    permitted_status.add(429)
            return {key: "context_limit_turn" if key in _CODE_FIELDS and child == "context_limit"
                    else project(child) for key, child in node.items()
                    if not (key in {"http_status", "status_code"} and child in permitted_status)}
        if isinstance(node, list):
            return [project(child) for child in node]
        return node
    return exhausted_retry(value) or infrastructure_failure(project(value))


class _SmokeDataset:
    """Three new questions using original private assets, never a rebuilt album."""
    def __init__(self, source, path):
        original = Dataset(source, strict=True)
        path = Path(path).absolute()
        if not path.is_file() or path.is_symlink():
            raise ValueError("Smoke questions require a regular JSONL file")
        path = path.resolve()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        original_ids = {row["qa_id"] for row in original.rows}
        original_questions = {turn["message"] for row in original.rows for turn in Dataset.turns(row)}
        allowed = {"qa_id", "question", "answer", "answerability", "expected_action", "retrieval_image_ids"}
        if len(rows) != 3 or any(not isinstance(row, dict) or set(row) - allowed for row in rows):
            raise ValueError("Nonformal smoke requires exactly three independent single-turn questions")
        ids = [row.get("qa_id") for row in rows]
        if len(set(ids)) != 3 or any(not isinstance(identifier, str) or
                not identifier.startswith("minicpm-smoke-") or identifier in original_ids for identifier in ids):
            raise ValueError("Smoke question identities must be new")
        filenames = {photo.name.casefold() for photo in original.photos}
        for row in rows:
            safe_id(row["qa_id"])
            if (not isinstance(row.get("question"), str) or not row["question"].strip()
                    or row["question"] in original_questions or not isinstance(row.get("answer"), str)
                    or row.get("expected_action") not in {"answer", "clarify", "refuse"}
                    or row.get("answerability") not in {"answerable", "unanswerable"}
                    or not isinstance(row.get("retrieval_image_ids"), list)
                    or any(not isinstance(name, str) or Path(name).name != name or name.casefold() not in filenames
                           for name in row["retrieval_image_ids"])):
                raise ValueError("Invalid independent smoke question or asset reference")
        if not any(row["answerability"] == "answerable" for row in rows):
            raise ValueError("Real-picture smoke must include an answerable picture question")
        self.rows, self.photos, self.manifest = rows, original.photos, original.manifest
        self.qa_set = "minicpm-real-smoke-3"
        self.source_integrity = original.integrity()
        self.plan_path, self.plan_sha256 = path, sha256(path)

    def integrity(self):
        if sha256(self.plan_path) != self.plan_sha256:
            raise ValueError("Smoke questions changed during preparation")
        return {**self.source_integrity, "nonformal_plan_sha256": self.plan_sha256,
                "nonformal_plan_object_sha256": object_hash(self.rows)}

    turns = staticmethod(Dataset.turns)


class MiniCPMRunner(GateRunner):
    protocol = PROTOCOL
    run_mode = "minicpm_compare"
    stream_model = False

    def __init__(self, *args, backend_factory=MiniCPMBackend, inference=None,
                 pricing_evidence=None, **kwargs):
        self.inference = inference_config(inference)
        self.protocol = self.inference.protocol_version
        self._smoke_questions = None
        if pricing_evidence is not None:
            from functools import partial
            backend_factory = partial(backend_factory, pricing_evidence=copy.deepcopy(pricing_evidence))
        super().__init__(*args, backend_factory=backend_factory, **kwargs)

    def _profile(self, group):
        profile = inference_config(group)
        if profile != self.inference:
            raise ValueError("Run profile differs from frozen runner configuration")
        return profile

    def _dataset(self, strict):
        if strict:
            if self._smoke_questions is not None:
                raise ValueError("Smoke questions cannot enter a formal run")
            return Dataset(self.source, strict=True)
        if self._smoke_questions is None:
            raise ValueError("Nonformal MiniCPM run requires independent smoke questions")
        return _SmokeDataset(self.source, self._smoke_questions)

    def _validate_dataset(self, parent, dataset, integrity):
        original = getattr(dataset, "source_integrity", integrity)
        if parent.get("input_integrity") != original:
            raise ValueError("MiniCPM source differs from the original frozen album")

    def _capture_binding(self, dataset, expected_snapshot, strict):
        # Both smoke and formal use the original complete W38Max-provenance
        # snapshot. The inference model never changes this index provenance.
        return IndexBinding.capture(self._memory, self._parent_path,
            expected_snapshot=expected_snapshot, expected_images=355, strict_source=True)

    def _infrastructure_failure(self, value):
        return minicpm_infrastructure_failure(value)

    def _failure_category(self, response, fatal, origin, checker_failure):
        if not fatal and origin == "system_failure" and _has_context_limit(response):
            return "context_limit"
        return super()._failure_category(response, fatal, origin, checker_failure)

    def _summary(self, state):
        result = gate_summary(state)
        result["summary_version"] = self.protocol
        result["retrieval_metric_scope"] = "fixed_minicpm_planned_image_delivery"
        result["context_limit_turn_count"] = sum(turn.get("failure_category") == "context_limit"
            for item in state.get("items", []) for turn in item.get("runtime_turns", []))
        return result

    def _save(self):
        self._state["base_adapter_sha256"] = self._state.get("base_adapter_sha256", self._state.get("adapter_sha256"))
        self._state["adapter_sha256"] = sha256(__file__)
        self._state["inference_complete"] = self._state.get("gate_complete", False)
        self._state["context_limit_policy"] = "unscored_turn_failure_continue_plan"
        self._state["comparison_scope"] = "pinned-index-minicpm-historical-f37-v1"
        self._state["seen_benchmark"] = True
        return super()._save()

    def run_minicpm(self, *, run_id, suite_id, source_album_id, source_run_id,
                    source_evaluation_root=None, smoke_questions=None, lock_held=False):
        self._smoke_questions = smoke_questions
        return super().run_gate(run_id=run_id, suite_id=suite_id, group=self.inference,
            source_album_id=source_album_id, source_run_id=source_run_id,
            source_evaluation_root=source_evaluation_root, strict=smoke_questions is None,
            expected_snapshot=FIXED_SNAPSHOT, lock_held=lock_held)


def client_class(profile):
    if profile.provider == "llamacpp-selfhost":
        from camroll_agent.minicpm_selfhost import MiniCPMSelfHostedClient
        return MiniCPMSelfHostedClient
    from camroll_agent.modelbest import ModelBestClient
    return ModelBestClient
