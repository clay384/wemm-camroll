"""Independent MiniCPM protocol, original assets, and per-turn capacity failures."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from test_photobench_eval import source
from test_supplement import frozen_parent
from test_gate_experiment import gate_fixture, Backend, FrozenJudge
from camroll_eval.common import read_json
from camroll_eval.dataset import Dataset
from camroll_eval.gate_experiment import GateRunner, IndexBinding, file_hashes, inference_config as flash_profile
from camroll_eval import minicpm_experiment as mini


def test_profile_is_explicit_frozen_and_flash_profiles_unchanged():
    profile = mini.MiniCPMInferenceConfig(decision_mode="json_decision", json_mode="prompt")
    assert mini.inference_config(profile.as_dict()) == profile
    assert mini.inference_config(profile).fingerprint == profile.fingerprint
    assert profile.enable_thinking is True and profile.stream is False
    assert profile.provider == profile.as_dict()["provider"] == "modelbest"
    assert profile.max_output_tokens == 8192 and profile.context_limit == 256000
    assert profile.gate_policy == "off" and profile.mode == "M4"
    with pytest.raises(FrozenInstanceError):
        profile.enable_thinking = False
    for key, value in (("enable_thinking", False), ("enable_thinking", 1), ("temperature", True),
                       ("deadline_s", 300), ("context_limit", 128000), ("max_output_tokens", 4096),
                       ("generation_model", "qwen3.7-flash"), ("provider", "qwen"), ("gate_policy", "repair-evidence-v1"),
                       ("decision_mode", "fallback"), ("json_mode", "relaxed")):
        with pytest.raises(ValueError):
            mini.inference_config({**profile.as_dict(), key: value})
    assert flash_profile("F37-Off").enable_thinking is False
    with pytest.raises((ValueError, TypeError)):
        flash_profile(profile.as_dict())


def test_original_minicpm_profile_keeps_exact_frozen_serialization_and_fingerprint():
    profile = mini.MiniCPMInferenceConfig()
    assert profile.request_profile == "fixed-v1" and profile.thinking_mode == "on"
    assert "request_profile" not in profile.as_dict() and "thinking_mode" not in profile.as_dict()
    assert profile.fingerprint == "f7a9042159787f364baa785c7774b7fe70f0fd1b225ce094e4a0eedc109514bd"
    assert mini.MiniCPMInferenceConfig(**profile.as_dict()) == profile


def test_sdk_defaults_profile_is_versioned_and_unspecified_not_thinking_off():
    profile = mini.MiniCPMInferenceConfig.sdk_defaults(decision_mode="json_decision", json_mode="prompt")
    assert profile.group == "MiniCPM47-SDK-Defaults-Off"
    assert profile.protocol_version == "minicpm47-sdk-defaults-m4-off-v1"
    assert mini.SUPPORTED_PROTOCOLS == {mini.PROTOCOL, profile.protocol_version, mini.SDK_INTERFACE_PROTOCOL, mini.SELFHOST_PROTOCOL}
    assert profile.enable_thinking is profile.temperature is profile.stream is profile.max_output_tokens is None
    assert profile.as_dict()["thinking_mode"] == "unspecified"
    assert profile.as_dict()["request_profile"] == "sdk-defaults-v1"
    assert mini.inference_config(profile.as_dict()) == mini.inference_config(profile) == profile
    assert profile.fingerprint != mini.MiniCPMInferenceConfig().fingerprint
    assert profile.client_options() == {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified",
                                         "decision_mode": "json_decision", "json_mode": "prompt"}


def test_interface_acceptance_is_new_explicit_protocol_preserving_old_fingerprints():
    old = mini.MiniCPMInferenceConfig.sdk_defaults()
    assert "acceptance_policy" not in old.as_dict()
    assert old.fingerprint == "643892b8a78492a5e3e4cd338f05bdd7d3f96f09c87a4bd71c06e0fdf1ef48c7"
    new = mini.MiniCPMInferenceConfig.sdk_defaults(acceptance_policy="interface-only-v2")
    assert new.protocol_version == mini.SDK_INTERFACE_PROTOCOL
    assert new.as_dict()["acceptance_policy"] == "interface-only-v2"
    assert mini.inference_config(new.as_dict()) == new
    assert new.client_options() == {**old.client_options(), "compatibility_mode": "modelbest-v2"}
    assert new.as_dict()["adapter_version"] == "modelbest-json-v2"
    assert new.fingerprint != old.fingerprint
    with pytest.raises(ValueError):
        mini.inference_config({**old.as_dict(), "acceptance_policy": "interface-only-v2"})
    with pytest.raises(ValueError):
        mini.MiniCPMInferenceConfig(acceptance_policy="interface-only-v2")


@pytest.mark.parametrize("key,value", [("group", mini.GROUP), ("protocol_version", mini.PROTOCOL),
    ("enable_thinking", False), ("enable_thinking", True), ("temperature", 0), ("stream", False),
    ("max_output_tokens", 8192), ("thinking_mode", "on"), ("thinking_mode", "off"),
    ("request_profile", "fixed-v1"), ("context_limit", 1000000)])
def test_sdk_defaults_rejects_mixed_or_overridden_protocol(key, value):
    config = mini.MiniCPMInferenceConfig.sdk_defaults().as_dict()
    with pytest.raises(ValueError):
        mini.inference_config({**config, key: value})


def test_wire_contract_validation_requires_parameter_absence():
    fixed, defaults = mini.MiniCPMInferenceConfig(), mini.MiniCPMInferenceConfig.sdk_defaults()
    basic = {"model": mini.MODEL, "messages": [{"role": "user", "content": "hello"}]}
    thinking = {**basic, "max_tokens": 8192, "temperature": 0, "stream": False,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    assert fixed.matches_request(thinking) and not fixed.matches_request(basic)
    assert defaults.matches_request(basic) and not defaults.matches_request(thinking)
    flattened = {key: value for key, value in thinking.items() if key != "extra_body"}
    assert fixed.matches_request({**flattened, "chat_template_kwargs": {"enable_thinking": True}})
    for key in ("temperature", "stream", "max_tokens", "max_completion_tokens", "extra_body", "chat_template_kwargs"):
        assert not defaults.matches_request({**basic, key: None})
    assert not defaults.matches_request({**basic, "model": "qwen3.7-flash"})
    assert defaults.matches_request({**basic, "response_format": {"type": "json_object"}})


@pytest.mark.parametrize("code", ["usage_missing", "request_uncertain", "response_model_mismatch",
    "budget_limit_changed", "index_binding_changed", "retrieval_backend_error", "http_404"])
def test_only_local_context_limit_is_removed_from_fatal_policy(code):
    assert mini.minicpm_infrastructure_failure({"error_code": "context_limit"}) is None
    assert mini.minicpm_infrastructure_failure({"error_code": "invalid_json"}) is None
    value = {"error_code": "context_limit", "model_call_metrics": [{"error_code": code}]}
    assert mini.minicpm_infrastructure_failure(value) == code
    assert value["error_code"] == "context_limit"
    assert mini.minicpm_infrastructure_failure({"error_code": "context_limit", "http_status": 400}) is None
    assert mini.minicpm_infrastructure_failure({"error_code": "context_limit", "http_status": 404}) == "http_404"
    assert mini.minicpm_infrastructure_failure({"answer": "usage_missing", "input": {"error_code": code}}) is None


@pytest.mark.parametrize("code,fatal", [("http_404_empty", "http_404"), ("http_429", "http_429")])
def test_first_empty_404_or_429_can_retry_but_exhausted_transport_stops_group(code, fatal):
    first = {"error_code": code, "attempt": 1, "http_status": int(fatal.removeprefix("http_"))}
    assert mini.minicpm_infrastructure_failure(first) is None
    assert mini.minicpm_infrastructure_failure({**first, "attempt": 2}) == fatal
    assert mini.minicpm_infrastructure_failure({"tool_trace": [{"result": {"error": {
        "error_code": code}}}]}) == fatal
    assert mini.minicpm_infrastructure_failure({"input": {"error_code": code}, "answer": code}) is None


class FixtureRunner(mini.MiniCPMRunner):
    """Only test fixture loading differs; production cannot select this path."""
    def _dataset(self, strict):
        return Dataset(self.source, strict=False)

    def _capture_binding(self, dataset, expected_snapshot, strict):
        return IndexBinding.capture(self._memory, self._parent_path,
            expected_snapshot=expected_snapshot, expected_images=len(dataset.photos), strict_source=False)


def run_fixture(fixture, responses=None, *, budget=None, inference=None):
    backend, judges = Backend(responses=responses, budget=budget), []
    def judge_factory(**kwargs):
        judge = FrozenJudge(**kwargs)
        judges.append(judge)
        return judge
    runner = FixtureRunner(fixture["root"], fixture["source"], fixture["albums"],
        backend_factory=backend.factory, judge_factory=judge_factory, sample_resources=False, inference=inference)
    state = runner.run_gate(run_id="minicpm-fixture", suite_id="minicpm-fixture-suite", group=runner.inference,
        source_album_id=fixture["parent"]["album_id"], source_run_id=fixture["parent"]["run_id"],
        expected_snapshot=fixture["version"], strict=False)
    return runner, state, backend, judges


def test_sdk_defaults_runner_freezes_its_own_protocol_without_rebuilding(gate_fixture):
    profile = mini.MiniCPMInferenceConfig.sdk_defaults()
    before, parent = file_hashes(gate_fixture["memory"]), gate_fixture["parent_path"].read_bytes()
    runner, state, backend, judges = run_fixture(gate_fixture, inference=profile)
    assert state["protocol_label"] == state["gate_summary_version"] == state["summary"]["summary_version"] == profile.protocol_version
    assert state["experiment"] == profile.group and state["inference_config"] == profile.as_dict()
    assert state["inference_fingerprint"] == profile.fingerprint
    assert state["inference_complete"] and state["build_skipped"] and state["source_unchanged"]
    assert len(backend.questions) == 3 and all(row["stream_model"] is False for row in backend.questions)
    assert file_hashes(gate_fixture["memory"]) == before and gate_fixture["parent_path"].read_bytes() == parent
    with pytest.raises(ValueError):
        runner._profile(mini.MiniCPMInferenceConfig())


@pytest.mark.parametrize("code,category", [("context_limit", "context_limit"), ("invalid_json", "model_contract_failure")])
def test_capacity_and_json_failure_skip_judge_but_keep_every_later_turn(gate_fixture, code, category):
    before, parent = file_hashes(gate_fixture["memory"]), gate_fixture["parent_path"].read_bytes()
    runner, state, backend, judges = run_fixture(gate_fixture, {"which event": {
        "origin": "system_failure", "turn_outcome": "error", "stopped_reason": code, "error_code": code}})
    assert state["status"] == "completed_with_errors" and state["inference_complete"] is True
    assert state["mode"] == "minicpm_compare" and state["protocol_label"] == mini.PROTOCOL
    assert state["gate_summary_version"] == state["summary"]["summary_version"] == mini.PROTOCOL
    assert state["source_unchanged"] is True and "stop_reason" not in state
    assert len(backend.questions) == 3 and all(row["stream_model"] is False for row in backend.questions)
    assert len(backend.questions[2]["history"]) == 2
    failed = state["items"][1]["runtime_turns"][0]
    assert failed["failure_category"] == category and failed["judge_status"] == "skipped_expected"
    assert failed["judge"]["score"] is None and failed["predicted_images"] == []
    assert len(judges[0].calls) == 4
    assert not any(call["question"] == "which event" for call in judges[0].calls)
    assert state["summary"]["context_limit_turn_count"] == int(code == "context_limit")
    assert state["summary"]["judge_expected_skip_turn_count"] == 1
    assert state["summary"]["retrieval_metric_count"] == 2
    assert read_json(runner._path / "turn_attempts/two/0.json")["response"]["error_code"] == code
    assert file_hashes(gate_fixture["memory"]) == before and gate_fixture["parent_path"].read_bytes() == parent


def test_new_unknown_stops_all_remaining_submissions(gate_fixture):
    _, state, backend, judges = run_fixture(gate_fixture, {"find photo": {
        "origin": "system_failure", "turn_outcome": "error", "stopped_reason": "request_uncertain",
        "error_code": "request_uncertain"}})
    assert state["status"] == "blocked" and state["inference_complete"] is False
    assert state["stop_reason"] == "request_uncertain" and len(backend.questions) == 1
    assert judges[0].calls == []


def test_smoke_and_formal_binding_both_require_original_complete_album(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mini.IndexBinding, "capture", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner = mini.MiniCPMRunner(tmp_path / "out", tmp_path / "source", sample_resources=False)
    runner._memory, runner._parent_path = tmp_path / "original-album", tmp_path / "original-run.json"
    for strict in (False, True):
        runner._capture_binding(object(), mini.FIXED_SNAPSHOT, strict)
    assert all(kwargs == {"expected_snapshot": mini.FIXED_SNAPSHOT, "expected_images": 355,
        "strict_source": True} for _, kwargs in calls)


def test_smoke_plan_uses_new_ids_and_original_assets_without_exposing_gt(source, tmp_path, monkeypatch):
    real_dataset = Dataset(source, strict=False)
    class SyntheticSourceDataset(Dataset):
        def __init__(self, root, strict):
            super().__init__(root, strict=False)
    monkeypatch.setattr(mini, "Dataset", SyntheticSourceDataset)
    rows = [{"qa_id": "minicpm-smoke-" + str(index), "question": "Describe this independent picture " + str(index),
             "answer": "Independent visual observation", "expected_action": "answer", "answerability": "answerable",
             "retrieval_image_ids": [real_dataset.photos[0].name]} for index in range(3)]
    path = tmp_path / "smoke.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = mini._SmokeDataset(source, path)
    assert dataset.photos == real_dataset.photos and dataset.rows == rows
    assert dataset.source_integrity == real_dataset.integrity()
    assert dataset.integrity()["nonformal_plan_sha256"]
    runner = mini.MiniCPMRunner(tmp_path / "out", source, sample_resources=False)
    runner._validate_dataset({"input_integrity": real_dataset.integrity()}, dataset, dataset.integrity())
    path.write_text("\n".join(json.dumps({**row, "question": real_dataset.rows[0]["question"]}) for row in rows))
    with pytest.raises(ValueError):
        mini._SmokeDataset(source, path)
    with pytest.raises(ValueError):
        dataset.integrity()


@pytest.mark.parametrize("factory", [mini.MiniCPMInferenceConfig, mini.MiniCPMInferenceConfig.sdk_defaults])
def test_backend_passes_frozen_capabilities_pricing_and_original_binding(tmp_path, monkeypatch, factory):
    import camroll_agent.retrieval as retrieval
    monkeypatch.setattr(retrieval, "make_retrieval_clients", lambda value: ("original-embedding", None))
    class Binding:
        experiment_config = {"generation_model": "qwen3.8-max", "embedding_backend": "wemm"}
    observed = {}
    def client(**options):
        observed.update(options)
        return object()
    pricing = {"model": mini.MODEL, "input_rate": 0, "output_rate": 0,
               "observed_at": "2026-10-14", "source": "official-test-evidence"}
    profile = factory(decision_mode="json_decision", json_mode="prompt")
    backend = mini.MiniCPMBackend(index_binding=Binding(), inference=profile, run_id="new-run",
        journal_path=tmp_path / "journal.sqlite", ledger="same-ledger", pricing_evidence=pricing,
        api_key="synthetic-test-key", client_factory=client)
    assert backend.embedding == "original-embedding" and backend.verifier is None
    assert observed["decision_mode"] == "json_decision" and observed["json_mode"] == "prompt"
    assert observed["request_profile"] == profile.request_profile and observed["thinking_mode"] == profile.thinking_mode
    assert observed["pricing_evidence"] == pricing and observed["ledger"] == "same-ledger"
    assert observed["request_namespace"] == "new-run" and observed["strict_response_model"] is True


def test_new_entrypoint_only_selects_strict_formal_or_named_smoke(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(GateRunner, "run_gate", lambda self, **options: calls.append(options) or options)
    runner = mini.MiniCPMRunner(tmp_path / "out", tmp_path / "source", sample_resources=False)
    arguments = dict(run_id="new", suite_id="suite", source_album_id="album", source_run_id="original")
    runner.run_minicpm(**arguments)
    runner.run_minicpm(**arguments, smoke_questions=tmp_path / "smoke.jsonl")
    assert [row["strict"] for row in calls] == [True, False]
    assert all(row["expected_snapshot"] == mini.FIXED_SNAPSHOT for row in calls)
    assert all(row["group"] == runner.inference for row in calls)
