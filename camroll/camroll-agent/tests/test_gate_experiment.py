"""Gate A/B runner tests: real immutable Library, synthetic cloud/Judge only."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from test_photobench_eval import source, FakeJudge
from test_supplement import frozen_parent
from camroll_agent.experiment import get_experiment_profile
from camroll_agent.library import Library
from camroll_agent.retrieval import experiment_metadata
from camroll_eval.common import atomic_json, read_json
from camroll_eval.gate_experiment import (FIXED_SNAPSHOT, GROUPS, GateRunner, IndexBinding,
    InferenceConfig, file_hashes, inference_config, infrastructure_failure, gate_summary, validator_failure)


@pytest.fixture
def gate_fixture(frozen_parent):
    fixture = frozen_parent
    parent = fixture["parent"]
    parent.update(experiment="W37", synthetic_nonformal=True)
    profile = get_experiment_profile("W37").with_fingerprints(embedding_fingerprint="synthetic-wemm-fingerprint")
    parent["experiment_config"] = {**profile.as_dict(), "cache_namespace": parent["run_id"]}
    records_path = fixture["memory"] / "snapshots" / fixture["version"] / "records.json"
    records = read_json(records_path)
    records["metadata"].update(experiment_metadata(profile, parent["run_id"]))
    parent["snapshot_metadata"] = records["metadata"]
    atomic_json(records_path, records)
    atomic_json(fixture["parent_path"], parent)
    return fixture


class Backend:
    def __init__(self, *, responses=None, budget=None):
        self.responses = responses or {}
        self.questions, self.options = [], None
        self.status = budget or {"limit_cny": 600, "remaining_cny": 575, "uncertain_requests": 0}

    def factory(self, **options):
        self.options = options
        return self

    def budget(self):
        return self.status.copy()

    def agent(self, memory, config):
        backend = self
        library = Library(memory, config=config, experiment_config=self.options["index_binding"].experiment_config)
        class FakeAgent:
            def ask(self, question, history=None, **kwargs):
                backend.questions.append({"question": question, "history": copy.deepcopy(history), **kwargs})
                result = {"final_text": "synthetic answer", "origin": "model", "gate_status": "off", "gate_attempts": 0,
                    "index_version": library.version, "turn_outcome": "answer", "stopped_reason": "completed",
                    "selected_image_ids": list(library.images)[:1], "retrieved_image_ids": list(library.images),
                    "evidence_image_ids": list(library.images)[:1], "tool_trace": [],
                    "model_call_metrics": [{"call_id": "new-" + str(len(backend.questions)), "call_type": "agent", "duration_ms": 1}],
                    "conversation_history": (history or []) + [{"role": "user", "content": question},
                        {"role": "assistant", "content": "synthetic answer"}]}
                result.update(copy.deepcopy(backend.responses.get(question) or {}))
                return result
        agent = FakeAgent()
        agent.library = library
        return agent


class FrozenJudge(FakeJudge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options, self.calls, self.contexts = kwargs, [], []

    @contextlib.contextmanager
    def request_context(self, value):
        self.contexts.append(value)
        yield

    def _judge_task_action(self, question, answer, expected, status, reason, **kwargs):
        self.calls.append({"kind": "action", "question": question, **copy.deepcopy(kwargs)})
        return {"actual_action": expected, "correct": True}

    def _judge(self, question, reference, answer, **kwargs):
        self.calls.append({"kind": "quality", "question": question, "reference": reference, **copy.deepcopy(kwargs)})
        return {"score": 2, "reason": "synthetic"}


def run_gate(fixture, *, backend=None, group="F37-Off", run_id="gate-test", judge_type=FrozenJudge, **options):
    backend = backend or Backend()
    judges = []
    def factory(**kwargs):
        judge = judge_type(**kwargs)
        judges.append(judge)
        return judge
    engine = GateRunner(fixture["root"], fixture["source"], fixture["albums"],
        backend_factory=backend.factory, judge_factory=factory, sample_resources=False)
    state = engine.run_gate(run_id=run_id, suite_id="synthetic-suite", group=group,
        source_album_id=fixture["parent"]["album_id"], source_run_id=fixture["parent"]["run_id"],
        expected_snapshot=fixture["version"], strict=False, **options)
    return state, backend, judges, engine


def test_configs_are_frozen_and_do_not_allow_max_or_protocol_changes():
    for name in GROUPS:
        profile = inference_config(name)
        assert profile.generation_model in {"qwen3.7-flash", "qwen3.8-flash"}
        assert inference_config(profile.as_dict()) == profile
        with pytest.raises(FrozenInstanceError):
            profile.generation_model = "qwen3.8-max"
        for key, value in (("generation_model", "qwen3.8-max"), ("deadline_s", 300), ("agent_concurrency", True),
                           ("enable_thinking", 0), ("temperature", True)):
            with pytest.raises(ValueError):
                inference_config({**profile.as_dict(), key: value})


@pytest.mark.parametrize("group", list(GROUPS))
def test_four_groups_use_original_index_fresh_context_and_own_judge_ids(gate_fixture, group):
    before, parent = file_hashes(gate_fixture["memory"]), gate_fixture["parent_path"].read_bytes()
    state, backend, judges, engine = run_gate(gate_fixture, group=group)
    assert state["status"] == "completed" and state["gate_complete"] is True
    assert state["gate_summary_version"] == "camroll-flash-gate-v1"
    assert state["qa_count"] == 2 and state["planned_turn_count"] == 3
    assert state["source_unchanged"] is True and state["build_skipped"] is True
    assert state["index_binding"]["experiment_config"]["generation_model"] == "qwen3.7-flash"
    assert state["inference_config"]["group"] == group
    assert len(backend.questions) == 3
    assert backend.questions[0]["history"] == backend.questions[1]["history"] == []
    assert len(backend.questions[2]["history"]) == 2
    assert all("reference_answer" not in question for question in backend.questions)
    assert len(judges[0].calls) == 6 and len(set(judges[0].contexts)) == 3
    assert judges[0].options["expected_limit_cny"] == 600
    assert judges[0].options["request_namespace"] == "gate-test"
    assert Path(state["request_journal_path"]).parent == engine._path
    assert state["summary"]["quality_distribution_planned"] == {"correct": 2, "partial": 0, "incorrect": 0, "unscored": 0}
    assert state["summary"]["judge_valid_turn_count"] == 3
    assert len(list((engine._path / "turn_attempts").rglob("*.json"))) == 3
    assert file_hashes(gate_fixture["memory"]) == before and gate_fixture["parent_path"].read_bytes() == parent
    with pytest.raises((ValueError, FileExistsError)):
        run_gate(gate_fixture, group=group, run_id="different-id")


def test_fresh_destination_reads_separate_frozen_source_and_keeps_one_shot_claims(gate_fixture, tmp_path):
    source_root = gate_fixture["root"]
    source_before = file_hashes(source_root)
    album_before = file_hashes(gate_fixture["memory"])
    destination = tmp_path / "new-gate-evaluation"
    assert not destination.exists()
    fixture = {**gate_fixture, "root": destination}

    state, backend, judges, engine = run_gate(fixture, source_evaluation_root=source_root)

    assert state["status"] == "completed" and state["gate_complete"] is True
    assert len(backend.questions) == 3 and len(judges[0].calls) == 6
    assert engine._parent_path == gate_fixture["parent_path"]
    assert engine._path == destination / "runs" / "gate-test"
    assert engine._path.stat().st_mode & 0o777 == 0o700
    assert read_json(engine._path / "run.json")["source_unchanged"] is True
    assert Path(state["request_journal_path"]).parent == engine._path
    claim = destination / "gate_claims" / "synthetic-suite" / "F37-Off.json"
    assert read_json(claim)["run_id"] == "gate-test"

    output_before = file_hashes(engine._path)
    claim_before = claim.read_bytes()
    repeated_backend = Backend()
    for repeated_id in ("gate-test", "different-id"):
        with pytest.raises((ValueError, FileExistsError)):
            run_gate(fixture, run_id=repeated_id, backend=repeated_backend,
                     source_evaluation_root=source_root)
    assert repeated_backend.questions == []
    assert file_hashes(engine._path) == output_before and claim.read_bytes() == claim_before
    assert not (destination / "runs" / "different-id").exists()
    assert file_hashes(source_root) == source_before
    assert file_hashes(gate_fixture["memory"]) == album_before


@pytest.mark.parametrize("failure", ["no_progress", "final_response_error", "invalid_evidence_reference"])
def test_model_contract_failure_skips_only_affected_turn_and_keeps_multiturn_history(gate_fixture, failure):
    backend = Backend(responses={"which event": {"origin": "system_failure", "turn_outcome": "error", "stopped_reason": failure}})
    state, backend, judges, _ = run_gate(gate_fixture, backend=backend)
    assert state["status"] == "completed_with_errors" and state["gate_complete"] is True
    assert len(backend.questions) == 3 and len(backend.questions[2]["history"]) == 2
    failed = state["items"][1]["runtime_turns"][0]
    assert failed["failure_category"] == "model_contract_failure" and failed["judge_status"] == "skipped_expected"
    assert failed["judge"]["score"] is None
    assert not any(call["question"] == "which event" for call in judges[0].calls)
    last = next(call for call in judges[0].calls if call["question"] == "that day" and call["kind"] == "quality")
    assert len(last["conversation"]) == 2 and last["reference"] == "photo b"
    assert state["summary"]["model_contract_failure_turn_count"] == 1
    assert state["summary"]["judge_expected_skip_turn_count"] == 1


def test_gate_exhaustion_never_becomes_correct_refusal_or_delivery(gate_fixture):
    backend = Backend(responses={"that day": {"origin": "gate_exhausted", "turn_outcome": "error", "gate_status": "exhausted",
        "gate_attempts": 4, "gate_events": [{"state": "exhausted", "gate_id": "final"}]}})
    state, _, judges, engine = run_gate(gate_fixture, backend=backend, group="F38-On")
    last = state["items"][1]
    assert state["gate_complete"] is True and len(judges[0].calls) == 4
    assert last["judge_status"] == "skipped_expected" and last["judge"]["score"] is None
    assert last["predicted_images"] == [] and last["gate_events"][0]["turn_index"] == 1
    assert state["summary"]["quality_distribution_planned"] == {"correct": 1, "partial": 0, "incorrect": 0, "unscored": 1}
    assert state["summary"]["quality_correct_planned_rate"] == .5
    assert state["summary"]["gate_attempt_count"] == 4
    assert state["summary"]["retrieval_metric_count"] == 2
    assert read_json(engine._path / "turn_attempts/two/1.json")["response"]["selected_image_ids"]


@pytest.mark.parametrize("metrics,events,expected", [
    ([{"call_type": "evidence_gate", "error_code": "gate_invalid_verdict"}], [], "gate_invalid_verdict"),
    ([{"role": "evidence_gate", "error_code": "refusal"}], [], "refusal"),
    ([{"role": "evidence_gate", "parse_status": "invalid", "format_attempt": 0},
      {"role": "evidence_gate", "parse_status": "invalid", "format_attempt": 1}],
     [{"state": "blocked", "error_code": "gate_invalid_verdict"}], "gate_invalid_verdict"),
])
def test_checker_failure_is_not_attributed_to_flash_and_continues_plan(gate_fixture, metrics, events, expected):
    backend = Backend(responses={"which event": {"origin": "system_failure", "turn_outcome": "error",
        "stopped_reason": "provider_error", "model_call_metrics": metrics, "gate_events": events}})
    state, backend, judges, _ = run_gate(gate_fixture, backend=backend, group="F37-On")
    failed = state["items"][1]["runtime_turns"][0]
    assert failed["origin"] == "system_failure" and failed["failure_category"] == "validator_failure"
    assert failed["validator_failure_code"] == expected
    assert failed["judge_status"] == "skipped_expected" and failed["judge"]["score"] is None
    assert not any(call["question"] == "which event" for call in judges[0].calls)
    assert state["gate_complete"] is True and len(backend.questions) == 3
    assert state["summary"]["validator_failure_turn_count"] == 1
    assert state["summary"]["model_contract_failure_turn_count"] == 0
    assert state["summary"]["infrastructure_failure_turn_count"] == 0


def test_checker_repair_success_flash_refusal_and_quoted_codes_do_not_count_as_validator_failure():
    assert validator_failure({"model_call_metrics": [{"call_type": "agent", "error_code": "refusal"}]}) is None
    assert validator_failure({"model_call_metrics": [{"role": "evidence_gate", "parse_status": "invalid", "format_attempt": 0},
        {"role": "evidence_gate", "parse_status": "valid", "format_attempt": 1}]}) is None
    assert validator_failure({"model_call_metrics": [{"role": "evidence_gate", "parse_status": "valid",
        "input": {"error_code": "refusal"}, "output": {"error_code": "gate_invalid_verdict"}}]}) is None
    assert validator_failure({"answer": "gate_invalid_verdict", "gate_events": [{"state": "blocked", "error_code": "gate_invalid_verdict"}]}) is None


def test_explicit_provider_configuration_faults_stop_without_breaking_off_schema_fallback():
    for code in ("http_400", "http_404", "http_422", "context_limit", "missing_credentials"):
        assert infrastructure_failure({"model_call_metrics": [{"error_code": code}]}) == code
    assert infrastructure_failure({"tool_trace": [{"result": {"error": {"provider_code": "context_limit"}}}]}) == "context_limit"
    for code in ("invalid_request", "invalid_image", "invalid_schema", "unknown_model", "missing_endpoint"):
        assert infrastructure_failure({"tool_trace": [{"result": {"error": {
            "code": "provider_error", "exception_type": "ProviderError", "error_code": code}}}]}) == code
    assert infrastructure_failure({"gate_events": [{"state": "blocked", "error_code": "context_limit"}]}) == "context_limit"
    assert infrastructure_failure({"gate_events": [{"state": "blocked", "error_code": "unsupported_schema"}]}) == "unsupported_schema"
    assert infrastructure_failure({"model_call_metrics": [{"error_code": "unsupported_schema"}, {"status": "success"}]}) is None
    assert infrastructure_failure({"model_call_metrics": [{"status": "success", "input": {"error_code": "http_400"}}],
        "answer": "context_limit", "gate_events": [{"state": "rejected", "error": {"code": "invalid_schema"}}]}) is None


def test_off_preflight_error_metadata_blocks_before_later_questions(gate_fixture):
    backend = Backend(responses={"find photo": {"origin": "system_failure", "turn_outcome": "error",
        "model_call_metrics": [], "tool_trace": [{"tool": "agent", "result": {"error": {
            "code": "provider_error", "exception_type": "ProviderError", "error_code": "context_limit"}}}]}})
    state, backend, judges, _ = run_gate(gate_fixture, backend=backend)
    assert state["status"] == "blocked" and len(backend.questions) == 1 and not judges[0].calls
    assert state["stop_reason"] == "context_limit"
    assert state["summary"]["infrastructure_failure_turn_count"] == 1
    assert state["summary"]["model_contract_failure_turn_count"] == 0


@pytest.mark.parametrize("code", ["http_401", "http_403", "request_uncertain", "usage_missing", "asset_binding_error",
                                 "response_model_mismatch", "gate_response_model_mismatch", "retrieval_backend_error"])
def test_infrastructure_error_stops_new_questions_and_preserves_entire_plan(gate_fixture, code):
    backend = Backend(responses={"find photo": {"origin": "system_failure", "model_call_metrics": [{"error_code": code}]}})
    state, backend, judges, _ = run_gate(gate_fixture, backend=backend)
    assert state["status"] == "blocked" and state["gate_complete"] is False
    assert len(backend.questions) == 1 and not judges[0].calls
    assert state["stop_reason"] == code
    assert state["items"][0]["failure_category"] == "infrastructure_failure"
    assert len(state["planned_status"]) == 2 and state["planned_status"][1]["execution_status"] == "not_started"
    assert state["summary"]["unexecuted_turn_count"] == 2


def test_answer_text_cannot_trigger_operational_stop():
    assert infrastructure_failure({"answer": "http_403", "code": {}, "model_call_metrics": [{"input": {"error_code": "http_403"}}]}) is None
    assert infrastructure_failure({"tool_trace": [{"result": {"error": {"http_status": 403}}}]}) == "http_403"


@pytest.mark.parametrize("budget", [{"limit_cny": 300, "remaining_cny": 300},
    {"limit_cny": 600, "remaining_cny": 400, "uncertain_requests": 1}, {"limit_cny": 600, "remaining_cny": 0}])
def test_budget_guard_stops_before_any_paid_ask(gate_fixture, budget):
    state, backend, judges, _ = run_gate(gate_fixture, backend=Backend(budget=budget))
    assert state["status"] == "blocked" and not backend.questions and not judges[0].calls
    assert len(state["planned_status"]) == 2


def test_asset_failure_persists_plan_before_backend_or_judge_creation(gate_fixture):
    next((gate_fixture["memory"] / "assets").iterdir()).unlink()
    called = []
    engine = GateRunner(gate_fixture["root"], gate_fixture["source"], gate_fixture["albums"],
        backend_factory=lambda **kw: called.append("backend"), judge_factory=lambda **kw: called.append("judge"), sample_resources=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        engine.run_gate(run_id="missing-assets", suite_id="suite", group="F37-Off", strict=False,
            source_run_id=gate_fixture["parent"]["run_id"], source_album_id=gate_fixture["memory"].name,
            expected_snapshot=gate_fixture["version"])
    state = read_json(engine._path / "run.json")
    assert len(state["planned_status"]) == 2 and state["status"] == "blocked" and not called


def test_formal_capture_cannot_accept_other_version_or_synthetic_bypass(gate_fixture):
    with pytest.raises(ValueError, match="Formal Gate source"):
        IndexBinding.capture(gate_fixture["memory"], gate_fixture["parent_path"], expected_snapshot=gate_fixture["version"])
    parent = read_json(gate_fixture["parent_path"])
    parent.pop("synthetic_nonformal")
    atomic_json(gate_fixture["parent_path"], parent)
    with pytest.raises(ValueError, match="synthetic fixture"):
        IndexBinding.capture(gate_fixture["memory"], gate_fixture["parent_path"], strict_source=False,
            expected_snapshot=gate_fixture["version"], expected_images=2)


def test_missing_or_boolean_judge_score_prevents_completion(gate_fixture):
    class BadJudge(FrozenJudge):
        def _judge(self, *args, **kwargs):
            return {"score": True}
    state, _, _, _ = run_gate(gate_fixture, judge_type=BadJudge)
    assert not state["gate_complete"] and state["summary"]["judge_valid_turn_count"] == 0
    assert state["summary"]["quality_valid_final_count"] == 0


def test_snapshot_mutation_blocks_later_questions_and_final_integrity(gate_fixture):
    class Mutating(Backend):
        def agent(self, *args):
            agent = super().agent(*args)
            original = agent.ask
            def ask(*a, **kw):
                response = original(*a, **kw)
                (gate_fixture["memory"] / "state.sqlite").write_bytes(b"changed")
                return response
            agent.ask = ask
            return agent
    state, backend, _, _ = run_gate(gate_fixture, backend=Mutating())
    assert len(backend.questions) == 1
    assert state["source_unchanged"] is False and not state["gate_complete"]


@pytest.mark.parametrize("group", ["F37-Off", "F38-Off"])
def test_real_backend_library_agent_view_final_uses_original_assets_with_new_inference(gate_fixture, monkeypatch, group):
    from test_supplement import RealAgentVisionBackend
    from camroll_eval.gate_experiment import GateBackend
    synthetic = RealAgentVisionBackend(gate_fixture["images"][0]["id"])
    synthetic.qwen.model = inference_config(group).generation_model
    synthetic.qwen.usage_summary = lambda: {"limit_cny": 600, "remaining_cny": 575}
    constructions = []
    def cloud(**options):
        constructions.append(options)
        return synthetic.qwen
    monkeypatch.setattr("camroll_agent.qwen.QwenClient", cloud)
    monkeypatch.setattr("camroll_agent.retrieval.make_retrieval_clients", lambda config: (synthetic.embedding, None))
    engine = GateRunner(gate_fixture["root"], gate_fixture["source"], gate_fixture["albums"],
        backend_factory=GateBackend, judge_factory=FrozenJudge, sample_resources=False)
    before = file_hashes(gate_fixture["memory"])
    state = engine.run_gate(run_id="real-chain", suite_id="real-chain", group=group,
        source_album_id=gate_fixture["memory"].name, source_run_id=gate_fixture["parent"]["run_id"],
        expected_snapshot=gate_fixture["version"], strict=False)
    assert state["gate_complete"] is True
    assert constructions[0]["model"] == inference_config(group).generation_model
    assert constructions[0]["strict_response_model"] is True
    assert constructions[0]["request_namespace"] == "real-chain"
    assert [call["role"] for call in synthetic.vision_calls] == ["view", "final"] * 3
    assert [call["role"] for call in synthetic.query_calls] == ["query"] * 3
    assert all(Path(path).is_relative_to(gate_fixture["memory"] / "assets")
        for call in synthetic.vision_calls for path in call["paths"])
    assert all(turn["origin"] == "model" for row in state["items"] for turn in row["runtime_turns"])
    assert file_hashes(gate_fixture["memory"]) == before


@pytest.mark.parametrize("outcome", ["refuse", "clarify"])
def test_normal_model_refusal_and_clarification_remain_judge_eligible(gate_fixture, outcome):
    backend = Backend(responses={"find photo": {"turn_outcome": outcome, "origin": "model"}})
    state, _, judges, _ = run_gate(gate_fixture, backend=backend)
    assert state["items"][0]["judge_eligible"] is True
    assert state["items"][0]["judge"]["score"] == 2
    assert len(judges[0].calls) == 6


def test_no_last_turn_score_is_borrowed_when_second_turn_hits_infrastructure(gate_fixture):
    backend = Backend(responses={"that day": {"origin": "system_failure", "error_code": "http_403"}})
    state, _, _, _ = run_gate(gate_fixture, backend=backend)
    assert state["items"][1]["judge_eligible"] is False
    assert state["items"][1]["judge"]["score"] is None
    assert state["gate_complete"] is False


@pytest.mark.parametrize("group", ["F37-On", "F38-On"])
def test_gate_on_real_streaming_client_and_backend_keep_all_attempts_and_asset_binding(gate_fixture, monkeypatch, group):
    import json
    from types import SimpleNamespace
    from test_supplement import RealAgentVisionBackend
    from test_output_gate import call, response
    from camroll_agent.budget import BudgetLedger
    from camroll_agent.qwen import QwenClient
    from camroll_eval.gate_experiment import GateBackend
    model = inference_config(group).generation_model
    image_id = gate_fixture["images"][0]["id"]
    synthetic = RealAgentVisionBackend(image_id)
    outcomes = []
    for _ in range(3):
        outcomes.extend([
            response(calls=[call("search", {"query": "red"})]),
            response(calls=[call("view_image", {"image_ids": [image_id], "prompt": "describe"})]),
            response(json.dumps({"analysis": "red", "evidence": [{"image_id": image_id, "observation": "red"}], "limitations": ""})),
            response(json.dumps({"answer": "red", "outcome": "answer", "selected_image_ids": [image_id]})),
            response(json.dumps({"answer": "red", "outcome": "answer", "evidence_image_ids": [image_id]}))])
    received = []
    def create(**kwargs):
        received.append(copy.deepcopy(kwargs))
        raw = copy.deepcopy(outcomes[len(received) - 1])
        raw["model"] = model
        if not kwargs.get("stream"):
            return raw
        message = raw["choices"][0]["message"]
        for i, tool in enumerate(message.get("tool_calls", [])):
            tool["index"] = i
        return iter([{"model": model, "choices": [{"delta": message, "finish_reason": "stop"}], "usage": raw["usage"]}])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ledger = BudgetLedger(gate_fixture["root"].parent / "gate-budget.sqlite", 600)
    def cloud(**options):
        return QwenClient(sdk=sdk, ledger=ledger, **options)
    verifier_calls = []
    class Verifier:
        def __init__(self, *, client, on_call):
            assert client.model == model
        def verify(self, **kwargs):
            verifier_calls.append(kwargs)
            return {"status": "supported", "reasons": [], "evidence_ids": [image_id]}
    monkeypatch.setattr("camroll_agent.qwen.QwenClient", cloud)
    monkeypatch.setattr("camroll_agent.retrieval.make_retrieval_clients", lambda config: (synthetic.embedding, None))
    monkeypatch.setattr("camroll_agent.evidence_gate.EvidenceGateVerifier", Verifier)
    engine = GateRunner(gate_fixture["root"], gate_fixture["source"], gate_fixture["albums"],
        backend_factory=GateBackend, judge_factory=FrozenJudge, sample_resources=False)
    state = engine.run_gate(run_id="on-chain", suite_id="on-chain", group=group,
        source_album_id=gate_fixture["memory"].name, source_run_id=gate_fixture["parent"]["run_id"],
        expected_snapshot=gate_fixture["version"], strict=False)
    assert state["gate_complete"] is True and len(received) == 15
    assert len(verifier_calls) == 3
    assert all(call["model"] == model and call["temperature"] == 0 and call["extra_body"]["enable_thinking"] is False for call in received)
    assert all("reference_answer" not in call and "gt_images" not in call for call in verifier_calls)
    assert all(turn["gate_status"] == "accepted" for item in state["items"] for turn in item["runtime_turns"])
    assert state["summary"]["gate"]["first_pass_denominator"] == 15
    events = [json.loads(line) for line in (engine._path / "gate_events.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (engine._path / "model_calls.jsonl").read_text().splitlines()]
    assert len(calls) == len({entry["call_id"] for entry in calls}) == 15
    assert all(entry["evaluation_context"]["qa_id"] in {"one", "two"} for entry in calls)
    assert len([entry for entry in events if entry["state"] == "sending"]) == 15
    assert (engine._path / "request-journal.sqlite").is_file()
    assert state["source_unchanged"] is True
