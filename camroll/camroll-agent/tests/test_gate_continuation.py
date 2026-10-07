"""Independent continuation never regenerates or rejudges a recorded turn."""
import copy
import sqlite3
from types import SimpleNamespace

import pytest

from test_photobench_eval import source
from test_supplement import frozen_parent
from test_gate_experiment import gate_fixture, Backend, FrozenJudge, run_gate
from camroll_eval.common import atomic_json, read_json, sha256
from camroll_eval.gate_continuation import validate_history
from camroll_eval.gate_experiment import GateRunner, file_hashes


@pytest.fixture
def stopped_parent(gate_fixture):
    state, _, _, engine = run_gate(gate_fixture, group="F37-On", run_id="stopped-source")
    item = state["items"][-1]
    item["runtime_turns"] = item["runtime_turns"][:1]
    turn = item["runtime_turns"][0]
    turn.update(origin="system_failure", answer_origin="system_failure", judge_eligible=False,
        failure_category="infrastructure_failure", error_code="request_uncertain", judge_status="skipped_expected",
        judge={"score": None, "reason": "infrastructure_failure"},
        task_judge={"correct": None, "actual_action": None, "reason": "infrastructure_failure"})
    raw = read_json(engine._path / turn["raw_attempt_path"])
    item.update(conversation_history=raw["response"]["conversation_history"], final_turn_executed=False,
                conversation_turn_count=1, judge_eligible=False, execution_status="incomplete")
    state.update(status="blocked", stop_reason="request_uncertain", gate_complete=False)
    atomic_json(engine._path / "run.json", state)
    with sqlite3.connect(engine._path / "request-journal.sqlite") as c:
        c.execute("CREATE TABLE cloud_requests (call_id TEXT, status TEXT)")
        c.execute("INSERT INTO cloud_requests VALUES ('acknowledged-call','uncertain')")
    return gate_fixture, state, engine._path


def continue_run(fixture, parent, tmp_path, *, mutate=None):
    spec = {"path": str(parent / "run.json"), "sha256": sha256(parent / "run.json")}
    if mutate:
        mutate(spec)
    backend = Backend(budget={"limit_cny": 600, "remaining_cny": 500, "uncertain_requests": 1,
                              "blocking_uncertain_requests": 0})
    ledger = SimpleNamespace(acknowledged_uncertain=(SimpleNamespace(call_id="acknowledged-call"),))
    judges = []
    def judge_factory(**options):
        value = FrozenJudge(**options); judges.append(value); return value
    runner = GateRunner(tmp_path / "continued-evaluation", fixture["source"], fixture["albums"],
        backend_factory=backend.factory, judge_factory=judge_factory, sample_resources=False, budget_ledger=ledger)
    result = runner.run_gate(run_id="continued-run", suite_id="new-suite", group="F37-On",
        source_album_id=fixture["parent"]["album_id"], source_run_id=fixture["parent"]["run_id"],
        source_evaluation_root=fixture["root"], expected_snapshot=fixture["version"], strict=False,
        continuation_parent=spec, acknowledged_call_ids=["acknowledged-call"])
    return result, backend, judges, runner


def test_only_unexecuted_turn_is_generated_and_scored(stopped_parent, tmp_path):
    fixture, original, parent = stopped_parent
    before = file_hashes(parent)
    result, backend, judges, runner = continue_run(fixture, parent, tmp_path)
    assert result["gate_complete"] is True
    assert result["status"] == "completed_with_errors"
    assert len(backend.questions) == 1
    assert backend.questions[0]["history"] == original["items"][-1]["conversation_history"]
    assert backend.questions[0]["request_context"]["turn_index"] == 1
    assert len(judges[0].calls) == 2
    assert all(":turn:1:judge" in namespace for namespace in judges[0].contexts)
    assert result["items"][0]["judge"] == original["items"][0]["judge"]
    assert result["items"][-1]["runtime_turns"][0]["error_code"] == "request_uncertain"
    assert result["items"][-1]["runtime_turns"][0]["judge_status"] == "skipped_expected"
    assert len(result["items"][-1]["runtime_turns"]) == 2
    assert result["continuation"]["inherited_turn_count"] == 2
    assert result["continuation"]["new_turn_count"] == 1
    assert result["summary"]["executed_turn_count"] == 3
    assert len(list((runner._path / "turn_attempts").rglob("*.json"))) == 1
    assert len(list((runner._path / "inherited_turn_attempts").rglob("*.json"))) == 2
    assert file_hashes(parent) == before
    assert "reference_answer" not in backend.questions[0]


def test_parent_hash_mismatch_stops_before_new_calls(stopped_parent, tmp_path):
    fixture, _, parent = stopped_parent
    with pytest.raises(ValueError, match="hash mismatch"):
        continue_run(fixture, parent, tmp_path, mutate=lambda spec: spec.update(sha256="0" * 64))


@pytest.mark.parametrize("field,value", [("inference_fingerprint", "changed"), ("index_binding_fingerprint", "changed"),
    ("input_integrity", {}), ("stop_reason", "something_else")])
def test_changed_source_contract_is_rejected(stopped_parent, tmp_path, field, value):
    fixture, state, parent = stopped_parent
    changed = copy.deepcopy(state); changed[field] = value
    atomic_json(parent / "run.json", changed)
    with pytest.raises(ValueError):
        continue_run(fixture, parent, tmp_path)


def test_unfinished_persisted_tool_cannot_be_reexecuted(stopped_parent, tmp_path):
    fixture, state, parent = stopped_parent
    state["items"][-1]["conversation_history"] = [{"role": "assistant", "tool_calls": [{"id": "pending"}]}]
    atomic_json(parent / "run.json", state)
    with pytest.raises(ValueError, match="unfinished"):
        continue_run(fixture, parent, tmp_path)


def test_new_unknown_still_stops_runner(stopped_parent, tmp_path):
    fixture, state, parent = stopped_parent
    result, backend, judges, runner = continue_run(fixture, parent, tmp_path)
    backend.status["blocking_uncertain_requests"] = 1
    assert runner._cancelled() is True
    assert runner._stop_reason == "request_uncertain"


def test_valid_completed_tool_history_has_no_replay_step():
    validate_history([{"role": "user", "content": "synthetic"},
        {"role": "assistant", "tool_calls": [{"id": "call1"}]},
        {"role": "tool", "tool_call_id": "call1", "content": "done"},
        {"role": "assistant", "content": "failed turn preserved"}])


@pytest.mark.parametrize("change", ["missing", "modified", "symlink"])
def test_inherited_attempt_copy_must_remain_intact(stopped_parent, tmp_path, change):
    fixture, _, parent = stopped_parent
    _, _, _, runner = continue_run(fixture, parent, tmp_path)
    target = next((runner._path / "inherited_turn_attempts").rglob("*.json"))
    content = target.read_bytes()
    target.unlink()
    if change == "modified":
        target.write_bytes(content + b" ")
    elif change == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(content)
        target.symlink_to(outside)
    with pytest.raises(ValueError, match="copy changed"):
        runner._continuation.validate()
