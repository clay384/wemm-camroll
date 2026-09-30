import copy
import json
import sqlite3
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from four_export import ORDER, export


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run_state(group):
    items = []
    for index in range(100):
        score = 2 if index < 75 else 1 if index < 89 else 0
        turn = {"judge": {"score": score}, "task_judge": {"actual_action": "answer", "correct": True}}
        item = {"qa_id": f"q{index}", "question": f"问题{index}", "answer": "回答", "judge": {"score": score},
                "runtime_turns": [copy.deepcopy(turn) for _ in range(2 if index < 11 else 1)],
                "judge_status": "completed", "execution_status": "complete",
                "answerability": "answerable" if index < 81 else "unanswerable",
                "timing_breakdown": {"agent_wall_ms": 1000 + index}}
        if index < 81:
            item["image_retrieval_counts"] = {"gt": 2, "predicted": 2, "matched": 1}
        items.append(item)
    items[0]["model_call_metrics"] = [{"call_id": "agent1", "call_type": "agent", "model": "qwen-actual-revision",
                                       "prompt_tokens": 100, "completion_tokens": 10, "duration_ms": 123}]
    return {"run_id": group, "status": "completed", "qa_count": 100, "formal_protocol": True,
            "planned_turn_count": 111, "items": items, "album_id": "eval-" + group,
            "qa_plan": [{"qa_id": item["qa_id"], "question": item["question"], "planned_turn_count": len(item["runtime_turns"])} for item in items],
            "media_mapping": {str(index): {} for index in range(355)},
            "build_summary": {"image_count": 355, "image_embedding_count": 355, "caption_count": 354,
                              "text_embedding_count": 354, "event_count": 12, "event_embedding_count": 12},
            "phases": {"pipeline": {"wall_ms": 2000}},
            "budget_before": {"estimated_spent_cny": 20, "reserved_cny": 0, "limit_cny": 600},
            "budget_after": {"estimated_spent_cny": 30, "reserved_cny": 0, "uncertain_requests": 0, "limit_cny": 600},
            "resource_summary": {"samples": 10, "gpu_memory_mib_peak": 6200}}


@pytest.fixture
def runtime(tmp_path):
    suite = {"status": "passed", "suite_id": "formal-test", "groups": [],
             "restoration": {"state": "restored", "no_experiment_gpu_residual": True}}
    for group in ORDER:
        suite["groups"].append({"name": group, "run_id": group, "status": "completed"})
        write_json(tmp_path / "evaluation/runs" / group / "run.json", run_state(group))
    write_json(tmp_path / "formal-suite.json", suite)
    return tmp_path


def test_complete_report_fixed_denominators_cost_models_and_all_item_links(runtime):
    result = export(runtime)
    assert result["status"] == "complete"
    assert len(result["groups"]) == 4
    group = result["groups"][0]
    assert group["quality_distribution"] == {"correct": 75, "partial": 14, "incorrect": 11, "unscored": 0}
    assert group["accuracy_fixed_planned"] == .75
    assert group["quality_mean_scored"] == 1.64
    assert group["action_accuracy_scored"] == 1
    assert group["final_delivery_macro"]["metric_count"] == 81
    assert group["final_delivery_macro"]["f1"] == .5
    assert group["cost"]["estimated_spent_delta_cny"] == 10
    assert group["cost"]["actual_subscription_bill_cny"] is None
    assert group["calls_and_tokens"]["observed_generation_models"] == ["qwen-actual-revision"]
    assert group["calls_and_tokens"]["token_totals"]["total_tokens"] == 110
    assert group["sampled_resources"]["samples"] == 10
    assert len(group["questions"]) == 100
    text = (runtime / "reports/four-models.md").read_text()
    assert text.count("/items/") == 400
    assert "历史 M1" in text and "套餐实际账单未取得" in text
    assert (runtime / "reports/four-models.json").stat().st_mode & 0o077 == 0


def test_missing_question_preserves_planned_link_and_does_not_shrink_denominator(runtime):
    path = runtime / "evaluation/runs/Q37/run.json"
    state = json.loads(path.read_text())
    state["items"].pop(0)
    write_json(path, state)
    result = export(runtime)
    assert result["status"] == "incomplete"
    group = result["groups"][0]
    assert group["quality_distribution"]["unscored"] == 1
    assert group["accuracy_fixed_planned"] == .74
    assert group["final_delivery_macro"]["f1"] == pytest.approx(40 / 81)
    assert group["questions"][0]["execution_status"] == "not_started"
    assert len(group["questions"]) == 100


def test_resumed_report_preserves_noncomparable_segment_scope_without_inventing_totals(runtime):
    path = runtime / "evaluation/runs/Q37/run.json"
    state = json.loads(path.read_text())
    state.update(resumed_at=["2026-09-30T12:00:00Z"], timing_comparable=False)
    write_json(path, state)
    group = export(runtime)["groups"][0]
    assert group["resumed_at"] == state["resumed_at"]
    assert group["timing_comparable"] is False
    assert group["timings"]["build_wall_ms"] == 2000
    assert group["timings"]["scope"]["build_wall_ms"] == "last_recorded_pipeline_invocation_segment_only"
    assert group["sampled_resources"] == state["resource_summary"]
    assert group["sampled_resources_scope"] == "latest_evaluator_process_segment_only"
    text = (runtime / "reports/four-models.md").read_text()
    assert "Q37 已发生恢复" in text
    assert "未重建整组总耗时或全程峰值" in text
    assert "单题时延为该题全部已记录轮次的 Agent 墙钟，不含 Judge" in text


@pytest.mark.parametrize("restoration", [{"state": "restore_failed", "no_experiment_gpu_residual": True},
                                         {"state": "restored", "no_experiment_gpu_residual": False}, {}])
def test_complete_scores_require_restoration_and_no_gpu_residual(runtime, restoration):
    path = runtime / "formal-suite.json"
    suite = json.loads(path.read_text())
    suite["restoration"] = restoration
    write_json(path, suite)
    result = export(runtime)
    assert result["status"] == "incomplete"
    assert all(group["acceptance_complete"] for group in result["groups"])


def test_absent_suite_retains_all_four_unstarted_groups_without_cloud_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("CAMROLL_BUDGET_DB", raising=False)
    result = export(tmp_path)
    assert result["status"] == "incomplete"
    assert [group["name"] for group in result["groups"]] == list(ORDER)
    assert all(group["quality_distribution"]["unscored"] == 100 for group in result["groups"])
    assert result["historical_reference"]["contemporary_baseline"] is False


def attributed_fixture(runtime):
    path = runtime / "evaluation/runs/Q37/run.json"
    state = json.loads(path.read_text())
    state["build_summary"]["model_calls_file"] = "pipeline_calls-test.jsonl"
    write_json(path, state)
    pipeline = runtime / "albums/eval-Q37/pipeline_calls-test.jsonl"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("\n".join(json.dumps(row) for row in [
        {"call_id": "build1", "call_type": "caption", "model": "qwen3.8-flash"},
        {"call_id": "build1", "call_type": "caption", "model": "qwen3.8-flash", "replayed": True},
        {"call_id": "uncertain1", "call_type": "event", "status": "error"},
        {"call_id": "local_gpu", "call_type": "embedding", "operation": "/encode"}]) + "\n")
    (path.parent / "judge_calls.jsonl").write_text(json.dumps({"budget_call_id": "judge1", "role": "judge"}) + "\n")
    ledger = runtime / "shared.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE budget_calls(id TEXT PRIMARY KEY,status TEXT,reserved_units INTEGER,charged_units INTEGER,metadata TEXT)")
        cloud = {"model": "qwen3.8-flash", "response_model": "qwen3.8-flash-actual", "pricing_version": "qwen3.8-flash-cn-2026-09-30",
                 "prompt_tokens": 100, "completion_tokens": 10}
        judge = {"model": "doubao-seed-2.0-lite", "response_model": "doubao-actual", "input_rate": 1.8, "output_rate": 10.8,
                 "prompt_tokens": 50, "completion_tokens": 5}
        for identifier, status, reserved, charged, metadata in [
                ("build1", "settled", 2_000_000, 1_250_000, cloud),
                ("agent1", "settled", 1_000_000, 750_000, cloud),
                ("judge1", "settled", 1_000_000, 500_000, judge),
                ("uncertain1", "uncertain", 3_000_000, 0, {"model": "qwen3.8-flash"}),
                ("other-project", "settled", 99_000_000, 50_000_000, {"model": "qwen3.8-max", "role": "native-wiki"})]:
            connection.execute("INSERT INTO budget_calls VALUES (?,?,?,?,?)", (identifier, status, reserved, charged, json.dumps(metadata)))
    return ledger


def test_attributed_cost_uses_unique_owned_ids_excludes_concurrent_other_project_and_does_not_write_ledger(runtime):
    ledger = attributed_fixture(runtime)
    original = ledger.read_bytes()
    result = export(runtime, ledger_path=ledger)
    audit = result["groups"][0]["cost_audit"]
    assert audit["complete"] is True
    assert audit["request_count"] == 4
    assert audit["own_estimated_cny"] == 2.5
    assert audit["own_reserved_cny"] == 3
    assert audit["uncertain_requests"] == 1
    assert audit["missing_call_ids"] == []
    assert "local_gpu" not in audit["request_ids"] and "other-project" not in audit["request_ids"]
    by_model = {row["model"]: row for row in audit["by_model"]}
    assert set(by_model) == {"qwen3.8-flash", "doubao-seed-2.0-lite"}
    assert by_model["qwen3.8-flash"]["total_tokens"] == 220
    assert by_model["qwen3.8-flash"]["actual_models"] == ["qwen3.8-flash-actual"]
    assert by_model["qwen3.8-flash"]["rates"][0]["input_rate_cny_per_million"] == .8
    assert by_model["doubao-seed-2.0-lite"]["rates"][0]["source"] == "ledger_metadata"
    assert result["groups"][0]["cost"]["global_interval_estimated_spent_delta_cny"] == 10
    assert ledger.read_bytes() == original
    saved = json.loads((runtime / "evaluation/cost-audit.json").read_text())
    assert saved["runs"]["Q37"]["own_estimated_cny"] == 2.5
    assert "本组已记账估算" in (runtime / "reports/four-models.md").read_text()


def test_missing_ledger_row_leaves_own_total_unknown_and_reports_known_subtotal(runtime):
    ledger = attributed_fixture(runtime)
    with sqlite3.connect(ledger) as connection:
        connection.execute("DELETE FROM budget_calls WHERE id='judge1'")
    result = export(runtime, ledger_path=ledger)
    audit = result["groups"][0]["cost_audit"]
    assert audit["status"] == "partial"
    assert audit["own_estimated_cny"] is None
    assert audit["known_estimated_cny"] == 2
    assert audit["missing_call_ids"] == ["judge1"]


def test_resumed_build_attributes_all_exclusive_album_traces_once(runtime):
    ledger = attributed_fixture(runtime)
    earlier = runtime / "albums/eval-Q37/pipeline_calls-earlier.jsonl"
    earlier.write_text("\n".join(json.dumps(row) for row in [
        {"call_id": "cached-build", "call_type": "caption", "prompt_tokens": 20, "completion_tokens": 2},
        {"call_id": "build1", "call_type": "caption", "replayed": True}]) + "\n")
    other = runtime / "albums/unrelated-run/pipeline_calls-other.jsonl"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"call_id": "other-project", "call_type": "caption"}) + "\n")
    with sqlite3.connect(ledger) as connection:
        connection.execute("INSERT INTO budget_calls VALUES (?,?,?,?,?)",
                           ("cached-build", "settled", 1_000_000, 500_000, json.dumps({"model": "qwen3.8-flash"})))
    result = export(runtime, ledger_path=ledger)["groups"][0]
    audit = result["cost_audit"]
    assert audit["complete"] is True
    assert audit["own_estimated_cny"] == 3
    assert audit["request_count"] == 5
    assert "cached-build" in audit["request_ids"] and "other-project" not in audit["request_ids"]
    assert audit["pipeline_call_artifacts"] == sorted([str(earlier), str(earlier.with_name("pipeline_calls-test.jsonl"))])
    assert audit["album_id"] == "eval-Q37" and "exclusive to this run" in audit["attribution_assumption"]
    assert result["calls_and_tokens"]["by_role"]["caption"]["calls"] == 2
    assert result["calls_and_tokens"]["token_totals"]["total_tokens"] == 132


def test_invalid_earlier_build_trace_keeps_resumed_build_cost_partial(runtime):
    ledger = attributed_fixture(runtime)
    earlier = runtime / "albums/eval-Q37/pipeline_calls-earlier.jsonl"
    earlier.write_text('{"call_id":"not-persisted')
    audit = export(runtime, ledger_path=ledger)["groups"][0]["cost_audit"]
    assert audit["status"] == "partial"
    assert audit["own_estimated_cny"] is None
    assert audit["known_estimated_cny"] == 2.5
    assert audit["artifact_issues"] == [{"kind": "incomplete_or_invalid_call_record", "path": str(earlier), "line": 1}]


def test_unidentified_cloud_calls_are_partial_not_absent(runtime):
    ledger = attributed_fixture(runtime)
    path = runtime / "evaluation/runs/Q37/run.json"
    state = json.loads(path.read_text())
    state["items"][0]["model_call_metrics"] = []
    write_json(path, state)
    (runtime / "albums/eval-Q37/pipeline_calls-test.jsonl").write_text(json.dumps({"call_type": "caption", "status": "error"}) + "\n")
    (path.parent / "judge_calls.jsonl").write_text("")
    audit = export(runtime, ledger_path=ledger)["groups"][0]["cost_audit"]
    assert audit["status"] == "partial"
    assert audit["request_count"] == 0 and audit["unidentified_call_records"] == 1
    assert audit["own_estimated_cny"] is None


def test_ongoing_build_with_partial_jsonl_tail_never_claims_complete_or_zero_cost(runtime):
    ledger = attributed_fixture(runtime)
    path = runtime / "evaluation/runs/Q37/run.json"
    state = json.loads(path.read_text())
    state["status"] = "building"
    state["build_summary"].pop("model_calls_file")
    write_json(path, state)
    pipeline = runtime / "albums/eval-Q37/pipeline_calls-test.jsonl"
    with pipeline.open("a") as stream:
        stream.write('{"call_id":"inflight')
    audit = export(runtime, ledger_path=ledger)["groups"][0]["cost_audit"]
    assert audit["traces_complete"] is False
    assert audit["complete"] is False
    assert audit["own_estimated_cny"] is None
    assert audit["known_estimated_cny"] == 2.5
    assert audit["artifact_issues"][0]["kind"] == "incomplete_or_invalid_call_record"


def test_missing_ledger_does_not_create_database_and_keeps_explicit_unknown(runtime):
    ledger = runtime / "does-not-exist.sqlite"
    audit = export(runtime, ledger_path=ledger)["groups"][0]["cost_audit"]
    assert audit["status"] == "unavailable"
    assert audit["own_estimated_cny"] is None
    assert audit["missing_call_ids"] == ["agent1"]
    assert not ledger.exists()
