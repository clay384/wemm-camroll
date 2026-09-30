import copy
import json
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
