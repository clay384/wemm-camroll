"""Report projections use synthetic data and never rewrite frozen run records."""
from __future__ import annotations

import copy
import json
import os

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json
from camroll_eval.metrics import add_media_metrics, initial_item
from camroll_eval.service import create_app


@pytest.fixture
def report(tmp_path):
    root = tmp_path / "evaluation"
    media = {"img-a": {"asset_id": "img-a", "media_id": "synthetic.jpg",
                       "file_name": "synthetic.jpg", "media_type": "image"}}
    item = initial_item({"qa_id": "synthetic-question", "question": "Find the synthetic image",
                         "answer": "Synthetic reference answer", "answerability": "answerable",
                         "retrieval_image_ids": ["synthetic.jpg"]})
    turns = [{"index": 0, "message": "Find the image", "answer": "Which day?",
              "expected_action": "clarify", "turn_outcome": "clarify", "agent_status": "complete",
              "judge": {"score": 2, "reason": "Synthetic clarification"}},
             {"index": 1, "message": "The next day", "answer": "Synthetic final answer",
              "expected_action": "answer", "turn_outcome": "error", "agent_status": "error",
              "error_type": "SyntheticTimeout", "termination_reason": "timeout",
              "judge": {"score": 1, "reason": "Synthetic partial answer"}}]
    item.update(answer="Synthetic final answer", execution_status="error", agent_status="error",
                error_type="SyntheticTimeout", termination_reason="timeout", judge_status="completed",
                judge={"score": 1, "reason": "Synthetic partial answer"},
                task_judges=[{"expected_action": "clarify", "actual_action": "clarify", "correct": True},
                             {"expected_action": "answer", "actual_action": "error", "correct": False}],
                runtime_turns=turns, conversation=copy.deepcopy(turns), conversation_turn_count=2,
                conversation_context_mode="shared_history", timing_breakdown={"agent_wall_ms": 123},
                model_call_metrics=[
                    {"call_id": "synthetic-call-0", "call_type": "agent", "conversation_turn": 0,
                     "model": "synthetic-model", "status": "success", "duration_ms": 41,
                     "prompt_tokens": 100, "completion_tokens": 20,
                     "input": {"messages": [{"role": "user", "content": "input-" * 12000}]},
                     "output": {"message": "output-" * 12000}, "context": "context-" * 12000},
                    {"call_id": "synthetic-call-1", "call_type": "agent", "conversation_turn": 1,
                     "status": "error", "error_type": "SyntheticTimeout", "duration_ms": 42,
                     "input": {"messages": [{"role": "user", "content": "second-" * 12000}]},
                     "output": {"error": "SyntheticTimeout"}}],
                tool_trace=[{"tool": "search", "status": "ok", "duration_ms": 40,
                             "model_call_id": "synthetic-call-0", "model_call_index": 0,
                             "conversation_turn": 0, "args": {"query": "query-" * 12000},
                             "result": {"images": [{"id": "img-a", "caption": "caption-" * 12000}],
                                        "retrieval_trace": {"images": {"candidates": [
                                            {"id": "img-a", "recall_rank": 2, "final_rank": 1,
                                             "rerank_score": .9}]}}}}])
    add_media_metrics(item, {"selected_image_ids": ["img-a"], "retrieved_image_ids": ["img-a"],
                            "evidence_image_ids": ["img-a"]}, media)
    pending = {"qa_id": "synthetic-pending", "question": "Unstarted synthetic question",
               "execution_status": "not_started", "judge_status": "pending", "planned_turn_count": 1}
    state = {"run_id": "synthetic-run", "system_id": "camroll_qwen", "mode": "full",
             "status": "completed_with_errors", "qa_count": 2, "planned_turn_count": 3,
             "formal_protocol": True, "created_at": "2026-01-01T00:00:00Z",
             "config": "M4", "album_id": "synthetic-album", "items": [item],
             "qa_ids": [item["qa_id"], pending["qa_id"]], "media_mapping": media,
             "qa_plan": [{"qa_id": item["qa_id"], "question": item["question"], "planned_turn_count": 2}, pending],
             "planned_status": [pending], "hardware_context": {"gpu": {"name": "synthetic-gpu"}},
             "input_integrity": {"qa_sha256": "synthetic-qa", "dataset_sha256": "synthetic-images"}}
    path = root / "runs" / state["run_id"] / "run.json"
    atomic_json(path, state)
    source = tmp_path / "source"
    source.mkdir()
    app = create_app(root, source, tmp_path / "albums", read_only=True)
    with TestClient(app) as client:
        yield {"root": root, "path": path, "state": state, "item": item,
               "repo": app.state.repository, "client": client, "bytes": path.read_bytes()}


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def test_light_item_removes_large_payloads_but_keeps_review_evidence(report):
    full = report["repo"].item("synthetic-run", "synthetic-question")
    light = report["repo"].item("synthetic-run", "synthetic-question", light=True)
    assert full == report["item"]
    assert encoded_size(light) < encoded_size(full) / 20
    for field in ("answer", "reference_answer", "judge", "task_judges", "runtime_turns", "conversation",
                  "conversation_context_mode", "execution_status", "agent_status", "error_type",
                  "termination_reason", "predicted_media", "evidence_source_media", "gt_media",
                  "retrieval_precision", "retrieval_recall", "retrieval_f1"):
        assert light[field] == full[field], field
    assert light["trace_counts"] == {"model": 2, "tool": 1}
    for index, call in enumerate(light["model_call_metrics"]):
        assert call["trace_index"] == index
        assert not {"input", "output", "context"} & call.keys()
        assert call["call_id"] == full["model_call_metrics"][index]["call_id"]
        assert call["duration_ms"] == full["model_call_metrics"][index]["duration_ms"]
    for index, call in enumerate(light["tool_trace"]):
        assert call["trace_index"] == index
        assert not {"args", "result"} & call.keys()
        assert call["tool"] == full["tool_trace"][index]["tool"]


def test_run_projection_keeps_summary_and_pending_items_without_trace_payloads(report):
    detail = report["repo"].detail("synthetic-run")
    full = report["repo"].load("synthetic-run")
    assert detail["summary"] == full["summary"]
    assert detail["hardware_context"] == full["hardware_context"]
    assert detail["qa_plan"] == full["qa_plan"]
    assert {row["qa_id"] for row in detail["items"]} == {"synthetic-question", "synthetic-pending"}
    first = next(row for row in detail["items"] if row["qa_id"] == "synthetic-question")
    assert first["answer"] == report["item"]["answer"]
    assert first["judge"]["score"] == 1
    assert encoded_size(detail) < encoded_size(full) / 20
    assert "media_mapping" not in detail and "qa_ids" not in detail


@pytest.mark.parametrize("kind,field,index", [("model", "model_call_metrics", 0),
                                             ("model", "model_call_metrics", 1),
                                             ("tool", "tool_trace", 0)])
def test_on_demand_trace_is_exact_original_record(report, kind, field, index):
    expected = report["item"][field][index]
    assert report["repo"].item_trace("synthetic-run", "synthetic-question", kind, index) == expected
    response = report["client"].get(f"/api/runs/synthetic-run/items/synthetic-question/trace/{kind}/{index}")
    assert response.status_code == 200
    assert response.json() == expected


def test_new_view_and_legacy_full_item_and_export_remain_distinct(report):
    client = report["client"]
    full = client.get("/api/runs/synthetic-run/items/synthetic-question")
    light = client.get("/api/runs/synthetic-run/items/synthetic-question/view")
    detail = client.get("/api/runs/synthetic-run")
    assert full.status_code == light.status_code == detail.status_code == 200
    assert full.json() == report["item"]
    assert light.json() == report["repo"].item("synthetic-run", "synthetic-question", light=True)
    assert len(light.content) < len(full.content) / 20
    assert detail.json()["summary"] == report["repo"].load("synthetic-run")["summary"]
    # None of the read APIs may alter the file behind the immutable download.
    client.get("/api/runs")
    client.get("/api/compare?ids=synthetic-run")
    exported = client.get("/api/runs/synthetic-run/export")
    assert exported.status_code == 200
    assert exported.content == report["bytes"] == report["path"].read_bytes()


@pytest.mark.parametrize("suffix", ["trace/model/-1", "trace/model/2", "trace/tool/1",
                                    "trace/model/not-an-index", "trace/unknown/0"])
def test_invalid_trace_kind_or_index_is_404(report, suffix):
    response = report["client"].get(f"/api/runs/synthetic-run/items/synthetic-question/{suffix}")
    assert response.status_code == 404


@pytest.mark.parametrize("url", ["/api/runs/missing-run", "/api/runs/missing-run/items/synthetic-question/view",
                                 "/api/runs/synthetic-run/items/missing-question/view",
                                 "/api/runs/synthetic-run/items/missing-question/trace/model/0",
                                 "/api/runs/synthetic-run/items/synthetic-pending/trace/model/0"])
def test_missing_run_item_or_unstarted_trace_is_404(report, url):
    assert report["client"].get(url).status_code == 404


@pytest.mark.parametrize("method,url", [("POST", "/api/runs"),
                                       ("POST", "/api/runs/synthetic-run/cancel"),
                                       ("PUT", "/api/runs/synthetic-run"),
                                       ("PATCH", "/api/runs/synthetic-run/items/synthetic-question/view"),
                                       ("DELETE", "/api/runs/synthetic-run/items/synthetic-question/trace/model/0")])
def test_read_only_report_rejects_all_mutation_methods(report, method, url):
    response = report["client"].request(method, url, json={}, headers={"Origin": "http://testserver"})
    assert response.status_code == 405
    assert response.json()["error"] == "read_only_report"
    assert report["path"].read_bytes() == report["bytes"]


def test_cached_accessors_are_isolated_from_caller_mutations(report):
    repo = report["repo"]
    full = repo.load("synthetic-run")
    full["items"][0]["answer"] = "mutated load"
    full["items"][0]["model_call_metrics"][0]["input"]["messages"].clear()
    detail = repo.detail("synthetic-run")
    detail["hardware_context"]["gpu"]["name"] = "mutated detail"
    detail["items"][0]["judge"]["score"] = 0
    light = repo.item("synthetic-run", "synthetic-question", light=True)
    light["runtime_turns"][0]["answer"] = "mutated light"
    light["model_call_metrics"][0]["call_id"] = "mutated summary"
    item = repo.item("synthetic-run", "synthetic-question")
    item["model_call_metrics"][0]["input"]["messages"].clear()
    trace = repo.item_trace("synthetic-run", "synthetic-question", "model", 0)
    trace["input"]["messages"].clear()
    listing = repo.list()
    listing[0]["hardware_context"]["gpu"]["name"] = "mutated listing"
    comparison = repo.compare(["synthetic-run"])
    comparison["runs"][0]["hardware_context"]["gpu"]["name"] = "mutated comparison"
    assert repo.load("synthetic-run")["items"][0] == report["item"]
    assert repo.item("synthetic-run", "synthetic-question") == report["item"]
    assert repo.item_trace("synthetic-run", "synthetic-question", "model", 0) == report["item"]["model_call_metrics"][0]
    assert repo.item("synthetic-run", "synthetic-question", light=True)["runtime_turns"] == report["item"]["runtime_turns"]
    assert repo.detail("synthetic-run")["hardware_context"] == report["state"]["hardware_context"]
    assert repo.detail("synthetic-run")["items"][0]["judge"]["score"] == 1
    assert report["path"].read_bytes() == report["bytes"]


def test_run_file_stamp_refreshes_all_cached_projections(report):
    repo = report["repo"]
    repo.load("synthetic-run")
    repo.detail("synthetic-run")
    repo.item("synthetic-run", "synthetic-question", light=True)
    repo.item_trace("synthetic-run", "synthetic-question", "model", 0)
    repo.list()
    repo.compare(["synthetic-run"])
    updated = copy.deepcopy(report["state"])
    updated["items"][0]["answer"] = "Updated synthetic final answer"
    updated["items"][0]["judge"]["score"] = 2
    updated["items"][0]["model_call_metrics"][0]["input"] = {"messages": ["updated synthetic payload"]}
    updated["items"][0]["tool_trace"][0]["result"] = {"images": []}
    before = report["path"].stat()
    atomic_json(report["path"], updated)
    os.utime(report["path"], ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert repo.load("synthetic-run")["items"][0] == updated["items"][0]
    assert repo.detail("synthetic-run")["items"][0]["answer"] == updated["items"][0]["answer"]
    assert repo.item("synthetic-run", "synthetic-question", light=True)["judge"]["score"] == 2
    assert repo.item("synthetic-run", "synthetic-question") == updated["items"][0]
    assert repo.item_trace("synthetic-run", "synthetic-question", "model", 0) == updated["items"][0]["model_call_metrics"][0]
    assert repo.item_trace("synthetic-run", "synthetic-question", "tool", 0) == updated["items"][0]["tool_trace"][0]
    assert repo.list()[0]["summary"]["judge_distribution"]["2"] == 1
    assert repo.compare(["synthetic-run"])["rows"][0]["results"][0]["score"] == 2


def supplement_report(report):
    state = copy.deepcopy(report["state"])
    state.update(run_id="supplement-64", mode="supplement", formal_protocol=False,
                 parent_run_id="synthetic-run", qa_count=64, planned_turn_count=71,
                 planned_eligible_qa_count=52, supplement_summary_version="http403-supplement-v1",
                 build_skipped=True, cache_mode="reused_parent_snapshot_fresh_queries",
                 query_cache_policy="fresh_clients_no_parent_cache", source_unchanged=True,
                 summary={"retrieval_metric_count": 52, "retrieval_recall_macro": 1 / 52,
                          "quality_correct_planned_rate": 1 / 64, "planned_count": 64,
                          "task_decision_planned_count": 71, "judge_valid_count": 1,
                          "quality_unscored_planned_count": 63})
    path = report["root"] / "runs" / state["run_id"] / "run.json"
    atomic_json(path, state)
    return state, path


def test_supplement_uses_frozen_subset_summary_without_observed_denominator_recalculation(report, monkeypatch):
    state, path = supplement_report(report)
    before = path.read_bytes()
    def forbidden_recalculation(*args):
        raise AssertionError("Supplement subset metrics must not be recomputed from observed items")
    monkeypatch.setattr("camroll_eval.repository.summary", forbidden_recalculation)
    for result in (report["repo"].load(state["run_id"]), report["repo"].get_brief(state["run_id"]),
                   report["repo"].detail(state["run_id"]),
                   report["client"].get("/api/runs/" + state["run_id"]).json()):
        assert result["summary"] == state["summary"]
        assert result["planned_eligible_qa_count"] == 52
        assert result["source_unchanged"] is True
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [None, "unsupported-summary-version"])
def test_supplement_missing_version_never_silently_shrinks_denominator(report, version):
    state, path = supplement_report(report)
    state["supplement_summary_version"] = version
    atomic_json(path, state)
    with pytest.raises(ValueError, match="subset summary"):
        report["repo"].load(state["run_id"])


def test_supplement_is_excluded_from_formal_comparison(report):
    state, _ = supplement_report(report)
    with pytest.raises(ValueError, match="separately"):
        report["repo"].compare(["synthetic-run", state["run_id"]])
    assert report["client"].get("/api/compare?ids=synthetic-run,supplement-64").status_code == 404


def test_supplement_cost_sidecar_overlays_only_supplement_without_rewriting_first_run_audit(report):
    state, path = supplement_report(report)
    audit_path = report["root"] / "cost-audit.json"
    atomic_json(audit_path, {"runs": {"synthetic-run": {"own_estimated_cny": 2.5}}})
    original_audit = audit_path.read_bytes()
    atomic_json(report["root"] / "supplement-cost-audit.json", {"runs": {
        "supplement-64": {"own_estimated_cny": 1.25},
        "synthetic-run": {"own_estimated_cny": 999}}})
    before = path.read_bytes()
    client = report["client"]
    rows = {row["run_id"]: row for row in client.get("/api/runs").json()["runs"]}
    assert rows["synthetic-run"]["cost_audit"]["own_estimated_cny"] == 2.5
    assert rows["supplement-64"]["cost_audit"]["own_estimated_cny"] == 1.25
    assert client.get("/api/runs/supplement-64").json()["cost_audit"]["own_estimated_cny"] == 1.25
    assert client.get("/api/runs/supplement-64/export").content == before
    assert path.read_bytes() == before
    assert audit_path.read_bytes() == original_audit


def test_invalid_supplement_annotation_preserves_raw_export_and_original_runs(report):
    _, path = supplement_report(report)
    before = path.read_bytes()
    audit = {'valid': False, 'reason': 'Asset boundary mismatch', 'code': 'asset_binding_failure'}
    atomic_json(report['root'] / 'supplement-validity-audit.json', {'runs': {
        'supplement-64': audit, 'synthetic-run': audit}})
    client = report['client']
    rows = {row['run_id']: row for row in client.get('/api/runs').json()['runs']}
    assert rows['supplement-64']['validity_audit'] == audit
    assert 'validity_audit' not in rows['synthetic-run']
    assert client.get('/api/runs/supplement-64').json()['validity_audit'] == audit
    assert client.get('/api/runs/supplement-64/export').content == before
    assert path.read_bytes() == before
