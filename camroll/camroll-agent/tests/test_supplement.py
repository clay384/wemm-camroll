"""Independent W38Max HTTP 403 supplements; synthetic data and no paid calls."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from test_photobench_eval import source, FakeJudge
from camroll_agent.experiment import get_experiment_profile
from camroll_agent.retrieval import cache_key, experiment_metadata
from camroll_eval.common import atomic_json, read_json, sha256
from camroll_eval.dataset import Dataset
from camroll_eval.metrics import initial_item, summary
from camroll_eval.supplement import SupplementRunner, select_http403_items


def error_item(qa_id, **values):
    return {"qa_id": qa_id, "execution_status": "error", "judge": {"score": None}, **values}


def test_selects_only_structured_http403_failures_including_valid_judged_items():
    chosen = [
        error_item("model-code", model_call_metrics=[{"error_code": "http_403"}]),
        error_item("model-status", model_call_metrics=[{"http_status": 403}]),
        error_item("tool-status", tool_trace=[{"tool": "view_image", "result": {
            "error": {"http_status": 403, "type": "provider_error"}}}]),
        error_item("valid-judge", model_call_metrics=[{"error_code": "http_403"}],
                   judge={"score": 2, "reason": "A valid prior judgment must remain unchanged"}),
    ]
    excluded = [
        error_item("answer-only", answer="The string http_403 and HTTP 403 is part of an answer"),
        error_item("reason-only", judge={"score": 0, "reason": "http_403"}),
        error_item("different-status", model_call_metrics=[{"error_code": "http_500", "http_status": 500}]),
        error_item("successful", execution_status="complete", model_call_metrics=[{"error_code": "http_403"}]),
        error_item("error-free", model_call_metrics=[{"status": "success"}]),
        error_item("quoted-request", model_call_metrics=[{"input": {"error_code": "http_403"},
                                                          "output": {"example": {"http_status": 403}}}]),
    ]
    parent = {"mode": "full", "status": "completed_with_errors",
              "items": [chosen[0], excluded[0], chosen[1], excluded[1], *chosen[2:], *excluded[2:]]}
    before = copy.deepcopy(parent)
    assert select_http403_items(parent) == chosen
    assert parent == before


def tree_hashes(root):
    return {str(path.relative_to(root)): sha256(path) for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def frozen_parent(tmp_path, source, monkeypatch):
    root, albums = tmp_path / "evaluation", tmp_path / "albums"
    parent_id, album_id, version = "original-w38max", "eval-original-w38max", "1" * 32
    dataset = Dataset(source, strict=False)
    configuration = {**get_experiment_profile("W38Max").as_dict(), "cache_namespace": parent_id}
    profile = get_experiment_profile("W38Max").with_fingerprints(embedding_fingerprint="synthetic-wemm-fingerprint")
    images = [{"id": "img_" + sha256(path), "asset_id": sha256(path), "kind": "image",
               "filename": path.name, "path": str(path), "source_path": str(path),
               "caption": "Synthetic frozen caption", "visible_text": "", "event_id": None}
              for path in dataset.photos]
    vectors = np.zeros((len(images), 4096), dtype=np.float32)
    vectors[:, 0] = 1
    metadata = {**experiment_metadata(profile, parent_id), "embedding_fingerprint": profile.embedding_fingerprint,
                "dimensions": 4096, "image_count": len(images), "image_embedding_count": len(images),
                "vector_hashes": {"image": hashlib.sha256(vectors.tobytes()).hexdigest()}}
    memory = albums / album_id
    snapshot = memory / "snapshots" / version
    snapshot.mkdir(parents=True)
    atomic_json(snapshot / "records.json", {"images": images, "events": [], "metadata": metadata,
                                            "vector_ids": {"image": [row["id"] for row in images]}})
    np.savez(snapshot / "vectors.npz", image=vectors)
    (memory / "CURRENT").write_text(version + "\n")
    atomic_json(memory / "experiment.json", {"profile": "W38Max", "cache_namespace": parent_id,
               "configuration_hash": cache_key([], profile, parent_id)})
    (memory / "state.sqlite").write_bytes(b"Synthetic source state must remain untouched")
    items = []
    for row in dataset.rows:
        item = initial_item(row)
        item.update(execution_status="error", answer="Original first-attempt answer",
                    model_call_metrics=[{"call_id": "original-" + row["qa_id"], "call_type": "agent",
                                         "error_code": "http_403", "http_status": 403}],
                    judge_status="completed", judge={"score": 2, "reason": "Original judgment"},
                    runtime_turns=[{"index": 0, "message": row["question"], "answer": "Original partial answer"}],
                    resume_history=[{"role": "user", "content": "Do not reuse this original context"}])
        items.append(item)
    parent = {"run_id": parent_id, "album_id": album_id, "index_version": version,
              "system_id": "camroll_wemm", "experiment": "W38Max", "experiment_config": configuration,
              "mode": "full", "formal_protocol": True, "status": "completed_with_errors",
              "config": "M4", "qa_concurrency": 1, "judge_concurrency": 4, "caption_concurrency": 4,
              "qa_set": "100qa-full", "dataset_id": dataset.manifest["album_id"],
              "qa_count": len(dataset.rows), "planned_turn_count": 3,
              "qa_ids": [row["qa_id"] for row in dataset.rows],
              "qa_plan": [{"qa_id": row["qa_id"], "question": row["question"],
                           "planned_turn_count": len(dataset.turns(row))} for row in dataset.rows],
              "input_integrity": dataset.integrity(), "snapshot_metadata": metadata,
              "code_revision": "four-8874328", "build_summary": {"published": True, "index_version": version,
                  "image_count": len(images), "image_embedding_count": len(images)},
              "judge_protocol": FakeJudge().protocol(), "judge_model": "fake-judge", "items": items,
              "metric_source": {"name": "synthetic-frozen-scoring"},
              "phases": {}, "created_at": "2026-09-30T00:00:00Z", "finished_at": "2026-09-30T01:00:00Z"}
    parent["summary"] = summary(parent)
    parent_path = root / "runs" / parent_id / "run.json"
    atomic_json(parent_path, parent)
    freeze = tmp_path / "synthetic-freeze.json"
    atomic_json(freeze, {"synthetic": True})
    checks = []
    def frozen_core(parent_value, freeze_path):
        checks.append((copy.deepcopy(parent_value), Path(freeze_path)))
        return {"verified": True, "synthetic": True, "code_revision": "four-8874328"}
    monkeypatch.setattr("camroll_eval.supplement.verify_frozen_core", frozen_core)
    return {"root": root, "albums": albums, "source": source, "parent": parent,
            "parent_path": parent_path, "memory": memory, "version": version,
            "freeze": freeze, "checks": checks, "images": images}


class SupplementBackend:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.options = []
        self.questions = []
        self.agent_memories = []

    def factory(self, **options):
        self.options.append(options)
        return self

    def budget(self):
        return {"estimated_spent_cny": 25, "remaining_cny": 575, "limit_cny": 600}

    def build(self, *args, **kwargs):
        raise AssertionError("A supplement must never cold-build or change the parent album")

    def agent(self, memory, config):
        memory = Path(memory)
        self.agent_memories.append((memory, config))
        version = (memory / "CURRENT").read_text().strip()
        ids = [row["id"] for row in read_json(memory / "snapshots" / version / "records.json")["images"]]
        backend = self
        class Agent:
            def ask(self, question, history=None, **kwargs):
                backend.questions.append({"question": question, "history": copy.deepcopy(history),
                                          "request_context": kwargs.get("request_context")})
                response = {"final_text": "Synthetic supplementary answer", "index_version": version,
                            "turn_outcome": "answer", "stopped_reason": "completed",
                            "selected_image_ids": ids[:1], "retrieved_image_ids": ids, "evidence_image_ids": ids[:1],
                            "conversation_history": (history or []) + [{"role": "user", "content": question},
                                {"role": "assistant", "content": "Synthetic supplementary answer"}],
                            "model_call_metrics": [{"call_id": "supplement-call-" + str(len(backend.questions)),
                                                    "call_type": "agent", "duration_ms": 1,
                                                    "prompt_tokens": 10, "completion_tokens": 2}],
                            "tool_trace": []}
                if backend.failure:
                    response.update(turn_outcome="error", error_type=backend.failure,
                                    stopped_reason="provider_error")
                    if backend.failure == "http_403":
                        response["model_call_metrics"][0].update(error_code="http_403", http_status=403)
                return response
        return Agent()


class SupplementJudge(FakeJudge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs
        self.scored = []

    def score_item(self, item, row):
        self.scored.append(row["qa_id"])
        return super().score_item(item, row)


def run_synthetic(fixture, backend=None, *, run_id="supplement-one", judge_type=SupplementJudge, **options):
    backend = backend or SupplementBackend()
    judges = []
    def judge_factory(**kwargs):
        judge = judge_type(**kwargs)
        judges.append(judge)
        return judge
    runner = SupplementRunner(fixture["root"], fixture["source"], fixture["albums"],
                              backend_factory=backend.factory, judge_factory=judge_factory,
                              sample_resources=False)
    backend.runner = runner
    result = runner.run_supplement(fixture["parent"]["run_id"], run_id=run_id,
                                  freeze_path=fixture["freeze"], strict=False,
                                  expected_qa_count=options.pop("expected_qa_count", 2),
                                  expected_turn_count=options.pop("expected_turn_count", 3), **options)
    return result, backend, judges


def test_supplement_is_separate_immutable_snapshot_reuse_with_fresh_dialogue(frozen_parent):
    parent_bytes = frozen_parent["parent_path"].read_bytes()
    before = tree_hashes(frozen_parent["memory"])
    result, backend, judges = run_synthetic(frozen_parent)
    assert result["run_id"] == "supplement-one" and result["mode"] == "supplement"
    assert result["parent_run_id"] == "original-w38max"
    assert result["parent_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    assert result["source_unchanged"] is True
    assert result["snapshot_hashes"]
    assert result["query_cache_policy"] == "fresh_clients_no_parent_cache"
    assert result["index_version"] == frozen_parent["version"]
    assert result["qa_ids"] == ["one", "two"]
    assert result["qa_count"] == 2 and result["planned_turn_count"] == 3
    assert result["config"] == "M4" and (result["qa_concurrency"], result["judge_concurrency"]) == (1, 4)
    assert len(backend.questions) == 3
    assert [entry["history"] for entry in backend.questions[:2]] == [[], []]
    assert backend.questions[2]["history"] == [
        {"role": "user", "content": "which event"},
        {"role": "assistant", "content": "Synthetic supplementary answer"}]
    assert [entry["request_context"]["turn_index"] for entry in backend.questions] == [0, 0, 1]
    assert {entry["request_context"]["run_id"] for entry in backend.questions} == {"supplement-one"}
    assert sorted(qa for judge in judges for qa in judge.scored) == ["one", "two"]
    assert backend.options[0]["run_id"] == "supplement-one"
    assert Path(backend.options[0]["journal_path"]) != frozen_parent["root"] / "request-journal.sqlite"
    assert all(judge.options["request_namespace"] == "supplement-one" for judge in judges)
    assert all(Path(judge.options["request_journal_path"]) == Path(backend.options[0]["journal_path"]) for judge in judges)
    assert backend.options[0]["experiment_config"]["cache_namespace"] == "original-w38max"
    for memory, config in backend.agent_memories:
        assert memory != frozen_parent["memory"] and config == "M4"
        assert memory == Path(result["pinned_memory"])
        assert not (memory / "state.sqlite").exists()
        assert (memory / "CURRENT").read_text().strip() == frozen_parent["version"]
        assert (memory / "snapshots" / frozen_parent["version"] / "records.json").read_bytes() == (
            frozen_parent["memory"] / "snapshots" / frozen_parent["version"] / "records.json").read_bytes()
    assert frozen_parent["parent_path"].read_bytes() == parent_bytes
    assert tree_hashes(frozen_parent["memory"]) == before
    assert frozen_parent["checks"]
    assert result["mode"] != "composite" and "repaired_valid_count" not in result
    assert {path.name for path in (frozen_parent["root"] / "runs").iterdir()} == {"original-w38max", "supplement-one"}


def test_http403_stops_later_questions_but_judges_completed_failure(frozen_parent):
    result, backend, judges = run_synthetic(frozen_parent, SupplementBackend(failure="http_403"))
    assert len(backend.questions) == 1
    assert [row["qa_id"] for row in result["items"]] == ["one"]
    assert result["items"][0]["execution_status"] == "error"
    assert result["items"][0]["judge"]["score"] == 2
    assert [qa for judge in judges for qa in judge.scored] == ["one"]
    assert result["status"] == "blocked"
    assert result["stop_reason"] == "http_403"
    pending = next(row for row in result["planned_status"] if row["qa_id"] == "two")
    assert pending["execution_status"] == "not_started" and pending["executed_turn_count"] == 0
    assert result["qa_count"] == 2 and result["planned_turn_count"] == 3


@pytest.mark.parametrize("failure", ["RequestUncertain", "BudgetExceeded", "BudgetLimitChanged"])
def test_unknown_request_or_budget_stop_never_submits_next_question(frozen_parent, failure):
    result, backend, _ = run_synthetic(frozen_parent, SupplementBackend(failure=failure))
    assert len(backend.questions) == 1
    assert result["status"] not in {"completed", "completed_with_errors"}
    assert result.get("stop_reason")
    pending = next(row for row in result["planned_status"] if row["qa_id"] == "two")
    assert pending["execution_status"] == "not_started"


@pytest.mark.parametrize("counts", [{"expected_qa_count": 64, "expected_turn_count": 71},
                                    {"expected_qa_count": 2, "expected_turn_count": 71},
                                    {"expected_qa_count": 64, "expected_turn_count": 3}])
def test_selection_count_mismatch_blocks_before_backend_or_new_run(frozen_parent, counts):
    backend = SupplementBackend()
    with pytest.raises(ValueError):
        run_synthetic(frozen_parent, backend, **counts)
    assert not backend.options and not backend.questions
    assert not (frozen_parent["root"] / "runs" / "supplement-one").exists()


def test_existing_run_is_never_resumed_or_overwritten(frozen_parent):
    result, _, _ = run_synthetic(frozen_parent)
    path = frozen_parent["root"] / "runs" / result["run_id"] / "run.json"
    before = path.read_bytes()
    backend = SupplementBackend()
    with pytest.raises((ValueError, FileExistsError)):
        run_synthetic(frozen_parent, backend)
    assert not backend.questions and not backend.options
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("experiment", "W37"), ("mode", "supplement"),
                                        ("status", "evaluating"), ("config", "M1"),
                                        ("qa_concurrency", 2), ("judge_concurrency", 1)])
def test_parent_must_be_terminal_original_w38max_with_frozen_execution_settings(frozen_parent, field, value):
    parent = copy.deepcopy(frozen_parent["parent"])
    parent[field] = value
    atomic_json(frozen_parent["parent_path"], parent)
    backend = SupplementBackend()
    with pytest.raises(ValueError):
        run_synthetic(frozen_parent, backend)
    assert not backend.options and not backend.questions


def test_snapshot_vector_dimension_is_validated_before_submission(frozen_parent):
    vector_path = frozen_parent["memory"] / "snapshots" / frozen_parent["version"] / "vectors.npz"
    np.savez(vector_path, image=np.ones((2, 2048), dtype=np.float32))
    backend = SupplementBackend()
    with pytest.raises(ValueError):
        run_synthetic(frozen_parent, backend)
    assert not backend.options and not backend.questions


@pytest.mark.parametrize("invalid", ["final_missing", "final_boolean", "turn_boolean", "final_float",
                                    "task_missing_correct", "task_nonboolean_correct"])
def test_invalid_final_or_turn_score_never_marks_supplement_complete(frozen_parent, invalid):
    class MalformedJudge(SupplementJudge):
        def score_item(self, item, row):
            super().score_item(item, row)
            item["judge"] = copy.deepcopy(item["judge"])
            if invalid == "final_missing":
                item.pop("judge")
            elif invalid == "final_boolean":
                item["judge"]["score"] = True
            elif invalid == "final_float":
                item["judge"]["score"] = 2.0
            elif invalid == "turn_boolean":
                item["runtime_turns"][0]["judge"]["score"] = True
            elif invalid == "task_missing_correct":
                item["runtime_turns"][0]["task_judge"].pop("correct")
            else:
                item["runtime_turns"][0]["task_judge"]["correct"] = 1
            return item
    result, backend, _ = run_synthetic(frozen_parent, judge_type=MalformedJudge)
    assert len(backend.questions) == 3
    assert result["supplement_complete"] is False
    assert result["status"] == "completed_with_errors"
    assert result["source_unchanged"] is True and result["pinned_snapshot_unchanged"] is True


def test_budget_final_read_failure_still_persists_answers_and_integrity_evidence(frozen_parent):
    class FinalBudgetFailure(SupplementBackend):
        def budget(self):
            if self.runner._state and self.runner._state.get("status") == "completed":
                raise RuntimeError("Synthetic final budget read unavailable")
            return super().budget()
    original = frozen_parent["parent_path"].read_bytes()
    before = tree_hashes(frozen_parent["memory"])
    result, backend, _ = run_synthetic(frozen_parent, FinalBudgetFailure())
    persisted = read_json(frozen_parent["root"] / "runs" / result["run_id"] / "run.json")
    assert len(backend.questions) == 3
    assert result == persisted
    assert len(result["items"]) == 2 and sum(len(item["runtime_turns"]) for item in result["items"]) == 3
    assert result["status"] == "failed" and result["supplement_complete"] is False
    assert result["stop_reason"] == "budget_check_failed" and result["budget_after"] is None
    assert {"phase": "budget_after", "type": "RuntimeError"} in result["finalization_errors"]
    assert result["source_unchanged"] is True and result["pinned_snapshot_unchanged"] is True
    assert frozen_parent["parent_path"].read_bytes() == original
    assert tree_hashes(frozen_parent["memory"]) == before
