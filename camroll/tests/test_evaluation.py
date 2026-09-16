import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("evaluate_retrieval", Path(__file__).resolve().parents[1] / "scripts/evaluate_retrieval.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_scoring_deduplicates_candidates_and_targets():
    metrics, ranks = evaluation.score_ranking(["wrong", "wrong", "a", "a", "b"], ["a", "b", "b"])
    assert ranks == {"a": 2, "b": 3}
    assert metrics["Hit@1"] == 0
    assert metrics["Hit@5"] == 1
    assert metrics["Recall@10"] == 1
    assert metrics["AllCoverage@10"] == 1
    assert metrics["MRR@50"] == 0.5


def test_multiple_targets_retains_cardinality_and_missing_target():
    metrics, ranks = evaluation.score_ranking(["a"], ["a", "b", "c"])
    assert metrics["Hit@1"] == 1
    assert metrics["Recall@20"] == pytest.approx(1 / 3)
    assert metrics["AllCoverage@20"] == 0
    assert ranks == {"a": 1, "b": None, "c": None}


def test_not_found_failed_or_rank_beyond_fifty_scores_zero():
    for hits, failed in [([], False), (["a"], True), ([str(i) for i in range(50)] + ["a"], False)]:
        metrics, ranks = evaluation.score_ranking(hits, ["a"], failed=failed)
        assert not any(metrics.values())
        assert ranks == {"a": None}
    with pytest.raises(ValueError):
        evaluation.score_ranking(["a"], [])


class FakeEmbedding:
    fingerprint = "test-fingerprint"
    last_timings = {"embedding_s": 0.01}

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return np.array([1, 0], dtype=np.float32)


class FakeLibrary:
    metadata = {"dimensions": 2}

    def __init__(self, asset, *, fail=False):
        self.asset = asset
        self.fail = fail
        self.vectors = []

    def search(self, query, *, kind, top_k, query_vector):
        assert kind == "images" and top_k == 50
        self.vectors.append(query_vector)
        if self.fail:
            raise RuntimeError("private server error")
        return {"results": [{"asset_id": self.asset, "kind": "image", "id": "img_" + self.asset,
                             "rank": 1, "score": 0.9, "branch_ranks": {"image": {"rank": 1, "score": 0.9}}}]}


def question():
    return {"qa_id": "q1", "question": "原始中文问题", "event_id": "event1", "gold_assets": ["a" * 64],
            "target_count": 1, "frozen_eligibility": {"cohort": "main", "composition": "image"}}


def test_encode_once_reuses_identical_vector_and_preserves_arm_failure(tmp_path):
    embedding = FakeEmbedding()
    libs = {"M1": FakeLibrary("a" * 64), "M3": FakeLibrary("b" * 64, fail=True)}
    row = evaluation.evaluate_question(question(), libs, embedding, tmp_path, "lock")
    assert embedding.calls == ["原始中文问题"]
    assert libs["M1"].vectors[0] is libs["M3"].vectors[0]
    assert row["outcomes"]["M1"]["metrics"]["Hit@1"] == 1
    assert row["outcomes"]["M3"]["status"] == "ERROR"
    assert row["outcomes"]["M3"]["metrics"]["Hit@1"] == 0
    assert "private server" not in json.dumps(row)
    evaluation.evaluate_question(question(), libs, embedding, tmp_path, "lock")
    assert len(embedding.calls) == 1  # Verified query-vector cache is reused.


def test_resume_retains_failed_row_and_repairs_only_partial_tail(tmp_path):
    embedding = FakeEmbedding()
    libs = {"M1": FakeLibrary("a" * 64), "M3": FakeLibrary("b" * 64, fail=True)}
    row = evaluation.evaluate_question(question(), libs, embedding, tmp_path, "lock")
    complete_bytes = (evaluation.json_text(row) + "\n").encode()
    path = tmp_path / "results.jsonl"
    path.write_bytes(complete_bytes + b'{"incomplete":')
    resumed = evaluation.load_resume(tmp_path, [question()], "lock")
    assert resumed["q1"]["outcomes"]["M3"]["status"] == "ERROR"
    assert path.read_bytes() == complete_bytes
    assert len(list(tmp_path.glob("incomplete-tail-*"))) == 1


def test_resume_refuses_changed_lock_or_query_vector(tmp_path):
    embedding = FakeEmbedding()
    libs = {"M1": FakeLibrary("a" * 64), "M3": FakeLibrary("b" * 64)}
    row = evaluation.evaluate_question(question(), libs, embedding, tmp_path, "lock")
    (tmp_path / "results.jsonl").write_text(evaluation.json_text(row) + "\n")
    with pytest.raises(ValueError, match="frozen run"):
        evaluation.load_resume(tmp_path, [question()], "changed")
    vector_path = tmp_path / "query_vectors" / (row["query_vector"]["key"] + ".npy")
    vector_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="vector was changed"):
        evaluation.load_resume(tmp_path, [question()], "lock")


def test_event_bootstrap_is_paired_deterministic_and_cluster_weighted():
    rows = []
    for event, gain, n in [("large", 1.0, 4), ("small", 0.0, 1)]:
        for _ in range(n):
            rows.append({"event_id": event, "outcomes": {
                "M1": {"status": "OK", "metrics": {m: 0.0 for m in evaluation.METRICS}},
                "M3": {"status": "OK", "metrics": {m: gain for m in evaluation.METRICS}},
            }})
    result = evaluation.paired_bootstrap(rows, samples=1000)
    assert result == evaluation.paired_bootstrap(rows, samples=1000)
    assert result["Hit@1"]["delta_M3_minus_M1"] == 0.8
    assert result["Hit@1"]["ci95"] == [0.0, 1.0]
    assert result["failure_rate"]["ci95"] == [0.0, 0.0]


def test_source_hash_mismatch_refused_before_data_use(tmp_path):
    for name in evaluation.APPROVED_HASHES:
        (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="frozen source hash mismatch"):
        evaluation.load_frozen(tmp_path)
