import shutil

import numpy as np
import pytest
from PIL import Image

from camroll_agent.library import Library
from camroll_agent.pipeline import _publish


class Embedding:
    fingerprint = "test-fingerprint"

    def encode(self, **kwargs):
        return np.array([1, 0], dtype=np.float32)


@pytest.fixture
def library_factory(tmp_path):
    memory = tmp_path / "album"
    assets = memory / "assets"
    assets.mkdir(parents=True)
    records = []
    for i, date in enumerate(["2020-01-01", "2021-01-01", None]):
        iid = f"img_{i}"
        path = assets / f"{i}.jpg"
        Image.new("RGB", (12, 12), (i * 50, 0, 0)).save(path)
        records.append({"id": iid, "image_id": iid, "kind": "image", "asset_id": str(i),
                        "path": str(path), "source_path": str(path), "date": date,
                        "caption": ["中文咖啡杯", "绿色帐篷", ""][i], "visible_text": "",
                        "event_id": "ev_a" if i < 2 else "ev_b", "event_name": "咖啡旅行",
                        "location": "北京", "people": [], "sources": []})
    events = [{"id": "ev_a", "event_id": "ev_a", "kind": "event", "name": "咖啡旅行",
               "description": "北京的帐篷和咖啡杯", "date_start": "2020-01-01", "date_end": "2021-01-01",
               "image_ids": ["img_0", "img_1"], "location": "北京", "people": []},
              {"id": "ev_b", "event_id": "ev_b", "kind": "event", "name": "其他",
               "description": "其他照片", "date_start": None, "date_end": None,
               "image_ids": ["img_2"], "location": "", "people": []}]
    vector_ids = {"image": ["img_0", "img_1", "img_2"],
                  "caption": ["img_1", "img_1", "img_0"], "event": ["ev_a", "ev_b"]}
    vectors = {"image": [[1, 0], [.8, .6], [0, 1]],
               "caption": [[1, 0], [.1, .995], [0, 1]], "event": [[0, 1], [1, 0]]}
    metadata = {"embedding_fingerprint": "test-fingerprint", "text_coverage": 2 / 3}

    def factory(config="M1"):
        _publish(memory, records, events, vector_ids, vectors, metadata)
        return Library(memory, Embedding(), config=config)
    factory.records = records
    factory.memory = memory
    return factory


def test_image_ranking_date_before_rank_and_event_aggregation(library_factory):
    lib = library_factory()
    assert lib.search("咖啡", kind="images")["results"][0]["id"] == "img_0"
    assert [r["id"] for r in lib.search("咖啡", kind="images", date_from="2021-01-01")["results"]] == ["img_1"]
    event = lib.search("咖啡", kind="events", date_from="2021-01-01")["results"][0]
    assert event["id"] == "ev_a"
    assert event["score"] == pytest.approx(.8)
    assert [r["kind"] for r in lib.search("咖啡", kind="both", top_k=4)["results"]] == ["image", "event", "image", "event"]


def test_rrf_chunks_max_dedup_ranks_and_ties(library_factory):
    lib = library_factory("M3")
    result = lib.search("咖啡", kind="images")["results"]
    assert [r["id"] for r in result] == ["img_0", "img_1", "img_2"]
    assert result[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert result[1]["branch_ranks"]["text"]["rank"] == 1
    assert result[2]["branch_ranks"].keys() == {"image"}
    assert result[0]["score"] == result[1]["score"]  # stable ID tie break
    assert len({r["id"] for r in result}) == len(result)


def test_grep_chinese_and_empty_notice(library_factory):
    lib = library_factory()
    assert lib.grep("绿色帐篷", kind="images")["results"][0]["id"] == "img_1"
    assert lib.grep("绿色帐篷", kind="images", date_to="2020-12-31")["count"] == 0
    missing = lib.grep("不存在词汇", kind="images")
    assert missing["count"] == 0
    assert "不能证明" in missing["notice"]
    assert lib.grep('" OR *', kind="images")["count"] == 0


def test_snapshot_pinned_and_image_ids_strict(library_factory):
    old = library_factory()
    old_version = old.version
    library_factory.records[0]["caption"] = "新文字"
    new = library_factory()
    assert new.version != old_version
    assert old.get("img_0")["caption"] == "中文咖啡杯"
    assert new.get("img_0")["caption"] == "新文字"
    assert old.image_paths(["img_0"])
    with pytest.raises(ValueError, match="Unknown image_id"):
        old.image_paths(["../../etc/passwd"])
    shutil.rmtree(library_factory.memory / "assets")
    with pytest.raises(ValueError, match="unavailable"):
        old.image_paths(["img_0"])


def test_fingerprint_and_validation_fail_explicitly(library_factory):
    lib = library_factory()
    with pytest.raises(ValueError):
        lib.search("x", top_k=True)
    with pytest.raises(ValueError):
        lib.search("x", date_from="2021-02-30")
    with pytest.raises(ValueError):
        lib.search("x", date_from="2022-01-01", date_to="2020-01-01")
    with pytest.raises(ValueError):
        lib.search("x", kind="paths")
    lib.embedding.fingerprint = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        lib.search("x")


def test_shared_query_vector_avoids_reencoding(library_factory):
    lib = library_factory()
    def forbidden(**kwargs):
        raise AssertionError("Query was encoded twice")
    lib.embedding.encode = forbidden
    assert lib.search("咖啡", kind="images", query_vector=[1, 0])["results"][0]["id"] == "img_0"


def test_default_client_is_lazy_and_reused(library_factory, monkeypatch):
    from camroll_agent import wemm
    library_factory()
    created = []
    def factory():
        created.append(True)
        return Embedding()
    monkeypatch.setattr(wemm, "EmbeddingClient", factory)
    lib = Library(library_factory.memory)
    assert lib.stats()["image_count"] == 3
    assert lib.get("img_0")["id"] == "img_0"
    assert not created
    assert lib.search("照片", kind="images")["count"] == 3
    lib.search("咖啡", kind="images")
    assert len(created) == 1


def test_duplicate_source_dates_are_searchable(library_factory):
    library_factory.records[0]["sources"] = [{"date": "2022-01-01"}]
    lib = library_factory()
    result = lib.list_by_date(date_from="2022-01-01", kind="images")
    assert result["count"] == 1
    assert result["results"][0]["id"] == "img_0"


@pytest.mark.parametrize("config", ["M1", "M3"])
@pytest.mark.parametrize("kind", ["images", "captions", "events"])
def test_search_skips_unused_branches_and_copies_only_top_k(library_factory, config, kind):
    lib = library_factory(config)
    original_rank, original_decorate = lib._rank, lib._decorate
    ranked, copied = [], []
    def rank(name, vector, eligible):
        ranked.append(name)
        return original_rank(name, vector, eligible)
    def decorate(rows, records):
        copied.extend(item[0] for item in rows)
        return original_decorate(rows, records)
    lib._rank, lib._decorate = rank, decorate
    result = lib.search("咖啡", kind=kind, top_k=1)
    expected = ["image"]
    if config == "M3":
        expected.append("event" if kind == "events" else "caption")
    assert ranked == expected
    assert copied == [result["results"][0]["id"]]
    assert result["count"] == (2 if kind == "events" else 3)


@pytest.mark.parametrize("config", ["M1", "M3"])
def test_both_prefix_keeps_interleaving_count_and_branch_ranks(library_factory, config):
    lib = library_factory(config)
    expected = [("img_0", 1), ("ev_a", 1), ("img_1", 2), ("ev_b", 2), ("img_2", 3)]
    for top_k in (1, 2, 3, 4, 5, 8):
        result = lib.search("咖啡", kind="both", top_k=top_k)
        assert [(row["id"], row["rank"]) for row in result["results"]] == expected[:top_k]
        assert result["count"] == 5


class QwenEmbedding(Embedding):
    provider = "qwen"
    dimension = 2048
    reranker_fingerprint = "test-reranker-fingerprint"

    def __init__(self):
        self.roles, self.rerank_calls = [], []
        self.scores = {"img_0": .1, "img_1": .9, "img_2": .9, "ev_a": .2, "ev_b": .8}

    def encode(self, **kwargs):
        self.roles.append(kwargs["role"])
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = 1
        return vector

    def rerank(self, query, candidates):
        self.rerank_calls.append((query, candidates))
        return [{"id": item["id"], "score": self.scores.get(item["id"], .5), "chunk_count": 1}
                for item in candidates]


def _qwen_library(library_factory, *, image_count=None):
    from camroll_agent.experiment import get_experiment_profile
    from camroll_agent.retrieval import experiment_metadata
    old = library_factory("M4")
    client = QwenEmbedding()
    profile = get_experiment_profile("Q37").with_fingerprints(
        embedding_fingerprint=client.fingerprint, reranker_fingerprint=client.reranker_fingerprint)
    images = list(old.images.values())
    vector_ids = old._vector_ids
    vectors = {name: np.pad(value, ((0, 0), (0, 2046))).tolist() for name, value in old._vectors.items()}
    if image_count:
        images = [dict(images[0], id=f"img_{number:03}", image_id=f"img_{number:03}")
                  for number in range(image_count)]
        vector_ids = {"image": [row["id"] for row in images], "caption": [], "event": []}
        vectors = {"image": [[1.0] + [0.0] * 2047 for row in images], "caption": [], "event": []}
    metadata = {**old.metadata, **experiment_metadata(profile, "test-q37")}
    _publish(old.memory_dir, images, list(old.events.values()), vector_ids, vectors, metadata)
    return Library(old.memory_dir, client, experiment_config="Q37"), client


def test_qwen_reranks_modalities_before_interleaving(library_factory):
    library, embedding = _qwen_library(library_factory)
    result = library.search("咖啡", top_k=4)
    assert embedding.roles == ["query"]
    assert [row["id"] for row in result["results"]] == ["img_1", "ev_b", "img_2", "ev_a"]
    image_candidates = embedding.rerank_calls[0][1]
    event_candidates = embedding.rerank_calls[1][1]
    assert all(item["image_path"] for item in image_candidates)
    assert image_candidates[0]["text"] == "中文咖啡杯"
    assert all("image_path" not in item for item in event_candidates)
    assert event_candidates[0]["text"] == "咖啡旅行\n北京的帐篷和咖啡杯"
    assert result["results"][0]["recall_rank"] == 2
    assert result["results"][0]["final_rank"] == 1
    assert result["results"][0]["branch_ranks"]["text"]["rank"] == 1
    assert result["retrieval_trace"]["images"]["reranked_count"] == 3


def test_qwen_filters_before_reranker_and_preserves_event_members(library_factory):
    library, embedding = _qwen_library(library_factory)
    result = library.search("咖啡", date_from="2021-01-01", top_k=4)
    assert [row["id"] for row in result["results"]] == ["img_1", "ev_a"]
    assert [item["id"] for item in embedding.rerank_calls[0][1]] == ["img_1"]
    assert result["results"][1]["matched_image_ids"] == ["img_1"]
    assert library.search("咖啡", location="不存在地点")["results"] == []
    assert len(embedding.rerank_calls) == 2


def test_qwen_reranks_first_thirty_and_marks_original_tail(library_factory):
    library, embedding = _qwen_library(library_factory, image_count=60)
    embedding.scores = {f"img_{number:03}": number / 100 for number in range(60)}
    result = library.search("照片", kind="images", top_k=50)
    assert len(embedding.rerank_calls[0][1]) == 30
    assert result["count"] == 50
    assert [row["id"] for row in result["results"][:30]] == [f"img_{number:03}" for number in reversed(range(30))]
    tail = result["results"][30:]
    assert [row["id"] for row in tail] == [f"img_{number:03}" for number in range(30, 50)]
    assert all(not row["reranked"] and row["rerank_score"] is None for row in tail)
    assert all(row["recall_rank"] == row["final_rank"] for row in tail)


@pytest.mark.parametrize("scores", [[], [{"id": "outside", "score": 1}],
                                     [{"id": "img_0", "score": float("nan")}]])
def test_qwen_invalid_reranking_never_falls_back(library_factory, scores):
    library, embedding = _qwen_library(library_factory)
    embedding.rerank = lambda *args: scores
    with pytest.raises(ValueError, match="Reranker"):
        library.search("照片", kind="images")


def test_qwen_snapshot_rejects_mixed_model_profile_and_fingerprint(library_factory):
    library, embedding = _qwen_library(library_factory)
    with pytest.raises(ValueError, match="configuration"):
        Library(library.memory_dir, embedding, experiment_config="Q38")
    with pytest.raises(ValueError, match="namespace"):
        Library(library.memory_dir, embedding, cache_namespace="another-run")
    with pytest.raises(ValueError, match="mode"):
        Library(library.memory_dir, embedding, config="M1")
    with pytest.raises(ValueError, match="provider"):
        Library(library.memory_dir, Embedding()).search("照片")
    embedding.reranker_fingerprint = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        library.search("照片")


def test_snapshot_vector_hash_checked_before_queries(library_factory):
    library = library_factory()
    values = dict(library._vectors)
    values["image"] = values["image"].copy()
    values["image"][0, 0] = .9
    np.savez(library.snapshot_dir / "vectors.npz", **values)
    with pytest.raises(ValueError, match="hash"):
        Library(library.memory_dir)
