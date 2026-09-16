import json
import shutil
import sqlite3
import threading

import numpy as np
import pytest
from PIL import Image

from camroll_agent.library import Library
from camroll_agent.pipeline import BuildBudgetExhausted, BuildProviderFailed, build, encode_images, enrich, inspect, reindex


class FakeEmbedding:
    fingerprint = "fake-wemm-v1"

    def __init__(self):
        self.calls = []
        self.active = 0
        self.lock = threading.Lock()

    def encode(self, *, text=None, image_path=None):
        with self.lock:
            self.active += 1
            assert self.active == 1, "WeMM work must be serial"
        try:
            self.calls.append((text, image_path))
            if image_path:
                color = Image.open(image_path).getpixel((0, 0))[0]
                return np.array([1 + color, 2], dtype=np.float32)
            return np.array([max(len(text), 1), 3], dtype=np.float32)
        finally:
            with self.lock:
                self.active -= 1

    def split_text(self, text):
        return [text[i:i + 12] for i in range(0, len(text), 12)]


class FakeQwen:
    model = "qwen3.7-flash"

    def __init__(self, fail_caption=False, budget=False):
        self.calls = []
        self.event_number = 0
        self.fail_caption = fail_caption
        self.budget = budget

    def generate_json(self, *, prompt, image_paths, schema, role, max_attempts):
        self.calls.append({"role": role, "prompt": prompt, "images": image_paths})
        assert max_attempts == 3
        if self.budget:
            raise BudgetExceeded("budget exhausted")
        if role == "caption":
            if self.fail_caption and Image.open(image_paths[0]).getpixel((0, 0))[0] > 100:
                raise RuntimeError("Injected caption failure")
            return {"caption": "中文照片中的红色杯子和花朵", "visible_text": "招牌咖啡",
                    "location": "模型观察地点", "people": ["穿蓝衣的人"]}
        self.event_number += 1
        return {"operation": "ADD" if self.event_number == 1 else "UPDATE",
                "name": "初始事件" if self.event_number == 1 else "改名事件", "description": "看到杯子与花朵"}


class BudgetExceeded(RuntimeError):
    pass


@pytest.fixture
def manifest(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    for name, color in (("a.png", 30), ("b.png", 180)):
        Image.new("RGB", (20, 20), (color, 10, 30)).save(source / name)
    shutil.copyfile(source / "a.png", source / "duplicate.png")
    spec = tmp_path / "input.json"
    spec.write_text(json.dumps({"album_id": "test-album", "root_folder": str(source),
                               "turns": [{"date": "2020-01-01", "user": {"image": "a.png"},
                                          "location": "真实地点", "answer": "秘密答案不应进入提示词"},
                                         {"date": "2021-01-01", "user": {"image": "b.png"}},
                                         {"date": "2022-01-01", "user": {"image": "duplicate.png"}}]}))
    return spec, tmp_path / "memory"


def test_build_dedup_event_rename_metadata_and_resume(manifest):
    spec, memory = manifest
    embedding, qwen = FakeEmbedding(), FakeQwen()
    result = build(spec, memory, qwen, embedding)
    assert result["source_count"] == 3
    assert result["image_count"] == 2
    assert result["event_count"] == 1
    assert result["text_coverage"] == 1
    assert inspect(spec)["profile_image"] is None
    lib = Library(memory, embedding)
    event = next(iter(lib.events.values()))
    assert event["name"] == "改名事件"
    assert len(event["image_ids"]) == 2
    assert all(r["event_id"] == event["id"] and r["event_name"] == "改名事件" for r in lib.images.values())
    first = next(r for r in lib.images.values() if r["date"] == "2020-01-01")
    assert len(first["sources"]) == 2
    assert first["location"] == "真实地点"
    assert first["observed_location"] == "模型观察地点"
    assert all("秘密答案" not in call["prompt"] for call in qwen.calls)
    assert all("memory/assets/" in path for _, path in embedding.calls if path)
    assert lib.image_paths([first["id"]]) == [first["path"]]
    calls = (len(qwen.calls), len(embedding.calls))
    resumed = build(spec, memory, qwen, embedding)
    assert (len(qwen.calls), len(embedding.calls)) == calls
    assert resumed["index_version"] == result["index_version"]
    assert resumed["cache_hits"]["caption"] == 2
    assert (memory / "source_spec.json").is_file()


def test_failed_caption_keeps_image_and_retries_only_missing_stage(manifest):
    spec, memory = manifest
    qwen, embedding = FakeQwen(fail_caption=True), FakeEmbedding()
    first = build(spec, memory, qwen, embedding)
    assert first["image_embedding_count"] == 2
    assert first["caption_count"] == 1
    assert first["text_coverage"] == .5
    assert len(Library(memory, embedding).search("照片", kind="images")["results"]) == 2
    calls = len([c for c in qwen.calls if c["role"] == "caption"])
    qwen.fail_caption = False
    second = build(spec, memory, qwen, embedding)
    assert len([c for c in qwen.calls if c["role"] == "caption"]) == calls + 1
    assert second["cache_hits"]["image_embedding"] == 2
    assert second["text_coverage"] == 1


def test_budget_stops_cloud_and_saves_resumable_image_index(manifest):
    spec, memory = manifest
    qwen, embedding = FakeQwen(budget=True), FakeEmbedding()
    with pytest.raises(BuildBudgetExhausted) as error:
        build(spec, memory, qwen, embedding, concurrency=1)
    assert len(qwen.calls) == 1
    assert error.value.summary["image_embedding_count"] == 2
    assert error.value.summary["budget_exhausted"]
    assert len(Library(memory, embedding).images) == 2
    with sqlite3.connect(memory / "state.sqlite") as db:
        assert db.execute("SELECT count(*) FROM stages WHERE status='ok'").fetchone()[0] == 2
    qwen.budget = False
    assert build(spec, memory, qwen, embedding)["text_coverage"] == 1


def test_permanent_provider_error_stops_further_cloud_submissions(manifest):
    spec, memory = manifest
    class Unauthorized(RuntimeError):
        status_code = 401
    class Reject(FakeQwen):
        def generate_json(self, **kwargs):
            self.calls.append(kwargs)
            raise Unauthorized("auth failed")
    qwen = Reject()
    with pytest.raises(BuildProviderFailed) as error:
        build(spec, memory, qwen, FakeEmbedding(), concurrency=1)
    assert len(qwen.calls) == 1
    assert not error.value.summary["budget_exhausted"]
    assert error.value.summary["stop_reason"] == "provider_error"
    assert len(Library(memory).images) == 2


def test_reindex_no_qwen_and_changed_model_version(manifest):
    spec, memory = manifest
    embedding = FakeEmbedding()
    old = build(spec, memory, FakeQwen(), embedding)
    old_reader = Library(memory, embedding)
    embedding.fingerprint = "fake-wemm-v2"
    new = reindex(memory, embedding)
    assert new["cloud_calls"] == 0
    assert new["index_version"] != old["index_version"]
    assert old_reader.version == old["index_version"]
    assert Library(memory, embedding).fingerprint == "fake-wemm-v2"


def test_reindex_rejects_preprocess_version_change_before_model_access(manifest, monkeypatch):
    from camroll_agent import pipeline
    spec, memory = manifest
    build(spec, memory, FakeQwen(), FakeEmbedding())
    pointer = (memory / "CURRENT").read_bytes()
    state = (memory / "state.sqlite").read_bytes()
    class ForbiddenEmbedding:
        @property
        def fingerprint(self):
            raise AssertionError("A rejected migration must not access the model")
    monkeypatch.setattr(pipeline, "PREPROCESS_VERSION", "future-preprocess-version")
    with pytest.raises(ValueError, match="build --resume"):
        reindex(memory, ForbiddenEmbedding())
    assert (memory / "CURRENT").read_bytes() == pointer
    assert (memory / "state.sqlite").read_bytes() == state
    assert not (memory / "last_index.json").exists()


def test_enrich_has_no_embedding_client_and_build_reuses_all_cloud_work(manifest):
    spec, memory = manifest
    qwen = FakeQwen()
    enriched = enrich(spec, memory, qwen)
    assert enriched["mode"] == "enrich" and not enriched["published"]
    assert enriched["embedding_fingerprint"] is None
    assert enriched["image_embedding_count"] == 0
    assert enriched["caption_count"] == 2 and enriched["event_count"] == 1
    assert not (memory / "CURRENT").exists()
    assert (memory / "last_enrich.json").exists()
    records = json.loads((memory / "enriched_records.json").read_text())
    assert all(row["embedding_status"] == "pending" for row in records["images"])
    calls = len(qwen.calls)
    built = build(spec, memory, qwen, FakeEmbedding())
    assert len(qwen.calls) == calls
    assert built["published"] and built["text_coverage"] == 1
    assert built["cache_hits"]["caption"] == 2 and built["cache_hits"]["event"] == 2
    original_current = (memory / "CURRENT").read_text()
    enrich(spec, memory, qwen)
    assert (memory / "CURRENT").read_text() == original_current
    assert len(qwen.calls) == calls


def test_enrich_small_then_full_reuses_successful_captions(manifest):
    spec, memory = manifest
    qwen = FakeQwen()
    first = enrich(spec, memory, qwen, max_images=1)
    assert first["caption_count"] == 1
    second = enrich(spec, memory, qwen)
    assert second["caption_count"] == 2
    assert second["cache_hits"]["caption"] == 1
    assert len([call for call in qwen.calls if call["role"] == "caption"]) == 2
    assert not (memory / "CURRENT").exists()


def test_all_embedding_failures_never_publish_empty_index(manifest):
    spec, memory = manifest
    class Unavailable(FakeEmbedding):
        def encode(self, **kwargs):
            raise RuntimeError("worker not available")
    qwen = FakeQwen()
    with pytest.raises(BuildProviderFailed) as error:
        build(spec, memory, qwen, Unavailable())
    assert not error.value.summary["published"]
    assert error.value.summary["stop_reason"] == "embedding_unavailable"
    assert not (memory / "CURRENT").exists()
    count = len(qwen.calls)
    build(spec, memory, qwen, FakeEmbedding())
    assert len(qwen.calls) == count


def test_encode_images_empty_memory_then_build_reuses_cache(manifest):
    spec, memory = manifest
    embedding = FakeEmbedding()
    encoded = encode_images(spec, memory, embedding)
    assert encoded["mode"] == "encode_images"
    assert encoded["source_count"] == 3 and encoded["image_count"] == 2
    assert encoded["image_embedding_count"] == 2 and encoded["new_embedding_count"] == 2
    assert encoded["cloud_calls"] == 0 and not encoded["published"]
    assert (memory / "last_image_encoding.json").exists()
    for name in ("CURRENT", "source_spec.json", "enriched_records.json"):
        assert not (memory / name).exists()
    with sqlite3.connect(memory / "state.sqlite") as db:
        assert db.execute("SELECT DISTINCT stage FROM stages").fetchall() == [("image_embedding",)]
        assert db.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    image_calls = len([path for _, path in embedding.calls if path])
    built = build(spec, memory, FakeQwen(), embedding)
    assert len([path for _, path in embedding.calls if path]) == image_calls
    assert built["cache_hits"]["image_embedding"] == 2
    assert built["published"]


def test_encode_images_preserves_enrichment_and_current(manifest):
    spec, memory = manifest
    qwen, embedding = FakeQwen(), FakeEmbedding()
    enrich(spec, memory, qwen)
    before = {name: (memory / name).read_bytes()
              for name in ("source_spec.json", "enriched_records.json", "last_enrich.json")}
    calls = len(qwen.calls)
    first = encode_images(spec, memory, embedding, max_images=1)
    assert first["image_embedding_count"] == 1
    second = encode_images(spec, memory, embedding)
    assert second["cache_hits"]["image_embedding"] == 1
    assert len(qwen.calls) == calls
    for name, content in before.items():
        assert (memory / name).read_bytes() == content
    assert not (memory / "CURRENT").exists()
    build(spec, memory, qwen, embedding)
    pointer = (memory / "CURRENT").read_bytes()
    again = encode_images(spec, memory, embedding)
    assert again["new_embedding_count"] == 0
    assert (memory / "CURRENT").read_bytes() == pointer
    assert len(qwen.calls) == calls


def test_encode_images_fingerprint_change_is_reported(manifest):
    spec, memory = manifest
    class Changing(FakeEmbedding):
        def encode(self, **kwargs):
            result = super().encode(**kwargs)
            self.fingerprint = "changed-during-run"
            return result
    with pytest.raises(BuildProviderFailed) as error:
        encode_images(spec, memory, Changing())
    assert error.value.summary["stop_reason"] == "fingerprint_changed"
    assert error.value.summary["image_embedding_count"] == 0
    with sqlite3.connect(memory / "state.sqlite") as db:
        assert db.execute("SELECT count(*) FROM stages WHERE status='ok'").fetchone()[0] == 0
    assert (memory / "last_image_encoding.json").is_file()
    assert not (memory / "CURRENT").exists()


def test_encode_images_can_run_alongside_enrichment(manifest):
    import concurrent.futures
    spec, memory = manifest
    cloud_started, release_cloud = threading.Event(), threading.Event()
    class WaitingQwen(FakeQwen):
        def generate_json(self, **kwargs):
            if kwargs["role"] == "caption":
                cloud_started.set()
                assert release_cloud.wait(5)
            return super().generate_json(**kwargs)
    qwen, embedding = WaitingQwen(), FakeEmbedding()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(enrich, spec, memory, qwen)
        assert cloud_started.wait(5)
        try:
            encoded = encode_images(spec, memory, embedding)
        finally:
            release_cloud.set()
        enriched = pending.result(timeout=5)
    assert encoded["image_embedding_count"] == 2 and enriched["caption_count"] == 2
    assert not (memory / "CURRENT").exists()
    calls = len(qwen.calls)
    built = build(spec, memory, qwen, embedding)
    assert built["cache_hits"]["image_embedding"] == 2
    assert len(qwen.calls) == calls


def test_exif_orientation_date_and_unknown_date(tmp_path):
    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6
    exif[306] = "2019:03:04 05:06:07"
    Image.new("RGB", (30, 10), "red").save(source, exif=exif)
    other = tmp_path / "other.png"
    Image.new("RGB", (10, 10), "green").save(other)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"images": [str(source), str(other)]}))
    memory = tmp_path / "memory"
    build(spec, memory, FakeQwen(), FakeEmbedding())
    rows = list(Library(memory).images.values())
    dated = next(r for r in rows if r["date"])
    assert dated["date"] == "2019-03-04" and dated["date_source"] == "exif"
    assert Image.open(dated["path"]).size == (10, 30)
    assert Image.open(dated["embedding_path"]).size == (10, 30)
    assert Image.open(dated["original_path"]).size == (30, 10)
    assert Image.open(source).size == (30, 10)
    assert next(r for r in rows if not r["date"])["date_source"] is None


def test_wemm_preparation_matches_frozen_image_bytes(tmp_path):
    from camroll_agent.pipeline import _prepare
    from PIL import ImageOps
    source = tmp_path / "large.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (3000, 1000), "purple").save(source, exif=exif)
    memory = tmp_path / "memory"
    (memory / "assets").mkdir(parents=True)
    _, view, embedding_path, original = _prepare(
        {"source_path": str(source), "date": None, "date_source": None}, memory)
    expected = tmp_path / "expected.jpg"
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((2048, 2048))
        image.save(expected)
    from pathlib import Path
    assert Path(embedding_path).read_bytes() == expected.read_bytes()
    assert Image.open(embedding_path).size == (683, 2048)
    assert Image.open(view).size == (1000, 3000)
    assert Path(original).read_bytes() == source.read_bytes()


def test_same_named_events_keep_distinct_stable_ids(manifest):
    spec, memory = manifest
    class NewEvents(FakeQwen):
        def generate_json(self, **kwargs):
            if kwargs["role"] == "event":
                return {"operation": "ADD", "name": "同名活动", "description": "不同的活动"}
            return super().generate_json(**kwargs)
    result = build(spec, memory, NewEvents(), FakeEmbedding())
    library = Library(memory)
    assert result["event_count"] == 2
    assert len({r["event_id"] for r in library.images.values()}) == 2
    assert {r["name"] for r in library.events.values()} == {"同名活动"}


def test_duplicate_sources_choose_known_date_and_retain_metadata(tmp_path):
    first = tmp_path / "first.png"
    alias = tmp_path / "alias.png"
    Image.new("RGB", (10, 10), "red").save(first)
    shutil.copyfile(first, alias)
    spec = tmp_path / "sources.json"
    spec.write_text(json.dumps({"images": [
        {"path": str(first), "date": None, "people": ["乙"]},
        {"path": str(alias), "date": "2020-05-06", "location": "明确地点", "people": ["甲", "乙"]},
        {"path": str(first), "date": "2021-01-01", "people": []}]}))
    memory = tmp_path / "memory"
    qwen = FakeQwen()
    enrich(spec, memory, qwen)
    records = json.loads((memory / "enriched_records.json").read_text())
    row = records["images"][0]
    assert row["date"] == "2020-05-06" and row["date_source"] == "manifest"
    assert row["date_source_path"] == str(alias)
    assert row["location"] == "明确地点" and set(row["people"]) == {"甲", "乙"}
    assert [source["date"] for source in row["sources"]] == [None, "2020-05-06", "2021-01-01"]
    caption_prompt = next(call["prompt"] for call in qwen.calls if call["role"] == "caption")
    assert "2020-05-06" in caption_prompt and "明确地点" in caption_prompt
    assert records["events"][0]["dates"] == ["2020-05-06", "2021-01-01"]


def test_duplicate_manifest_date_takes_priority_over_exif(tmp_path):
    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[306] = "2010:01:02 03:04:05"
    Image.new("RGB", (10, 10), "red").save(source, exif=exif)
    alias = tmp_path / "alias.jpg"
    shutil.copyfile(source, alias)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"images": [str(source), {"path": str(alias), "date": "2020-01-01"}]}))
    memory = tmp_path / "memory"
    enrich(spec, memory, FakeQwen())
    row = json.loads((memory / "enriched_records.json").read_text())["images"][0]
    assert row["date"] == "2020-01-01" and row["date_source"] == "manifest"
    assert {source["date"] for source in row["sources"]} == {"2010-01-02", "2020-01-01"}
