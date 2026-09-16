import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from camroll_agent.identity import (POLICY, expanded_person_box, load_identity_spec,
                                   match_embedding, photo_faces, resolve_person, seed_references)
from camroll_agent.metadata import OfflineGeocoder, extract_exif, place_matches
from camroll_agent.library import Library
from camroll_agent.pipeline import BuildProviderFailed, _prepare, build
from camroll_agent.schemas import TOOL_NAMES, ToolValidationError, validate_arguments
from camroll_agent.tools import query_photo_people, view_image
from test_pipeline import FakeEmbedding, FakeQwen


class Exif(dict):
    def get_ifd(self, tag):
        return self.get(tag, {})


def test_original_time_offset_and_invalid_gps():
    tags = Exif({34665: {36867: "2019:07:22 16:18:34"},
                 34853: {1: "S", 2: (30, 15, 0), 3: "W", 4: (120, 30, 0)}})
    image = SimpleNamespace(getexif=lambda: tags)
    data = extract_exif(image)
    assert data["captured_at"] == "2019-07-22T16:18:34"
    assert data["capture_timezone"] == "unknown"
    assert data["gps"]["latitude"] == -30.25
    assert data["gps"]["longitude"] == -120.5
    tags[34665][36881] = "+08:00"
    assert extract_exif(image)["captured_at"] == "2019-07-22T16:18:34+08:00"
    tags[34853][2] = (30, 61, 0)
    assert extract_exif(image)["gps"] is None
    tags[34665][36867] = "not a date"
    assert extract_exif(image)["captured_at"] is None


def test_geonames_is_nearest_city_not_venue(monkeypatch):
    monkeypatch.setitem(sys.modules, "reverse_geocoder", SimpleNamespace(search=lambda *a, **kw: [
        {"lat": "10", "lon": "20", "name": "Example City", "admin1": "Example Province", "admin2": "Example District", "cc": "ZZ"}]))
    geo = OfflineGeocoder()
    result = geo.lookup({"latitude": 10.1, "longitude": 20.1})
    assert result["precision"] == "nearest_city" and not result["poi_supported"]
    assert result["distance_km"] > 0
    assert place_matches("Example City", result)
    assert not place_matches("Example Theme Park", result)
    assert geo.lookup(None)["status"] == "missing_gps"
    assert geo.lookup({"latitude": 200, "longitude": 20})["status"] == "invalid_gps"


def test_matching_ambiguity_threshold_and_invalid_vector():
    prototypes = {"person_a": [1, 0], "person_b": [0, 1]}
    good = match_embedding([1, 0], prototypes)
    assert good["identity_status"] == "matched" and good["person_id"] == "person_a"
    assert good["score_semantics"] == "cosine_similarity_not_probability"
    assert match_embedding([1, 1], prototypes)["identity_status"] == "unknown"
    assert match_embedding([-1, -1], prototypes)["identity_status"] == "unknown"
    assert match_embedding([1, 0], {})["reason"] == "no_named_references"
    with pytest.raises(ValueError):
        match_embedding([float("nan"), 0], prototypes)
    assert POLICY["cosine_min"] == .4 and POLICY["top_two_margin_min"] == .08


class FakeFaces:
    provenance = {"version": "independent-synthetic", "policy": POLICY, "model_sha256": {"fake": "abc"}}

    def detect(self, path):
        with Image.open(path) as image:
            width, height = image.size
        return [{"bbox": [2, 2, min(width, 32), min(height, 32)], "image_size": [width, height],
                 "detection_score": .99, "eligible": True, "embedding": [1, 0]}]


class FakeGeo:
    provenance = {"version": "independent-synthetic", "network": False}

    def lookup(self, gps):
        return {"status": "resolved", "source": "synthetic", "precision": "district",
                "label": "测试市测试区", "city": "测试市", "district": "测试区", "poi_supported": False}


@pytest.fixture
def identity_library(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    for name in ("reference.jpg", "photo.jpg"):
        image = Image.new("RGB", (80, 100), (30, 10, 20))
        exif = Image.Exif()
        exif[36867] = "2022:06:23 12:34:56"
        image.save(root / name, exif=exif)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"album_id": "independent-fixture", "root_folder": str(root),
        "input_policy": "originals_and_identity_references", "images": [{"path": "photo.jpg"}],
        "identity": {"persons": [{"person_id": "person_a", "name": "甲", "aliases": ["阿甲"],
                                    "reference_image": "reference.jpg"}], "relationships": []}}))
    memory = tmp_path / "memory"
    events = []
    summary = build(spec, memory, FakeQwen(), FakeEmbedding(), face_engine=FakeFaces(), geocoder=FakeGeo(), progress_callback=events.append)
    return spec, memory, Library(memory, FakeEmbedding()), summary, events


def test_build_registers_only_references_and_records_stage_spans(identity_library):
    spec, memory, lib, summary, events = identity_library
    row = next(iter(lib.images.values()))
    assert row["captured_at"] == "2022-06-23T12:34:56" and row["capture_timezone"] == "unknown"
    assert row["people"] == ["甲"] and row["observed_people"] == ["穿蓝衣的人"]
    assert row["face_instances"][0]["person_id"] == "person_a"
    assert "embedding" not in row["face_instances"][0]
    assert "reference_image" not in lib.metadata["people_registry"][0]
    assert summary["identity_summary"]["registered_people"] == 1
    assert {s["name"] for s in summary["stages"]} >= {"prepare_metadata", "identity_seed", "face_matching", "caption", "image_embedding", "event_aggregation", "text_embedding", "publish"}
    assert all(s["finished_at"] >= s["started_at"] and s["wall_ms"] >= 0 for s in summary["stages"])
    assert any(e["type"] == "stage_finished" for e in events)
    second = build(spec, memory, FakeQwen(), FakeEmbedding(), face_engine=FakeFaces(), geocoder=FakeGeo())
    assert second["index_version"] == summary["index_version"]
    assert second["cache_hits"]["face_matching"] == 1
    raw = json.loads(spec.read_text())
    raw["images"][0]["people"] = ["从答案偷来的名字"]
    spec.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="forbids"):
        build(spec, memory, FakeQwen(), FakeEmbedding(), face_engine=FakeFaces(), geocoder=FakeGeo())


def test_registered_filter_and_observed_facts_never_authenticate(identity_library):
    _, _, lib, _, _ = identity_library
    assert lib.search("照片", kind="images", person="阿甲")["count"] == 1
    assert lib.list_by_date(persons=["person_a"], location="测试区", kind="images")["count"] == 1
    assert lib.list_by_date(person="穿蓝衣的人", kind="images")["count"] == 0
    assert lib.list_by_date(location="模型观察地点", kind="images")["count"] == 0
    assert lib.list_by_date(location="测试公园", kind="images")["count"] == 0
    assert lib.list_by_date(person="未登记", kind="images")["filter_coverage"]["unregistered_person_filters"] == ["未登记"]


def test_photo_people_is_read_only_scoped_and_unknown_is_explicit(identity_library):
    _, memory, lib, _, _ = identity_library
    iid = next(iter(lib.images))
    before = (lib.snapshot_dir / "records.json").read_bytes()
    result = query_photo_people(image_ids=[iid], memory_dir=str(memory), library=lib)
    assert result["images"][0]["identity_status"] == "processed"
    assert result["images"][0]["face_instances"][0]["name"] == "甲"
    result = query_photo_people(image_ids=[iid], memory_dir=str(memory), library=lib, person="陌生人")
    assert result["requested_person_status"] == "not_registered"
    assert (lib.snapshot_dir / "records.json").read_bytes() == before
    with pytest.raises(ToolValidationError):
        query_photo_people(image_ids=["img_other_album"], memory_dir=str(memory), library=lib)
    assert "query_photo_people" in TOOL_NAMES and len(TOOL_NAMES) == 6


def test_target_crop_checks_identity_before_cloud_and_cleans_up(identity_library):
    _, memory, lib, _, _ = identity_library
    iid = next(iter(lib.images))
    class Vision:
        calls = []
        def generate_json(self, **kwargs):
            self.calls.append(kwargs)
            assert len(kwargs["image_paths"]) == 2
            assert all(Path(p).is_file() for p in kwargs["image_paths"])
            assert "扩展区域" in kwargs["prompt"]
            return {"analysis": "所见", "evidence": [{"image_id": iid, "observation": "颜色"}], "limitations": "不是人工确认"}
    client = Vision()
    result = view_image(image_ids=[iid], prompt="甲穿什么", memory_dir=str(memory), image_client=client, library=lib, target_person="甲")
    assert result["target_regions"][0]["person_id"] == "person_a"
    assert not Path(client.calls[0]["image_paths"][1]).exists()
    with pytest.raises(ToolValidationError, match="not registered"):
        view_image(image_ids=[iid], prompt="陌生人穿什么", memory_dir=str(memory), image_client=client, library=lib, target_person="陌生人")
    assert len(client.calls) == 1


def test_exif_rotation_and_crop_bounds(tmp_path):
    original = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (80, 40), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(original, exif=exif)
    memory = tmp_path / "memory"
    (memory / "assets").mkdir(parents=True)
    source = {"source_path": str(original), "date": None}
    _, derived, _, _ = _prepare(source, memory)
    with Image.open(derived) as corrected:
        assert corrected.size == (40, 80)
    assert source["exif_orientation"] == 6
    assert expanded_person_box([2, 3, 22, 30], [40, 80]) == [0, 0, 38, 80]


def test_reference_requires_single_face_and_alias_ambiguity(identity_library):
    spec, _, _, _, _ = identity_library
    data = load_identity_spec(spec)
    class Many(FakeFaces):
        def detect(self, path):
            return super().detect(path) * 2
    with pytest.raises(ValueError, match="exactly one"):
        seed_references(data, Many())
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_person("同名", [{"person_id": "person_1", "name": "同名"}, {"person_id": "person_2", "name": "同名"}])
    with pytest.raises(ToolValidationError):
        validate_arguments("view_image", {"image_ids": ["img_a"], "prompt": "看图", "face_id": "../../x"})


def test_reference_detector_path_is_distinct_from_full_photo(identity_library):
    spec, _, _, _, _ = identity_library
    data = load_identity_spec(spec)
    class ReferenceOnly(FakeFaces):
        def detect(self, path):
            raise AssertionError("Tightly cropped named references must use reference-specific padding")
        def detect_reference(self, path):
            return FakeFaces.detect(self, path)
    assert seed_references(data, ReferenceOnly())["references"][0]["status"] == "registered"
    assert POLICY["reference_padding_fraction"] == .25


def test_cancellation_settles_inflight_cache_without_publishing(identity_library, tmp_path):
    spec, _, _, _, _ = identity_library
    memory = tmp_path / "cancelled-build"
    cancelled = [False]
    class CancelAfterCaption(FakeQwen):
        def generate_json(self, **kwargs):
            result = super().generate_json(**kwargs)
            if kwargs["role"] == "caption":
                cancelled[0] = True
            return result
    qwen = CancelAfterCaption()
    with pytest.raises(BuildProviderFailed) as failure:
        build(spec, memory, qwen, FakeEmbedding(), face_engine=FakeFaces(), geocoder=FakeGeo(),
              cancel_callback=lambda: cancelled[0], concurrency=1)
    assert failure.value.summary["stop_reason"] == "cancelled"
    assert not failure.value.summary["published"] and not (memory / "CURRENT").exists()
    assert not any(c["role"] == "event" for c in qwen.calls)
    with sqlite3.connect(memory / "state.sqlite") as db:
        assert db.execute("SELECT count(*) FROM stages WHERE stage='caption' AND status='ok'").fetchone()[0] == 1
    assert json.loads((memory / "last_build.json").read_text())["stop_reason"] == "cancelled"


def test_cancel_before_work_keeps_existing_snapshot(identity_library):
    spec, memory, _, _, _ = identity_library
    before = (memory / "CURRENT").read_bytes()
    with pytest.raises(BuildProviderFailed) as failure:
        build(spec, memory, FakeQwen(), FakeEmbedding(), cancel_callback=lambda: True)
    assert failure.value.summary["stop_reason"] == "cancelled"
    assert (memory / "CURRENT").read_bytes() == before
