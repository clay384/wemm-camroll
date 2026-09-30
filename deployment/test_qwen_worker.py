import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("camroll_qwen_worker_test", ROOT / "qwen_worker.py")
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


class Tokenizer:
    padding_side = "right"

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        for index, character in enumerate(text):
            weight = 3 if ord(character) > 127 else 1
            ids.extend([ord(character)] * weight)
            offsets.extend([(index, index + 1)] * weight)
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    value = worker.Runtime()
    value.cache_root = tmp_path / "vectors"
    value.telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(value, "processor", lambda model: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(value, "fingerprint", lambda model: "e" * 64 if model == "qwen" else "f" * 64)
    return value


@pytest.mark.parametrize("text", ["", "中文😀 " * 2300, " " * 7000, "x" * 2049, "a\n\t b"])
def test_split_preserves_unicode_and_whitespace(runtime, text):
    result = runtime.split_text(worker.SplitRequest(text=text, cache_namespace="test"))
    assert "".join(result["chunks"]) == text
    assert all(0 < count <= 2048 for count in result["token_counts"])
    assert all(result["chunks"])


def test_encode_requires_explicit_role_and_namespace():
    with pytest.raises(ValidationError):
        worker.EncodeRequest(texts=["q"])
    with pytest.raises(ValidationError):
        worker.EncodeRequest(model="wemm", texts=["q"], role="document", cache_namespace="test")
    with pytest.raises(worker.WorkerError, match="exactly one"):
        worker.request_items(worker.EncodeRequest(texts=["q"], images=["image"], role="query", cache_namespace="test"))


def test_document_query_instructions_differ():
    query = worker.messages_for("qwen", "one", role="query")
    document = worker.messages_for("qwen", "one", role="document")
    assert query[0] != document[0]
    assert query[1] == document[1]


def test_candidate_validation():
    worker.Candidate(id="image", type="image", image_path="image.png")
    worker.Candidate(id="event", type="text", text="event text")
    for payload in ({"id": "x", "type": "text", "text": ""},
                    {"id": "x", "type": "image"},
                    {"id": "x", "type": "text", "text": "event", "image_path": "x"}):
        with pytest.raises(ValidationError):
            worker.Candidate(**payload)


def test_rerank_chunks_image_and_text_max_score(runtime, monkeypatch):
    captured = []
    monkeypatch.setattr(worker, "read_image", lambda path: ("IMAGE", {"input_sha256": "i" * 64}))
    monkeypatch.setattr(runtime, "load", lambda model: 0.0)

    def prepare(model, messages, count, expected_images, trace):
        text = messages[-1]["content"][-1]["text"]
        captured.append((text, expected_images, count))
        return {"text": text}, {"effective_input_tokens": count, "truncated": False}

    monkeypatch.setattr(runtime, "prepare", prepare)
    monkeypatch.setattr(runtime, "run_rerank", lambda inputs: (0.9 if "TARGET" in inputs["text"] else 0.1, {"inference_ms": 1.0}))
    request = worker.RerankRequest(query="query", cache_namespace="test", candidates=[
        worker.Candidate(id="photo", type="image", image_path="image.png", caption="x" * 2200 + "TARGET"),
        worker.Candidate(id="event", type="text", text="z" * 4100 + "TARGET"),
    ])
    result = runtime.rerank(request)
    assert [(row["id"], row["score"], row["chunk_count"]) for row in result["scores"]] == [("photo", 0.9, 2), ("event", 0.9, 3)]
    assert [entry[1] for entry in captured] == [1, 1, 0, 0, 0]
    assert "".join(entry[0] for entry in captured[:2]) == request.candidates[0].caption
    assert all(entry[2] <= 2048 + 5 for entry in captured)
    assert not result["telemetry"]["cache_hit"]


def test_rerank_no_silent_fallback(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "prepare", lambda *args: (_ for _ in ()).throw(worker.WorkerError("total_length_exceeded", "too long", 413)))
    request = worker.RerankRequest(query="q", cache_namespace="test", candidates=[worker.Candidate(id="event", type="text", text="event")])
    with pytest.raises(worker.WorkerError) as error:
        runtime.rerank(request)
    assert error.value.category == "total_length_exceeded"


def test_image_allowlist_rejects_symlink_escape(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "outside.png"
    secret.write_bytes(b"image")
    (allowed / "escape.png").symlink_to(secret)
    monkeypatch.setattr(worker, "ALLOWED_IMAGE_ROOTS", (allowed,))
    with pytest.raises(worker.WorkerError) as error:
        worker.validate_image_path(str(allowed / "escape.png"))
    assert error.value.category == "image_path_forbidden"


def test_model_audit_checks_content_not_only_size(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model = model_root / "Qwen3-VL-Embedding-2B"
    model.mkdir(parents=True)
    weight = model / "model.safetensors"
    weight.write_bytes(b"original")
    manifest = {"models": {"Qwen/Qwen3-VL-Embedding-2B": {"complete": True, "files": [
        {"path": weight.name, "size": 8, "sha256": hashlib.sha256(b"original").hexdigest()}]}}}
    manifest_path = model_root / "MODEL_WORKER_DOWNLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(worker, "MODEL_ROOT", model_root)
    monkeypatch.setitem(worker.PATHS, "qwen", model)
    monkeypatch.setattr(worker, "MANIFEST_SHA256", hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    value = worker.Runtime()
    value.telemetry_path = tmp_path / "audit.jsonl"
    assert value.verify_model("qwen")[weight.name]["size"] == 8
    weight.write_bytes(b"tampered")
    fresh = worker.Runtime()
    fresh.telemetry_path = tmp_path / "audit.jsonl"
    with pytest.raises(worker.WorkerError) as error:
        fresh.verify_model("qwen")
    assert error.value.category == "model_unverified"


def test_rope_reset_only_owned_model(runtime):
    state = SimpleNamespace(rope_deltas="stale")
    runtime.loaded = SimpleNamespace(model=SimpleNamespace(model=state))
    runtime.reset_independent_state("qwen")
    assert state.rope_deltas is None
    runtime.loaded = SimpleNamespace(model=state)
    state.rope_deltas = "stale"
    runtime.reset_independent_state("reranker")
    assert state.rope_deltas is None


def test_gpu_disabled_never_loads(runtime):
    runtime.gpu_enabled = False
    with pytest.raises(worker.WorkerError) as error:
        runtime.load("qwen")
    assert error.value.category == "GPU_WINDOW_CLOSED"


def test_foreign_gpu1_process_blocks_load_without_signaling(runtime, monkeypatch):
    runtime.gpu_enabled = True
    monkeypatch.setattr(runtime, "verify_model", lambda model: {})
    answers = iter(["0, GPU-unrelated\n1, " + worker.GPU_UUID, worker.GPU_UUID + ", 123456"])
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    with pytest.raises(worker.WorkerError) as error:
        runtime.load("qwen")
    assert error.value.category == "gpu_in_use"
    assert runtime.loaded is None


def test_vendor_matches_locked_sources():
    vendor = ROOT / "vendor/qwen_vl"
    source = json.loads((vendor / "SOURCE_LOCK.json").read_text())
    patch = json.loads((vendor / "COMPATIBILITY_PATCH.json").read_text())
    assert hashlib.sha256((vendor / "qwen3_vl_embedding.py").read_bytes()).hexdigest() == patch["patched_sha256"]
    assert hashlib.sha256((vendor / "qwen3_vl_reranker.py").read_bytes()).hexdigest() == source["files"]["qwen3_vl_reranker.py"]["sha256"]
