"""Synthetic acceptance against an already owned Qwen GPU window."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "camroll/camroll-agent"))

from camroll_agent.qwen_retrieval import QwenRetrievalClient


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _vector(vector, client, expected_role):
    array = np.asarray(vector)
    _require(array.shape == (2048,) and np.isfinite(array).all(), "Embedding must be finite and 2048 dimensional")
    _require(abs(float(np.linalg.norm(array)) - 1) <= 1e-4, "Embedding must be unit normalized")
    items = client.last_timings.get("items", [])
    _require(len(items) == 1 and items[0].get("cache_hit") is False, "Acceptance embedding must execute uncached")
    _require(items[0].get("role") == expected_role, "Embedding role was not preserved")
    _require(items[0].get("truncated") is False, "Embedding input was truncated or lacks a receipt")
    _require(0 < items[0].get("effective_input_tokens", 0) <= 8192, "Embedding tensor budget missing or exceeded")
    return {"dimension": len(array), "norm": float(np.linalg.norm(array)),
            "sha256": hashlib.sha256(array.astype(np.float32).tobytes()).hexdigest(),
            "telemetry": client.last_timings}


def _scores(scores, identifiers, client):
    _require([item["id"] for item in scores] == identifiers, "Reranker changed candidate identity/order")
    _require(all(math.isfinite(item["score"]) and 0 <= item["score"] <= 1 for item in scores), "Invalid rerank scores")
    telemetry = client.last_timings
    _require(telemetry.get("cache_hit") is False, "Rerank acceptance must execute uncached")
    items = telemetry.get("items", [])
    _require(bool(items) and all(item.get("truncated") is False for item in items), "Reranker truncated input or lacks a receipt")
    _require(all(0 < item.get("effective_input_tokens", 0) <= 10240 for item in items), "Reranker tensor budget missing or exceeded")
    _require(all(item.get("document_content_tokens", 2049) <= 2048 for item in items), "Reranker document chunk exceeds budget")
    return {"scores": scores, "telemetry": telemetry}


def _fixtures(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("checker", (255, 255, 255))):
        image = Image.new("RGB", (96, 96), color)
        if name == "checker":
            draw = ImageDraw.Draw(image)
            for row in range(0, 96, 16):
                for column in range(0, 96, 16):
                    if (row + column) // 16 % 2 == 0:
                        draw.rectangle((column, row, column + 15, row + 15), fill=(0, 0, 0))
        path = root / (name + ".png")
        image.save(path)
        paths[name] = path
    return paths


def run_acceptance(runtime_root, *, base_url=None, client_factory=None):
    runtime_root = Path(runtime_root).resolve()
    factory = client_factory or QwenRetrievalClient
    acceptance_id = "acceptance_gpu_" + uuid.uuid4().hex
    report_path = runtime_root / "gpu-acceptance" / acceptance_id / "result.json"
    report = {"status": "running", "acceptance_id": acceptance_id, "started_at_epoch": time.time(),
              "checks": [], "call_records": [], "synthetic_only": True,
              "worker_telemetry_path": str(runtime_root / "qwen-worker/telemetry.jsonl")}

    def check(name, action):
        tick = time.monotonic()
        try:
            detail = action()
        except Exception as error:
            report["checks"].append({"name": name, "passed": False, "error_type": type(error).__name__,
                                     "error": str(error), "duration_ms": (time.monotonic() - tick) * 1000})
            raise
        report["checks"].append({"name": name, "passed": True, "detail": detail,
                                 "duration_ms": (time.monotonic() - tick) * 1000})
        _write(report_path, report)
        return detail

    try:
        fixtures = _fixtures(runtime_root / "fixtures" / acceptance_id)
        report["fixtures"] = {name: {"path": str(path), "sha256": _sha(path)} for name, path in fixtures.items()}
        source_paths = [Path(__file__), RELEASE / "deployment/qwen_worker.py",
                        RELEASE / "camroll/camroll-agent/camroll_agent/qwen_retrieval.py",
                        RELEASE / "deployment/vendor/qwen_vl/qwen3_vl_embedding.py",
                        RELEASE / "deployment/vendor/qwen_vl/qwen3_vl_reranker.py"]
        report["source_hashes"] = {str(path.relative_to(RELEASE)): _sha(path) for path in source_paths}
        with contextlib.ExitStack() as stack:
            first = factory(base_url, cache_namespace=acceptance_id + "_first")
            second = factory(base_url, cache_namespace=acceptance_id + "_second")
            for client in (first, second):
                stack.enter_context(client.capture_calls(report["call_records"].append))
            report["worker_fingerprints"] = check("model_fingerprints", lambda: {
                "embedding": first.fingerprint, "reranker": first.reranker_fingerprint})
            text = "查找一张纯红色图片。"
            query_before = first.encode(text=text, role="query")
            check("query_before_image", lambda: _vector(query_before, first, "query"))
            red = first.encode(image_path=fixtures["red"], role="document")
            check("red_image", lambda: _vector(red, first, "document"))
            query_after = second.encode(text=text, role="query")
            check("query_after_image", lambda: _vector(query_after, second, "query"))

            def embedding_consistency():
                cosine = float(np.dot(query_before, query_after) / (np.linalg.norm(query_before) * np.linalg.norm(query_after)))
                _require(cosine >= 0.99999, "Text/image/text embedding consistency failed")
                return {"cosine": cosine, "threshold": 0.99999}

            check("embedding_rope_consistency", embedding_consistency)
            blue = second.encode(image_path=fixtures["blue"], role="document")
            check("blue_image", lambda: _vector(blue, second, "document"))
            checker = second.encode(image_path=fixtures["checker"], role="document")
            check("checker_image", lambda: _vector(checker, second, "document"))

            def image_distinction():
                _require(not np.allclose(red, blue, atol=1e-6), "Different image bytes returned the same embedding")
                return {"red_blue_cosine": float(np.dot(red, blue)), "red_checker_cosine": float(np.dot(red, checker))}

            check("different_images", image_distinction)
            document = second.encode(text="这是一张红色图片。", role="document")
            check("document_text", lambda: _vector(document, second, "document"))
            long_text = "这是用于验证完整分块的中文记录。\n红色蓝色与换行必须保留。😀\n" * 450

            def lossless_chunking():
                result = second._request("/split_text", {"model": "qwen", "text": long_text})
                chunks, counts = result["chunks"], result["token_counts"]
                _require("".join(chunks) == long_text, "Long Unicode split lost content")
                _require(sum(counts) > 4096 and all(0 < count <= 2048 for count in counts), "Long test did not exercise the required token boundaries")
                for offset in range(0, len(chunks), 32):
                    receipt = second._request("/tokenize", {"model": "qwen", "texts": chunks[offset:offset + 32], "role": "document"})
                    _require(all(item.get("truncated") is False and item["effective_input_tokens"] <= 8192 for item in receipt["items"]), "Chunk tensor validation failed")
                return {"characters": len(long_text), "chunk_count": len(chunks), "token_counts": counts,
                        "input_sha256": hashlib.sha256(long_text.encode()).hexdigest()}

            check("long_chinese_lossless_chunking", lossless_chunking)
            event_candidates = [{"id": "red_event", "text": "观察纯红色图片。"}, {"id": "blue_event", "text": "观察纯蓝色图片。"}]
            event_before = second.rerank(text, event_candidates)
            check("text_event_rerank", lambda: _scores(event_before, ["red_event", "blue_event"], second))
            image_candidates = [{"id": "red_image", "image_path": str(fixtures["red"]), "text": "纯红色画面。"},
                                {"id": "blue_image", "image_path": str(fixtures["blue"]), "text": "纯蓝色画面。"}]
            image_scores = second.rerank(text, image_candidates)
            check("actual_image_rerank", lambda: _scores(image_scores, ["red_image", "blue_image"], second))

            def color_ranking():
                _require(image_scores[0]["score"] > image_scores[1]["score"], "Red image was not ranked above blue for a red-image query")
                return {"red": image_scores[0]["score"], "blue": image_scores[1]["score"]}

            check("red_blue_relevance", color_ranking)
            event_after = second.rerank(text, event_candidates)
            check("text_event_rerank_after_images", lambda: _scores(event_after, ["red_event", "blue_event"], second))

            def rerank_consistency():
                delta = max(abs(before["score"] - after["score"]) for before, after in zip(event_before, event_after))
                _require(delta <= 1e-6, "Text/image/text rerank consistency failed")
                return {"max_absolute_delta": delta, "tolerance": 1e-6}

            check("reranker_rope_consistency", rerank_consistency)
            mixed = second.rerank(text, [image_candidates[0], event_candidates[1]])
            check("mixed_modality_rerank", lambda: _scores(mixed, ["red_image", "blue_event"], second))
            long_scores = second.rerank(text, [{"id": "long_event", "text": long_text}])

            def long_rerank():
                result = _scores(long_scores, ["long_event"], second)
                row = long_scores[0]
                _require(row["chunk_count"] >= 3 and len(row.get("chunk_scores", [])) == row["chunk_count"], "Long rerank did not report every chunk")
                _require(all(math.isfinite(score) for score in row["chunk_scores"]) and row["score"] == max(row["chunk_scores"]), "Long rerank did not use the maximum chunk score")
                _require(sum(item["document_content_tokens"] for item in second.last_timings["items"]) > 4096, "Rerank document did not cross 4096 tokens")
                return result

            check("long_event_max_chunk_score", long_rerank)
            query_reload = first.encode(text="换载后再次编码新的检索问题。", role="query")
            check("embedding_after_reranker_swap", lambda: _vector(query_reload, first, "query"))
            final_scores = second.rerank(text, event_candidates)
            check("reranker_after_embedding_swap", lambda: _scores(final_scores, ["red_event", "blue_event"], second))
            _require(first.fingerprint == second.fingerprint and first.reranker_fingerprint == second.reranker_fingerprint,
                     "Worker fingerprints changed between acceptance namespaces")

        def model_audit():
            audit = {}
            path = Path(report["worker_telemetry_path"])
            with path.open() as handle:
                for line in handle:
                    entry = json.loads(line)
                    if entry.get("operation") == "model_sha256_audit":
                        audit[entry["model"]] = entry
            _require(set(audit) >= {"qwen", "reranker"}, "Missing model weight audit records")
            _require(all("model.safetensors" in audit[name]["files"] for name in ("qwen", "reranker")), "Weight hashes missing in audit")
            report["model_audit"] = audit
            return {name: {"file_count": len(audit[name]["files"]), "pid": audit[name]["pid"]} for name in ("qwen", "reranker")}

        check("model_file_sha256_audit", model_audit)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        report["finished_at_epoch"] = time.time()
        report["duration_seconds"] = report["finished_at_epoch"] - report["started_at_epoch"]
        _write(report_path, report)
        _write(runtime_root / "gpu-acceptance.json", report)
    return report


def main():
    from four_config import RUNTIME, configure
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    arguments = parser.parse_args()
    configure(RELEASE)
    result = run_acceptance(arguments.runtime)
    print(json.dumps({"status": result["status"], "acceptance_id": result["acceptance_id"],
                      "checks": len(result["checks"]), "report": str(arguments.runtime / "gpu-acceptance.json")}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
