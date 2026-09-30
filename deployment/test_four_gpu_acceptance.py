import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("four_gpu_acceptance_test", ROOT / "four_gpu_acceptance.py")
acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acceptance
SPEC.loader.exec_module(acceptance)


class SyntheticClient:
    fingerprint = "a" * 64
    reranker_fingerprint = "b" * 64
    instances = []
    cached = False
    poison = False
    lose_chunks = False

    def __init__(self, base_url=None, *, cache_namespace):
        self.cache_namespace = cache_namespace
        self.last_timings = {}
        self.observers = []
        self.instances.append(self)

    @contextlib.contextmanager
    def capture_calls(self, callback):
        self.observers.append(callback)
        try:
            yield
        finally:
            self.observers.remove(callback)

    def _record(self, kind):
        for callback in self.observers:
            callback({"call_type": kind, "status": "success", "cache_namespace": self.cache_namespace})

    def encode(self, text=None, image_path=None, role="document"):
        vector = np.zeros(2048, dtype=np.float32)
        index = {"red": 0, "blue": 1, "checker": 2}[Path(image_path).stem] if image_path else 3
        if self.poison and self.cache_namespace.endswith("second") and role == "query":
            index = 8
        vector[index] = 1
        self.last_timings = {"items": [{"cache_hit": self.cached, "role": role, "truncated": False,
                                       "effective_input_tokens": 50}]}
        self._record("embedding")
        return vector

    def _request(self, path, payload):
        self._record("embedding")
        if path == "/split_text":
            text = payload["text"]
            chunks = [text[start:start + 2048] for start in range(0, len(text), 2048)]
            return {"chunks": chunks, "token_counts": [len(chunk) for chunk in chunks]}
        assert path == "/tokenize"
        return {"items": [{"truncated": False, "effective_input_tokens": len(text) + 20} for text in payload["texts"]]}

    def rerank(self, query, candidates):
        self._record("rerank")
        results, items = [], []
        for candidate in candidates:
            text = candidate["text"]
            chunks = [text[start:start + 2048] for start in range(0, len(text), 2048)]
            score = 0.1 if "blue" in candidate["id"] else 0.9
            results.append({"id": candidate["id"], "score": score, "chunk_count": len(chunks),
                            "chunk_scores": [score] * len(chunks)})
            if self.lose_chunks:
                results[-1].pop("chunk_scores")
            items.extend({"truncated": False, "effective_input_tokens": len(chunk) + 50,
                          "document_content_tokens": len(chunk)} for chunk in chunks)
        self.last_timings = {"cache_hit": False, "items": items}
        return results


@pytest.fixture
def runtime(tmp_path):
    SyntheticClient.instances = []
    SyntheticClient.cached = False
    SyntheticClient.poison = False
    SyntheticClient.lose_chunks = False
    telemetry = tmp_path / "qwen-worker/telemetry.jsonl"
    telemetry.parent.mkdir()
    telemetry.write_text("\n".join(json.dumps({"operation": "model_sha256_audit", "model": model, "pid": 123,
                                            "files": {"model.safetensors": {"sha256": "c" * 64, "size": 4255140312}}})
                                   for model in ("qwen", "reranker")) + "\n")
    return tmp_path


def test_complete_synthetic_acceptance_without_gpu_or_network(runtime):
    result = acceptance.run_acceptance(runtime, client_factory=SyntheticClient)
    assert result["status"] == "passed", result
    assert len(result["checks"]) == 20
    assert all(item["passed"] for item in result["checks"])
    assert result["worker_fingerprints"] == {"embedding": "a" * 64, "reranker": "b" * 64}
    assert len(result["fixtures"]) == 3
    assert len({client.cache_namespace for client in SyntheticClient.instances}) == 2
    assert all(Path(value["path"]).is_file() for value in result["fixtures"].values())
    assert (runtime / "gpu-acceptance.json").is_file()
    assert (runtime / "gpu-acceptance" / result["acceptance_id"] / "result.json").is_file()
    assert result["call_records"]


@pytest.mark.parametrize("flag,failed_check", [("cached", "query_before_image"),
                                              ("poison", "embedding_rope_consistency"),
                                              ("lose_chunks", "long_event_max_chunk_score")])
def test_rejects_cache_stale_rope_and_lost_chunks(runtime, flag, failed_check):
    setattr(SyntheticClient, flag, True)
    result = acceptance.run_acceptance(runtime, client_factory=SyntheticClient)
    assert result["status"] == "failed"
    assert result["checks"][-1]["name"] == failed_check
    assert result["checks"][-1]["passed"] is False
    assert json.loads((runtime / "gpu-acceptance.json").read_text())["status"] == "failed"


def test_missing_weight_audit_fails_acceptance(runtime):
    (runtime / "qwen-worker/telemetry.jsonl").write_text("")
    result = acceptance.run_acceptance(runtime, client_factory=SyntheticClient)
    assert result["status"] == "failed"
    assert result["checks"][-1]["name"] == "model_file_sha256_audit"


def test_rerun_keeps_prior_acceptance_artifact(runtime):
    first = acceptance.run_acceptance(runtime, client_factory=SyntheticClient)
    previous_path = runtime / "gpu-acceptance" / first["acceptance_id"] / "result.json"
    before = previous_path.read_bytes()
    second = acceptance.run_acceptance(runtime, client_factory=SyntheticClient)
    assert first["acceptance_id"] != second["acceptance_id"]
    assert previous_path.read_bytes() == before
    assert json.loads((runtime / "gpu-acceptance.json").read_text())["acceptance_id"] == second["acceptance_id"]
