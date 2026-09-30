"""HTTP client for CamRoll's local Qwen embedding and reranking worker."""
from __future__ import annotations

import contextlib
import copy
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests

from .wemm import _bounded_timeout


class QwenRetrievalError(RuntimeError):
    def __init__(self, message, *, code="qwen_retrieval_error"):
        super().__init__(message)
        self.code = code


class QwenRetrievalClient:
    dimension = 2048
    provider = "qwen"

    def __init__(self, base_url=None, *, cache_namespace, timeout=600,
                 expected_fingerprint=None, expected_reranker_fingerprint=None):
        self.base_url = (base_url or os.getenv("CAMROLL_QWEN_BASE_URL") or os.getenv("CAMROLL_QWEN_WORKER_URL", "http://127.0.0.1:18778")).rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Qwen retrieval worker URL must be a loopback HTTP endpoint")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", cache_namespace):
            raise ValueError("A distinct safe cache_namespace is required")
        self.cache_namespace = cache_namespace
        self.timeout = timeout
        self._fingerprints = {"qwen": expected_fingerprint, "reranker": expected_reranker_fingerprint}
        self._local = threading.local()
        self._lock = threading.Lock()
        self._observers = []

    @property
    def last_timings(self):
        return getattr(self._local, "last_timings", {})

    @last_timings.setter
    def last_timings(self, value):
        self._local.last_timings = value

    @property
    def call_records(self):
        if not hasattr(self._local, "call_records"):
            self._local.call_records = []
        return self._local.call_records

    @contextlib.contextmanager
    def capture_calls(self, callback):
        with self._lock:
            self._observers.append(callback)
        try:
            yield
        finally:
            with self._lock:
                self._observers.remove(callback)

    def _session(self):
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            self._local.session = session
        return self._local.session

    def _accept_fingerprint(self, model, fingerprint):
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            raise QwenRetrievalError("Invalid worker fingerprint", code="invalid_response")
        with self._lock:
            current = self._fingerprints[model]
            if current and current != fingerprint:
                raise QwenRetrievalError("Qwen fingerprint changed; create a new index version", code="fingerprint_mismatch")
            self._fingerprints[model] = fingerprint

    def _request(self, path, payload=None, *, model="qwen"):
        started = time.monotonic()
        metric = {"call_id": uuid.uuid4().hex, "call_type": "rerank" if path == "/rerank" else "embedding",
                  "operation": path, "provider": "qwen", "model": model,
                  "cache_namespace": self.cache_namespace, "started_at_epoch": time.time(), "status": "running"}
        if payload:
            payload = {**payload, "cache_namespace": self.cache_namespace}
            metric.update(input_type="image" if "images" in payload else "text", role=payload.get("role"))
        try:
            session = self._session()
            try:
                timeout = _bounded_timeout(self.timeout)
                response = (session.get(self.base_url + path, timeout=timeout, allow_redirects=False) if payload is None
                            else session.post(self.base_url + path, json=payload, timeout=timeout, allow_redirects=False))
            except (requests.Timeout, requests.ConnectionError) as error:
                _bounded_timeout(1)
                raise QwenRetrievalError("Qwen transport failure: " + type(error).__name__, code="qwen_unavailable") from None
            _bounded_timeout(1)
            try:
                result = response.json()
            except ValueError:
                raise QwenRetrievalError("Qwen worker returned invalid JSON", code="invalid_response") from None
            if not isinstance(result, dict):
                raise QwenRetrievalError("Qwen worker returned invalid object", code="invalid_response")
            if response.status_code != 200:
                code = result.get("error", "qwen_retrieval_error")
                if code in {"content_length_exceeded", "total_length_exceeded"}:
                    code = "query_too_long"
                raise QwenRetrievalError(f"Qwen worker HTTP {response.status_code}: {code}", code=code)
            self._accept_fingerprint(model, result.get("fingerprint"))
            if path == "/fingerprints":
                self._accept_fingerprint("reranker", result.get("reranker_fingerprint"))
            timings = result.get("telemetry", {})
            if not isinstance(timings, dict):
                raise QwenRetrievalError("Invalid Qwen timing metadata", code="invalid_response")
            self.last_timings = timings
            metric.update(status="success", backend_timings=timings, fingerprint=result["fingerprint"])
            return result
        except Exception as error:
            metric.update(status="error", error_code=getattr(error, "code", type(error).__name__))
            raise
        finally:
            metric.update(finished_at_epoch=time.time(), duration_ms=(time.monotonic() - started) * 1000)
            self.call_records.append(metric)
            with self._lock:
                observers = list(self._observers)
            for callback in observers:
                callback(copy.deepcopy(metric))

    @property
    def fingerprint(self):
        if self._fingerprints["qwen"] is None:
            self._request("/fingerprints")
        return self._fingerprints["qwen"]

    @property
    def reranker_fingerprint(self):
        if self._fingerprints["reranker"] is None:
            self._request("/fingerprints")
        return self._fingerprints["reranker"]

    def encode(self, text=None, image_path=None, *, role="document"):
        if (text is None) == (image_path is None):
            raise ValueError("Supply text or image_path exclusively")
        if role not in {"document", "query"}:
            raise ValueError("Embedding role must be document or query")
        payload = {"model": "qwen", "role": role, "dimension": 2048}
        payload.update({"texts": [text]} if text is not None else {"images": [str(Path(image_path).resolve())]})
        result = self._request("/encode", payload)
        try:
            vectors = np.asarray(result["embeddings"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            raise QwenRetrievalError("Invalid Qwen embedding payload", code="invalid_response") from None
        if result.get("dimension") != 2048 or vectors.shape != (1, 2048) or not np.isfinite(vectors).all() or not np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-4):
            raise QwenRetrievalError("Invalid Qwen embedding shape/normalization", code="invalid_response")
        return vectors[0]

    def split_text(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        result = self._request("/split_text", {"model": "qwen", "text": text})
        chunks, counts = result.get("chunks"), result.get("token_counts")
        if (result.get("limit") != 2048 or not isinstance(chunks, list) or not isinstance(counts, list)
                or len(chunks) != len(counts) or any(not isinstance(chunk, str) or not chunk for chunk in chunks)
                or any(not isinstance(count, int) or not 0 <= count <= 2048 for count in counts)
                or "".join(chunks) != text):
            raise QwenRetrievalError("Qwen tokenizer split lost text or exceeded budget", code="invalid_response")
        return chunks

    def embed_many(self, texts, *, role="document"):
        return [self.encode(text=text, role=role).tolist() for text in texts]

    def rerank(self, query, candidates):
        if not 1 <= len(candidates) <= 30:
            raise ValueError("Rerank requires between 1 and 30 candidates")
        normalized = []
        for candidate in candidates:
            row = dict(candidate)
            if "type" not in row:
                row["type"] = "image" if row.get("image_path") else "text"
            if row["type"] == "image" and "text" in row:
                row["caption"] = "\n\n".join(part for part in (row.get("caption"), row.pop("text")) if part)
            normalized.append(row)
        identifiers = [row.get("id") for row in normalized]
        if any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Rerank candidate IDs must be non-empty and unique")
        result = self._request("/rerank", {"query": query, "candidates": normalized}, model="reranker")
        scores = result.get("scores")
        if not isinstance(scores, list) or len(scores) != len(identifiers):
            raise QwenRetrievalError("Reranker did not return every candidate", code="invalid_response")
        for identifier, row in zip(identifiers, scores):
            if (not isinstance(row, dict) or row.get("id") != identifier
                    or not isinstance(row.get("score"), (int, float)) or not math.isfinite(row["score"])
                    or not 0 <= row["score"] <= 1 or not isinstance(row.get("chunk_count"), int) or row["chunk_count"] < 1):
                raise QwenRetrievalError("Invalid reranker score/identity", code="invalid_response")
        return scores
