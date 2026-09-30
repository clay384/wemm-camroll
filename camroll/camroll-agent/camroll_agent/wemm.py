"""HTTP-only client for the existing single GPU WeMM worker."""
from __future__ import annotations

import os
import contextlib
import copy
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests


class WeMMError(RuntimeError):
    def __init__(self, message, *, code='wemm_error'):
        super().__init__(message)
        self.code = code


def _bounded_timeout(default):
    # Embedding does not consume a Qwen request, but must honor the same question
    # deadline. The import constructs no client and never reads credentials.
    from .qwen import DeadlineExceeded, _scope
    scope = _scope.get()
    if scope is None:
        return default
    remaining = scope.deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded()
    return min(default, remaining)


def _backoff(attempt):
    wait = 0.5 * 2 ** attempt
    if _bounded_timeout(wait) < wait:
        from .qwen import DeadlineExceeded
        raise DeadlineExceeded()
    time.sleep(wait)
    _bounded_timeout(1)


class EmbeddingClient:
    provider = "wemm"
    dimension = 4096

    def __init__(self, base_url=None, key_path=None, *, timeout=180):
        self.base_url = (base_url or os.getenv("WEMM_BASE_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.key_path = Path(key_path or os.getenv("WEMM_KEY_PATH", "/home/realmagic/wemm-embedding-lab/run/access.key"))
        self.timeout = timeout
        self._local = threading.local()
        self._fingerprint = None
        self._lock = threading.Lock()
        self._observers = []
        self.last_timings = {}

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
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            try:
                self._login(session)
            except Exception:
                session.close()
                raise
            self._local.session = session
        return session

    def _login(self, session):
        for attempt in range(3):
            try:
                response = session.post(self.base_url + "/api/login", json={"password": self.key_path.read_text().strip()}, timeout=_bounded_timeout(15), allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError):
                _bounded_timeout(1)
                if attempt == 2:
                    raise WeMMError('WeMM is unreachable; retain progress and resume when the worker is ready', code='wemm_unavailable') from None
                _backoff(attempt)
                continue
            _bounded_timeout(1)
            if response.status_code != 200:
                raise WeMMError(f"WeMM authentication failed: HTTP {response.status_code}")
            return

    def _request(self, path, payload=None):
        started = time.monotonic()
        metric = {"call_id": uuid.uuid4().hex, "call_type": "embedding", "operation": path,
                  "started_at_epoch": time.time(), "status": "running",
                  "input_type": "image" if payload and "image_path" in payload else "text" if payload else "status"}
        try:
            result = self._request_impl(path, payload)
            metric.update(status="success", backend_timings=result.get("timings", {}),
                          fingerprint=result.get("fingerprint"))
            return result
        except Exception as exc:
            metric.update(status="error", error_code=getattr(exc, "code", type(exc).__name__))
            raise
        finally:
            metric.update(finished_at_epoch=time.time(), duration_ms=(time.monotonic()-started)*1000)
            self.call_records.append(metric)
            with self._lock:
                observers = list(self._observers)
            for callback in observers:
                callback(copy.deepcopy(metric))

    def _request_impl(self, path, payload=None):
        session = self._session()
        for attempt in range(3):
            try:
                timeout = _bounded_timeout(self.timeout)
                response = (session.get(self.base_url + path, timeout=timeout, allow_redirects=False) if payload is None
                            else session.post(self.base_url + path, json=payload, timeout=timeout, allow_redirects=False))
            except (requests.Timeout, requests.ConnectionError) as exc:
                _bounded_timeout(1)
                if attempt == 2:
                    raise WeMMError(f"WeMM transport failure: {type(exc).__name__}") from None
                _backoff(attempt)
                continue
            _bounded_timeout(1)
            if response.status_code == 401 and attempt < 2:
                self._login(session)
                continue
            if response.status_code in {429, 502, 503, 504} and attempt < 2:
                _backoff(attempt)
                continue
            if response.status_code != 200:
                try:
                    detail = response.json().get("detail")
                except Exception:
                    detail = "invalid server response"
                code = 'query_too_long' if isinstance(detail, dict) and detail.get('code') == 'text_too_long' else 'wemm_error'
                raise WeMMError(f"WeMM HTTP {response.status_code}: {code}", code=code)
            try:
                result = response.json()
            except ValueError:
                raise WeMMError("WeMM returned invalid JSON", code="invalid_response") from None
            if not isinstance(result, dict):
                raise WeMMError("WeMM returned an invalid response object", code="invalid_response")
            fingerprint = result.get("fingerprint")
            if (path != "/api/status" or fingerprint is not None) and (
                    not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
                raise WeMMError("WeMM response has no valid fingerprint", code="invalid_response")
            if fingerprint:
                with self._lock:
                    if self._fingerprint and self._fingerprint != fingerprint:
                        raise WeMMError("WeMM fingerprint changed; create a new index version", code="fingerprint_mismatch")
                    self._fingerprint = fingerprint
            return result
        raise WeMMError("WeMM request attempts exhausted")

    @property
    def fingerprint(self):
        if self._fingerprint is None:
            status = self._request("/api/status")
            if status.get("state") not in {"ready", "busy"} or not status.get("fingerprint"):
                raise WeMMError("WeMM worker not ready")
        return self._fingerprint

    def encode(self, text=None, image_path=None, *, role="document"):
        if role not in {"document", "query"}:
            raise ValueError("Embedding role must be document or query")
        if (text is None) == (image_path is None):
            raise ValueError("Supply text or image_path exclusively")
        payload = {"text": text} if text is not None else {"image_path": str(Path(image_path).resolve())}
        result = self._request("/api/internal/embeddings", payload)
        try:
            vector = np.asarray(result["embedding"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            raise WeMMError("Invalid WeMM embedding payload", code="invalid_response") from None
        if result.get("dimension") != 4096 or vector.shape != (4096,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4):
            raise WeMMError("Invalid WeMM embedding shape/normalization")
        self.last_timings = result.get("timings", {})
        if not isinstance(self.last_timings, dict):
            raise WeMMError("Invalid WeMM timing metadata", code="invalid_response")
        return vector

    def split_text(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        result = self._request("/api/internal/tokenize", {"text": text})
        chunks = result.get("chunks")
        if (result.get("limit") != 512 or not isinstance(chunks, list)
                or any(not isinstance(chunk, str) or not chunk for chunk in chunks)
                or "".join(chunks) != text):
            raise WeMMError("Tokenizer split lost input text")
        return chunks

    def embed_many(self, texts, *, role="document"):
        return [self.encode(text=text, role=role).tolist() for text in texts]
