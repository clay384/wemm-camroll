import sys
from pathlib import Path

import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "camroll-agent"))
from camroll_agent import qwen_retrieval
from camroll_agent.qwen_retrieval import QwenRetrievalClient, QwenRetrievalError


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, url, **kwargs):
        self.requests.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    get = request
    post = request


def client_with(*responses, **kwargs):
    client = QwenRetrievalClient(cache_namespace="Q37_test", **kwargs)
    session = Session(responses)
    client._local.session = session
    return client, session


def vector_response(fingerprint="a" * 64):
    vector = np.zeros(2048)
    vector[0] = 1
    return {"dimension": 2048, "embeddings": [vector.tolist()], "fingerprint": fingerprint,
            "telemetry": {"inference_ms": 12}}


def test_encode_query_role_namespace_and_observer():
    client, session = client_with(Response(vector_response()))
    records = []
    with client.capture_calls(records.append):
        vector = client.encode(text="测试", role="query")
    assert vector.shape == (2048,)
    assert vector.dtype == np.float32
    payload = session.requests[0][1]["json"]
    assert payload["role"] == "query"
    assert payload["cache_namespace"] == "Q37_test"
    assert client.last_timings["inference_ms"] == 12
    assert records[0]["status"] == "success"
    assert records[0]["role"] == "query"


def test_fingerprint_cannot_change_in_one_client():
    client, session = client_with(Response(vector_response()), Response(vector_response("b" * 64)))
    client.encode(text="document")
    with pytest.raises(QwenRetrievalError) as error:
        client.encode(text="query", role="query")
    assert error.value.code == "fingerprint_mismatch"


@pytest.mark.parametrize("mutation", ["dimension", "nan", "zero", "shape"])
def test_invalid_vectors_fail(mutation):
    response = vector_response()
    if mutation == "dimension":
        response["dimension"] = 4096
    elif mutation == "nan":
        response["embeddings"][0][0] = float("nan")
    elif mutation == "zero":
        response["embeddings"][0][0] = 0
    else:
        response["embeddings"].append(response["embeddings"][0])
    client, session = client_with(Response(response))
    with pytest.raises(QwenRetrievalError):
        client.encode(text="test")


@pytest.mark.parametrize("chunks,counts", [(["a"], [1]), (["abc"], [2049]), (["ab", "c"], [2])])
def test_split_rejects_loss_or_token_overflow(chunks, counts):
    client, session = client_with(Response({"chunks": chunks, "token_counts": counts, "limit": 2048, "fingerprint": "a" * 64}))
    with pytest.raises(QwenRetrievalError):
        client.split_text("abc")


def test_rerank_preserves_input_identity_and_compact_interface():
    scores = [{"id": "p", "score": 0.7, "chunk_count": 2}, {"id": "e", "score": 0.2, "chunk_count": 1}]
    client, session = client_with(Response({"scores": scores, "fingerprint": "b" * 64}))
    returned = client.rerank("query", [{"id": "p", "text": "caption", "image_path": "/safe/image.jpg"}, {"id": "e", "text": "event"}])
    assert returned == scores
    assert session.requests[0][1]["json"]["candidates"] == [
        {"id": "p", "type": "image", "image_path": "/safe/image.jpg", "caption": "caption"},
        {"id": "e", "type": "text", "text": "event"},
    ]
    assert client.call_records[0]["call_type"] == "rerank"
    assert client.reranker_fingerprint == "b" * 64


@pytest.mark.parametrize("scores", [[{"id": "other", "score": 0.5, "chunk_count": 1}],
                                   [{"id": "p", "score": float("nan"), "chunk_count": 1}], []])
def test_rerank_rejects_missing_or_corrupt_scores(scores):
    client, session = client_with(Response({"scores": scores, "fingerprint": "b" * 64}))
    with pytest.raises(QwenRetrievalError):
        client.rerank("query", [{"id": "p", "text": "text"}])


def test_transport_failure_has_no_hidden_retry():
    client, session = client_with(requests.Timeout())
    with pytest.raises(QwenRetrievalError) as error:
        client.encode(text="query", role="query")
    assert error.value.code == "qwen_unavailable"
    assert len(session.requests) == 1
    assert client.call_records[0]["status"] == "error"


def test_deadline_checked_before_network(monkeypatch):
    client, session = client_with()

    def expired(timeout):
        raise TimeoutError("expired")

    monkeypatch.setattr(qwen_retrieval, "_bounded_timeout", expired)
    with pytest.raises(TimeoutError):
        client.encode(text="query", role="query")
    assert not session.requests


def test_worker_url_and_namespace_boundaries():
    for url in ("http://example.org:18778", "http://127.0.0.1/private", "https://127.0.0.1:18778"):
        with pytest.raises(ValueError):
            QwenRetrievalClient(url, cache_namespace="test")
    with pytest.raises(ValueError):
        QwenRetrievalClient(cache_namespace="../other-group")
