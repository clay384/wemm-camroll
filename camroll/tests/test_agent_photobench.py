"""Multi-turn and delivery invariants, using synthetic records only."""
import json
from dataclasses import asdict
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from camroll_agent.agent import Agent, AgentResult
from camroll_agent.service import create_app
from camroll_agent.qwen import QwenClient
from camroll_agent.wemm import EmbeddingClient, WeMMError
from test_agent import make, response, tool


def final(answer, outcome="answer", ids=None):
    return response(content=json.dumps({"answer": answer, "outcome": outcome,
                                      "selected_image_ids": ids or []}, ensure_ascii=False))


def test_clarification_survives_without_tools_and_next_turn_has_context():
    agent, client, _ = make([final("你说的是哪一年？", "clarify"), response(tool()),
                              final("在公园", ids=["img_1"])])
    first = agent.ask("那次在哪里？", structured_output=True)
    assert first.final_text == "你说的是哪一年？"
    assert first.turn_outcome == "clarify" and first.stopped_reason == "ok"
    assert first.selected_image_ids == []
    second = agent.ask("2026年。", history=first.conversation_history, structured_output=True)
    assert second.selected_image_ids == ["img_1"]
    assert set(second.retrieved_image_ids) == {"img_1", "img_2"}
    text = json.dumps(client.messages[1], ensure_ascii=False)
    assert "那次在哪里" in text and "你说的是哪一年" in text and "2026年" in text


def test_refusal_does_not_need_a_search_or_lose_its_reason():
    agent, _, library = make([final("无法根据外貌确认这个陌生人的身份。", "refuse")])
    result = agent.ask("这个陌生人叫什么？", structured_output=True)
    assert result.turn_outcome == "refuse"
    assert "陌生人" in result.final_text
    assert result.selected_image_ids == [] and not library.calls


@pytest.mark.parametrize("outcome", ["clarify", "refuse"])
def test_final_review_does_not_override_terminal_behavior(outcome):
    agent, client, _ = make([response(tool()), final("请补充时间" if outcome == "clarify" else "无法确认", outcome)], config="M2")
    result = agent.ask("那一次呢", structured_output=True)
    assert result.turn_outcome == outcome and result.selected_image_ids == []
    assert not client.generations


def test_candidate_and_explicit_delivery_are_not_conflated():
    agent, _, _ = make([response(tool()), final("公园", ids=["img_2"])])
    result = agent.ask("公园照片", structured_output=True)
    assert result.selected_image_ids == ["img_2"]
    assert result.retrieved_image_ids == ["img_1", "img_2"]
    assert result.evidence_image_ids == ["img_2"]


def test_unknown_delivery_is_repaired_and_never_counted():
    agent, _, _ = make([response(tool()), final("公园", ids=["img_UNKNOWN"]),
                        final("公园", ids=["img_1"])])
    result = agent.ask("公园照片", structured_output=True)
    assert result.selected_image_ids == ["img_1"] and result.stopped_reason == "ok"


def test_clarification_repair_explains_conflicting_delivery():
    agent, client, _ = make([response(tool()), final("哪一年？ [img_1]", "clarify", ["img_1"]),
                             final("哪一年？", "clarify")])
    result = agent.ask("找那张", structured_output=True)
    assert result.turn_outcome == "clarify" and result.selected_image_ids == []
    repair = json.dumps(client.messages[-1], ensure_ascii=False)
    assert "Clarification/refusal" in repair and "selected_image_ids必须是[]" in repair


def test_prior_turn_tool_evidence_can_be_used_but_other_session_cannot():
    agent, _, _ = make([response(tool()), final("这里有公园。", ids=["img_1"]),
                        final("这一张", ids=["img_1"])])
    first = agent.ask("找公园", structured_output=True)
    second = agent.ask("给我这一张", history=first.conversation_history, structured_output=True)
    assert second.selected_image_ids == ["img_1"]
    isolated, _, _ = make([final("无法确认", "refuse")])
    assert isolated.ask("给我这一张", structured_output=True).retrieved_image_ids == []


def test_real_ttft_ignores_role_only_and_usage_chunks(monkeypatch):
    clock = [5.0]
    monkeypatch.setattr("camroll_agent.qwen.time.monotonic", lambda: clock[0])
    def chunks():
        clock[0] = 5.1
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        clock[0] = 5.4
        yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call1",
                   "function": {"name": "search", "arguments": "{}"}}]}}]}
        clock[0] = 5.8
        yield {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": []}
    result = QwenClient._collect_stream(chunks(), started=5.0)
    assert result["_ttft_ms"] == pytest.approx(400)
    assert result["usage"]["completion_tokens"] == 2


class StatefulStub:
    library = SimpleNamespace(version="v1")
    def __init__(self):
        self.calls = []
    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        history = kwargs.get("history", []) + [{"role": "user", "content": question},
                                              {"role": "assistant", "content": "回答"}]
        return AgentResult("回答", [], 1, 0, "ok", .01, index_version=self.library.version,
                           conversation_history=history)


def test_http_idempotency_context_and_version_pin(tmp_path):
    agent = StatefulStub()
    client = TestClient(create_app(tmp_path, agent_factory=lambda _: agent))
    body = {"album_id": "album1", "question": "第一问", "conversation_id": "qa1",
            "request_id": "run1.qa1.1", "index_version": "v1"}
    first = client.post("/api/ask", json=body)
    assert first.status_code == 200
    assert client.post("/api/ask", json=body).json() == first.json()
    assert len(agent.calls) == 1
    assert client.post("/api/ask", json={**body, "question": "偷换问题"}).status_code == 409
    assert client.get("/api/requests/run1.qa1.1").json()["status"] == "complete"
    assert client.post("/api/ask", json={**body, "question": "第二问", "request_id": "run1.qa1.2"}).status_code == 200
    assert "第一问" in json.dumps(agent.calls[-1][1]["history"], ensure_ascii=False)
    agent.library = SimpleNamespace(version="v2")
    assert client.post("/api/ask", json={**body, "request_id": "run1.qa1.3"}).status_code == 409
    assert len(agent.calls) == 2


def test_embedding_observer_keeps_each_call_without_vector_or_path(monkeypatch):
    client = EmbeddingClient()
    results = iter([{"embedding": [1.0], "timings": {"queue_ms": 8, "encode_ms": 4}, "fingerprint": "a"*64},
                    {"embedding": [2.0], "timings": {"queue_ms": 1, "encode_ms": 7}, "fingerprint": "a"*64}])
    monkeypatch.setattr(client, "_request_impl", lambda *args: next(results))
    records = []
    with client.capture_calls(records.append):
        client._request("/api/internal/embeddings", {"image_path": "/private/photo.jpg"})
        client._request("/api/internal/embeddings", {"text": "private query"})
    assert [r["backend_timings"]["queue_ms"] for r in records] == [8, 1]
    assert len(client.call_records) == 2
    assert "/private" not in json.dumps(records) and "private query" not in json.dumps(records)
    assert all("embedding" not in r and r["duration_ms"] >= 0 for r in records)
