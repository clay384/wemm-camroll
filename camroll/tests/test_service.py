"""HTTP boundary tests with injected agents and a real Qwen adapter/fake SDK."""
from contextvars import Context
from dataclasses import asdict
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from camroll_agent.agent import Agent, AgentResult
from camroll_agent.budget import BudgetLedger
from camroll_agent.qwen import QwenClient
from camroll_agent.service import create_app


def _result():
    return AgentResult(final_text="在公园 [img_1]", tool_trace=[], steps=2,
        view_image_calls=0, stopped_reason="ok", latency_s=0.1,
        evidence_image_ids=["img_1"], index_version="frozen-1", config="M3",
        usage={"requests": 2, "estimated_cny": 0.001})


class StubAgent:
    def __init__(self):
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        return _result()

    def ask_streaming(self, question):
        self.questions.append(question)
        yield "answer", {"response": _result().final_text, **asdict(_result())}


@pytest.mark.parametrize("override", [
    {"album_id": "../outside"}, {"album_id": "a/../outside"}, {"album_id": "/absolute"},
    {"album_id": "."}, {"album_id": ".."}, {"album_id": "a\\outside"},
    {"album_id": "a%2foutside"}, {"album_id": ""}, {"album_id": "a"*81},
    {"config": "M5"}, {"config": "m1"}, {"config": ""}, {"config": "M1\n"},
    {"question": ""}, {"question": " \t\n"}, {"question": "a"*20001},
])
def test_invalid_input_rejected_before_agent_creation(tmp_path, override):
    created = []
    client = TestClient(create_app(tmp_path, agent_factory=lambda request: created.append(request)))
    response = client.post("/api/ask", json={"album_id": "album_1", "question": "哪里？", **override})
    assert response.status_code == 422
    assert created == []


def test_sync_response_keeps_required_metadata_and_usage(tmp_path):
    agent = StubAgent()
    requests = []
    def factory(request):
        requests.append(request)
        return agent
    client = TestClient(create_app(tmp_path, agent_factory=factory))
    response = client.post("/api/ask", json={"album_id": "album-1", "question": "哪里？", "config": "M3"})
    assert response.status_code == 200
    value = response.json()
    assert value["answer"] == value["final_text"] == "在公园 [img_1]"
    assert value["album_id"] == "album-1"
    assert value["evidence_image_ids"] == ["img_1"]
    assert value["index_version"] == "frozen-1"
    assert value["config"] == "M3"
    assert value["stopped_reason"] == "ok"
    assert value["usage"]["estimated_cny"] == 0.001
    assert all(key in value for key in ("steps", "view_image_calls", "tool_trace", "latency_s"))
    assert requests[0].config == "M3"
    assert agent.questions == ["哪里？"]


def test_missing_album_returns_404_before_cloud_client_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/ask", json={"album_id": "missing", "question": "哪里？"})
    assert response.status_code == 404
    assert "published index" in response.json()["detail"]


def test_album_symlink_cannot_escape_configured_root(tmp_path, monkeypatch):
    albums, outside = tmp_path / "albums", tmp_path / "private"
    albums.mkdir()
    outside.mkdir()
    (outside / "CURRENT").write_text("fake")
    (albums / "escaped").symlink_to(outside, target_is_directory=True)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    client = TestClient(create_app(albums))
    response = client.post("/api/ask", json={"album_id": "escaped", "question": "哪里？"})
    assert response.status_code == 404
    assert str(outside) not in response.text


class FrozenLibrary:
    version = "frozen-http-1"
    def __init__(self):
        self.record = {"id": "img_1", "image_id": "img_1", "kind": "image", "caption": "公园里的树",
                       "path": "/private/photo.jpg", "source_path": "/private/original.jpg"}
    def search(self, **kwargs):
        return {"results": [self.record], "count": 1}
    def get(self, id):
        return self.record if id == "img_1" else {"error": "unknown ID"}


def _api_reply(content=None, arguments=None):
    message = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [{"id": "call1", "type": "function", "function": {
            "name": "search", "arguments": arguments}}]
    return {"model": "qwen3.7-flash", "choices": [{"finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}


def test_sse_real_agent_preserves_qwen_scope_across_starlette_worker_contexts(tmp_path):
    class HTTPError(Exception):
        status_code = 429
    outcomes = iter([HTTPError(), _api_reply(arguments="null"),
                     _api_reply(arguments='{"query":"公园"}'), _api_reply(content="在公园 [img_1]")])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        item = next(outcomes)
        if isinstance(item, Exception):
            raise item
        return item
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    qwen = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite"), sleep=lambda _: None)
    agent = Agent("unused", llm=qwen, library=FrozenLibrary())
    client = TestClient(create_app(tmp_path, agent_factory=lambda _: agent))
    with client.stream("POST", "/api/ask", json={"album_id": "photos", "question": "哪里？", "stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = list(response.iter_lines())
    events = []
    kind = None
    for line in lines:
        if line.startswith("event: "):
            kind = line[7:]
        elif line.startswith("data: "):
            events.append((kind, json.loads(line[6:])))
    assert events[-1][0] == "answer"
    final = events[-1][1]
    assert final["stopped_reason"] == "ok"
    assert len(calls) == final["usage"]["requests"] == 4
    assert final["usage"]["max_requests"] == 50
    assert final["index_version"] == "frozen-http-1"
    assert final["evidence_image_ids"] == ["img_1"]
    assert final["response"] == final["final_text"]
    assert all(kind != "error" for kind, _ in events)
    assert "/private" not in "\n".join(lines)


def test_stream_exception_does_not_disclose_message_or_secrets(tmp_path):
    class BrokenAgent:
        def ask_streaming(self, question):
            yield "status", {"message": "started"}
            raise RuntimeError("sk-SECRET /private/path confidential")
    client = TestClient(create_app(tmp_path, agent_factory=lambda _: BrokenAgent()))
    response = client.post("/api/ask", json={"album_id": "photos", "question": "哪里？", "stream": True})
    assert "event: error" in response.text
    assert "RuntimeError" in response.text
    assert "SECRET" not in response.text
    assert "/private" not in response.text


def test_sync_exception_does_not_disclose_message_or_secrets(tmp_path):
    class BrokenAgent:
        def ask(self, question):
            raise RuntimeError("sk-SECRET /private/path confidential")
    client = TestClient(create_app(tmp_path, agent_factory=lambda _: BrokenAgent()), raise_server_exceptions=False)
    response = client.post("/api/ask", json={"album_id": "photos", "question": "哪里？"})
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert "/private" not in response.text


def test_stream_disconnect_closes_agent_iterator(tmp_path):
    """Drive ASGI directly: TestClient buffers streaming bodies until completion."""
    import asyncio
    import time
    closed = []
    class DisconnectAgent:
        def ask_streaming(self, question):
            try:
                for i in range(10):
                    yield "status", {"step": i}
                    time.sleep(0.005)
            finally:
                closed.append(True)
    app = create_app(tmp_path, agent_factory=lambda _: DisconnectAgent())
    async def drive():
        received_body = False
        first_chunk = asyncio.Event()
        async def receive():
            nonlocal received_body
            if not received_body:
                received_body = True
                return {"type": "http.request", "body": json.dumps({"album_id": "photos", "question": "哪里？", "stream": True}).encode(), "more_body": False}
            await first_chunk.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk.set()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/api/ask", "raw_path": b"/api/ask",
                 "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000)}
        await asyncio.wait_for(app(scope, receive, send), timeout=2)
    asyncio.run(drive())
    gc.collect()
    assert closed == [True]


def test_publish_during_library_load_and_rollback_use_actual_cached_version(tmp_path, monkeypatch):
    from camroll_agent import agent as agent_module, library as library_module, qwen as qwen_module, wemm as wemm_module
    directory = tmp_path / 'photos'
    directory.mkdir()
    pointer = directory / 'CURRENT'
    pointer.write_text('version-1')
    loads = []
    class ChangingLibrary:
        def __init__(self, memory_dir, embedding, config):
            # Simulate atomic publication after service.read(CURRENT), before
            # Library's own read. The first response pins the loaded v2.
            if not loads:
                pointer.write_text('version-2')
            self.version = pointer.read_text()
            self.config = config
            loads.append(self.version)
    class VersionAgent:
        def __init__(self, memory_dir, *, llm, vlm, library, config):
            self.library, self.config = library, config
        def ask(self, question):
            value = _result()
            value.index_version, value.config = self.library.version, self.config
            return value
    monkeypatch.setattr(library_module, 'Library', ChangingLibrary)
    monkeypatch.setattr(agent_module, 'Agent', VersionAgent)
    monkeypatch.setattr(qwen_module, 'QwenClient', lambda: object())
    monkeypatch.setattr(wemm_module, 'EmbeddingClient', lambda: object())
    client = TestClient(create_app(tmp_path))
    payload = {'album_id': 'photos', 'question': '哪里？'}
    first = client.post('/api/ask', json=payload)
    assert first.status_code == 200
    assert first.json()['index_version'] == 'version-2'
    # Roll back before any second request. A cache incorrectly keyed v1->v2
    # would return v2 here instead of constructing the requested v1 snapshot.
    pointer.write_text('version-1')
    rolled_back = client.post('/api/ask', json=payload)
    assert rolled_back.json()['index_version'] == 'version-1'
    assert loads == ['version-2', 'version-1']
    repeated = client.post('/api/ask', json=payload)
    assert repeated.json()['index_version'] == 'version-1'
    assert loads == ['version-2', 'version-1']
