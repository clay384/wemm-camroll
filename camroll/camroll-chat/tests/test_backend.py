from __future__ import annotations

import json
import hashlib
import io
import asyncio
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from camroll_chat.app import ChatRequest, create_app
from camroll_chat.config import ChatConfig, DEFAULT_ALBUM_ID, DEFAULT_INDEX_VERSION


ORIGIN = {"Origin": "http://testserver"}


def make_test_picture_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1200, 600), "#3344aa").save(stream, "JPEG")
    return stream.getvalue()


PICTURE_BYTES = make_test_picture_bytes()
ASSET_ID = hashlib.sha256(PICTURE_BYTES).hexdigest()
IMAGE_ID = "img_" + ASSET_ID


def fixture(tmp_path: Path, *, current: str = DEFAULT_INDEX_VERSION):
    album = tmp_path / DEFAULT_ALBUM_ID
    asset = album / "assets" / ASSET_ID
    snapshot = album / "snapshots" / DEFAULT_INDEX_VERSION
    asset.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (album / "CURRENT").write_text(current)
    picture = asset / "original.jpg"
    picture.write_bytes(PICTURE_BYTES)
    rows = [{
        "id": IMAGE_ID,
        "image_id": IMAGE_ID,
        "asset_id": ASSET_ID,
        "original_path": str(picture),
    }]
    for index in range(1, 355):
        digest = f"{index:064x}"
        item = album / "assets" / digest
        item.mkdir(parents=True)
        path = item / "original.jpg"
        path.write_bytes(picture.read_bytes())
        rows.append({"id": "img_" + digest, "asset_id": digest, "original_path": str(path)})
    (snapshot / "records.json").write_text(json.dumps({"images": rows}))
    (snapshot / "search.sqlite").write_bytes(b"test-search-snapshot")
    (snapshot / "vectors.npz").write_bytes(b"test-vector-snapshot")
    snapshot_hashes = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in ("records.json", "search.sqlite", "vectors.npz")
    }
    config = ChatConfig(
        album_root=tmp_path,
        static_dir=tmp_path / "static",
        allowed_hosts=("testserver",),
        snapshot_hashes=snapshot_hashes,
    )
    return config


def upstream(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, json={
            "status": "ready", "albums": [DEFAULT_ALBUM_ID], "configs": ["M1", "M2", "M3", "M4"],
        })
    body = json.loads(request.content)
    assert body == {
        "album_id": DEFAULT_ALBUM_ID,
        "question": "2019年在哪里？",
        "config": "M4",
        "stream": True,
        "index_version": DEFAULT_INDEX_VERSION,
        "structured_output": True,
        "stream_model": True,
    }
    fake_secret = "sk-" + "test_redaction_12345678"
    stream = (
        'event: status\ndata: {"message":"started","index_version":"' + DEFAULT_INDEX_VERSION + '"}\n\n'
        'event: tool_call\ndata: {"step":1,"tool":"search","args":{"query":"2019"}}\n\n'
        'event: tool_result\ndata: {"step":1,"tool":"search","observation":"/home/private ' + fake_secret + '","cached":false}\n\n'
        'event: answer\ndata: ' + json.dumps({
            "response": "在秦皇岛。", "final_text": "在秦皇岛。", "index_version": DEFAULT_INDEX_VERSION,
            "config": "M4", "selected_image_ids": [IMAGE_ID], "retrieved_image_ids": [IMAGE_ID],
            "turn_outcome": "answer", "stopped_reason": "ok",
            "usage": {"prompt_tokens": 123, "completion_tokens": 45, "estimated_cny": 0.02},
            "model_call_metrics": [{
                "role": "tool_loop", "model": "qwen3.7-flash", "status": "success",
                "prompt_tokens": 123, "completion_tokens": 45, "duration_ms": 321,
                "input": {"messages": [{"role": "system", "content": "secret system prompt"}]},
                "output": {"raw": "secret raw output"},
            }],
            "tool_trace": [{
                "step": 1, "tool": "search", "status": "success", "result": {
                    "results": [{
                        "id": "ev_1234", "name": "秦皇岛合影", "date_start": "2019-07-22",
                        "date_end": "2019-07-22", "location": "秦皇岛", "people": ["明明", "乐乐"],
                        "description_preview": "在主题沙雕前合影", "image_ids": [IMAGE_ID],
                        "face_instances": [{"name": "明明", "image_ids": [IMAGE_ID]}],
                    }]
                },
            }],
            "conversation_history": [{"role": "system", "content": "secret"}],
        }, ensure_ascii=False) + '\n\n'
    )
    return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})


def test_health_album_media_and_security_headers(tmp_path):
    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(upstream)))
    health = client.get("/api/health")
    assert health.json()["status"] == "ready"
    assert health.json()["album"]["photo_count"] == 355
    assert "album_root" not in health.text
    assert health.headers["content-security-policy"].startswith("default-src 'self'")
    assert client.get("/api/albums").json()["default_album_id"] == DEFAULT_ALBUM_ID
    thumb = client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/{IMAGE_ID}")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert Image.open(__import__("io").BytesIO(thumb.content)).size == (960, 480)
    assert client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/not-an-id").status_code == 404
    assert client.get(f"/api/albums/other/media/{IMAGE_ID}").status_code == 404


def test_chat_forces_policy_normalizes_stream_and_redacts(tmp_path):
    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(upstream)))
    with client.stream("POST", "/api/chat/stream", headers=ORIGIN,
                       json={"album_id": DEFAULT_ALBUM_ID, "question": "2019年在哪里？"}) as response:
        text = "\n".join(response.iter_lines())
    assert response.status_code == 200
    assert "event: request_started" in text
    assert "event: model_status" in text
    assert "event: tool_call" in text and "event: tool_result" in text
    assert "event: answer" in text and "在秦皇岛。" in text
    assert "conversation_history" not in text
    assert "secret system prompt" not in text and "secret raw output" not in text
    assert '"prompt_tokens":123' in text and '"completion_tokens":45' in text
    assert '"evidence_metadata"' in text and "2019-07-22" in text and "主题沙雕前合影" in text
    assert "/home/private" not in text and "test_redaction_12345678" not in text


def test_post_requires_same_origin_and_rejects_policy_override(tmp_path):
    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(upstream)))
    body = {"album_id": DEFAULT_ALBUM_ID, "question": "2019年在哪里？"}
    assert client.post("/api/chat/stream", json=body).status_code == 403
    override = client.post("/api/chat/stream", headers=ORIGIN, json={**body, "config": "M1"})
    assert override.status_code == 422


def test_index_change_blocks_chat_and_media(tmp_path):
    config = fixture(tmp_path, current="changed")
    client = TestClient(create_app(config, transport=httpx.MockTransport(upstream)))
    response = client.post("/api/chat/stream", headers=ORIGIN,
                           json={"album_id": DEFAULT_ALBUM_ID, "question": "问题"})
    assert response.status_code == 409
    assert client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/{IMAGE_ID}").status_code == 404
    assert client.get("/api/health").json()["status"] == "degraded"


def test_media_never_serves_record_path_outside_assets(tmp_path):
    config = fixture(tmp_path)
    records = (tmp_path / DEFAULT_ALBUM_ID / "snapshots" / DEFAULT_INDEX_VERSION / "records.json")
    value = json.loads(records.read_text())
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"private")
    value["images"][0]["original_path"] = str(secret)
    records.write_text(json.dumps(value))
    client = TestClient(create_app(config, transport=httpx.MockTransport(upstream)))
    assert client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/{IMAGE_ID}").status_code == 404


def test_upstream_failure_is_not_retried(tmp_path):
    calls = []

    def unavailable(request: httpx.Request):
        calls.append(request)
        return httpx.Response(503, text="secret provider exception /home/provider")

    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(unavailable)))
    response = client.post(
        "/api/chat/stream",
        headers=ORIGIN,
        json={"album_id": DEFAULT_ALBUM_ID, "question": "问题"},
    )
    assert response.status_code == 200
    assert "event: error" in response.text and "upstream_unavailable" in response.text
    assert "secret provider" not in response.text and "/home/provider" not in response.text
    assert len(calls) == 1


def test_host_header_is_allowlisted_before_same_origin(tmp_path):
    calls = []

    def counting(request: httpx.Request):
        calls.append(request)
        return upstream(request)

    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(counting)))
    response = client.post(
        "/api/chat/stream",
        headers={"Host": "evil.example", "Origin": "http://evil.example"},
        json={"album_id": DEFAULT_ALBUM_ID, "question": "问题"},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "host_not_allowed"}
    assert calls == []


def test_snapshot_hash_change_degrades_and_blocks_chat_and_media(tmp_path):
    config = fixture(tmp_path)
    snapshot = tmp_path / DEFAULT_ALBUM_ID / "snapshots" / DEFAULT_INDEX_VERSION
    (snapshot / "vectors.npz").write_bytes(b"changed")
    client = TestClient(create_app(config, transport=httpx.MockTransport(upstream)))
    assert client.get("/api/health").json()["status"] == "degraded"
    assert client.post(
        "/api/chat/stream",
        headers=ORIGIN,
        json={"album_id": DEFAULT_ALBUM_ID, "question": "问题"},
    ).status_code == 409
    assert client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/{IMAGE_ID}").status_code == 404


def test_changed_media_content_is_never_served(tmp_path):
    config = fixture(tmp_path)
    picture = tmp_path / DEFAULT_ALBUM_ID / "assets" / ASSET_ID / "original.jpg"
    picture.write_bytes(b"changed private media")
    client = TestClient(create_app(config, transport=httpx.MockTransport(upstream)))
    assert client.get(f"/api/albums/{DEFAULT_ALBUM_ID}/media/{IMAGE_ID}").status_code == 404


def test_runtime_model_policy_mismatch_never_emits_answer(tmp_path):
    def wrong_model(request: httpx.Request) -> httpx.Response:
        response = upstream(request)
        if request.method == "POST":
            content = response.text.replace('"model": "qwen3.7-flash"', '"model": "other-model"')
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        return response

    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(wrong_model)))
    response = client.post(
        "/api/chat/stream", headers=ORIGIN,
        json={"album_id": DEFAULT_ALBUM_ID, "question": "2019年在哪里？"},
    )
    assert response.status_code == 200
    assert "model_policy_mismatch" in response.text
    assert "event: answer" not in response.text


def test_preflight_error_without_model_call_keeps_real_terminal_state(tmp_path):
    def budget_exhausted(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return upstream(request)
        answer = {
            "response": "本轮费用预算已用尽。", "index_version": DEFAULT_INDEX_VERSION,
            "config": "M4", "turn_outcome": "error", "stopped_reason": "budget_exhausted",
            "model_call_metrics": [], "usage": {"requests": 0, "estimated_cny": 0},
        }
        return httpx.Response(
            200,
            text="event: answer\ndata: " + json.dumps(answer, ensure_ascii=False) + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = TestClient(create_app(fixture(tmp_path), transport=httpx.MockTransport(budget_exhausted)))
    response = client.post(
        "/api/chat/stream", headers=ORIGIN,
        json={"album_id": DEFAULT_ALBUM_ID, "question": "问题"},
    )
    assert "event: answer" in response.text and "budget_exhausted" in response.text
    assert "model_policy_mismatch" not in response.text


def test_production_config_ignores_environment_overrides(monkeypatch):
    monkeypatch.setenv("CAMROLL_CHAT_UPSTREAM_URL", "http://attacker.invalid")
    monkeypatch.setenv("QWEN_API_KEY", "must-not-be-read")
    config = ChatConfig.from_env()
    assert config.upstream_url == "http://127.0.0.1:8772/camroll"
    assert "attacker" not in repr(config)


def test_inflight_request_keeps_global_lock_without_stream_consumer(tmp_path):
    """A disconnected browser must not release the billed upstream execution."""
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class DelayedAnswer(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await release.wait()
                answer = {
                    "response": "完成", "index_version": DEFAULT_INDEX_VERSION, "config": "M4",
                    "turn_outcome": "answer", "stopped_reason": "ok",
                    "model_call_metrics": [{"role": "agent", "model": "qwen3.7-flash"}],
                }
                yield ("event: answer\ndata: " + json.dumps(answer, ensure_ascii=False) + "\n\n").encode()

            async def aclose(self):
                return None

        def delayed_upstream(request: httpx.Request):
            if request.method == "GET":
                return upstream(request)
            return httpx.Response(
                200, stream=DelayedAnswer(), headers={"content-type": "text/event-stream"}
            )

        app = create_app(fixture(tmp_path), transport=httpx.MockTransport(delayed_upstream))
        endpoint = next(
            route.endpoint for route in app.routes
            if getattr(route, "path", None) == "/api/chat/stream"
        )
        response = await endpoint(ChatRequest(album_id=DEFAULT_ALBUM_ID, question="问题一"))
        assert response.media_type == "text/event-stream"
        await asyncio.wait_for(started.wait(), timeout=1)
        assert app.state.execution_lock.locked()
        with pytest.raises(HTTPException) as caught:
            await endpoint(ChatRequest(album_id=DEFAULT_ALBUM_ID, question="问题二"))
        assert caught.value.status_code == 429
        release.set()
        for _ in range(100):
            if not app.state.background_tasks:
                break
            await asyncio.sleep(0.01)
        assert not app.state.background_tasks
        assert not app.state.execution_lock.locked()

    asyncio.run(scenario())
