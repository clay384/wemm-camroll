"""Independent LAN UI backend for one frozen CamRoll PhotoBench album."""
from __future__ import annotations

import asyncio
import io
import json
import math
from pathlib import Path
import re
import threading
import uuid
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .album import (
    AlbumUnavailable,
    FrozenAlbum,
    IndexVersionChanged,
    SnapshotIntegrityChanged,
)
from .config import ChatConfig


SERVICE_VERSION = "0.1.0"
_PATH_RE = re.compile(r"(?<![\w.])/(?:home|Users|private|var|tmp|etc|opt|root|srv|mnt)/[^\s\"'<>]+")
_SECRET_RE = re.compile(r"\b(?:sk|ak|ark)[-_][A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_PRIVATE_KEYS = {
    "access_key", "api_key", "authorization", "base_url", "conversation_history",
    "headers", "messages", "prompt", "raw_request", "raw_response", "secret",
    "system", "system_prompt",
}
_IMAGE_ID_RE = re.compile(r"^img_[0-9a-f]{64}$")
_METRIC_FIELDS = (
    "call_type", "role", "model", "status", "error_code", "duration_ms", "ttft_ms",
    "prompt_tokens", "completion_tokens", "tokens_per_second", "estimated_cny", "attempt",
    "streaming", "parse_status",
)
_USAGE_FIELDS = (
    "requests", "max_requests", "elapsed_s", "prompt_tokens", "completion_tokens",
    "total_tokens", "estimated_cny", "reserved_cny",
)
_EVIDENCE_SOURCE_FIELDS = (
    "date_source", "capture_precision", "capture_source", "identity_status",
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    album_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=20_000)


def _clean_string(value: str) -> str:
    value = _PATH_RE.sub("[redacted-path]", value)
    return _SECRET_RE.sub("[redacted-secret]", value)


def _public_value(value: Any, depth: int = 0) -> Any:
    if depth > 16:
        return "[redacted-depth]"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            lower = name.lower()
            if (lower in _PRIVATE_KEYS or lower.endswith("_path") or lower.endswith("_dir")
                    or lower.endswith("_root") or "credential" in lower or "password" in lower):
                continue
            result[name] = _public_value(item, depth + 1)
        return result
    if isinstance(value, list):
        return [_public_value(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return [_public_value(item, depth + 1) for item in value]
    if isinstance(value, str):
        return _clean_string(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clean_string(str(value))


def _bounded_public(value: Any, depth: int = 0) -> Any:
    """Bound every browser payload after removing private fields."""
    value = _public_value(value)
    if depth >= 7:
        return "[truncated]"
    if isinstance(value, dict):
        return {str(key): _bounded_public(item, depth + 1)
                for key, item in list(value.items())[:80]}
    if isinstance(value, list):
        return [_bounded_public(item, depth + 1) for item in value[:100]]
    if isinstance(value, str):
        return value if len(value) <= 12_000 else value[:12_000] + "…"
    return value


def _sse(event: str, value: dict[str, Any]) -> str:
    payload = json.dumps(_bounded_public(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _unique_image_ids(value: Any) -> list[str]:
    result = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str) and _IMAGE_ID_RE.fullmatch(item) and item not in result:
            result.append(item)
        if len(result) == 355:
            break
    return result


def _tool_image_ids(value: Any) -> list[str]:
    found: list[str] = []

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 6 or len(found) >= 100:
            return
        if isinstance(item, list):
            for child in item[:100]:
                visit(child, depth + 1)
            return
        if not isinstance(item, dict):
            return
        for key in ("image_id", "id"):
            candidate = item.get(key)
            if isinstance(candidate, str) and _IMAGE_ID_RE.fullmatch(candidate) and candidate not in found:
                found.append(candidate)
        for candidate in item.get("image_ids") if isinstance(item.get("image_ids"), list) else []:
            if isinstance(candidate, str) and _IMAGE_ID_RE.fullmatch(candidate) and candidate not in found:
                found.append(candidate)
        for key, child in list(item.items())[:100]:
            if not any(token in str(key).lower() for token in ("path", "root", "prompt", "message", "secret")):
                visit(child, depth + 1)

    visit(value)
    return found


def _observation(value: Any) -> tuple[Any, str]:
    """Return a small audit summary, never the complete tool payload."""
    parsed = value
    if isinstance(value, str) and "] " in value:
        try:
            parsed = json.loads(value.split("] ", 1)[1])
        except (TypeError, ValueError):
            parsed = value
    status = "error" if isinstance(parsed, dict) and parsed.get("status") == "error" else "success"
    payload = parsed.get("data") if isinstance(parsed, dict) and parsed.get("status") == "ok" else parsed
    count = None
    if isinstance(payload, dict):
        if isinstance(payload.get("count"), int):
            count = payload["count"]
        else:
            for key in ("results", "images", "records", "items"):
                if isinstance(payload.get(key), list):
                    count = len(payload[key])
                    break
    image_ids = _tool_image_ids(payload)
    return {
        "status": status,
        "summary": _clean_string(_result_summary(payload))[:500],
        "count": count,
        "image_ids": image_ids,
    }, status


def _result_summary(result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        error = result.get("error")
        if isinstance(error, dict):
            return str(error.get("reason") or error.get("code") or "工具执行失败")[:500]
        return str(error)[:500]
    if isinstance(result, dict):
        for key in ("results", "images", "records", "items"):
            if isinstance(result.get(key), list):
                return f"返回 {len(result[key])} 条公开证据"
        return str(result.get("summary") or result.get("caption") or result.get("description")
                   or "工具调用完成")[:500]
    if isinstance(result, list):
        return f"返回 {len(result)} 条公开证据"
    return str(result or "工具调用完成")[:500]


def _compact_tool_trace(value: Any) -> list[dict[str, Any]]:
    result = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        tool_result = item.get("result")
        status = item.get("status") or (
            "error" if isinstance(tool_result, dict) and "error" in tool_result else "success"
        )
        result.append({
            "step": item.get("step"),
            "tool": str(item.get("tool") or "unknown")[:80],
            "args": _bounded_public(item.get("args") or item.get("arguments") or {}),
            "cached": bool(item.get("cached")),
            "status": status if status in {"success", "error"} else "error",
            "duration_ms": item.get("duration_ms"),
            "summary": _clean_string(_result_summary(tool_result)),
        })
        if len(result) == 100:
            break
    return result


def _evidence_metadata(trace: Any, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def first(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
        return next((value[key] for key in keys if value.get(key) not in (None, "", [])), None)

    def compact(value: Any, *, max_string: int = 300, depth: int = 0) -> Any:
        value = _public_value(value)
        if depth >= 4:
            return "[truncated]"
        if isinstance(value, str):
            return value if len(value) <= max_string else value[:max_string] + "…"
        if isinstance(value, list):
            return [compact(item, max_string=max_string, depth=depth + 1) for item in value[:20]]
        if isinstance(value, dict):
            return {
                str(key): compact(item, max_string=max_string, depth=depth + 1)
                for key, item in list(value.items())[:20]
            }
        return value

    def metadata(value: dict[str, Any]) -> dict[str, Any]:
        start, finish = value.get("date_start"), value.get("date_end")
        date = first(value, ("captured_at", "date_taken", "datetime", "date", "time"))
        if date is None and start:
            date = start if not finish or finish == start else f"{start} 至 {finish}"
        description_keys = ["description_preview", "caption", "description", "summary", "event_name"]
        record_id = value.get("id")
        if value.get("kind") == "event" or (isinstance(record_id, str) and record_id.startswith("ev_")):
            description_keys.extend(("name", "event"))
        canonical = {
            "date": date,
            "location": first(value, ("location", "place", "trusted_location", "observed_location")),
            "people": first(value, ("people", "trusted_people", "observed_people")),
            "description": first(value, tuple(description_keys)),
            "filename": first(value, ("file_name", "filename")),
        }
        public = {
            key: compact(item, max_string=800 if key == "description" else 300)
            for key, item in canonical.items() if item is not None
        }
        public.update({
            key: compact(value[key]) for key in _EVIDENCE_SOURCE_FIELDS if key in value
        })
        return public

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 8 or len(found) >= 355:
            return
        if isinstance(value, list):
            for item in value[:355]:
                visit(item, depth + 1)
            return
        if not isinstance(value, dict):
            return
        image_ids: list[str] = []
        direct_id = value.get("image_id") or value.get("id")
        if isinstance(direct_id, str) and direct_id in allowed_ids:
            image_ids.append(direct_id)
        for key in ("image_ids", "matched_image_ids"):
            for image_id in _unique_image_ids(value.get(key)):
                if image_id in allowed_ids and image_id not in image_ids:
                    image_ids.append(image_id)
        public = metadata(value)
        for image_id in image_ids:
            if public:
                current = found.get(image_id, {})
                merged = dict(current)
                for key, item in public.items():
                    if key not in merged or merged[key] in (None, "", []):
                        merged[key] = item
                    elif key == "description" and len(str(item)) > len(str(merged[key])):
                        merged[key] = item
                found[image_id] = merged
        for key, item in list(value.items())[:100]:
            if not any(token in str(key).lower() for token in
                       ("path", "root", "prompt", "message", "secret")):
                visit(item, depth + 1)

    for row in trace if isinstance(trace, list) else []:
        if isinstance(row, dict):
            visit(row.get("result"))
    return found


def _compact_answer(value: dict[str, Any], request_id: str, config: ChatConfig) -> dict[str, Any]:
    selected = _unique_image_ids(value.get("selected_image_ids"))
    retrieved = _unique_image_ids(value.get("retrieved_image_ids"))
    evidence = _unique_image_ids(value.get("evidence_image_ids"))
    trace = value.get("tool_trace")
    metrics = []
    source_metrics = value.get("model_call_metrics")
    for item in source_metrics if isinstance(source_metrics, list) else []:
        if isinstance(item, dict):
            metrics.append({key: _bounded_public(item.get(key))
                            for key in _METRIC_FIELDS if key in item})
        if len(metrics) == 50:
            break
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    return {
        "answer": value.get("response") or value.get("final_text") or "",
        "request_id": request_id,
        "album_id": config.album_id,
        "model": config.model,
        "steps": value.get("steps"),
        "view_image_calls": value.get("view_image_calls"),
        "stopped_reason": value.get("stopped_reason"),
        "latency_s": value.get("latency_s"),
        "selected_image_ids": selected,
        "retrieved_image_ids": retrieved,
        "evidence_image_ids": evidence,
        "index_version": value.get("index_version"),
        "config": value.get("config"),
        "turn_outcome": value.get("turn_outcome"),
        "usage": {key: _bounded_public(usage.get(key)) for key in _USAGE_FIELDS if key in usage},
        "model_call_metrics": metrics,
        "tool_trace": _compact_tool_trace(trace),
        "evidence_metadata": _evidence_metadata(trace, set(selected + retrieved + evidence)),
    }


def _model_policy_matches(value: dict[str, Any], config: ChatConfig) -> bool:
    """Require runtime evidence that every recorded model call used Qwen."""
    metrics = value.get("model_call_metrics")
    if not isinstance(metrics, list):
        return False
    if not metrics:
        return value.get("turn_outcome") == "error"
    return all(isinstance(item, dict) and item.get("model") == config.model for item in metrics)


async def _upstream_events(response: httpx.Response) -> AsyncIterator[tuple[str, Any]]:
    event = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                raw = "\n".join(data)
                try:
                    yield event, json.loads(raw)
                except (TypeError, ValueError):
                    yield "invalid", None
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        try:
            yield event, json.loads("\n".join(data))
        except (TypeError, ValueError):
            yield "invalid", None


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return False
    parsed = urlsplit(origin)
    return (parsed.scheme in {"http", "https"} and parsed.netloc.lower() == request.headers.get("host", "").lower()
            and not parsed.path and not parsed.query and not parsed.fragment)


def create_app(
    config: ChatConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the app without loading CamRoll, WeMM, or any model."""

    config = config or ChatConfig.from_env()
    album = FrozenAlbum(config)
    execution_lock = threading.Lock()
    media_work_limit = asyncio.Semaphore(4)
    background_tasks: set[asyncio.Task] = set()
    app = FastAPI(
        title="WeMM–CamRoll Album Chat",
        version=SERVICE_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.execution_lock = execution_lock
    app.state.background_tasks = background_tasks

    def client(timeout: httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        host = request.headers.get("host", "").lower()
        if host not in {value.lower() for value in config.allowed_hosts}:
            response = JSONResponse({"error": "host_not_allowed"}, status_code=400)
        elif request.method not in {"GET", "HEAD", "OPTIONS"} and not _same_origin(request):
            response: Response = JSONResponse({"error": "same_origin_required"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
            "form-action 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _exc: Exception):
        return JSONResponse({"error": "internal_error"}, status_code=500)

    @app.get("/")
    async def index():
        path = Path(config.static_dir or "") / "index.html"
        if not path.is_file():
            raise HTTPException(404, "frontend unavailable")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @app.get("/app.js")
    async def javascript():
        path = Path(config.static_dir or "") / "app.js"
        if not path.is_file():
            raise HTTPException(404, "frontend unavailable")
        return FileResponse(path, media_type="application/javascript; charset=utf-8")

    @app.get("/style.css")
    async def stylesheet():
        path = Path(config.static_dir or "") / "style.css"
        if not path.is_file():
            raise HTTPException(404, "frontend unavailable")
        return FileResponse(path, media_type="text/css; charset=utf-8")

    async def upstream_ready() -> bool:
        timeout = httpx.Timeout(3.0, connect=config.upstream_connect_timeout_s)
        try:
            async with client(timeout) as session:
                response = await session.get(config.upstream_url + "/api/health")
            if response.status_code != 200:
                return False
            value = response.json()
            return (isinstance(value, dict) and value.get("status") == "ready"
                    and config.album_id in (value.get("albums") or [])
                    and config.agent_config in (value.get("configs") or []))
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    @app.get("/api/health")
    async def health():
        album_state = album.inspect()
        upstream = "ready" if await upstream_ready() else "unavailable"
        status = "ready" if album_state.status == "ready" and upstream == "ready" else "degraded"
        return {
            "status": status,
            "version": SERVICE_VERSION,
            "busy": execution_lock.locked(),
            "upstream": upstream,
            "album": album.public_metadata(),
            "policy": {"config": config.agent_config, "model": config.model},
        }

    @app.get("/api/albums")
    async def albums():
        return {"default_album_id": config.album_id, "albums": [album.public_metadata()]}

    @app.get("/api/albums/{album_id}/media/{image_id}")
    async def media(album_id: str, image_id: str, variant: Literal["thumb", "full"] = "thumb"):
        if album_id != config.album_id:
            raise HTTPException(404, "media not found")
        def create_thumbnail() -> bytes:
            path = album.media_path(image_id)
            with Image.open(path) as source:
                picture = ImageOps.exif_transpose(source).convert("RGB")
                picture.thumbnail((960, 960))
                output = io.BytesIO()
                picture.save(output, "JPEG", quality=85)
                return output.getvalue()

        try:
            async with media_work_limit:
                if variant == "full":
                    path = await asyncio.to_thread(album.media_path, image_id)
                else:
                    output = await asyncio.to_thread(create_thumbnail)
        except (
            KeyError, AlbumUnavailable, IndexVersionChanged, SnapshotIntegrityChanged,
            OSError, UnidentifiedImageError, Image.DecompressionBombError,
        ):
            raise HTTPException(404, "media not found") from None
        if variant == "full":
            return FileResponse(path, media_type="image/jpeg")
        return Response(output, media_type="image/jpeg")

    def error_event(code: str, *, uncertain: bool) -> str:
        return _sse("error", {
            "code": code,
            "uncertain": uncertain,
            "retriable": False,
            "message": (
                "请求状态不确定；重新提交会产生一次新的云端调用。"
                if uncertain else "本次请求未执行，请检查服务状态。"
            ),
        })

    @app.post("/api/chat/stream")
    async def chat(body: ChatRequest):
        if body.album_id != config.album_id:
            raise HTTPException(404, "album not found")
        if not body.question.strip():
            raise HTTPException(422, "question cannot be blank")
        try:
            album.ensure_ready()
        except (IndexVersionChanged, SnapshotIntegrityChanged):
            raise HTTPException(409, "frozen album index changed") from None
        except AlbumUnavailable:
            raise HTTPException(503, "frozen album unavailable") from None
        if not execution_lock.acquire(blocking=False):
            raise HTTPException(429, "another question is already running")

        request_id = "chat-" + uuid.uuid4().hex
        payload = {
            "album_id": config.album_id,
            "question": body.question.strip(),
            "config": config.agent_config,
            "stream": True,
            "index_version": config.index_version,
            "structured_output": True,
            "stream_model": True,
        }

        queue: asyncio.Queue = asyncio.Queue()
        end = object()

        async def emit(event: str, value: dict[str, Any]) -> None:
            queue.put_nowait(_sse(event, value))

        async def producer() -> None:
            answered = False
            try:
                await emit("request_started", {
                    "request_id": request_id,
                    "album_id": config.album_id,
                    "index_version": config.index_version,
                    "config": config.agent_config,
                    "model": config.model,
                })
                await emit("model_status", {
                    "phase": "agent",
                    "status": "running",
                    "message": "正在等待 Qwen3.7-Flash 决策",
                    "model": config.model,
                })
                timeout = httpx.Timeout(
                    config.upstream_read_timeout_s,
                    connect=config.upstream_connect_timeout_s,
                    write=config.upstream_write_timeout_s,
                    pool=config.upstream_pool_timeout_s,
                )
                parsed_upstream = urlsplit(config.upstream_url)
                origin = f"{parsed_upstream.scheme}://{parsed_upstream.netloc}"
                async with client(timeout) as session:
                    async with session.stream(
                        "POST",
                        config.upstream_url + "/api/ask",
                        json=payload,
                        headers={"Accept": "text/event-stream", "Origin": origin},
                    ) as upstream:
                        if upstream.status_code != 200:
                            codes = {409: "index_changed", 422: "upstream_rejected", 429: "upstream_busy"}
                            queue.put_nowait(error_event(
                                codes.get(upstream.status_code, "upstream_unavailable"), uncertain=False
                            ))
                            return
                        async for kind, value in _upstream_events(upstream):
                            if kind == "status" and isinstance(value, dict):
                                await emit("model_status", {
                                    "phase": "agent",
                                    "status": "running",
                                    "message": value.get("message") or "Agent 正在处理",
                                    "model": config.model,
                                    "index_version": value.get("index_version"),
                                    "config": config.agent_config,
                                })
                            elif kind == "tool_call" and isinstance(value, dict):
                                await emit("tool_call", {
                                    "step": value.get("step"),
                                    "tool": value.get("tool"),
                                    "args": value.get("args") if isinstance(value.get("args"), dict) else {},
                                    "status": "running",
                                })
                            elif kind == "tool_result" and isinstance(value, dict):
                                observation, tool_status = _observation(value.get("observation"))
                                await emit("tool_result", {
                                    "step": value.get("step"),
                                    "tool": value.get("tool"),
                                    "observation": observation,
                                    "cached": bool(value.get("cached")),
                                    "status": tool_status,
                                })
                                await emit("model_status", {
                                    "phase": "agent",
                                    "status": "running",
                                    "message": "工具返回完成，等待 Qwen3.7-Flash 下一步决策",
                                    "model": config.model,
                                })
                            elif kind == "answer" and isinstance(value, dict):
                                if (value.get("index_version") != config.index_version
                                        or value.get("config") != config.agent_config):
                                    queue.put_nowait(error_event("index_or_policy_mismatch", uncertain=True))
                                    return
                                if not _model_policy_matches(value, config):
                                    queue.put_nowait(error_event("model_policy_mismatch", uncertain=True))
                                    return
                                public = _compact_answer(value, request_id, config)
                                successful = (public.get("turn_outcome") != "error"
                                              and public.get("stopped_reason") in {"ok", "insufficient_evidence"})
                                await emit("model_status", {
                                    "phase": "answer",
                                    "status": "complete" if successful else "error",
                                    "message": ("Qwen3.7-Flash 已完成回答" if successful
                                                else "Agent 未能完成可靠回答"),
                                    "model": config.model,
                                })
                                await emit("answer", public)
                                answered = True
                            elif kind == "error":
                                queue.put_nowait(error_event("agent_failed", uncertain=True))
                                return
                            elif kind == "invalid":
                                queue.put_nowait(error_event("invalid_upstream_stream", uncertain=True))
                                return
                if not answered:
                    queue.put_nowait(error_event("incomplete_upstream_stream", uncertain=True))
            except (httpx.HTTPError, TimeoutError):
                queue.put_nowait(error_event("upstream_connection_failed", uncertain=True))
            except asyncio.CancelledError:
                # This only occurs during process shutdown. Browser disconnects
                # cancel the consumer, while this task continues draining the
                # already-started billed request to its terminal state.
                raise
            except Exception:
                # Runtime failures are reduced to a bounded public event.
                queue.put_nowait(error_event("chat_failed", uncertain=True))
            finally:
                execution_lock.release()
                queue.put_nowait(end)

        async def events() -> AsyncIterator[str]:
            while True:
                item = await queue.get()
                if item is end:
                    return
                yield item

        task = asyncio.create_task(producer(), name=request_id)
        background_tasks.add(task)

        def task_done(completed: asyncio.Task) -> None:
            background_tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(task_done)

        response = StreamingResponse(events(), media_type="text/event-stream")
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


app = create_app()
