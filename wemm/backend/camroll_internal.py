"""Install in WeMM backend, mounted from app.py. Never imports a second model."""
import asyncio
import copy
import json
import queue
import threading
from pathlib import Path

import numpy as np
from fastapi import HTTPException, Request


def install(app, get_worker, allowed_root, *, session_validator):
    root = Path(allowed_root).resolve()
    token_lock = threading.Lock()
    tokenizer_cache = {}

    def token_count(encoder, text):
        # Fast tokenizers can mutate padding/truncation settings in encode().
        # Keep the API's CPU tokenizer independent of the active GPU processor.
        with token_lock:
            key = (id(encoder.processor.tokenizer), encoder.fingerprint)
            if tokenizer_cache.get("key") != key:
                tokenizer_cache.update(key=key, tokenizer=copy.deepcopy(encoder.processor.tokenizer))
            return len(tokenizer_cache["tokenizer"].encode(text, add_special_tokens=False))

    def ready(request):
        if not request.client or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "Internal endpoint accepts loopback clients only")
        if not session_validator(request.cookies.get("wemm_session", "")):
            raise HTTPException(401, "WeMM session authentication is required")
        worker = get_worker()
        if (worker is None or worker.encoder is None or worker.state != "ready"
                or getattr(worker, "stop", False)):
            raise HTTPException(503, "WeMM worker is not ready")
        return worker

    async def body(request):
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        if declared < 0:
            raise HTTPException(400, "Invalid Content-Length")
        if declared > 2_000_000:
            raise HTTPException(413, "Request too large")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > 2_000_000:
                raise HTTPException(413, "Request too large")
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "Invalid JSON body") from None
        if not isinstance(data, dict):
            raise HTTPException(422, "Expected JSON object")
        return data

    @app.post("/api/internal/embeddings")
    async def embeddings(request: Request):
        worker = ready(request)
        data = await body(request)
        if set(data) - {"text", "image_path"}:
            raise HTTPException(422, "Unknown embedding fields")
        encoder = worker.encoder
        fingerprint = encoder.fingerprint
        text, image_path = data.get("text"), data.get("image_path")
        if (text is None) == (image_path is None):
            raise HTTPException(422, "Supply exactly one of text or image_path")
        if text is not None:
            if not isinstance(text, str) or not text or len(text) > 200_000:
                raise HTTPException(422, "text must be a nonempty string")
            count = await asyncio.to_thread(token_count, encoder, text)
            if count > 512:
                raise HTTPException(422, {"code": "text_too_long", "tokens": count, "limit": 512})
            kwargs = {"text": text}
        else:
            if not isinstance(image_path, str):
                raise HTTPException(422, "image_path must be a string")
            path = Path(image_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise HTTPException(403, "Image must exist inside the CamRoll asset root")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise HTTPException(422, "Unsupported image format")
            if path.stat().st_size > 128 * 1024 * 1024:
                raise HTTPException(413, "Image exceeds 128 MiB")
            kwargs = {"image": path}
        try:
            if ready(request) is not worker or worker.encoder is not encoder:
                raise HTTPException(503, "WeMM worker changed before encoding")
            future = asyncio.wrap_future(worker.request(kwargs))
            # A disconnected HTTP consumer must not cancel the worker Future:
            # worker.set_result on a cancelled Future would corrupt scheduling.
            vector, timings = await asyncio.shield(future)
        except queue.Full:
            raise HTTPException(429, "WeMM queue is full")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        vector = np.asarray(vector, dtype=np.float32)
        if (vector.shape != (4096,) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)):
            raise HTTPException(500, "WeMM returned an invalid normalized 4096-dimensional vector")
        if encoder is not worker.encoder or timings.get("fingerprint", fingerprint) != fingerprint:
            raise HTTPException(409, "WeMM fingerprint changed during encoding")
        return {"embedding": vector.tolist(), "dimension": len(vector),
                "fingerprint": fingerprint, "timings": timings}

    @app.post("/api/internal/tokenize")
    async def tokenize(request: Request):
        worker = ready(request)
        data = await body(request)
        if set(data) - {"text"}:
            raise HTTPException(422, "Unknown tokenizer fields")
        text = data.get("text")
        if not isinstance(text, str) or len(text) > 200_000:
            raise HTTPException(422, "text must be a string of at most 200000 characters")
        encoder = worker.encoder
        count = lambda value: token_count(encoder, value)
        # Split by character boundaries, testing the exact re-encoded string.
        # Decoding fixed token windows can split multi-token Unicode characters.
        def split():
            total = count(text)
            chunks, remaining = [], text
            while remaining:
                lo, hi = 1, min(len(remaining), 4096)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count(remaining[:mid]) <= 512:
                        lo = mid
                    else:
                        hi = mid - 1
                chunk = remaining[:lo]
                if count(chunk) > 512:
                    raise HTTPException(422, "Cannot split token sequence safely")
                chunks.append(chunk)
                remaining = remaining[lo:]
            return total, chunks
        total, chunks = await asyncio.to_thread(split)
        return {"tokens": total, "chunks": chunks,
                "fingerprint": encoder.fingerprint, "limit": 512}
