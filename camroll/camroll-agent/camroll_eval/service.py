"""LAN-accessible independent 8772 UI/API. The original 8771 service is untouched."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .common import read_json, safe_id, now_iso
from .repository import Repository
from .runner import Runner, configured_paths


class StartRequest(BaseModel):
    config: str = Field(default="M1", pattern=r"^M[1-4]$")


def create_app(root=None, source=None, albums=None, *, runner_factory=Runner, mount_agent=True):
    default_root, default_source, default_albums = configured_paths()
    root, source, albums = Path(root or default_root), Path(source or default_source), Path(albums or default_albums)
    app = FastAPI(title="CamRoll × PhotoBench", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
    repo = Repository(root)
    execution_lock = threading.Lock()
    active = {"runner": None, "thread": None}
    static = Path(__file__).with_name("static")

    def origin_ok(request):
        origin = request.headers.get("origin")
        if not origin:
            return False
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == request.headers.get("host") and not parsed.path

    @app.middleware("http")
    async def guard(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not origin_ok(request):
            return JSONResponse({"error": "same_origin_required"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/app.js")
    def javascript():
        return FileResponse(static / "app.js", media_type="application/javascript")

    @app.get("/style.css")
    def styles():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/health")
    def health():
        live = active["thread"] is not None and active["thread"].is_alive()
        return {"status": "ready", "version": "0.1.0", "source_available": (source / "manifest.json").is_file(),
                "active_run": active["runner"]._state.get("run_id") if live and active["runner"]._state else None}

    @app.get("/api/runs")
    def runs():
        return {"runs": repo.list()}

    @app.get("/api/comparison-profile")
    def comparison_profile():
        path = root / "comparison-profile.json"
        return read_json(path) if path.is_file() else {}

    @app.get("/api/compare")
    def compare(ids: str):
        values = ids.split(",")
        if not 1 <= len(values) <= 6:
            raise HTTPException(422, "select one to six runs")
        try:
            return repo.compare(values)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "run not found") from None

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        try:
            state = repo.load(run_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "run not found") from None
        result = {key: value for key, value in state.items() if key not in {"items", "media_mapping", "qa_ids"}}
        result["items"] = [{key: row.get(key) for key in ("qa_id", "question", "answer", "task_type", "angle", "difficulty",
                           "answerability", "execution_status", "judge_status", "judge", "retrieval_precision",
                           "retrieval_recall", "retrieval_f1", "agent_status", "wall_clock_ms")}
                           for row in state.get("items", [])]
        present = {row["qa_id"] for row in result["items"]}
        result["items"].extend({**row, "judge": {"score": None, "reason": "not_started"}}
                                for row in state.get("planned_status", []) if row["qa_id"] not in present)
        for row in result["items"]:
            judge = row.get("judge") or {}
            row["judge"] = {key: judge.get(key) for key in ("score", "reason", "judge_status")}
        progress = repo.path(run_id).parent / "progress.json"
        if progress.is_file():
            result["live_progress"] = read_json(progress)
        return result

    @app.get("/api/runs/{run_id}/items/{qa_id}")
    def item(run_id: str, qa_id: str):
        try:
            state = repo.load(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "run not found") from None
        value = next((row for row in state.get("items", []) if row.get("qa_id") == qa_id), None)
        if value is None:
            planned = next((row for row in state.get("planned_status", []) if row["qa_id"] == qa_id), None)
            if planned:
                return {**planned, "answer": "", "judge": {"score": None, "reason": "not_started"}, "runtime_turns": []}
            raise HTTPException(404, "item not found")
        return value

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str):
        try:
            path = repo.path(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "run not found") from None
        return FileResponse(path, filename=run_id + ".json", media_type="application/json")

    @app.get("/api/runs/{run_id}/resources")
    def resources(run_id: str):
        try:
            state = repo.load(run_id)
            resource_id = state.get("parent_run_id") if state.get("mode") == "composite" else run_id
            directory = repo.path(resource_id).parent
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "run not found") from None
        path = directory / "resources.jsonl"
        if not path.is_file():
            path = directory / "gpu_samples.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        # Keep endpoints and extrema available through the original JSON export;
        # the interactive chart has a bounded payload for long historical runs.
        stride = max(1, len(rows) // 1500)
        return {"sample_count": len(rows), "stride": stride, "samples": rows[::stride]}

    @app.get("/api/runs/{run_id}/pipeline-calls")
    def pipeline_calls(run_id: str):
        try:
            state = repo.load(run_id)
            safe_id(state.get("album_id"))
        except (FileNotFoundError, ValueError, TypeError):
            raise HTTPException(404, "run not found") from None
        filename = (state.get("build_summary") or {}).get("model_calls_file")
        if state.get("system_id") != "camroll_wemm" or not filename:
            raise HTTPException(404, "pipeline calls were not recorded")
        permitted = (albums / state["album_id"]).resolve()
        path = (permitted / filename).resolve()
        if not path.is_relative_to(permitted) or not path.is_file():
            raise HTTPException(404, "pipeline calls unavailable")
        return FileResponse(path, filename=run_id + "-pipeline-calls.jsonl", media_type="application/x-ndjson")

    @app.get("/api/media/{filename:path}")
    def media(filename: str, thumbnail: bool = True):
        if Path(filename).name != filename:
            raise HTTPException(404, "media not found")
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(404, "source unavailable")
        manifest = read_json(manifest_path)
        paths = [*manifest.get("photos", []), *[r["ref_image"] for r in manifest.get("faces", [])]]
        matches = [source / value for value in paths if Path(value).name == filename]
        if len(matches) != 1 or not matches[0].is_file() or not matches[0].resolve().is_relative_to(source.resolve()):
            raise HTTPException(404, "media not found")
        if not thumbnail:
            return FileResponse(matches[0])
        import io
        from PIL import Image, ImageOps
        with Image.open(matches[0]) as picture:
            picture = ImageOps.exif_transpose(picture).convert("RGB")
            picture.thumbnail((960, 960))
            output = io.BytesIO()
            picture.save(output, "JPEG", quality=85)
        return Response(output.getvalue(), media_type="image/jpeg")

    @app.post("/api/runs")
    def start(body: StartRequest):
        with execution_lock:
            if active["thread"] is not None and active["thread"].is_alive():
                raise HTTPException(409, "an evaluation is already active")
            from .dataset import Dataset
            try:
                Dataset(source)
            except (OSError, ValueError, KeyError):
                raise HTTPException(422, "complete 355-photo / 100-QA source is unavailable") from None
            runner = runner_factory(root, source, albums)
            try:
                runner.acquire()
            except RuntimeError:
                raise HTTPException(409, "another evaluation process is active") from None
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-camroll-" + secrets.token_hex(4)
            def execute():
                try:
                    runner.run(run_id=run_id, config=body.config, lock_held=True)
                except Exception:
                    # Run state contains the bounded diagnostic; no provider
                    # traceback can leak authorization headers to service logs.
                    pass
                finally:
                    runner.release()
            active["runner"] = runner
            active["thread"] = threading.Thread(target=execute, name="camroll-evaluation", daemon=True)
            active["thread"].start()
            return {"run_id": run_id, "status": "starting"}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        try:
            path = repo.path(run_id)
            state = repo.load(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "run not found") from None
        if state.get("system_id") != "camroll_wemm" or state.get("status") not in {"preparing", "building", "evaluating", "judging", "cancelling"}:
            raise HTTPException(409, "only an active independent run can be cancelled")
        (path.parent / "cancel.requested").write_text(now_iso())
        return {"status": "cancel_requested", "note": "Already running requests finish safely; no process is terminated."}

    if mount_agent:
        from camroll_agent.service import create_app as agent_app
        app.mount("/camroll", agent_app(album_root=albums))
    app.state.repository = repo
    app.state.active = active
    return app
