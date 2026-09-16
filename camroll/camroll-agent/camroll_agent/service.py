"""Loopback-only CamRoll question/answer service. No model loads at import."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    album_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')
    question: str = Field(min_length=1, max_length=20000)
    config: str = Field(default='M1', pattern=r'^M[1-4]$')
    stream: bool = False


def create_app(album_root=None, agent_factory=None):
    app = FastAPI(title='CamRoll Qwen + WeMM', version='0.2.0')
    root = Path(album_root or os.getenv('CAMROLL_ALBUM_ROOT', 'runtime/albums')).resolve()
    lock = threading.Lock()
    libraries = {}
    clients = {}

    def make_agent(request):
        if agent_factory:
            return agent_factory(request)
        from .agent import Agent
        from .library import Library
        from .qwen import QwenClient
        from .wemm import EmbeddingClient

        directory = (root/request.album_id).resolve()
        if not directory.is_relative_to(root) or not (directory/'CURRENT').is_file():
            raise HTTPException(404, 'Album does not have a published index')
        version = (directory/'CURRENT').read_text().strip()
        with lock:
            if not clients:
                clients.update(qwen=QwenClient(), embedding=EmbeddingClient())
            key = (request.album_id, version, request.config)
            if key not in libraries:
                loaded = Library(directory, clients['embedding'], config=request.config)
                # CURRENT can atomically change between the earlier read and
                # Library construction. Cache under the version actually loaded,
                # so a later rollback never retrieves a mislabelled snapshot.
                version = loaded.version
                key = (request.album_id, version, request.config)
                libraries[key] = loaded
                # Active requests retain their immutable instance independently.
                for old in list(libraries):
                    if old[0] == request.album_id and old[1] != version:
                        del libraries[old]
            library = libraries[key]
        return Agent(str(directory), llm=clients['qwen'], vlm=clients['qwen'],
                     library=library, config=request.config)

    @app.get('/api/health')
    def health():
        return {'status': 'ready', 'configs': ['M1','M2','M3','M4'],
                'albums': sorted(p.parent.name for p in root.glob('*/CURRENT')) if root.exists() else []}

    @app.post('/api/ask')
    def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(422, 'question cannot be blank')
        agent = make_agent(request)
        if request.stream:
            def events():
                try:
                    for kind, value in agent.ask_streaming(request.question):
                        yield 'event: ' + kind + '\ndata: ' + json.dumps(value, ensure_ascii=False, default=str) + '\n\n'
                except Exception as exc:
                    # Never expose exception messages from external SDKs.
                    yield 'event: error\ndata: ' + json.dumps({'error':type(exc).__name__}) + '\n\n'
            return StreamingResponse(events(), media_type='text/event-stream')
        result = agent.ask(request.question)
        value = asdict(result)
        value['answer'] = value['final_text']
        value['album_id'] = request.album_id
        return value

    return app


app = create_app()
