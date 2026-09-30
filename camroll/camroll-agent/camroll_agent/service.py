"""Loopback-only CamRoll question/answer service. No model loads at import."""
from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    album_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')
    question: str = Field(min_length=1, max_length=20000)
    config: Optional[str] = Field(default=None, pattern=r'^M[1-4]$')
    stream: bool = False
    conversation_id: Optional[str] = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$')
    request_id: Optional[str] = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$')
    index_version: Optional[str] = Field(default=None, max_length=120)
    structured_output: bool = False
    stream_model: bool = False


def create_app(album_root=None, agent_factory=None):
    app = FastAPI(title='CamRoll Qwen + WeMM', version='0.2.0')
    root = Path(album_root or os.getenv('CAMROLL_ALBUM_ROOT', 'runtime/albums')).resolve()
    lock = threading.Lock()
    libraries = {}
    clients = {}
    session_locks = {}
    request_locks = {}

    def state_path(kind, identifier):
        return root / ('.' + kind) / (hashlib.sha256(identifier.encode()).hexdigest() + '.json')

    def save_state(path, value):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

    def make_agent(request):
        if agent_factory:
            return agent_factory(request)
        from .agent import Agent
        from .library import Library
        from .runtime_clients import build_clients

        directory = (root/request.album_id).resolve()
        if not directory.is_relative_to(root) or not (directory/'CURRENT').is_file():
            raise HTTPException(404, 'Album does not have a published index')
        version = (directory/'CURRENT').read_text().strip()
        with lock:
            key = (request.album_id, version, request.config)
            if key not in libraries:
                loaded = Library(directory, None, config=request.config)
                cloud, embedding, reranker, experiment_config = build_clients(memory=directory, snapshot=loaded)
                loaded.embedding, loaded.reranker = embedding, reranker
                # CURRENT can atomically change between the earlier read and
                # Library construction. Cache under the version actually loaded,
                # so a later rollback never retrieves a mislabelled snapshot.
                version = loaded.version
                key = (request.album_id, version, request.config)
                libraries[key] = loaded
                clients[key] = cloud
                # Active requests retain their immutable instance independently.
                for old in list(libraries):
                    if old[0] == request.album_id and old[1] != version:
                        del libraries[old]
                        clients.pop(old, None)
            library = libraries[key]
            cloud = clients[key]
        options = {'deadline_s': 600} if getattr(library, 'experiment', None) else {}
        return Agent(str(directory), llm=cloud, vlm=cloud,
                     library=library, config=library.config, **options)

    @app.get('/api/health')
    def health():
        return {'status': 'ready', 'configs': ['M1','M2','M3','M4'],
                'albums': sorted(p.parent.name for p in root.glob('*/CURRENT')) if root.exists() else []}

    @app.post('/api/ask')
    def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(422, 'question cannot be blank')
        if request.stream and (request.conversation_id or request.request_id):
            raise HTTPException(422, 'Stateful requests use non-SSE mode; model streaming is independent')
        if request.conversation_id or request.request_id:
            return ask_stateful(request)
        agent = make_agent(request)
        if request.index_version and request.index_version != str(agent.library.version):
            raise HTTPException(409, 'Index version changed')
        options = {}
        if request.structured_output or request.stream_model:
            options = dict(structured_output=request.structured_output, stream_model=request.stream_model)
        if request.stream:
            def events():
                try:
                    for kind, value in agent.ask_streaming(request.question, **options):
                        yield 'event: ' + kind + '\ndata: ' + json.dumps(value, ensure_ascii=False, default=str) + '\n\n'
                except Exception as exc:
                    # Never expose exception messages from external SDKs.
                    yield 'event: error\ndata: ' + json.dumps({'error':type(exc).__name__}) + '\n\n'
            return StreamingResponse(events(), media_type='text/event-stream')
        result = agent.ask(request.question, **options)
        value = asdict(result)
        value['answer'] = value['final_text']
        value['album_id'] = request.album_id
        return value

    def ask_stateful(request):
        # A session is pinned to album/config/version. Each turn is serialized,
        # while different QA sessions can run concurrently.
        session_id = request.conversation_id or ('request:' + request.request_id)
        with lock:
            guard = session_locks.setdefault(session_id, threading.Lock())
            request_guard = request_locks.setdefault(request.request_id, threading.Lock()) if request.request_id else None
        if not guard.acquire(blocking=False):
            raise HTTPException(409, 'Conversation already has an active turn')
        if request_guard and not request_guard.acquire(blocking=False):
            guard.release()
            raise HTTPException(409, 'Request already has an active execution')
        request_path = state_path('requests', request.request_id) if request.request_id else None
        session_path = state_path('sessions', session_id)
        fingerprint = hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest()
        try:
            if request_path and request_path.exists():
                cached = json.loads(request_path.read_text())
                if cached.get('fingerprint') != fingerprint:
                    raise HTTPException(409, 'request_id is bound to different input')
                if cached.get('status') == 'complete':
                    return cached['result']
                raise HTTPException(409, 'Request is in progress or has an uncertain outcome; inspect its status')
            state = json.loads(session_path.read_text()) if session_path.exists() else None
            agent = make_agent(request)
            version = str(agent.library.version)
            if request.index_version and request.index_version != version:
                raise HTTPException(409, 'Index version changed')
            binding = [request.album_id, getattr(agent, 'config', request.config), version]
            if state and state['binding'] != binding:
                raise HTTPException(409, 'Conversation album, config or snapshot changed')
            if request_path:
                save_state(request_path, {'status': 'running', 'fingerprint': fingerprint,
                    'created_at': time.time(), 'request_id': request.request_id})
            try:
                result = agent.ask(request.question, history=state['history'] if state else [],
                    structured_output=True, stream_model=request.stream_model,
                    request_context={'request_id': request.request_id, 'conversation_id': session_id})
                value = asdict(result)
                value.update(answer=value['final_text'], album_id=request.album_id, conversation_id=session_id,
                             request_id=request.request_id)
                save_state(session_path, {'binding': binding, 'history': value['conversation_history'],
                                          'updated_at': time.time()})
                if request_path:
                    save_state(request_path, {'status': 'complete', 'fingerprint': fingerprint,
                                             'request_id': request.request_id, 'result': value})
                return value
            except Exception as exc:
                if request_path:
                    save_state(request_path, {'status': 'uncertain', 'fingerprint': fingerprint,
                        'request_id': request.request_id, 'error_type': type(exc).__name__})
                raise HTTPException(500, 'Agent request failed; do not automatically repeat a billed request') from None
        finally:
            if request_guard:
                request_guard.release()
            guard.release()

    @app.get('/api/requests/{request_id}')
    def request_status(request_id: str):
        path = state_path('requests', request_id)
        if not path.exists():
            raise HTTPException(404, 'Unknown request')
        value = json.loads(path.read_text())
        value.pop('fingerprint', None)
        return value

    return app


app = create_app()
