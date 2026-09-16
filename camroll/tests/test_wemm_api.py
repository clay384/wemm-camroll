import concurrent.futures
import importlib.util
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from camroll_agent.budget import BudgetLedger
from camroll_agent.qwen import QwenClient, DeadlineExceeded
from camroll_agent.wemm import EmbeddingClient, WeMMError

FINGERPRINT = 'a' * 64


spec = importlib.util.spec_from_file_location('wemm_internal', Path(__file__).parents[1]/'deployment/wemm_internal.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))


def client_for(tmp_path, host='127.0.0.1', *, authenticated=True, worker_state='ready', tokenizer=None, outcome=None):
    calls = []
    def request(kw):
        calls.append(kw)
        if isinstance(outcome, Exception):
            raise outcome
        future = concurrent.futures.Future()
        v = np.zeros(4096,dtype=np.float32);v[0]=1
        future.set_result((v if outcome is None else outcome,{'embedding_s':0.01,'fingerprint':FINGERPRINT}))
        return future
    worker=SimpleNamespace(state=worker_state,request=request,encoder=SimpleNamespace(fingerprint=FINGERPRINT,processor=SimpleNamespace(tokenizer=tokenizer or Tokenizer())))
    app=FastAPI()
    module.install(app,lambda:worker,tmp_path,session_validator=lambda cookie:cookie=='valid')
    client = TestClient(app,client=(host,12345))
    if authenticated:
        client.cookies.set('wemm_session','valid')
    return client,calls


def test_encoding_and_path_confinement(tmp_path):
    client,calls=client_for(tmp_path)
    path=tmp_path/'image.jpg';path.write_bytes(b'pretend')
    assert client.post('/api/internal/embeddings',json={'image_path':str(path)}).status_code==200
    assert len(calls)==1
    assert client.post('/api/internal/embeddings',json={'image_path':'/etc/passwd'}).status_code==403
    assert len(calls)==1
    assert client.post('/api/internal/embeddings',json={'text':'a'*513}).status_code==422
    assert len(calls)==1


def test_loopback_and_xor(tmp_path):
    client,calls=client_for(tmp_path,'192.168.0.2')
    assert client.post('/api/internal/embeddings',json={'text':'hi'}).status_code==403
    assert not calls
    client,calls=client_for(tmp_path)
    assert client.post('/api/internal/embeddings',json={'text':'hi','image_path':'/x'}).status_code==422
    assert client.post('/api/internal/embeddings',json=[]).status_code==422


def test_unicode_splits_losslessly(tmp_path):
    client,_=client_for(tmp_path)
    text='这是一段中文，含emoji🙂和标点。'*200
    result=client.post('/api/internal/tokenize',json={'text':text}).json()
    assert ''.join(result['chunks'])==text
    assert all(len(Tokenizer().encode(x))<=512 for x in result['chunks'])


def test_explicit_authentication_and_worker_readiness(tmp_path):
    client, calls = client_for(tmp_path, authenticated=False)
    assert client.post('/api/internal/embeddings', json={'text':'x'}).status_code == 401
    client.cookies.set('wemm_session','invalid')
    assert client.post('/api/internal/tokenize', json={'text':'x'}).status_code == 401
    assert calls == []
    client, calls = client_for(tmp_path, worker_state='loading')
    assert client.post('/api/internal/embeddings', json={'text':'x'}).status_code == 503
    assert calls == []


def test_bad_json_header_and_chunked_limit_are_bounded(tmp_path):
    client, calls = client_for(tmp_path)
    assert client.post('/api/internal/tokenize', content=b'{bad').status_code == 400
    assert client.post('/api/internal/tokenize', content=b'{}', headers={'Content-Length':'abc'}).status_code == 400
    assert client.post('/api/internal/tokenize', content=iter([b'x' * 1_000_001, b'x' * 1_000_001])).status_code == 413
    assert client.post('/api/internal/embeddings', json={'text':'x', '_reference':True}).status_code == 422
    assert calls == []


def test_exact_512_boundary_queue_and_normalization(tmp_path):
    client, calls = client_for(tmp_path)
    assert client.post('/api/internal/embeddings', json={'text':'a'*512}).status_code == 200
    assert calls == [{'text':'a'*512}]  # No role, profile, or template override.
    full, _ = client_for(tmp_path, outcome=queue.Full())
    assert full.post('/api/internal/embeddings', json={'text':'x'}).status_code == 429
    invalid, _ = client_for(tmp_path, outcome=np.zeros(4096))
    assert invalid.post('/api/internal/embeddings', json={'text':'x'}).status_code == 500


def test_symlink_cannot_escape_asset_root(tmp_path):
    assets = tmp_path/'assets'; assets.mkdir()
    outside = tmp_path/'outside.jpg'; outside.write_bytes(b'not sent')
    (assets/'link.jpg').symlink_to(outside)
    client, calls = client_for(assets)
    assert client.post('/api/internal/embeddings', json={'image_path':str(assets/'link.jpg')}).status_code == 403
    assert calls == []


def test_api_tokenizer_is_independent_of_encoder_tokenizer(tmp_path):
    class BusyTokenizer:
        def __init__(self, worker_copy=True):
            self.worker_copy = worker_copy
        def __deepcopy__(self, memo):
            return BusyTokenizer(False)
        def encode(self, text, **kwargs):
            assert not self.worker_copy, 'API borrowed the worker tokenizer'
            return list(text.encode())
    client, _ = client_for(tmp_path, tokenizer=BusyTokenizer())
    assert client.post('/api/internal/embeddings', json={'text':'中文'}).status_code == 200
    result = client.post('/api/internal/tokenize', json={'text':'中文🙂 '*500}).json()
    assert ''.join(result['chunks']) == '中文🙂 '*500


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item
    get = post


def embedding_client(session):
    client = EmbeddingClient()
    client._local.session = session
    return client


def test_client_rejects_missing_fingerprint_or_malformed_success():
    for body in [{}, [], ValueError('bad json'), {'fingerprint':'wrong'}]:
        client = embedding_client(Session([Response(body)]))
        with pytest.raises(WeMMError):
            client.encode(text='x')
    session = Session([Response({'fingerprint':FINGERPRINT}), Response({'fingerprint':'b'*64})])
    client = embedding_client(session)
    client._request('/api/internal/tokenize', {'text':'x'})
    with pytest.raises(WeMMError) as exc:
        client._request('/api/internal/tokenize', {'text':'y'})
    assert exc.value.code == 'fingerprint_mismatch'


def test_client_truncates_timeout_to_question_deadline(tmp_path):
    qwen = QwenClient(sdk=object(), ledger=BudgetLedger(tmp_path/'budget.sqlite'))
    session = Session([Response({'fingerprint':FINGERPRINT, 'tokens':1, 'chunks':['x'], 'limit':512})])
    client = embedding_client(session)
    with qwen.request_scope(deadline_s=0.2):
        assert client.split_text('x') == ['x']
    assert 0 < session.calls[0][1]['timeout'] <= 0.2
    assert session.calls[0][1]['allow_redirects'] is False


def test_client_timeout_and_backoff_stop_at_scope_deadline(tmp_path):
    qwen = QwenClient(sdk=object(), ledger=BudgetLedger(tmp_path/'budget.sqlite'))
    session = Session([requests.Timeout(), Response({})])
    client = embedding_client(session)
    with qwen.request_scope(deadline_s=0.2):
        with pytest.raises(DeadlineExceeded):
            client.encode(text='x')
    assert len(session.calls) == 1
    with qwen.request_scope() as scope:
        scope.deadline = 0
        with pytest.raises(DeadlineExceeded):
            client.encode(text='x')
    assert len(session.calls) == 1


def test_login_uses_scope_deadline_and_never_redirects(tmp_path):
    key = tmp_path/'access.key';key.write_text('test-only-password')
    qwen = QwenClient(sdk=object(), ledger=BudgetLedger(tmp_path/'budget.sqlite'))
    client = EmbeddingClient(key_path=key)
    session = Session([Response({'ok':True})])
    with qwen.request_scope(deadline_s=0.2):
        client._login(session)
    assert 0 < session.calls[0][1]['timeout'] <= 0.2
    assert session.calls[0][1]['allow_redirects'] is False
