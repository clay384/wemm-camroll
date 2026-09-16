"""Protocol, failure accounting, isolation and auth tests; no cloud calls."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import threading
import urllib.error

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json, sha256, read_json
from camroll_eval.dataset import Dataset
from camroll_eval.metrics import initial_item, add_media_metrics, summary
from camroll_eval.judge import Judge
from camroll_eval.repository import Repository
from camroll_eval.runner import Runner
from camroll_eval.service import create_app
from camroll_agent.budget import BudgetLedger, BudgetExceeded


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'photos').mkdir(parents=True)
    (root / 'faces').mkdir()
    (root / 'qa').mkdir()
    (root / 'photos/a.jpg').write_bytes(b'photo-a')
    (root / 'photos/b.jpg').write_bytes(b'photo-b')
    (root / 'faces/ref.jpg').write_bytes(b'reference')
    atomic_json(root / 'manifest.json', {'album_id': 'synthetic', 'photos': ['photos/a.jpg', 'photos/b.jpg'],
        'faces': [{'face_id': '1', 'canonical_name': 'Person', 'aliases': ['Me'], 'family_role': 'self', 'ref_image': 'faces/ref.jpg'}],
        'family_relationships': [], 'qa_sets': {'100qa-full': 'qa/questions.jsonl'}})
    rows = [{'qa_id': 'one', 'question': 'find photo', 'answer': 'photo a', 'retrieval_image_ids': ['a.jpg'], 'answerability': 'answerable', 'expected_action': 'answer'},
            {'qa_id': 'two', 'question': 'which event', 'answer': 'photo b', 'retrieval_image_ids': ['b.jpg'], 'answerability': 'answerable', 'expected_action': 'answer',
             'conversation': [{'message': 'which event', 'expected_action': 'clarify', 'reference_answer': 'which day'}, {'message': 'that day', 'expected_action': 'answer', 'reference_answer': 'photo b'}]}]
    (root / 'qa/questions.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    return root


class Backend:
    def __init__(self):
        self.questions = []
        self.cancel_after = None
        self.ids = []
        self.error = False
    def budget(self):
        return {'estimated_spent_cny': 0, 'remaining_cny': 300}
    def build(self, spec, memory, progress, cancel):
        data = read_json(spec)
        root = Path(data['root_folder'])
        memory = Path(memory)
        rows = []
        for image in data['images']:
            path = root / image['path']
            digest = sha256(path)
            rows.append({'id': 'img_' + digest, 'asset_id': digest, 'filename': path.name})
        self.ids = [r['id'] for r in rows]
        atomic_json(memory / 'snapshots/version/records.json', {'images': rows, 'metadata': {'image_embedding_count': len(rows)}})
        (memory / 'CURRENT').write_text('version\n')
        return {'published': True, 'index_version': 'version', 'stages': []}
    def agent(self, memory, config):
        backend = self
        if not self.ids:
            self.ids = [r['id'] for r in read_json(Path(memory) / 'snapshots/version/records.json')['images']]
        class FakeAgent:
            def ask(self, question, history=None, **kwargs):
                backend.questions.append((question, copy.deepcopy(history)))
                if backend.cancel_after:
                    backend.cancel_after(question)
                index = 1 if question == 'that day' else 0
                return {'final_text': 'which day' if question == 'which event' else 'photo',
                        'turn_outcome': 'error' if backend.error else 'clarify' if question == 'which event' else 'answer',
                        'stopped_reason': 'timeout' if backend.error else 'completed',
                        'index_version': 'version', 'selected_image_ids': [backend.ids[index]],
                        'retrieved_image_ids': backend.ids, 'evidence_image_ids': [backend.ids[index]],
                        'conversation_history': (history or []) + [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': 'photo'}],
                        'model_call_metrics': [{'call_id': 'c1', 'call_type': 'agent', 'duration_ms': 100, 'parse_status': 'valid', 'prompt_tokens': 100, 'completion_tokens': 20}],
                        'tool_trace': [{'tool': 'search', 'model_call_id': 'c1', 'duration_ms': 30, 'status': 'success'}]}
        return FakeAgent()


class FakeJudge:
    stop_reason = None
    def __init__(self, **kwargs):
        self.cancel = kwargs.get('cancel')
    def protocol(self):
        return {'model': 'fake-judge', 'prompts': 'fixed'}
    def score_item(self, item, row):
        for record in item['runtime_turns']:
            record['judge'] = {'score': 2, 'reason': 'correct'}
            record['task_judge'] = {'expected_action': record['expected_action'], 'actual_action': record['expected_action'], 'correct': True}
        item['judge'] = item['runtime_turns'][-1]['judge']
        item['task_judges'] = [r['task_judge'] for r in item['runtime_turns']]
        item['judge_status'] = 'completed'
        return item


def runner(tmp_path, source, backend=None):
    backend = backend or Backend()
    return Runner(tmp_path / 'eval', source, backend_factory=lambda: backend,
                  judge_factory=FakeJudge, sample_resources=False), backend


def test_build_spec_excludes_qa_answers_and_photo_annotations(source):
    dataset = Dataset(source, strict=False)
    spec = dataset.build_spec('new-run')
    assert spec['input_policy'] == 'originals_and_identity_references'
    assert spec['images'] == [{'path': 'photos/a.jpg'}, {'path': 'photos/b.jpg'}]
    assert 'question' not in json.dumps(spec) and 'photo a' not in json.dumps(spec)
    assert spec['identity']['persons'][0]['person_id'] == 'person_1'
    assert dataset.integrity()['files_checked'] == 5
    with pytest.raises(ValueError):
        Dataset(source)


def test_full_runner_multiturn_media_and_model_tool_binding(tmp_path, source):
    engine, backend = runner(tmp_path, source)
    result = engine.run(run_id='test', strict=False, qa_concurrency=2, judge_concurrency=2)
    assert result['status'] == 'completed'
    assert result['summary']['judge_valid_count'] == 2
    assert result['summary']['task_decision_valid_count'] == 3
    assert result['summary']['retrieval_recall_macro'] == 1
    assert len(next(h for q, h in backend.questions if q == 'that day')) == 2
    first = result['items'][0]
    assert first['model_call_metrics'][0]['agent_loop_total_ms'] == 130
    assert first['timing_breakdown']['tool_ms'] == 30
    assert first['agent_stability']['native_protocol_valid_count'] == 1
    assert first['agent_stability']['json_parse_total'] is None
    assert (tmp_path / 'eval/runs/test/results.jsonl').is_file()


def test_explicit_error_is_counted_and_never_auto_retried(tmp_path, source):
    engine, backend = runner(tmp_path, source)
    backend.error = True
    result = engine.run(run_id='error', strict=False, qa_concurrency=1, judge_concurrency=1)
    assert result['status'] == 'completed_with_errors'
    assert result['summary']['execution_error_count'] == 2
    assert all(row['execution_status'] == 'error' for row in result['items'])
    assert len(backend.questions) == 2  # No invented second-round request after failure.


def test_cancelled_dialogue_resume_keeps_paid_first_turn(tmp_path, source):
    engine, backend = runner(tmp_path, source)
    backend.cancel_after = lambda q: engine.cancel.set() if q == 'which event' else None
    result = engine.run(run_id='resume', strict=False, qa_concurrency=1, judge_concurrency=1)
    assert result['status'] == 'cancelled'
    partial = next(r for r in result['items'] if r['qa_id'] == 'two')
    assert partial['execution_status'] == 'cancelled' and len(partial['runtime_turns']) == 1
    backend.cancel_after = None
    engine2, _ = runner(tmp_path, source, backend)
    resumed = engine2.run(run_id='resume', strict=False, qa_concurrency=1, judge_concurrency=1, resume=True)
    assert resumed['status'] == 'completed'
    assert [q for q, h in backend.questions].count('which event') == 1
    assert [q for q, h in backend.questions].count('that day') == 1
    assert resumed['timing_comparable'] is False


def test_resume_rejects_code_change_without_mutating_saved_run(tmp_path, source, monkeypatch):
    engine, backend = runner(tmp_path, source)
    backend.cancel_after = lambda q: engine.cancel.set()
    engine.run(run_id='locked', strict=False, qa_concurrency=1, judge_concurrency=1)
    path = tmp_path / 'eval/runs/locked/run.json'
    digest = sha256(path)
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', 'different')
    new, _ = runner(tmp_path, source, backend)
    with pytest.raises(ValueError, match='revision'):
        new.run(run_id='locked', strict=False, qa_concurrency=1, judge_concurrency=1, resume=True)
    assert sha256(path) == digest


def test_completed_run_cannot_be_resumed_or_overwritten(tmp_path, source):
    engine, _ = runner(tmp_path, source)
    engine.run(run_id='immutable', strict=False, qa_concurrency=1, judge_concurrency=1)
    path = tmp_path / 'eval/runs/immutable/run.json'
    digest = sha256(path)
    new, _ = runner(tmp_path, source)
    with pytest.raises(ValueError, match='immutable'):
        new.run(run_id='immutable', strict=False, qa_concurrency=1, judge_concurrency=1, resume=True)
    assert sha256(path) == digest


def test_denominators_and_unknown_delivery_do_not_disappear():
    items = []
    for index in range(100):
        row = {'qa_id': str(index), 'question': 'question', 'answer': 'answer', 'retrieval_image_ids': ['a.jpg'],
               'answerability': 'unanswerable' if index < 19 else 'answerable'}
        item = initial_item(row)
        item['judge'] = {'score': [0,1,2][index % 3], 'reason': 'result'}
        item['task_judges'] = [{'expected_action': 'answer', 'actual_action': 'answer', 'correct': True}] * (2 if index < 11 else 1)
        items.append(item)
    result = summary({'system_id': 'camroll_wemm', 'qa_count': 100, 'items': items})
    assert result['retrieval_metric_count'] == 81 and result['judge_valid_count'] == 100
    assert result['task_decision_valid_count'] == 111 and result['retrieval_recall_macro'] == 0
    add_media_metrics(items[-1], {'selected_image_ids': ['unknown']}, {})
    assert items[-1]['media_retrieval_counts']['predicted'] == 1 and items[-1]['mapping_error_count'] == 1


def test_invalid_judge_output_is_not_completed_success():
    judge = Judge(transport=lambda p: {'choices': [{'message': {'content': 'invalid'}}]}, interval=0)
    row = {'answer': 'a'}
    item = {'runtime_turns': [{'message': 'q', 'answer': 'a', 'expected_action': 'answer'}]}
    judge.score_item(item, row)
    assert item['judge']['score'] is None and item['judge_status'] == 'failed'


def test_unexecuted_final_turn_never_uses_clarification_quality():
    judge = Judge(transport=lambda p: None, interval=0)
    judge._judge = lambda *a, **k: {'score': 2, 'judge_status': 'completed'}
    judge._judge_task_action = lambda *a, **k: {'expected_action': 'clarify', 'actual_action': 'clarify', 'correct': True}
    row = {'question': 'which event', 'answer': 'the event', 'conversation': [
        {'message': 'which event', 'expected_action': 'clarify'},
        {'message': 'that day', 'expected_action': 'answer'}]}
    item = initial_item(row)
    item['runtime_turns'] = [{'message': 'which event', 'answer': 'which day?', 'expected_action': 'clarify', 'agent_status': 'error'}]
    judge.score_item(item, row)
    assert item['runtime_turns'][0]['judge']['score'] == 2
    assert item['task_judges'][0]['correct'] is True
    assert item['judge']['score'] is None and item['judge_status'] == 'incomplete'
    assert item['judge']['reason'] == 'final_turn_not_executed'
    result = summary({'system_id': 'camroll_wemm', 'qa_count': 1, 'planned_turn_count': 2, 'items': [item]})
    assert result['judge_valid_count'] == 0 and result['judge_invalid_count'] == 1
    assert result['task_decision_valid_count'] == result['task_decision_labeled_count'] == 1
    assert result['task_decision_planned_count'] == 2 and result['task_decision_unjudged_count'] == 1
    assert result['unexecuted_turn_count'] == 1


def test_judge_retries_account_each_uncertain_call(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite')
    attempts = []
    class NoWait:
        def wait(self, delay):
            return False
        def is_set(self):
            return False
    def transport(payload):
        attempts.append(payload)
        if len(attempts) < 3:
            raise TimeoutError()
        return {'choices': [{'message': {'content': '{"score":2,"reason":"correct"}'}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}
    judge = Judge(transport=transport, ledger=ledger, cancel=NoWait(), interval=0)
    result = judge._judge('q', 'a', 'a')
    assert result['score'] == 2
    assert ledger.summary()['requests'] == 3 and ledger.summary()['uncertain_requests'] == 2
    assert ledger.summary()['estimated_spent_cny'] == pytest.approx(.000288)


def test_explicit_rejection_releases_budget_and_missing_usage_retains_it(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite')
    def bad(payload):
        raise urllib.error.HTTPError('https://invalid', 401, 'secret must not appear', {}, None)
    judge = Judge(transport=bad, ledger=ledger, interval=0)
    value = judge._judge('q','a','a')
    assert 'secret must not appear' not in json.dumps(value)
    assert ledger.summary()['reserved_cny'] == 0
    judge.transport = lambda p: {'choices': [{'message': {'content': '{"score":2,"reason":"correct"}'}}]}
    judge._judge('q','a','a')
    assert ledger.summary()['uncertain_requests'] == 1


def test_budget_exhaustion_stops_without_network(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', .1)
    calls = []
    judge = Judge(transport=lambda p: calls.append(p), ledger=ledger, interval=0)
    result = judge._judge('q','a','a')
    assert result['score'] is None and calls == []
    assert judge.stop_reason == 'budget_exhausted' and judge._cancel.is_set()


def test_auth_all_data_paths_and_same_origin_write_gate(tmp_path, source):
    app = create_app(tmp_path / 'eval', source, access_key='test-key', mount_agent=False)
    client = TestClient(app)
    assert client.get('/healthz').status_code == 200
    for path in ['/api/health','/api/runs','/api/media/a.jpg','/api/runs/no/export','/api/runs/no/pipeline-calls']:
        assert client.get(path).status_code == 401
    assert client.post('/api/login', json={'access_key':'test-key'}).status_code == 403
    assert client.post('/api/login', headers={'Origin':'http://evil.invalid'}, json={'access_key':'test-key'}).status_code == 403
    response = client.post('/api/login', headers={'Origin':'http://testserver'}, json={'access_key':'test-key'})
    assert response.status_code == 200
    assert 'httponly' in response.headers['set-cookie'].lower() and 'samesite=strict' in response.headers['set-cookie'].lower()
    assert client.get('/api/runs').status_code == 200
    assert client.get('/api/media/..%2Fmanifest.json').status_code == 404
    assert client.post('/api/runs', json={}).status_code == 403
    assert client.post('/api/runs', json={}, headers={'Origin':'http://testserver'}).status_code == 422  # Formal source gate.


def test_missing_access_key_fails_closed(tmp_path, source, monkeypatch):
    monkeypatch.delenv('CAMROLL_EVAL_ACCESS_KEY', raising=False)
    monkeypatch.delenv('CAMROLL_EVAL_ACCESS_KEY_FILE', raising=False)
    with pytest.raises(ValueError, match='not configured'):
        create_app(tmp_path/'eval', source, mount_agent=False)


def test_frozen_real_history_replay_when_fixture_supplied():
    fixture = os.getenv('PHOTOBENCH_HISTORY_FIXTURE')
    if not fixture:
        pytest.skip('Private fixture remains outside Git; set PHOTOBENCH_HISTORY_FIXTURE for live parity check')
    state = read_json(Path(fixture) / 'run.json')
    state['system_id'] = 'sentrix'
    result = summary(state)
    assert result['judge_distribution'] == {'0':42,'1':19,'2':39}
    assert result['retrieval_precision_macro'] == .185
    assert result['retrieval_recall_macro'] == .129
    assert result['retrieval_f1_macro'] == .141
    assert result['judge_valid_count'] == 100 and result['retrieval_metric_count'] == 81
    assert result['task_decision_valid_count'] == 111
    assert result['benchmark_e2e_latency_excluding_judge_ms'] == 1516744.4
