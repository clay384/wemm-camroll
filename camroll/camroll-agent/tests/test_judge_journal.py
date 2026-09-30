import json
import sqlite3
import urllib.error

import pytest

from camroll_agent.budget import BudgetLedger
from camroll_eval.common import object_hash
from camroll_eval.judge import Judge


PAYLOAD = {'model': 'doubao-seed-2.0-lite', 'messages': [{'role': 'user', 'content': 'synthetic'}],
           'max_tokens': 64, 'temperature': 0}
RESPONSE = {'model': 'doubao-seed-2.0-lite', 'choices': [{'message': {'content': '{"score":2,"reason":"ok"}'}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}


def make_judge(tmp_path, transport, **kwargs):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600, expected_limit_cny=600)
    return Judge(ledger=ledger, transport=transport, interval=0, request_namespace='run',
                 request_journal_path=tmp_path / 'requests.sqlite', expected_limit_cny=600, **kwargs)


def test_rejected_429_and_completed_attempt_replay_without_resending(tmp_path, monkeypatch):
    calls = []
    def transport(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise urllib.error.HTTPError('https://invalid', 429, 'rate limit', {}, None)
        return RESPONSE
    original = make_judge(tmp_path, transport)
    monkeypatch.setattr(original._cancel, 'wait', lambda seconds: False)
    with original.request_context('item'):
        result, attempt = original._judge_request(PAYLOAD)
    assert attempt == 2 and result == RESPONSE
    restarted = make_judge(tmp_path, lambda payload: pytest.fail('Completed request must not resend'))
    with restarted.request_context('item'):
        replayed, attempt = restarted._judge_request(PAYLOAD)
    assert replayed == RESPONSE and attempt == 2
    assert len(calls) == 2
    assert restarted.ledger.summary()['requests'] == 2
    assert restarted.ledger.summary()['estimated_spent_cny'] == pytest.approx(.000288)


def test_missing_usage_returns_no_successful_score_and_blocks_other_clients(tmp_path):
    calls = []
    def transport(payload):
        calls.append(payload)
        return {key: value for key, value in RESPONSE.items() if key != 'usage'}
    original = make_judge(tmp_path, transport)
    with original.request_context('item'):
        score = original._judge('q', 'a', 'a')
    assert score['score'] is None
    assert original.stop_reason == 'request_uncertain'
    assert original.ledger.summary()['uncertain_requests'] == 1
    restarted = make_judge(tmp_path, lambda payload: pytest.fail('Uncertain requests must not resend'))
    with restarted.request_context('another-item'):
        score = restarted._judge('q2', 'a2', 'a2')
    assert score['score'] is None
    assert restarted.stop_reason == 'request_uncertain'
    assert len(calls) == 1
    with sqlite3.connect(tmp_path / 'requests.sqlite') as connection:
        status, raw = connection.execute('SELECT status,response FROM cloud_requests').fetchone()
    assert status == 'uncertain' and json.loads(raw)['choices']


def test_completed_response_replay_settles_previously_interrupted_ledger(tmp_path):
    judge = make_judge(tmp_path, lambda payload: pytest.fail('Persisted request must not resend'))
    request_id = 'run:judge:item:1:1'
    claim = judge.journal.claim(request_id, object_hash(PAYLOAD), lambda: judge.ledger.reserve(1))
    judge.journal.complete(request_id, RESPONSE)
    assert judge.ledger.summary()['reserved_cny'] == 1
    with judge.request_context('item'):
        response, _ = judge._judge_request(PAYLOAD)
    assert response == RESPONSE
    assert judge.ledger.summary()['reserved_cny'] == 0
    assert judge.ledger.summary()['estimated_spent_cny'] == pytest.approx(.000288)
    assert judge.ledger.summary()['requests'] == 1


def test_success_event_keeps_actual_response_model(tmp_path):
    events = []
    actual = dict(RESPONSE, model='doubao-seed-2.0-lite-provider-snapshot')
    judge = make_judge(tmp_path, lambda payload: actual, on_call=events.append)
    with judge.request_context('item'):
        judge._judge_request(PAYLOAD)
    assert events[0]['model'] == 'doubao-seed-2.0-lite'
    assert events[0]['response_model'] == actual['model']


def test_unknown_transport_never_retries_or_replays_after_restart(tmp_path):
    calls = []
    def transport(payload):
        calls.append(payload)
        raise TimeoutError('provider private details')
    judge = make_judge(tmp_path, transport)
    with judge.request_context('item'):
        score = judge._judge('q', 'a', 'a')
    assert score['score'] is None and 'private details' not in str(score)
    assert len(calls) == 1
    restarted = make_judge(tmp_path, lambda payload: pytest.fail('Unknown request must not resend'))
    with restarted.request_context('item'):
        score = restarted._judge('q', 'a', 'a')
    assert score['score'] is None
    assert restarted.ledger.summary()['requests'] == 1
