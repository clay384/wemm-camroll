"""Explicit continuation keeps an old unknown charge held; only synthetic data."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from camroll_agent.budget import (BudgetExceeded, BudgetLedger, BudgetLimitChanged, RequestJournal,
                                 RequestUncertain, UncertainRequestAcknowledgement)
from camroll_agent.evidence_gate import EvidenceGateVerifier
from camroll_agent.qwen import QwenClient
from camroll_eval.judge import Judge


@pytest.fixture
def held(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600, expected_limit_cny=600)
    journal = RequestJournal(tmp_path / 'original-journal.sqlite')
    request_id = 'original-run:unknown-request'
    payload_hash = hashlib.sha256(b'synthetic original payload').hexdigest()
    claim = journal.claim(request_id, payload_hash, lambda: ledger.reserve(1.268122,
        {'request_id': request_id, 'role': 'agent', 'model': 'qwen3.7-flash'}))
    ledger.finish(claim['call_id'], None, {'error_code': 'transport_error'})
    journal.fail(request_id, 'transport_error', uncertain=True)
    ack = UncertainRequestAcknowledgement.capture(ledger.path, journal.path, call_id=claim['call_id'],
        request_id=request_id, payload_hash=payload_hash, reserved_units=1268122)
    return ledger, journal, ack


def scoped(held):
    ledger, _, ack = held
    return BudgetLedger(ledger.path, expected_limit_cny=600, acknowledged_uncertain=(ack,))


def old_rows(held):
    ledger, journal, ack = held
    with sqlite3.connect(ledger.path) as db:
        one = db.execute('SELECT * FROM budget_calls WHERE id=?', (ack.call_id,)).fetchone()
    with sqlite3.connect(journal.path) as db:
        two = db.execute('SELECT * FROM cloud_requests WHERE request_id=?', (ack.request_id,)).fetchone()
    return one, two


def test_explicit_scope_preserves_original_unknown_and_default_strict_behavior(held):
    original = old_rows(held)
    ledger, _, ack = held
    with pytest.raises(RequestUncertain):
        ledger.reserve(.01)
    continuing = scoped(held)
    summary = continuing.summary()
    assert summary['limit_cny'] == 600
    assert summary['uncertain_requests'] == summary['acknowledged_uncertain_requests'] == 1
    assert summary['blocking_uncertain_requests'] == 0
    assert summary['reserved_cny'] == summary['acknowledged_reserved_cny'] == 1.268122
    new_call = continuing.reserve(.05)
    continuing.finish(new_call, .02)
    assert continuing.summary()['remaining_cny'] == pytest.approx(600 - 1.268122 - .02)
    assert old_rows(held) == original
    with pytest.raises(RequestUncertain):
        BudgetLedger(ledger.path, expected_limit_cny=600).reserve(.01)
    assert ledger.summary()['blocking_uncertain_requests'] == 1
    assert ledger.summary()['acknowledged_uncertain_requests'] == 0
    assert continuing.with_expected_limit(600).acknowledged_uncertain == (ack,)


def test_acknowledgement_roundtrip_is_immutable_and_rejects_changed_contract(held):
    ack = held[2]
    assert UncertainRequestAcknowledgement.from_dict(json.loads(json.dumps(ack.as_dict()))) == ack
    with pytest.raises(FrozenInstanceError):
        ack.reserved_units = 0
    for bad in ({**ack.as_dict(), 'auto_discover': True}, {k: v for k, v in ack.as_dict().items() if k != 'payload_hash'}):
        with pytest.raises(ValueError):
            UncertainRequestAcknowledgement.from_dict(bad)
    with pytest.raises(ValueError):
        BudgetLedger(held[0].path, expected_limit_cny=600, acknowledged_uncertain=(ack, ack))
    with pytest.raises(ValueError):
        BudgetLedger(held[0].path, expected_limit_cny=300, acknowledged_uncertain=(ack,))


@pytest.mark.parametrize('changed', ['call_id', 'request_id', 'payload_hash', 'reserved_units'])
def test_capture_requires_all_explicit_authorized_identity_fields(held, changed):
    ledger, journal, ack = held
    fields = {k: getattr(ack, k) for k in ('call_id', 'request_id', 'payload_hash', 'reserved_units')}
    fields[changed] = 1 if changed == 'reserved_units' else '0' * 64
    with pytest.raises(RequestUncertain):
        UncertainRequestAcknowledgement.capture(ledger.path, journal.path, **fields)


@pytest.mark.parametrize('target,statement,value', [
    ('ledger', 'UPDATE budget_calls SET reserved_units=? WHERE id=?', 1),
    ('ledger', 'UPDATE budget_calls SET charged_units=? WHERE id=?', 1),
    ('ledger', 'UPDATE budget_calls SET created=? WHERE id=?', 0),
    ('ledger', 'UPDATE budget_calls SET status=? WHERE id=?', 'settled'),
    ('ledger', 'UPDATE budget_calls SET metadata=? WHERE id=?', '{}'),
    ('ledger', 'UPDATE budget_calls SET metadata=? WHERE id=?', '[]'),
    ('journal', 'UPDATE cloud_requests SET payload_hash=? WHERE request_id=?', '0' * 64),
    ('journal', 'UPDATE cloud_requests SET status=? WHERE request_id=?', 'pending'),
    ('journal', 'UPDATE cloud_requests SET call_id=? WHERE request_id=?', 'other-call'),
    ('journal', 'UPDATE cloud_requests SET error_code=? WHERE request_id=?', 'other'),
    ('journal', 'UPDATE cloud_requests SET response=? WHERE request_id=?', '{}'),
])
def test_changed_old_evidence_blocks_next_reservation_and_cannot_be_hidden_in_summary(held, target, statement, value):
    continuing = scoped(held)
    ledger, journal, ack = held
    path = ledger.path if target == 'ledger' else journal.path
    with sqlite3.connect(path) as db:
        db.execute(statement, (value, ack.call_id if target == 'ledger' else ack.request_id))
    for operation in (lambda: continuing.reserve(.01), continuing.summary):
        with pytest.raises(RequestUncertain):
            operation()
    with sqlite3.connect(ledger.path) as db:
        assert db.execute('SELECT COUNT(*) FROM budget_calls').fetchone()[0] == 1


def test_acknowledgement_is_bound_to_ledger_path_and_file_identity(held, tmp_path):
    ledger, _, ack = held
    copy_path = tmp_path / 'different-budget.sqlite'
    with sqlite3.connect(ledger.path) as src, sqlite3.connect(copy_path) as dest:
        src.backup(dest)
    with pytest.raises(RequestUncertain):
        BudgetLedger(copy_path, expected_limit_cny=600, acknowledged_uncertain=(ack,))
    continuing = scoped(held)
    copy_path.replace(ledger.path)
    with pytest.raises(RequestUncertain):
        continuing.reserve(.01)


def test_replaced_or_missing_old_journal_fails_closed(held, tmp_path):
    continuing = scoped(held)
    _, journal, _ = held
    replacement = tmp_path / 'replacement.sqlite'
    with sqlite3.connect(journal.path) as src, sqlite3.connect(replacement) as dest:
        src.backup(dest)
    replacement.replace(journal.path)
    with pytest.raises(RequestUncertain):
        continuing.reserve(.01)
    journal.path.unlink()
    with pytest.raises(RequestUncertain):
        continuing.summary()


@pytest.mark.parametrize('amount,rejected', [(None, False), (0, True), (.1, False)])
def test_acknowledgement_never_settles_releases_or_rewrites_old_request(held, amount, rejected):
    before = old_rows(held)
    with pytest.raises(RequestUncertain):
        scoped(held).finish(held[2].call_id, amount, rejected=rejected)
    assert old_rows(held) == before


def test_old_journal_never_replays_or_resends_an_acknowledged_unknown(held):
    before = old_rows(held)
    _, journal, ack = held
    continuing = scoped(held)
    with pytest.raises(RequestUncertain):
        journal.claim(ack.request_id, ack.payload_hash, lambda: continuing.reserve(.1))
    assert old_rows(held) == before
    assert continuing.summary()['requests'] == 1


def test_new_unknown_stops_all_scoped_clients_without_losing_old_hold(held):
    one, two = scoped(held), scoped(held)
    new_call = one.reserve(.5)
    one.finish(new_call, None, {'error_code': 'transport_error'})
    summary = two.summary()
    assert summary['uncertain_requests'] == 2 and summary['acknowledged_uncertain_requests'] == 1
    assert summary['blocking_uncertain_requests'] == 1 and summary['reserved_cny'] == 1.768122
    with pytest.raises(RequestUncertain):
        two.reserve(.01)
    assert two.summary()['requests'] == 2


def test_concurrent_reservations_include_the_old_hold_and_enforce_original_600_cap(held):
    def attempt(_):
        try:
            return scoped(held).reserve(300)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(value is not None for value in results) == 1
    assert scoped(held).summary()['reserved_cny'] == 301.268122
    with sqlite3.connect(held[0].path) as db:
        db.execute("UPDATE budget_settings SET value=300000000 WHERE key='limit_units'")
    with pytest.raises(BudgetLimitChanged):
        scoped(held).reserve(.01)


def test_qwen_fact_verifier_and_judge_keep_acknowledgement_scope(held, tmp_path):
    before = old_rows(held)
    sdk_calls, fact_calls, judge_calls = [], [], []
    def qwen_response(**body):
        sdk_calls.append(body)
        return {'model': 'qwen3.7-flash', 'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'synthetic'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=qwen_response)))
    client = QwenClient(sdk=sdk, ledger=scoped(held), expected_limit_cny=600, request_namespace='new-run',
                        request_journal_path=tmp_path / 'new-run.sqlite')
    client.chat([{'role': 'user', 'content': 'synthetic new request'}], request_id='new-decision')
    def fact_response(body, *, timeout):
        fact_calls.append(body)
        return {'model': 'doubao-seed-2-0-lite-260215', 'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
            'content': json.dumps({'status': 'insufficient', 'evidence_ids': [], 'reasons': [
                {'claim': 'synthetic', 'assessment': 'insufficient', 'evidence_ids': [], 'reason': 'No primary evidence supplied.'}]})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    fact = EvidenceGateVerifier(client=client, transport=fact_response)
    with client.request_scope():
        assert fact.verify(question='synthetic', candidate_answer='unknown', outcome='refuse', images=[],
            request_id='new-fact')['status'] == 'insufficient'
    def judge_response(body):
        judge_calls.append(body)
        return {'model': 'doubao-seed-2-0-lite', 'choices': [{'message': {'content': '{"score":2,"reason":"synthetic"}'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    judge = Judge(ledger=scoped(held), transport=judge_response, interval=0, expected_limit_cny=600,
                  request_namespace='new-run', request_journal_path=tmp_path / 'new-run.sqlite')
    with judge.request_context('new-score'):
        judge._judge_request({'model': 'doubao-seed-2.0-lite', 'messages': [{'role': 'user', 'content': 'synthetic'}], 'max_tokens': 64})
    for ledger in (client.ledger, fact.ledger, judge.ledger):
        assert ledger.acknowledged_uncertain == (held[2],)
        assert ledger.summary()['acknowledged_uncertain_requests'] == 1
    assert len(sdk_calls) == len(fact_calls) == len(judge_calls) == 1
    assert old_rows(held) == before
    assert client.ledger.summary()['reserved_cny'] == 1.268122
    extra = client.ledger.reserve(.1)
    client.ledger.finish(extra, None, {'error_code': 'transport_error'})
    with pytest.raises(RequestUncertain):
        client.chat([{'role': 'user', 'content': 'must stop'}], request_id='another-decision')
    with client.request_scope(), pytest.raises(RequestUncertain):
        fact.verify(question='synthetic', candidate_answer='unknown', outcome='refuse', images=[], request_id='another-fact')
    with judge.request_context('another-score'), pytest.raises(RequestUncertain):
        judge._judge_request({'model': 'doubao-seed-2.0-lite', 'messages': [{'role': 'user', 'content': 'must stop'}], 'max_tokens': 64})
    assert len(sdk_calls) == len(fact_calls) == len(judge_calls) == 1
