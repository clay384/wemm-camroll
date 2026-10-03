from pathlib import Path
import sys

import pytest
import sqlite3

sys.path.insert(0, str(Path(__file__).parent))
from gate_export import export, gate_overhead, journal_requests
from gate_config import ORDER
from test_gate_suite import complete_run
from camroll_agent.budget import BudgetLedger, RequestJournal
from camroll_eval.common import atomic_json, read_json


def test_partial_export_keeps_unexecuted_denominators_and_all_reserved_calls(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600)
    groups = [{'name': name, 'run_id': name.lower(), 'status': 'not_started'} for name in ORDER]
    suite = {'status': 'blocked', 'groups': groups, 'restoration': {'state': 'restored'}}
    atomic_json(tmp_path / 'formal-suite.json', suite)
    run = complete_run()
    run.update(run_id='f37-off', status='blocked', inference_config={'group': 'F37-Off'},
               summary={'quality_distribution_planned': {'correct': 0, 'partial': 0, 'incorrect': 0, 'unscored': 100}})
    directory = tmp_path / 'evaluation/runs/f37-off'
    atomic_json(directory / 'run.json', run)
    before = (directory / 'run.json').read_bytes()
    journal = RequestJournal(directory / 'request-journal.sqlite')
    journal.claim('interrupted-before-item-save', 'body', lambda: ledger.reserve(.25, {'model': 'qwen3.7-flash'}))
    report = export(tmp_path, ledger_path=ledger.path)
    assert report['acceptance_complete'] is False and report['status'] == 'incomplete'
    assert report['groups'][0]['cost_audit']['own_reserved_cny'] == .25
    assert report['groups'][0]['cost_audit']['journal_statuses'] == {'pending': 1}
    assert report['groups'][1]['summary']['quality_distribution_planned']['unscored'] == 100
    assert report['groups'][1]['summary']['unexecuted_turn_count'] == 111
    assert report['groups'][1]['summary']['retrieval_metric_count'] == 81
    assert (directory / 'run.json').read_bytes() == before
    assert (tmp_path / 'reports/flash-gate.md').is_file()
    assert read_json(tmp_path / 'evaluation/cost-audit.json')['runs']['f37-off']['own_reserved_cny'] == .25


def test_extra_cost_excludes_initial_and_final_judge_and_deduplicates_verdict_reuse():
    run = {'items': [{'gate_events': [
        {'state': 'sending', 'attempt': 2, 'semantic_request_id': 'repeat'},
        {'state': 'rejected', 'attempt': 2, 'call_id': 'extra'},
        {'state': 'evidence_checked', 'verifier_call_ids': ['fact']},
        {'state': 'evidence_checked', 'verifier_call_ids': ['fact'], 'reused_verdict': True}]}]}
    def row(role, amount): return {'metadata': {'role': role, 'prompt_tokens': 10, 'completion_tokens': 5},
                                  'charged_units': amount, 'reserved_units': 0, 'status': 'settled'}
    snapshot = {'status': 'available', 'rows': {'first': row('agent', 100), 'extra': row('agent', 200),
                  'fact': row('evidence_gate', 300), 'score': row('judge', 400)}}
    calls = [{'call_id': 'first'}, {'call_id': 'extra', 'semantic_request_id': 'repeat'}]
    result = gate_overhead(run, calls, snapshot)
    assert result['combined']['known_estimated_cny'] == .0005
    assert result['combined']['prompt_tokens'] == 20
    assert result['fact_verifier']['request_count'] == 1
    assert result['mapping_complete'] is True


@pytest.mark.parametrize('request_state', ['missing', 'corrupt', 'pending', 'uncertain', 'completed', 'failed',
                                         'ledger_missing', 'ledger_pending_zero', 'ledger_status_invalid'])
def test_delivery_requires_complete_attribution_and_terminal_settlement(tmp_path, request_state):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600)
    groups = [{'name': name, 'run_id': name.lower(), 'status': 'completed', 'returncode': 0} for name in ORDER]
    atomic_json(tmp_path / 'formal-suite.json', {'status': 'passed', 'groups': groups,
        'restoration': {'state': 'restored'}, 'model_files_unchanged': True, 'frozen_inputs_unchanged': True})
    originals = {}
    for group in groups:
        directory = tmp_path / 'evaluation/runs' / group['run_id']
        run = complete_run()
        run.update(run_id=group['run_id'], inference_config={'group': group['name']}, summary={})
        atomic_json(directory / 'run.json', run)
        originals[directory / 'run.json'] = (directory / 'run.json').read_bytes()
        if request_state == 'missing':
            continue
        journal_path = directory / 'request-journal.sqlite'
        if request_state == 'corrupt':
            journal_path.write_bytes(b'not a sqlite database')
            continue
        journal = RequestJournal(journal_path)
        request_id = group['run_id'] + ':request'
        reserve = 0 if request_state == 'ledger_pending_zero' else .25
        claim = journal.claim(request_id, 'body', lambda: ledger.reserve(reserve, {'model': 'qwen3.7-flash'}))
        if request_state in {'completed', 'ledger_missing', 'ledger_status_invalid'}:
            journal.complete(request_id, {'synthetic': True})
            ledger.finish(claim['call_id'], .1)
            if request_state == 'ledger_status_invalid':
                with sqlite3.connect(ledger.path) as connection:
                    connection.execute("UPDATE budget_calls SET status='unrecognized' WHERE id=?", (claim['call_id'],))
        elif request_state == 'ledger_pending_zero':
            journal.complete(request_id, {'synthetic': True})
        elif request_state == 'failed':
            journal.fail(request_id, 'http_403')
            ledger.finish(claim['call_id'], 0, rejected=True)
        elif request_state == 'uncertain':
            journal.fail(request_id, 'request_uncertain', uncertain=True)
            ledger.finish(claim['call_id'], None)
    report = export(tmp_path, ledger_path=tmp_path / 'absent.sqlite' if request_state == 'ledger_missing' else ledger.path)
    expected = request_state in {'completed', 'failed'}
    assert report['execution_complete'] is True
    assert report['delivery_complete'] is report['acceptance_complete'] is expected
    assert all(group['execution_complete'] for group in report['groups'])
    assert all(group['delivery_complete'] is expected for group in report['groups'])
    assert all(pair['complete_pair'] is expected for pair in report['paired_comparisons'])
    if request_state in {'pending', 'uncertain'}:
        assert all(group['cost_audit']['complete'] is True for group in report['groups'])
        assert all(group['cost_audit']['settlement_complete'] is False for group in report['groups'])
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_unopenable_journal_is_reported_without_crashing_or_creating_a_database(tmp_path, monkeypatch):
    path = tmp_path / 'journal.sqlite'
    path.write_bytes(b'existing private journal')
    def denied(*args, **kwargs):
        assert '?mode=ro' in args[0]
        raise sqlite3.OperationalError('unable to open database')
    monkeypatch.setattr(sqlite3, 'connect', denied)
    records, issues = journal_requests(path)
    assert records == [] and issues == [{'kind': 'request_journal_unreadable'}]
    assert path.read_bytes() == b'existing private journal'
