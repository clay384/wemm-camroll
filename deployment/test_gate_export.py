from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from gate_export import export, gate_overhead
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
