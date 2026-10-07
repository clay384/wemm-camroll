import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_remaining_export as module
import gate_continuation_export as continuation
from test_gate_continuation_export import build_continuation, formal, add_call
from camroll_agent.budget import BudgetLedger
from camroll_eval.common import atomic_json, read_json, sha256
from camroll_eval.repository import Repository, ComparisonScopeError


@pytest.fixture
def remaining(tmp_path, monkeypatch):
    f, g, h = (tmp_path / name for name in ('f', 'g', 'h'))
    monkeypatch.setattr(continuation, 'PARENT_RUNTIME', f)
    monkeypatch.setattr(module, 'PARENT_RUNTIME', g)
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600)
    old_off = f / 'evaluation/runs/old-off/run.json'
    atomic_json(old_off, formal('F37-Off', 'old-off', 'old-suite'))
    add_call(old_off.parent, ledger, 'old-off')
    build_continuation((f, g, ledger, old_off))
    parent = read_json(g / 'formal-suite.json')
    parent.update(suite_id=module.PARENT_SUITE, status='blocked')
    references = {}
    integrity = {'qa_sha256': 'a' * 64, 'dataset_sha256': 'b' * 64}
    for index, group in enumerate(parent['groups']):
        path = g / 'evaluation/runs' / group['run_id'] / 'run.json'
        run = read_json(path)
        run.update(suite_id=module.PARENT_SUITE, input_integrity=integrity)
        if index == 2:
            run.update(status='blocked', gate_complete=False, stop_reason='context_limit')
            run['items'] = run['items'][:1]
            run['items'][0]['runtime_turns'] = run['items'][0]['runtime_turns'][:1]
            run['items'][0]['runtime_turns'][0].update(answer_origin='system_failure', judge_status='skipped_expected', judge={'score': None})
            group.update(status='blocked', returncode=2, acceptance_complete=False)
        if index == 3:
            group.update(status='not_started', returncode=None, acceptance_complete=None)
        atomic_json(path, run)
        if index < 3:
            references[group['name']] = {'path': str(path), 'sha256': sha256(path)}
    atomic_json(g / 'formal-suite.json', parent)
    atomic_json(g / 'freeze.json', {'acknowledgement': parent['acknowledgement'], 'synthetic': True})
    groups = copy.deepcopy(parent['groups'])
    for row in groups[:3]:
        row.update(reused=True, parent_run_path=references[row['name']]['path'],
                   parent_run_sha256=references[row['name']]['sha256'])
    groups[3].update(run_id='h-f38-on', status='completed_with_errors', returncode=0)
    suite = {'schema_version': 'camroll-flash-gate-remaining-suite-v1', 'suite_id': 'h-suite',
        'execution_order': ['F38-On'], 'status': 'passed', 'groups': groups,
        'acknowledgement': parent['acknowledgement'], 'restoration': {'state': 'restored'},
        'model_files_unchanged': True, 'wemm_weights_unchanged': True, 'frozen_inputs_unchanged': True}
    atomic_json(h / 'formal-suite.json', suite)
    atomic_json(h / 'freeze.json', {'schema_version': 'camroll-flash-gate-remaining-freeze-v1',
        'parent_runs': references, 'acknowledgement': parent['acknowledgement'],
        'parent_suite': {'path': str(g / 'formal-suite.json'), 'sha256': sha256(g / 'formal-suite.json')},
        'parent_freeze': {'path': str(g / 'freeze.json'), 'sha256': sha256(g / 'freeze.json')}})
    run = formal('F38-On', 'h-f38-on', 'h-suite')
    run['input_integrity'] = integrity
    path = h / 'evaluation/runs/h-f38-on/run.json'
    atomic_json(path, run)
    add_call(path.parent, ledger, 'new-h')
    return f, g, h, ledger


def test_only_new_on_is_added_old_blocked_state_and_costs_remain(remaining):
    f, g, h, ledger = remaining
    # SQLite read-only connections may update transient shared-memory locks.
    # Preserve all source payloads, DBs and WALs, as the supervisor does.
    original = {path: path.read_bytes() for path in g.rglob('*')
                if path.is_file() and not path.name.endswith('-shm')}
    result = module.export(h, ledger_path=ledger.path)
    assert result['status'] == 'incomplete' and result['work_complete'] is result['acceptance_complete'] is False
    assert result['remaining_group_complete'] is True and result['financial_settlement_complete'] is False
    assert result['groups'][2]['status'] == 'blocked' and result['groups'][2]['work_complete'] is False
    assert result['comparison_scope_ready'] is True
    assert result['comparison_cost_audit']['request_count'] == 5
    assert result['comparison_cost_audit']['known_estimated_cny'] == .4
    assert result['comparison_cost_audit']['known_reserved_cny'] == .25
    assert len(result['acknowledged_outstanding']) == 1
    assert all(row['source_kind'] == 'read_only_g_reference' for row in result['groups'][:3])
    assert [row['suite_id'] for row in result['groups']] == [module.PARENT_SUITE] * 3 + ['h-suite']
    assert all(path.read_bytes() == data for path, data in original.items())
    assert sorted(path.name for path in (h / 'evaluation/runs').iterdir()) == ['h-f38-on']
    report = Repository(h / 'evaluation', history_roots=[g / 'evaluation', f / 'evaluation'])
    comparison = report.compare([group['run_id'] for group in result['groups']])
    assert comparison['remaining_scope']['source_hashes_verified'] is True
    assert comparison['remaining_scope']['comparison_complete'] is False
    assert comparison['same_qa_hash'] is comparison['same_dataset_hash'] is True
    assert '四组比较尚未完整' in comparison['warning']


@pytest.mark.parametrize('change', ['missing', 'active'])
def test_pending_on_keeps_fixed_denominators_without_cross_suite_permission(remaining, change):
    f, g, h, ledger = remaining
    path = h / 'evaluation/runs/h-f38-on/run.json'
    if change == 'missing':
        path.unlink()
    else:
        run = read_json(path)
        run.update(status='evaluating', gate_complete=False)
        atomic_json(path, run)
    result = module.export(h, ledger_path=ledger.path)
    assert result['remaining_group_complete'] is result['work_complete'] is False
    assert result['comparison_scope_ready'] is False
    assert not (h / 'evaluation/remaining-comparison.json').exists()
    if change == 'missing':
        assert result['groups'][3]['summary']['unexecuted_turn_count'] == 111
        assert result['groups'][3]['summary']['quality_distribution_planned']['unscored'] == 100
        assert result['groups'][3]['summary']['retrieval_metric_count'] == 81
    else:
        with pytest.raises(ComparisonScopeError):
            Repository(h / 'evaluation', history_roots=[g / 'evaluation']).compare(['new-off', 'h-f38-on'])


@pytest.mark.parametrize('change', ['parent_hash', 'parent_path', 'new_inputs', 'new_index', 'new_plan', 'new_suite', 'acknowledgement', 'unknown_schema'])
def test_export_rejects_unverified_source_or_scope(remaining, change):
    f, g, h, ledger = remaining
    freeze = read_json(h / 'freeze.json')
    suite = read_json(h / 'formal-suite.json')
    path = h / 'evaluation/runs/h-f38-on/run.json'
    run = read_json(path)
    if change == 'parent_hash': freeze['parent_runs']['F37-Off']['sha256'] = '0' * 64
    elif change == 'parent_path': freeze['parent_runs']['F37-Off']['path'] = '/unrelated/private/run.json'
    elif change == 'new_inputs': run['input_integrity']['qa_sha256'] = 'c' * 64
    elif change == 'new_index': run['index_binding']['snapshot_version'] = 'other'
    elif change == 'new_plan': run['qa_plan'][0]['planned_turn_count'] = 1
    elif change == 'new_suite': run['suite_id'] = 'unrelated-suite'
    elif change == 'acknowledgement': suite['acknowledgement']['call_id'] = 'new-unknown'
    elif change == 'unknown_schema': suite['schema_version'] = 'unsupported'
    atomic_json(h / 'freeze.json', freeze)
    atomic_json(h / 'formal-suite.json', suite)
    atomic_json(path, run)
    with pytest.raises(ValueError): module.export(h, ledger_path=ledger.path)


@pytest.mark.parametrize('change', ['schema', 'run_id', 'hash', 'source', 'suite', 'inputs', 'index', 'plan_hash', 'symlink', 'active_after_export'])
def test_scope_requires_all_four_exact_terminal_sources(remaining, change):
    f, g, h, ledger = remaining
    module.export(h, ledger_path=ledger.path)
    path = h / 'evaluation/remaining-comparison.json'
    scope = read_json(path)
    if change == 'schema': scope['schema_version'] = 'unknown'
    elif change == 'run_id': scope['runs'][2]['run_id'] = 'other'
    elif change == 'hash': scope['runs'][2]['run_sha256'] = '0' * 64
    elif change == 'source': scope['runs'][2]['source'] = 'current'
    elif change == 'suite': scope['runs'][2]['suite_id'] = 'h-suite'
    elif change == 'inputs': scope['input_integrity']['qa_sha256'] = 'c' * 64
    elif change == 'index': scope['index_binding'] = {'other': True}
    elif change == 'plan_hash': scope['qa_plan_sha256'] = '0' * 64
    elif change == 'active_after_export':
        run_path = h / 'evaluation/runs/h-f38-on/run.json'
        run = read_json(run_path)
        run['status'] = 'evaluating'
        atomic_json(run_path, run)
        scope['runs'][3]['run_sha256'] = sha256(run_path)
    atomic_json(path, scope)
    if change == 'symlink':
        target = path.with_name('elsewhere.json')
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ComparisonScopeError):
        Repository(h / 'evaluation', history_roots=[g / 'evaluation']).compare(['new-off', 'h-f38-on'])


def test_refuses_export_over_parent_and_preserves_old_compare_rules(remaining):
    f, g, h, ledger = remaining
    with pytest.raises(ValueError, match='independent runtime'):
        module.export(g, ledger_path=ledger.path)
    # Same-suite historical analysis remains available without any new scope.
    comparison = Repository(g / 'evaluation').compare(['new-off', 'new-on'])
    assert comparison['remaining_scope'] is None


def test_export_rejects_root_alias_before_normalizing(remaining):
    f, g, h, ledger = remaining
    linked = h.with_name('alias-h')
    linked.symlink_to(h, target_is_directory=True)
    with pytest.raises(ValueError):
        module.export(linked, ledger_path=ledger.path)
    assert not (h / 'reports').exists()
