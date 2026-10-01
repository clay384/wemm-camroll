import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from gate_suite import ORDER, MODELS, ledger_ready, validate_run
from gate_export import journal_requests, paired_deltas


def complete_run():
    plans = [{'qa_id': str(i), 'planned_turn_count': 2 if i < 11 else 1} for i in range(100)]
    good = {'answer_origin': 'model', 'judge': {'score': 0}, 'task_judge': {'correct': False, 'actual_action': 'answer'},
            'judge_status': 'completed', 'execution_status': 'complete'}
    return {'mode': 'gate_ab', 'status': 'completed_with_errors', 'source_unchanged': True,
            'gate_complete': True, 'formal_protocol': True, 'planned_eligible_qa_count': 81,
            'qa_plan': plans, 'items': [{'qa_id': p['qa_id'], 'runtime_turns': [copy.deepcopy(good) for _ in range(p['planned_turn_count'])]} for p in plans]}


def test_only_two_flash_models_in_four_fixed_groups():
    assert ORDER == ('F37-Off', 'F37-On', 'F38-Off', 'F38-On')
    assert set(MODELS.values()) == {'qwen3.7-flash', 'qwen3.8-flash'}


def test_execution_complete_does_not_require_correct_answers():
    run = complete_run()
    assert validate_run(run)
    run['items'][0]['runtime_turns'][0].update(answer_origin='gate_exhausted', judge={'score': None}, judge_status='skipped_expected')
    assert validate_run(run)
    run['items'][-1]['runtime_turns'].clear()
    assert not validate_run(run)


def test_unscored_model_answer_is_incomplete():
    run = complete_run()
    run['items'][0]['runtime_turns'][0]['judge'] = {'score': None}
    assert not validate_run(run)


def test_budget_preflight_never_raises_limit():
    class Ledger:
        def check_limit(self): pass
        def summary(self): return {'reserved_cny': 1, 'uncertain_requests': 0, 'remaining_cny': 100}
        def raise_limit(self, *args): raise AssertionError('unauthorized raise')
    with pytest.raises(RuntimeError, match='reconciliation'):
        ledger_ready(Ledger())


def test_pairwise_report_marks_partial_pairs_and_keeps_missing_values():
    rows = [{'name': 'F37-Off', 'acceptance_complete': True, 'summary': {'quality_correct_planned_rate': .5}},
            {'name': 'F37-On', 'acceptance_complete': False, 'summary': {'quality_correct_planned_rate': .6}}]
    result = paired_deltas(rows)
    assert len(result) == 4
    assert result[0]['delta_to_minus_from']['quality_correct_planned_rate'] == pytest.approx(.1)
    assert result[0]['complete_pair'] is False
    assert result[1]['delta_to_minus_from']['quality_correct_planned_rate'] is None


def test_journal_attribution_includes_interrupted_requests(tmp_path):
    from camroll_agent.budget import RequestJournal
    path = tmp_path / 'journal.sqlite'
    journal = RequestJournal(path)
    journal.claim('unknown', 'payload', lambda: 'reserved-call')
    rows, issues = journal_requests(path)
    assert not issues
    assert rows == [{'request_id': 'unknown', 'call_id': 'reserved-call', 'status': 'pending',
                     'payload_hash': 'payload', 'error_code': None}]


@pytest.mark.parametrize('budget_read_fails', [False, True])
def test_supervisor_restores_and_persists_even_if_child_cleanup_or_budget_read_fails(tmp_path, monkeypatch, budget_read_fails):
    import gate_suite
    import four_gpu_window
    import four_supplement
    import camroll_agent.budget
    import camroll_eval.dataset
    from camroll_eval.common import read_json
    class Ledger:
        reads = 0
        def __init__(self, *args, **kwargs): pass
        def check_limit(self): pass
        def summary(self):
            self.reads += 1
            if budget_read_fails and self.reads >= 3:
                raise OSError('synthetic ledger read failure')
            return {'reserved_cny': 0, 'uncertain_requests': 0, 'remaining_cny': 10}
    restored = []
    class Window:
        def __init__(self, *args, **kwargs): self.state_path = tmp_path / 'window.json'
        def enter(self):
            self.state_path.write_text('{}')
            return {'before': {'fingerprint': 'same'}}
        def switch(self, kind): raise RuntimeError('synthetic switch failed')
        def restore(self, reason):
            restored.append(reason)
            return {'state': 'restored'}
    class Dataset:
        rows = []
        def __init__(self, *args): pass
    monkeypatch.setattr(gate_suite, 'RUNTIME', tmp_path)
    monkeypatch.setattr(gate_suite, 'bootstrap', lambda: None)
    monkeypatch.setenv('CAMROLL_BUDGET_DB', str(tmp_path / 'fake-ledger'))
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', 'synthetic')
    monkeypatch.setattr(camroll_agent.budget, 'BudgetLedger', Ledger)
    monkeypatch.setattr(camroll_eval.dataset, 'Dataset', Dataset)
    monkeypatch.setattr(gate_suite, 'freeze_inputs', lambda path: {'wemm_fingerprint': 'same', 'model_files': {'models': {}}})
    monkeypatch.setattr(gate_suite, 'verify_frozen', lambda freeze: None)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(four_gpu_window, 'verify_model_files', lambda: {'models': {}})
    monkeypatch.setattr(gate_suite.signal, 'signal', lambda *args: None)
    def failed_cleanup(child): raise OSError('synthetic cleanup failed')
    monkeypatch.setattr(four_supplement, 'stop_owned_child', failed_cleanup)
    assert gate_suite.main(['--execute-paid']) == 2
    state = read_json(tmp_path / 'formal-suite.json')
    assert restored and state['restoration']['state'] == 'restored'
    assert state['child_cleanup']['state'] == 'failed'
    assert state['status'] == 'blocked'
    assert all(group['status'] == 'not_started' for group in state['groups'])
    if budget_read_fails:
        assert state['budget_after'] is None and state['budget_audit_error'] == 'OSError'


def test_adapter_incomplete_flag_is_never_promoted_to_complete():
    run = complete_run()
    run['gate_complete'] = False
    assert not validate_run(run)
    run['gate_complete'] = True
    run['finalization_errors'] = [{'phase': 'resources'}]
    assert not validate_run(run)
