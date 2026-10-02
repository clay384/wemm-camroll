"""Synthetic acceptance measures a working chain, never error-free answers."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from gate_cloud_smoke import synthetic_chain_acceptance, text_probe_status
from gate_config import MODELS


def test_text_capability_preserves_instruction_mismatch_without_rejecting_period():
    for answer in ('OK.', 'OK。'):
        status = text_probe_status(answer)
        assert status['passed'] is True
        assert status['exact_output_match'] is False
        assert status['observed_output'] == answer
    assert text_probe_status('OK')['exact_output_match'] is True
    for answer in (None, '', ' ', 'unavailable', 'not OK'):
        assert text_probe_status(answer)['passed'] is False


def chain_run(group='F37-Off', *, model_failures=False):
    on = group.endswith('-On')
    items = []
    for index, name in enumerate(('synthetic-red', 'synthetic-blue', 'synthetic-missing')):
        turn = {'index': 0, 'origin': 'model', 'answer_origin': 'model', 'failure_category': None,
                'judge_eligible': True, 'judge_status': 'completed', 'judge': {'score': 0},
                'task_judge': {'correct': False, 'actual_action': 'refuse' if index == 2 else 'answer'},
                'agent_status': 'complete', 'turn_outcome': 'refuse' if index == 2 else 'answer',
                'gate_status': 'accepted' if on else 'off', 'gate_attempts': 1 if on else 0,
                'termination_reason': 'ok', 'validator_failure_code': None, 'gate_events': [],
                'selected_asset_ids': [], 'predicted_media': [], 'predicted_images': []}
        if model_failures and index < 2:
            turn.update(origin='system_failure', answer_origin='system_failure', failure_category='model_contract_failure',
                        judge_eligible=False, judge_status='skipped_expected', judge={'score': None},
                        task_judge={'correct': None, 'actual_action': None}, agent_status='error', turn_outcome='error',
                        termination_reason='invalid_evidence_reference', gate_status='blocked' if on else 'off')
        calls = [{'call_id': name + '-agent', 'call_type': 'agent', 'status': 'success'},
                 {'call_id': name + '-view', 'call_type': 'tool_internal', 'original_call_type': 'view', 'status': 'success'},
                 {'call_id': name + '-final', 'call_type': 'final_review', 'status': 'success'}] if index < 2 else []
        items.append({'qa_id': name, **copy.deepcopy(turn), 'runtime_turns': [turn],
                      'final_turn_executed': True, 'execution_status': 'error' if not turn['judge_eligible'] else 'complete',
                      'model_call_metrics': calls, 'gate_events': [], 'tool_trace': []})
    run = {'mode': 'gate_ab', 'formal_protocol': False, 'status': 'completed_with_errors' if model_failures else 'completed',
           'gate_complete': True, 'source_unchanged': True, 'qa_count': 3, 'planned_turn_count': 3,
           'qa_plan': [{'qa_id': item['qa_id'], 'planned_turn_count': 1} for item in items], 'items': items,
           'inference_config': {'group': group, 'generation_model': MODELS[group],
                                'gate_policy': 'repair-evidence-v1' if on else 'off'},
           'summary': {'gate': {'valid_fact_verdict_count': 0}}}
    if on:
        add_fact(run)
    return run


def add_fact(run, status='supported'):
    event = {'state': 'evidence_checked', 'verdict': {'status': status}, 'verifier_call_ids': ['fact-call']}
    call = {'call_id': 'fact-call', 'call_type': 'evidence_gate', 'status': 'complete',
            'parse_status': 'valid', 'api_sent': True, 'replayed': False,
            'response_model': 'doubao-seed-2-0-lite-260215', 'verdict_status': status}
    run['items'][-1]['gate_events'] = [event]
    run['items'][-1]['runtime_turns'][0]['gate_events'] = [copy.deepcopy(event)]
    run['items'][-1]['model_call_metrics'] = [call]
    run['summary']['gate']['valid_fact_verdict_count'] = 1


def update_turn(run, index=0, **changes):
    turn = run['items'][index]['runtime_turns'][0]
    turn.update(changes)
    run['items'][index].update(copy.deepcopy(changes))


def test_realistic_off_reference_failures_are_recorded_not_selected_away():
    run = chain_run(model_failures=True)
    before = copy.deepcopy(run)
    # This was the old, overstrict condition that blocked the real smoke.
    assert not all(turn['answer_origin'] == 'model' for item in run['items'] for turn in item['runtime_turns'])
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert result['passed'] and not result['issues']
    assert result['execution_counts']['model_contract_failure_turn_count'] == 2
    assert result['execution_counts']['expected_unscored_failure_turn_count'] == 2
    assert result['execution_counts']['normal_model_turn_count'] == result['execution_counts']['scored_model_turn_count'] == 1
    assert result['answer_correctness_required'] is False
    assert run == before


@pytest.mark.parametrize('group', list(MODELS))
def test_completed_wrong_answers_are_valid_measurements(group):
    result = synthetic_chain_acceptance(chain_run(group), group=group)
    assert result['passed']
    assert result['execution_counts']['scored_model_turn_count'] == 3
    assert result['integrated_fact_verdict_count'] == int(group.endswith('-On'))


def test_gate_exhaustion_with_valid_negative_fact_verdict_remains_unscored():
    run = chain_run('F38-On', model_failures=True)
    event = {'state': 'exhausted'}
    update_turn(run, origin='gate_exhausted', answer_origin='gate_exhausted', failure_category='gate_exhausted',
                termination_reason='gate_exhausted', gate_status='exhausted', gate_attempts=4, gate_events=[event])
    add_fact(run, 'contradicted')
    result = synthetic_chain_acceptance(run, group='F38-On')
    assert result['passed']
    assert result['execution_counts']['gate_exhausted_turn_count'] == 1
    assert result['execution_counts']['model_contract_failure_turn_count'] == 1
    assert result['execution_counts']['expected_unscored_failure_turn_count'] == 2


@pytest.mark.parametrize('changes,issue', [
    ({'failure_category': 'infrastructure_failure', 'error_code': 'http_403'}, 'infrastructure_failure'),
    ({'failure_category': 'model_contract_failure', 'error_code': 'usage_missing'}, 'infrastructure_failure'),
    ({'failure_category': 'validator_failure', 'validator_failure_code': 'gate_invalid_verdict'}, 'validator_failure'),
    ({'origin': 'model'}, 'inconsistent_origin'),
    ({'termination_reason': 'tool_error'}, 'unaccepted_failure_origin'),
    ({'judge_eligible': True}, 'invalid_failure_bookkeeping'),
    ({'judge': {'score': 2}}, 'invalid_failure_bookkeeping'),
    ({'task_judge': {'correct': True, 'actual_action': 'refuse'}}, 'invalid_failure_bookkeeping'),
    ({'judge_status': 'completed'}, 'invalid_failure_bookkeeping'),
    ({'agent_status': 'complete'}, 'invalid_failure_bookkeeping'),
    ({'turn_outcome': 'refuse'}, 'invalid_failure_bookkeeping'),
    ({'gate_status': 'exhausted'}, 'unaccepted_failure_origin'),
    ({'selected_asset_ids': ['img-invented']}, 'failed_turn_delivers_assets'),
])
def test_failure_metadata_cannot_disguise_faults_or_scored_fallback(changes, issue):
    run = chain_run(model_failures=True)
    update_turn(run, **changes)
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert not result['passed'] and issue in result['issues']


@pytest.mark.parametrize('changes,issue', [
    ({'judge': {'score': None}}, 'missing_model_judgment'),
    ({'judge': {'score': True}}, 'missing_model_judgment'),
    ({'task_judge': {'correct': None, 'actual_action': 'answer'}}, 'missing_model_judgment'),
    ({'judge_status': 'pending'}, 'missing_model_judgment'),
    ({'failure_category': 'model_contract_failure'}, 'invalid_model_origin_metadata'),
    ({'judge_eligible': False}, 'invalid_model_origin_metadata'),
    ({'turn_outcome': 'error'}, 'invalid_model_origin_metadata'),
])
def test_model_answers_need_consistent_origin_and_complete_quality_and_behavior_scores(changes, issue):
    run = chain_run()
    update_turn(run, **changes)
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert not result['passed'] and issue in result['issues']


@pytest.mark.parametrize('damage,issue', [
    (lambda r: r.update(qa_plan=[]), 'synthetic_plan_incomplete'),
    (lambda r: r.update(planned_turn_count=2), 'synthetic_plan_incomplete'),
    (lambda r: r['qa_plan'].append(copy.deepcopy(r['qa_plan'][0])), 'synthetic_plan_incomplete'),
    (lambda r: r['items'].pop(), 'missing_or_duplicate_question'),
    (lambda r: r['items'][0].update(runtime_turns=[]), 'missing_or_duplicate_turn'),
    (lambda r: r['items'][0]['runtime_turns'][0].update(index=2), 'missing_or_duplicate_turn'),
    (lambda r: r['items'][0].update(final_turn_executed=False), 'missing_or_duplicate_turn'),
    (lambda r: r['items'][0].update(answer_origin='system_failure'), 'inconsistent_final_item'),
    (lambda r: r['items'][0].update(execution_status='incomplete'), 'inconsistent_execution_status'),
    (lambda r: r.update(source_unchanged=False), 'source_changed'),
    (lambda r: r.update(gate_complete=False), 'incomplete_execution'),
    (lambda r: r.update(status='blocked'), 'incomplete_execution'),
    (lambda r: r.update(finalization_errors=[{'phase': 'resources'}]), 'run_fault'),
    (lambda r: r.update(stop_reason='request_uncertain'), 'run_fault'),
    (lambda r: r['inference_config'].update(gate_policy='repair-evidence-v1'), 'inference_configuration'),
])
def test_missing_execution_and_restoration_preconditions_never_pass(damage, issue):
    run = chain_run()
    damage(run)
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert not result['passed'] and issue in result['issues']


@pytest.mark.parametrize('role', ['view', 'final_review'])
@pytest.mark.parametrize('damage', ['missing', 'failed', 'replayed'])
def test_chain_requires_real_successful_view_and_final_review(role, damage):
    run = chain_run(model_failures=True)
    for item in run['items']:
        calls = item['model_call_metrics']
        for call in calls:
            if (call.get('original_call_type') or call['call_type']) == role:
                call.update(status='error' if damage == 'failed' else 'success', replayed=damage == 'replayed')
        if damage == 'missing':
            item['model_call_metrics'] = [call for call in calls if (call.get('original_call_type') or call['call_type']) != role]
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert not result['passed'] and 'missing_successful_view_final_chain' in result['issues']


@pytest.mark.parametrize('damage', [
    lambda r: r['items'][-1].update(gate_events=[]),
    lambda r: r['items'][-1].update(model_call_metrics=[]),
    lambda r: r['summary']['gate'].update(valid_fact_verdict_count=0),
    lambda r: r['items'][-1]['model_call_metrics'][0].update(parse_status='invalid'),
    lambda r: r['items'][-1]['model_call_metrics'][0].update(api_sent=False),
    lambda r: r['items'][-1]['model_call_metrics'][0].update(replayed=True),
    lambda r: r['items'][-1]['model_call_metrics'][0].update(response_model='different-model'),
    lambda r: r['items'][-1]['gate_events'][0].update(verifier_call_ids=['unrelated-probe']),
    lambda r: r['items'][-1]['gate_events'][0].update(verdict={'status': 'contradicted'}),
])
def test_gate_on_requires_in_run_valid_fact_judgment_not_a_direct_probe_or_summary_only(damage):
    run = chain_run('F37-On')
    damage(run)
    result = synthetic_chain_acceptance(run, group='F37-On')
    assert not result['passed'] and 'missing_integrated_fact_gate' in result['issues']


def test_gate_off_cannot_silently_have_a_fact_gate():
    run = chain_run()
    add_fact(run)
    result = synthetic_chain_acceptance(run, group='F37-Off')
    assert not result['passed'] and 'unexpected_fact_gate_off' in result['issues']
