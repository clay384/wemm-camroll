import pytest

from camroll_eval.gate_metrics import gate_metrics


def event(state, attempt, identity='same', **values):
    return {'state': state, 'attempt': attempt, 'semantic_request_id': identity, **values}


def test_rates_keep_started_and_unknown_attempts_in_denominators():
    events = [event('sending', 1, 'a', at_epoch=0), event('rejected', 1, 'a', error={'code': 'invalid_json'}),
              event('sending', 2, 'a2', at_epoch=1), event('accepted', 2, 'a2', at_epoch=3),
              event('sending', 1, 'b'), event('accepted', 1, 'b'),
              event('sending', 1, 'c'), event('rejected', 1, 'c'),
              event('sending', 2, 'c2'), event('blocked', 2, 'c2', error_code='request_uncertain'),
              {'state': 'evidence_checked', 'verdict': {'status': 'supported'}},
              {'state': 'evidence_checked', 'verdict': {'status': 'contradicted'}}]
    result = gate_metrics({'items': [{'gate_events': events}]})
    assert result['first_pass_rate'] == pytest.approx(1/3)
    assert result['exact_retry_recovery_rate'] == .5
    assert result['extra_generation_attempt_count'] == 2
    assert result['extra_attempt_wall_ms'] == 2000
    assert result['fact_rejection_rate'] == .5
    assert result['blocked_error_types'] == {'request_uncertain': 1}


def test_prompt_repair_counts_logical_outputs_not_each_of_two_repairs():
    events = [event('sending', i, str(i)) for i in range(1, 5)] + [event('accepted', 4, '4')]
    result = gate_metrics({'items': [{'gate_events': events}]})
    assert result['prompt_repair_logical_count'] == 1
    assert result['prompt_repair_recovered_count'] == 1
    assert result['prompt_repair_recovery_rate'] == 1


def test_off_rates_are_unobserved_and_execution_errors_remain_visible():
    result = gate_metrics({'items': [{'runtime_turns': [{'answer_origin': 'system_failure', 'termination_reason': 'no_progress'}]}]})
    assert result['first_pass_rate'] is None
    assert result['fact_rejection_rate'] is None
    assert result['execution_failure_types'] == {'no_progress': 1}


def test_journal_replays_and_same_candidate_verdict_reuse_are_not_new_attempts():
    events = [event('sending', 2), event('accepted', 2, replayed=True),
              {'state': 'evidence_checked', 'verdict': {'status': 'contradicted'}},
              {'state': 'evidence_checked', 'verdict': {'status': 'contradicted'}, 'reused_verdict': True}]
    result = gate_metrics({'items': [{'gate_events': events}]})
    assert result['exact_retry_count'] == 0 and result['exact_retry_recovered_count'] == 0
    assert result['journal_replayed_attempt_count'] == 1
    assert result['valid_fact_verdict_count'] == 1 and result['reused_fact_verdict_count'] == 1


def test_exact_repeat_payload_mismatch_blocks_instead_of_being_model_failure():
    from camroll_eval.gate_experiment import infrastructure_failure
    assert infrastructure_failure({'error_code': 'gate_payload_mismatch'}) == 'gate_payload_mismatch'
