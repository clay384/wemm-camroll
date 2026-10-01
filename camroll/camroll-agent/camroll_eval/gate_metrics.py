"""Gate audit counters with explicit denominators; no access to reference answers."""
from collections import Counter


def gate_metrics(state):
    turns = [turn for item in state.get('items', []) for turn in item.get('runtime_turns', [])]
    events = [event for item in state.get('items', []) for event in item.get('gate_events', [])]
    replayed_ids = {event.get('semantic_request_id') for event in events if event.get('replayed') is True}
    sends = [event for event in events if event.get('state') == 'sending'
             and event.get('semantic_request_id') not in replayed_ids]
    initial = [event for event in sends if event.get('attempt') == 1]
    repeat = [event for event in sends if event.get('attempt') == 2]
    repair = [event for event in sends if event.get('attempt') == 3]
    accepted = [event for event in events if event.get('state') == 'accepted' and event.get('replayed') is not True]
    facts = [event for event in events if event.get('state') == 'evidence_checked' and event.get('reused_verdict') is not True]
    first_ok = sum(event.get('attempt') == 1 for event in accepted)
    repeat_ok = sum(event.get('attempt') == 2 for event in accepted)
    repair_ok = sum(event.get('attempt') in (3, 4) for event in accepted)
    rejected_facts = sum((event.get('verdict') or {}).get('status') != 'supported' for event in facts)
    errors = Counter((event.get('error') or {}).get('code', 'unknown') for event in events if event.get('state') == 'rejected')
    blocked = Counter(event.get('error_code', 'unknown') for event in events if event.get('state') == 'blocked')
    stops = Counter(turn.get('termination_reason', 'unknown') for turn in turns
                    if turn.get('answer_origin', turn.get('origin')) != 'model')
    categories = Counter(turn.get('failure_category') or 'unclassified' for turn in turns
                         if turn.get('answer_origin', turn.get('origin')) != 'model')
    tool_errors = Counter()
    for item in state.get('items', []):
        for tool in item.get('tool_trace', []):
            error = (tool.get('result') or {}).get('error')
            if isinstance(error, dict):
                tool_errors[error.get('code', 'unknown')] += 1
    sent_at = {event.get('semantic_request_id'): event.get('at_epoch') for event in sends}
    extra_ms = 0
    extra_observed = 0
    for event in events:
        if event.get('state') not in ('accepted', 'rejected', 'blocked') or event.get('attempt', 0) < 2:
            continue
        start, end = sent_at.get(event.get('semantic_request_id')), event.get('at_epoch')
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            extra_ms += max(0, end - start) * 1000
            extra_observed += 1
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {'schema_version': 'camroll-gate-counters-v1',
            'logical_output_count': len(initial), 'generation_attempt_count': len(sends),
            'extra_generation_attempt_count': len(sends) - len(initial),
            'journal_replayed_attempt_count': len(replayed_ids),
            'first_pass_count': first_ok, 'first_pass_denominator': len(initial),
            'first_pass_rate': ratio(first_ok, len(initial)),
            'exact_retry_count': len(repeat), 'exact_retry_recovered_count': repeat_ok,
            'exact_retry_recovery_rate': ratio(repeat_ok, len(repeat)),
            'prompt_repair_logical_count': len(repair), 'prompt_repair_recovered_count': repair_ok,
            'prompt_repair_recovery_rate': ratio(repair_ok, len(repair)),
            'valid_fact_verdict_count': len(facts), 'fact_rejected_count': rejected_facts,
            'reused_fact_verdict_count': sum(event.get('state') == 'evidence_checked' and event.get('reused_verdict') is True for event in events),
            'fact_rejection_rate': ratio(rejected_facts, len(facts)),
            'exhausted_logical_count': sum(event.get('state') == 'exhausted' for event in events),
            'extra_attempt_wall_ms': extra_ms if extra_observed else None,
            'extra_attempt_timing_count': extra_observed,
            'repair_error_types': dict(errors), 'blocked_error_types': dict(blocked),
            'failure_categories': dict(categories),
            'execution_failure_types': dict(stops), 'tool_error_types': dict(tool_errors),
            'denominator_note': 'First pass: initial logical outputs. Exact retry: attempt 2 submitted. Prompt repair: logical outputs reaching attempt 3. Fact rejection: one valid verdict per distinct candidate/evidence. Journal replays and reused verdicts are excluded; in-flight outputs remain in denominators.',
            'timing_note': 'Extra semantic attempt wall time includes validation; separate verifier and model traces retain every physical request and usage.'}
