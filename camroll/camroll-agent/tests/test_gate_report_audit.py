"""Synthetic read-only projections of durable traces after interrupted writes."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json
from camroll_eval.repository import Repository
from camroll_eval.service import create_app


@pytest.fixture
def audit_report(tmp_path):
    root = tmp_path / 'evaluation'
    directory = root / 'runs/gate-run'
    plans = [{'qa_id': name, 'question': 'Synthetic ' + name, 'planned_turn_count': 2,
              'execution_status': 'not_started', 'judge_status': 'not_started'} for name in ('qa-a', 'qa-b')]
    state = {'run_id': 'gate-run', 'mode': 'gate_ab', 'gate_summary_version': 'camroll-flash-gate-v1',
             'status': 'interrupted', 'summary': {'judge_valid_count': 0, 'quality_mean_fixed_planned': 0},
             'qa_plan': plans, 'planned_status': plans, 'items': []}
    path = directory / 'run.json'
    atomic_json(path, state)
    return {'root': root, 'directory': directory, 'path': path, 'state': state, 'repo': Repository(root)}


def event(*, qa='qa-a', run='gate-run', turn=0, state='rejected', **extra):
    return {'state': state, 'semantic_request_id': 'attempt-1', 'attempt': 1,
            'at_epoch': 100, 'candidate': {'content': 'Synthetic candidate'},
            'evaluation_context': {'run_id': run, 'qa_id': qa, 'turn_index': turn}, **extra}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_normal_embedded_and_durable_records_are_deduplicated_per_turn(audit_report):
    report = audit_report
    raw = event()
    embedded = {key: value for key, value in raw.items() if key != 'evaluation_context'}
    embedded['turn_index'] = 0
    call = {'call_id': 'cloud-call', 'request_id': 'cloud-request', 'call_type': 'agent',
            'input': {'messages': ['synthetic']}, 'evaluation_context': raw['evaluation_context']}
    stored_call = {key: value for key, value in call.items() if key != 'evaluation_context'}
    stored_call.update(conversation_turn=0, total_ms=10)
    item = {'qa_id': 'qa-a', 'answer': 'Preserved answer', 'judge': {'score': 0},
            'runtime_turns': [], 'gate_events': [embedded], 'model_call_metrics': [stored_call]}
    report['state']['items'] = [item]
    atomic_json(report['path'], report['state'])
    write_rows(report['directory'] / 'gate_events.jsonl', [raw, raw, event(turn=1)])
    write_rows(report['directory'] / 'model_calls.jsonl', [call, call])
    write_rows(report['directory'] / 'verifier_calls.jsonl', [call])
    result = report['repo'].item('gate-run', 'qa-a')
    assert result['answer'] == item['answer'] and result['judge'] == item['judge']
    assert result['gate_events'][0] == embedded
    assert len(result['gate_events']) == 2
    assert result['gate_events'][1]['turn_index'] == 1
    assert result['model_call_metrics'] == [stored_call]
    assert result['trace_audit']['supplemented_count'] == 1


def test_hard_interruption_keeps_jsonl_candidates_without_inventing_an_answer(audit_report, tmp_path):
    report = audit_report
    write_rows(report['directory'] / 'gate_events.jsonl', [event(state='sending'), event()])
    call = {'call_id': 'fact-call', 'request_id': 'fact-request', 'role': 'evidence_gate',
            'output': {'status': 'insufficient'}, 'evaluation_context': event()['evaluation_context']}
    write_rows(report['directory'] / 'verifier_calls.jsonl', [call])
    originals = {path: path.read_bytes() for path in report['directory'].iterdir()}
    app = create_app(report['root'], tmp_path / 'source', tmp_path / 'albums', read_only=True, mount_agent=False)
    with TestClient(app) as client:
        item = client.get('/api/runs/gate-run/items/qa-a/view').json()
        assert item['answer'] == '' and item['judge']['score'] is None
        assert item['runtime_turns'] == [] and item['execution_status'] == 'not_started'
        assert item['trace_counts'] == {'gate': 2, 'model': 1, 'tool': 0}
        assert item['trace_audit']['supplemented_count'] == 3
        trace = client.get('/api/runs/gate-run/items/qa-a/trace/gate/1').json()
        assert trace['candidate'] == event()['candidate']
        assert trace['report_audit'] == {'source': 'gate_events.jsonl', 'not_in_run_snapshot': True}
        assert client.get('/api/runs/gate-run/items/qa-a/trace/model/0').json()['output'] == call['output']
        assert client.get('/api/runs/gate-run/export').content == originals[report['path']]
        assert client.get('/api/runs/gate-run').json()['summary'] == report['state']['summary']
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_context_isolation_rejects_other_run_question_and_invalid_turns(audit_report):
    rows = [event(), event(qa='qa-b'), event(run='unrelated-run'), event(qa='../../private'),
            *[event(turn=value) for value in (-1, 2, '0', None, True)],
            {**event(), 'evaluation_context': None},
            {key: value for key, value in event().items() if key != 'evaluation_context'}]
    write_rows(audit_report['directory'] / 'gate_events.jsonl', rows)
    repo = audit_report['repo']
    first = repo.item('gate-run', 'qa-a')
    second = repo.item('gate-run', 'qa-b')
    assert len(first['gate_events']) == len(second['gate_events']) == 1
    assert first['gate_events'][0]['evaluation_context']['qa_id'] == 'qa-a'
    assert second['gate_events'][0]['evaluation_context']['qa_id'] == 'qa-b'
    with pytest.raises(FileNotFoundError):
        repo.item('gate-run', '../../private')


def test_symlink_is_not_followed_and_corrupt_tail_keeps_prior_valid_traces(audit_report, tmp_path):
    report = audit_report
    outside = tmp_path / 'unrelated-private.jsonl'
    write_rows(outside, [{**event(), 'candidate': {'content': 'Must never be exposed'}}])
    (report['directory'] / 'model_calls.jsonl').symlink_to(outside)
    target = report['directory'] / 'gate_events.jsonl'
    write_rows(target, [event()])
    with target.open('a') as stream:
        stream.write('{"evaluation_context":')
    result = report['repo'].item('gate-run', 'qa-a')
    assert len(result['gate_events']) == 1 and not result.get('model_call_metrics')
    assert {row['issue'] for row in result['trace_audit']['issues']} == {
        'unsafe_trace_path', 'incomplete_or_invalid_record'}
    encoded = json.dumps(result)
    assert 'Must never be exposed' not in encoded and str(outside) not in encoded
    assert all(row['source'] in {'gate_events.jsonl', 'model_calls.jsonl'} for row in result['trace_audit']['issues'])


def test_projection_does_not_pollute_cached_run_and_observes_new_jsonl(audit_report):
    report = audit_report
    repo = report['repo']
    original = repo.load('gate-run')
    before = report['path'].read_bytes()
    target = report['directory'] / 'gate_events.jsonl'
    write_rows(target, [event()])
    first = repo.item('gate-run', 'qa-a')
    first['gate_events'][0]['candidate']['content'] = 'Caller mutation'
    first['judge']['score'] = 2
    trace = repo.item_trace('gate-run', 'qa-a', 'gate', 0)
    trace['candidate']['content'] = 'Another mutation'
    assert repo.item_trace('gate-run', 'qa-a', 'gate', 0)['candidate'] == event()['candidate']
    with target.open('a') as stream:
        stream.write(json.dumps(event(turn=1)) + '\n')
    assert repo.item('gate-run', 'qa-a', light=True)['trace_counts']['gate'] == 2
    assert repo.load('gate-run') == original
    assert repo.get_brief('gate-run')['summary'] == original['summary']
    assert report['path'].read_bytes() == before


def test_invalid_utf8_tail_does_not_hide_earlier_valid_records(audit_report):
    target = audit_report['directory'] / 'gate_events.jsonl'
    target.write_bytes((json.dumps(event()) + '\n').encode('utf-8') + b'{"candidate":"\x80')
    result = audit_report['repo'].item('gate-run', 'qa-a')
    assert len(result['gate_events']) == 1
    assert result['trace_audit']['issues'] == [
        {'source': 'gate_events.jsonl', 'issue': 'incomplete_or_invalid_record'}]


def test_legacy_report_ignores_new_gate_sidecars(audit_report):
    report = audit_report
    state = copy.deepcopy(report['state'])
    state.update(mode='supplement', supplement_summary_version='http403-supplement-v1')
    atomic_json(report['path'], state)
    write_rows(report['directory'] / 'gate_events.jsonl', [event()])
    result = report['repo'].item('gate-run', 'qa-a')
    assert 'gate_events' not in result and 'trace_audit' not in result
