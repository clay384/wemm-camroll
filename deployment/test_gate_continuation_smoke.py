"""Continuation smoke uses real accounting and fake transports, never paid calls."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_continuation_smoke as smoke
from camroll_agent.budget import BudgetLedger, RequestJournal, UncertainRequestAcknowledgement
from camroll_eval.common import atomic_json, read_json, sha256
from four_suite import source_manifest


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import four_gpu_window
    import four_supplement
    import camroll_agent.qwen as qwen_module
    import camroll_agent.evidence_gate as fact_module
    import camroll_eval.judge as judge_module
    runtime, parent = tmp_path / 'new', tmp_path / 'parent'
    release, old_release = tmp_path / 'release', tmp_path / 'old-release'
    for path in (runtime, parent, release, old_release):
        path.mkdir()
    (release / 'core.txt').write_text('same frozen protocol')
    (old_release / 'core.txt').write_text('same frozen protocol')
    monkeypatch.setattr(smoke, 'RELEASE', release)
    monkeypatch.setattr(smoke.continuation, 'protocol_manifest', lambda path: {'core': sha256(Path(path) / 'core.txt')})
    old_source = source_manifest(old_release)
    atomic_json(parent / 'freeze.json', {'source_files': old_source})
    atomic_json(parent / 'cloud-smoke.json', {'status': 'passed', 'source_files': old_source,
        'restoration': {'state': 'restored', 'release': str(old_release)}})
    atomic_json(runtime / 'validation.json', {'status': 'passed', 'source_files': source_manifest(release)})
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600, expected_limit_cny=600)
    journal = RequestJournal(parent / 'request-journal.sqlite')
    request_id, payload = 'synthetic-old:request', hashlib.sha256(b'synthetic old payload').hexdigest()
    claim = journal.claim(request_id, payload, lambda: ledger.reserve(1.268122, {'request_id': request_id}))
    ledger.finish(claim['call_id'], None, {'error_code': 'transport_error'})
    journal.fail(request_id, 'transport_error', uncertain=True)
    ack = UncertainRequestAcknowledgement.capture(ledger.path, journal.path, call_id=claim['call_id'],
        request_id=request_id, payload_hash=payload, reserved_units=1268122)
    ledger = BudgetLedger(ledger.path, expected_limit_cny=600, acknowledged_uncertain=(ack,))
    def old_rows():
        with sqlite3.connect(ledger.path) as db:
            one = db.execute('SELECT * FROM budget_calls WHERE id=?', (ack.call_id,)).fetchone()
        with sqlite3.connect(journal.path) as db:
            two = db.execute('SELECT * FROM cloud_requests WHERE request_id=?', (ack.request_id,)).fetchone()
        return one, two
    original, events, paid = old_rows(), [], []
    controls = {'fail': None, 'original_running': False, 'registration_failure': False,
                'cleanup_failure': False, 'restore_failure': False, 'residual': False, 'source_change': False}
    class Window:
        def __init__(self, release_arg, window_id, *, runtime):
            assert release_arg == release and runtime == smoke.HISTORY_RUNTIME
            self.state_path = tmp_path / 'window.json'
            self.window_id = window_id
        def enter(self):
            events.append(('enter', self.window_id))
            self.state_path.write_text('{}')
            return {'model_integrity': {'models': {'frozen': 'hash'}}}
        def _read(self): return {'state': 'active', 'current_kind': None}
        def register_child(self, pid, expected_command):
            events.append(('register', pid))
            if controls['registration_failure']: raise RuntimeError('synthetic registration failure')
        def unregister_child(self, pid): events.append(('unregister', pid))
        def restore(self, reason):
            events.append(('restore', reason))
            if controls['restore_failure']: raise RuntimeError('synthetic restoration failure')
            return {'state': 'restored', 'no_experiment_gpu_residual': not controls['residual'],
                    'original_running': controls['original_running']}
    class Child:
        def __init__(self, command, **options):
            assert options['cwd'] == release and options['start_new_session'] is True
            assert '--child' in command and '--execute-paid' in command
            assert command[command.index('--runtime') + 1] == str(runtime)
            self.pid, self.poll_count = 1234, 0
            if controls['source_change']:
                (release / 'core.txt').write_text('changed source')
            smoke_id = command[command.index('--smoke-id') + 1]
            self.returncode = 2 if controls['registration_failure'] else (0 if smoke.run_checks(runtime, smoke_id, ledger) else 2)
        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else self.returncode
    def cleanup(child):
        events.append(('cleanup', child.pid if child else None))
        if controls['cleanup_failure']: raise RuntimeError('synthetic cleanup failure')
        return {'state': 'absent_or_exited'}
    actual_qwen, actual_fact, actual_judge = qwen_module.QwenClient, fact_module.EvidenceGateVerifier, judge_module.Judge
    def fail(name):
        if controls['fail'] == name:
            raise OSError('synthetic unknown transport outcome')
    def qwen_factory(**kwargs):
        assert kwargs['ledger'] is ledger and kwargs['expected_limit_cny'] == 600
        def transport(**body):
            paid.append(('qwen', body))
            fail('qwen')
            if body.get('tools'):
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'synthetic-call', 'type': 'function', 'function': {'name': 'record_color', 'arguments': '{"color":"red"}'}}]}
            elif body.get('response_format'):
                message = {'role': 'assistant', 'content': '{"color":"red"}'}
            elif isinstance(body['messages'][-1]['content'], list):
                message = {'role': 'assistant', 'content': 'red square'}
            else:
                message = {'role': 'assistant', 'content': 'OK'}
            return {'model': kwargs['model'], 'choices': [{'finish_reason': 'stop', 'message': message}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))
        return actual_qwen(sdk=sdk, **kwargs)
    def fact_factory(**kwargs):
        assert kwargs['ledger'] is ledger
        def transport(body, *, timeout):
            paid.append(('fact', body))
            fail('fact')
            ids = ['image:img-red', 'image:img-blue']
            verdict = {'status': 'supported', 'evidence_ids': ids, 'reasons': [
                {'claim': 'colored shapes', 'assessment': 'supported', 'evidence_ids': ids, 'reason': 'Both original images support the shapes.'}]}
            return {'model': 'doubao-seed-2-0-lite-260215', 'choices': [{'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': json.dumps(verdict)}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        return actual_fact(transport=transport, **kwargs)
    def judge_factory(**kwargs):
        assert kwargs['ledger'] is ledger and kwargs['expected_limit_cny'] == 600
        def transport(body):
            paid.append(('judge', body))
            fail('judge')
            value = ({'score': 2, 'reason': '回答正确。'} if body['max_tokens'] == 1024 else
                     {'actual_action': 'answer', 'reason': '直接回答。'})
            return {'model': 'doubao-seed-2-0-lite-260215', 'choices': [{'message': {'content': json.dumps(value)}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        return actual_judge(transport=transport, interval=0, **kwargs)
    monkeypatch.setattr(qwen_module, 'QwenClient', qwen_factory)
    monkeypatch.setattr(fact_module, 'EvidenceGateVerifier', fact_factory)
    monkeypatch.setattr(judge_module, 'Judge', judge_factory)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(four_gpu_window, 'verify_model_files', lambda: {'models': {'frozen': 'hash'}})
    monkeypatch.setattr(four_supplement, 'stop_owned_child', cleanup)
    monkeypatch.setattr(smoke.subprocess, 'Popen', Child)
    monkeypatch.setattr(smoke.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(smoke.time, 'sleep', lambda *args: None)
    yield SimpleNamespace(runtime=runtime, parent=parent, ledger=ledger, ack=ack, controls=controls,
                          events=events, paid=paid, release=release, old_release=old_release)
    assert old_rows() == original


@pytest.mark.parametrize('original_running', [False, True])
def test_scoped_real_clients_complete_only_connectivity_checks_and_restore(environment, original_running):
    e = environment
    e.controls['original_running'] = original_running
    assert smoke.execute(e.runtime, e.parent, e.ledger, e.ack) == 0
    report = read_json(e.runtime / 'cloud-smoke.json')
    assert report['status'] == 'passed' and report['restoration']['original_running'] is original_running
    assert report['source_files'] == source_manifest(e.release) and report['source_unchanged'] is True
    assert report['parent_cloud_smoke']['sha256'] == sha256(e.parent / 'cloud-smoke.json')
    assert report['parent_cloud_smoke']['restoration_state'] == 'restored'
    assert report['build_performed'] is report['formal_questions_submitted'] is report['local_model_loaded'] is False
    assert [name for name, _ in e.paid] == ['qwen'] * 8 + ['fact', 'judge', 'judge']
    assert all(value['status'] == 'passed' and set(value['checks']) == {'text', 'image', 'tool', 'structured'}
               for value in report['models'].values())
    assert report['after']['uncertain_requests'] == report['after']['acknowledged_uncertain_requests'] == 1
    assert report['after']['reserved_cny'] == 1.268122
    fact_payload = next(body for name, body in e.paid if name == 'fact')
    assert '参考答案' not in json.dumps(fact_payload, ensure_ascii=False)
    assert sum(row.get('type') == 'image_url' for row in fact_payload['messages'][-1]['content']) == 2
    assert report['verifier']['protocol']['max_tokens'] == 2048
    assert report['verifier']['protocol']['reference_answers_allowed'] is False
    assert report['judge']['quality']['score'] == 2 and report['judge']['behavior']['correct'] is True
    assert not (e.runtime / 'smoke-evaluation').exists()
    with pytest.raises(RuntimeError, match='already_recorded'):
        smoke.execute(e.runtime, e.parent, e.ledger, e.ack)


@pytest.mark.parametrize('role,count', [('qwen', 1), ('fact', 9), ('judge', 10)])
def test_new_unknown_stops_all_later_calls_retains_both_holds_and_restores(environment, role, count):
    e = environment
    e.controls['fail'] = role
    assert smoke.execute(e.runtime, e.parent, e.ledger, e.ack) == 2
    report = read_json(e.runtime / 'cloud-smoke.json')
    assert report['status'] == 'blocked' and report['restoration']['state'] == 'restored'
    assert len(e.paid) == count
    assert report['after']['uncertain_requests'] == 2 and report['after']['blocking_uncertain_requests'] == 1
    assert report['after']['acknowledged_reserved_cny'] == 1.268122
    assert report['after']['reserved_cny'] > 1.268122
    assert (e.runtime / 'smoke.cancel').exists()


@pytest.mark.parametrize('filename,change', [
    ('validation.json', {'status': 'failed'}), ('validation.json', {'source_files': {}}),
    ('parent/cloud-smoke.json', {'status': 'blocked'}),
    ('parent/cloud-smoke.json', {'restoration': {'state': 'restore_failed'}}),
    ('parent/cloud-smoke.json', {'source_files': {}}),
    ('parent/freeze.json', {'source_files': {}}),
])
def test_invalid_local_or_parent_acceptance_blocks_before_window_or_paid_call(environment, filename, change):
    e = environment
    path = e.parent / filename.split('/', 1)[1] if filename.startswith('parent/') else e.runtime / filename
    atomic_json(path, {**read_json(path), **change})
    with pytest.raises(RuntimeError):
        smoke.execute(e.runtime, e.parent, e.ledger, e.ack)
    assert not e.events and not e.paid and not (e.runtime / 'smoke.claim').exists()


def test_protocol_change_is_not_hidden_by_updated_source_metadata(environment):
    e = environment
    (e.release / 'core.txt').write_text('changed protocol')
    atomic_json(e.runtime / 'validation.json', {'status': 'passed', 'source_files': source_manifest(e.release)})
    with pytest.raises(RuntimeError, match='core_protocol_changed'):
        smoke.execute(e.runtime, e.parent, e.ledger, e.ack)
    assert not e.events and not e.paid


@pytest.mark.parametrize('flag', ['registration_failure', 'cleanup_failure', 'restore_failure', 'residual', 'source_change'])
def test_registration_finalization_and_restoration_failures_cannot_report_pass(environment, flag):
    e = environment
    e.controls[flag] = True
    assert smoke.execute(e.runtime, e.parent, e.ledger, e.ack) == 2
    report = read_json(e.runtime / 'cloud-smoke.json')
    assert report['status'] == 'blocked'
    assert any(name == 'restore' for name, value in e.events)
    if flag == 'source_change': assert not e.paid


def test_explicit_claim_is_one_shot_even_without_previous_report(environment):
    e = environment
    (e.runtime / 'smoke.claim').mkdir()
    with pytest.raises(FileExistsError):
        smoke.execute(e.runtime, e.parent, e.ledger, e.ack)
    assert not e.events and not e.paid


def test_parent_runtime_is_never_a_smoke_destination(environment):
    e = environment
    with pytest.raises(ValueError, match='independent_runtime'):
        smoke.execute(e.parent, e.parent, e.ledger, e.ack)
    assert not e.events and not e.paid


def test_legacy_fixtures_accept_runtime_without_mutating_default(tmp_path, monkeypatch):
    import gate_cloud_smoke
    original = gate_cloud_smoke.RUNTIME
    first = gate_cloud_smoke.fixtures(runtime=tmp_path)
    assert first == tmp_path / 'fixtures/synthetic-flash-gate-v1'
    assert (first / 'photos/red.jpg').is_file() and (first / 'photos/blue.jpg').is_file()
    assert gate_cloud_smoke.RUNTIME == original
    with pytest.raises(RuntimeError, match='fixture_exists'):
        gate_cloud_smoke.fixtures(runtime=tmp_path)
