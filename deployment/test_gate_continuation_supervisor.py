"""Isolated continuation supervisor; no network, model, or service operations."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_continuation as continuation
from camroll_eval.common import atomic_json, read_json
from test_gate_suite import complete_run


def ready_summary():
    return {'limit_cny': 600, 'reserved_cny': 1.268122, 'uncertain_requests': 1,
            'acknowledged_reserved_cny': 1.268122, 'acknowledged_uncertain_requests': 1,
            'blocking_uncertain_requests': 0, 'remaining_cny': 100}


@pytest.mark.parametrize('change', [
    {'blocking_uncertain_requests': 1, 'uncertain_requests': 2},
    {'reserved_cny': 1.5}, {'acknowledged_uncertain_requests': 0},
    {'uncertain_requests': 0}, {'remaining_cny': 0},
])
def test_only_exact_acknowledged_pending_charge_is_allowed(change):
    class Ledger:
        def check_limit(self): pass
        def summary(self): return {**ready_summary(), **change}
    with pytest.raises(RuntimeError):
        continuation.ledger_ready(Ledger())


def test_acknowledgement_does_not_hide_global_unknown_or_reservation():
    class Ledger:
        def check_limit(self): pass
        def summary(self): return ready_summary()
    result = continuation.ledger_ready(Ledger())
    assert result['uncertain_requests'] == 1
    assert result['reserved_cny'] == 1.268122


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    import four_gpu_window
    import four_supplement
    import camroll_eval.dataset
    runtime, parent = tmp_path / 'new', tmp_path / 'old'
    parent.mkdir()
    original = parent / 'run.json'
    original.write_bytes(b'unchanged parent bytes')
    before = original.read_bytes()
    events = []
    controls = {'new_unknown': False, 'raise_cleanup': False, 'raise_register': False,
                'original_running': False, 'block': False}
    class Ack:
        def as_dict(self): return {'call_id': 'prior-unknown', 'journal_path': str(parent / 'journal.sqlite')}
    ack = Ack()
    class Ledger:
        def check_limit(self): pass
        def summary(self):
            value = ready_summary()
            if controls['block']:
                value.update(blocking_uncertain_requests=1, uncertain_requests=2, reserved_cny=2)
            return value
    ledger = Ledger()
    frozen = {'parent_runs': {name: {'path': str(original), 'sha256': 'frozen'} for name in ('F37-Off', 'F37-On')},
              'acknowledgement': ack.as_dict(), 'wemm_fingerprint': 'same', 'model_files': {'models': {}},
              'wemm_weights': {'actual_sha256': 'unchanged', 'file_identity_after': {'inode': 1}}}
    class Dataset:
        rows = [{'qa_id': str(i), 'question': 'Synthetic question'} for i in range(100)]
        def __init__(self, *args): pass
        @staticmethod
        def turns(row): return [{}] * (2 if int(row['qa_id']) < 11 else 1)
    class Window:
        def __init__(self, release, window_id, *, runtime):
            assert runtime == continuation.HISTORY_RUNTIME
            self.state_path = tmp_path / 'window.json'
            events.append(('window', window_id))
        def enter(self):
            self.state_path.write_text('{}')
            events.append(('enter', controls['original_running']))
            return {'before': {'fingerprint': 'same'}}
        def switch(self, kind): events.append(('switch', kind))
        def register_child(self, pid, expected_command):
            events.append(('register', pid))
            if controls['raise_register']: raise RuntimeError('synthetic child registration failure')
        def unregister_child(self, pid): events.append(('unregister', pid))
        def restore(self, reason):
            events.append(('restore', reason))
            return {'state': 'restored', 'original_running': controls['original_running'],
                    'after': {'state': 'loaded' if controls['original_running'] else 'stopped'}}
    def reference(old_path, new_root, run_id, suite_id, expected_sha):
        events.append(('reference', run_id))
        assert old_path == str(original) and expected_sha == 'frozen'
        result = complete_run()
        result.update(run_id=run_id, suite_id=suite_id, summary={})
        atomic_json(new_root / 'runs' / run_id / 'run.json', result)
        return result
    class Child:
        serial = 0
        def __init__(self, command, **kwargs):
            self.pid = 100 + Child.serial
            Child.serial += 1
            assert kwargs['cwd'] == continuation.RELEASE
            group = command[command.index('--child-group') + 1]
            run_id = command[command.index('--run-id') + 1]
            suite_id = command[command.index('--suite-id') + 1]
            assert group != 'F37-Off'
            events.append(('child', group))
            self.returncode = 2 if controls['new_unknown'] else 0
            controls['block'] = controls['new_unknown']
            result = complete_run()
            result.update(run_id=run_id, suite_id=suite_id, summary={})
            if controls['new_unknown']: result.update(status='blocked', gate_complete=False)
            atomic_json(runtime / 'evaluation/runs' / run_id / 'run.json', result)
            self.polls = 0
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode
    def cleanup(child):
        events.append(('cleanup', child.pid if child else None))
        if controls['raise_cleanup']: raise RuntimeError('synthetic cleanup failure')
        return {'state': 'absent_or_exited'}
    monkeypatch.setattr(continuation, 'freeze_inputs', lambda *args: copy.deepcopy(frozen))
    monkeypatch.setattr(continuation, 'verify_frozen', lambda *args: None)
    monkeypatch.setattr(continuation, 'verify_wemm_weights', lambda: {
        'actual_sha256': 'unchanged', 'file_identity_before': {'inode': 1}})
    monkeypatch.setattr(continuation.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(continuation.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(continuation.subprocess, 'Popen', Child)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(four_gpu_window, 'verify_model_files', lambda: {'models': {}})
    monkeypatch.setattr(four_supplement, 'stop_owned_child', cleanup)
    monkeypatch.setattr(camroll_eval.dataset, 'Dataset', Dataset)
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', 'synthetic-continuation')
    monkeypatch.setitem(sys.modules, 'gate_continuation_export', types.SimpleNamespace(create_reference_run=reference))
    yield runtime, parent, ledger, ack, events, controls
    assert original.read_bytes() == before


@pytest.mark.parametrize('original_running', [False, True])
def test_only_remaining_groups_execute_in_new_shared_registry_window(supervisor, original_running):
    runtime, parent, ledger, ack, events, controls = supervisor
    controls['original_running'] = original_running
    assert continuation.execute(runtime, parent, ledger, ack) == 0
    state = read_json(runtime / 'formal-suite.json')
    assert state['status'] == 'passed'
    assert [value for name, value in events if name == 'child'] == list(continuation.EXECUTION_ORDER)
    assert len([row for row in events if row[0] == 'reference']) == 1
    assert state['groups'][0]['reused'] is True
    assert state['restoration']['original_running'] is original_running
    assert state['wemm_weights_unchanged'] is True
    assert state['budget_after']['uncertain_requests'] == 1
    assert state['budget_after']['reserved_cny'] == 1.268122
    planned = read_json(runtime / 'evaluation/planned-groups.json')
    assert len(planned['groups']) == 4
    assert all(len(row['qa_plan']) == 100 and sum(p['planned_turn_count'] for p in row['qa_plan']) == 111
               for row in planned['groups'])
    with pytest.raises(FileExistsError):
        continuation.execute(runtime, parent, ledger, ack)


def test_new_unknown_stops_remaining_groups_and_restores(supervisor):
    runtime, parent, ledger, ack, events, controls = supervisor
    controls['new_unknown'] = True
    assert continuation.execute(runtime, parent, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['status'] == 'blocked' and state['restoration']['state'] == 'restored'
    assert [value for name, value in events if name == 'child'] == ['F37-On']
    assert state['groups'][1]['new_submissions_blocked'] == 'new_uncertain_cloud_request_stop'
    assert (runtime / 'evaluation/runs' / state['groups'][1]['run_id'] / 'cancel.requested').is_file()
    assert all(row['status'] == 'not_started' for row in state['groups'][2:])
    assert state['budget_after']['uncertain_requests'] == 2


def test_child_registration_and_cleanup_failure_still_restore(supervisor):
    runtime, parent, ledger, ack, events, controls = supervisor
    controls.update(raise_register=True, raise_cleanup=True)
    assert continuation.execute(runtime, parent, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['child_cleanup']['state'] == 'failed'
    assert state['restoration']['state'] == 'restored'
    assert any(row[0] == 'restore' for row in events)


def test_runtime_overlap_is_rejected_before_any_parent_write(supervisor):
    runtime, parent, ledger, ack, events, controls = supervisor
    for target in (parent, parent / 'nested', parent.parent):
        with pytest.raises(ValueError, match='independent_runtime'):
            continuation.execute(target, parent, ledger, ack)
    assert not (parent / 'formal-suite.claim').exists()


def test_changed_weight_after_run_never_reports_restoration_acceptance(supervisor, monkeypatch):
    runtime, parent, ledger, ack, events, controls = supervisor
    monkeypatch.setattr(continuation, 'verify_wemm_weights', lambda: {
        'actual_sha256': 'changed', 'file_identity_before': {'inode': 1}})
    assert continuation.execute(runtime, parent, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['status'] == 'restore_failed'
    assert state['restoration']['state'] == 'restored'
    assert not state.get('model_files_unchanged')
    assert not state.get('wemm_weights_unchanged')


def test_full_weight_hash_uses_every_byte_and_keeps_identity(tmp_path):
    path = tmp_path / 'model.safetensors'
    payload = b'x' * (8 * 1024 * 1024 + 17)
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    result = continuation.verify_wemm_weights(path, expected_sha256=expected, expected_size=len(payload))
    assert result['actual_sha256'] == expected
    assert result['file_identity_before'] == result['file_identity_after']
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'y')
    with pytest.raises(RuntimeError, match='hash_changed'):
        continuation.verify_wemm_weights(path, expected_sha256=expected, expected_size=len(payload))


def test_weight_size_and_symlink_are_rejected(tmp_path):
    path = tmp_path / 'weight'
    path.write_bytes(b'original')
    with pytest.raises(RuntimeError, match='size_or_type'):
        continuation.verify_wemm_weights(path, expected_size=9)
    linked = tmp_path / 'linked'
    linked.symlink_to(path)
    with pytest.raises(RuntimeError, match='regular_file'):
        continuation.verify_wemm_weights(linked, expected_size=8)


def test_weight_replaced_during_hash_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'weight'
    path.write_bytes(b'original')
    expected = hashlib.sha256(b'original').hexdigest()
    original_sha = hashlib.sha256
    class Digest:
        def __init__(self): self.value = original_sha()
        def update(self, value):
            self.value.update(value)
            path.unlink()
            path.write_bytes(b'original')
        def hexdigest(self): return self.value.hexdigest()
    monkeypatch.setattr(continuation.hashlib, 'sha256', Digest)
    with pytest.raises(RuntimeError, match='identity_changed'):
        continuation.verify_wemm_weights(path, expected_sha256=expected, expected_size=8)


def test_child_rejects_identity_mismatch_before_backend_or_paid_call(tmp_path, monkeypatch):
    atomic_json(tmp_path / 'formal-suite.json', {'suite_id': 'new', 'status': 'running',
        'groups': [{'name': 'F37-On', 'run_id': 'owned', 'status': 'running'}]})
    atomic_json(tmp_path / 'freeze.json', {})
    with pytest.raises(RuntimeError, match='identity'):
        continuation.run_child(tmp_path, 'new', 'not-owned', 'F37-On', object())


def test_frozen_protocol_allows_only_exact_accounting_clone_change(tmp_path):
    source = continuation.RELEASE / 'camroll/camroll-agent'
    target = tmp_path / 'camroll/camroll-agent'
    names = ['camroll_agent/agent.py', 'camroll_agent/gate.py', 'camroll_agent/prompts.py',
             'camroll_eval/judge_prompts.json', 'camroll_eval/task_actions.json', 'camroll_eval/gate_metrics.py',
             'camroll_agent/qwen.py', 'camroll_agent/evidence_gate.py', 'camroll_eval/judge.py',
             'camroll_eval/gate_experiment.py']
    for name in names:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target / name)
    expected = continuation.protocol_manifest(tmp_path)
    qwen = target / 'camroll_agent/qwen.py'
    text = qwen.read_text()
    qwen.write_text(text.replace('ledger.with_expected_limit(expected_limit_cny)',
                                'BudgetLedger(ledger.path, expected_limit_cny=expected_limit_cny)'))
    assert continuation.protocol_manifest(tmp_path) == expected
    qwen.write_text(text.replace('self.model = model', 'self.model = model\n        self.accounting_scope = None', 1))
    assert continuation.protocol_manifest(tmp_path) != expected
    qwen.write_text(text)
    qwen.write_text(qwen.read_text().replace('MAX_OUTPUT_TOKENS = 32_768', 'MAX_OUTPUT_TOKENS = 16_384', 1))
    assert continuation.protocol_manifest(tmp_path) != expected
    qwen.write_text(text)
    evidence = target / 'camroll_agent/evidence_gate.py'
    evidence.write_text(evidence.read_text().replace('MAX_IMAGES = 6', 'MAX_IMAGES = 5', 1))
    assert continuation.protocol_manifest(tmp_path) != expected


@pytest.mark.parametrize('group', continuation.EXECUTION_ORDER)
def test_child_passes_scoped_ledger_and_only_on_parent_to_runner(tmp_path, monkeypatch, group):
    import camroll_eval.gate_experiment
    import four_gpu_window
    parent = {'path': '/synthetic/parent/run.json', 'sha256': 'immutable'}
    freeze = {'parent_runs': {'F37-On': parent}, 'acknowledgement': {'call_id': 'prior-unknown'}}
    atomic_json(tmp_path / 'freeze.json', freeze)
    atomic_json(tmp_path / 'formal-suite.json', {'suite_id': 'new', 'status': 'running',
        'groups': [{'name': group, 'run_id': 'owned', 'status': 'running'}]})
    class Ledger:
        def check_limit(self): pass
        def summary(self): return ready_summary()
    ledger = Ledger()
    seen = {}
    class Runner:
        def __init__(self, *args, budget_ledger): seen['ledger'] = budget_ledger
        def run_gate(self, **kwargs): seen.update(kwargs); return complete_run()
    class Window:
        def __init__(self, *args, **kwargs): assert kwargs['runtime'] == continuation.HISTORY_RUNTIME
        def _read(self): return {'state': 'active', 'current_kind': 'wemm'}
    monkeypatch.setattr(camroll_eval.gate_experiment, 'GateRunner', Runner)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(continuation, 'verify_frozen', lambda *args: None)
    continuation.run_child(tmp_path, 'new', 'owned', group, ledger)
    assert seen['ledger'] is ledger
    if group == 'F37-On':
        assert seen['continuation_parent'] == parent
        assert seen['acknowledged_call_ids'] == ['prior-unknown']
    else:
        assert 'continuation_parent' not in seen and 'acknowledged_call_ids' not in seen
