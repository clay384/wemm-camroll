"""No network/GPU/paid calls: parent eligibility and one-group ownership."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_remaining as remaining
from camroll_eval.common import atomic_json, read_json, sha256
from four_suite import source_manifest
from test_gate_suite import complete_run
from test_gate_continuation_supervisor import ready_summary


class Ack:
    def as_dict(self): return {'call_id': 'existing-only'}


@pytest.fixture
def parent(tmp_path, monkeypatch):
    import camroll_eval.dataset
    root, release, original, ancestor_release = [tmp_path / name for name in ('g', 'h-release', 'g-release', 'f-release')]
    for path in (root, release, original, ancestor_release): path.mkdir()
    names = ['camroll_agent.py', *remaining.REPORT_ONLY_CHANGES]
    for name in names:
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original bytes')
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    (release / 'new-entry.py').write_text('new orchestration only')
    monkeypatch.setattr(remaining, 'RELEASE', release)
    monkeypatch.setattr(remaining, 'PARENT_RELEASE', original)
    monkeypatch.setattr(remaining, 'PARENT_RUNTIME', root)
    protocol = {'frozen': 'protocol'}
    monkeypatch.setattr(remaining.continuation, 'protocol_manifest', lambda *args: protocol)
    albums, history = tmp_path / 'albums', tmp_path / 'history'
    (albums / remaining.SOURCE_ALBUM_ID).mkdir(parents=True)
    monkeypatch.setattr(remaining, 'ALBUMS', albums)
    monkeypatch.setattr(remaining, 'HISTORY_RUNTIME', history)
    source_run = history / 'evaluation/runs' / remaining.SOURCE_RUN_ID / 'run.json'
    atomic_json(source_run, {'immutable': 'source'})
    integrity = {'input': 'same'}
    class Dataset:
        def __init__(self, *args): pass
        def integrity(self): return integrity
    monkeypatch.setattr(camroll_eval.dataset, 'Dataset', Dataset)
    ancestor_freeze = tmp_path / 'f/freeze.json'
    atomic_json(ancestor_freeze, {'source_files': {}, 'album_files': {}, 'input_integrity': integrity,
                                 'source_run_sha256': sha256(source_run)})
    ancestor_suite = tmp_path / 'f/formal-suite.json'
    atomic_json(ancestor_suite, {'immutable': 'ancestor'})
    groups, hashes = [], {}
    for name in remaining.ORDER:
        identifier = name.lower() + '-g'
        group = {'name': name, 'run_id': identifier, 'status': 'not_started', 'qa_count': 100, 'planned_turn_count': 111}
        if name != remaining.GROUP:
            path = root / 'evaluation/runs' / identifier / 'run.json'
            run = complete_run()
            run.update(suite_id=remaining.PARENT_SUITE, run_id=identifier, inference_config={'group': name})
            if name == 'F38-Off':
                run.update(status='blocked', stop_reason='context_limit', gate_complete=False,
                    items=[{'qa_id': '0', 'runtime_turns': [{'index': 0, 'answer_origin': 'system_failure',
                        'judge_status': 'skipped_expected', 'raw_attempt_path': 'turn_attempts/0/0.json'}]}])
                atomic_json(path.parent / 'turn_attempts/0/0.json', {'response': {'tool_trace': [
                    {'error': {'error_code': 'context_limit', 'exception_type': 'ProviderError'}}]}})
            atomic_json(path, run)
            hashes[name] = sha256(path)
            group.update(status=run['status'], returncode=2 if name == 'F38-Off' else 0,
                         acceptance_complete=name != 'F38-Off')
            if name != 'F37-Off':
                db = sqlite3.connect(path.parent / 'request-journal.sqlite')
                db.execute('CREATE TABLE cloud_requests (request_id,payload_hash,call_id,status)')
                db.execute("INSERT INTO cloud_requests VALUES ('request','payload','unique','completed')")
                db.commit(); db.close()
        groups.append(group)
    monkeypatch.setattr(remaining, 'PARENT_RUN_SHA256', hashes)
    suite = {'suite_id': remaining.PARENT_SUITE, 'status': 'blocked', 'error': 'group_blocked_F38-Off',
        'finished_at': 'time', 'groups': groups, 'acknowledgement': Ack().as_dict(),
        'restoration': {'state': 'restored', 'no_experiment_gpu_residual': True},
        'model_files_unchanged': True, 'wemm_weights_unchanged': True, 'frozen_inputs_unchanged': True}
    atomic_json(root / 'formal-suite.json', suite)
    monkeypatch.setattr(remaining, 'PARENT_SUITE_SHA256', sha256(root / 'formal-suite.json'))
    frozen = {'source_files': source_manifest(original), 'acknowledgement': Ack().as_dict(),
        'protocol_files': protocol, 'snapshot': remaining.SNAPSHOT, 'wemm_fingerprint': 'same',
        'parent_freeze': {'path': str(ancestor_freeze), 'sha256': sha256(ancestor_freeze)},
        'parent_suite': {'path': str(ancestor_suite), 'sha256': sha256(ancestor_suite)},
        'parent_runs': {}, 'parent_release': str(ancestor_release)}
    atomic_json(root / 'freeze.json', frozen)
    monkeypatch.setattr(remaining, 'PARENT_FREEZE_SHA256', sha256(root / 'freeze.json'))
    atomic_json(root / 'cloud-smoke.json', {'status': 'passed', 'restoration': {'state': 'restored'},
                                           'source_files': frozen['source_files']})
    atomic_json(root / 'reports/flash-gate.json', {'status': 'incomplete'})
    (root / 'budget-final-snapshot.sqlite').write_bytes(b'finalizer snapshot')
    atomic_json(root / 'completion-audit.json', {'suite_id': remaining.PARENT_SUITE, 'suite_status': 'blocked',
        'export_returncode': 0, 'report_sha256': sha256(root / 'reports/flash-gate.json'),
        'ledger_snapshot_sha256': sha256(root / 'budget-final-snapshot.sqlite')})
    return root, release, suite, hashes


def test_exact_parent_and_unchanged_core_with_report_only_allowlist(parent):
    root, release, suite, hashes = parent
    result = remaining.checked_parent(root, Ack())
    assert set(result['parent_runs']) == {'F37-Off', 'F37-On', 'F38-Off'}
    assert result['parent_journal_counts'] == {'F37-On': 1, 'F38-Off': 1}
    for name in remaining.REPORT_ONLY_CHANGES: (release / name).write_text('new report only')
    assert remaining.checked_parent(root, Ack()) == result
    (release / 'camroll_agent.py').write_text('changed core')
    with pytest.raises(RuntimeError, match='byte_identical'): remaining.checked_parent(root, Ack())


@pytest.mark.parametrize('name', ['freeze.json', 'formal-suite.json', 'evaluation/runs/f38-off-g/run.json'])
def test_changed_parent_freeze_suite_or_run_is_rejected(parent, name):
    root, *_ = parent
    path = root / name; path.write_text(path.read_text() + ' ')
    with pytest.raises(RuntimeError, match='changed'): remaining.checked_parent(root, Ack())


@pytest.mark.parametrize('state', ['pending', 'uncertain'])
def test_any_new_parent_unknown_or_pending_is_rejected(parent, state):
    root, *_ = parent
    with sqlite3.connect(root / 'evaluation/runs/f38-off-g/request-journal.sqlite') as db:
        db.execute('UPDATE cloud_requests SET status=?', (state,))
    with pytest.raises(RuntimeError, match='new_pending_or_unknown'): remaining.checked_parent(root, Ack())


@pytest.mark.parametrize('mutation', ['started', 'claim', 'raw_symlink', 'raw_error', 'not_restored', 'second_turn'])
def test_only_never_started_on_after_single_context_fault_is_eligible(parent, monkeypatch, mutation):
    root, release, suite, hashes = parent
    raw = root / 'evaluation/runs/f38-off-g/turn_attempts/0/0.json'
    if mutation == 'started': (root / 'evaluation/runs/f38-on-g').mkdir()
    elif mutation == 'claim': atomic_json(root / 'evaluation/gate_claims' / remaining.PARENT_SUITE / 'F38-On.json', {})
    elif mutation == 'raw_symlink':
        target = raw.with_suffix('.original'); raw.rename(target); raw.symlink_to(target)
    elif mutation == 'raw_error': atomic_json(raw, {'response': {'error_code': 'request_uncertain'}})
    elif mutation == 'not_restored':
        suite['restoration']['state'] = 'active'; atomic_json(root / 'formal-suite.json', suite)
        monkeypatch.setattr(remaining, 'PARENT_SUITE_SHA256', sha256(root / 'formal-suite.json'))
    else:
        path = raw.parents[2] / 'run.json'; run = read_json(path)
        run['items'][0]['runtime_turns'].append(copy.deepcopy(run['items'][0]['runtime_turns'][0]))
        atomic_json(path, run); hashes['F38-Off'] = sha256(path)
    with pytest.raises((RuntimeError, ValueError)): remaining.checked_parent(root, Ack())


def test_parent_root_alias_rejected_before_resolve(parent, tmp_path):
    root, *_ = parent
    linked = tmp_path / 'linked-g'; linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'): remaining.checked_parent(linked, Ack())


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    import four_gpu_window, four_supplement, camroll_eval.dataset
    runtime, parent = tmp_path / 'h', tmp_path / 'g'
    parent.mkdir()
    old = {'groups': [{'name': name, 'run_id': name.lower() + '-g',
                       'status': 'blocked' if name == 'F38-Off' else 'completed_with_errors'}
                      for name in remaining.ORDER[:3]]}
    atomic_json(parent / 'formal-suite.json', old)
    old_bytes = (parent / 'formal-suite.json').read_bytes()
    inputs = {'parent_runs': {name: {'path': '/original/' + name, 'sha256': 'same'} for name in remaining.ORDER[:3]}}
    frozen = {'model_files': {'models': {}}, 'wemm_fingerprint': 'same',
              'wemm_weights': {'actual_sha256': 'same', 'file_identity_after': {'inode': 1}}}
    events, flags = [], {'unknown': False, 'blocked': False, 'restore_failure': False, 'register_failure': False,
                         'original_running': False, 'weight_changed': False, 'restore_bad_state': False,
                         'cleanup_failure': False}
    class Ledger:
        def check_limit(self): pass
        def summary(self):
            value = ready_summary()
            if flags['unknown'] and flags['blocked']: value.update(blocking_uncertain_requests=1, uncertain_requests=2)
            return value
    class Dataset:
        rows = [{'qa_id': str(i), 'question': 'Synthetic'} for i in range(100)]
        def __init__(self, *args): pass
        @staticmethod
        def turns(row): return [{}] * (2 if int(row['qa_id']) < 11 else 1)
    class Window:
        def __init__(self, release, identifier, *, runtime):
            assert runtime == remaining.HISTORY_RUNTIME
            self.state_path = tmp_path / 'window.json'
            events.append(('window', identifier))
        def enter(self): self.state_path.write_text('{}'); return {'before': {'fingerprint': 'same'}}
        def switch(self, kind): events.append(('switch', kind))
        def register_child(self, pid, expected_command):
            if flags['register_failure']: raise RuntimeError('registration fault')
        def unregister_child(self, pid): events.append(('unregister', pid))
        def restore(self, reason):
            events.append(('restore', reason))
            if flags['restore_failure']: raise RuntimeError('restore fault')
            if flags['restore_bad_state']: return {'state': 'restore_failed'}
            return {'state': 'restored', 'no_experiment_gpu_residual': True,
                    'after': {'state': 'loaded' if flags['original_running'] else 'stopped'}}
    class Child:
        def __init__(self, command, **kwargs):
            assert kwargs['cwd'] == remaining.RELEASE
            assert '--child' in command and '--child-group' not in command
            events.append(('child', command)); self.pid = 123; self.polls = 0
            identifier = command[command.index('--run-id') + 1]
            run = complete_run(); run.update(run_id=identifier, summary={})
            if flags['unknown']: run.update(status='blocked', gate_complete=False)
            atomic_json(runtime / 'evaluation/runs' / identifier / 'run.json', run)
            flags['blocked'] = flags['unknown']; self.returncode = 2 if flags['unknown'] else 0
        def poll(self): self.polls += 1; return None if self.polls == 1 else self.returncode
    monkeypatch.setattr(remaining, 'disk_ready', lambda path: {'free_bytes': 10 ** 10})
    monkeypatch.setattr(remaining, 'checked_parent', lambda *args: copy.deepcopy(inputs))
    monkeypatch.setattr(remaining, 'freeze_inputs', lambda *args: copy.deepcopy(frozen))
    monkeypatch.setattr(remaining, 'verify_frozen', lambda *args: None)
    monkeypatch.setattr(remaining.continuation, 'verify_wemm_weights', lambda: {
        'actual_sha256': 'changed' if flags['weight_changed'] else 'same', 'file_identity_before': {'inode': 1}})
    monkeypatch.setattr(remaining, 'subprocess', types.SimpleNamespace(
        Popen=Child, STDOUT=remaining.subprocess.STDOUT))
    monkeypatch.setattr(remaining.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(remaining.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(four_gpu_window, 'verify_model_files', lambda: {'models': {}})
    monkeypatch.setattr(four_supplement, 'stop_owned_child', lambda child: {
        'state': 'failed' if flags['cleanup_failure'] else 'exited'})
    monkeypatch.setattr(camroll_eval.dataset, 'Dataset', Dataset)
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', 'synthetic-h')
    yield runtime, parent, Ledger(), events, flags
    assert (parent / 'formal-suite.json').read_bytes() == old_bytes


@pytest.mark.parametrize('original_running', [False, True])
def test_only_new_on_executes_and_four_group_comparison_remains_incomplete(supervisor, original_running):
    runtime, parent, ledger, events, flags = supervisor
    flags['original_running'] = original_running
    assert remaining.execute(runtime, parent, ledger, Ack()) == 0
    suite = read_json(runtime / 'formal-suite.json')
    assert len([row for row in events if row[0] == 'child']) == 1
    assert suite['execution_order'] == ['F38-On'] and suite['remaining_group_complete'] is True
    assert suite['groups'][2]['status'] == 'blocked'
    assert all(row['reused'] for row in suite['groups'][:3])
    assert suite['restoration']['after']['state'] == ('loaded' if original_running else 'stopped')
    profile = read_json(runtime / 'evaluation/comparison-profile.json')
    assert profile['execution_order'] == suite['execution_order'] == ['F38-On']
    assert profile['comparison_status'] == 'incomplete' and profile['remaining_group_status'] == 'passed'
    plans = read_json(runtime / 'evaluation/planned-groups.json')['groups']
    assert len(plans[-1]['qa_plan']) == 100 and sum(row['planned_turn_count'] for row in plans[-1]['qa_plan']) == 111
    with pytest.raises(FileExistsError): remaining.execute(runtime, parent, ledger, Ack())


@pytest.mark.parametrize('fault', ['unknown', 'register_failure', 'restore_failure', 'restore_bad_state',
                                  'cleanup_failure', 'weight_changed'])
def test_failure_restores_or_reports_restore_fault_without_success(supervisor, fault):
    runtime, parent, ledger, events, flags = supervisor
    flags[fault] = True
    assert remaining.execute(runtime, parent, ledger, Ack()) == 2
    suite = read_json(runtime / 'formal-suite.json')
    assert suite['status'] != 'passed' and any(row[0] == 'restore' for row in events)
    if fault == 'unknown':
        assert suite['restoration']['state'] == 'restored'
        assert suite['budget_after']['uncertain_requests'] == 2
        assert (runtime / 'evaluation/runs' / suite['groups'][-1]['run_id'] / 'cancel.requested').is_file()


def test_overlap_symlink_and_low_disk_block_before_writes(tmp_path, monkeypatch):
    parent = tmp_path / 'g'; parent.mkdir()
    with pytest.raises(ValueError): remaining.independent_runtime(parent / 'h', parent)
    linked = tmp_path / 'alias'; linked.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError): remaining.independent_runtime(linked / 'h', parent)
    monkeypatch.setattr(remaining.shutil, 'disk_usage', lambda path: types.SimpleNamespace(free=1024))
    with pytest.raises(RuntimeError, match='2GiB'): remaining.disk_ready(tmp_path / 'h')
    assert not (tmp_path / 'h').exists()


def test_child_only_passes_on_and_scoped_ledger_without_continuation(tmp_path, monkeypatch):
    import four_gpu_window, camroll_eval.gate_experiment
    atomic_json(tmp_path / 'formal-suite.json', {'suite_id': 'h', 'status': 'running', 'execution_order': ['F38-On'],
        'groups': [{'name': 'F38-On', 'run_id': 'on-h', 'status': 'running'}]})
    atomic_json(tmp_path / 'freeze.json', {})
    ledger = types.SimpleNamespace(check_limit=lambda: None, summary=ready_summary)
    seen = {}
    class Runner:
        def __init__(self, *args, budget_ledger): seen['ledger'] = budget_ledger
        def run_gate(self, **kwargs): seen.update(kwargs); return complete_run()
    monkeypatch.setattr(camroll_eval.gate_experiment, 'GateRunner', Runner)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', lambda *args, **kwargs: types.SimpleNamespace(
        _read=lambda: {'state': 'active', 'current_kind': 'wemm'}))
    monkeypatch.setattr(remaining, 'verify_frozen', lambda *args: None)
    remaining.run_child(tmp_path, 'h', 'on-h', ledger, Ack())
    assert seen['ledger'] is ledger and seen['group'] == 'F38-On'
    assert 'continuation_parent' not in seen and 'acknowledged_call_ids' not in seen
    with pytest.raises(RuntimeError, match='identity'): remaining.run_child(tmp_path, 'h', 'other', ledger, Ack())


def test_default_cli_never_configures_credentials_or_starts_work(monkeypatch, capsys):
    monkeypatch.setattr(remaining, 'bootstrap', lambda *args: pytest.fail('Unexpected bootstrap'))
    assert remaining.main([]) == 0
    assert json.loads(capsys.readouterr().out)['execution_order'] == ['F38-On']
