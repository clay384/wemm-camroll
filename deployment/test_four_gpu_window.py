"""GPU control tests use synthetic process tables and temporary paths only."""
import json
from pathlib import Path
import signal
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_gpu_window as gpu


@pytest.fixture
def window(tmp_path, monkeypatch):
    old = tmp_path / 'legacy'
    (old / 'run').mkdir(parents=True)
    (old / 'data').mkdir()
    (old / 'backend').mkdir()
    protected = old / 'backend/encoder.py'
    protected.write_text('frozen encoder')
    vector = old / 'data/vector.npy'
    vector.write_bytes(b'unchanged vector')
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'protected_files': {'backend/encoder.py': gpu._sha(protected)},
                                  'index_files': [{'path': 'data/vector.npy', 'sha256': gpu._sha(vector)}]}))
    monkeypatch.setattr(gpu, 'OLD', old)
    monkeypatch.setattr(gpu, 'LEGACY_BASELINE', baseline)
    monkeypatch.setattr(gpu, 'verify_model_files', lambda: {'manifest_sha256': 'locked', 'models': {}})
    monkeypatch.setattr(gpu, '_legacy_fingerprint', lambda: 'fixed-fingerprint')
    monkeypatch.setattr(gpu, '_check_legacy_database', lambda fingerprint: None)
    monkeypatch.setattr(gpu, '_legacy_command', lambda: ['python', 'backend.app:app', '--port', '8765'])
    monkeypatch.setattr(gpu.time, 'sleep', lambda seconds: None)
    processes, ports, stopped, events = {}, {}, [], []
    monkeypatch.setattr(gpu, '_process', lambda pid: processes.get(pid))
    monkeypatch.setattr(gpu, '_port_free', lambda port: port not in ports)
    monkeypatch.setattr(gpu, '_owns_port', lambda process, port: ports.get(port) == process['pid'])
    monkeypatch.setattr(gpu, '_gpu_processes', lambda: [(gpu.GPU_UUID, pid) for pid in processes])
    status = {'state': 'ready', 'fingerprint': 'fixed-fingerprint', 'queue': 0,
              'busy_asset': None, 'counts': {}, 'encoding_calls': 5}
    monkeypatch.setattr(gpu, '_legacy_status', lambda: dict(status))
    def stop(record):
        gpu._verify(record)
        stopped.append(record['pid'])
        processes.pop(record['pid'], None)
        for port in list(ports):
            if ports[port] == record['pid']:
                ports.pop(port)
    monkeypatch.setattr(gpu, '_stop', stop)
    def forbid(*args, **kwargs):
        raise AssertionError('Unexpected real external operation')
    monkeypatch.setattr(gpu.subprocess, 'Popen', forbid)
    monkeypatch.setattr(gpu.subprocess, 'run', forbid)
    monkeypatch.setattr(gpu.os, 'kill', forbid)
    value = gpu.GpuWindow(tmp_path / 'release', 'window-1', runtime=tmp_path / 'runtime')
    monkeypatch.setattr(value, '_start_heartbeat', lambda: None)
    def guard():
        assert value._read()['state'] == 'entering'
        events.append('guard-started')
        return {'pid': 999, 'start_ticks': '1', 'cwd': str(value.release), 'cmdline': ['guard']}
    monkeypatch.setattr(value, '_launch_guard', guard)
    def launch(kind, state):
        gpu._assert_gpu()
        port = 8765 if kind == 'wemm' else 18778
        if not gpu._port_free(port):
            raise RuntimeError('Required model port is occupied')
        pid = 100 + len(state['history'])
        record = {'pid': pid, 'start_ticks': str(pid),
                  'cwd': str(old if kind == 'wemm' else value.release),
                  'cmdline': gpu._legacy_command() if kind == 'wemm' else ['qwen']}
        processes[pid], ports[port] = record, pid
        state['current_kind'], state['current_process'] = kind, record
        state['history'].append({'kind': kind, 'process': record})
        value._write(state)
        if kind == 'wemm':
            gpu._save(old / 'run/service.json', {**record, 'camroll_window_id': value.window_id})
        events.append('launch-' + kind)
        return record
    monkeypatch.setattr(value, '_launch_model', launch)
    value.synthetic = {'processes': processes, 'ports': ports, 'stopped': stopped,
                       'events': events, 'status': status, 'protected': protected}
    return value


def _running_original(window):
    record = {'pid': 50, 'start_ticks': '50', 'cwd': str(gpu.OLD), 'cmdline': gpu._legacy_command()}
    window.synthetic['processes'][50] = record
    window.synthetic['ports'][8765] = 50
    gpu._save(gpu.OLD / 'run/service.json', record)
    return record


def test_originally_stopped_remains_stopped_and_switches_are_exclusive(window):
    assert window.enter()['original_running'] is False
    assert window.synthetic['events'] == ['guard-started']
    first = window.switch('qwen')['current_process']['pid']
    second = window.switch('wemm')['current_process']['pid']
    assert first in window.synthetic['stopped']
    assert list(window.synthetic['processes']) == [second]
    result = window.restore('completed')
    assert result['state'] == 'restored'
    assert result['after']['state'] == 'stopped'
    assert result['no_experiment_gpu_residual']
    assert not window.synthetic['processes']
    assert not (gpu.OLD / 'run/service.json').exists()
    assert (window.directory / 'receipt.json').is_file()
    assert window.restore('again') == result


def test_original_running_is_restored_to_its_fingerprint_and_command(window):
    original = _running_original(window)
    window.enter()
    assert original['pid'] in window.synthetic['stopped']
    window.switch('qwen')
    result = window.restore('completed')
    assert result['original_running']
    assert result['after']['fingerprint'] == 'fixed-fingerprint'
    restored = result['restored_process']
    assert restored['cmdline'] == original['cmdline']
    assert list(window.synthetic['processes']) == [restored['pid']]
    assert json.loads((gpu.OLD / 'run/service.json').read_text())['pid'] == restored['pid']


@pytest.mark.parametrize('port', [8765, 18778])
def test_port_conflict_refuses_before_launch_or_stop(window, port):
    window.synthetic['ports'][port] = 700
    with pytest.raises(RuntimeError, match='occup'):
        window.enter()
    assert window.synthetic['events'] == []
    assert window.synthetic['stopped'] == []
    assert not window.state_path.exists()


def test_unknown_gpu_owner_blocks_entry_without_control(window):
    window.synthetic['processes'][700] = {'pid': 700}
    with pytest.raises(RuntimeError, match='Unexpected GPU1 owner'):
        window.enter()
    assert window.synthetic['stopped'] == []


def test_busy_original_is_not_stopped(window):
    _running_original(window)
    window.synthetic['status']['queue'] = 1
    with pytest.raises(RuntimeError, match='not idle'):
        window.enter()
    assert window.synthetic['stopped'] == []


def test_known_pid_using_gpu0_is_not_stopped(window, monkeypatch):
    _running_original(window)
    monkeypatch.setattr(gpu, '_gpu_processes', lambda: [('GPU-OTHER', 50)])
    with pytest.raises(RuntimeError, match='another GPU'):
        window.enter()
    assert window.synthetic['stopped'] == []


@pytest.mark.parametrize('state', ['entering', 'active', 'switching'])
def test_expired_lease_restores_initial_state(window, state):
    window.enter()
    window.switch('qwen')
    current = window._read()
    current.update(state=state, lease_expires_at=1)
    window._write(current)
    assert window.watch_once(now=601) is False
    assert window._read()['state'] == 'restored'
    assert window._read()['restore_reason'] == 'lease_expired'
    assert window.synthetic['processes'] == {}


def test_keepalive_preserves_window_identity_and_extends_lease(window):
    window.enter()
    previous = window._read()['lease_expires_at']
    renewed = window.keepalive()
    assert renewed['window_id'] == window.window_id
    assert renewed['lease_expires_at'] >= previous
    assert window.watch_once(now=0)
    window.restore()
    with pytest.raises(RuntimeError, match='not active'):
        window.keepalive()


def test_protected_file_drift_stops_new_worker_and_marks_restore_failed(window):
    window.enter()
    worker = window.switch('qwen')['current_process']['pid']
    window.synthetic['protected'].write_text('changed')
    with pytest.raises(RuntimeError, match='Protected WeMM file changed'):
        window.restore('error')
    assert worker in window.synthetic['stopped']
    assert window._read()['state'] == 'restore_failed'
    assert not (window.directory / 'receipt.json').exists()


def test_original_running_restore_port_conflict_never_uses_fallback_port(window):
    _running_original(window)
    window.enter()
    window.switch('qwen')
    window.synthetic['ports'][8765] = 700
    with pytest.raises(RuntimeError, match='port is occupied'):
        window.restore('error')
    assert 'launch-wemm' not in window.synthetic['events']
    assert window._read()['state'] == 'restore_failed'


def test_switch_failure_triggers_restore(window, monkeypatch):
    window.enter()
    def fail(kind, state):
        raise RuntimeError('model_start_failed')
    monkeypatch.setattr(window, '_launch_model', fail)
    with pytest.raises(RuntimeError, match='model_start_failed'):
        window.switch('qwen')
    assert window._read()['state'] == 'restored'
    assert window._read()['restore_reason'] == 'switch_failed'


def test_stale_window_object_cannot_restore_replaced_state(window):
    window.enter()
    state = window._read()
    state['window_id'] = 'different'
    window._write(state)
    with pytest.raises(RuntimeError, match='ownership mismatch'):
        window.restore()
    assert window.synthetic['stopped'] == []


def test_watchdog_stops_registered_evaluator_before_models(window, monkeypatch):
    window.enter()
    worker_pid = window.switch('qwen')['current_process']['pid']
    evaluator = {'pid': 500, 'start_ticks': '500', 'cwd': str(window.release), 'cmdline': ['python', 'evaluate']}
    window.synthetic['processes'][500] = evaluator
    monkeypatch.setattr(gpu, '_gpu_processes', lambda: [(gpu.GPU_UUID, pid)
                                                     for pid in window.synthetic['processes'] if pid != 500])
    window.register_child(500, ['python', 'evaluate'])
    with pytest.raises(RuntimeError, match='running evaluator'):
        window.unregister_child(500)
    state = window._read()
    state['lease_expires_at'] = 1
    window._write(state)
    assert window.watch_once(now=700) is False
    assert window.synthetic['stopped'] == [500, worker_pid]
    assert window._read()['children'][0]['state'] == 'stopped'


def test_completed_child_is_preserved_in_receipt_without_signalling(window, monkeypatch):
    window.enter()
    evaluator = {'pid': 500, 'start_ticks': '500', 'cwd': str(window.release), 'cmdline': ['python', 'evaluate']}
    window.synthetic['processes'][500] = evaluator
    window.register_child(500, ['python', 'evaluate'])
    window.synthetic['processes'].pop(500)
    window.unregister_child(500)
    result = window.restore()
    assert result['children'][0]['state'] == 'finished'
    assert window.synthetic['stopped'] == []


def test_owner_exit_triggers_restore_without_waiting_full_lease(window):
    window.enter()
    window.switch('qwen')
    state = window._read()
    state['owner_process'] = {'pid': 9999, 'start_ticks': '1', 'cwd': str(window.release), 'cmdline': ['controller']}
    window._write(state)
    assert window.watch_once(now=0) is False
    assert window._read()['restore_reason'] == 'owner_exited'


@pytest.mark.parametrize('field', ['cwd', 'cmdline', 'start_ticks'])
def test_identity_change_refuses_signal(monkeypatch, field):
    record = {'pid': 5, 'cwd': '/owned', 'cmdline': ['python', 'owned'], 'start_ticks': '50'}
    changed = dict(record, **{field: 'changed'})
    monkeypatch.setattr(gpu, '_process', lambda pid: changed)
    with pytest.raises(RuntimeError, match='identity changed'):
        gpu._stop(record)


@pytest.mark.parametrize('exits', [True, False])
def test_pidfd_stop_never_forces_kill(monkeypatch, exits):
    record = {'pid': 5, 'cwd': '/owned', 'cmdline': ['python', 'owned'], 'start_ticks': '50'}
    signals = []
    monkeypatch.setattr(gpu, '_process', lambda pid: record)
    monkeypatch.setattr(gpu.os, 'pidfd_open', lambda pid: 100, raising=False)
    monkeypatch.setattr(gpu.signal, 'pidfd_send_signal', lambda fd, sig: signals.append((fd, sig)), raising=False)
    monkeypatch.setattr(gpu.os, 'close', lambda fd: None)
    class Poll:
        def register(self, descriptor, events):
            assert descriptor == 100
        def poll(self, timeout):
            return [(100, 1)] if exits else []
    monkeypatch.setattr(gpu.select, 'poll', Poll)
    if exits:
        gpu._stop(record)
    else:
        with pytest.raises(RuntimeError, match='no forced kill'):
            gpu._stop(record)
    assert signals == [(100, signal.SIGTERM)]


def test_readonly_database_preflight_rejects_pending_or_mismatched_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, 'OLD', tmp_path)
    (tmp_path / 'data').mkdir()
    path = tmp_path / 'data/lab.sqlite'
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE assets(status TEXT); CREATE TABLE units(fingerprint TEXT);")
        connection.execute("INSERT INTO assets VALUES ('indexed')")
        connection.execute("INSERT INTO units VALUES ('fixed')")
    original = path.read_bytes()
    gpu._check_legacy_database('fixed')
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match='incompatible frozen vectors'):
        gpu._check_legacy_database('other')
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO assets VALUES ('pending')")
    with pytest.raises(RuntimeError, match='pending assets'):
        gpu._check_legacy_database('fixed')


def test_old_configuration_wrong_port_or_gpu_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(gpu, 'OLD', tmp_path)
    monkeypatch.setitem(sys.modules, 'yaml', SimpleNamespace(safe_load=lambda value: {'port': 8766, 'gpu_uuid': 'other'}))
    (tmp_path / 'config.yaml').write_text('port: 8766\ngpu_uuid: other\n')
    with pytest.raises(RuntimeError, match='configuration GPU or port'):
        gpu._config()
