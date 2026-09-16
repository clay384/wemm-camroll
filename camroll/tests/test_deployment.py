"""Deployment control tests. No OS processes are started or signalled."""
import importlib.util
import json
import os
from pathlib import Path
import signal

import pytest


spec = importlib.util.spec_from_file_location('camroll_manager', Path(__file__).parents[1] / 'deployment/manage_camroll.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


def record(root, **updates):
    value = {'pid': 12345, 'start_ticks': '100', 'cwd': str(root), 'host': '127.0.0.1', 'port': 8770,
             'cmdline': [str(root / '.venv/bin/python'), '-u', str(root / 'deployment/run_service.py')]}
    value.update(updates)
    return value


def write_record(root, value):
    (root / 'runtime').mkdir(exist_ok=True)
    (root / 'runtime/camroll-service.json').write_text(json.dumps(value))


def test_process_identity_reads_creation_tick_and_parenthesized_process_name(tmp_path):
    proc = tmp_path / 'proc'
    path = proc / '12345'
    path.mkdir(parents=True)
    (path / 'stat').write_text('12345 (camroll worker) ' + ' '.join(['S'] + ['0']*18 + ['100'] + ['0']*30))
    (path / 'cwd').symlink_to(tmp_path, target_is_directory=True)
    (path / 'cmdline').write_bytes(b'/python\x00-u\x00/script.py\x00')
    value = manager._process(12345, proc)
    assert value['start_ticks'] == '100'
    assert value['cwd'] == str(tmp_path)
    assert value['cmdline'] == ['/python', '-u', '/script.py']


@pytest.mark.parametrize('updates', [
    {'start_ticks': '101'}, {'cwd': '/other/project'}, {'cmdline': ['/python', '/other/script.py']},
])
def test_identity_mismatch_and_pid_reuse_refuse_control(tmp_path, monkeypatch, updates):
    original = record(tmp_path)
    current = {**original, **updates}
    monkeypatch.setattr(manager, '_process', lambda pid: current)
    with pytest.raises(RuntimeError, match='refusing process control'):
        manager._verify(original, tmp_path)


def test_stop_pid_reuse_never_opens_pidfd_or_signals(tmp_path, monkeypatch):
    write_record(tmp_path, record(tmp_path))
    monkeypatch.setattr(manager, '_process', lambda pid: record(tmp_path, start_ticks='200'))
    calls = []
    monkeypatch.setattr(manager.os, 'pidfd_open', lambda pid: calls.append(pid), raising=False)
    with pytest.raises(RuntimeError, match='refusing process control'):
        manager.stop(tmp_path)
    assert calls == []


def test_stop_uses_pidfd_and_removes_only_own_record(tmp_path, monkeypatch):
    expected = record(tmp_path)
    write_record(tmp_path, expected)
    states = iter([expected, expected, None])
    monkeypatch.setattr(manager, '_process', lambda pid: next(states))
    fd = os.open(tmp_path / 'fake-pidfd', os.O_CREAT | os.O_RDWR, 0o600)
    opened, sent = [], []
    def pidfd_open(pid):
        opened.append(pid)
        return fd
    monkeypatch.setattr(manager.os, 'pidfd_open', pidfd_open, raising=False)
    monkeypatch.setattr(manager.signal, 'pidfd_send_signal', lambda handle, sig: sent.append((handle, sig)), raising=False)
    result = manager.stop(tmp_path)
    assert result['state'] == 'stopped'
    assert opened == [12345]
    assert sent == [(fd, signal.SIGTERM)]
    assert not (tmp_path / 'runtime/camroll-service.json').exists()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_stop_without_pidfd_support_refuses_unsafe_fallback(tmp_path, monkeypatch):
    expected = record(tmp_path)
    write_record(tmp_path, expected)
    monkeypatch.setattr(manager, '_process', lambda pid: expected)
    monkeypatch.delattr(manager.os, 'pidfd_open', raising=False)
    with pytest.raises(RuntimeError, match='pidfd support'):
        manager.stop(tmp_path)
    assert (tmp_path / 'runtime/camroll-service.json').exists()


def test_start_does_not_touch_unmanaged_listener(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, '_port_open', lambda: True)
    calls = []
    monkeypatch.setattr(manager.subprocess, 'Popen', lambda *a, **kw: calls.append(a))
    with pytest.raises(RuntimeError, match='unmanaged listener'):
        manager.start(tmp_path)
    assert calls == []


def test_start_refuses_readable_secrets_without_opening_contents(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    secret = runtime / 'secrets.json'
    secret.write_text('SECRET-DO-NOT-READ')
    secret.chmod(0o644)
    monkeypatch.setattr(manager, '_port_open', lambda: False)
    with pytest.raises(RuntimeError, match='permissions'):
        manager.start(tmp_path)


def test_status_with_no_record_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, '_port_open', lambda: False)
    assert manager.status(tmp_path)['state'] == 'stopped'
    assert not (tmp_path / 'runtime').exists()


def test_cli_index_failure_outputs_recoverable_summary(tmp_path, monkeypatch, capsys):
    from camroll_agent import cli, pipeline, wemm
    summary = {'published': False, 'stop_reason': 'embedding_unavailable', 'cloud_calls': 0,
               'memory_dir': str(tmp_path)}
    def reindex(*args, **kwargs):
        raise pipeline.BuildProviderFailed(summary)
    monkeypatch.setattr(pipeline, 'reindex', reindex)
    monkeypatch.setattr(wemm, 'EmbeddingClient', lambda: object())
    assert cli.main(['index', str(tmp_path)]) == 3
    assert json.loads(capsys.readouterr().out) == summary
