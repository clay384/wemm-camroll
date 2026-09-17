"""Safety tests for the independent 8773 chat process manager."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manage_chat


def _release(tmp_path, monkeypatch):
    base = tmp_path / 'camroll-wemm-chat'
    staging = base / 'staging'
    for relative in manage_chat.REQUIRED_RELEASE_FILES:
        entry = staging / relative
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(relative)
    subprocess.run(['git', 'init', '-q', str(staging)], check=True)
    subprocess.run(['git', '-C', str(staging), 'config', 'user.email', 'tests@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(staging), 'config', 'user.name', 'Chat Tests'], check=True)
    subprocess.run(['git', '-C', str(staging), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(staging), 'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.run(
        ['git', '-C', str(staging), 'rev-parse', 'HEAD'], check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    release = base / 'releases' / commit
    release.parent.mkdir(parents=True)
    staging.rename(release)
    for entry in (release, *release.rglob('*')):
        entry.chmod(entry.stat().st_mode & ~0o222)
    monkeypatch.setattr(manage_chat, 'BASE', base)
    monkeypatch.setattr(manage_chat, 'PYTHON', base / '.venv/bin/python')
    return base, release


def _record(release):
    return {
        'pid': 123,
        'start_ticks': '1234',
        'cwd': str(release),
        'cmdline': manage_chat._command(release),
        'release': str(release),
        'host': '0.0.0.0',
        'port': 8773,
        **manage_chat._git_identity(release),
    }


def test_release_must_be_commit_directory_with_chat_entrypoint(tmp_path, monkeypatch):
    base, release = _release(tmp_path, monkeypatch)
    assert manage_chat._release(release) == release
    with pytest.raises(RuntimeError):
        manage_chat._release(base)
    target = release / 'camroll/camroll-chat/camroll_chat/app.py'
    target.parent.chmod(0o755)
    target.unlink()
    with pytest.raises(RuntimeError, match='working tree|regular required'):
        manage_chat._release(release)


def test_release_rejects_writable_tree_and_commit_name_mismatch(tmp_path, monkeypatch):
    base, release = _release(tmp_path, monkeypatch)
    release.chmod(0o755)
    with pytest.raises(RuntimeError, match='read-only'):
        manage_chat._release(release)
    release.chmod(0o755)
    release.parent.chmod(0o755)
    renamed = base / 'releases' / ('f' * 40)
    release.rename(renamed)
    with pytest.raises(RuntimeError, match='HEAD/tree'):
        manage_chat._release(renamed)


def test_stop_identity_does_not_depend_on_mutable_release_contents(tmp_path, monkeypatch):
    _, release = _release(tmp_path, monkeypatch)
    record = _record(release)
    target = release / 'camroll/camroll-chat/static/app.js'
    target.parent.chmod(0o755)
    target.unlink()
    identity = {key: record[key] for key in ['pid', 'start_ticks', 'cwd', 'cmdline']}
    monkeypatch.setattr(manage_chat, '_process', lambda _pid: identity)
    assert manage_chat._verify(record) == identity


@pytest.mark.parametrize('field,value', [
    ('cwd', '/home/realmagic/camroll-wemm'),
    ('start_ticks', '9999'),
    ('cmdline', ['python', 'other_service.py']),
])
def test_pid_identity_mismatch_never_passes_verification(
    tmp_path, monkeypatch, field, value
):
    _, release = _release(tmp_path, monkeypatch)
    record = _record(release)
    identity = {key: record[key] for key in ['pid', 'start_ticks', 'cwd', 'cmdline']}
    identity[field] = value
    monkeypatch.setattr(manage_chat, '_process', lambda _pid: identity)
    with pytest.raises(RuntimeError, match='refusing control'):
        manage_chat._verify(record)


def test_recorded_cwd_must_match_release_even_when_process_does(
    tmp_path, monkeypatch
):
    _, release = _release(tmp_path, monkeypatch)
    record = _record(release)
    identity = {key: record[key] for key in ['pid', 'start_ticks', 'cwd', 'cmdline']}
    record['cwd'] = '/home/realmagic/another-checkout'
    monkeypatch.setattr(manage_chat, '_process', lambda _pid: identity)
    with pytest.raises(RuntimeError, match='refusing control'):
        manage_chat._verify(record)


@pytest.mark.parametrize('port', [8765, 8770, 8772, 8774])
def test_other_service_ports_are_rejected(tmp_path, monkeypatch, port):
    _, release = _release(tmp_path, monkeypatch)
    record = _record(release)
    record['port'] = port
    identity = {key: record[key] for key in ['pid', 'start_ticks', 'cwd', 'cmdline']}
    monkeypatch.setattr(manage_chat, '_process', lambda _pid: identity)
    with pytest.raises(RuntimeError, match='refusing control'):
        manage_chat._verify(record)


def test_unmanaged_listener_blocks_start_without_spawning(tmp_path, monkeypatch):
    _, release = _release(tmp_path, monkeypatch)
    git_identity = manage_chat._git_identity(release)
    monkeypatch.setattr(manage_chat, 'RELEASE', release)
    monkeypatch.setattr(manage_chat, '_release', lambda _path: release)
    monkeypatch.setattr(manage_chat, '_git_identity', lambda _path: git_identity)
    monkeypatch.setattr(manage_chat, '_lock', nullcontext)
    monkeypatch.setattr(manage_chat, '_read_record', lambda: None)
    monkeypatch.setattr(manage_chat, '_listening', lambda: True)
    spawned = []
    monkeypatch.setattr(manage_chat.subprocess, 'Popen', lambda *a, **k: spawned.append(a))
    with pytest.raises(RuntimeError, match='already occupied'):
        manage_chat.start()
    assert spawned == []


def test_stop_uses_owned_pidfd_and_never_forced_kill(tmp_path, monkeypatch):
    record = {'pid': 123}
    path = tmp_path / 'chat-service.json'
    path.write_text('{}')
    path.chmod(0o600)
    signals = []
    monkeypatch.setattr(manage_chat, 'RECORD', path)
    monkeypatch.setattr(manage_chat, '_lock', nullcontext)
    monkeypatch.setattr(manage_chat, '_read_record', lambda: record)
    monkeypatch.setattr(manage_chat, '_verify', lambda _record: record)
    monkeypatch.setattr(manage_chat.os, 'pidfd_open', lambda pid: 999, raising=False)
    monkeypatch.setattr(
        manage_chat.signal,
        'pidfd_send_signal',
        lambda fd, sig: signals.append((fd, sig)),
        raising=False,
    )
    monkeypatch.setattr(manage_chat.os, 'close', lambda _fd: None)

    class Poll:
        def register(self, fd, event):
            assert (fd, event) == (999, manage_chat.select.POLLIN)

        def poll(self, timeout):
            assert timeout == 10000
            return [(999, manage_chat.select.POLLIN)]

    monkeypatch.setattr(manage_chat.select, 'poll', Poll)
    assert manage_chat.stop()['state'] == 'stopped'
    assert signals == [(999, manage_chat.signal.SIGTERM)]
    assert not path.exists()


def test_restricted_record_rejects_group_read_without_leaking_content(
    tmp_path, monkeypatch
):
    record = tmp_path / 'chat-service.json'
    record.write_text(json.dumps({'secret': 'do-not-echo'}))
    record.chmod(0o640)
    monkeypatch.setattr(manage_chat, 'RECORD', record)
    with pytest.raises(RuntimeError) as caught:
        manage_chat._read_record()
    assert 'do-not-echo' not in str(caught.value)


def test_restart_stops_before_starting(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manage_chat,
        'stop',
        lambda: calls.append('stop') or {'state': 'stopped'},
    )
    monkeypatch.setattr(
        manage_chat,
        'start',
        lambda: calls.append('start') or {'state': 'running'},
    )
    result = manage_chat.restart()
    assert calls == ['stop', 'start']
    assert result['restart'] == {'previous': 'stopped', 'current': 'running'}


def test_child_environment_is_minimal_and_drops_provider_secrets(tmp_path, monkeypatch):
    _, release = _release(tmp_path, monkeypatch)
    for key in (
        'DASHSCOPE_API_KEY', 'QWEN_API_KEY', 'CAMROLL_BUDGET_PATH',
        'PYTHONHOME', 'LD_PRELOAD', 'HTTP_PROXY', 'HTTPS_PROXY',
    ):
        monkeypatch.setenv(key, 'must-not-reach-child')
    env = manage_chat._child_environment(release)
    assert set(env) <= {
        'PATH', 'HOME', 'LANG', 'PYTHONPATH', 'PYTHONUNBUFFERED',
        'PYTHONDONTWRITEBYTECODE', 'LC_ALL', 'TZ',
    }
    assert 'must-not-reach-child' not in env.values()
    assert str(release / 'camroll/camroll-chat') in env['PYTHONPATH']


def test_failed_start_cleanup_removes_record_only_after_owned_child_stops(
    tmp_path, monkeypatch
):
    _, release = _release(tmp_path, monkeypatch)
    record = tmp_path / 'chat-service.json'
    record.write_text('{}')
    record.chmod(0o600)
    monkeypatch.setattr(manage_chat, 'RECORD', record)
    process = type('Process', (), {'poll': lambda self: None})()
    calls = []
    monkeypatch.setattr(
        manage_chat,
        '_terminate_spawned',
        lambda child, child_release, command: calls.append((child, child_release, command)) or True,
    )
    command = manage_chat._command(release)
    assert manage_chat._cleanup_failed_start(process, release, command) is True
    assert calls == [(process, release, command)]
    assert not record.exists()


def test_failed_start_cleanup_retains_record_when_stop_is_unverified(
    tmp_path, monkeypatch
):
    _, release = _release(tmp_path, monkeypatch)
    record = tmp_path / 'chat-service.json'
    record.write_text('{}')
    record.chmod(0o600)
    monkeypatch.setattr(manage_chat, 'RECORD', record)
    process = type('Process', (), {'poll': lambda self: None})()
    monkeypatch.setattr(manage_chat, '_terminate_spawned', lambda *_args: False)
    assert manage_chat._cleanup_failed_start(process, release, manage_chat._command(release)) is False
    assert record.exists()
