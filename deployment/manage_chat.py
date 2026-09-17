"""Manage only the isolated CamRoll chat process on 8773.

This manager deliberately knows nothing about the existing 8765, 8770, or
8772 processes.  It refuses to signal a process unless Linux /proc identity,
the recorded start tick, cwd, complete argv, release, host, and port all match.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import time
import urllib.request


BASE = Path('/home/realmagic/camroll-wemm-chat')
PYTHON = BASE / '.venv/bin/python'
RUNTIME = BASE / 'runtime'
RELEASE = Path(__file__).resolve().parents[1]
PROC = Path('/proc')
HOST, PORT = '0.0.0.0', 8773
RECORD = RUNTIME / 'chat-service.json'
NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
REQUIRED_RELEASE_FILES = (
    'deployment/manage_chat.py',
    'deployment/run_chat.py',
    'camroll/camroll-chat/camroll_chat/app.py',
    'camroll/camroll-chat/camroll_chat/album.py',
    'camroll/camroll-chat/camroll_chat/config.py',
    'camroll/camroll-chat/static/index.html',
    'camroll/camroll-chat/static/app.js',
    'camroll/camroll-chat/static/style.css',
)


def _release_location(path):
    raw = Path(path)
    if raw.is_symlink():
        raise RuntimeError('Release directory must not be a symbolic link')
    path = raw.resolve(strict=True)
    releases = (BASE / 'releases').resolve()
    if path.parent != releases or not re.fullmatch(r'[0-9a-f]{40}', path.name):
        raise RuntimeError('Release must be an existing releases/<40-character-commit> directory')
    return path


def _git(release, *arguments, check=True):
    env = {
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'HOME': '/nonexistent',
        'LANG': 'C.UTF-8',
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_OPTIONAL_LOCKS': '0',
    }
    try:
        result = subprocess.run(
            ['git', '-C', str(release), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError('Release Git verification failed') from None
    if check and result.returncode:
        raise RuntimeError('Release Git verification failed')
    return result


def _git_identity(release):
    head = _git(release, 'rev-parse', '--verify', 'HEAD').stdout.strip()
    tree = _git(release, 'rev-parse', '--verify', 'HEAD^{tree}').stdout.strip()
    if head != release.name or not re.fullmatch(r'[0-9a-f]{40}', tree):
        raise RuntimeError('Release HEAD/tree does not match its commit directory')
    status = _git(release, 'status', '--porcelain=v1', '--untracked-files=all').stdout
    if status:
        raise RuntimeError('Release working tree is not clean')
    for relative in REQUIRED_RELEASE_FILES:
        entry = release / relative
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError('Release is missing a regular required chat file')
    for entry in (release, *release.rglob('*')):
        if entry.is_symlink() or entry.stat().st_mode & 0o222:
            raise RuntimeError('Release tree must be non-symlinked and read-only')
    return {'git_commit': head, 'git_tree': tree}


def _release(path):
    """Accept only a clean, read-only Git checkout of the named commit."""
    path = _release_location(path)
    _git_identity(path)
    return path


def _command(release):
    return [str(PYTHON), '-u', str(release / 'deployment/run_chat.py')]


def _process(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 2:
        raise RuntimeError('Invalid process record; no process was signalled')
    process = PROC / str(pid)
    try:
        fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {
            'pid': pid,
            'start_ticks': fields[19],
            'cwd': str((process / 'cwd').resolve(strict=True)),
            'cmdline': [
                item.decode()
                for item in (process / 'cmdline').read_bytes().split(b'\0')
                if item
            ],
        }
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        raise RuntimeError('Cannot verify process identity; no process was signalled') from None


def _read_record():
    if not RECORD.exists():
        return None
    if RECORD.is_symlink() or RECORD.stat().st_mode & 0o077:
        raise RuntimeError('Unsafe chat process record; no process was changed')
    try:
        record = json.loads(RECORD.read_text())
        if not isinstance(record, dict):
            raise ValueError()
        return record
    except (OSError, ValueError):
        raise RuntimeError('Invalid chat process record; no process was changed') from None


def _verify(record):
    identity = _process(record.get('pid'))
    if identity is None:
        return None
    release = _release_location(record.get('release', ''))
    expected = _command(release)
    if (
        identity['cwd'] != str(release)
        or record.get('cwd') != str(release)
        or identity['start_ticks'] != record.get('start_ticks')
        or identity['cmdline'] != expected
        or record.get('cmdline') != expected
        or record.get('host') != HOST
        or record.get('port') != PORT
        or record.get('git_commit') != release.name
        or not re.fullmatch(r'[0-9a-f]{40}', str(record.get('git_tree', '')))
    ):
        raise RuntimeError('Recorded PID does not identify the owned 8773 server; refusing control')
    return identity


def _listening():
    try:
        with socket.create_connection(('127.0.0.1', PORT), timeout=0.2):
            return True
    except OSError:
        return False


def _http_alive():
    """Check the chat application itself, independently of upstream readiness."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{PORT}/', timeout=0.5) as response:
            return response.status == 200
    except OSError:
        return False


@contextmanager
def _lock():
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    if RUNTIME.is_symlink() or RUNTIME.stat().st_mode & 0o077:
        raise RuntimeError('Chat runtime directory must have mode 0700')
    lock_path = RUNTIME / '.chat-manager.lock'
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def status():
    record = _read_record()
    if record is None:
        return {
            'state': 'unmanaged_listener' if _listening() else 'stopped',
            'host': HOST,
            'port': PORT,
        }
    identity = _verify(record)
    integrity = 'not_running'
    if identity:
        try:
            release = _release(record.get('release', ''))
            git_identity = _git_identity(release)
            integrity = 'verified' if all(
                record.get(key) == value for key, value in git_identity.items()
            ) else 'changed'
        except (RuntimeError, OSError, subprocess.SubprocessError):
            integrity = 'changed'
    return {
        'state': 'running' if identity else 'stopped',
        'host': HOST,
        'port': PORT,
        'pid': record.get('pid'),
        'release': record.get('release'),
        'git_commit': record.get('git_commit'),
        'git_tree': record.get('git_tree'),
        'release_integrity': integrity,
        'listening': _listening(),
        'http_alive': bool(identity) and _http_alive(),
        'note': 'HTTP liveness does not establish upstream 8772, album, WeMM, or Qwen readiness',
    }


def _write_record(identity):
    temporary = RECORD.with_suffix('.tmp')
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError('Unsafe temporary process record; no process was changed')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(identity, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, RECORD)


def _child_environment(release):
    """The facade needs no provider credentials or inherited Python hooks."""
    env = {
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'HOME': '/home/realmagic',
        'LANG': os.environ.get('LANG', 'C.UTF-8'),
        'PYTHONPATH': os.pathsep.join([
            str(release / 'camroll/camroll-chat'),
            str(release),
        ]),
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
    }
    for key in ('LC_ALL', 'TZ'):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _terminate_spawned(process, release, command):
    """Best-effort cleanup only after the exact newly spawned child is verified."""
    identity = _process(process.pid)
    if (identity is None or identity['cwd'] != str(release)
            or identity['cmdline'] != command):
        return False
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        return False
    try:
        process_fd = os.pidfd_open(process.pid)
    except ProcessLookupError:
        return True
    try:
        signal.pidfd_send_signal(process_fd, signal.SIGTERM)
        poller = select.poll()
        poller.register(process_fd, select.POLLIN)
        return bool(poller.poll(10000))
    except ProcessLookupError:
        return True
    finally:
        os.close(process_fd)


def _cleanup_failed_start(process, release, command):
    """Remove the record only after the exact child is known to be gone."""
    exited = process.poll() is not None
    stopped = exited or _terminate_spawned(process, release, command)
    if stopped:
        RECORD.unlink(missing_ok=True)
    return stopped


def start():
    release = _release(RELEASE)
    git_identity = _git_identity(release)
    with _lock():
        record = _read_record()
        if record is not None:
            identity = _verify(record)
            if identity is not None:
                if record.get('release') != str(release):
                    raise RuntimeError('Another owned release is running; stop it before switching')
                return status()
            RECORD.unlink(missing_ok=True)
        if _listening():
            raise RuntimeError('8773 is already occupied; no process was changed')
        if not PYTHON.is_file():
            raise RuntimeError('Independent chat Python environment is missing')

        env = _child_environment(release)
        command = _command(release)
        log_path = RUNTIME / 'chat-service.log'
        log_fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | NOFOLLOW, 0o600)
        os.fchmod(log_fd, 0o600)
        try:
            process = subprocess.Popen(
                command,
                cwd=release,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=log_fd,
                start_new_session=True,
            )
        finally:
            os.close(log_fd)

        identity = None
        for _ in range(40):
            identity = _process(process.pid)
            if identity and identity['cmdline'] == command and identity['cwd'] == str(release):
                break
            time.sleep(0.025)
        if not identity or identity['cmdline'] != command or identity['cwd'] != str(release):
            _terminate_spawned(process, release, command)
            raise RuntimeError('Chat server exited before identity was recorded; inspect its restricted log')
        identity.update(
            release=str(release),
            host=HOST,
            port=PORT,
            created_at=time.time(),
            **git_identity,
        )
        try:
            _write_record(identity)
        except Exception:
            _terminate_spawned(process, release, command)
            raise

        for _ in range(100):
            if process.poll() is not None:
                RECORD.unlink(missing_ok=True)
                raise RuntimeError('Chat server failed to start; inspect its restricted log')
            if _http_alive():
                return status()
            time.sleep(0.1)
        if _cleanup_failed_start(process, release, command):
            raise RuntimeError('Chat server HTTP was not ready and the owned child was stopped safely')
        raise RuntimeError(
            'Chat server HTTP is not ready; its owned record was retained because safe cleanup could not be verified'
        )


def stop():
    with _lock():
        record = _read_record()
        if not record:
            return {
                'state': 'stopped',
                'port': PORT,
                'note': 'No owned record; no process was signalled',
            }
        identity = _verify(record)
        if identity is None:
            RECORD.unlink(missing_ok=True)
            return {
                'state': 'stopped',
                'port': PORT,
                'note': 'Removed stale owned record only',
            }
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Linux pidfd support is required; no process was signalled')
        try:
            process_fd = os.pidfd_open(identity['pid'])
        except ProcessLookupError:
            RECORD.unlink(missing_ok=True)
            return {'state': 'stopped', 'port': PORT}
        try:
            if _verify(record) is None:
                RECORD.unlink(missing_ok=True)
                return {'state': 'stopped', 'port': PORT}
            try:
                signal.pidfd_send_signal(process_fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            poller = select.poll()
            poller.register(process_fd, select.POLLIN)
            if poller.poll(10000):
                RECORD.unlink(missing_ok=True)
                return {'state': 'stopped', 'pid': identity['pid'], 'port': PORT}
        finally:
            os.close(process_fd)
        raise RuntimeError('Owned chat process did not exit within 10 seconds; no forced kill was sent')


def restart():
    stopped = stop()
    started = start()
    started['restart'] = {'previous': stopped.get('state'), 'current': started.get('state')}
    return started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop', 'restart', 'rollback'])
    args = parser.parse_args(argv)
    if not PROC.is_dir():
        print(json.dumps({'error': 'Linux /proc is required; no service was changed'}))
        return 2
    try:
        actions = {
            'start': start,
            'status': status,
            'stop': stop,
            'restart': restart,
            'rollback': stop,
        }
        result = actions[args.command]()
        if args.command == 'rollback':
            result['rollback'] = (
                'Stopped independent 8773 only; 8765/8770/8772, frozen album, and budget are unchanged'
            )
    except (RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
