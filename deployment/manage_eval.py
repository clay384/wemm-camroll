"""Manage only the owned PhotoBench process on 8772; rollback stops it only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

from eval_config import BASE, PYTHON, RUNTIME

RELEASE = Path(__file__).resolve().parents[1]
PROC = Path('/proc')
HOST, PORT = '0.0.0.0', 8772
RECORD = RUNTIME / 'eval-service.json'


def _release(path):
    path = Path(path).resolve(strict=True)
    if path.parent != (BASE / 'releases').resolve() or not re.fullmatch(r'[0-9a-f]{40}', path.name):
        raise RuntimeError('Release must be an existing releases/<40-character-commit> directory')
    if not (path / 'deployment/run_eval.py').is_file():
        raise RuntimeError('Release is missing its independent server entrypoint')
    return path


def _command(release):
    return [str(PYTHON), '-u', str(release / 'deployment/run_eval.py')]


def _process(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 2:
        raise RuntimeError('Invalid process record; no process was signalled')
    p = PROC / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'cwd': str((p / 'cwd').resolve(strict=True)),
                'cmdline': [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        raise RuntimeError('Cannot verify process identity; no process was signalled') from None


def _read_record():
    if not RECORD.exists():
        return None
    if RECORD.is_symlink() or RECORD.stat().st_mode & 0o077:
        raise RuntimeError('Unsafe independent process record')
    try:
        record = json.loads(RECORD.read_text())
        if not isinstance(record, dict):
            raise ValueError()
        return record
    except (OSError, ValueError):
        raise RuntimeError('Invalid independent process record') from None


def _verify(record):
    identity = _process(record.get('pid'))
    if identity is None:
        return None
    release = _release(record.get('release', ''))
    expected = _command(release)
    if (identity['cwd'] != str(release) or identity['start_ticks'] != record.get('start_ticks')
            or identity['cmdline'] != expected or record.get('cmdline') != expected
            or record.get('host') != HOST or record.get('port') != PORT):
        raise RuntimeError('Recorded PID does not identify the owned 8772 server; refusing control')
    return identity


def _listening():
    try:
        with socket.create_connection(('127.0.0.1', PORT), timeout=0.2):
            return True
    except OSError:
        return False


def _healthy():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{PORT}/healthz', timeout=0.5) as response:
            return response.status == 200 and json.load(response).get('status') == 'ok'
    except (OSError, ValueError, AttributeError):
        return False


@contextmanager
def _lock():
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    if RUNTIME.is_symlink() or RUNTIME.stat().st_mode & 0o077:
        raise RuntimeError('Independent runtime directory must have mode 0700')
    fd = os.open(RUNTIME / '.eval-manager.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def status():
    record = _read_record()
    if record is None:
        return {'state': 'unmanaged_listener' if _listening() else 'stopped', 'port': PORT}
    identity = _verify(record)
    return {'state': 'running' if identity else 'stopped', 'port': PORT,
            'pid': record['pid'], 'release': record.get('release'),
            'listening': _listening(), 'http_alive': bool(identity) and _healthy(),
            'note': 'HTTP liveness does not establish album, WeMM or Judge readiness'}


def start():
    release = _release(RELEASE)
    with _lock():
        record = _read_record()
        if record and _verify(record):
            if record['release'] != str(release):
                raise RuntimeError('Another owned release is running; stop it explicitly before switching')
            return status()
        if _listening():
            raise RuntimeError('8772 is already occupied; no process was changed')
        if not PYTHON.is_file():
            raise RuntimeError('Independent PhotoBench Python environment is missing')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(release / 'camroll/camroll-agent'), str(release)])
        env['PYTHONUNBUFFERED'] = '1'
        command = _command(release)
        log_fd = os.open(RUNTIME / 'eval-service.log', os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.fchmod(log_fd, 0o600)
        try:
            process = subprocess.Popen(command, cwd=release, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log_fd, stderr=log_fd, start_new_session=True)
        finally:
            os.close(log_fd)
        identity = None
        for _ in range(40):
            identity = _process(process.pid)
            if identity and identity['cmdline'] == command and identity['cwd'] == str(release):
                break
            time.sleep(0.025)
        if not identity or identity['cmdline'] != command or identity['cwd'] != str(release):
            raise RuntimeError('Server exited before identity was recorded; inspect its restricted log')
        identity.update(release=str(release), host=HOST, port=PORT, created_at=time.time())
        temporary = RECORD.with_suffix('.tmp')
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(identity, stream, indent=2)
        os.replace(temporary, RECORD)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Server failed to start; inspect its restricted log')
            if _healthy():
                return status()
            time.sleep(0.1)
        raise RuntimeError('Server identity recorded but /healthz not ready; inspect its restricted log')


def stop():
    with _lock():
        record = _read_record()
        if not record:
            return {'state': 'stopped', 'note': 'No owned record; no process was signalled', 'port': PORT}
        identity = _verify(record)
        if identity is None:
            RECORD.unlink(missing_ok=True)
            return {'state': 'stopped', 'note': 'Removed stale owned record only', 'port': PORT}
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Linux pidfd support is required; no process was signalled')
        try:
            fd = os.pidfd_open(identity['pid'])
        except ProcessLookupError:
            RECORD.unlink(missing_ok=True)
            return {'state': 'stopped', 'port': PORT}
        try:
            if _verify(record) is None:
                RECORD.unlink(missing_ok=True)
                return {'state': 'stopped', 'port': PORT}
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        finally:
            os.close(fd)
        for _ in range(100):
            if _verify(record) is None:
                RECORD.unlink(missing_ok=True)
                return {'state': 'stopped', 'pid': identity['pid'], 'port': PORT}
            time.sleep(0.1)
        raise RuntimeError('Owned process did not exit within 10 seconds; no forced kill was sent')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop', 'rollback'])
    args = parser.parse_args(argv)
    if not PROC.is_dir():
        print(json.dumps({'error': 'Linux /proc is required; no service was changed'}))
        return 2
    try:
        result = {'start': start, 'status': status, 'stop': stop, 'rollback': stop}[args.command]()
        if args.command == 'rollback':
            result['rollback'] = 'Stopped independent 8772 only; original 8765/8770 and budget remain unchanged'
    except (RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
