"""Manage only this checkout's CamRoll HTTP process on Linux, never WeMM/GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROC = Path('/proc')
HOST, PORT = '127.0.0.1', 8770


def _paths(root):
    runtime = root / 'runtime'
    return runtime, runtime / 'camroll-service.json', runtime / 'camroll-service.log'


def _process(pid, proc_root=PROC):
    """Return Linux process identity including creation tick, or None if gone."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 2:
        raise RuntimeError('Invalid recorded process ID; no process was signalled')
    directory = proc_root / str(pid)
    try:
        stat = (directory / 'stat').read_text()
        fields = stat.rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'cwd': str((directory / 'cwd').resolve(strict=True)),
                'cmdline': [p.decode() for p in (directory / 'cmdline').read_bytes().split(b'\0') if p]}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        raise RuntimeError('Cannot verify recorded process identity; no process was signalled') from None


def _read_record(path):
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError()
        return record
    except (ValueError, OSError):
        raise RuntimeError('Invalid CamRoll process record; inspect it before continuing') from None


def _verify(record, root=ROOT):
    identity = _process(record.get('pid'))
    if identity is None:
        return None
    script = str(root / 'deployment/run_service.py')
    expected = record.get('cmdline')
    if (identity['cwd'] != str(root) or identity['start_ticks'] != record.get('start_ticks')
            or identity['cmdline'] != expected or not isinstance(expected, list)
            or expected != [str(root / '.venv/bin/python'), '-u', script]
            or record.get('host') != HOST or record.get('port') != PORT):
        raise RuntimeError('Recorded PID no longer identifies this CamRoll service; refusing process control')
    return identity


def _port_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=0.2):
            return True
    except OSError:
        return False


def status(root=ROOT):
    _, record_path, _ = _paths(root)
    record = _read_record(record_path)
    if record is None:
        return {'state': 'unmanaged_listener' if _port_open() else 'stopped', 'host': HOST, 'port': PORT}
    identity = _verify(record, root)
    return {'state': 'running' if identity else 'stopped', 'pid': record.get('pid'),
            'host': HOST, 'port': PORT, 'listening': _port_open(),
            'note': 'Process/listener status only; this does not verify a published album or WeMM readiness'}


@contextmanager
def _lock(root):
    runtime, _, _ = _paths(root)
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(runtime / '.camroll-manager.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def start(root=ROOT):
    runtime, record_path, log_path = _paths(root)
    with _lock(root):
        record = _read_record(record_path)
        if record is not None and _verify(record, root) is not None:
            return status(root)
        if _port_open():
            raise RuntimeError('127.0.0.1:8770 already has an unmanaged listener; no process was changed')
        if not (runtime / 'secrets.json').is_file():
            raise RuntimeError('Protected runtime/secrets.json is missing')
        if (runtime / 'secrets.json').stat().st_mode & 0o077:
            raise RuntimeError('runtime/secrets.json permissions must exclude group/other access (0600)')
        command = [str(root / '.venv/bin/python'), '-u', str(root / 'deployment/run_service.py')]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'camroll-agent')
        env['PYTHONUNBUFFERED'] = '1'
        log_fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.fchmod(log_fd, 0o600)
        try:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log_fd, stderr=log_fd, start_new_session=True)
        finally:
            os.close(log_fd)
        # Store identity before waiting for readiness, so a failed readiness
        # check leaves a controllable, precisely identified process.
        identity = None
        for _ in range(20):
            identity = _process(process.pid)
            if identity is not None and identity['cmdline'] == command and identity['cwd'] == str(root):
                break
            time.sleep(0.025)
        if identity is None or identity['cmdline'] != command or identity['cwd'] != str(root):
            raise RuntimeError('CamRoll exited before its process identity could be recorded; inspect its runtime log')
        identity.update(host=HOST, port=PORT, created_at=time.time())
        temporary = record_path.with_suffix('.tmp')
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(identity, stream, indent=2)
        os.replace(temporary, record_path)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('CamRoll failed to start; inspect runtime/camroll-service.log')
            if _port_open():
                return status(root)
            time.sleep(0.1)
        return {**status(root), 'note': 'Process started; listener not observed within 10 seconds. Inspect the runtime log.'}


def stop(root=ROOT):
    _, record_path, _ = _paths(root)
    with _lock(root):
        record = _read_record(record_path)
        if record is None:
            return {'state': 'stopped', 'note': 'No owned process record; no process was signalled'}
        identity = _verify(record, root)
        if identity is None:
            record_path.unlink(missing_ok=True)
            return {'state': 'stopped', 'note': 'Removed stale CamRoll process record only'}
        # Linux pidfds bind the signal to a process instance, preventing PID
        # reuse between the identity check and the signal syscall.
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Linux pidfd support is required for safe stop; no process was signalled')
        try:
            process_fd = os.pidfd_open(identity['pid'])
        except ProcessLookupError:
            record_path.unlink(missing_ok=True)
            return {'state': 'stopped'}
        try:
            if _verify(record, root) is None:
                record_path.unlink(missing_ok=True)
                return {'state': 'stopped'}
            try:
                signal.pidfd_send_signal(process_fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        finally:
            os.close(process_fd)
        for _ in range(100):
            if _verify(record, root) is None:
                record_path.unlink(missing_ok=True)
                return {'state': 'stopped', 'pid': identity['pid']}
            time.sleep(0.1)
        raise RuntimeError('Owned CamRoll process did not stop within 10 seconds; no forced kill was sent')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop'])
    args = parser.parse_args(argv)
    if not PROC.is_dir():
        print(json.dumps({'error': 'This manager requires Linux /proc and must run on .100'}))
        return 2
    try:
        result = {'start': start, 'status': status, 'stop': stop}[args.command]()
    except (RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
