"""Lease-bounded, identity-checked GPU1 ownership for CamRoll experiments."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request

from four_config import GPU_UUID, RUNTIME, WORKER_PYTHON, configure

OLD = Path('/home/realmagic/wemm-embedding-lab')
LEGACY_BASELINE = Path('/home/realmagic/weknora-album-lab/locks/LEGACY_BASELINE.json')
MODEL_ROOT = Path('/home/realmagic/weknora-album-lab/models')
MODEL_MANIFEST_SHA256 = 'f6ef1a9b61c70cfe836e836e510ace33f40bc9def99e5e8e4417630e53fc68ba'
PROC = Path('/proc')
LEASE_SECONDS = 600
KEEPALIVE_SECONDS = 30


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_files():
    baseline = json.loads(LEGACY_BASELINE.read_text())
    entries = list(baseline['protected_files'].items())
    entries.extend((entry['path'], entry['sha256']) for entry in baseline['index_files'])
    for relative, expected in entries:
        target = (OLD / relative).resolve(strict=True)
        if not target.is_relative_to(OLD.resolve()) or _sha(target) != expected:
            raise RuntimeError('Protected WeMM file changed: ' + relative)
    return {'protected_files': len(baseline['protected_files']),
            'index_files': len(baseline['index_files']),
            'baseline_sha256': _sha(LEGACY_BASELINE), 'verified_at': time.time()}


def verify_model_files():
    manifest_path = MODEL_ROOT / 'MODEL_WORKER_DOWNLOAD_MANIFEST.json'
    if _sha(manifest_path) != MODEL_MANIFEST_SHA256:
        raise RuntimeError('Qwen model download manifest changed')
    manifest = json.loads(manifest_path.read_text())
    verified = {}
    for model in ('Qwen3-VL-Embedding-2B', 'Qwen3-VL-Reranker-2B'):
        entry = manifest['models']['Qwen/' + model]
        if not entry.get('complete') or not entry.get('files'):
            raise RuntimeError('Qwen model manifest is incomplete')
        root = (MODEL_ROOT / model).resolve(strict=True)
        files = {}
        for item in entry['files']:
            path = (root / item['path']).resolve(strict=True)
            if (not path.is_relative_to(root) or path.stat().st_size != item['size']
                    or _sha(path) != item['sha256']):
                raise RuntimeError('Qwen model file integrity mismatch: ' + model + '/' + item['path'])
            files[item['path']] = {'size': item['size'], 'sha256': item['sha256']}
        verified[model] = files
    return {'manifest_sha256': MODEL_MANIFEST_SHA256, 'models': verified, 'verified_at': time.time()}


def _config():
    import yaml
    config = yaml.safe_load((OLD / 'config.yaml').read_text())
    if config.get('port') != 8765 or config.get('gpu_uuid') != GPU_UUID:
        raise RuntimeError('WeMM configuration GPU or port differs from the authorized service')
    return config


def _legacy_command():
    config = _config()
    return [str(OLD / '.venv/bin/python'), '-m', 'uvicorn', 'backend.app:app',
            '--host', config['host'], '--port', '8765', '--workers', '1']


def _legacy_fingerprint():
    config = _config()
    model = OLD / config['model_path']
    parts = {key: config[key] for key in ('model_id', 'revision', 'precision', 'dimension',
                                          'profiles', 'profile', 'video_segment_seconds')}
    parts['implementation_sha256'] = _sha(OLD / 'backend/encoder.py')
    parts['pipeline_sha256'] = {name: _sha(OLD / 'backend' / name) for name in ('preprocess.py', 'worker.py')}
    parts['candidate_modes'] = ['image', 'visual_pdf', 'native_video', 'text_chunk', 'image_text_joint']
    parts['files'] = {name: _sha(model / name) for name in ('config.json', 'modeling_wemm_embedding.py',
        'tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja',
        'embedding_chat_template.jinja')}
    parts.update(normalize='float32 L2', aggregation='max unit score; ties asset ID',
                 split='pack whole paragraphs to 512 tokens; oversized paragraph fixed token windows, no overlap',
                 weights='b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4')
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _check_legacy_database(fingerprint):
    connection = sqlite3.connect((OLD / 'data/lab.sqlite').as_uri() + '?mode=ro', uri=True)
    try:
        pending = connection.execute("SELECT COUNT(*) FROM assets WHERE status IN ('pending','processing','encoding','preprocessing')").fetchone()[0]
        mismatched = connection.execute('SELECT COUNT(*) FROM units WHERE fingerprint != ?', (fingerprint,)).fetchone()[0]
    finally:
        connection.close()
    if pending or mismatched:
        raise RuntimeError('WeMM has pending assets or incompatible frozen vectors; startup is blocked')


def _process(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 2:
        raise RuntimeError('Invalid process identity')
    directory = PROC / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'cwd': str((directory / 'cwd').resolve(strict=True)),
                'cmdline': [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]}
    except FileNotFoundError:
        return None
    except (OSError, IndexError, UnicodeDecodeError, ValueError):
        raise RuntimeError('Cannot verify process identity') from None


def _verify(record):
    identity = _process(record['pid'])
    if identity is not None and any(identity[key] != record[key] for key in ('pid', 'start_ticks', 'cwd', 'cmdline')):
        raise RuntimeError('Process identity changed; no signal was sent')
    return identity


def _port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False


def _owns_port(identity, port):
    listening = set()
    for filename in ('tcp', 'tcp6'):
        for line in (PROC / 'net' / filename).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(':', 1)[1], 16) == port and fields[3] == '0A':
                listening.add(fields[9])
    owned = set()
    for descriptor in (PROC / str(identity['pid']) / 'fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            owned.add(target[8:-1])
    return bool(listening) and listening <= owned


def _gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            gpu, pid = line.split(',')
            rows.append((gpu.strip(), int(pid.strip())))
    return rows


def _assert_gpu(allowed=()):
    rows = _gpu_processes()
    if any((gpu == GPU_UUID and pid not in allowed) or (pid in allowed and gpu != GPU_UUID) for gpu, pid in rows):
        raise RuntimeError('Unexpected GPU1 owner or authorized process using another GPU')
    return [{'gpu_uuid': gpu, 'pid': pid} for gpu, pid in rows if gpu == GPU_UUID]


def _request_json(url, *, headers=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers=headers or {}), timeout=10) as response:
        return json.load(response)


def _legacy_status():
    timestamp = str(int(time.time()))
    key = (OLD / 'run/access.key').read_text().strip()
    token = timestamp + '.' + hmac.new(key.encode(), timestamp.encode(), 'sha256').hexdigest()
    return _request_json('http://127.0.0.1:8765/api/status', headers={'Cookie': 'wemm_session=' + token})


def _idle_twice():
    first = _legacy_status()
    time.sleep(2)
    second = _legacy_status()
    for status in (first, second):
        if (status.get('queue') != 0 or status.get('busy_asset') is not None or not status.get('fingerprint')
                or status.get('state') not in ('ready', 'loaded')
                or any(status.get('counts', {}).get(name, 0) for name in ('pending', 'processing', 'encoding', 'preprocessing'))):
            raise RuntimeError('WeMM is not idle; no service was stopped')
    if first.get('encoding_calls') is None or first['encoding_calls'] != second.get('encoding_calls'):
        raise RuntimeError('WeMM is active; no service was stopped')
    return second


def _stop(record):
    if _verify(record) is None:
        return
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required for safe process control')
    try:
        descriptor = os.pidfd_open(record['pid'])
    except ProcessLookupError:
        return
    try:
        if _verify(record) is None:
            return
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        except ProcessLookupError:
            pass
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        if not poller.poll(60000):
            raise RuntimeError('Owned process did not exit gracefully; no forced kill was sent')
    finally:
        os.close(descriptor)


class GpuWindow:
    def __init__(self, release, window_id, *, runtime=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', window_id):
            raise ValueError('Invalid GPU window identifier')
        self.release = Path(release).resolve()
        self.window_id = window_id
        self.runtime = Path(runtime or RUNTIME)
        self.directory = self.runtime / 'windows' / window_id
        self.state_path = self.directory / 'state.json'
        self.current_path = self.runtime / 'gpu-window-current.json'
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    @contextlib.contextmanager
    def _lock(self):
        self.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.runtime / 'gpu-window.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self):
        state = json.loads(self.state_path.read_text())
        if state.get('window_id') != self.window_id or state.get('release') != str(self.release):
            raise RuntimeError('GPU window ownership mismatch')
        return state

    def _write(self, state):
        _save(self.state_path, state)

    def _spawn(self, command, cwd, name, env):
        descriptor = os.open(self.directory / (name + '.log'), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=descriptor, stderr=descriptor, start_new_session=True)
        finally:
            os.close(descriptor)
        for _ in range(80):
            identity = _process(child.pid)
            if identity and identity['cwd'] == str(cwd) and identity['cmdline'] == command:
                return identity
            if child.poll() is not None:
                break
            time.sleep(.025)
        raise RuntimeError('Started process identity could not be verified: ' + name)

    def _launch_guard(self):
        command = [sys.executable, str(self.release / 'deployment/four_gpu_window.py'), 'watch',
                   '--window-id', self.window_id, '--runtime', str(self.runtime), '--release', str(self.release)]
        return self._spawn(command, self.release, 'watchdog', os.environ.copy())

    def _start_heartbeat(self):
        def beat():
            while not self._heartbeat_stop.wait(KEEPALIVE_SECONDS):
                try:
                    self.keepalive()
                except Exception:
                    return
        self._heartbeat_thread = threading.Thread(target=beat, name='camroll-gpu-lease', daemon=True)
        self._heartbeat_thread.start()

    def enter(self):
        with self._lock():
            if self.state_path.exists():
                raise RuntimeError('GPU window IDs cannot be reused')
            if self.current_path.exists():
                current = json.loads(self.current_path.read_text())
                previous_path = self.runtime / 'windows' / current['window_id'] / 'state.json'
                if json.loads(previous_path.read_text()).get('state') != 'restored':
                    raise RuntimeError('An earlier GPU window requires recovery')
            integrity = verify_files()
            model_integrity = verify_model_files()
            fingerprint = _legacy_fingerprint()
            _check_legacy_database(fingerprint)
            if not _port_free(18778):
                raise RuntimeError('Qwen worker port 18778 is occupied')
            record_path = OLD / 'run/service.json'
            initial_record = json.loads(record_path.read_text()) if record_path.exists() else None
            original = _process(initial_record['pid']) if initial_record else None
            if original and (original['cwd'] != str(OLD) or original['cmdline'] != _legacy_command()):
                raise RuntimeError('Existing WeMM service identity is not recognized')
            if original:
                if not _owns_port(original, 8765):
                    raise RuntimeError('Recorded WeMM process does not own port 8765')
                before = _idle_twice()
                if before['fingerprint'] != fingerprint:
                    raise RuntimeError('Running WeMM fingerprint differs from protected sources')
            else:
                if not _port_free(8765):
                    raise RuntimeError('Unowned listener occupies WeMM port 8765')
                before = {'state': 'stopped', 'fingerprint': fingerprint}
            gpu_before = _assert_gpu([original['pid']] if original else [])
            self.directory.mkdir(parents=True, mode=0o700)
            state = {'window_id': self.window_id, 'release': str(self.release), 'state': 'entering',
                     'created_at': time.time(), 'lease_expires_at': time.time() + LEASE_SECONDS,
                     'original_running': bool(original), 'original_process': original,
                     'initial_service_record': initial_record, 'before': before, 'initial_integrity': integrity,
                     'gpu_before': gpu_before, 'current_kind': None, 'current_process': None, 'history': [],
                     'children': [], 'owner_process': _process(os.getpid()), 'model_integrity': model_integrity}
            self._write(state)
            _save(self.current_path, {'window_id': self.window_id})
            try:
                state['guard_process'] = self._launch_guard()
                self._write(state)
                if original:
                    _stop(original)
                _assert_gpu()
                state['state'] = 'active'
                self._write(state)
            except BaseException:
                self._restore_locked(state, 'enter_failed')
                raise
        self._start_heartbeat()
        return state

    def keepalive(self):
        with self._lock():
            state = self._read()
            if state['state'] not in ('entering', 'active', 'switching'):
                raise RuntimeError('GPU window is not active')
            state['lease_expires_at'] = time.time() + LEASE_SECONDS
            self._write(state)
            return {'window_id': self.window_id, 'lease_expires_at': state['lease_expires_at']}

    def register_child(self, pid, expected_command=None):
        with self._lock():
            state = self._read()
            if state['state'] != 'active':
                raise RuntimeError('GPU window is not active')
            process = None
            for _ in range(40):
                process = _process(pid)
                if (process is not None and process['cwd'] == str(self.release)
                        and (expected_command is None or process['cmdline'] == list(expected_command))):
                    break
                time.sleep(.025)
            if process is None or process['cwd'] != str(self.release):
                raise RuntimeError('Evaluator process is not owned by this release')
            if expected_command is not None and process['cmdline'] != list(expected_command):
                raise RuntimeError('Evaluator command differs from the registered command')
            if any(item['process']['pid'] == pid and item['state'] == 'running' for item in state['children']):
                raise RuntimeError('Evaluator process is already registered')
            state['children'].append({'process': process, 'state': 'running', 'registered_at': time.time()})
            self._write(state)
            return process

    def unregister_child(self, pid):
        with self._lock():
            state = self._read()
            for child in state['children']:
                if child['process']['pid'] == pid and child['state'] == 'running':
                    if _verify(child['process']) is not None:
                        raise RuntimeError('A running evaluator cannot be unregistered')
                    child.update(state='finished', finished_at=time.time())
                    self._write(state)
                    return
            raise RuntimeError('Evaluator is not registered')

    def _launch_model(self, kind, state):
        _assert_gpu()
        port = 18778 if kind == 'qwen' else 8765
        if not _port_free(port):
            raise RuntimeError('Required model port is occupied: ' + str(port))
        if kind == 'qwen':
            configure(self.release)
            env = os.environ.copy()
            command = [str(WORKER_PYTHON), '-u', str(self.release / 'deployment/qwen_worker.py')]
            cwd = self.release
        else:
            config = _config()
            _check_legacy_database(state['before']['fingerprint'])
            env = os.environ.copy()
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                env.pop(key, None)
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(OLD / '.hf'),
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(config['cpu_threads']), OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES=GPU_UUID)
            command, cwd = _legacy_command(), OLD
        identity = self._spawn(command, cwd, kind, env)
        state['current_kind'], state['current_process'] = kind, identity
        state['history'].append({'kind': kind, 'process': identity, 'started_at': time.time()})
        self._write(state)
        if kind == 'wemm':
            _save(OLD / 'run/service.json', {**identity, 'created': time.time(), 'camroll_window_id': self.window_id})
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if _verify(identity) is None:
                raise RuntimeError('Model service exited before becoming ready')
            try:
                health = (_request_json('http://127.0.0.1:18778/health') if kind == 'qwen' else _legacy_status())
                if kind == 'qwen':
                    ready = (health.get('status') == 'ok' and health.get('pid') == identity['pid']
                             and health.get('gpu_uuid') == GPU_UUID and health.get('gpu_loading_enabled') is True)
                else:
                    ready = health.get('state') in ('ready', 'loaded') and health.get('fingerprint') == state['before']['fingerprint']
                if ready and _owns_port(identity, port):
                    _assert_gpu([identity['pid']])
                    return identity
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError('Model service failed readiness or fingerprint validation')

    def switch(self, kind):
        if kind not in ('qwen', 'wemm'):
            raise ValueError('Unknown GPU model family')
        with self._lock():
            state = self._read()
            if state['state'] != 'active':
                raise RuntimeError('GPU window is not active')
            if state['current_kind'] == kind and _verify(state['current_process']):
                _assert_gpu([state['current_process']['pid']])
                return state
            state['state'] = 'switching'
            state['lease_expires_at'] = time.time() + LEASE_SECONDS
            self._write(state)
            try:
                if state['current_process']:
                    if state['current_kind'] == 'wemm' and _verify(state['current_process']):
                        _idle_twice()
                    _stop(state['current_process'])
                    state['current_process'], state['current_kind'] = None, None
                    self._write(state)
                verify_files()
                self._launch_model(kind, state)
                state['state'] = 'active'
                self._write(state)
                return state
            except BaseException:
                self._restore_locked(state, 'switch_failed')
                raise

    def _restore_locked(self, state, reason):
        if state['state'] == 'restored':
            return state
        state.update(state='restoring', restore_reason=reason)
        self._write(state)
        try:
            for child in state.get('children', []):
                if child['state'] == 'running':
                    _stop(child['process'])
                    child.update(state='stopped', stopped_at=time.time())
                    self._write(state)
            if state.get('current_process'):
                _stop(state['current_process'])
                state['current_process'], state['current_kind'] = None, None
                self._write(state)
            original = state.get('original_process')
            original_alive = _verify(original) if original else None
            _assert_gpu([original_alive['pid']] if original_alive else [])
            integrity = verify_files()
            if integrity['baseline_sha256'] != state['initial_integrity']['baseline_sha256']:
                raise RuntimeError('Legacy integrity manifest changed during GPU window')
            if state['original_running']:
                if original_alive:
                    restored_process = original_alive
                else:
                    restored_process = self._launch_model('wemm', state)
                after = _legacy_status()
                if after.get('fingerprint') != state['before']['fingerprint'] or after.get('state') not in ('ready', 'loaded'):
                    raise RuntimeError('Restored WeMM fingerprint or readiness differs from original')
                state['restored_process'] = restored_process
                state['current_process'], state['current_kind'] = None, None
                final_gpu = _assert_gpu([restored_process['pid']])
            else:
                if not _port_free(8765):
                    raise RuntimeError('WeMM was originally stopped but port 8765 is now occupied')
                service_path = OLD / 'run/service.json'
                if service_path.exists():
                    current_record = json.loads(service_path.read_text())
                    if (current_record != state.get('initial_service_record')
                            and current_record.get('camroll_window_id') != self.window_id):
                        raise RuntimeError('WeMM service record changed outside this window')
                if state.get('initial_service_record') is not None:
                    _save(service_path, state['initial_service_record'])
                elif service_path.exists():
                    record = json.loads(service_path.read_text())
                    if record.get('camroll_window_id') != self.window_id:
                        raise RuntimeError('WeMM service record changed outside this window')
                    service_path.unlink()
                after, final_gpu = {'state': 'stopped'}, _assert_gpu()
            if not _port_free(18778):
                raise RuntimeError('Qwen worker port remains occupied after restore')
            for item in state['history']:
                if item['process'] != state.get('restored_process') and _verify(item['process']) is not None:
                    raise RuntimeError('An experiment model process remains after restore')
            state.update(state='restored', restored_at=time.time(), after=after, final_gpu=final_gpu,
                         final_integrity=verify_files(), no_experiment_gpu_residual=True)
            self._write(state)
            _save(self.directory / 'receipt.json', state)
            return state
        except BaseException as exc:
            state.update(state='restore_failed', error=type(exc).__name__ + ': ' + str(exc))
            self._write(state)
            raise

    def restore(self, reason='explicit_exit'):
        self._heartbeat_stop.set()
        with self._lock():
            return self._restore_locked(self._read(), reason)

    def watch_once(self, now=None):
        with self._lock():
            state = self._read()
            if state['state'] in ('restored', 'restore_failed'):
                return False
            owner = state.get('owner_process')
            if owner:
                current_owner = _process(owner['pid'])
                if current_owner is None or any(current_owner[key] != owner[key] for key in ('start_ticks', 'cwd', 'cmdline')):
                    self._restore_locked(state, 'owner_exited')
                    return False
            if (time.time() if now is None else now) >= state['lease_expires_at']:
                self._restore_locked(state, 'lease_expired')
                return False
            return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('watch', 'restore', 'status'))
    parser.add_argument('--window-id', required=True)
    parser.add_argument('--runtime', default=str(RUNTIME))
    parser.add_argument('--release', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    window = GpuWindow(args.release, args.window_id, runtime=args.runtime)
    if args.action == 'watch':
        while window.watch_once():
            time.sleep(10)
    elif args.action == 'restore':
        print(json.dumps(window.restore('operator_restore'), ensure_ascii=False))
    else:
        print(json.dumps(window._read(), ensure_ascii=False))


if __name__ == '__main__':
    main()
