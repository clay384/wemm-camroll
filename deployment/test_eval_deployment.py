"""Safety boundaries for independent process control and configuration."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_config
import manage_eval


def _release(tmp_path, monkeypatch):
    base = tmp_path / 'camroll-wemm'
    release = base / 'releases' / ('a' * 40)
    (release / 'deployment').mkdir(parents=True)
    (release / 'deployment/run_eval.py').write_text('')
    monkeypatch.setattr(manage_eval, 'BASE', base)
    monkeypatch.setattr(manage_eval, 'PYTHON', base / '.venv-photobench/bin/python')
    return base, release


def test_refuses_source_checkout_as_release(tmp_path, monkeypatch):
    base, release = _release(tmp_path, monkeypatch)
    assert manage_eval._release(release) == release
    with pytest.raises(RuntimeError):
        manage_eval._release(base)


@pytest.mark.parametrize('field,value', [
    ('cwd', '/home/realmagic/wemm-embedding-lab'),
    ('start_ticks', '9999'),
    ('cmdline', ['python', 'other_service.py']),
])
def test_pid_identity_mismatch_never_signals(tmp_path, monkeypatch, field, value):
    base, release = _release(tmp_path, monkeypatch)
    record = {'pid': 123, 'start_ticks': '1234', 'cwd': str(release),
              'cmdline': manage_eval._command(release), 'release': str(release),
              'host': '0.0.0.0', 'port': 8772}
    identity = {k: record[k] for k in ['pid', 'start_ticks', 'cwd', 'cmdline']}
    identity[field] = value
    monkeypatch.setattr(manage_eval, '_process', lambda pid: identity)
    with pytest.raises(RuntimeError, match='refusing control'):
        manage_eval._verify(record)


def test_old_service_port_is_rejected_even_with_matching_pid(tmp_path, monkeypatch):
    _, release = _release(tmp_path, monkeypatch)
    record = {'pid': 123, 'start_ticks': '1234', 'cwd': str(release),
              'cmdline': manage_eval._command(release), 'release': str(release),
              'host': '0.0.0.0', 'port': 8770}
    monkeypatch.setattr(manage_eval, '_process', lambda pid: record)
    with pytest.raises(RuntimeError):
        manage_eval._verify(record)


def test_stop_waits_on_owned_pidfd_without_exit_time_proc_race(tmp_path, monkeypatch):
    record = {'pid': 123}
    path = tmp_path / 'service.json'
    path.write_text('{}')
    verifies, signals = [], []
    monkeypatch.setattr(manage_eval, 'RECORD', path)
    monkeypatch.setattr(manage_eval, '_lock', nullcontext)
    monkeypatch.setattr(manage_eval, '_read_record', lambda: record)
    def verify(r):
        verifies.append(r)
        if len(verifies) > 2:
            raise RuntimeError('process is exiting and argv disappeared')
        return record
    monkeypatch.setattr(manage_eval, '_verify', verify)
    monkeypatch.setattr(manage_eval.os, 'pidfd_open', lambda pid: 999, raising=False)
    monkeypatch.setattr(manage_eval.signal, 'pidfd_send_signal', lambda fd, sig: signals.append((fd, sig)), raising=False)
    monkeypatch.setattr(manage_eval.os, 'close', lambda fd: None)
    class Poll:
        def register(self, fd, event):
            assert fd == 999
        def poll(self, timeout):
            return [(999, manage_eval.select.POLLIN)]
    monkeypatch.setattr(manage_eval.select, 'poll', Poll)
    assert manage_eval.stop()['state'] == 'stopped'
    assert len(signals) == 1 and not path.exists()


def test_budget_cannot_be_redirected_and_albums_are_isolated(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    runtime = base / 'runtime/photobench-v2'
    runtime.mkdir(parents=True, mode=0o700)
    old = base / 'runtime/secrets.json'
    old.write_text(json.dumps({'CAMROLL_BUDGET_DB': '/wrong/old.sqlite',
                               'CAMROLL_BUDGET_CNY': '300', 'CAMROLL_ALBUM_ROOT': '/old/albums'}))
    new = runtime / 'secrets.json'
    new.write_text(json.dumps({'CAMROLL_BUDGET_DB': '/wrong/new.sqlite'}))
    for p in [old, new, runtime / 'access.key', runtime / 'judge.key']:
        if not p.exists():
            p.write_text('fixture-key')
        p.chmod(0o600)
    monkeypatch.setattr(eval_config, 'BASE', base)
    monkeypatch.setattr(eval_config, 'RUNTIME', runtime)
    values = eval_config.configure(tmp_path / 'release')
    assert values['CAMROLL_BUDGET_DB'] == str(base / 'runtime/budget.sqlite')
    assert values['CAMROLL_BUDGET_CNY'] == '300'
    assert values['CAMROLL_ALBUM_ROOT'] == str(runtime / 'albums')
    assert values['CAMROLL_EVAL_ROOT'] == str(runtime / 'evaluation')
    assert values['WEMM_BASE_URL'] == 'http://127.0.0.1:8765'


def test_rejects_world_readable_secret_without_echoing_values(tmp_path):
    path = tmp_path / 'secret.json'
    path.write_text('{"key":"do-not-echo-this-value"}')
    path.chmod(0o644)
    with pytest.raises(RuntimeError) as caught:
        eval_config._protected_json(path)
    assert 'do-not-echo' not in str(caught.value)
