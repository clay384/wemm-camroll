from pathlib import Path
import os
import subprocess
import sys

import pytest

import gate_config
import minicpm_report as report
from test_gate_remaining_export import remaining, module as remaining_export
from camroll_eval.repository import Repository


def test_report_preserves_all_roots():
    assert [path.parent.name for path in report.history_roots()] == [
        'photobench-flash-gate-20261007h', 'photobench-flash-gate-20261007g',
        'photobench-flash-gate-20261006f', 'photobench-flash-gate-20261002d',
        gate_config.HISTORY_RUNTIME.name]


def test_runtime_required_and_script_bootstraps_without_pythonpath(tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, str(Path(report.__file__).resolve()), '--help'],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and '--runtime' in result.stdout
    with pytest.raises(SystemExit):
        report.main([])


def test_bind_conflict_does_not_construct_app_or_take_over_owner(monkeypatch):
    class Socket:
        def bind(self, address):
            assert address == ('0.0.0.0', 8774)
            raise OSError('address already in use')
        def listen(self, value):
            raise AssertionError('must not serve')
    monkeypatch.setattr(report.socket, 'socket', lambda *args: Socket())
    monkeypatch.setattr(report, 'create_report', lambda *args: pytest.fail('must not create app'))
    with pytest.raises(OSError, match='already in use'):
        report.main(['--runtime', str(gate_config.BASE / 'runtime/minicpm47-thinking-m4-off-test')])


def test_foreign_runtime_is_rejected_before_binding(monkeypatch):
    monkeypatch.setattr(report.socket, 'socket', lambda *args: pytest.fail('must not bind'))
    with pytest.raises(SystemExit):
        report.main(['--runtime', str(gate_config.BASE / 'runtime/photobench-flash-gate-20261007h')])


def test_new_report_retains_hash_bound_h_g_history_comparison(remaining, tmp_path):
    f, g, h, ledger = remaining
    result = remaining_export.export(h, ledger_path=ledger.path)
    repo = Repository(tmp_path / 'new/evaluation',
        history_roots=[h / 'evaluation', g / 'evaluation', f / 'evaluation'])
    comparison = repo.compare([group['run_id'] for group in result['groups']])
    assert comparison['remaining_scope']['source_hashes_verified'] is True
    assert comparison['comparison_scope'] == 'frozen_index_flash_gate_comparison'
