"""The remaining-group report cannot start an Agent or replace a port owner."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_remaining_report as report
import gate_remaining_export as export
import gate_config


def test_report_only_binds_read_only_routes_and_explicit_history_roots(monkeypatch):
    from camroll_eval import service
    import uvicorn
    calls = {}
    class Socket:
        def bind(self, address): calls['bind'] = address
        def listen(self, value): calls['listen'] = value
    socket = Socket()
    monkeypatch.setattr(report.socket, 'socket', lambda *args: socket)
    def create_app(root, source, albums, **kwargs):
        calls['app'] = (root, source, albums, kwargs)
        return 'synthetic-readonly-app'
    monkeypatch.setattr(service, 'create_app', create_app)
    monkeypatch.setattr(uvicorn, 'Config', lambda app, **kwargs: app)
    class Server:
        def __init__(self, app): assert app == 'synthetic-readonly-app'
        def run(self, **kwargs): calls['serve'] = kwargs
    monkeypatch.setattr(uvicorn, 'Server', Server)
    report.main()
    root, source, albums, kwargs = calls['app']
    assert root == export.RUNTIME / 'evaluation'
    assert kwargs['read_only'] is True and kwargs['mount_agent'] is False
    assert kwargs['history_roots'] == [export.PARENT_RUNTIME / 'evaluation',
        gate_config.BASE / 'runtime/photobench-flash-gate-20261006f/evaluation',
        gate_config.BASE / 'runtime/photobench-flash-gate-20261002d/evaluation',
        gate_config.HISTORY_RUNTIME / 'evaluation']
    assert calls['bind'] == ('0.0.0.0', 8774) and calls['serve'] == {'sockets': [socket]}


def test_port_conflict_stops_before_creating_report_app(monkeypatch):
    from camroll_eval import service
    class Conflict:
        def bind(self, address): raise OSError('address already in use')
        def listen(self, value): raise AssertionError('must not serve')
    monkeypatch.setattr(report.socket, 'socket', lambda *args: Conflict())
    def create_app(*args, **kwargs): raise AssertionError('must not mount anything on conflict')
    monkeypatch.setattr(service, 'create_app', create_app)
    with pytest.raises(OSError, match='already in use'):
        report.main()


def test_export_cli_bootstraps_without_pythonpath(tmp_path):
    import os
    import subprocess
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, str(Path(export.__file__).resolve()), '--help'],
                            cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--runtime' in result.stdout and '--ledger' in result.stdout


# Exercise the actual producer with its fully synthetic supervisor fixture.
from test_gate_remaining import supervisor, Ack, remaining
from camroll_eval.common import read_json


def test_real_producer_profile_renders_incomplete_source_notice(supervisor):
    import json
    sys.path.insert(0, str(Path(__file__).parents[1] / 'camroll/camroll-agent/tests'))
    from test_report_ui import evaluate
    runtime, parent, ledger, events, flags = supervisor
    assert remaining.execute(runtime, parent, ledger, Ack()) == 0
    profile = read_json(runtime / 'evaluation/comparison-profile.json')
    assert profile['execution_order'] == ['F38-On']
    expression = """(()=>{const profile=PROFILE;state.profile=profile;
        state.runs=profile.groups.map(g=>({run_id:g.run_id,mode:'gate_ab',qa_count:100,status:g.status,summary:{}}));
        renderProfile(profile);return {html:$('#comparison-profile').innerHTML,
          old:gateBatchBadge(state.runs[0]),fresh:gateBatchBadge(state.runs[3])};})()""".replace('PROFILE', json.dumps(profile))
    result = evaluate(expression)
    assert '四组比较仍未完整' in result['html']
    assert '本次F38-On已执行及完成必要评分' in result['html']
    assert '只读引用此前封存的结果' in result['html']
    assert '引用封存结果' in result['old'] and '独立新增' in result['fresh']
