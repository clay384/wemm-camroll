"""The continuation report cannot start an Agent or replace a port owner."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_continuation_report as report
import gate_continuation_export as export
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
