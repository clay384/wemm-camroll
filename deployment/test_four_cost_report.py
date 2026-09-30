from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json
from camroll_eval.service import create_app


def test_cost_audit_is_report_overlay_and_keeps_run_bytes(tmp_path):
    root = tmp_path / 'evaluation'
    state = {'run_id': 'one', 'items': [], 'qa_plan': [], 'qa_count': 100, 'formal_protocol': True,
             'budget_before': {'estimated_spent_cny': 10}, 'budget_after': {'estimated_spent_cny': 20}}
    path = root / 'runs/one/run.json'
    atomic_json(path, state)
    original = path.read_bytes()
    audit = {'own_estimated_cny': 2, 'own_reserved_cny': 0, 'request_count': 3, 'missing_call_ids': [], 'by_model': []}
    atomic_json(root / 'cost-audit.json', {'runs': {'one': audit}})
    client = TestClient(create_app(root, tmp_path / 'source', tmp_path / 'albums', read_only=True, mount_agent=False))
    assert client.get('/api/runs').json()['runs'][0]['cost_audit'] == audit
    assert client.get('/api/runs/one').json()['cost_audit'] == audit
    assert client.get('/api/compare?ids=one').json()['runs'][0]['cost_audit'] == audit
    assert path.read_bytes() == original
    assert 'cost_audit' not in client.get('/api/runs/one/export').json()
