"""Capability smoke failures must prevent subsequent paid submissions."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_cloud_smoke as smoke
from camroll_agent.budget import BudgetLedger, RequestUncertain
import camroll_agent.qwen as qwen
import camroll_eval.judge as judge


def test_uncertain_model_result_skips_other_models_and_judge(tmp_path, monkeypatch):
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600, expected_limit_cny=600)
    monkeypatch.setenv('CAMROLL_BUDGET_DB', str(ledger.path))
    monkeypatch.setattr(smoke, 'RUNTIME', tmp_path)
    monkeypatch.setattr(smoke, 'configure', lambda release: None)
    monkeypatch.setattr(sys, 'argv', ['four_cloud_smoke.py', '--execute-paid'])
    calls = []
    class UnknownClient:
        def __init__(self, **kwargs):
            calls.append(kwargs['model'])
        def chat(self, *args, **kwargs):
            pending = ledger.reserve(1)
            ledger.finish(pending, None)
            raise RequestUncertain('Synthetic transport uncertainty')
    def forbidden_judge(*args, **kwargs):
        raise AssertionError('Judge must not run after an uncertain request')
    monkeypatch.setattr(qwen, 'QwenClient', UnknownClient)
    monkeypatch.setattr(judge, 'Judge', forbidden_judge)
    smoke.main()
    report = json.loads((tmp_path / 'cloud-smoke.json').read_text())
    assert report['status'] == 'blocked'
    assert report['judge']['error_type'] == 'RequestUncertain'
    assert calls == ['qwen3.7-flash']
    assert ledger.summary()['requests'] == 1
