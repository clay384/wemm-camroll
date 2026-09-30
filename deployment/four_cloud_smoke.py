"""Paid, journaled capability checks using synthetic inputs only."""
import argparse
import json
import os
import sys
from pathlib import Path

from four_config import RUNTIME, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raise-budget', action='store_true')
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    configure(release)
    sys.path.insert(0, str(release / 'camroll/camroll-agent'))
    from camroll_agent.budget import BudgetLedger
    from camroll_agent.qwen import QwenClient, image_messages
    from camroll_eval.common import atomic_json, now_iso
    from camroll_eval.judge import Judge
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'])
    if args.raise_budget:
        before = ledger.summary()
        if before['reserved_cny'] or before['uncertain_requests']:
            raise RuntimeError('pending_cloud_requests_require_reconciliation')
        audit = ledger.raise_limit(600, expected_previous_cny=300,
                                   reason='User approved four-group CamRoll M4 experiment on 2026-09-30')
        atomic_json(RUNTIME / 'budget-authorization.json', audit)
    if not args.execute_paid:
        print(json.dumps(ledger.summary()))
        return
    ledger.check_limit()
    from PIL import Image
    fixture = RUNTIME / 'fixtures/red.png'
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if not fixture.exists():
        Image.new('RGB', (160, 120), 'red').save(fixture)
    destination = RUNTIME / 'cloud-smoke.json'
    report = {'started_at': now_iso(), 'before': ledger.summary(), 'models': {}, 'source': 'synthetic-nonformal'}
    tools = [{'type': 'function', 'function': {'name': 'record_color', 'description': 'Record the color',
        'parameters': {'type': 'object', 'properties': {'color': {'type': 'string'}}, 'required': ['color'],
                       'additionalProperties': False}}}]
    for model in ('qwen3.7-flash', 'qwen3.8-flash', 'qwen3.8-max'):
        client = QwenClient(model=model, expected_limit_cny=600, request_namespace='cloud-smoke-20260930-v1',
                            request_journal_path=RUNTIME / 'cloud-smoke-journal.sqlite')
        result = {'checks': {}, 'status': 'running'}
        report['models'][model] = result
        atomic_json(destination, report)
        try:
            response = client.chat([{'role': 'user', 'content': 'Reply exactly OK.'}], max_tokens=64,
                                   role='probe', request_id='text', stream=True)
            result['checks']['text'] = {'model': response['_usage']['model'], 'usage': response['_usage'],
                                        'passed': bool(response.get('content'))}
            response = client.chat(image_messages('What is the dominant color? Reply with its English name.', [fixture]),
                                   role='probe', max_tokens=64, request_id='image')
            result['checks']['image'] = {'model': response['_usage']['model'], 'usage': response['_usage'],
                                         'passed': 'red' in (response.get('content') or '').lower()}
            response = client.chat([{'role': 'user', 'content': 'Call record_color with color red.'}],
                tools, tool_choice={'type': 'function', 'function': {'name': 'record_color'}},
                role='probe', max_tokens=128, request_id='tool')
            result['checks']['tool'] = {'model': response['_usage']['model'], 'usage': response['_usage'],
                                        'passed': any(row['function']['name'] == 'record_color' for row in response.get('tool_calls', []))}
            with client.request_context('json'):
                output = client.generate_json('Return the color red.', [],
                    {'type': 'object', 'properties': {'color': {'type': 'string'}}, 'required': ['color'],
                     'additionalProperties': False}, role='probe')
            result['checks']['structured'] = {'passed': isinstance(output, dict) and output.get('color') == 'red'}
            result['status'] = 'passed' if all(value['passed'] for value in result['checks'].values()) else 'failed'
        except Exception as exc:
            result.update(status='blocked', error_type=type(exc).__name__, error_code=getattr(exc, 'code', None))
        atomic_json(destination, report)
        if ledger.summary()['uncertain_requests']:
            break
    if ledger.summary()['uncertain_requests']:
        report['judge'] = {'status': 'blocked', 'error_type': 'RequestUncertain', 'reason': 'Earlier cloud request remains uncertain'}
    else:
        try:
            judge = Judge(expected_limit_cny=600, request_namespace='cloud-smoke-20260930-v1',
                          request_journal_path=RUNTIME / 'cloud-smoke-journal.sqlite')
            with judge.request_context('arithmetic'):
                result = judge._judge('2+2等于几？', '4', '4', expected_action='answer')
            report['judge'] = {'status': 'passed' if result.get('score') == 2 else 'failed', 'score': result.get('score'),
                                'protocol': judge.protocol()}
        except Exception as exc:
            report['judge'] = {'status': 'blocked', 'error_type': type(exc).__name__}
    report.update(after=ledger.summary(), finished_at=now_iso())
    report['status'] = 'passed' if len(report['models']) == 3 and all(row['status'] == 'passed' for row in report['models'].values()) and report['judge']['status'] == 'passed' else 'blocked'
    atomic_json(destination, report)
    print(json.dumps({'status': report['status'], 'models': {key: value['status'] for key, value in report['models'].items()},
                      'judge': report['judge']['status'], 'budget': report['after']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
