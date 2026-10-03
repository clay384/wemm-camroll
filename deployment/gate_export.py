"""Private Flash Gate report and cost attribution; no cloud or GPU operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from four_export import _jsonl, _ledger_snapshot, _own_cost, _unique_calls, _usage
from gate_config import BASE, ORDER, RUNTIME
from gate_suite import validate_run
from camroll_eval.common import atomic_json, now_iso, read_json


def journal_requests(path):
    if not path.is_file():
        return [], [{'kind': 'request_journal_missing'}]
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute('SELECT request_id,call_id,status,payload_hash,error_code FROM cloud_requests').fetchall()
        return [dict(zip(('request_id', 'call_id', 'status', 'payload_hash', 'error_code'), row)) for row in rows], []
    except (sqlite3.Error, OSError):
        return [], [{'kind': 'request_journal_unreadable'}]
    finally:
        if connection is not None:
            connection.close()


def paired_deltas(groups):
    by_name = {row['name']: row for row in groups}
    fields = ('quality_correct_planned_rate', 'quality_mean_fixed_planned', 'retrieval_precision_macro',
              'retrieval_recall_macro', 'retrieval_f1_macro', 'task_decision_accuracy', 'agent_task_latency_mean_ms')
    pairs = [('F37-Off', 'F37-On'), ('F38-Off', 'F38-On'), ('F37-Off', 'F38-Off'), ('F37-On', 'F38-On')]
    result = []
    for left, right in pairs:
        a, b = by_name.get(left, {}), by_name.get(right, {})
        delta = {}
        for key in fields:
            x, y = a.get('summary', {}).get(key), b.get('summary', {}).get(key)
            delta[key] = y - x if isinstance(x, (float, int)) and isinstance(y, (float, int)) else None
        result.append({'from': left, 'to': right, 'complete_pair': bool(a.get('acceptance_complete') and b.get('acceptance_complete')),
                       'delta_to_minus_from': delta})
    return result


def gate_overhead(run, calls, snapshot):
    events = [event for item in run.get('items', []) for event in item.get('gate_events', [])]
    extra_semantics = {event['semantic_request_id'] for event in events
                       if event.get('state') == 'sending' and event.get('attempt', 0) >= 2}
    retry_ids = {call['call_id'] for call in calls if call.get('call_id')
                 and call.get('semantic_request_id') in extra_semantics}
    retry_ids.update(event['call_id'] for event in events if event.get('call_id') and event.get('attempt', 0) >= 2)
    fact_ids = {identifier for identifier, row in snapshot['rows'].items()
                if row['metadata'].get('role') == 'evidence_gate'}
    def totals(ids):
        rows = [snapshot['rows'][identifier] for identifier in ids if identifier in snapshot['rows']]
        return {'known_estimated_cny': sum(row['charged_units'] for row in rows) / 1e6,
                'known_reserved_cny': sum(row['reserved_units'] for row in rows if row['status'] in ('pending', 'reserved', 'uncertain')) / 1e6,
                'request_count': len(ids), 'matched_count': len(rows),
                'prompt_tokens': sum(row['metadata'].get('prompt_tokens') or 0 for row in rows),
                'completion_tokens': sum(row['metadata'].get('completion_tokens') or 0 for row in rows)}
    traced = {call['call_id'] for call in calls if call.get('call_id')}
    unmapped = [identifier for identifier, row in snapshot['rows'].items()
                if row['metadata'].get('role') in ('agent', 'view', 'final') and identifier not in traced]
    return {'extra_generation': totals(retry_ids), 'fact_verifier': totals(fact_ids),
            'combined': totals(retry_ids | fact_ids), 'unmapped_generation_call_ids': unmapped,
            'mapping_complete': not unmapped and snapshot['status'] == 'available',
            'scope': 'Additional Gate semantic generations (attempts 2–4), plus all fact-verifier requests including format repairs. Final quality/action Judge is excluded. Known charges/reservations only; unmapped in-flight requests remain in full run cost.'}


def export(runtime=RUNTIME, *, ledger_path=None):
    runtime = Path(runtime)
    suite = read_json(runtime / 'formal-suite.json')
    groups, audits = [], {}
    for planned in suite['groups']:
        directory = runtime / 'evaluation/runs' / planned['run_id']
        path = directory / 'run.json'
        if not path.is_file():
            groups.append({**planned, 'execution_complete': False, 'delivery_complete': False,
                           'acceptance_complete': False, 'summary': {
                'quality_distribution_planned': {'correct': 0, 'partial': 0, 'incorrect': 0, 'unscored': 100},
                'quality_correct_planned_rate': 0., 'quality_mean_fixed_planned': 0., 'quality_mean_scored': None,
                'retrieval_precision_macro': 0., 'retrieval_recall_macro': 0., 'retrieval_f1_macro': 0.,
                'retrieval_metric_count': 81, 'unexecuted_turn_count': 111, 'judge_valid_count': 0}})
            continue
        run = read_json(path)
        if run.get('mode') != 'gate_ab':
            raise ValueError('Gate export cannot rewrite a historical experiment')
        journal, issues = journal_requests(directory / 'request-journal.sqlite')
        identifiers = sorted({row['call_id'] for row in journal})
        calls = []
        for filename in ('model_calls.jsonl', 'verifier_calls.jsonl', 'judge_calls.jsonl'):
            call_path = directory / filename
            if call_path.is_file():
                calls.extend(_jsonl(call_path, issues))
        # The dedicated journal includes requests interrupted before item/trace persistence.
        ledger = _ledger_snapshot(Path(ledger_path or BASE / 'runtime/budget.sqlite'), identifiers)
        terminal = run['status'] in {'completed', 'completed_with_errors', 'blocked', 'failed', 'interrupted', 'cancelled'}
        audit = _own_cost(identifiers, 0, ledger, traces_complete=terminal, artifact_issues=issues)
        audit['attribution_assumption'] = 'Every cloud request in the exclusive run journal, including repairs, verifier and final Judge; no historical build or smoke calls.'
        audit['journal_statuses'] = {status: sum(row['status'] == status for row in journal)
                                     for status in sorted({row['status'] for row in journal})}
        # Attribution can be complete while a known request still holds funds.
        # Delivery requires both durable terminal requests and settled costs.
        audit['settlement_complete'] = bool(audit['complete'] and journal
            and all(row['status'] in {'completed', 'failed'} for row in journal)
            and all(row['status'] in {'settled', 'released'} for row in ledger['rows'].values())
            and audit['known_reserved_cny'] == 0 and audit['uncertain_requests'] == 0)
        audit['gate_overhead'] = gate_overhead(run, calls, ledger)
        audits[run['run_id']] = audit
        execution_complete = planned.get('returncode') == 0 and validate_run(run)
        delivery_complete = execution_complete and audit['settlement_complete']
        groups.append({**planned, 'status': run['status'], 'execution_complete': execution_complete,
                       'delivery_complete': delivery_complete, 'acceptance_complete': delivery_complete,
                       'summary': run['summary'], 'inference_config': run['inference_config'],
                       'index_binding': run.get('index_binding'), 'cost_audit': audit,
                       'calls_and_tokens': _usage(_unique_calls(calls)), 'resources': run.get('resource_summary'),
                       'wall_ms': run.get('wall_ms'), 'run_json': str(path),
                       'all_attempts': str(directory / 'gate_events.jsonl'),
                       'request_journal': str(directory / 'request-journal.sqlite'),
                       'full_answers_and_traces': str(directory / 'results.jsonl')})
    execution_complete = tuple(row['name'] for row in groups) == ORDER and all(row['execution_complete'] for row in groups)
    complete = (execution_complete and all(row['delivery_complete'] for row in groups)
                and suite.get('restoration', {}).get('state') == 'restored'
                and suite.get('status') == 'passed' and suite.get('model_files_unchanged') is True
                and suite.get('frozen_inputs_unchanged') is True)
    report = {'schema_version': 'camroll-flash-gate-report-v1', 'observed_at': now_iso(),
              'status': 'complete' if complete else 'incomplete', 'acceptance_complete': complete,
              'execution_complete': execution_complete, 'delivery_complete': complete,
              'groups': groups, 'paired_comparisons': paired_deltas(groups), 'restoration': suite.get('restoration'),
              'budget_before': suite.get('budget_before'), 'budget_after': suite.get('budget_after'),
              'actual_subscription_bill_cny': None, 'actual_bill_status': 'unavailable',
              'gate_judge_correlation': 'Doubao Seed 2.0 Lite performs both independent final judging and evidence Gate.',
              'interpretation': 'Seen album3 benchmark; shared frozen WeMM index. Not independent generalization evidence.',
              'private_artifact': True}
    atomic_json(runtime / 'reports/flash-gate.json', report)
    atomic_json(runtime / 'evaluation/cost-audit.json', {'observed_at': now_iso(), 'runs': audits})
    atomic_json(runtime / 'evaluation/gate-comparisons.json', report['paired_comparisons'])
    lines = ['# Qwen Flash Gate 四组对照', '',
             f"观测：{report['observed_at']}；交付状态：{report['status']}。", '',
             '共享冻结WeMM索引，未建库。正确率与质量固定分母100题；交付宏平均固定81题。系统故障与Gate耗尽不作为正常拒答送评。', '',
             'Doubao兼任事实Gate与独立最终评分，存在同模型关联；Gate通过不保证正确。album3为已见基准。', '',
             '| 组别 | 状态 | 正确/部分/错误/未评分 | 有效质量均分 | 固定100题均分 | 归属等价费用 |',
             '|---|---|---|---|---|---|']
    for group in groups:
        s, c = group['summary'], group.get('cost_audit', {})
        d = s.get('quality_distribution_planned', {})
        dist = '/'.join(str(d.get(key, '—')) for key in ('correct', 'partial', 'incorrect', 'unscored'))
        lines.append(f"| {group['name']} | {group['status']} | {dist} | {s.get('quality_mean_scored')} | {s.get('quality_mean_fixed_planned')} | {c.get('own_estimated_cny')} |")
    lines += ['', '费用包含该组全部尝试、Gate和最终Judge，套餐实付尚未接入。独立小样费用保存在cloud-smoke.json，不并入正式组费用。', '',
              '[只读逐题报告](http://192.168.0.100:8774/)；全部候选输出、判据、Token、耗时、原始请求身份和恢复证据见同目录JSON及各run。', '',
              '## 成对差值（后者减前者）', '', '```json', json.dumps(report['paired_comparisons'], ensure_ascii=False, indent=2), '```', '']
    target = runtime / 'reports/flash-gate.md'
    target.write_text('\n'.join(lines)); target.chmod(0o600)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    result = export(args.runtime, ledger_path=args.ledger)
    print(json.dumps({'status': result['status'], 'acceptance_complete': result['acceptance_complete']}))
