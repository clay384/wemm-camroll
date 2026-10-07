"""Report the separately authorized F38-On run alongside immutable g results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from four_export import _ledger_snapshot, _own_cost
from camroll_eval.common import atomic_json, now_iso, read_json, safe_id, sha256, object_hash
from gate_config import BASE, ORDER, SNAPSHOT
from gate_continuation_export import _audit, _lineage, _regular_path, TERMINAL, PROTOCOL
from gate_export import paired_deltas
from gate_suite import validate_run

RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007h'
PARENT_RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007g'
PARENT_SUITE = 'gate-continuation-20261007-102950-966c02'
SCOPE_SCHEMA = 'camroll-flash-gate-remaining-comparison-v1'


def _frozen_parent(runtime, suite):
    freeze = read_json(_regular_path(runtime / 'freeze.json', runtime))
    if freeze.get('schema_version') != 'camroll-flash-gate-remaining-freeze-v1':
        raise ValueError('Expected frozen remaining-group provenance')
    for key, filename in (('parent_suite', 'formal-suite.json'), ('parent_freeze', 'freeze.json')):
        expected = Path(PARENT_RUNTIME).absolute() / filename
        record = freeze[key]
        if Path(record['path']).absolute() != expected or sha256(_regular_path(expected, PARENT_RUNTIME)) != record['sha256']:
            raise ValueError('Frozen g provenance changed')
    parent = read_json(Path(PARENT_RUNTIME) / 'formal-suite.json')
    if (parent.get('suite_id') != PARENT_SUITE or parent.get('status') != 'blocked'
            or tuple(row['name'] for row in parent['groups']) != ORDER
            or parent['groups'][-1].get('status') != 'not_started'
            or suite.get('acknowledgement') != freeze.get('acknowledgement')
            or suite.get('acknowledgement') != parent.get('acknowledgement')):
        raise ValueError('Original blocked suite or acknowledgement scope changed')
    return freeze, parent


def _read_group(runtime, planned, position, freeze, parent):
    identifier = safe_id(planned['run_id'])
    root = Path(PARENT_RUNTIME) if position < 3 else runtime
    path = root / 'evaluation/runs' / identifier / 'run.json'
    if position < 3:
        original = parent['groups'][position]
        frozen = freeze['parent_runs'][planned['name']]
        if (identifier != original['run_id'] or planned.get('reused') is not True
                or Path(frozen['path']).absolute() != path.absolute()
                or Path(planned.get('parent_run_path', '')).absolute() != path.absolute()
                or planned.get('parent_run_sha256') != frozen['sha256']):
            raise ValueError('Read-only group reference identity changed')
        if sha256(_regular_path(path, root)) != frozen['sha256']:
            raise ValueError('Original g run changed')
    elif not path.exists():
        return path, None
    else:
        _regular_path(path, root)
    run = read_json(path)
    if (run.get('run_id') != identifier or run.get('inference_config', {}).get('group') != planned['name']
            or run.get('mode') != 'gate_ab' or run.get('formal_protocol') is not True
            or run.get('protocol_label') != PROTOCOL or run.get('gate_summary_version') != PROTOCOL
            or run.get('index_version') != SNAPSHOT or run.get('qa_count') != 100
            or run.get('planned_turn_count') != 111 or run.get('planned_eligible_qa_count') != 81):
        raise ValueError('Remaining report requires the unchanged formal protocol')
    if position < 3 and run.get('suite_id') != PARENT_SUITE:
        raise ValueError('Original group suite identity changed')
    return path, run


def _same_inputs(runs):
    first = runs[0]
    if (not first.get('index_binding') or not first.get('input_integrity')
            or any(not first['input_integrity'].get(key) for key in ('qa_sha256', 'dataset_sha256'))):
        raise ValueError('Exact input and index evidence is required')
    if any(any(run.get(key) != first.get(key) for key in ('index_binding', 'input_integrity', 'qa_plan')) for run in runs[1:]):
        raise ValueError('Cross-source report inputs or frozen index differ')


def _empty_summary():
    return {'quality_distribution_planned': {'correct': 0, 'partial': 0, 'incorrect': 0, 'unscored': 100},
        'quality_correct_planned_rate': 0., 'quality_mean_fixed_planned': 0., 'quality_mean_scored': None,
        'retrieval_precision_macro': 0., 'retrieval_recall_macro': 0., 'retrieval_f1_macro': 0.,
        'retrieval_metric_count': 81, 'unexecuted_turn_count': 111, 'judge_valid_count': 0}


def export(runtime=RUNTIME, *, ledger_path=None):
    runtime = Path(runtime).absolute()
    parent_root = Path(PARENT_RUNTIME).absolute()
    if runtime == parent_root or runtime in parent_root.parents or parent_root in runtime.parents:
        raise ValueError('Remaining export requires an independent runtime')
    # Check caller-provided roots before normalization can hide an alias.
    suite = read_json(_regular_path(runtime / 'formal-suite.json', runtime))
    if (suite.get('schema_version') != 'camroll-flash-gate-remaining-suite-v1'
            or suite.get('execution_order') != ['F38-On']
            or tuple(row['name'] for row in suite.get('groups', [])) != ORDER):
        raise ValueError('Only the separately authorized F38-On scope is supported')
    freeze, parent = _frozen_parent(runtime, suite)
    ledger_path = Path(ledger_path or BASE / 'runtime/budget.sqlite')
    groups, audits, records, scope_rows = [], {}, [], []
    for position, planned in enumerate(suite['groups']):
        path, run = _read_group(runtime, planned, position, freeze, parent)
        if run is None:
            groups.append({**planned, 'summary': _empty_summary(), 'execution_complete': False,
                'work_complete': False, 'acceptance_complete': False, 'financial_settlement_complete': False})
            continue
        if position == 3 and (run.get('suite_id') != suite['suite_id'] or run.get('reference') or run.get('continuation')):
            raise ValueError('F38-On must be a new independent run with its own identity')
        records.append(run)
        _same_inputs(records)
        sources, lineage = _lineage(run, path.parent)
        audit, usage = _audit(run, sources, ledger_path, suite['acknowledgement'])
        execution = planned.get('returncode') == 0 and validate_run(run)
        accepted = bool(execution and audit['complete'] and not audit['unacknowledged_outstanding_call_ids'])
        audit.update(work_complete=accepted, financial_settlement_complete=audit['settlement_complete'])
        audits[run['run_id']] = audit
        group = {**planned, 'status': run['status'], 'suite_id': run['suite_id'],
            'summary': run['summary'], 'execution_complete': execution, 'work_complete': accepted,
            'acceptance_complete': accepted, 'financial_settlement_complete': audit['settlement_complete'],
            'inference_config': run['inference_config'], 'index_binding': run['index_binding'],
            'cost_audit': audit, 'calls_and_tokens': usage, 'source_run_sha256': sha256(path),
            'source_kind': 'read_only_g_reference' if position < 3 else 'independent_h_F38_On',
            'inner_provenance': lineage, 'new_paid_execution_in_this_suite': position == 3}
        groups.append(group)
        scope_rows.append({'group': planned['name'], 'run_id': run['run_id'], 'suite_id': run['suite_id'],
            'run_sha256': group['source_run_sha256'], 'source': 'history' if position < 3 else 'current'})
    identifiers = sorted({identifier for audit in audits.values() for identifier in audit['request_ids']})
    issues = [{'kind': 'group_cost_attribution_incomplete', 'group': group['name']}
        for group in groups if group.get('cost_audit', {}).get('artifact_issues')
        or group.get('cost_audit', {}).get('missing_call_ids')]
    all_terminal = len(records) == 4 and all(run['status'] in TERMINAL for run in records)
    total = _own_cost(identifiers, 0, _ledger_snapshot(ledger_path, identifiers),
                      traces_complete=all_terminal, artifact_issues=issues)
    total['scope'] = 'Unique union of the three frozen g sources (including f ancestry) and new h F38-On requests. Prior attempts are retained, not resubmitted; call_id is counted once.'
    total['acknowledged_outstanding'] = list({row['call_id']: row for audit in audits.values()
        for row in audit['acknowledged_outstanding']}.values())
    total['source_run_count'] = len(audits)
    total['summed_group_request_count_before_deduplication'] = sum(audit['request_count'] for audit in audits.values())
    remaining_complete = bool(groups[-1]['work_complete'] and suite.get('status') == 'passed'
        and suite.get('restoration', {}).get('state') == 'restored'
        and all(suite.get(key) is True for key in ('model_files_unchanged', 'wemm_weights_unchanged', 'frozen_inputs_unchanged')))
    scope = None
    if all_terminal and suite.get('status') in {'passed', 'blocked', 'interrupted', 'failed', 'incomplete'}:
        scope = {'schema_version': SCOPE_SCHEMA, 'suite_id': suite['suite_id'], 'source_suite_id': PARENT_SUITE,
            'protocol': PROTOCOL, 'comparison_complete': False, 'runs': scope_rows,
            'index_binding': records[0]['index_binding'], 'input_integrity': records[0]['input_integrity'],
            'qa_plan_sha256': object_hash(records[0]['qa_plan']),
            'note': 'Only F38-On is newly executed in h. Original g suite identities and F38-Off blocked outcome remain unchanged.'}
        atomic_json(runtime / 'evaluation/remaining-comparison.json', scope)
    report = {'schema_version': 'camroll-flash-gate-remaining-report-v1', 'observed_at': now_iso(),
        'suite_id': suite['suite_id'], 'parent_suite_id': PARENT_SUITE, 'status': 'incomplete',
        'work_complete': False, 'acceptance_complete': False, 'remaining_group_complete': remaining_complete,
        'financial_settlement_complete': bool(len(audits) == 4 and all(audit['settlement_complete'] for audit in audits.values())),
        'groups': groups, 'paired_comparisons': paired_deltas(groups), 'comparison_cost_audit': total,
        'acknowledged_outstanding': total['acknowledged_outstanding'], 'comparison_scope_ready': bool(scope),
        'restoration': suite.get('restoration'), 'budget_before': suite.get('budget_before'), 'budget_after': suite.get('budget_after'),
        'actual_subscription_bill_cny': None, 'actual_bill_status': 'unavailable', 'private_artifact': True,
        'interpretation': 'Only the previously unexecuted F38-On group is newly run. F38-Off stays blocked with its missing turns. Fixed denominators: 100 questions, 111 turns, 81 delivery questions. Seen benchmark, not independent generalization evidence.',
        'gate_judge_correlation': 'Doubao Seed 2.0 Lite performs both independent final judging and evidence Gate.'}
    atomic_json(runtime / 'reports/flash-gate.json', report)
    atomic_json(runtime / 'evaluation/cost-audit.json', {'observed_at': now_iso(), 'runs': audits})
    atomic_json(runtime / 'evaluation/gate-comparisons.json', report['paired_comparisons'])
    lines = ['# Qwen Flash Gate：独立完成剩余F38-On', '',
        f"观测：{report['observed_at']}；四组比较仍不完整；本次F38-On执行及评分完整：{remaining_complete}。", '',
        'F37-Off、F37-On及F38-Off只读引用g原记录。F38-Off首轮阻断及其未执行轮次保持原样；本次仅新增F38-On，不重发旧未知请求、不重新执行其他组。', '',
        '| 组别 | 来源 | 状态 | 正确/部分/错误/未评分 | 有效评分数 | 有效 / 固定100题质量均分 |',
        '|---|---|---|---|---|---|']
    for group in groups:
        s = group['summary']
        distribution = '/'.join(str(s.get('quality_distribution_planned', {}).get(key, '—')) for key in ('correct', 'partial', 'incorrect', 'unscored'))
        lines.append(f"| {group['name']} | {group.get('source_kind', '待执行')} | {group['status']} | {distribution} | {s.get('judge_valid_count')} | {s.get('quality_mean_scored')} / {s.get('quality_mean_fixed_planned')} |")
    lines += ['', '费用按全部来源请求ID去重；原预留仍单列，套餐实际账单未接入。交付宏平均P/R/F1固定81题。Doubao事实Gate与最终Judge存在同模型关联；album3是已见基准。', '',
              '[只读报告](http://192.168.0.100:8774/)；原g/f/d及9月30日记录均独立保留。', '']
    path = runtime / 'reports/flash-gate.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    path.chmod(0o600)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    result = export(args.runtime, ledger_path=args.ledger)
    print(json.dumps({key: result[key] for key in ('status', 'work_complete', 'remaining_group_complete', 'financial_settlement_complete')}))
