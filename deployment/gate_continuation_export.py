"""Read-only ancestry and cost union for the authorized independent continuation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from four_export import _jsonl, _ledger_snapshot, _own_cost, _unique_calls, _usage
from camroll_eval.common import atomic_json, now_iso, read_json, safe_id, sha256
from gate_config import BASE, ORDER, SNAPSHOT
from gate_export import gate_overhead, journal_requests, paired_deltas
from gate_suite import validate_run

RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007g'
PARENT_RUNTIME = BASE / 'runtime/photobench-flash-gate-20261006f'
PROTOCOL = 'camroll-flash-gate-v1'
TERMINAL = {'completed', 'completed_with_errors', 'blocked', 'failed', 'interrupted', 'cancelled'}


def _regular_path(path, root):
    """Only fixed managed files, without following an ancestry symlink."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if root not in path.parents or any(part.is_symlink() for part in (path, *path.parents) if part == root or root in part.parents):
        raise ValueError('Ancestry path leaves the configured source root')
    if path.resolve() != path or not path.is_file():
        raise ValueError('Ancestry requires a regular managed file')
    return path


def _protocol(run, group):
    if (run.get('mode') != 'gate_ab' or run.get('formal_protocol') is not True
            or run.get('protocol_label') != PROTOCOL or run.get('gate_summary_version') != PROTOCOL
            or run.get('inference_config', {}).get('group') != group
            or run.get('index_version') != SNAPSHOT or run.get('qa_count') != 100
            or run.get('planned_turn_count') != 111 or run.get('planned_eligible_qa_count') != 81
            or run.get('reference') or run.get('continuation')):
        raise ValueError('Ancestry is not the frozen formal source group')


def _parent(path, digest, group, *, run_id=None):
    path = Path(path).absolute()
    identifier = safe_id(run_id or path.parent.name)
    expected = Path(PARENT_RUNTIME).absolute() / 'evaluation/runs' / identifier / 'run.json'
    if path != expected:
        raise ValueError('Parent is outside the configured source run')
    _regular_path(path, PARENT_RUNTIME)
    if sha256(path) != digest:
        raise ValueError('Parent run hash mismatch')
    run = read_json(path)
    _protocol(run, group)
    if run.get('run_id') != identifier or sha256(path) != digest:
        raise ValueError('Parent identity or hash changed')
    if group == 'F37-Off' and not validate_run(run):
        raise ValueError('Reference requires the complete original F37-Off group')
    if group == 'F37-On' and (run.get('status') != 'blocked' or run.get('stop_reason') != 'request_uncertain'):
        raise ValueError('Continuation requires the original uncertain F37-On group')
    if run.get('source_unchanged') is not True or run.get('finalization_errors'):
        raise ValueError('Original source integrity is not verified')
    return path, run


def create_reference_run(old_path, new_eval_root, new_run_id, new_suite_id, expected_sha256):
    """Create a derived result, never a new request or a mutable copy of a journal."""
    path, original = _parent(old_path, expected_sha256, 'F37-Off')
    safe_id(new_run_id)
    safe_id(new_suite_id)
    root = Path(new_eval_root).resolve()
    parent_root = Path(PARENT_RUNTIME).resolve()
    if root == parent_root or parent_root in root.parents or root in parent_root.parents:
        raise ValueError('Reference requires an independent destination')
    if new_run_id == original['run_id'] or new_suite_id == original['suite_id']:
        raise ValueError('Reference requires new run and suite identities')
    result = copy.deepcopy(original)
    result.update(run_id=new_run_id, suite_id=new_suite_id, view_kind='gate_reference')
    result['reference'] = {'schema_version': 'camroll-gate-reference-v1', 'reused': True,
        'parent_run_id': original['run_id'], 'parent_run_path': str(path),
        'parent_run_sha256': expected_sha256, 'source_suite_id': original['suite_id'],
        'inherited_turn_count': 111, 'new_turn_count': 0, 'source_records_unchanged': True,
        'new_cloud_requests': 0, 'created_at': now_iso()}
    destination = root / 'runs' / new_run_id
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.mkdir(mode=0o700)  # Exclusive: never overwrite an existing derived run.
    if sha256(path) != expected_sha256:
        raise ValueError('Parent run changed before reference creation')
    atomic_json(destination / 'run.json', result)
    return result


def _lineage(run, directory):
    reference, continuation = run.get('reference'), run.get('continuation')
    if reference and continuation:
        raise ValueError('Run cannot be both a reference and a continuation')
    metadata = reference or continuation
    if not metadata:
        return [(directory, run, 'new')], None
    group = run.get('inference_config', {}).get('group')
    expected_group = 'F37-Off' if reference else 'F37-On'
    if group != expected_group:
        raise ValueError('Unexpected inherited group')
    path, parent = _parent(metadata['parent_run_path'], metadata['parent_run_sha256'], expected_group,
                           run_id=metadata['parent_run_id'])
    if (metadata.get('source_suite_id') != parent['suite_id'] or run.get('suite_id') == parent['suite_id']
            or run.get('run_id') == parent['run_id']):
        raise ValueError('Ancestry suite identity mismatch')
    for key in ('qa_plan', 'index_binding', 'index_version', 'inference_config'):
        if run.get(key) != parent.get(key):
            raise ValueError('Derived run differs from parent ' + key)
    if reference:
        if not metadata.get('reused') or run.get('items') != parent.get('items') or run.get('summary') != parent.get('summary'):
            raise ValueError('Reference has changed original measurements')
    else:
        expected = {(item['qa_id'], turn['index']): turn for item in parent['items'] for turn in item['runtime_turns']}
        declared = [(row['qa_id'], row['turn_index']) for row in metadata.get('preserved_turns', [])]
        observed = [(item['qa_id'], turn) for item in run.get('items', []) for turn in item.get('runtime_turns', [])]
        actual = {(qa_id, turn['index']): turn for qa_id, turn in observed}
        if (len(actual) != len(observed) or len(set(declared)) != len(declared) or set(declared) != set(expected)
                or metadata.get('inherited_turn_count') != len(expected)
                or metadata.get('new_turn_count') != 111 - len(expected)):
            raise ValueError('Inherited turn plan differs from source')
        for key, original in expected.items():
            inherited = copy.deepcopy(actual.get(key))
            if not isinstance(inherited, dict):
                raise ValueError('Inherited turn is missing')
            relative = Path('inherited_turn_attempts') / safe_id(key[0]) / (str(key[1]) + '.json')
            if inherited.get('raw_attempt_path') != str(relative):
                raise ValueError('Inherited raw attempt path differs from the managed copy')
            source_relative = Path(original.get('raw_attempt_path', ''))
            if source_relative.is_absolute() or not source_relative.parts:
                raise ValueError('Original raw attempt path is invalid')
            source_raw = _regular_path(path.parent / source_relative, path.parent)
            copied_raw = _regular_path(directory / relative, directory)
            if sha256(source_raw) != sha256(copied_raw):
                raise ValueError('Inherited raw attempt copy hash mismatch')
            provenance = inherited.pop('record_provenance', {})
            inherited['raw_attempt_path'] = original.get('raw_attempt_path')
            if (inherited != original or provenance.get('kind') != 'inherited'
                    or provenance.get('run_id') != parent['run_id']
                    or provenance.get('run_sha256') != metadata['parent_run_sha256']):
                raise ValueError('Inherited measurements changed')
    summary = {key: metadata.get(key) for key in ('parent_run_id', 'parent_run_sha256', 'source_suite_id',
               'inherited_turn_count', 'new_turn_count', 'acknowledged_call_ids')}
    summary.update(kind='reference' if reference else 'continuation', source_verified=True)
    sources = [(path.parent, parent, 'inherited')]
    if continuation:
        sources.append((directory, run, 'new'))
    return sources, summary


def _audit(run, sources, ledger_path, acknowledgement):
    issues, journal_by_id, calls, segments = [], {}, [], []
    all_terminal = True
    inherited_ids = set()
    for directory, source, kind in sources:
        source_issues = []
        journal_path = directory / 'request-journal.sqlite'
        if journal_path.is_symlink():
            journal, source_issues = [], [{'kind': 'request_journal_symlink'}]
        else:
            journal, source_issues = journal_requests(journal_path)
        source_calls = []
        for filename in ('model_calls.jsonl', 'verifier_calls.jsonl', 'judge_calls.jsonl'):
            call_path = directory / filename
            if call_path.is_symlink():
                source_issues.append({'kind': 'trace_symlink', 'file': filename})
            elif call_path.is_file():
                source_calls.extend(_jsonl(call_path, source_issues))
        ids = set()
        for row in journal:
            identifier = row.get('call_id')
            if not identifier:
                source_issues.append({'kind': 'journal_call_id_missing'})
                continue
            ids.add(identifier)
            if identifier in journal_by_id and journal_by_id[identifier] != row:
                source_issues.append({'kind': 'conflicting_duplicate_call_id', 'call_id': identifier})
            journal_by_id.setdefault(identifier, row)
        if kind == 'inherited':
            inherited_ids.update(ids)
        calls.extend(source_calls)
        issues.extend({**issue, 'source_run_id': source['run_id']} for issue in source_issues)
        segments.append({'kind': kind, 'run_id': source['run_id'], 'call_ids': ids,
                         'wall_ms': source.get('wall_ms'), 'resources': source.get('resource_summary')})
        all_terminal = all_terminal and source.get('status') in TERMINAL
    ids = sorted(journal_by_id)
    ledger = _ledger_snapshot(Path(ledger_path), ids)
    audit = _own_cost(ids, 0, ledger, traces_complete=all_terminal, artifact_issues=issues)
    audit['attribution_assumption'] = 'Union of verified original and independent continuation journals, deduplicated by call_id. No smoke/build calls; the original unknown reservation is retained.'
    audit['journal_statuses'] = {status: sum(row['status'] == status for row in journal_by_id.values())
                               for status in sorted({row['status'] for row in journal_by_id.values()})}
    audit['settlement_complete'] = bool(audit['complete'] and ids
        and all(row['status'] in {'completed', 'failed'} for row in journal_by_id.values())
        and all(row['status'] in {'settled', 'released'} for row in ledger['rows'].values())
        and audit['known_reserved_cny'] == 0 and audit['uncertain_requests'] == 0)
    acknowledged = set((run.get('continuation') or {}).get('acknowledged_call_ids', []))
    ack_id = (acknowledgement or {}).get('call_id')
    # An arbitrary metadata list cannot exempt a fresh uncertain request.
    if acknowledged and (acknowledged != {ack_id} or not acknowledged <= inherited_ids):
        raise ValueError('Acknowledged request is not the fixed inherited request')
    if acknowledged:
        request = journal_by_id[ack_id]
        if any(acknowledgement.get(key) != request[key] for key in ('request_id', 'payload_hash')):
            raise ValueError('Acknowledged request identity or payload changed')
    outstanding = []
    for identifier in sorted(acknowledged):
        row = ledger['rows'].get(identifier)
        request = journal_by_id[identifier]
        if row and (request['status'] not in {'completed', 'failed'} or row['status'] not in {'settled', 'released'}):
            outstanding.append({'call_id': identifier, 'request_id': request['request_id'],
                'payload_hash': request['payload_hash'], 'journal_status': request['status'],
                'ledger_status': row['status'], 'reserved_cny': row['reserved_units'] / 1e6,
                'policy': 'preserve_reservation_skip_request', 'source': 'inherited'})
    audit['acknowledged_outstanding'] = outstanding
    attributed_ids = set()
    for segment in segments:
        unique_ids = segment.pop('call_ids') - attributed_ids
        attributed_ids.update(unique_ids)
        rows = [ledger['rows'][identifier] for identifier in unique_ids if identifier in ledger['rows']]
        segment.update(request_count=len(unique_ids), matched_request_count=len(rows),
            known_estimated_cny=sum(row['charged_units'] for row in rows) / 1e6,
            known_reserved_cny=sum(row['reserved_units'] for row in rows if row['status'] in {'pending', 'reserved', 'uncertain'}) / 1e6,
            calls_and_tokens=_usage(_unique_calls([call for call in calls if (call.get('budget_call_id') or call.get('call_id')) in unique_ids])))
    if not any(segment['kind'] == 'new' for segment in segments):
        segments.append({'kind': 'new', 'run_id': run['run_id'], 'request_count': 0,
                         'matched_request_count': 0, 'known_estimated_cny': 0., 'known_reserved_cny': 0., 'wall_ms': 0})
    audit['cost_segments'] = segments
    audit['gate_overhead'] = gate_overhead(run, _unique_calls(calls), ledger)
    audit['unacknowledged_outstanding_call_ids'] = sorted(identifier for identifier, request in journal_by_id.items()
        if identifier not in acknowledged and (request['status'] not in {'completed', 'failed'}
            or ledger['rows'].get(identifier, {}).get('status') not in {'settled', 'released'}))
    return audit, _usage(_unique_calls(calls))


def export(runtime=RUNTIME, *, ledger_path=None):
    runtime = Path(runtime).resolve()
    if runtime == Path(PARENT_RUNTIME).resolve():
        raise ValueError('Continuation export must not rewrite the original runtime')
    suite = read_json(runtime / 'formal-suite.json')
    if suite.get('schema_version') != 'camroll-flash-gate-continuation-suite-v1':
        raise ValueError('Expected a separately authorized continuation suite')
    groups, audits = [], {}
    for planned in suite['groups']:
        directory = runtime / 'evaluation/runs' / safe_id(planned['run_id'])
        path = directory / 'run.json'
        if not path.is_file():
            groups.append({**planned, 'execution_complete': False, 'work_complete': False,
                'financial_settlement_complete': False, 'acceptance_complete': False, 'summary': {
                'quality_distribution_planned': {'correct': 0, 'partial': 0, 'incorrect': 0, 'unscored': 100},
                'quality_correct_planned_rate': 0., 'quality_mean_fixed_planned': 0., 'quality_mean_scored': None,
                'retrieval_precision_macro': 0., 'retrieval_recall_macro': 0., 'retrieval_f1_macro': 0.,
                'retrieval_metric_count': 81, 'unexecuted_turn_count': 111, 'judge_valid_count': 0}})
            continue
        _regular_path(path, runtime)
        run = read_json(path)
        if (run.get('mode') != 'gate_ab' or run.get('protocol_label') != PROTOCOL
                or run.get('suite_id') != suite['suite_id'] or run.get('run_id') != planned['run_id']
                or run.get('inference_config', {}).get('group') != planned['name']):
            raise ValueError('Continuation run identity or protocol mismatch')
        sources, lineage = _lineage(run, directory)
        if (planned['name'] == 'F37-Off' and not run.get('reference')) or (planned['name'] == 'F37-On' and not run.get('continuation')):
            raise ValueError('Continuation suite is missing the required source lineage')
        audit, usage = _audit(run, sources, ledger_path or BASE / 'runtime/budget.sqlite', suite.get('acknowledgement'))
        execution_complete = planned.get('returncode') == 0 and validate_run(run)
        # A known, explicitly retained old reservation does not make executed
        # work disappear. Missing attribution or a fresh unknown is different:
        # it blocks acceptance of the independently assembled delivery.
        work_complete = bool(execution_complete and audit['complete']
                             and not audit['unacknowledged_outstanding_call_ids'])
        audit['work_complete'] = work_complete
        audit['financial_settlement_complete'] = audit['settlement_complete']
        audits[run['run_id']] = audit
        groups.append({**planned, 'status': run['status'], 'execution_complete': execution_complete,
            'work_complete': work_complete, 'acceptance_complete': work_complete,
            'financial_settlement_complete': audit['settlement_complete'], 'summary': run['summary'],
            'inference_config': run['inference_config'], 'index_binding': run.get('index_binding'),
            'cost_audit': audit, 'calls_and_tokens': usage, 'provenance': lineage,
            'timing_scope': 'Source and continuation elapsed time/resources are separate segments; inherited per-turn measurements retain original values.',
            'new_run_wall_ms': 0 if run.get('reference') else run.get('wall_ms'), 'run_json': str(path),
            'all_attempt_sources': [{'run_id': source['run_id'], 'kind': kind,
                'directory': str(folder)} for folder, source, kind in sources]})
    execution_complete = tuple(row['name'] for row in groups) == ORDER and all(row['execution_complete'] for row in groups)
    work_complete = bool(execution_complete and all(row['work_complete'] for row in groups) and suite.get('status') == 'passed'
        and suite.get('restoration', {}).get('state') == 'restored'
        and suite.get('model_files_unchanged') is True and suite.get('frozen_inputs_unchanged') is True
        and suite.get('wemm_weights_unchanged') is True)
    financial_complete = bool(groups and all(row['financial_settlement_complete'] for row in groups))
    outstanding = [row for group in groups for row in group.get('cost_audit', {}).get('acknowledged_outstanding', [])]
    report = {'schema_version': 'camroll-flash-gate-continuation-report-v1', 'observed_at': now_iso(),
        'status': ('complete' if financial_complete else 'work_complete_with_outstanding_settlement') if work_complete else 'incomplete',
        'work_complete': work_complete, 'execution_complete': execution_complete, 'acceptance_complete': work_complete,
        'financial_settlement_complete': financial_complete, 'acknowledged_outstanding': outstanding,
        'groups': groups, 'paired_comparisons': paired_deltas(groups), 'restoration': suite.get('restoration'),
        'budget_before': suite.get('budget_before'), 'budget_after': suite.get('budget_after'),
        'actual_subscription_bill_cny': None, 'actual_bill_status': 'unavailable', 'private_artifact': True,
        'interpretation': 'Independent continuation of the seen album3 benchmark. Original results/failures and unknown reservations are preserved. Fixed denominators: 100 questions, 111 turns, 81 delivery questions.',
        'gate_judge_correlation': 'Doubao Seed 2.0 Lite performs both independent final judging and evidence Gate.'}
    atomic_json(runtime / 'reports/flash-gate.json', report)
    atomic_json(runtime / 'evaluation/cost-audit.json', {'observed_at': now_iso(), 'runs': audits})
    atomic_json(runtime / 'evaluation/gate-comparisons.json', report['paired_comparisons'])
    lines = ['# Qwen Flash Gate 独立续测', '',
        f"观测：{report['observed_at']}；执行及必要评分完整：{work_complete}；费用结算完整：{financial_complete}。", '',
        'F37-Off引用既有完整结果；F37-On保留已执行轮次与失败，独立继续剩余轮次；F38两组独立执行。原未知请求没有重发，预留仍保留。', '',
        '正确率与质量固定100题，交付宏平均P/R/F1固定81题。正常回答、拒答和澄清进入评分，系统故障与Gate耗尽不作为正常拒答评分。', '',
        '| 组别 | 执行及评分完整 | 正确/部分/错误/未评分 | 有效评分数 | 有效 / 固定100题质量均分 | 已知等价费用 / 待结算预留 |',
        '|---|---|---|---|---|---|']
    for group in groups:
        s, c = group['summary'], group.get('cost_audit', {})
        d = s.get('quality_distribution_planned', {})
        distribution = '/'.join(str(d.get(key, '—')) for key in ('correct', 'partial', 'incorrect', 'unscored'))
        lines.append(f"| {group['name']} | {group['work_complete']} | {distribution} | {s.get('judge_valid_count')} | {s.get('quality_mean_scored')} / {s.get('quality_mean_fixed_planned')} | {c.get('known_estimated_cny')} / {c.get('known_reserved_cny')} |")
    lines += ['', '各来源的请求按call_id去重，原运行与新增运行费用、耗时、资源分列于JSON；未结算预留不计作已消费。套餐实付未接入。', '',
        'Doubao兼任事实Gate和独立最终评分，存在同模型关联；Gate通过不保证正确。album3为已见基准，不作为独立泛化能力证明。', '',
        '[只读逐题报告](http://192.168.0.100:8774/)；原记录及全部尝试保持独立可追溯。', '']
    target = runtime / 'reports/flash-gate.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    target.chmod(0o600)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    result = export(args.runtime, ledger_path=args.ledger)
    print(json.dumps({key: result[key] for key in ('status', 'work_complete', 'financial_settlement_complete')}))
