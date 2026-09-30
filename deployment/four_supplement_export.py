"""Read-only, separately attributed report for a requested post-403 supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from four_config import BASE, RUNTIME
from four_export import _cloud_ids, _fixed_macro, _jsonl, _ledger_snapshot, _own_cost, _unique_calls, _usage
from four_supplement import validate_supplement
from camroll_eval.common import atomic_json, now_iso, read_json, safe_id
from camroll_eval.supplement import supplement_summary

TERMINAL = {'completed', 'completed_with_errors', 'failed', 'cancelled', 'blocked', 'interrupted'}


def export(runtime_root, run_id, *, ledger_path=None, report_url='http://192.168.0.100:8774'):
    runtime, run_id = Path(runtime_root), safe_id(run_id)
    run_path = runtime / 'evaluation/runs' / run_id / 'run.json'
    state = read_json(run_path)
    if state.get('mode') != 'supplement':
        raise ValueError('only separately recorded supplements may be exported')
    plans, items = state.get('qa_plan') or [], state.get('items') or []
    n, turns = len(plans), sum(p['planned_turn_count'] for p in plans)
    eligible_n = state.get('planned_eligible_qa_count')
    if (n, turns, eligible_n) != (64, 71, 52):
        raise ValueError('supplement plan differs from the authorized 64/71/52 scope')
    scores = [(i.get('judge') or {}).get('score') for i in items]
    scores = [v for v in scores if type(v) is int and v in (0, 1, 2)]
    distribution = dict(correct=scores.count(2), partial=scores.count(1), incorrect=scores.count(0), unscored=n-len(scores))
    actions = [t.get('task_judge') or {} for i in items for t in i.get('runtime_turns') or []]
    valid_actions = [a for a in actions if type(a.get('correct')) is bool and a.get('actual_action') in {'answer', 'refuse', 'clarify', 'none'}]
    eligible = [i for i in items if i.get('answerability') != 'unanswerable']
    issues = []
    judge_path = run_path.parent / 'judge_calls.jsonl'
    judge_calls = _jsonl(judge_path, issues) if judge_path.exists() or state['status'] in TERMINAL else []
    calls = _unique_calls([c for i in items for c in i.get('model_call_metrics') or []] + judge_calls)
    identifiers, unidentified = _cloud_ids(calls)
    ledger = _ledger_snapshot(Path(ledger_path or BASE/'runtime/budget.sqlite'), identifiers)
    audit = _own_cost(identifiers, unidentified, ledger, traces_complete=state['status'] in TERMINAL, artifact_issues=issues)
    audit.update(attribution_assumption='Only this new supplement run; no parent building, answers, Judge calls or recharge probe costs included.',
                 album_id=None, pipeline_call_artifacts=[], parent_run_id=state.get('parent_run_id'))
    supervisor_path = runtime / 'supplements' / run_id / 'supervisor.json'
    supervisor = read_json(supervisor_path) if supervisor_path.exists() else {}
    observed = {i['qa_id']: i for i in items}
    questions = []
    for p in plans:
        i = observed.get(p['qa_id'], {})
        questions.append({'qa_id': p['qa_id'], 'planned_turn_count': p['planned_turn_count'],
            'executed_turn_count': len(i.get('runtime_turns') or []), 'execution_status': i.get('execution_status', 'not_started'),
            'judge_status': i.get('judge_status', 'not_started'), 'score': (i.get('judge') or {}).get('score'),
            'termination_reason': i.get('termination_reason'),
            'full_record_url': report_url.rstrip('/')+'/api/runs/'+quote(run_id, safe='')+'/items/'+quote(p['qa_id'], safe='')})
    restoration = supervisor.get('restoration') or {}
    report = {'schema_version': 'camroll-post-403-supplement-report-v1', 'observed_at': now_iso(),
        'run_id': run_id, 'parent_run_id': state.get('parent_run_id'), 'mode': 'supplement',
        'status': state['status'], 'acceptance_complete': validate_supplement(state) and restoration.get('state') == 'restored',
        'title': 'W38Max 恢复后64题补测 · 复用原快照 · 独立报告', 'formal_first_attempt_replaced': False,
        'scope': '64 questions selected solely from original HTTP403 calls. Fresh question contexts and new paid request identities. No cold build; no merging with first-attempt scores.',
        'planned_questions': n, 'planned_turns': turns, 'planned_eligible_questions': eligible_n,
        'executed_questions': len(items), 'executed_turns': sum(len(i.get('runtime_turns') or []) for i in items),
        'quality_distribution': distribution, 'accuracy_fixed_planned': distribution['correct']/n,
        'quality_mean_scored': sum(scores)/len(scores) if scores else None,
        'quality_mean_fixed_planned': sum(scores)/n,
        'action_accuracy_scored': sum(a['correct'] for a in valid_actions)/len(valid_actions) if valid_actions else None,
        'action_scored_turns': len(valid_actions), 'action_unscored_turns': turns-len(valid_actions),
        'final_delivery_macro': _fixed_macro(eligible, 'image_retrieval_counts', eligible_n),
        'candidate_retrieval_macro': _fixed_macro(eligible, 'candidate_retrieval_counts', eligible_n),
        'snapshot_version': state.get('index_version'), 'build_performed': False, 'build_cost_cny': 0.0,
        'timing_comparable_to_cold_runs': False, 'summary': supplement_summary(state),
        'calls_and_tokens': _usage(calls), 'cost_audit': audit, 'sampled_resources': state.get('resource_summary'),
        'restoration': restoration, 'questions': questions, 'run_json': str(run_path), 'private_artifact': True}
    output = runtime / 'reports' / run_id
    validity_path = runtime / 'evaluation/supplement-validity-audit.json'
    if validity_path.is_file():
        validity = (read_json(validity_path).get('runs') or {}).get(run_id)
        if isinstance(validity, dict):
            report['validity_audit'] = validity
            if validity.get('valid') is False:
                report['acceptance_complete'] = False
                report['title'] = 'W38Max 补测实施无效 · 仅故障存证 · 不用于模型比较'
    atomic_json(output.with_suffix('.json'), report)
    overlay_path = runtime / 'evaluation/supplement-cost-audit.json'
    overlay = read_json(overlay_path) if overlay_path.exists() else {'runs': {}}
    overlay['observed_at'] = ledger['observed_at']; overlay['runs'][run_id] = audit
    atomic_json(overlay_path, overlay)
    d=distribution; m=report['final_delivery_macro']
    fmt=lambda v: '未观测' if v is None else f'{v:.6f}'
    lines=['# W38Max 充值恢复后64题补测', '', f"观测：{report['observed_at']}；状态：{state['status']}；完整性验收：{report['acceptance_complete']}。", '',
        '沿用原W38Max快照，只重做原HTTP403影响的64题、71轮。正确率固定64题，检索宏平均固定52道可回答题；这份补测不替换原四组首跑，也不与原答案合并。', '',
        f"正确/部分/错误/未评分：{d['correct']}/{d['partial']}/{d['incorrect']}/{d['unscored']}；已执行{len(items)}/64题、{report['executed_turns']}/71轮。", '',
        f"正确率：{100*report['accuracy_fixed_planned']:.2f}%；有效质量均分：{fmt(report['quality_mean_scored'])}/2。交付宏P/R/F1：{100*m['precision']:.2f}%/{100*m['recall']:.2f}%/{100*m['f1']:.2f}%。", '',
        f"本补测精确归属等价费用：¥{fmt(audit['own_estimated_cny'])}；已知小计¥{audit['known_estimated_cny']:.6f}。不含原建库、首跑或充值验证费用。套餐实付未知。", '',
        f"原快照：{state.get('index_version')}；没有重新建库，冷建库时延不适用。恢复状态：{restoration.get('state','尚未结束')}。", '',
        f"[只读报告]({report_url})；完整记录：`{run_path}`。", '',
        '| 题号 | 执行轮次/计划 | 执行状态 | 评分状态 | 最终分数 |', '|---|---|---|---|---|']
    for q in questions:
        lines.append(f"| [{q['qa_id']}]({q['full_record_url']}) | {q['executed_turn_count']}/{q['planned_turn_count']} | {q['execution_status']} | {q['judge_status']} | {q['score'] if q['score'] is not None else '未评分'} |")
    if report.get('validity_audit', {}).get('valid') is False:
        lines[2:2] = ['**实施无效，不用于模型比较；下列答案、评分与费用仅作故障存证。**', '',
                       report['validity_audit'].get('reason', 'Implementation validity audit failed.'), '']
    path=output.with_suffix('.md'); tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.chmod(0o600);tmp.replace(path)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--ledger', type=Path)
    args=parser.parse_args(argv)
    report=export(args.runtime,args.run_id,ledger_path=args.ledger)
    print(json.dumps({'run_id':report['run_id'],'status':report['status'],'acceptance_complete':report['acceptance_complete'],
                      'distribution':report['quality_distribution'],'cost_cny':report['cost_audit']['own_estimated_cny']},ensure_ascii=False))


if __name__=='__main__':
    main()
