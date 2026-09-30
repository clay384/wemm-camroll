"""Export private four-group results without cloud calls or model operations."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import quote

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "camroll/camroll-agent"))

from camroll_eval.common import atomic_json, now_iso, read_json, safe_id
from camroll_eval.metrics import _fixed_macro, summary
from four_suite import ORDER, validate_run


def _jsonl(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _delta(before, after, key):
    left, right = before.get(key), after.get(key)
    return right - left if _number(left) and _number(right) else None


def _cost(before, after):
    return {"estimated_spent_delta_cny": _delta(before, after, "estimated_spent_cny"),
            "reserved_before_cny": before.get("reserved_cny"), "reserved_after_cny": after.get("reserved_cny"),
            "uncertain_requests_after": after.get("uncertain_requests"),
            "limit_cny": after.get("limit_cny", before.get("limit_cny")),
            "actual_subscription_bill_cny": None, "actual_bill_status": "unavailable",
            "billing_basis": "shared_cumulative_ledger_usage_equivalent_estimate",
            "attribution_limit": "Ledger interval can include other authorized calls made concurrently."}


def _usage(calls):
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    known = 0
    models = set()
    groups = {}
    for call in calls:
        role = call.get("original_call_type") or call.get("call_type") or call.get("role") or "unknown"
        group = groups.setdefault(role, {"calls": 0, "errors": 0, "duration_ms_total": 0.0})
        group["calls"] += 1
        group["errors"] += int(call.get("status") == "error")
        if _number(call.get("duration_ms")):
            group["duration_ms_total"] += call["duration_ms"]
        usage = call.get("usage") or call
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if _number(prompt) and _number(completion):
            known += 1
            tokens["prompt_tokens"] += prompt
            tokens["completion_tokens"] += completion
            tokens["total_tokens"] += prompt + completion
        if call.get("model") and role not in {"embedding", "rerank", "judge"}:
            models.add(call["model"])
    return {"calls": len(calls), "calls_with_known_tokens": known, "token_totals": tokens,
            "observed_generation_models": sorted(models), "by_role": groups,
            "judge_response_models": sorted({call["response_model"] for call in calls if call.get("response_model")}),
            "token_scope": "recorded_build_answer_and_judge_calls; missing_usage_excluded"}


def _group(runtime, group, *, formal, report_url):
    question_count, turn_count, retrieval_count = (100, 111, 81) if formal else (3, 4, 2)
    run_id = group.get("run_id")
    root = runtime / ("evaluation" if formal else "smoke-evaluation")
    run_path = root / "runs" / safe_id(run_id) / "run.json" if run_id else None
    state = read_json(run_path) if run_path and run_path.is_file() else {}
    items = state.get("items") or []
    scores = [item["judge"]["score"] for item in items if isinstance((item.get("judge") or {}).get("score"), int)
              and not isinstance(item["judge"]["score"], bool) and item["judge"]["score"] in (0, 1, 2)]
    distribution = {"correct": scores.count(2), "partial": scores.count(1), "incorrect": scores.count(0),
                    "unscored": max(0, question_count - len(scores))}
    actions = [turn.get("task_judge") or {} for item in items for turn in item.get("runtime_turns") or []]
    valid_actions = [value for value in actions if isinstance(value.get("correct"), bool)
                     and value.get("actual_action") in {"answer", "clarify", "refuse", "none"}]
    metrics = summary(state) if state else {}
    eligible = [item for item in items if item.get("answerability") != "unanswerable"]
    build = state.get("build_summary") or {}
    pipeline_calls = []
    pipeline_name = build.get("model_calls_file")
    album_id = state.get("album_id")
    if pipeline_name and album_id:
        if Path(pipeline_name).name != pipeline_name:
            raise ValueError("Pipeline call artifact must be an album-local filename")
        pipeline_calls = _jsonl(runtime / "albums" / safe_id(album_id) / pipeline_name)
    calls = list(pipeline_calls)
    calls.extend(call for item in items for call in item.get("model_call_metrics") or [])
    judge_calls = _jsonl(run_path.parent / "judge_calls.jsonl") if state else []
    calls.extend(judge_calls)
    records_by_id = {item["qa_id"]: item for item in items}
    plan = state.get("qa_plan") or state.get("planned_status") or [{"qa_id": item["qa_id"]} for item in items]
    questions = []
    for planned in plan:
        item = records_by_id.get(planned["qa_id"], {})
        questions.append({"qa_id": planned["qa_id"], "question": item.get("question", planned.get("question")),
                          "planned_turn_count": planned.get("planned_turn_count"),
                          "execution_status": item.get("execution_status", "not_started"),
                          "judge_status": item.get("judge_status", "not_started"),
                          "judge": item.get("judge"), "answer": item.get("answer"),
                          "selected_asset_ids": item.get("selected_asset_ids", []),
                          "image_retrieval_counts": item.get("image_retrieval_counts"),
                          "runtime_turn_count": len(item.get("runtime_turns") or []),
                          "full_record_url": report_url + "/api/runs/" + quote(str(run_id), safe="") + "/items/" + quote(str(planned["qa_id"]), safe=""),
                          "full_record_file": str(run_path) if run_path else None,
                          "full_record_selector": {"items.qa_id": planned["qa_id"]}})
    return {"name": group["name"], "run_id": run_id,
            "status": state.get("status", group.get("status", "not_started")),
            "acceptance_complete": bool(state) and validate_run(state, formal=formal),
            "run_json": str(run_path) if run_path else None, "formal": formal,
            "model_profile": state.get("experiment_config"), "index_version": state.get("index_version"),
            "planned_questions": question_count, "planned_turns": turn_count,
            "executed_questions": len(items), "executed_turns": sum(len(item.get("runtime_turns") or []) for item in items),
            "quality_distribution": distribution, "accuracy_fixed_planned": distribution["correct"] / question_count,
            "quality_mean_scored": sum(scores) / len(scores) if scores else None,
            "quality_mean_fixed_planned": sum(scores) / question_count,
            "action_accuracy_scored": sum(value["correct"] for value in valid_actions) / len(valid_actions) if valid_actions else None,
            "action_accuracy_fixed_planned": sum(value["correct"] for value in valid_actions) / turn_count,
            "action_scored_turns": len(valid_actions), "action_unscored_turns": max(0, turn_count - len(valid_actions)),
            "final_delivery_macro": _fixed_macro(eligible, "image_retrieval_counts", retrieval_count),
            "candidate_retrieval_macro": _fixed_macro(eligible, "candidate_retrieval_counts", retrieval_count),
            "retrieval_diagnostics": metrics.get("retrieval_diagnostics"),
            "coverage": {key: build.get(key) for key in ("image_count", "image_embedding_count", "caption_count",
                         "text_embedding_count", "text_coverage", "event_count", "event_embedding_count")},
            "timings": {"build_wall_ms": ((state.get("phases") or {}).get("pipeline") or {}).get("wall_ms"),
                        "question_mean_ms": metrics.get("agent_task_latency_mean_ms"),
                        "question_p50_ms": metrics.get("agent_task_latency_p50_ms"),
                        "question_p95_ms": metrics.get("agent_task_latency_p95_ms"),
                        "question_samples": metrics.get("agent_task_latency_sample_count"),
                        "local_retrieval": metrics.get("local_retrieval_performance", {}),
                        "cloud_roles": metrics.get("model_role_performance", {}), "build_stages": build.get("stages", [])},
            "calls_and_tokens": _usage(calls), "cost": _cost(state.get("budget_before") or {}, state.get("budget_after") or {}),
            "sampled_resources": state.get("resource_summary"), "resources_jsonl": str(run_path.parent / "resources.jsonl") if state else None,
            "build_errors": build.get("errors", []), "fatal_error": state.get("fatal_error"), "questions": questions}


def _format(value, *, percent=False, digits=2):
    return "未观测" if value is None else (f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}")


def markdown(report):
    lines = ["# CamRoll 四组完整系统评测", "", f"观测时间：{report['observed_at']}。交付状态：**{report['status']}**。",
             f"报告页面：[8774 比较页]({report['report_url']})。完整逐题记录及工具轨迹保留在私有运行 JSON。", "",
             "| 组别 | 状态 | 正确/部分/错误/未评分 | 固定题数正确率 | 已评分质量均分 /2 | 行为准确率 | 交付 P/R/F1 |",
             "|---|---|---|---:|---:|---:|---|" ]
    for group in report["groups"]:
        distribution, delivery = group["quality_distribution"], group["final_delivery_macro"]
        values = "/".join(str(distribution[name]) for name in ("correct", "partial", "incorrect", "unscored"))
        prf = "/".join(_format(delivery[name], percent=True) for name in ("precision", "recall", "f1"))
        lines.append(f"| {group['name']} | {group['status']} | {values} | {_format(group['accuracy_fixed_planned'], percent=True)} | {_format(group['quality_mean_scored'])} | {_format(group['action_accuracy_scored'], percent=True)} | {prf} |")
    lines += ["", "正确率分母为全部计划题目；质量均分、行为准确率使用已评分样本，并在 JSON 单列缺失数及固定分母指标。",
              f"交付 P/R/F1 分母为 {'81 道可回答题' if report['formal'] else '2 道合成可回答题'}；未执行题贡献为 0。四组比较完整系统效果，不能单独归因于 embedding 或 reranker。", "",
              "| 组别 | 图片向量/描述覆盖 | 事件向量/事件 | 建库秒 | 单题均值/P50/P95 秒 | 累计账本费用增量 | Token 总数 |",
              "|---|---|---|---:|---|---:|---:|"]
    for group in report["groups"]:
        coverage, timings = group["coverage"], group["timings"]
        durations = "/".join(_format(timings.get(name) / 1000 if _number(timings.get(name)) else None)
                             for name in ("question_mean_ms", "question_p50_ms", "question_p95_ms"))
        build = timings["build_wall_ms"]
        lines.append(f"| {group['name']} | {coverage['image_embedding_count']}/{coverage['caption_count']}（共 {coverage['image_count']} 图） | {coverage['event_embedding_count']}/{coverage['event_count']} | {_format(build / 1000 if _number(build) else None)} | {durations} | ¥{_format(group['cost']['estimated_spent_delta_cny'], digits=4)} | {group['calls_and_tokens']['token_totals']['total_tokens']} |")
    lines += ["", "费用为共享累计账本的等价估算；套餐实际账单未取得，不将估算标为实付。未记录 usage 的调用不进入 Token 合计。", "",
              f"恢复状态：{report['restoration'].get('state', '未观测')}；本轮 GPU 残留清理证据：{report['restoration'].get('no_experiment_gpu_residual', '未观测')}。", "",
              "历史 M1：75 正确、14 部分正确、11 错误，仅作历史参考，不作为本轮同期基线。", ""]
    for group in report["groups"]:
        lines += [f"## {group['name']} 逐题记录", "", f"运行状态：{group['status']}；完整私有文件：{group['run_json'] or '尚未创建'}。", ""]
        if not group["questions"]:
            lines += ["尚未生成题目记录。", ""]
        for item in group["questions"]:
            score = (item.get("judge") or {}).get("score")
            lines.append(f"- [{item['qa_id']}]({item['full_record_url']})：{item['execution_status']} / {item['judge_status']}，质量评分 {score if score is not None else '未评分'}。")
        lines.append("")
    return "\n".join(lines)


def export(runtime_root, *, formal=True, report_url="http://192.168.0.100:8774"):
    runtime = Path(runtime_root).resolve()
    suite_path = runtime / ("formal-suite.json" if formal else "smoke-suite.json")
    suite = read_json(suite_path) if suite_path.is_file() else {}
    candidates = {group["name"]: group for group in suite.get("groups", [])}
    groups = [_group(runtime, candidates.get(name, {"name": name, "status": "not_started"}), formal=formal,
                     report_url=report_url.rstrip("/")) for name in ORDER]
    restoration = suite.get("restoration") or {}
    complete = all(group["acceptance_complete"] for group in groups) and restoration.get("state") == "restored" and restoration.get("no_experiment_gpu_residual") is True
    report = {"schema_version": "camroll-four-models-report-v1", "observed_at": now_iso(), "status": "complete" if complete else "incomplete",
              "suite_status": suite.get("status", "not_started"), "suite_id": suite.get("suite_id"), "formal": formal,
              "suite_json": str(suite_path), "report_url": report_url.rstrip("/"), "groups": groups,
              "restoration": restoration, "suite_cost": _cost(suite.get("budget_before") or {}, suite.get("budget_after") or {}),
              "historical_reference": {"label": "Historical M1 only", "correct": 75, "partial": 14, "incorrect": 11,
                                       "contemporary_baseline": False}, "private_artifact": True}
    output = runtime / "reports" / ("four-models" if formal else "four-models-smoke")
    atomic_json(output.with_suffix(".json"), report)
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text(markdown(report))
    temporary.chmod(0o600)
    os.replace(temporary, output.with_suffix(".md"))
    return report


def main(argv=None):
    from four_config import RUNTIME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-url", default="http://192.168.0.100:8774")
    args = parser.parse_args(argv)
    report = export(args.runtime, formal=not args.smoke, report_url=args.report_url)
    print(json.dumps({"status": report["status"], "groups": {group["name"]: group["status"] for group in report["groups"]},
                      "reports": str(args.runtime / "reports")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
