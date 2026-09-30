"""Export private four-group results without cloud calls or model operations."""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "camroll/camroll-agent"))

from camroll_eval.common import atomic_json, now_iso, read_json, safe_id
from camroll_eval.metrics import _fixed_macro, summary
from four_suite import ORDER, validate_run


def _jsonl(path, issues=None):
    if not path.is_file():
        if issues is not None:
            issues.append({"kind": "missing_call_artifact", "path": str(path)})
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("call_record_not_object")
                rows.append(value)
            except ValueError:
                if issues is not None:
                    issues.append({"kind": "incomplete_or_invalid_call_record", "path": str(path), "line": number})
    return rows


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _delta(before, after, key):
    left, right = before.get(key), after.get(key)
    return right - left if _number(left) and _number(right) else None


def _cost(before, after):
    return {"estimated_spent_delta_cny": _delta(before, after, "estimated_spent_cny"),
            "global_interval_estimated_spent_delta_cny": _delta(before, after, "estimated_spent_cny"),
            "reserved_before_cny": before.get("reserved_cny"), "reserved_after_cny": after.get("reserved_cny"),
            "uncertain_requests_after": after.get("uncertain_requests"),
            "limit_cny": after.get("limit_cny", before.get("limit_cny")),
            "actual_subscription_bill_cny": None, "actual_bill_status": "unavailable",
            "billing_basis": "shared_cumulative_ledger_usage_equivalent_estimate",
            "attribution_limit": "Ledger interval can include other authorized calls made concurrently."}


def _cloud_ids(calls):
    identifiers, unidentified = set(), 0
    for call in calls:
        role = call.get("call_type") or call.get("role")
        if role in {"embedding", "rerank", "retrieval_setup"} or str(call.get("operation") or "").startswith("/"):
            continue
        identifier = call.get("budget_call_id") or call.get("call_id")
        if isinstance(identifier, str) and identifier:
            identifiers.add(identifier)
        else:
            unidentified += 1
    return sorted(identifiers), unidentified


def _unique_calls(calls):
    """Count journal replays once; retain unidentifiable calls as audit gaps."""
    result, positions = [], {}
    for call in calls:
        identifier = call.get("budget_call_id") or call.get("call_id")
        if not isinstance(identifier, str) or not identifier:
            result.append(call)
            continue
        if identifier not in positions:
            positions[identifier] = len(result)
            result.append(call)
        elif result[positions[identifier]].get("replayed") and not call.get("replayed"):
            result[positions[identifier]] = call
    return result


def _ledger_snapshot(path, identifiers):
    result = {"observed_at": now_iso(), "ledger_path": str(path), "rows": {}, "status": "unavailable"}
    if not path.is_file():
        result["reason"] = "ledger_unavailable"
        return result
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        rows = {}
        for offset in range(0, len(identifiers), 900):
            selected = identifiers[offset:offset + 900]
            markers = ",".join("?" for _ in selected)
            for row in connection.execute("SELECT id,status,reserved_units,charged_units,metadata FROM budget_calls WHERE id IN (" + markers + ")", selected):
                metadata = json.loads(row[4])
                if not isinstance(metadata, dict) or any(not isinstance(value, int) or value < 0 for value in row[2:4]):
                    raise ValueError("invalid_ledger_record")
                rows[row[0]] = {"status": row[1], "reserved_units": row[2], "charged_units": row[3], "metadata": metadata}
        result.update(status="available", rows=rows)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        result["reason"] = "ledger_read_failed"
    finally:
        if connection is not None:
            connection.close()
    return result


def _recorded_rates(metadata):
    version = metadata.get("pricing_version")
    input_rate, output_rate = metadata.get("input_rate"), metadata.get("output_rate")
    source = "ledger_metadata"
    if not (_number(input_rate) and _number(output_rate)):
        prompt = metadata.get("prompt_tokens")
        if version == "qwen3.7-flash-cn-2026-09-15" and _number(prompt):
            input_rate, output_rate = (0.2, 0.8) if prompt <= 32768 else (0.6, 2.4) if prompt <= 262144 else (1.2, 4.8)
        elif version == "qwen3.8-flash-cn-2026-09-30":
            input_rate, output_rate = 0.8, 2.7
        elif version == "qwen3.8-max-cn-2026-09-30":
            input_rate, output_rate = 12.0, 36.0
        else:
            input_rate, output_rate = None, None
        source = "recorded_pricing_version_and_token_tier" if input_rate is not None else "unavailable"
    return {"pricing_version": version, "input_rate_cny_per_million": input_rate,
            "output_rate_cny_per_million": output_rate, "source": source}


def _own_cost(identifiers, unidentified, snapshot, *, traces_complete, artifact_issues):
    rows = snapshot["rows"]
    missing = [identifier for identifier in identifiers if identifier not in rows]
    models = {}
    for identifier in identifiers:
        if identifier not in rows:
            continue
        row = rows[identifier]
        metadata = row["metadata"]
        model = metadata.get("model") or "unknown"
        group = models.setdefault(model, {"model": model, "actual_models": set(), "request_count": 0,
            "charged_units": 0, "reserved_units": 0, "uncertain_requests": 0, "statuses": {},
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests_with_usage": 0, "rates": []})
        group["request_count"] += 1
        group["charged_units"] += row["charged_units"]
        if row["status"] in {"reserved", "uncertain"}:
            group["reserved_units"] += row["reserved_units"]
        group["uncertain_requests"] += int(row["status"] == "uncertain")
        group["statuses"][row["status"]] = group["statuses"].get(row["status"], 0) + 1
        if metadata.get("response_model"):
            group["actual_models"].add(metadata["response_model"])
        prompt, completion = metadata.get("prompt_tokens"), metadata.get("completion_tokens")
        if _number(prompt) and _number(completion):
            group["requests_with_usage"] += 1
            group["prompt_tokens"] += prompt
            group["completion_tokens"] += completion
            group["total_tokens"] += prompt + completion
        rates = _recorded_rates(metadata)
        if rates not in group["rates"]:
            group["rates"].append(rates)
    charged_units = sum(group["charged_units"] for group in models.values())
    reserved_units = sum(group["reserved_units"] for group in models.values())
    complete = bool(identifiers) and snapshot["status"] == "available" and not missing and not unidentified and traces_complete and not artifact_issues
    model_rows = []
    for model in sorted(models):
        group = models[model]
        group["estimated_cny"] = group.pop("charged_units") / 1_000_000
        group["reserved_cny"] = group.pop("reserved_units") / 1_000_000
        group["actual_models"] = sorted(group["actual_models"])
        if not group["requests_with_usage"]:
            group.update(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        model_rows.append(group)
    return {"status": "complete" if complete else "no_recorded_calls" if not identifiers and not unidentified else "unavailable" if snapshot["status"] != "available" else "partial",
            "complete": complete, "traces_complete": traces_complete,
            "own_estimated_cny": charged_units / 1_000_000 if complete else None,
            "own_reserved_cny": reserved_units / 1_000_000 if complete else None,
            "known_estimated_cny": charged_units / 1_000_000, "known_reserved_cny": reserved_units / 1_000_000,
            "request_count": len(identifiers), "matched_request_count": len(identifiers) - len(missing),
            "request_ids": identifiers, "missing_call_ids": missing, "unidentified_call_records": unidentified,
            "uncertain_requests": sum(group["uncertain_requests"] for group in model_rows),
            "by_model": model_rows, "artifact_issues": artifact_issues,
            "ledger_path": snapshot["ledger_path"], "ledger_snapshot_at": snapshot["observed_at"],
            "ledger_read_status": snapshot["status"], "ledger_read_reason": snapshot.get("reason"),
            "actual_subscription_bill_cny": None, "actual_bill_status": "unavailable",
            "scope": "Unique cloud request IDs recorded by this run; live in-flight requests may not yet have trace records."}


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
    resumed_at = state.get("resumed_at") or []
    pipeline_calls = []
    artifact_issues = []
    pipeline_artifacts = []
    pipeline_name = build.get("model_calls_file")
    album_id = state.get("album_id")
    if album_id:
        if pipeline_name and Path(pipeline_name).name != pipeline_name:
            raise ValueError("Pipeline call artifact must be an album-local filename")
        # Runner allocates eval-<run_id> exclusively to one run. A resumed build
        # writes a new trace; paid calls reused from its stage cache remain only
        # in earlier traces, so the final build filename is not the full scope.
        album = runtime / "albums" / safe_id(album_id)
        paths = set(album.glob("pipeline_calls-*.jsonl"))
        if pipeline_name:
            paths.add(album / pipeline_name)
        if not paths:
            artifact_issues.append({"kind": "unbound_pipeline_call_artifact", "candidate_count": 0})
        for path in sorted(paths):
            pipeline_artifacts.append(str(path))
            pipeline_calls.extend(_jsonl(path, artifact_issues))
        pipeline_calls = _unique_calls(pipeline_calls)
    else:
        artifact_issues.append({"kind": "album_not_started"})
    calls = list(pipeline_calls)
    calls.extend(call for item in items for call in item.get("model_call_metrics") or [])
    judge_calls = _jsonl(run_path.parent / "judge_calls.jsonl", artifact_issues) if state else []
    calls.extend(judge_calls)
    identifiers, unidentified = _cloud_ids(calls)
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
            "resumed_at": resumed_at, "timing_comparable": state.get("timing_comparable"),
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
            "timings": {"scope": {
                            "build_wall_ms": "last_recorded_pipeline_invocation_segment_only" if resumed_at else "recorded_pipeline_invocation",
                            "question_latency": "recorded_agent_wall_per_question_including_retained_turns_excluding_judge",
                            "call_durations": "recorded_calls_including_retained_turns; not_reconstructed_process_wall_time"},
                        "build_wall_ms": ((state.get("phases") or {}).get("pipeline") or {}).get("wall_ms"),
                        "question_mean_ms": metrics.get("agent_task_latency_mean_ms"),
                        "question_p50_ms": metrics.get("agent_task_latency_p50_ms"),
                        "question_p95_ms": metrics.get("agent_task_latency_p95_ms"),
                        "question_samples": metrics.get("agent_task_latency_sample_count"),
                        "local_retrieval": metrics.get("local_retrieval_performance", {}),
                        "cloud_roles": metrics.get("model_role_performance", {}), "build_stages": build.get("stages", [])},
            "calls_and_tokens": _usage(calls), "cost": _cost(state.get("budget_before") or {}, state.get("budget_after") or {}),
            "cost_trace": {"request_ids": identifiers, "unidentified_call_records": unidentified,
                           "traces_complete": state.get("status") in {"completed", "completed_with_errors"} and bool(pipeline_name),
                           "album_id": album_id, "pipeline_call_artifacts": pipeline_artifacts,
                           "attribution_assumption": "The album is exclusive to this run; all of its pipeline call traces, including resumed builds, belong to this run.",
                           "artifact_issues": artifact_issues},
            "sampled_resources": state.get("resource_summary"),
            "sampled_resources_scope": "latest_evaluator_process_segment_only" if resumed_at else "recorded_evaluator_process_samples",
            "resources_jsonl": str(run_path.parent / "resources.jsonl") if state else None,
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
              "| 组别 | 图片向量/描述覆盖 | 事件向量/事件 | 建库秒 | 单题均值/P50/P95 秒 | 本组已记账估算 | 全局区间增量 | Token 总数 |",
              "|---|---|---|---:|---|---:|---:|---:|"]
    for group in report["groups"]:
        coverage, timings = group["coverage"], group["timings"]
        durations = "/".join(_format(timings.get(name) / 1000 if _number(timings.get(name)) else None)
                             for name in ("question_mean_ms", "question_p50_ms", "question_p95_ms"))
        build = timings["build_wall_ms"]
        lines.append(f"| {group['name']} | {coverage['image_embedding_count']}/{coverage['caption_count']}（共 {coverage['image_count']} 图） | {coverage['event_embedding_count']}/{coverage['event_count']} | {_format(build / 1000 if _number(build) else None)} | {durations} | ¥{_format(group['cost_audit']['own_estimated_cny'], digits=4)} | ¥{_format(group['cost']['global_interval_estimated_spent_delta_cny'], digits=4)} | {group['calls_and_tokens']['token_totals']['total_tokens']} |")
    lines += ["", "本组费用按建库、问答和 Judge 轨迹的唯一请求 ID，从共享账本只读核对；全局区间增量可能包含其他并发项目，不能归为本组。记录尚未齐全时本组总费用显示未观测，JSON 单列已找到请求的小计、预留金额、缺失 ID 和模型费率。",
              "费用为等价估算；套餐实际账单未取得，不将估算标为实付。未记录 usage 的调用不进入 Token 合计。", "",
              "单题时延为该题全部已记录轮次的 Agent 墙钟，不含 Judge；资源峰值为实际采样峰值，宿主机 RAM 包含其他进程。", "",
              f"恢复状态：{report['restoration'].get('state', '未观测')}；本轮 GPU 残留清理证据：{report['restoration'].get('no_experiment_gpu_residual', '未观测')}。", "",
              "历史 M1：75 正确、14 部分正确、11 错误，仅作历史参考，不作为本轮同期基线。", ""]
    for group in report["groups"]:
        if group.get("resumed_at"):
            lines += [f"{group['name']} 已发生恢复：保留 timing_comparable={group['timing_comparable']}，不可将恢复片段视为完整冷启动时延。建库墙钟仅为最后记录的一次流水线调用片段，资源峰值仅覆盖最后一次评测进程；未重建整组总耗时或全程峰值。", ""]
    for group in report["groups"]:
        lines += [f"## {group['name']} 逐题记录", "", f"运行状态：{group['status']}；完整私有文件：{group['run_json'] or '尚未创建'}。", ""]
        if not group["questions"]:
            lines += ["尚未生成题目记录。", ""]
        for item in group["questions"]:
            score = (item.get("judge") or {}).get("score")
            lines.append(f"- [{item['qa_id']}]({item['full_record_url']})：{item['execution_status']} / {item['judge_status']}，质量评分 {score if score is not None else '未评分'}。")
        lines.append("")
    return "\n".join(lines)


def export(runtime_root, *, formal=True, report_url="http://192.168.0.100:8774", ledger_path=None):
    from four_config import BASE
    runtime = Path(runtime_root).resolve()
    ledger_path = Path(ledger_path) if ledger_path is not None else BASE / "runtime/budget.sqlite"
    suite_path = runtime / ("formal-suite.json" if formal else "smoke-suite.json")
    suite = read_json(suite_path) if suite_path.is_file() else {}
    candidates = {group["name"]: group for group in suite.get("groups", [])}
    groups = [_group(runtime, candidates.get(name, {"name": name, "status": "not_started"}), formal=formal,
                     report_url=report_url.rstrip("/")) for name in ORDER]
    identifiers = sorted({identifier for group in groups for identifier in group["cost_trace"]["request_ids"]})
    snapshot = _ledger_snapshot(ledger_path, identifiers)
    for group in groups:
        trace = group.pop("cost_trace")
        group["cost_audit"] = _own_cost(trace["request_ids"], trace["unidentified_call_records"], snapshot,
                                       traces_complete=trace["traces_complete"], artifact_issues=trace["artifact_issues"])
        group["cost_audit"].update({key: trace[key] for key in ("album_id", "pipeline_call_artifacts", "attribution_assumption")})
    suite_audit = _own_cost(identifiers, sum(group["cost_audit"]["unidentified_call_records"] for group in groups), snapshot,
                           traces_complete=all(group["cost_audit"]["traces_complete"] for group in groups),
                           artifact_issues=[issue for group in groups for issue in group["cost_audit"]["artifact_issues"]])
    restoration = suite.get("restoration") or {}
    complete = all(group["acceptance_complete"] for group in groups) and restoration.get("state") == "restored" and restoration.get("no_experiment_gpu_residual") is True
    report = {"schema_version": "camroll-four-models-report-v1", "observed_at": now_iso(), "status": "complete" if complete else "incomplete",
              "suite_status": suite.get("status", "not_started"), "suite_id": suite.get("suite_id"), "formal": formal,
              "suite_json": str(suite_path), "report_url": report_url.rstrip("/"), "groups": groups,
              "restoration": restoration, "suite_cost": _cost(suite.get("budget_before") or {}, suite.get("budget_after") or {}),
              "suite_cost_audit": suite_audit,
              "historical_reference": {"label": "Historical M1 only", "correct": 75, "partial": 14, "incorrect": 11,
                                       "contemporary_baseline": False}, "private_artifact": True}
    output = runtime / "reports" / ("four-models" if formal else "four-models-smoke")
    atomic_json(output.with_suffix(".json"), report)
    atomic_json(runtime / ("evaluation" if formal else "smoke-evaluation") / "cost-audit.json",
                {"observed_at": snapshot["observed_at"], "runs": {group["run_id"]: group["cost_audit"] for group in groups if group["run_id"]}})
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
    parser.add_argument("--ledger", type=Path, help="Existing shared budget SQLite; opened read-only")
    args = parser.parse_args(argv)
    report = export(args.runtime, formal=not args.smoke, report_url=args.report_url, ledger_path=args.ledger)
    print(json.dumps({"status": report["status"], "groups": {group["name"]: group["status"] for group in report["groups"]},
                      "reports": str(args.runtime / "reports")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
