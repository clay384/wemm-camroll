"""Normalize CamRoll observations into the frozen PhotoBench scoring contract."""
from __future__ import annotations

import copy
from pathlib import Path

from .compat import (BenchmarkRun, OrchestratorRepository, _media_key,
                     _normalize_media_refs, _modality_metrics, _macro_metrics_from_counts, nearest_rank_percentile)


def resolve_ids(ids, media):
    """Unknown delivery IDs remain false positives, never silently disappear."""
    return [copy.deepcopy(media.get(value) or {
        "asset_id": value, "media_id": "__unmapped__" + str(value),
        "file_name": "__unmapped__" + str(value), "media_type": "image", "mapping_error": True,
    }) for value in dict.fromkeys(ids or [])]


def add_media_metrics(item, response, media):
    gt = _normalize_media_refs(item)
    deliveries = resolve_ids(response.get("selected_image_ids", []), media)
    retrieved = resolve_ids(response.get("retrieved_image_ids", []), media)
    evidence = resolve_ids(response.get("evidence_image_ids", []), media)
    metrics = _modality_metrics(gt, deliveries)
    candidate_metrics = _modality_metrics(gt, retrieved)
    item.update(predicted_media=deliveries, predicted_images=deliveries,
                retrieved_candidate_media=retrieved, retrieved_candidate_images=retrieved,
                evidence_source_media=evidence, evidence_source_images=evidence,
                selected_asset_ids=response.get("selected_image_ids") or [],
                retrieved_asset_ids=response.get("retrieved_image_ids") or [],
                evidence_asset_ids=response.get("evidence_image_ids") or [],
                mapping_error_count=sum(bool(x.get("mapping_error")) for x in deliveries))
    gt_keys = {_media_key(ref["media_type"], ref["media_id"]) for ref in gt}
    for prefix, values in [("predicted", deliveries), ("retrieved", retrieved), ("evidence_source", evidence)]:
        item[prefix + "_file_names"] = sorted({x["file_name"] for x in values})
    for key, values in [("matched_file_names", deliveries), ("retrieved_matched_file_names", retrieved),
                        ("evidence_matched_file_names", evidence)]:
        item[key] = sorted({x["file_name"] for x in values
                            if _media_key(x["media_type"], x["media_id"]) in gt_keys})
    item["delivery_matched_file_names"] = item["matched_file_names"]
    item["selected_delivery_file_names"] = item["predicted_file_names"]
    item["gt_media"] = [{**ref, "file_name": Path(ref["media_id"]).name} for ref in gt]
    item["gt_images"] = [ref for ref in item["gt_media"] if ref["media_type"] == "image"]
    for kind in ("media", "image", "video"):
        item[kind + "_retrieval_counts"] = metrics[kind]
        for field in ("precision", "recall", "f1"):
            item[kind + "_retrieval_" + field] = metrics[kind][field]
    for field in ("precision", "recall", "f1"):
        item["retrieval_" + field] = metrics["media"][field]
    item["candidate_retrieval_counts"] = candidate_metrics["media"]


def initial_item(row):
    fields = ("qa_id", "question", "task_type", "question_type", "tags", "angle", "difficulty",
              "answerability", "scope", "scope_anchor", "required_evidence_sources", "query_anchors",
              "expected_action", "answer_claims")
    item = {key: copy.deepcopy(row[key]) for key in fields if key in row}
    refs = _normalize_media_refs(row)
    item.update(reference_answer=row.get("answer", ""), retrieval_media_refs=refs,
                retrieval_image_ids=[x["media_id"] for x in refs if x["media_type"] == "image"],
                retrieval_video_ids=[x["media_id"] for x in refs if x["media_type"] == "video"],
                answer_evidence_media_refs=_normalize_media_refs(row, "answer_evidence") or refs,
                execution_status="scheduled", answer="", runtime_turns=[], model_call_metrics=[],
                tool_trace=[], execution_trace=[], judge_status="pending",
                judge={"score": None, "reason": "pending_judge"}, task_judges=[],
                evidence_judge={"score": None, "reason": "disabled"})
    add_media_metrics(item, {}, {})
    return item


def normalize_calls(calls, turn_index):
    result = []
    for index, original in enumerate(calls or []):
        row = dict(original)
        row.setdefault("call_id", f"turn-{turn_index}-call-{index}")
        row["conversation_turn"] = turn_index
        row.setdefault("call_type", row.get("role") or "agent")
        row.setdefault("total_ms", row.get("duration_ms"))
        if row.get("call_type") in {"view", "view_image", "visual", "image_review"}:
            row["original_call_type"] = row["call_type"]
            row["call_type"] = "tool_internal"
        elif row.get("call_type") == "final":
            row["call_type"] = "final_review"
        result.append(row)
    return result


def normalize_tools(traces, turn_index, call_offset=0):
    result = []
    for original in traces or []:
        row = dict(original)
        row.setdefault("tool", row.get("name"))
        row["conversation_turn"] = turn_index
        if not row.get("status"):
            observation = row.get("result") or {}
            row["status"] = "error" if isinstance(observation, dict) and observation.get("error") else "ok"
        if row.get("duration_ms") is not None:
            row.setdefault("latency_s", row["duration_ms"] / 1000)
        if isinstance(row.get("model_call_index"), int):
            row["model_call_index"] += call_offset
        result.append(row)
    return result


def native_stability(item):
    calls = [row for row in item.get("model_call_metrics", []) if row.get("call_type") in {"agent", "recovery"}]
    parsed = [row for row in calls if row.get("parse_status") is not None]
    stop = str(item.get("termination_reason") or "")
    complete = bool(item.get("answer")) and item.get("turn_outcome") in {"answer", "clarify", "refuse"} and \
        not any(marker in stop for marker in ("timeout", "budget", "max_steps", "error", "request_limit"))
    return {"json_parse_total": None, "json_parse_success": None, "json_parse_rate": None,
            "native_protocol_valid_count": sum(row["parse_status"] in {"success", "valid"} for row in parsed),
            "native_protocol_count": len(parsed), "completed_within_steps": complete,
            "final_turn_outcome": item.get("turn_outcome"), "protocol": "openai_native_tool_calls"}


def summary(state):
    if state.get("system_id") == "sentrix":
        result = OrchestratorRepository._effective_summary(copy.deepcopy(state))
    else:
        items = state.get("items") or []
        result = BenchmarkRun._capability_summary(items, state.get("phases") or {})
        result.update(total=state.get("qa_count", len(items)),
                      completed=sum(row.get("judge_status") in {"completed", "failed", "skipped"} for row in items),
                      benchmark_e2e_latency_excluding_judge_ms=BenchmarkRun._benchmark_e2e_latency_excluding_judge_ms(
                          state.get("phases") or {}, items),
                      tool_performance=OrchestratorRepository._aggregate_tool_performance(items))
        main_calls = [call for row in items for call in row.get("model_call_metrics", [])
                      if call.get("call_type") in {"agent", "recovery", "final_review"}]
        def values(key):
            return [c[key] for c in main_calls if isinstance(c.get(key), (int, float))]
        for key, out in [("ttft_ms", "llm_ttft_ms_mean"), ("tokens_per_second", "llm_tokens_per_second_mean")]:
            samples = values(key)
            result[out] = sum(samples) / len(samples) if samples else None
        for key, out in [("prompt_tokens", "prompt_tokens_total"), ("completion_tokens", "completion_tokens_total")]:
            samples = values(key)
            result[out] = sum(samples) if samples else None
        result["model_call_count"] = len(main_calls)
        prompts, completions = values("prompt_tokens"), values("completion_tokens")
        contexts = [call["prompt_tokens"] + call["completion_tokens"] for call in main_calls
                    if isinstance(call.get("prompt_tokens"), (int, float)) and isinstance(call.get("completion_tokens"), (int, float))]
        for name, samples in [("prompt_tokens", prompts), ("completion_tokens", completions), ("context_tokens", contexts)]:
            result["llm_" + name + "_max"] = max(samples) if samples else None
            result["llm_" + name + "_p95"] = nearest_rank_percentile(samples, .95)
        result["llm_context_samples_count"] = len(contexts)
        role_groups = {}
        for row in items:
            for call in row.get("model_call_metrics", []):
                group = role_groups.setdefault(call.get("original_call_type") or call.get("call_type") or "unknown",
                    {"calls": 0, "durations_ms": [], "prompt_tokens": 0, "completion_tokens": 0, "estimated_cny": 0})
                group["calls"] += 1
                if isinstance(call.get("duration_ms"), (int, float)):
                    group["durations_ms"].append(call["duration_ms"])
                for field in ("prompt_tokens", "completion_tokens", "estimated_cny"):
                    if isinstance(call.get(field), (int, float)):
                        group[field] += call[field]
        for group in role_groups.values():
            durations = group.pop("durations_ms")
            group.update(p50_ms=nearest_rank_percentile(durations, .5), p95_ms=nearest_rank_percentile(durations, .95),
                         total_ms=sum(durations) if durations else None)
        result["model_role_performance"] = role_groups
        parse_counts = [row.get("agent_stability") or {} for row in items]
        result["native_protocol_count"] = sum(r.get("native_protocol_count", 0) for r in parse_counts)
        result["native_protocol_valid_count"] = sum(r.get("native_protocol_valid_count", 0) for r in parse_counts)
        result["native_protocol_success_rate"] = (result["native_protocol_valid_count"] / result["native_protocol_count"]
                                                  if result["native_protocol_count"] else None)
    eligible = [r for r in state.get("items", []) if r.get("answerability") != "unanswerable"]
    result["candidate_retrieval_macro"] = _macro_metrics_from_counts(eligible, "candidate_retrieval_counts")
    result["planned_count"] = state.get("qa_count", len(state.get("items", [])))
    result["execution_error_count"] = sum(r.get("execution_status") in {"error", "failed"} for r in state.get("items", []))
    result["judge_invalid_count"] = sum((r.get("judge") or {}).get("score") not in {0, 1, 2} for r in state.get("items", []))
    result["mapping_error_count"] = sum(r.get("mapping_error_count", 0) for r in state.get("items", []))
    result["executed_turn_count"] = sum(len(r.get("runtime_turns") or []) for r in state.get("items", []))
    # Keep frozen labeled/valid counts tied to observed judgments, and expose
    # the full planned denominator separately (111 for the formal protocol).
    result["task_decision_planned_count"] = state.get("planned_turn_count", result.get("task_decision_labeled_count", 0))
    result["task_decision_unjudged_count"] = max(0, result["task_decision_planned_count"] - result.get("task_decision_valid_count", 0))
    result["not_started_count"] = max(0, result["planned_count"] - len(state.get("items", [])))
    result["unexecuted_turn_count"] = max(0, state.get("planned_turn_count", result["executed_turn_count"]) - result["executed_turn_count"])
    candidates = [r["candidate_retrieval_counts"] for r in eligible
                  if isinstance(r.get("candidate_retrieval_counts"), dict) and r["candidate_retrieval_counts"].get("gt", 0) > 0]
    result["candidate_coverage_count"] = len(candidates)
    result["candidate_any_hit_rate"] = sum(r.get("matched", 0) > 0 for r in candidates) / len(candidates) if candidates else None
    result["candidate_all_hit_rate"] = sum(r.get("matched", 0) >= r["gt"] for r in candidates) / len(candidates) if candidates else None
    items = state.get("items") or []
    latencies = [(r.get("timing_breakdown") or {}).get("agent_wall_ms") for r in items]
    latencies = [v for v in latencies if isinstance(v, (int, float))]
    result["agent_task_latency_p50_ms"] = nearest_rank_percentile(latencies, .5)
    result["agent_task_latency_p95_ms"] = nearest_rank_percentile(latencies, .95)
    result["agent_task_latency_sample_count"] = len(latencies)
    # Serialized distribution keys may be integers in fresh calculations.
    dist = result.get("judge_distribution", {})
    result["quality_correct_planned_rate"] = (dist.get("2", dist.get(2, 0)) / result["planned_count"]
        if result["planned_count"] else None)
    if state.get("system_id") == "sentrix":
        if not any(r.get("execution_status") is not None for r in items):
            result["execution_error_count"] = None
        result["task_decision_planned_count"] = 111 if state.get("qa_set") == "100qa-full" and len(items) == 100 else result["task_decision_planned_count"]
        result["task_decision_unjudged_count"] = max(0, result["task_decision_planned_count"] - result.get("task_decision_valid_count", 0))
    if state.get("mode") in {"reuse", "repair"}:
        result["benchmark_e2e_latency_excluding_judge_ms"] = None
    if state.get("mode") == "composite":
        original = state.get("original_timing_summary") or {}
        for key, value in original.items():
            if any(word in key for word in ("latency", "tokens", "ttft", "per_second")) or key in {
                "tool_performance", "model_role_performance", "model_call_count", "agent_task_latency_sample_count"}:
                result[key] = copy.deepcopy(value)
        result["performance_source"] = state["parent_run_id"]
    return result
