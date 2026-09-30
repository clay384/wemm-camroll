import copy

import pytest

from camroll_eval.metrics import (add_media_metrics, initial_item, local_retrieval_performance,
                                 retrieval_diagnostics, summary)


MEDIA = {identifier: {"media_id": filename, "file_name": filename, "media_type": "image"}
         for identifier, filename in (("img_a", "a.jpg"), ("img_b", "b.jpg"))}


def question(identifier="qa-a", *, unanswerable=False):
    return initial_item({"qa_id": identifier, "question": "照片", "answer": "答案",
                         "retrieval_image_ids": ["a.jpg"],
                         "answerability": "unanswerable" if unanswerable else "answerable"})


def trace(candidates):
    return {"tool": "search", "result": {"retrieval_trace": {
        "images": {"candidate_count": len(candidates), "candidates": candidates}}}}


def test_formal_partial_results_keep_fixed_eighty_one_and_hundred_denominators():
    item = question()
    add_media_metrics(item, {"selected_image_ids": ["img_a", "img_b"],
                            "retrieved_image_ids": ["img_a"]}, MEDIA)
    item["judge"] = {"score": 2}
    item["judge_status"] = "completed"
    unanswerable = question("qa-unanswerable", unanswerable=True)
    unanswerable["judge"] = {"score": 1}
    add_media_metrics(unanswerable, {"selected_image_ids": ["img_a"]}, MEDIA)
    result = summary({"experiment_config": {"name": "Q37"}, "formal_protocol": True,
                      "system_id": "camroll_qwen", "qa_count": 100, "planned_turn_count": 111,
                      "items": [item, unanswerable], "media_mapping": MEDIA})
    assert result["retrieval_precision_macro"] == pytest.approx(.5 / 81)
    assert result["retrieval_recall_macro"] == pytest.approx(1 / 81)
    assert result["retrieval_f1_macro"] == pytest.approx((2 / 3) / 81)
    assert result["retrieval_metric_count"] == 81
    assert result["retrieval_observed_metric_count"] == 1
    assert result["final_delivery_macro"]["unobserved_count"] == 80
    assert result["quality_correct_planned_rate"] == .01
    assert result["quality_distribution_planned"] == {"correct": 1, "partial": 1, "incorrect": 0, "unscored": 98}
    assert result["retrieval_excluded_unanswerable_count"] == 19
    assert result["retrieval_diagnostics"]["pre_top10_recall_macro"] == 0


def test_formal_not_started_results_have_zero_metrics_and_explicit_unscored():
    result = summary({"experiment_config": {"name": "W37"}, "formal_protocol": True,
                      "qa_count": 100, "items": []})
    assert result["retrieval_recall_macro"] == result["retrieval_f1_macro"] == 0
    assert result["final_delivery_macro"]["unobserved_count"] == 81
    assert result["quality_distribution_planned"]["unscored"] == 100
    assert result["not_started_count"] == 100


def test_rerank_diagnostics_distinguish_candidate_pool_and_top_ten():
    item = question()
    item["tool_trace"] = [trace([
        {"id": "img_a", "recall_rank": 11, "final_rank": 1},
        {"id": "img_b", "recall_rank": 1, "final_rank": 11}])]
    state = {"items": [item, question("not-searched")], "media_mapping": MEDIA}
    result = retrieval_diagnostics(state, 81)
    assert result["pre_top10_recall_macro"] == 0
    assert result["post_top10_recall_macro"] == pytest.approx(1 / 81)
    assert result["candidate_pool_recall_macro"] == pytest.approx(1 / 81)
    assert result["questions_with_image_search"] == 1
    assert result["per_question"][1]["post_top10_recall"] == 0
    assert result["scope"] == "per_question_union_of_image_branch_candidates_across_all_search_calls"


def test_rerank_diagnostics_union_calls_deduplicate_and_exclude_unanswerable():
    item = question()
    item["tool_trace"] = [trace([{"id": "img_a", "recall_rank": 11, "final_rank": 1}]),
                          trace([{"id": "img_a", "recall_rank": 1, "final_rank": 11},
                                 {"id": "unknown", "recall_rank": 1, "final_rank": 1}])]
    excluded = question("excluded", unanswerable=True)
    excluded["tool_trace"] = copy.deepcopy(item["tool_trace"])
    result = retrieval_diagnostics({"items": [item, excluded], "media_mapping": MEDIA})
    assert result["metric_count"] == 1
    assert result["pre_top10_recall_macro"] == result["post_top10_recall_macro"] == 1
    assert result["per_question"][0]["post_top10_count"] == 2
    assert result["per_question"][0]["post_top10_matched"] == 1


def test_local_retrieval_latency_separates_encoding_reranking_and_setup():
    embedding = {"call_id": "encode-one", "call_type": "embedding", "operation": "/encode",
                 "duration_ms": 100, "status": "success"}
    rerank = {"call_id": "rerank-one", "call_type": "rerank", "operation": "/rerank",
              "duration_ms": 300, "status": "error"}
    setup = {"call_id": "fingerprints", "call_type": "embedding", "operation": "/fingerprints",
             "duration_ms": 10, "status": "success"}
    item = question()
    item["tool_trace"] = [{"embedding_calls": [embedding, rerank, setup]},
                          {"embedding_calls": [copy.deepcopy(embedding)]}]
    result = local_retrieval_performance([item])
    assert result["embedding"]["calls"] == 1
    assert result["embedding"]["mean_ms"] == result["embedding"]["p95_ms"] == 100
    assert result["rerank"]["errors"] == 1
    assert result["rerank"]["total_ms"] == 300
    assert result["retrieval_setup"]["total_ms"] == 10
