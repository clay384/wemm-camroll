import base64
import copy
import io
import json
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace
import urllib.error

from PIL import Image
import pytest

from camroll_agent.budget import BudgetLedger, RequestJournal, RequestUncertain
from camroll_agent.evidence_gate import CONTEXT_EVIDENCE_ID, EvidenceGateVerifier, MODEL, SYSTEM_PROMPT
from camroll_agent.qwen import DeadlineExceeded, ProviderError, QwenClient, RequestLimitExceeded, RequestScope


def verdict(status="supported", evidence="metadata:date"):
    ids = [evidence] if evidence else []
    return {"status": status, "evidence_ids": ids, "reasons": [
        {"claim": "The proposed date", "assessment": status, "evidence_ids": ids,
         "reason": "The supplied date supports this claim." if status == "supported" else "The supplied evidence does not support it."}]}


def response(value=None, **changes):
    result = {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(verdict() if value is None else value)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    result.update(changes)
    return result


def inputs(**changes):
    result = dict(question="When was this taken?", candidate_answer="On 2026-09-30.", outcome="answer",
        images=[], trusted_metadata=[{"evidence_id": "metadata:date", "source": "EXIF",
            "fields": {"date": "2026-09-30"}}], derived_observations=[],
        request_id="run:question:0:action:1", deadline=None)
    result.update(changes)
    return result


class Transport:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, payload, *, timeout):
        self.calls.append((copy.deepcopy(payload), timeout))
        if not self.results:
            pytest.fail("An unexpected cloud request was attempted")
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def gate(tmp_path, results=None, **kwargs):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", 600, expected_limit_cny=600)
    journal = RequestJournal(tmp_path / "new-run-journal.sqlite")
    counts = {"requests": 0}
    def hook():
        counts["requests"] += 1
    client = SimpleNamespace(ledger=ledger, journal=journal, request_namespace="new-run",
                             _gate_request_budget_hook=hook)
    transport = kwargs.pop("transport", Transport(results or []))
    verifier = EvidenceGateVerifier(client=client, transport=transport, sleep=lambda seconds: None, **kwargs)
    return verifier, transport, counts


def journal_rows(verifier):
    with sqlite3.connect(verifier.journal.path) as db:
        return db.execute("SELECT request_id,call_id,status,response,error_code FROM cloud_requests ORDER BY created").fetchall()


def test_independent_body_and_shared_accounting_with_durable_verdict_event(tmp_path):
    events = []
    verifier, transport, count = gate(tmp_path, [response()], on_call=events.append)
    result = verifier.verify(**inputs())
    payload, timeout = transport.calls[0]
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0 and payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 2048 and payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT
    data = json.loads(payload["messages"][1]["content"][0]["text"])
    assert set(data) == {"question", "candidate_answer", "candidate_outcome", "raw_images_in_order",
                         "actual_delivery_image_ids", "remaining_reviewed_context_image_ids",
                         "trusted_metadata", "derived_observations_not_independent_facts", "question_and_evidence_availability"}
    assert data["actual_delivery_image_ids"] == data["remaining_reviewed_context_image_ids"] == []
    assert result["status"] == "supported" and result["attempts"] == 1 and count["requests"] == 1
    assert timeout == 180 and events == result["model_calls"]
    assert events[0]["parse_status"] == "valid" and events[0]["verdict_status"] == "supported"
    assert events[0]["api_sent"] is True and events[0]["replayed"] is False
    rows = journal_rows(verifier)
    assert len(rows) == 1 and rows[0][0].startswith("new-run:evidence_gate:") and rows[0][2] == "completed"
    with sqlite3.connect(verifier.ledger.path) as db:
        status, charged, meta = db.execute("SELECT status,charged_units,metadata FROM budget_calls").fetchone()
    assert status == "settled" and charged == 288
    assert json.loads(meta)["role"] == "evidence_gate" and json.loads(meta)["request_id"] == rows[0][0]
    assert verifier.ledger.summary()["reserved_cny"] == 0


def delivery_images(tmp_path):
    rows = []
    for color in ("red", "blue", "green"):
        path = tmp_path / (color + ".jpg")
        Image.new("RGB", (24, 24), color).save(path)
        rows.append({"image_id": "img_" + color, "path": path})
    return rows


@pytest.mark.parametrize("delivery,context", [
    (["img_blue"], ["img_red", "img_green"]),
    (["img_green", "img_red"], ["img_blue"]),
])
def test_delivery_is_separate_from_ordered_reviewed_context_and_wrong_delivery_stays_negative(tmp_path, delivery, context):
    images = delivery_images(tmp_path)
    negative = verdict("contradicted", "image:" + delivery[0])
    negative["reasons"][0].update(claim="actual delivery", reason="The delivered image is not the requested red image.")
    verifier, transport, counts = gate(tmp_path, [response(negative)])
    result = verifier.verify(**inputs(question="Deliver only the red image.", candidate_answer="Here is the red image.",
        images=images, trusted_metadata=[], delivery_image_ids=delivery))
    data = json.loads(transport.calls[0][0]["messages"][1]["content"][0]["text"])
    assert [row["image_id"] for row in data["raw_images_in_order"]] == ["img_red", "img_blue", "img_green"]
    assert data["actual_delivery_image_ids"] == delivery
    assert data["remaining_reviewed_context_image_ids"] == context
    assert result["status"] == "contradicted" and counts["requests"] == 1
    assert "A matching object in a context image does not excuse delivering a different, incorrect image." in SYSTEM_PROMPT


@pytest.mark.parametrize("delivery", ["img_red", ["img_red", "img_red"], ["img_missing"],
                                      ["image:img_red"], [True], [{}], ["bad id"]])
def test_invalid_delivery_is_rejected_before_reservation_or_transport(tmp_path, delivery):
    verifier, transport, counts = gate(tmp_path)
    with pytest.raises(ValueError, match="delivery image identifiers"):
        verifier.verify(**inputs(images=delivery_images(tmp_path), delivery_image_ids=delivery))
    assert not transport.calls and counts["requests"] == 0
    assert journal_rows(verifier) == [] and verifier.ledger.summary()["requests"] == 0


@pytest.mark.parametrize("outcome", ["clarify", "refuse"])
def test_non_answer_cannot_deliver_a_reviewed_image(tmp_path, outcome):
    verifier, transport, counts = gate(tmp_path)
    with pytest.raises(ValueError, match="cannot deliver images"):
        verifier.verify(**inputs(outcome=outcome, images=delivery_images(tmp_path), delivery_image_ids=["img_red"]))
    assert not transport.calls and counts["requests"] == 0 and journal_rows(verifier) == []


def test_default_empty_delivery_keeps_direct_multimodal_smoke_compatible(tmp_path):
    verifier, transport, counts = gate(tmp_path, [response(verdict(evidence="image:img_red"))])
    images = delivery_images(tmp_path)
    verifier.verify(**inputs(images=images, trusted_metadata=[]))
    result = verifier.verify(**inputs(images=images, trusted_metadata=[], delivery_image_ids=[]))
    data = json.loads(transport.calls[0][0]["messages"][1]["content"][0]["text"])
    assert data["actual_delivery_image_ids"] == []
    assert data["remaining_reviewed_context_image_ids"] == ["img_red", "img_blue", "img_green"]
    assert result["status"] == "supported" and result["model_calls"][0]["replayed"] is True
    assert counts["requests"] == len(transport.calls) == 1


def test_delivery_change_cannot_replay_a_journaled_verdict_for_another_delivery(tmp_path):
    verifier, transport, counts = gate(tmp_path, [response(verdict(evidence="image:img_red"))])
    images = delivery_images(tmp_path)
    verifier.verify(**inputs(images=images, trusted_metadata=[], delivery_image_ids=["img_red"]))
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs(images=images, trusted_metadata=[], delivery_image_ids=["img_blue"]))
    assert caught.value.code == "request_identity_mismatch"
    assert counts["requests"] == len(transport.calls) == 1


@pytest.mark.parametrize("outcome", ["clarify", "refuse"])
def test_context_can_support_nonfactual_refusal_or_clarification_without_images(tmp_path, outcome):
    value = verdict(evidence=CONTEXT_EVIDENCE_ID)
    value["reasons"][0]["claim"] = "candidate_outcome"
    verifier, transport, count = gate(tmp_path, [response(value)])
    result = verifier.verify(**inputs(candidate_answer="I need more evidence to answer that.",
        outcome=outcome, trusted_metadata=[]))
    assert result["status"] == "supported" and len(transport.calls) == count["requests"] == 1


@pytest.mark.parametrize("outcome,claim", [("answer", "candidate_outcome"), ("refuse", "The proposed date")])
def test_context_cannot_support_an_answer_or_an_album_fact(tmp_path, outcome, claim):
    value = verdict(evidence=CONTEXT_EVIDENCE_ID)
    value["reasons"][0]["claim"] = claim
    verifier, _, count = gate(tmp_path, [response(value), response(value)])
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs(outcome=outcome, trusted_metadata=[]))
    assert caught.value.code == "gate_invalid_verdict" and count["requests"] == 2


def test_shared_scope_accounts_tokens_cost_and_desensitized_call_once(tmp_path):
    verifier, transport, _ = gate(tmp_path, [response(usage={"prompt_tokens": 100, "completion_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": 0}})])
    scope = RequestScope(50, 600)
    verifier._scope = lambda: scope
    result = verifier.verify(**inputs(request_budget_hook=scope.start_request))
    assert scope.summary()["requests"] == 1
    assert scope.summary()["prompt_tokens"] == 100 and scope.summary()["completion_tokens"] == 10
    assert scope.summary()["estimated_cny"] == pytest.approx(.000288)
    assert len(scope.calls) == 1 and scope.calls[0]["parse_status"] == "valid"
    event = result["model_calls"][0]
    assert event["input"] == transport.calls[0][0]
    assert event["output"][0]["message"]["content"] == json.dumps(verdict())
    assert event["finish_reasons"] == ["stop"] and event["reasoning_tokens"] == 0
    replay = verifier.verify(**inputs(request_budget_hook=scope.start_request))
    assert replay["model_calls"][0]["replayed"] and scope.summary()["requests"] == 1
    assert scope.summary()["prompt_tokens"] == 100 and scope.summary()["estimated_cny"] == pytest.approx(.000288)


@pytest.mark.parametrize("status,evidence", [("contradicted", "metadata:date"), ("insufficient", None)])
def test_valid_negative_verdict_is_final_not_repaired(tmp_path, status, evidence):
    verifier, transport, count = gate(tmp_path, [response(verdict(status, evidence))])
    result = verifier.verify(**inputs())
    assert result["status"] == status and result["attempts"] == 1
    assert len(transport.calls) == count["requests"] == 1


def test_completed_request_replays_without_new_call_or_budget_hook(tmp_path):
    verifier, _, _ = gate(tmp_path, [response()])
    first = verifier.verify(**inputs())
    restarted, transport, count = gate(tmp_path)
    second = restarted.verify(**inputs())
    assert not transport.calls and count["requests"] == 0
    assert second["model_calls"][0]["call_id"] == first["model_calls"][0]["call_id"]
    assert second["model_calls"][0]["replayed"] and not second["model_calls"][0]["api_sent"]
    assert second["model_calls"][0]["estimated_cny"] == 0
    assert restarted.ledger.summary()["requests"] == 1


def test_changed_candidate_cannot_reuse_same_identity(tmp_path):
    verifier, transport, _ = gate(tmp_path, [response()])
    verifier.verify(**inputs())
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs(candidate_answer="A different answer."))
    assert caught.value.code == "request_identity_mismatch" and len(transport.calls) == 1


def test_one_format_repair_is_paid_separately_and_replayable(tmp_path):
    invalid = response()
    invalid["choices"][0]["message"]["content"] = "not JSON"
    verifier, transport, count = gate(tmp_path, [invalid, response()])
    result = verifier.verify(**inputs())
    assert result["attempts"] == count["requests"] == 2
    assert [event["parse_status"] for event in result["model_calls"]] == ["invalid", "valid"]
    assert len({event["request_id"] for event in result["model_calls"]}) == 2
    assert len({event["payload_hash"] for event in result["model_calls"]}) == 2
    assert "Preserve your substantive assessment" in transport.calls[1][0]["messages"][-1]["content"]
    assert verifier.ledger.summary()["estimated_spent_cny"] == pytest.approx(.000576)
    restarted, replay_transport, replay_count = gate(tmp_path)
    replay = restarted.verify(**inputs())
    assert replay["status"] == "supported" and replay["attempts"] == 2
    assert all(event["replayed"] for event in replay["model_calls"])
    assert not replay_transport.calls and replay_count["requests"] == 0


def two_image_outcome_case(tmp_path):
    images = []
    for color in ("red", "blue"):
        path = tmp_path / (color + ".png")
        Image.new("RGB", (8, 8), color).save(path)
        images.append({"image_id": "img_" + color, "path": path})
    value = {"status": "supported", "evidence_ids": ["image:img_red", "image:img_blue", CONTEXT_EVIDENCE_ID],
             "reasons": [
                 {"claim": "The first image is red", "assessment": "supported", "evidence_ids": ["image:img_red"], "reason": "The supplied original is red."},
                 {"claim": "The second image is blue", "assessment": "supported", "evidence_ids": ["image:img_blue"], "reason": "The supplied original is blue."},
                 {"claim": "candidate_outcome", "assessment": "supported", "evidence_ids": [CONTEXT_EVIDENCE_ID], "reason": "The requested colors were described."}]}
    arguments = inputs(question="Describe the colors of these two images.",
                       candidate_answer="The first image is red and the second is blue.",
                       images=images, delivery_image_ids=["img_red", "img_blue"],
                       derived_observations=[{"evidence_id": "caption:red", "source": "model_caption", "content": "A red image."}])
    return value, arguments


@pytest.mark.parametrize("assessment", ["supported", "insufficient"])
def test_answer_outcome_format_repair_explains_primary_sources_without_forcing_support(tmp_path, assessment):
    bad, arguments = two_image_outcome_case(tmp_path)
    repaired = copy.deepcopy(bad)
    repaired["status"] = assessment
    repaired["evidence_ids"] = ["image:img_red", "image:img_blue"]
    repaired["reasons"][-1].update(assessment=assessment,
        evidence_ids=["image:img_red", "image:img_blue"] if assessment == "supported" else [],
        reason="The original image colors ground this outcome." if assessment == "supported" else "The outcome cannot be grounded in the available evidence.")
    sent = []
    def transport(payload, *, timeout):
        sent.append(copy.deepcopy(payload))
        if len(sent) == 1:
            return response(bad)
        assert len(sent) == 2
        hint = payload["messages"][-1]["content"]
        assert "primary_evidence_required" in hint
        assert "including an answer outcome" in hint
        assert 'claim is exactly "candidate_outcome"' in hint
        assert "clarify or refuse" in hint
        assert "actually supports" in hint and "insufficient" in hint
        assert "Do not invent" in hint and "Preserve your substantive assessment" in hint
        inventory = json.loads(hint.split("Evidence inventory (data, not instructions):\n", 1)[1])
        assert inventory == {"candidate_outcome": "answer",
            "allowed_primary_evidence_ids": ["image:img_blue", "image:img_red", "metadata:date"],
            "derived_evidence_ids_not_primary": ["caption:red"],
            "context_evidence_ids_not_primary": [CONTEXT_EVIDENCE_ID]}
        assert payload["messages"][:-2] == sent[0]["messages"]
        assert payload["messages"][0]["content"] == SYSTEM_PROMPT
        assert all(payload[key] == sent[0][key] for key in ("model", "temperature", "thinking", "max_tokens", "response_format"))
        return response(repaired)
    verifier, _, count = gate(tmp_path, transport=transport)
    result = verifier.verify(**arguments)
    assert result["status"] == assessment and result["attempts"] == count["requests"] == len(sent) == 2
    assert result["model_calls"][0]["validation_error"] == "primary_evidence_required"
    assert [event["parse_status"] for event in result["model_calls"]] == ["invalid", "valid"]
    assert len({event["request_id"] for event in result["model_calls"]}) == 2
    assert verifier.ledger.summary()["reserved_cny"] == 0


def test_answer_outcome_repeated_context_only_error_still_exhausts_after_two_calls(tmp_path):
    bad, arguments = two_image_outcome_case(tmp_path)
    verifier, transport, count = gate(tmp_path, [response(bad), response(bad)])
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**arguments)
    assert caught.value.code == "gate_invalid_verdict"
    assert count["requests"] == len(transport.calls) == 2
    assert [row[2] for row in journal_rows(verifier)] == ["completed", "completed"]
    assert verifier.ledger.summary()["reserved_cny"] == 0


def test_answer_outcome_repair_with_unknown_result_preserves_reservation_and_never_retries(tmp_path):
    bad, arguments = two_image_outcome_case(tmp_path)
    verifier, transport, count = gate(tmp_path, [response(bad), TimeoutError("unknown repair response")])
    with pytest.raises(RequestUncertain):
        verifier.verify(**arguments)
    assert count["requests"] == len(transport.calls) == 2
    assert [row[2] for row in journal_rows(verifier)] == ["completed", "uncertain"]
    assert verifier.ledger.summary()["uncertain_requests"] == 1
    assert verifier.ledger.summary()["reserved_cny"] > 0
    restarted, new_transport, new_count = gate(tmp_path)
    for request_id in (arguments["request_id"], "another-candidate-request"):
        with pytest.raises(RequestUncertain):
            restarted.verify(**{**arguments, "request_id": request_id})
    assert not new_transport.calls and new_count["requests"] == 0


@pytest.mark.parametrize("value", [
    {**verdict(), "score": 2},
    verdict(evidence="invented:image"),
    {**verdict(), "status": "insufficient"},
    {**verdict(), "status": []},
    {**verdict(), "reasons": [{**verdict()["reasons"][0], "assessment": {}}]},
    {**verdict(), "reasons": []},
])
def test_schema_or_evidence_failure_stops_after_one_repair(tmp_path, value):
    verifier, transport, count = gate(tmp_path, [response(value), response(value)])
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs())
    assert caught.value.code == "gate_invalid_verdict"
    assert len(transport.calls) == count["requests"] == 2
    assert [row[2] for row in journal_rows(verifier)] == ["completed", "completed"]
    assert verifier.ledger.summary()["reserved_cny"] == 0


def test_derived_caption_alone_cannot_support_fact(tmp_path):
    verifier, transport, _ = gate(tmp_path, [response(verdict(evidence="caption:1")), response(verdict("insufficient", "caption:1"))])
    result = verifier.verify(**inputs(trusted_metadata=[], derived_observations=[
        {"evidence_id": "caption:1", "source": "qwen_caption", "content": "A synthetic observation"}]))
    assert result["status"] == "insufficient" and result["attempts"] == 2
    assert result["model_calls"][0]["validation_error"] == "primary_evidence_required"
    assert "derived_observations_not_independent_facts" in transport.calls[0][0]["messages"][1]["content"][0]["text"]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_known_rejection_does_not_trigger_format_or_transport_retry(tmp_path, code):
    error = urllib.error.HTTPError("https://synthetic.invalid", code, "SECRET provider body", {}, None)
    verifier, transport, count = gate(tmp_path, [error])
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs())
    assert caught.value.code == f"http_{code}" and "SECRET" not in str(caught.value)
    assert len(transport.calls) == count["requests"] == 1
    assert journal_rows(verifier)[0][2] == "failed"
    assert verifier.ledger.summary()["estimated_spent_cny"] == verifier.ledger.summary()["reserved_cny"] == 0


def test_429_attempts_count_against_shared_hook_and_resume_skips_rejections(tmp_path):
    error = urllib.error.HTTPError("https://synthetic.invalid", 429, "rate limit", {}, None)
    verifier, transport, count = gate(tmp_path, [error, response()])
    result = verifier.verify(**inputs())
    assert result["attempts"] == 1 and count["requests"] == len(transport.calls) == 2
    assert [event["transport_attempt"] for event in result["model_calls"]] == [0, 1]
    assert [row[2] for row in journal_rows(verifier)] == ["failed", "completed"]
    restarted, replay_transport, replay_count = gate(tmp_path)
    replay = restarted.verify(**inputs())
    assert replay["model_calls"][0]["transport_attempt"] == 1
    assert not replay_transport.calls and replay_count["requests"] == 0


@pytest.mark.parametrize("error", [TimeoutError("SECRET timeout"),
    urllib.error.HTTPError("https://synthetic.invalid", 500, "SECRET body", {}, None)], ids=["timeout", "server-error"])
def test_unknown_stops_all_new_identities_and_retains_reservation(tmp_path, error):
    verifier, transport, count = gate(tmp_path, [error])
    with pytest.raises(RequestUncertain) as caught:
        verifier.verify(**inputs())
    assert "SECRET" not in str(caught.value)
    assert len(transport.calls) == count["requests"] == 1
    assert verifier.ledger.summary()["uncertain_requests"] == 1
    assert verifier.ledger.summary()["reserved_cny"] > 0
    assert journal_rows(verifier)[0][2] == "uncertain"
    restarted, new_transport, new_count = gate(tmp_path)
    for request_id in (inputs()["request_id"], "different-id"):
        with pytest.raises(RequestUncertain):
            restarted.verify(**inputs(request_id=request_id))
    assert not new_transport.calls and new_count["requests"] == 0


def test_usage_missing_remains_uncertain_even_with_valid_verdict(tmp_path):
    verifier, transport, count = gate(tmp_path, [response(usage={})])
    with pytest.raises(RequestUncertain):
        verifier.verify(**inputs())
    row = journal_rows(verifier)[0]
    assert row[2] == "uncertain" and json.loads(row[3])["choices"]
    assert row[4] == "usage_missing" and count["requests"] == len(transport.calls) == 1
    assert verifier.ledger.summary()["uncertain_requests"] == 1


def test_crash_after_saved_response_can_settle_without_resending(tmp_path, monkeypatch):
    verifier, transport, _ = gate(tmp_path, [response()])
    def crash(*args, **kwargs):
        raise RuntimeError("synthetic crash after journal response")
    monkeypatch.setattr(verifier.ledger, "finish", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        verifier.verify(**inputs())
    assert journal_rows(verifier)[0][2] == "completed" and verifier.ledger.summary()["reserved_cny"] > 0
    restarted, replay_transport, replay_count = gate(tmp_path)
    assert restarted.verify(**inputs())["status"] == "supported"
    assert not replay_transport.calls and replay_count["requests"] == 0
    assert restarted.ledger.summary()["reserved_cny"] == 0
    assert restarted.ledger.summary()["estimated_spent_cny"] == pytest.approx(.000288)
    assert len(transport.calls) == 1


def test_shared_request_cap_rejection_releases_unsent_reservation(tmp_path):
    events = []
    verifier, transport, _ = gate(tmp_path, on_call=events.append)
    def rejected():
        raise RequestLimitExceeded()
    with pytest.raises(RequestLimitExceeded):
        verifier.verify(**inputs(request_budget_hook=rejected))
    assert not transport.calls and verifier.ledger.summary()["reserved_cny"] == 0
    assert events[0]["api_sent"] is False and journal_rows(verifier)[0][4] == "gate_request_guard"


def test_expired_deadline_cannot_reserve_or_send(tmp_path):
    verifier, transport, count = gate(tmp_path, clock=lambda: 100)
    with pytest.raises(DeadlineExceeded):
        verifier.verify(**inputs(deadline=99))
    assert not transport.calls and count["requests"] == 0 and not journal_rows(verifier)


def test_late_response_is_charged_but_not_accepted_or_repaired(tmp_path):
    now = [100.0]
    def late(payload, *, timeout):
        assert timeout == 1
        now[0] = 102
        return response()
    verifier, _, count = gate(tmp_path, transport=late, clock=lambda: now[0])
    with pytest.raises(DeadlineExceeded):
        verifier.verify(**inputs(deadline=101))
    assert count["requests"] == 1 and verifier.ledger.summary()["reserved_cny"] == 0
    assert verifier.ledger.summary()["estimated_spent_cny"] > 0


@pytest.mark.parametrize("model", [None, "qwen3.8-flash", [MODEL]])
def test_missing_or_wrong_actual_model_never_defaults_to_requested_model(tmp_path, model):
    verifier, transport, count = gate(tmp_path, [response(model=model)])
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs())
    assert caught.value.code == "gate_response_model_mismatch"
    assert len(transport.calls) == count["requests"] == 1
    assert verifier.ledger.summary()["estimated_spent_cny"] > 0


def test_six_real_preprocessed_images_do_not_count_base64_as_context(tmp_path):
    data = random.Random(17).randbytes(512 * 512 * 3)
    path = tmp_path / "synthetic.png"
    Image.frombytes("RGB", (512, 512), data).save(path)
    images = [{"image_id": f"img_{index}", "path": path} for index in range(6)]
    verifier, transport, count = gate(tmp_path, [response(verdict(evidence="image:img_0"))])
    result = verifier.verify(**inputs(images=images, trusted_metadata=[]))
    body = transport.calls[0][0]
    urls = [part["image_url"]["url"] for part in body["messages"][1]["content"] if part["type"] == "image_url"]
    assert len(urls) == 6 and sum(len(url) for url in urls) > 262144
    for url in urls:
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as picture:
            assert picture.format == "JPEG" and max(picture.size) <= 2048
    assert result["status"] == "supported" and count["requests"] == 1
    assert "base64" not in json.dumps(result["model_calls"])
    with pytest.raises(ValueError, match="at most six"):
        verifier.verify(**inputs(images=images + [{"image_id": "extra", "path": path}]))


def test_no_reference_interface_or_untrusted_metadata_field(tmp_path):
    verifier, transport, count = gate(tmp_path)
    with pytest.raises(TypeError):
        verifier.verify(**inputs(), reference_answer="not allowed")
    for metadata in [
        {"evidence_id": "metadata:1", "source": "exif", "fields": {"reference_answer": "never accepted"}},
        {"evidence_id": "metadata:1", "source": "exif", "fields": {"caption": "derived is not trusted"}},
    ]:
        with pytest.raises(ValueError):
            verifier.verify(**inputs(trusted_metadata=[metadata]))
    with pytest.raises(ValueError, match="reference fields"):
        verifier.verify(**inputs(derived_observations=[{"evidence_id": "derived:1", "source": "model",
            "fields": {"nested": {"ground_truth": "not allowed"}}}]))
    assert not transport.calls and count["requests"] == 0


def test_provider_endpoint_and_private_key_file_match_existing_entry_points(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("synthetic-key")
    key.chmod(0o600)
    monkeypatch.setenv("CAMROLL_EVAL_JUDGE_API_KEY_FILE", str(key))
    monkeypatch.setenv("CAMROLL_EVAL_JUDGE_API_KEY", "wrong-key")
    monkeypatch.setenv("CAMROLL_EVAL_JUDGE_URL", "https://synthetic.invalid/api/plan/v3")
    sent = []
    class Reply:
        def __enter__(self):
            return io.BytesIO(json.dumps(response()).encode())
        def __exit__(self, *args):
            pass
    def urlopen(request, *, timeout):
        sent.append((request, timeout))
        return Reply()
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    ledger = BudgetLedger(tmp_path / "budget.sqlite", 600, expected_limit_cny=600)
    verifier = EvidenceGateVerifier(ledger=ledger, request_namespace="run", request_journal_path=tmp_path / "journal.sqlite")
    verifier.verify(**inputs(request_budget_hook=lambda: None))
    request, _ = sent[0]
    assert request.full_url == "https://synthetic.invalid/api/plan/v3/chat/completions"
    assert request.get_header("Authorization") == "Bearer synthetic-key"
    assert json.loads(request.data)["thinking"] == {"type": "disabled"}
    assert request.method == "POST"


def test_unapproved_budget_or_missing_namespace_is_rejected(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", 300)
    with pytest.raises(Exception, match="ceiling changed"):
        EvidenceGateVerifier(ledger=ledger, request_namespace="run", request_journal_path=tmp_path / "journal.sqlite", transport=lambda payload, timeout: response())
    with pytest.raises(ValueError, match="600"):
        EvidenceGateVerifier(expected_limit_cny=300)


def test_unknown_charge_is_visible_in_shared_scope_and_never_double_counts_requests(tmp_path):
    verifier, _, _ = gate(tmp_path, [TimeoutError("synthetic unknown")])
    scope = RequestScope(50, 600)
    verifier._scope = lambda: scope
    with pytest.raises(RequestUncertain):
        verifier.verify(**inputs(request_budget_hook=scope.start_request))
    assert scope.summary()["requests"] == 1 and len(scope.calls) == 1
    assert scope.summary()["reserved_cny"] == pytest.approx(verifier.ledger.summary()["reserved_cny"])
    assert scope.calls[0]["status"] == "uncertain" and scope.calls[0]["api_sent"] is True


def test_429_is_bounded_to_three_physical_attempts(tmp_path):
    errors = [urllib.error.HTTPError("https://synthetic.invalid", 429, "rate limit", {}, None) for _ in range(3)]
    verifier, transport, count = gate(tmp_path, errors)
    with pytest.raises(ProviderError) as caught:
        verifier.verify(**inputs())
    assert caught.value.code == "http_429" and count["requests"] == len(transport.calls) == 3
    assert verifier.ledger.summary()["reserved_cny"] == verifier.ledger.summary()["estimated_spent_cny"] == 0


def test_real_qwen_and_verifier_share_one_physical_request_ceiling(tmp_path):
    class SDK:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            self.calls.append(kwargs)
            pytest.fail("The verifier already consumed the shared request allowance")
    sdk = SDK()
    client = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite", 600),
        expected_limit_cny=600, request_namespace="real-client-run", request_journal_path=tmp_path / "journal.sqlite")
    transport = Transport([response()])
    verifier = EvidenceGateVerifier(client=client, transport=transport)
    with client.request_scope(max_requests=1, deadline_s=600) as scope:
        result = verifier.verify(**inputs(deadline=scope.deadline))
        with pytest.raises(RequestLimitExceeded):
            client.chat([{"role": "user", "content": "another request"}])
        assert scope.requests == 1 and len(scope.calls) == 1
        assert scope.summary()["estimated_cny"] == pytest.approx(.000288)
    assert result["status"] == "supported" and len(transport.calls) == 1 and not sdk.calls


def test_real_qwen_is_blocked_by_verifier_unknown_request_in_shared_ledger(tmp_path):
    class SDK:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            pytest.fail("An unknown verifier charge must block a different provider")
    client = QwenClient(sdk=SDK(), ledger=BudgetLedger(tmp_path / "budget.sqlite", 600),
        expected_limit_cny=600, request_namespace="real-client-run", request_journal_path=tmp_path / "journal.sqlite")
    verifier = EvidenceGateVerifier(client=client, transport=Transport([TimeoutError("synthetic")]))
    with client.request_scope(max_requests=50, deadline_s=600) as scope:
        with pytest.raises(RequestUncertain):
            verifier.verify(**inputs(deadline=scope.deadline))
        with pytest.raises(RequestUncertain):
            client.chat([{"role": "user", "content": "another provider request"}], request_id="other")
    assert client.ledger.summary()["requests"] == 1 and scope.requests == 1


@pytest.mark.parametrize("explicit_deadline", [None, 10000.0])
def test_active_question_deadline_also_bounds_omitted_or_later_verifier_deadline(tmp_path, explicit_deadline):
    now = [100.0]
    def transport(payload, *, timeout):
        assert timeout == 2.0
        now[0] = 103.0
        return response()
    verifier, _, _ = gate(tmp_path, transport=transport, clock=lambda: now[0])
    scope = RequestScope(50, 600)
    scope.deadline = 102.0
    verifier._scope = lambda: scope
    with pytest.raises(DeadlineExceeded):
        verifier.verify(**inputs(deadline=explicit_deadline))
    assert scope.summary()["estimated_cny"] == pytest.approx(.000288)
    assert scope.calls[0]["status"] == "late_response"
