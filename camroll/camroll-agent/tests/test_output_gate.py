"""Real Library/Agent/Qwen transport regressions; only synthetic images and SDK."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from camroll_agent.agent import Agent
from camroll_agent.budget import BudgetLedger
from camroll_agent.experiment import get_experiment_profile
from camroll_agent.library import Library
from camroll_agent.qwen import QwenClient
from camroll_agent.retrieval import experiment_metadata


def response(content=None, calls=None):
    return {"model": "qwen3.7-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content,
                          **({"tool_calls": calls} if calls else {})}}]}


def call(name, args, ident="c1"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class SDK:
    def __init__(self, outcomes):
        self.outcomes, self.calls = iter(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value


class Verifier:
    def __init__(self, statuses=("supported",)):
        self.statuses, self.calls = iter(statuses), []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": next(self.statuses), "reasons": [], "evidence_ids": ["img_a"]}


@pytest.fixture
def album(tmp_path):
    memory = tmp_path / "album"
    (memory / "assets").mkdir(parents=True)
    Image.new("RGB", (12, 12), "red").save(memory / "assets" / "a.jpg")
    image = {"id": "img_a", "kind": "image", "path": str(memory / "assets" / "a.jpg"),
             "caption": "red square", "visible_text": "", "event_id": None}
    profile = get_experiment_profile("W38Max").with_fingerprints(embedding_fingerprint="synthetic")
    vector = np.zeros((1, 4096), dtype=np.float32)
    vector[:, 0] = 1
    metadata = {**experiment_metadata(profile, "old-build"), "embedding_fingerprint": "synthetic",
                "vector_hashes": {"image": hashlib.sha256(vector.tobytes()).hexdigest()}}
    snapshot = memory / "snapshots" / ("a" * 32)
    snapshot.mkdir(parents=True)
    (snapshot / "records.json").write_text(json.dumps({"images": [image], "events": [], "metadata": metadata,
                                                     "vector_ids": {"image": ["img_a"]}}))
    np.savez(snapshot / "vectors.npz", image=vector)
    (memory / "CURRENT").write_text("a" * 32)
    class Embedding:
        provider, dimension, fingerprint = "wemm", 4096, "synthetic"
        def encode(self, **kwargs):
            return vector[0].copy()
    return memory, Library(memory, embedding_client=Embedding(), config="M4")


def agent_for(tmp_path, album, outcomes, verifier=None, policy="repair-evidence-v1"):
    sdk = SDK(outcomes)
    client = QwenClient(sdk=sdk, model="qwen3.7-flash", ledger=BudgetLedger(tmp_path / "budget.sqlite", 600),
                        expected_limit_cny=600, request_namespace="synthetic-gate-run")
    agent = Agent(str(album[0]), library=album[1], llm=client, inference_model="qwen3.7-flash",
                  gate_policy=policy, evidence_verifier=verifier or Verifier())
    return agent, client, sdk


def test_real_chain_repeats_same_payload_once_then_repairs_without_repeating_search(tmp_path, album):
    outcomes = [response(calls=[call("search", {"query": "red"})]),
                response(calls=[call("view_image", {"image_ids": ["img_a"], "prompt": "color"})]),
                response("bad JSON"), response("bad JSON"),
                response(json.dumps({"analysis": "red", "evidence": [{"image_id": "img_a", "observation": "red"}], "limitations": ""})),
                response(json.dumps({"answer": "red", "outcome": "answer", "selected_image_ids": ["img_a"]})),
                response(json.dumps({"answer": "red", "outcome": "answer", "evidence_image_ids": ["img_a"]}))]
    verifier = Verifier()
    agent, client, sdk = agent_for(tmp_path, album, outcomes, verifier)
    result = agent.ask("color?", structured_output=True, request_context={"qa_id": "001"})
    assert result.stopped_reason == "ok" and result.origin == "model"
    assert len(sdk.calls) == result.usage["requests"] == 7
    assert {k:v for k,v in sdk.calls[2].items() if k != "timeout"} == {k:v for k,v in sdk.calls[3].items() if k != "timeout"}
    assert sdk.calls[4]["messages"] != sdk.calls[3]["messages"]
    repeat = next(event for event in result.gate_events if event.get("role") == "view" and event.get("attempt") == 2 and event["state"] == "rejected")
    initial = next(event for event in result.gate_events if event.get("role") == "view" and event.get("attempt") == 1 and event["state"] == "rejected")
    assert repeat["exact_payload_match"] is True and repeat["payload_hash"] == initial["payload_hash"]
    assert repeat["call_id"] != initial["call_id"] and repeat["replayed"] is False
    metrics = {metric["call_id"]: metric for metric in result.model_call_metrics}
    assert metrics[repeat["call_id"]]["semantic_request_id"] == repeat["semantic_request_id"]
    assert metrics[initial["call_id"]]["semantic_request_id"] == initial["semantic_request_id"]
    assert len([x for x in result.tool_trace if x["tool"] == "search"]) == 1
    assert len(verifier.calls) == 1
    assert verifier.calls[0]["delivery_image_ids"] == ["img_a"]
    assert result.gate_status == "accepted"
    assert album[1].experiment.generation_model == "qwen3.8-max"


def test_bad_batch_never_executes_valid_prefix_and_exhausts_exactly_four(tmp_path, album):
    bad = response(calls=[call("search", {"query": "red"}), call("view_image", {"image_ids": ["img_unknown"], "prompt": "x"}, "c2")])
    agent, _, sdk = agent_for(tmp_path, album, [bad] * 4)
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "gate_exhausted" and result.stopped_reason == "gate_exhausted"
    assert len(sdk.calls) == 4
    assert not result.tool_trace or not any(x["tool"] == "search" for x in result.tool_trace)


def test_search_image_event_reference_allows_get_of_the_returned_event(tmp_path, album):
    library = album[1]
    library.images["img_a"]["event_id"] = "ev_synthetic_returned"
    library.events["ev_synthetic_returned"] = {"id": "ev_synthetic_returned", "kind": "event",
        "name": "Synthetic red event", "description": "A synthetic red square.", "image_ids": ["img_a"]}
    outcomes = [response(calls=[call("search", {"query": "red", "kind": "images"})]),
        response(calls=[call("get", {"id": "ev_synthetic_returned"})]),
        response(json.dumps({"answer": "red", "outcome": "answer", "selected_image_ids": ["img_a"]})),
        response(json.dumps({"answer": "red", "outcome": "answer", "evidence_image_ids": ["img_a"]}))]
    agent, _, sdk = agent_for(tmp_path, album, outcomes)
    result = agent.ask("Describe the event for the red image.", structured_output=True)
    assert result.stopped_reason == "ok" and result.origin == "model"
    fetched = [row for row in result.tool_trace if row["tool"] == "get"]
    assert len(fetched) == 1 and fetched[0]["result"]["id"] == "ev_synthetic_returned"
    assert len(sdk.calls) == 4
    assert not any(row.get("error", {}).get("code") == "unknown_evidence_id" for row in result.gate_events)


@pytest.mark.parametrize("event_id", [None, "", [], {}, 0, True, "invalid", "ev_bad space", "ev_returned"])
def test_get_event_allowlist_uses_only_valid_explicit_returned_event_ids(tmp_path, album, event_id):
    from camroll_agent.gate import GateViolation
    album[1].events["ev_unreturned"] = {"id": "ev_unreturned", "kind": "event", "image_ids": ["img_a"]}
    records = copy.deepcopy(album[1].images)
    records["img_a"].update(event_id=event_id, caption="Text mention ev_unreturned is not an explicit event reference.")
    agent, _, sdk = agent_for(tmp_path, album, [])
    agent._gate_batch([call("get", {"id": "img_a"})], records)
    with pytest.raises(GateViolation) as caught:
        agent._gate_batch([call("get", {"id": "ev_unreturned"})], records)
    assert caught.value.detail["code"] == "unknown_evidence_id"
    assert caught.value.detail["allowed"] == (["ev_returned", "img_a"] if event_id == "ev_returned" else ["img_a"])
    assert not sdk.calls


def test_known_id_missing_asset_is_implementation_failure_without_retry(tmp_path, album):
    outcomes = [response(calls=[call("search", {"query": "red"})]),
                response(calls=[call("view_image", {"image_ids": ["img_a"], "prompt": "color"})])]
    agent, _, sdk = agent_for(tmp_path, album, outcomes)
    (album[0] / "assets" / "a.jpg").unlink()
    result = agent.ask("color?", structured_output=True)
    assert result.stopped_reason == "asset_binding_error"
    assert result.origin == "system_failure" and len(sdk.calls) == 2


def final_chain(*finals):
    return [response(calls=[call("search", {"query": "red"})]),
            response(json.dumps({"answer": "unverified draft", "outcome": "answer", "selected_image_ids": ["img_a"]})),
            *[response(json.dumps({"outcome": "answer", **final})) for final in finals]]


def test_verifier_input_fault_is_operational_not_a_candidate_repair(tmp_path, album):
    from camroll_agent.evidence_gate import EvidenceGateInputError
    from camroll_eval.gate_experiment import infrastructure_failure
    class InvalidInputVerifier:
        calls = 0
        def verify(self, **kwargs):
            self.calls += 1
            raise EvidenceGateInputError("Synthetic evidence assembly is invalid")
    verifier = InvalidInputVerifier()
    agent, client, sdk = agent_for(tmp_path, album,
        final_chain({"answer": "red", "evidence_image_ids": ["img_a"]}), verifier)
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "system_failure" and "unverified draft" not in result.final_text
    assert result.selected_image_ids == [] and verifier.calls == 1 and len(sdk.calls) == 3
    assert infrastructure_failure({"tool_trace": result.tool_trace, "gate_events": result.gate_events}) == "gate_input_error"
    assert result.gate_events[-1]["state"] == "blocked"
    assert not any(event.get("attempt", 0) > 1 for event in result.gate_events)
    assert client.usage_summary()["reserved_cny"] == 0


def test_fact_gate_separates_five_visual_observations_from_six_repeated_library_records(tmp_path, album):
    from camroll_agent.evidence_gate import EvidenceGateVerifier, SYSTEM_PROMPT
    from camroll_agent.gate import OutputGate
    from camroll_agent.prompts import public_evidence
    library = album[1]
    for index in range(6):
        iid = "img_" + chr(ord("a") + index)
        path = album[0] / "assets" / (iid + ".jpg")
        Image.new("RGB", (12, 12), "red").save(path)
        library.images[iid] = {"id": iid, "kind": "image", "path": str(path),
            "date": "2026-10-01", "date_source": "synthetic_EXIF",
            "caption": "caption " + str(index) + " 合成文字" * 100, "visible_text": "完整 OCR " + str(index),
            "face_instances": [{"face_id": "face_" + str(index), "bbox": [1, 2, 7, 8],
                "provenance": "登记资料" * 1000, "identity_status": "unknown"}]}
    records = copy.deepcopy(library.images)
    reviewed = list(records)
    trace, expected_observations = [], []
    for index in range(5):
        result = {"analysis": ("第 " + str(index) + ' 次完整观察：保留空格、"引号"和换行。\n') * 100,
            "evidence": [{"image_id": iid, "observation": "对应 " + iid + " 的全文观察。" * 30} for iid in reviewed],
            "limitations": "原图与扩展裁剪可能包含其他人物；不可按外观猜身份。" * 20,
            "image_ids": list(reviewed),
            "target_regions": [{"image_id": iid, "face_id": "face_" + str(i),
                "face_bbox": [1, 2, 7, 8], "crop_bbox": [0, 0, 12, 12],
                "coordinate_space": "synthetic_original", "limitation": "登记匹配，仅作为裁剪来源"}
                for i, iid in enumerate(reviewed)],
            "future_observation_extension": {"full_text": "未来扩展不得无声丢弃。" * 50},
            "images": public_evidence(list(records.values()))}
        trace.append({"tool": "view_image", "result": result})
        expected_observations.append({key: copy.deepcopy(value) for key, value in result.items() if key != "images"})
    original_trace = copy.deepcopy(trace)
    agent, client, sdk = agent_for(tmp_path, album, [])
    calls = []
    def transport(body, timeout):
        calls.append(copy.deepcopy(body))
        verdict = {"status": "supported", "evidence_ids": ["image:img_a"], "reasons": [
            {"claim": "red", "assessment": "supported", "evidence_ids": ["image:img_a"], "reason": "The raw image is red."}]}
        return {"model": "doubao-seed-2-0-lite-260215", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(verdict)}}]}
    agent.evidence_verifier = EvidenceGateVerifier(client=client, transport=transport)
    with client.request_scope() as scope:
        agent._gate_facts("Describe the synthetic colors.", {"answer": "red", "outcome": "answer", "evidence_image_ids": reviewed},
            records, reviewed, trace, "synthetic-fact-assembly", scope, OutputGate("synthetic-turn"))
    assert len(calls) == 1 and not sdk.calls
    assert calls[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert calls[0]["max_tokens"] == 2048 and calls[0]["temperature"] == 0
    body = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    trusted = {row["image_id"]: row for row in body["trusted_metadata"]}
    derived = body["derived_observations_not_independent_facts"]
    assert set(trusted) == set(reviewed)
    for iid in reviewed:
        assert trusted[iid]["fields"] == {"date": records[iid]["date"], "date_source": records[iid]["date_source"],
            "identity": public_evidence(records[iid]["face_instances"])}
        caption = next(row for row in derived if row["evidence_id"] == "derived:" + iid)
        assert caption["fields"] == {"caption": records[iid]["caption"], "visible_text": records[iid]["visible_text"]}
    observations = [row for row in derived if row["evidence_id"].startswith("observation:")]
    assert len(observations) == 5
    assert [json.loads(row["content"]) for row in observations] == expected_observations
    assert all(row["source"] == "prior_model_visual_observation_not_primary" for row in observations)
    assert trace == original_trace
    assert client.usage_summary()["reserved_cny"] == 0


@pytest.mark.parametrize("change", [
    None, {"analysis": []}, {"evidence": "invalid"}, {"image_ids": "img_a"},
    {"evidence": [{"image_id": "img_missing", "observation": "wrong source"}]},
    {"target_regions": {}}, {"target_regions": [{"image_id": "img_missing"}]},
])
def test_malformed_visual_evidence_assembly_is_an_input_fault_before_fact_request(tmp_path, album, change):
    from camroll_agent.evidence_gate import EvidenceGateInputError
    from camroll_agent.gate import OutputGate
    verifier = Verifier()
    agent, client, sdk = agent_for(tmp_path, album, [], verifier)
    result = {"analysis": "red", "evidence": [{"image_id": "img_a", "observation": "red"}],
        "limitations": "", "image_ids": ["img_a"], "target_regions": [], "images": []}
    result = None if change is None else {**result, **change}
    with client.request_scope() as scope, pytest.raises(EvidenceGateInputError) as caught:
        agent._gate_facts("color?", {"answer": "red", "outcome": "answer"}, album[1].images,
            ["img_a"], [{"tool": "view_image", "result": result}], "invalid-observation", scope, OutputGate("turn"))
    assert caught.value.code == "gate_input_error"
    assert not verifier.calls and not sdk.calls
    assert client.usage_summary()["reserved_cny"] == client.usage_summary()["estimated_spent_cny"] == 0


def test_final_facts_negative_repeats_exactly_once_then_frozen_repair(tmp_path, album):
    bad = {"answer": "blue", "evidence_image_ids": ["img_a"]}
    uncertain = {"answer": "perhaps red", "evidence_image_ids": ["img_a"]}
    payload = {"answer": "red", "evidence_image_ids": ["img_a"]}
    verifier = Verifier(("contradicted", "insufficient", "supported"))
    agent, _, sdk = agent_for(tmp_path, album, final_chain(bad, bad, uncertain, payload), verifier)
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "model" and result.final_text == "red"
    assert len(verifier.calls) == 3 and len(sdk.calls) == 6
    assert sdk.calls[2]["messages"] == sdk.calls[3]["messages"]
    assert sdk.calls[4]["messages"] != sdk.calls[3]["messages"]
    assert len({row["request_id"] for row in verifier.calls}) == 3
    facts = [event for event in result.gate_events if event["state"] == "evidence_checked"]
    assert [event["verdict"]["status"] for event in facts] == ["contradicted", "contradicted", "insufficient", "supported"]
    assert [event["reused_verdict"] for event in facts] == [False, True, False, False]


def test_exhausted_final_never_delivers_pre_gate_draft_or_images(tmp_path, album):
    payload = {"answer": "blue", "evidence_image_ids": ["img_a"]}
    agent, _, sdk = agent_for(tmp_path, album, final_chain(*([payload] * 4)), Verifier(("contradicted",) * 4))
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "gate_exhausted" and len(sdk.calls) == 6
    assert "unverified draft" not in result.final_text and "blue" not in result.final_text
    assert result.selected_image_ids == []


def test_inline_ids_subset_not_equality_with_reviewed_images(tmp_path, album):
    payload = {"answer": "red without inline citation", "evidence_image_ids": ["img_a"]}
    agent, _, sdk = agent_for(tmp_path, album, final_chain(payload))
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "model" and len(sdk.calls) == 3


def test_unknown_final_reference_uses_same_gate_not_outer_repair(tmp_path, album):
    bad = {"answer": "red [img_unknown]", "evidence_image_ids": ["img_a"]}
    good = {"answer": "red", "evidence_image_ids": ["img_a"]}
    verifier = Verifier()
    agent, _, sdk = agent_for(tmp_path, album, final_chain(bad, good), verifier)
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "model" and len(sdk.calls) == 4
    assert len(verifier.calls) == 1  # rejected reference incurs no verifier call
    assert sdk.calls[2]["messages"] == sdk.calls[3]["messages"]


def test_unknown_cloud_request_stops_without_new_semantic_identity(tmp_path, album):
    agent, client, sdk = agent_for(tmp_path, album, [TimeoutError("synthetic")])
    result = agent.ask("color?", structured_output=True)
    assert result.stopped_reason == "request_uncertain" and result.origin == "system_failure"
    assert len(sdk.calls) == 1 and client.usage_summary()["reserved_cny"] > 0
    assert result.gate_events[-1]["state"] == "blocked"


def test_provider403_is_not_output_repair(tmp_path, album):
    class Rejected(Exception):
        status_code = 403
    agent, client, sdk = agent_for(tmp_path, album, [Rejected("synthetic")])
    result = agent.ask("color?", structured_output=True)
    assert len(sdk.calls) == 1 and result.origin == "system_failure"
    assert client.usage_summary()["reserved_cny"] == 0


def test_durable_exact_repeat_has_new_call_and_same_hash_and_replays_after_restart(tmp_path):
    from camroll_agent.gate import OutputGate
    import sqlite3
    sdk = SDK([response("bad"), response('{"answer":"red"}')])
    ledger = BudgetLedger(tmp_path / "budget.sqlite", 600)
    client = QwenClient(sdk=sdk, ledger=ledger, expected_limit_cny=600, request_namespace="stable-run")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    for _ in range(2):
        with client.request_scope() as scope:
            assert client.generate_json("color?", [], schema, gate=OutputGate("qa001"), gate_id="final") == {"answer": "red"}
    assert len(sdk.calls) == 2 and scope.requests == 0
    with sqlite3.connect(client.journal.path) as connection:
        rows = connection.execute("SELECT request_id,payload_hash,call_id,status FROM cloud_requests").fetchall()
    assert len(rows) == 2 and rows[0][1] == rows[1][1]
    assert rows[0][0] != rows[1][0] and rows[0][2] != rows[1][2]
    assert all(row[3] == "completed" for row in rows)


def test_four_json_attempts_total_not_old_loop_times_four(tmp_path):
    from camroll_agent.gate import OutputGate, GateExhausted
    sdk = SDK([response("bad")] * 4)
    client = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite", 600))
    with client.request_scope() as scope:
        with pytest.raises(GateExhausted):
            client.generate_json("color?", [], {"type": "object"}, gate=OutputGate("qa001"), gate_id="view")
    assert scope.requests == len(sdk.calls) == 4


def test_off_keeps_original_three_json_attempts_and_no_gate_events(tmp_path, album):
    outcomes = [response(calls=[call("search", {"query": "red"})]),
                response(json.dumps({"answer": "red", "outcome": "answer", "selected_image_ids": ["img_a"]})),
                response(json.dumps({"answer": "red", "evidence_image_ids": ["img_a"]}))]
    verifier = Verifier()
    agent, _, sdk = agent_for(tmp_path, album, outcomes, verifier, policy="off")
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "model" and result.gate_status == "off"
    assert result.gate_attempts == 0 and result.gate_events == [] and verifier.calls == []
    assert len(sdk.calls) == 3


def test_off_system_fallback_never_becomes_model_refusal(tmp_path, album):
    agent, _, _ = agent_for(tmp_path, album, [TimeoutError("synthetic")], policy="off")
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "system_failure" and result.turn_outcome == "error"


def test_real_verifier_and_qwen_share_physical_request_cost_and_trace(tmp_path, album):
    from camroll_agent.evidence_gate import EvidenceGateVerifier
    payload = {"answer": "red", "evidence_image_ids": ["img_a"]}
    agent, client, sdk = agent_for(tmp_path, album, final_chain(payload))
    verifier_calls = []
    def verify_transport(body, timeout):
        verifier_calls.append(copy.deepcopy(body))
        verdict = {"status": "supported", "evidence_ids": ["image:img_a"], "reasons": [
            {"claim": "color", "assessment": "supported", "evidence_ids": ["image:img_a"], "reason": "raw color"}]}
        return {"model": "doubao-seed-2-0-lite-260215", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(verdict)}}]}
    agent.evidence_verifier = EvidenceGateVerifier(client=client, transport=verify_transport)
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "model" and len(sdk.calls) == 3 and len(verifier_calls) == 1
    assert result.usage["requests"] == 4
    assert len(result.model_call_metrics) == 4
    assert result.usage["prompt_tokens"] == 400
    assert result.usage["estimated_cny"] == pytest.approx(client.usage_summary()["estimated_spent_cny"])


def test_verifier_cannot_be_fifty_first_request_or_leave_reservation(tmp_path, album):
    from camroll_agent.evidence_gate import EvidenceGateVerifier
    payload = {"answer": "red", "evidence_image_ids": ["img_a"]}
    agent, client, sdk = agent_for(tmp_path, album, final_chain(payload))
    def no_request(*args, **kwargs):
        raise AssertionError("Verifier must not submit request 51")
    agent.evidence_verifier = EvidenceGateVerifier(client=client, transport=no_request)
    with client.request_scope(max_requests=50, deadline_s=600) as scope:
        scope.requests = 47
        result = agent.ask("color?", structured_output=True)
    assert result.stopped_reason == "request_limit" and result.origin == "system_failure"
    assert scope.requests == 50 and len(sdk.calls) == 3
    assert client.usage_summary()["reserved_cny"] == 0


def test_malformed_envelope_candidates_keep_original_call_identity(tmp_path, album):
    agent, _, sdk = agent_for(tmp_path, album, [response(None)] * 4)
    result = agent.ask("color?", structured_output=True)
    rejected = [event for event in result.gate_events if event["state"] == "rejected"]
    assert len(sdk.calls) == len(rejected) == 4
    assert all(event["candidate"] == {"role": "assistant", "content": None} for event in rejected)
    assert len({event["call_id"] for event in rejected}) == 4


def test_valid_no_image_refusal_is_model_output_not_gate_exhaustion(tmp_path, album):
    agent, _, sdk = agent_for(tmp_path, album, [response(json.dumps({
        "answer": "The question lacks enough detail", "outcome": "clarify", "selected_image_ids": []}))])
    result = agent.ask("which one?", structured_output=True)
    assert result.origin == "model" and result.turn_outcome == "clarify"
    assert len(sdk.calls) == 1 and result.selected_image_ids == []


def test_gate_rejects_legacy_vision_client_before_any_cloud_call(tmp_path, album):
    agent, client, sdk = agent_for(tmp_path, album, [])
    class LegacyVision:
        model = "qwen3.7-flash"
        def generate(self, prompt, paths):
            raise AssertionError("No legacy generation may bypass Gate-On")
    with pytest.raises(ValueError, match="single-layer"):
        Agent(str(album[0]), library=album[1], llm=client, vlm=LegacyVision(), inference_model="qwen3.7-flash",
              gate_policy="repair-evidence-v1", evidence_verifier=Verifier())
    assert sdk.calls == []


def test_repeated_successful_batch_keeps_three_round_stop_and_action_summary(tmp_path, album):
    search = response(calls=[call("search", {"query": "red"})])
    agent, _, sdk = agent_for(tmp_path, album, [search] * 4)
    result = agent.ask("color?", structured_output=True)
    assert result.stopped_reason == "no_progress" and len(sdk.calls) == 4
    assert len([entry for entry in result.tool_trace if entry["tool"] == "search" and not entry["cached"]]) == 1
    warning = [entry for entry in result.gate_events if entry["state"] == "no_progress_warning"]
    assert warning[0]["attempted_operations"][0]["args"]["query"] == "red"
    assert any("最近10项操作摘要" in str(row.get("content")) for row in sdk.calls[3]["messages"])


def test_clarification_gate_never_receives_unviewed_search_candidates(tmp_path, album):
    verifier = Verifier()
    agent, _, _ = agent_for(tmp_path, album, [response(calls=[call("search", {"query": "red"})]),
        response(json.dumps({"answer": "Which occasion?", "outcome": "clarify", "selected_image_ids": []}))], verifier)
    result = agent.ask("which photo?", structured_output=True)
    assert result.origin == "model" and result.turn_outcome == "clarify"
    assert verifier.calls[0]["images"] == []


@pytest.mark.parametrize("actual", [None, "qwen3.8-max", "unverified-version"])
def test_strict_actual_model_drift_is_settled_not_unknown(tmp_path, actual):
    from camroll_agent.qwen import ProviderError
    raw = response("ok")
    if actual is None:
        raw.pop("model")
    else:
        raw["model"] = actual
    sdk = SDK([raw])
    client = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite", 600), strict_response_model=True)
    with pytest.raises(ProviderError) as error:
        client.chat([{"role": "user", "content": "x"}], request_id="stable")
    assert error.value.code == "response_model_mismatch" and len(sdk.calls) == 1
    budget = client.usage_summary()
    assert budget["estimated_spent_cny"] > 0 and budget["reserved_cny"] == 0 and budget["uncertain_requests"] == 0
    with pytest.raises(ProviderError):
        client.chat([{"role": "user", "content": "x"}], request_id="stable")
    assert len(sdk.calls) == 1


@pytest.mark.parametrize("models", [[None, None], ["qwen3.8-max", "qwen3.7-flash"]])
def test_strict_stream_model_missing_or_mixed_cannot_be_hidden_by_fallback(tmp_path, models):
    from camroll_agent.qwen import ProviderError
    chunks = [{"choices": [{"index": 0, "delta": {"content": "text"}, "finish_reason": None}],
               **({"model": model} if model else {})} for model in models]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    sdk = SDK([iter(chunks)])
    client = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite", 600), strict_response_model=True)
    with pytest.raises(ProviderError) as error:
        client.chat([{"role": "user", "content": "x"}], stream=True)
    assert error.value.code == "response_model_mismatch"
    assert client.usage_summary()["estimated_spent_cny"] > 0
    assert client.usage_summary()["uncertain_requests"] == client.usage_summary()["reserved_cny"] == 0


def test_on_final_can_repair_answer_draft_to_honest_refusal(tmp_path, album):
    unsupported = {"answer": "blue", "outcome": "answer", "evidence_image_ids": ["img_a"]}
    refused = {"answer": "The exact date cannot be determined", "outcome": "refuse", "evidence_image_ids": []}
    verifier = Verifier(("insufficient", "supported"))
    agent, _, sdk = agent_for(tmp_path, album, final_chain(unsupported, unsupported, refused), verifier)
    result = agent.ask("exact date?", structured_output=True)
    assert result.origin == "model" and result.turn_outcome == "refuse"
    assert result.selected_image_ids == [] and "unverified draft" not in result.final_text
    assert [row["outcome"] for row in verifier.calls] == ["answer", "refuse"]
    assert [row["delivery_image_ids"] for row in verifier.calls] == [["img_a"], []]
    assert len(sdk.calls) == 5


@pytest.mark.parametrize("policy", ["off", "repair-evidence-v1"])
def test_new_experiment_retrieval_fault_stops_before_another_paid_decision(tmp_path, album, policy):
    from camroll_agent.wemm import WeMMError
    def broken(**kwargs):
        raise WeMMError("Synthetic backend unavailable", code="wemm_unavailable")
    album[1].embedding.encode = broken
    agent, _, sdk = agent_for(tmp_path, album, [response(calls=[call("search", {"query": "red"})])], policy=policy)
    result = agent.ask("color?", structured_output=True)
    assert result.stopped_reason == "retrieval_backend_error" and result.origin == "system_failure"
    assert len(sdk.calls) == 1


def test_off_preserves_pre_send_context_limit_for_group_stop_without_metric(tmp_path, album):
    agent, client, sdk = agent_for(tmp_path, album, [], policy="off")
    client.input_token_bound = 1
    result = agent.ask("color?", structured_output=True)
    assert result.origin == "system_failure" and result.stopped_reason == "provider_error"
    assert result.tool_trace[-1]["result"]["error"]["error_code"] == "context_limit"
    assert result.model_call_metrics == [] and sdk.calls == []
    assert client.usage_summary()["estimated_spent_cny"] == client.usage_summary()["reserved_cny"] == 0


def test_off_preserves_tool_pre_send_context_limit_without_extra_paid_call(tmp_path, album):
    outcomes = [response(calls=[call("search", {"query": "red"})]),
                response(calls=[call("view_image", {"image_ids": ["img_a"], "prompt": "color"})])]
    agent, client, sdk = agent_for(tmp_path, album, outcomes, policy="off")
    create = sdk.chat.completions.create
    def reduce_bound_after_decision(**kwargs):
        value = create(**kwargs)
        if len(sdk.calls) == 2:
            client.input_token_bound = 1
        return value
    sdk.chat.completions.create = reduce_bound_after_decision
    result = agent.ask("color?", structured_output=True)
    view = next(row for row in result.tool_trace if row["tool"] == "view_image")
    assert view["result"]["error"]["error_code"] == "context_limit"
    assert result.origin == "system_failure" and len(sdk.calls) == 2
    assert len(result.model_call_metrics) == 2


def test_same_candidate_four_generations_have_one_real_fact_judgment(tmp_path, album):
    from camroll_agent.evidence_gate import EvidenceGateVerifier
    payload = {"answer": "blue", "evidence_image_ids": ["img_a"]}
    agent, client, sdk = agent_for(tmp_path, album, final_chain(*([payload] * 4)))
    calls = []
    def transport(body, timeout):
        calls.append(body)
        verdict = {"status": "contradicted", "evidence_ids": ["image:img_a"], "reasons": [
            {"claim": "blue", "assessment": "contradicted", "evidence_ids": ["image:img_a"], "reason": "raw image is red"}]}
        return {"model": "doubao-seed-2-0-lite-260215", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(verdict)}}]}
    agent.evidence_verifier = EvidenceGateVerifier(client=client, transport=transport)
    result = agent.ask("color?", structured_output=True)
    facts = [event for event in result.gate_events if event["state"] == "evidence_checked"]
    assert result.origin == "gate_exhausted" and len(sdk.calls) == 6 and len(calls) == 1
    assert result.usage["requests"] == 7
    assert [event["reused_verdict"] for event in facts] == [False, True, True, True]
    assert len({event["verdict_cache_key"] for event in facts}) == 1
    assert len({tuple(event["verifier_call_ids"]) for event in facts}) == 1
    assert facts[0]["verifier_call_ids"] and all(event["verdict_source"] == "turn_cache" for event in facts[1:])


@pytest.mark.parametrize("change", ["question", "answer", "outcome", "delivery", "trusted", "derived", "image_content", "new_turn"])
def test_positive_fact_cache_requires_identical_candidate_and_all_evidence(tmp_path, album, change):
    from camroll_agent.gate import OutputGate
    verifier = Verifier(("supported",) * 3)
    agent, client, _ = agent_for(tmp_path, album, [], verifier)
    records = copy.deepcopy(album[1].images)
    gate = OutputGate("turn-one")
    question, payload = "color?", {"answer": "red", "outcome": "answer", "evidence_image_ids": ["img_a"]}
    with client.request_scope() as scope:
        for attempt in range(2):
            agent._gate_facts(question, payload, records, ["img_a"], [], "same-" + str(attempt), scope, gate)
        assert len(verifier.calls) == 1
        if change == "question":
            question = "shape?"
        elif change == "answer":
            payload["answer"] = "red square"
        elif change == "outcome":
            payload["outcome"] = "refuse"
        elif change == "delivery":
            payload["evidence_image_ids"] = []
        elif change == "trusted":
            records["img_a"].update(date="2026-10-01", date_source="explicit_metadata")
        elif change == "derived":
            records["img_a"]["caption"] = "different derived hint"
        elif change == "image_content":
            Image.new("RGB", (12, 12), "blue").save(album[0] / "assets" / "a.jpg")
        else:
            gate = OutputGate("turn-two")
        agent._gate_facts(question, payload, records, ["img_a"], [], "changed", scope, gate)
    assert len(verifier.calls) == 2


def test_journal_verdict_replay_is_labeled_reuse_not_new_judgment(tmp_path, album):
    from camroll_agent.gate import OutputGate
    class ReplayedVerifier:
        def verify(self, **kwargs):
            return {"status": "supported", "reasons": [], "evidence_ids": [],
                    "model_calls": [{"call_id": "original-call", "replayed": True, "api_sent": False}]}
    agent, client, _ = agent_for(tmp_path, album, [], ReplayedVerifier())
    gate = OutputGate("turn-one")
    with client.request_scope() as scope:
        agent._gate_facts("color?", {"answer": "red", "outcome": "answer"}, album[1].images,
                          ["img_a"], [], "replayed", scope, gate)
    event = gate.events[-1]
    assert event["reused_verdict"] is True and event["verifier_replayed"] is True
    assert event["verdict_source"] == "request_journal" and event["verifier_call_ids"] == ["original-call"]


def test_full_payload_mismatch_blocks_exact_repeat_before_accepting_output():
    from camroll_agent.gate import OutputGate, GatePayloadMismatch, GateViolation
    gate = OutputGate("one-turn")
    responses = iter([{"role": "assistant", "content": "bad", "_metric": {"payload_hash": "first"}},
                      {"role": "assistant", "content": "would otherwise pass", "_metric": {"payload_hash": "different"}}])
    def validate(raw, request_id):
        if raw["content"] == "bad":
            raise GateViolation("invalid_json")
        return raw
    with pytest.raises(GatePayloadMismatch):
        gate.run(gate_id="decision", role="agent", messages=[], send=lambda *args: next(responses), validate=validate)
    assert gate.events[-1]["state"] == "blocked" and gate.events[-1]["exact_payload_match"] is False
