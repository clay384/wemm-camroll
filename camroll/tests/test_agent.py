"""Behavioral agent regression tests; no server, model or filesystem assets."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import pytest

from camroll_agent.agent import Agent
from camroll_agent.prompts import format_observation
from camroll_agent.schemas import TOOLS, ToolValidationError, validate_arguments
from camroll_agent import tools


def tool(name="search", args=None, id="call_1", raw=None):
    return {"id": id, "type": "function", "function": {"name": name,
            "arguments": raw if raw is not None else json.dumps({"query": "公园"} if args is None else args)}}


def response(*calls, content=None):
    return {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}


class FakeLibrary:
    version = "frozen-snapshot-1"

    def __init__(self, count=2):
        self.calls = []
        self.images = {f"img_{i}": {"id": f"img_{i}", "image_id": f"img_{i}", "kind": "image",
            "path": f"/managed/assets/{i}.jpg", "source_path": f"/private/original/{i}.jpg",
            "caption": f"公园照片 {i}", "date": "2026-09-01", "event_id": "ev_1", "event_name": "公园"}
            for i in range(1, count+1)}
        self.events = {"ev_1": {"id": "ev_1", "kind": "event", "image_ids": list(self.images), "name": "公园"}}
        self.result = {"results": list(self.images.values()), "count": count}
        self.failure = None

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.result)

    def grep(self, **kwargs):
        self.calls.append(("grep", kwargs))
        return copy.deepcopy(self.result)

    def list_by_date(self, **kwargs):
        self.calls.append(("list_by_date", kwargs))
        return copy.deepcopy(self.result)

    def get(self, id):
        self.calls.append(("get", id))
        return copy.deepcopy(self.images.get(id) or self.events.get(id) or {"error": "not found"})

    def image_paths(self, ids):
        if any(i not in self.images for i in ids):
            raise ValueError("unknown")
        return [Path(self.images[i]["path"]) for i in ids]


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.generations = []
        self.scope_args = None
        self.final_bad_id = False
        self.requests = 0

    @contextmanager
    def request_scope(self, **kwargs):
        self.scope_args = kwargs
        yield self

    def summary(self):
        return {"requests": self.requests, "prompt_tokens": self.requests * 10, "completion_tokens": self.requests * 2}

    def chat(self, messages, tools=None, role="agent"):
        self.requests += 1
        self.messages.append(copy.deepcopy(messages))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value

    def generate_json(self, prompt, image_paths, schema, role="view", max_attempts=3):
        self.requests += 1
        self.generations.append({"prompt": prompt, "paths": image_paths, "schema": schema,
                                 "role": role, "max_attempts": max_attempts})
        if role == "final":
            ids = schema["properties"]["evidence_image_ids"]["items"]["enum"]
            return {"answer": f"照片在公园 [{ids[0]}]", "evidence_image_ids": ["img_UNKNOWN" if self.final_bad_id else ids[0]]}
        ids = schema["properties"]["evidence"]["items"]["properties"]["image_id"]["enum"]
        return {"analysis": "公园", "evidence": [{"image_id": i, "observation": f"看到树木 {len(self.generations)}"} for i in ids], "limitations": "无法判断具体地点"}


def make(responses, config="M1", lib=None, **kwargs):
    client = FakeClient(responses)
    library = lib or FakeLibrary()
    return Agent("unused", llm=client, library=library, config=config, **kwargs), client, library


@pytest.mark.parametrize("raw", ["{bad", "null", "[]", '"text"', "true", '{"query": null}',
    '{"query":"", "top_k":10}', '{"query":"x", "top_k":true}', '{"query":"x", "top_k":"5"}',
    '{"query":"x", "top_k":0}', '{"query":"x", "kind":"unknown"}', '{"query":"x", "path":"/tmp"}',
    '{"query":"x", "date_from":"2026-02-30"}', '{"query":"x", "date_from":"2026-09-10", "date_to":"2026-09-01"}',
    '{"query":"x", "top_k":NaN}'])
def test_invalid_arguments_not_executed_and_can_correct(raw):
    agent, client, library = make([response(tool(raw=raw)), response(tool(id="fixed")), response(content="在公园 [img_1]")])
    result = agent.ask("在哪里？")
    assert result.stopped_reason == "ok"
    assert len([x for x in library.calls if x[0] == "search"]) == 1
    observation = client.messages[1][-2]["content"]
    assert '"status": "error"' in observation
    assert '"field"' in observation
    assert result.tool_trace[0]["result"]["error"]["code"] in ("invalid_json", "invalid_arguments")


@pytest.mark.parametrize("bad", [
    {"role": "assistant", "tool_calls": "bad"},
    {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "search", "arguments": "{}"}}]},
    {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"arguments": "{}"}}]},
    {"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "search", "arguments": {}}}]},
    {"role": "assistant", "content": ["invalid"]},
    {"role": "user", "content": "invalid role"},
    response(),
])
def test_protocol_correction_rejects_envelope_before_transcript(bad):
    agent, client, library = make([bad, response(tool()), response(content="在公园 [img_1]")])
    result = agent.ask("在哪里？")
    assert result.stopped_reason == "ok"
    assert result.tool_trace[0]["result"]["error"]["code"] == "invalid_protocol"
    assert bad not in client.messages[1]
    assert len(library.calls) == 1


def test_protocol_has_only_two_correction_chances():
    agent, client, library = make([response(), response(), response()])
    result = agent.ask("问题")
    assert result.stopped_reason == "protocol_error"
    assert result.steps == 3
    assert library.calls == []


def test_unknown_tool_and_missing_required_fields():
    agent, _, library = make([response(tool("execute", {"command": "rm"})),
                            response(tool("search", {})), response(tool("search", {}))])
    result = agent.ask("问题")
    assert result.stopped_reason == "invalid_tool_arguments"
    assert library.calls == []
    assert result.tool_trace[0]["result"]["error"]["allowed"] == [t["function"]["name"] for t in TOOLS]


@pytest.mark.parametrize("name,args", [("get", {"id": "/tmp/a.jpg"}), ("get", {"id": "img_missing"}),
    ("view_image", {"image_ids": ["img_missing"], "prompt": "看图"}),
    ("view_image", {"image_ids": "img_1", "prompt": "看图"}),
    ("view_image", {"image_ids": ["img_1", "img_1"], "prompt": "看图"})])
def test_invalid_ids_cannot_cause_image_generation(name, args):
    agent, client, _ = make([response(tool(name, args)), response(content="无法确定")])
    result = agent.ask("问题")
    assert not client.generations
    assert result.view_image_calls == 0
    assert "error" in result.tool_trace[0]["result"]


def test_duplicate_successful_call_is_cached_ignoring_purpose():
    agent, _, library = make([response(tool(args={"query": "公园", "purpose": "寻找"})),
        response(tool(args={"query": "公园", "purpose": "再次寻找"})), response(content="公园 [img_1]")])
    result = agent.ask("哪里？")
    assert len(library.calls) == 1
    assert result.tool_trace[1]["cached"] is True


def test_failed_tool_is_explicit_and_never_reexecuted():
    lib = FakeLibrary()
    lib.failure = RuntimeError("PRIVATE /secret/key")
    agent, client, _ = make([response(tool()), response(tool()), response(tool())], lib=lib)
    result = agent.ask("哪里？")
    assert result.stopped_reason == "tool_error"
    assert len(lib.calls) == 1
    assert result.tool_trace[1]["cached"] is True
    assert "PRIVATE" not in json.dumps(result.tool_trace)
    assert "不能视为没有匹配" in result.final_text
    assert any("连续两轮" in str(message) for message in client.messages[-1])


def test_empty_results_end_after_strategy_change_and_do_not_claim_absence():
    lib = FakeLibrary()
    lib.result = {"results": [], "count": 0}
    agent, client, _ = make([response(tool()), response(tool(args={"query": "树木"})),
                            response(tool("grep", {"query": "树"}))], lib=lib)
    result = agent.ask("是否没有树？")
    assert result.stopped_reason == "no_progress"
    assert "无法确定" in result.final_text
    assert "不存在" not in result.final_text
    obs = format_observation("grep", {}, {"results": []})
    assert "不能据此" in obs


def test_hallucinated_answer_without_tools_is_not_returned():
    agent, _, _ = make([response(content="这是你去年在北京的照片")])
    result = agent.ask("哪里？")
    assert result.stopped_reason == "insufficient_evidence"
    assert "北京" not in result.final_text


def test_five_views_cap_and_all_cloud_work_uses_same_scope():
    calls = [tool("view_image", {"image_ids": ["img_1"], "prompt": f"详情{i}"}, id=f"v{i}") for i in range(6)]
    agent, client, _ = make([response(*calls), response(content="有树 [img_1]")])
    result = agent.ask("哪里？")
    assert result.view_image_calls == 5
    assert len(client.generations) == 5
    assert result.tool_trace[5]["result"]["error"]["code"] == "view_image_limit"
    assert client.scope_args == {"max_requests": 50, "deadline_s": 180}
    assert result.usage["requests"] == 7


@pytest.mark.parametrize("error_name,reason", [("BudgetExceeded", "budget_exhausted"),
    ("RequestLimitExceeded", "request_limit"), ("DeadlineExceeded", "timeout"), ("ProviderError", "provider_error")])
def test_provider_limits_preserved_and_do_not_become_no_results(error_name, reason):
    error = type(error_name, (RuntimeError,), {})()
    agent, _, _ = make([error])
    result = agent.ask("问题")
    assert result.stopped_reason == reason
    assert result.tool_trace[-1]["result"]["error"]["code"] == reason


@pytest.mark.parametrize("config", ["M2", "M4"])
def test_final_review_receives_at_most_six_actual_images_and_verified_ids(config):
    agent, client, _ = make([response(tool()), response(content="公园 [img_1]")], config=config, lib=FakeLibrary(9))
    result = agent.ask("哪里？")
    final = client.generations[-1]
    assert final["role"] == "final"
    assert final["max_attempts"] == 1
    assert len(final["paths"]) == 6
    assert final["paths"][0] == "/managed/assets/1.jpg"
    assert "photo_order" in final["prompt"]
    assert "/private" not in final["prompt"]
    assert result.evidence_image_ids == ["img_1"]
    assert result.index_version == "frozen-snapshot-1"


def test_final_review_prioritizes_viewed_images_before_search_rank():
    agent, client, _ = make([response(tool()), response(tool("view_image", {"image_ids": ["img_8"], "prompt": "看看"})),
                            response(content="公园 [img_8]")], config="M2", lib=FakeLibrary(9))
    result = agent.ask("哪里？")
    assert client.generations[-1]["paths"][0] == "/managed/assets/8.jpg"
    assert result.evidence_image_ids == ["img_8"]


def test_final_review_bad_source_is_rejected_without_another_generation():
    agent, client, _ = make([response(tool()), response(content="公园 [img_1]")], config="M2")
    client.final_bad_id = True
    result = agent.ask("哪里？")
    assert result.stopped_reason == "final_response_error"
    assert "img_UNKNOWN" not in result.final_text
    assert len(client.generations) == 1


def test_final_review_no_images_makes_no_cloud_review_request():
    agent, client, _ = make([response(content="无法确定")], config="M2")
    result = agent.ask("哪里？")
    assert client.generations == []
    assert result.stopped_reason == "insufficient_evidence"


def test_unretrieved_citation_is_not_exposed_as_valid_evidence():
    agent, _, _ = make([response(tool()), response(content="公园 [img_fake]")])
    result = agent.ask("哪里？")
    assert result.stopped_reason == "invalid_evidence_reference"
    assert "img_fake" not in result.final_text
    assert result.evidence_image_ids == []


def test_streaming_and_sync_are_same_state_machine():
    replies = [response(tool(raw="null")), response(tool()), response(content="公园 [img_1]")]
    agent, _, _ = make(copy.deepcopy(replies))
    sync = agent.ask("哪里？")
    streamed, _, _ = make(copy.deepcopy(replies))
    events = list(streamed.ask_streaming("哪里？"))
    answer = events[-1][1]
    assert events[-1][0] == "answer"
    assert sync.final_text == answer["response"]
    assert sync.tool_trace == answer["tool_trace"]
    assert sync.stopped_reason == answer["stopped_reason"]
    assert sync.usage == answer["usage"]
    assert "thought" not in [event for event, _ in events]


def test_tools_keep_paths_out_of_model_observations():
    data = FakeLibrary().result
    data["results"][0].update(embedding_path="/managed/embedding.png", original_path="/private/original.png",
                              date_source_path="/private/date-source.png")
    text = format_observation("search", {}, data)
    assert "/managed" not in text
    assert "/private" not in text
    error = format_observation("search", {}, {"error": {"code": "failed"}})
    assert '"status": "error"' in error
    assert "no results" not in error


def test_legacy_injected_clients_supported():
    class OldLLM:
        def __init__(self):
            self.calls = iter([response(tool()), response(content="公园 [img_1]")])
        def chat(self, messages, tools=None):
            return next(self.calls)
    result = Agent("unused", llm=OldLLM(), library=FakeLibrary()).ask("哪里？")
    assert result.stopped_reason == "ok"
    assert result.usage["requests"] == 2


def test_required_thought_was_removed_from_all_schemas():
    for spec in TOOLS:
        assert "thought" not in spec["function"]["parameters"]["required"]
        assert "purpose" not in spec["function"]["parameters"]["required"]


def test_query_too_long_gives_actionable_shortening_feedback():
    class TooLong(RuntimeError):
        code = "query_too_long"
    library = FakeLibrary()
    library.failure = TooLong("internal/private diagnostic")
    agent, client, _ = make([response(tool()), response(content="无法确定")], lib=library)
    result = agent.ask("问题")
    error = result.tool_trace[0]["result"]["error"]
    assert error["code"] == "query_too_long"
    assert error["max_tokens"] == 512
    assert "缩短" in error["reason"]
    assert "private" not in str(error)


def test_real_qwen_scope_survives_stream_resumes_in_fresh_contexts(tmp_path):
    from contextvars import Context
    from types import SimpleNamespace
    from camroll_agent.qwen import QwenClient
    from camroll_agent.budget import BudgetLedger

    payloads = iter([response(tool(raw="null")), response(tool()), response(content="公园 [img_1]")])
    api_calls = []
    def create(**kwargs):
        api_calls.append(kwargs)
        return {"model": "qwen3.7-flash", "choices": [{"finish_reason": "stop", "message": next(payloads)}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    qwen = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite"))
    stream = Agent("unused", llm=qwen, library=FakeLibrary()).ask_streaming("哪里？")
    final = None
    sentinel = object()
    while True:
        item = Context().run(next, stream, sentinel)
        if item is sentinel:
            break
        if item[0] == "answer":
            final = item[1]
    assert final["stopped_reason"] == "ok"
    assert len(api_calls) == final["usage"]["requests"] == 3
    assert all(not any(key.startswith("_") for key in message) for call in api_calls for message in call["messages"])
    # Cancellation/close must reset the same context token as normal exhaustion.
    cancelled = Agent("unused", llm=qwen, library=FakeLibrary()).ask_streaming("问题")
    Context().run(next, cancelled)
    Context().run(cancelled.close)


def test_provider_detected_empty_envelope_uses_protocol_repair_loop():
    from camroll_agent.qwen import ProviderError
    agent, _, library = make([ProviderError("empty", code="empty_response"), response(tool()),
                             response(content="公园 [img_1]")])
    result = agent.ask("哪里？")
    assert result.stopped_reason == "ok"
    assert result.tool_trace[0]["result"]["error"]["code"] == "invalid_protocol"
    assert len(library.calls) == 1


def test_failed_visual_generation_still_consumes_view_quota():
    from camroll_agent.qwen import ProviderError
    agent, client, _ = make([response(tool("view_image", {"image_ids": ["img_1"], "prompt": "看看"}))])
    def fail(**kwargs):
        raise ProviderError("failed")
    client.generate_json = fail
    result = agent.ask("哪里？")
    assert result.view_image_calls == 1
    assert result.stopped_reason == "provider_error"


@pytest.mark.parametrize('answer,refs', [
    ('错误引用 [img_8]', ['img_1']),
    ('引用不一致 [img_2]', ['img_1']),
    ('遗漏来源 [img_1] [img_2]', ['img_1']),
    ('引用事件 [ev_1]', ['img_1']),
])
def test_final_inline_references_match_selected_images_and_explicit_refs(answer, refs):
    agent, client, _ = make([response(tool()), response(content='原草稿 [img_1]')],
                            config='M2', lib=FakeLibrary(9))
    def final(**kwargs):
        return {'answer': answer, 'evidence_image_ids': refs}
    client.generate_json = final
    result = agent.ask('哪里？')
    assert result.stopped_reason == 'invalid_evidence_reference'
    assert result.evidence_image_ids == []
    assert answer not in result.final_text
    assert any(t['tool'] == 'final_image_review' and 'error' in t['result'] for t in result.tool_trace)


def test_final_explicit_references_are_allowed_without_inline_citations():
    agent, client, _ = make([response(tool()), response(content='原草稿 [img_1]')], config='M2')
    client.generate_json = lambda **kwargs: {'answer': '公园里有树木。', 'evidence_image_ids': ['img_1']}
    result = agent.ask('哪里？')
    assert result.stopped_reason == 'ok'
    assert result.final_text == '公园里有树木。'
    assert result.evidence_image_ids == ['img_1']


def test_final_summary_can_declare_all_reviewed_images_and_cite_one_example_inline():
    agent, client, _ = make([response(tool()), response(content='原草稿 [img_1]')],
                            config='M4', lib=FakeLibrary(5))
    declared = [f'img_{i}' for i in range(1, 6)]
    answer = '这五张图中有狗、猫和鸟。例如第一张里有一只狗 [img_1]。'
    captured = {}
    def final(**kwargs):
        captured.update(kwargs)
        return {'answer': answer, 'evidence_image_ids': declared}
    client.generate_json = final
    result = agent.ask('请总结这些动物照片，并举例。')
    assert result.stopped_reason == 'ok'
    assert result.final_text == answer
    assert result.evidence_image_ids == declared
    assert len(captured['image_paths']) == 5
    assert '该列表可以多于正文逐一引用的图片' in captured['prompt']
