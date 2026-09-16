import base64
import io
from types import SimpleNamespace

from PIL import Image
import pytest

from camroll_agent.budget import BudgetLedger, BudgetExceeded
from camroll_agent.qwen import (
    QwenClient, ProviderError, RequestLimitExceeded, DeadlineExceeded, image_messages,
)


SCHEMA = {"type": "object", "properties": {"caption": {"type": "string"}},
          "required": ["caption"], "additionalProperties": False}


class HTTPError(Exception):
    def __init__(self, status, text="request failed"):
        super().__init__(text)
        self.status_code = status


class APITimeoutError(Exception):
    pass


def answer(content='{"caption":"一只猫"}', *, usage=True, tool_calls=None):
    result = {"model": "qwen3.7-flash", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    if tool_calls:
        result["choices"][0]["message"]["tool_calls"] = tool_calls
    return result


class FakeSDK:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(tmp_path, outcomes, *, limit=300):
    sdk = FakeSDK(outcomes)
    value = QwenClient(sdk=sdk, ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=limit), sleep=lambda _: None)
    return value, sdk


def test_normalizes_tool_calls_and_disables_thinking(tmp_path):
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query":"猫"}'}}]
    qwen, sdk = client(tmp_path, [answer(None, tool_calls=tool_calls)])
    result = qwen.chat([{"role": "user", "content": "找猫"}], tools=[{"type": "function"}])
    assert result["tool_calls"] == tool_calls
    assert result["role"] == "assistant"
    assert sdk.calls[0]["model"] == "qwen3.7-flash"
    assert sdk.calls[0]["extra_body"] == {"enable_thinking": False}
    assert result["_usage"]["estimated_cny"] == pytest.approx(0.00028)


def test_transport_retries_429_and_timeout_are_actual_requests(tmp_path):
    qwen, sdk = client(tmp_path, [HTTPError(429), APITimeoutError(), answer()])
    with qwen.request_scope() as scope:
        qwen.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == scope.summary()["requests"] == 3
    assert qwen.usage_summary()["uncertain_requests"] == 1
    assert qwen.usage_summary()["reserved_cny"] > 1
    assert qwen.usage_summary()["estimated_spent_cny"] == pytest.approx(0.00028)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_error_never_retried_or_exposes_provider_payload(tmp_path, status):
    qwen, sdk = client(tmp_path, [HTTPError(status, "private-key and private-user-data")])
    with pytest.raises(ProviderError) as error:
        qwen.chat([{"role": "user", "content": "hi"}])
    assert "private" not in str(error.value)
    assert len(sdk.calls) == 1
    assert qwen.usage_summary()["reserved_cny"] == 0


def test_json_repairs_malformed_then_wrong_type(tmp_path):
    qwen, sdk = client(tmp_path, [answer('{"caption":'), answer('{"caption":3}'), answer()])
    assert qwen.generate_json("Describe", [], SCHEMA) == {"caption": "一只猫"}
    assert len(sdk.calls) == 3
    assert "Validation failed" in sdk.calls[2]["messages"][-1]["content"]


@pytest.mark.parametrize("value", ["null", "[]", '{"other":"x"}', '{"caption":null}'])
def test_invalid_json_outputs_exhaust_bounded_repair(tmp_path, value):
    qwen, sdk = client(tmp_path, [answer(value)] * 3)
    with pytest.raises(ProviderError) as error:
        qwen.generate_json("Describe", [], SCHEMA)
    assert error.value.code == "invalid_json"
    assert len(sdk.calls) == 3


def test_explicit_schema_unsupported_persists_fallback(tmp_path):
    qwen, sdk = client(tmp_path, [HTTPError(400, "json_schema is not supported"), answer(), answer()])
    assert qwen.generate_json("Describe", [], SCHEMA)["caption"]
    assert qwen.generate_json("Describe", [], SCHEMA)["caption"]
    assert [c["response_format"]["type"] for c in sdk.calls] == ["json_schema", "json_object", "json_object"]
    assert qwen.capabilities()["json_schema"] is False
    assert qwen.usage_summary()["reserved_cny"] == 0


def test_invalid_schema_response_does_not_trigger_fallback(tmp_path):
    qwen, sdk = client(tmp_path, [HTTPError(400, "Invalid schema: missing required")])
    with pytest.raises(ProviderError):
        qwen.generate_json("Describe", [], SCHEMA)
    assert len(sdk.calls) == 1
    assert qwen.capabilities()["json_schema"] is None


def test_request_limit_counts_retries_and_nested_clients_share_scope(tmp_path):
    qwen, sdk = client(tmp_path, [HTTPError(429), answer()])
    second = QwenClient(sdk=FakeSDK([answer()]), ledger=qwen.ledger)
    with qwen.request_scope(max_requests=2) as scope:
        qwen.chat([{"role": "user", "content": "hi"}])
        with second.request_scope() as nested:
            assert nested is scope
            with pytest.raises(RequestLimitExceeded):
                second.chat([{"role": "user", "content": "hi"}])
    assert scope.summary()["requests"] == 2


def test_budget_check_happens_before_sdk_call(tmp_path):
    qwen, sdk = client(tmp_path, [answer()], limit=1)
    with pytest.raises(BudgetExceeded):
        qwen.chat([{"role": "user", "content": "hi"}])
    assert sdk.calls == []


def test_missing_usage_retains_reservation(tmp_path):
    qwen, _ = client(tmp_path, [answer(usage=False)])
    qwen.chat([{"role": "user", "content": "hi"}])
    assert qwen.usage_summary()["uncertain_requests"] == 1
    assert qwen.usage_summary()["estimated_spent_cny"] == 0


def test_deadline_prevents_new_call(tmp_path):
    qwen, sdk = client(tmp_path, [answer()])
    with qwen.request_scope() as scope:
        scope.deadline = 0
        with pytest.raises(DeadlineExceeded):
            qwen.chat([{"role": "user", "content": "hi"}])
    assert sdk.calls == []


def test_real_sdk_requires_workspace_endpoint(monkeypatch):
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    with pytest.raises(ProviderError) as error:
        QwenClient()
    assert error.value.code == "missing_endpoint"


def test_images_are_oriented_resized_jpeg_and_preserve_prompt(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGBA", (4000, 2000), "red").save(path)
    messages = image_messages("Evidence ID photo-1", [path])
    assert messages[0]["content"][0]["text"] == "Evidence ID photo-1"
    image_data = messages[0]["content"][-1]["image_url"]["url"]
    assert image_data.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[1]))) as image:
        assert image.size == (2048, 1024)
        assert image.mode == "RGB"
    qwen, _ = client(tmp_path, [answer()])
    assert qwen.chat(messages)["content"]


def test_native_stream_assembles_tool_arguments_and_usage(tmp_path):
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call1", "function": {"name": "search", "arguments": '{"query":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"cat"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    ]
    qwen, sdk = client(tmp_path, [iter(chunks)])
    result = qwen.chat([{"role": "user", "content": "hi"}], stream=True)
    assert result["tool_calls"][0]["function"]["arguments"] == '{"query":"cat"}'
    assert result["_usage"]["total_tokens"] == 120
    assert sdk.calls[0]["stream_options"] == {"include_usage": True}


def test_final_generation_one_attempt(tmp_path):
    qwen, sdk = client(tmp_path, [answer("null"), answer()])
    with pytest.raises(ProviderError):
        qwen.generate_json("Final answer", [], SCHEMA, role="final", max_attempts=1)
    assert len(sdk.calls) == 1
