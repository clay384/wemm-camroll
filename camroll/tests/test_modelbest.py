"""Offline provider-contract tests; no ModelBest credentials or network."""
import copy
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

from camroll_agent.agent import Agent, _assistant_envelope
from camroll_agent.budget import BudgetLedger, RequestJournal, RequestUncertain
from camroll_agent.modelbest import MODEL, ModelBestClient
from camroll_agent.qwen import ProviderError, RequestLimitExceeded
from camroll_agent.schemas import TOOLS, validate_arguments, ToolValidationError


SCHEMA = {"type": "object", "properties": {"caption": {"type": "string"}},
          "required": ["caption"], "additionalProperties": False}


def evidence():
    return {"model": MODEL, "input_rate": 0, "output_rate": 0,
            "observed_at": datetime.now(timezone.utc).isoformat(), "source": "https://platform.modelbest.cn/models",
            "is_free": False, "units": "credits", "input_pricing": "0", "output_pricing": "0"}


def answer(content='{"caption":"一只猫"}', *, usage=True, tool_calls=None, model=MODEL, finish="stop"):
    message = {"role": "assistant", "content": content, "reasoning": "Private model reasoning"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    value = {"model": model, "choices": [{"finish_reason": finish, "message": message}]}
    if usage:
        value["usage"] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 5, "created_cache_tokens": 10, "multimodal_tokens": {"image": 60}},
            "completion_tokens_details": {"reasoning_tokens": 20}}
    return value


class HTTPError(Exception):
    def __init__(self, status, text="private provider payload", body=None, content=b"nonempty"):
        super().__init__(text)
        self.status_code, self.body = status, body
        self.response = SimpleNamespace(content=content)


class SDK:
    def __init__(self, outcomes):
        self.outcomes, self.calls = iter(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return result


def client(tmp_path, outcomes, **kwargs):
    sdk = SDK(outcomes)
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600, expected_limit_cny=600)
    return ModelBestClient(sdk=sdk, ledger=ledger, pricing_evidence=evidence(), sleep=lambda _: None, **kwargs), sdk


def test_wire_contract_reasoning_separation_and_zero_price_usage(tmp_path):
    c, sdk = client(tmp_path, [answer()])
    with c.request_scope() as scope:
        result = c.chat([{"role": "user", "content": "你好"}])
    wire = sdk.calls[0]
    assert wire["model"] == MODEL
    assert wire["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert wire["temperature"] == 0 and wire["max_tokens"] == 8192 and wire["stream"] is False
    assert wire["timeout"] <= 90
    assert "reasoning" not in result and "Private model reasoning" not in json.dumps(result)
    assert result["_usage"]["reasoning_tokens"] == 20
    assert result["_usage"]["completion_tokens"] == 50 and result["_usage"]["image_tokens"] == 60
    assert scope.summary()["requests"] == 1 and scope.summary()["total_tokens"] == 150
    assert c.usage_summary()["requests"] == 1 and c.usage_summary()["estimated_spent_cny"] == 0
    assert c.capabilities()["context_token_bound"] == 256000
    assert c.capabilities()["max_output_tokens"] == 0
    with sqlite3.connect(c.ledger.path) as db:
        metadata = json.loads(db.execute("SELECT metadata FROM budget_calls").fetchone()[0])
    assert metadata["provider"] == "modelbest" and metadata["reasoning_tokens"] == 20


def test_sdk_defaults_minimal_request_and_explicit_unspecified_audit(tmp_path):
    c, sdk = client(tmp_path, [answer()], request_profile="sdk-defaults-v1", thinking_mode="unspecified")
    result = c.chat([{"role": "user", "content": "hello"}], max_tokens=None)
    assert set(sdk.calls[0]) == {"model", "messages", "timeout"}
    assert result["_metric"]["input"] == {"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}
    assert result["_metric"]["thinking_mode"] == "unspecified"
    assert result["_metric"]["streaming"] is False
    caps = c.capabilities()
    assert caps["request_profile"] == "sdk-defaults-v1" and caps["thinking_mode"] == "unspecified"
    assert caps["enable_thinking"] is caps["requested_max_output_tokens"] is caps["stream"] is None
    assert caps["max_output_tokens"] == 0 and caps["context_token_bound"] == 256000
    with sqlite3.connect(c.ledger.path) as db:
        metadata = json.loads(db.execute("SELECT metadata FROM budget_calls").fetchone()[0])
    assert metadata["prompt_version"] == "camroll-modelbest-sdk-defaults-v1"
    assert metadata["enable_thinking"] is metadata["stream"] is metadata["max_tokens"] is None


@pytest.mark.parametrize("arguments", [{"max_tokens": 8192}, {"max_tokens": 1}, {"stream": True}])
def test_sdk_defaults_rejects_explicit_overrides_before_request(tmp_path, arguments):
    c, sdk = client(tmp_path, [], request_profile="sdk-defaults-v1", thinking_mode="unspecified")
    with pytest.raises(ProviderError) as caught:
        c.chat([{"role": "user", "content": "hello"}], **arguments)
    assert caught.value.code == "invalid_request" and not sdk.calls
    assert c.usage_summary()["requests"] == 0


@pytest.mark.parametrize("request_profile,thinking_mode", [("fixed-v1", "unspecified"),
    ("sdk-defaults-v1", "on"), ("sdk-defaults-v1", "off"), ("unknown", "unspecified")])
def test_request_profiles_never_silently_fallback(tmp_path, request_profile, thinking_mode):
    with pytest.raises(ValueError):
        client(tmp_path, [], request_profile=request_profile, thinking_mode=thinking_mode)


@pytest.mark.parametrize("change", [{"model": "other"}, {"input_rate": 1}, {"output_rate": False},
                                    {"observed_at": "2020-01-01T00:00:00+00:00"}, {"source": ""}])
def test_evidence_required_and_current(tmp_path, change):
    with pytest.raises(ValueError):
        ModelBestClient(pricing_evidence={**evidence(), **change}, sdk=SDK([]),
                        ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))


def test_absent_evidence_never_defaults_to_free(tmp_path):
    with pytest.raises(ValueError):
        ModelBestClient(pricing_evidence=None, sdk=SDK([]))


def test_sdk_serialization_retains_real_content_and_only_omits_none(tmp_path):
    from openai.types.chat import ChatCompletion
    from camroll_agent.qwen import _as_dict
    raw = answer()
    raw.update(id='offline-test', created=0, object='chat.completion')
    raw['choices'][0]['index'] = 0
    response = ChatCompletion.model_validate(raw)
    assert _as_dict(response)['choices'][0]['message']['content'] == '{"caption":"一只猫"}'
    c, sdk = client(tmp_path, [response])
    assert c.generate_json('Describe', [], SCHEMA, max_attempts=1) == {'caption': '一只猫'}
    raw['choices'][0]['message'].pop('content')
    response = ChatCompletion.model_validate(raw)
    assert response.choices[0].message.content is None
    assert 'content' not in _as_dict(response)['choices'][0]['message']


def test_reasoning_only_sdk_response_is_settled_failure_not_a_discarded_answer(tmp_path):
    from openai.types.chat import ChatCompletion
    raw = {'id': 'offline-thinking-only', 'created': 0, 'object': 'chat.completion', 'model': MODEL,
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'reasoning': 'synthetic'}}],
        'usage': {'prompt_tokens': 2671, 'completion_tokens': 18, 'total_tokens': 2689,
                  'completion_tokens_details': {'reasoning_tokens': 17}}}
    c, sdk = client(tmp_path, [ChatCompletion.model_validate(raw)])
    observed = []
    with c.capture_calls(observed.append):
        with pytest.raises(ProviderError) as caught:
            c.generate_json('Describe', [], SCHEMA, max_attempts=1)
    assert caught.value.code == 'invalid_json' and len(sdk.calls) == 1
    assert observed[0]['error_code'] == 'empty_response'
    assert observed[0]['reasoning_tokens'] == 17 and observed[0]['parse_status'] == 'invalid'
    assert c.usage_summary()['uncertain_requests'] == 0
    assert c.usage_summary()['estimated_spent_cny'] == 0
    with sqlite3.connect(c.ledger.path) as db:
        assert db.execute('SELECT status FROM budget_calls').fetchone()[0] == 'settled'


def test_actual_openai_sdk_wire_body_preserves_content_without_network(tmp_path):
    import httpx
    from openai import OpenAI
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        raw = answer()
        raw.update(id='offline-wire', object='chat.completion', created=0)
        raw['choices'][0]['index'] = 0
        return httpx.Response(200, json=raw)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key='offline-test-only', base_url='https://injected.invalid/v1',
                     http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(),
            ledger=BudgetLedger(tmp_path / 'budget.sqlite', limit_cny=600, expected_limit_cny=600))
        assert c.generate_json('Describe', [], SCHEMA, max_attempts=1) == {'caption': '一只猫'}
    assert len(requests) == 1
    assert requests[0]['chat_template_kwargs'] == {'enable_thinking': True}
    assert 'extra_body' not in requests[0] and requests[0]['max_tokens'] == 8192
    assert requests[0]['stream'] is False and requests[0]['temperature'] == 0


def test_actual_sdk_defaults_json_image_wire_omits_all_generation_parameters(tmp_path):
    import httpx
    from openai import OpenAI
    from PIL import Image
    requests = []
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), "red").save(photo)
    def respond(request):
        requests.append(json.loads(request.content))
        raw = answer()
        raw.update(id="offline-defaults", object="chat.completion", created=0)
        raw["choices"][0]["index"] = 0
        # SDK defaults may provide no reasoning fields; final JSON still counts.
        raw["choices"][0]["message"].pop("reasoning")
        raw["usage"].pop("completion_tokens_details")
        return httpx.Response(200, json=raw)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key="offline-test-only", base_url="https://injected.invalid/v1",
                     http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), request_profile="sdk-defaults-v1",
            thinking_mode="unspecified", ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        metrics = []
        with c.capture_calls(metrics.append):
            assert c.generate_json("Describe", [photo], SCHEMA, max_attempts=1) == {"caption": "一只猫"}
    assert len(requests) == len(metrics) == 1
    assert set(requests[0]) == {"model", "messages", "response_format"}
    assert requests[0]["response_format"]["type"] == "json_schema"
    images = [item for item in requests[0]["messages"][1]["content"] if item["type"] == "image_url"]
    assert len(images) == 1 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert metrics[0]["parse_status"] == "valid" and metrics[0]["reasoning_chars"] == 0
    assert "reasoning_tokens" not in metrics[0]


def _failure_audits(client):
    with sqlite3.connect(client.journal.path) as db:
        rows = db.execute("SELECT status,response FROM cloud_requests ORDER BY created").fetchall()
    return [(status, json.loads(response)["_failure"]) for status, response in rows if status != "completed"]


@pytest.mark.parametrize("mode", [None, "legacy-v1", "modelbest-v2"])
def test_real_sdk_decision_system_prefix_adapter_is_explicit_and_full_schema_is_retained(tmp_path, mode):
    import httpx
    from openai import OpenAI
    requests = []
    messages = [{"role": "system", "content": "Keep the original instructions exactly.\n第二行。"},
                {"role": "user", "content": "Call get for the selected image."}]
    original = copy.deepcopy(messages)
    schema = ModelBestClient._decision_schema(TOOLS)
    before = ModelBestClient._decision_messages(messages, TOOLS, schema)
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if any(message["role"] == "system" for message in body["messages"][1:]):
            return httpx.Response(400, json={"error": {"message": "System message must be at the beginning.",
                                                       "type": "BadRequestError", "code": 400}})
        raw = answer('{"tool_calls":[{"name":"get","arguments":{"id":"image_123456789abc"}}]}')
        raw.update(id="offline-prefix", object="chat.completion", created=0)
        raw["choices"][0]["index"] = 0
        return httpx.Response(200, json=raw)
    options = {} if mode is None else {"compatibility_mode": mode}
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key="offline-test-only", base_url="https://injected.invalid/v1",
                     http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), request_profile="sdk-defaults-v1",
            thinking_mode="unspecified", decision_mode="json_decision", **options,
            ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        if mode == "modelbest-v2":
            result = c.chat(messages, tools=TOOLS)
            assert result["tool_calls"][0]["function"]["name"] == "get"
            assert requests[0]["messages"] == [{"role": "system", "content":
                before[0]["content"] + "\n\n" + before[1]["content"]}, before[2]]
        else:
            with pytest.raises(ProviderError) as caught:
                c.chat(messages, tools=TOOLS)
            assert caught.value.code == "http_400"
            assert requests[0]["messages"] == before
            assert _failure_audits(c)[0][1]["body"]["error"]["message"] == "System message must be at the beginning."
    assert len(requests) == 1 and messages == original
    expected_wire_schema = copy.deepcopy(schema)
    if mode == "modelbest-v2":
        expected_wire_schema["oneOf"][0]["properties"]["selected_image_ids"].pop("uniqueItems")
        assert c.capabilities()["compatibility_mode"] == "modelbest-v2"
        assert c.capabilities()["protocol_version"] == "minicpm47-sdk-defaults-m4-off-v2"
        assert result["_metric"]["compatibility_mode"] == "modelbest-v2"
        with sqlite3.connect(c.ledger.path) as db:
            metadata = json.loads(db.execute("SELECT metadata FROM budget_calls").fetchone()[0])
        assert metadata["prompt_version"] == "minicpm47-sdk-defaults-m4-off-v2"
    assert requests[0]["response_format"]["json_schema"]["schema"] == expected_wire_schema
    assert set(requests[0]) == {"model", "messages", "response_format"}


@pytest.mark.parametrize("decision_mode", ["native_tools", "json_decision"])
def test_single_prefix_adapter_rejects_later_system_before_budget_or_network(tmp_path, decision_mode):
    c, sdk = client(tmp_path, [], request_profile="sdk-defaults-v1", thinking_mode="unspecified",
        decision_mode=decision_mode, compatibility_mode="modelbest-v2")
    messages = [{"role": "system", "content": "First"}, {"role": "user", "content": "Question"},
                {"role": "system", "content": "Do not move or discard this message"}]
    original = copy.deepcopy(messages)
    with pytest.raises(ProviderError) as caught:
        c.chat(messages, tools=TOOLS)
    assert caught.value.code == "invalid_system_layout"
    assert not sdk.calls and c.usage_summary()["requests"] == 0 and messages == original
    with sqlite3.connect(c.journal.path) as db:
        assert db.execute("SELECT COUNT(*) FROM cloud_requests").fetchone()[0] == 0


def test_real_sdk_v2_wire_omits_unique_items_but_local_validation_still_enforces_it(tmp_path):
    import httpx
    from openai import OpenAI
    requests = []
    schema = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"},
              "uniqueItems": True}}, "required": ["ids"], "additionalProperties": False}
    original = copy.deepcopy(schema)
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if "uniqueItems" in body["response_format"]["json_schema"]["schema"]["properties"]["ids"]:
            return httpx.Response(400, json={"error": {"message": 'Grammar error: Unimplemented keys: ["uniqueItems"]',
                                                       "type": "BadRequestError", "code": 400}})
        raw = answer('{"ids":["same","same"]}')
        raw.update(id="offline-grammar", object="chat.completion", created=0)
        raw["choices"][0]["index"] = 0
        return httpx.Response(200, json=raw)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key="offline-test-only", base_url="https://injected.invalid/v1",
                     http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), request_profile="sdk-defaults-v1",
            thinking_mode="unspecified", compatibility_mode="modelbest-v2",
            ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        metrics = []
        with c.capture_calls(metrics.append), pytest.raises(ProviderError) as caught:
            c.generate_json("Return unique IDs", [], schema, max_attempts=1)
    assert caught.value.code == "invalid_json" and len(requests) == 1 and schema == original
    assert metrics[0]["http_status"] == 200 and metrics[0]["parse_status"] == "invalid"
    assert json.dumps(schema, ensure_ascii=False) in requests[0]["messages"][1]["content"][0]["text"]
    wire = requests[0]["response_format"]["json_schema"]["schema"]
    assert wire["properties"]["ids"] == {"type": "array", "items": {"type": "string"}}


def test_v2_schema_conversion_preserves_instance_values_and_names(tmp_path):
    c, _ = client(tmp_path, [], request_profile="sdk-defaults-v1", thinking_mode="unspecified",
                  compatibility_mode="modelbest-v2")
    schema = {"type": "object", "properties": {"uniqueItems": {"type": "array", "uniqueItems": True}},
              "const": {"uniqueItems": True}, "description": "Keep uniqueItems in the prompt",
              "oneOf": [{"type": "array", "uniqueItems": False, "minItems": 1}]}
    original = copy.deepcopy(schema)
    wire = c._format(schema)["json_schema"]["schema"]
    assert schema == original
    assert wire == {"type": "object", "properties": {"uniqueItems": {"type": "array"}},
                    "const": {"uniqueItems": True}, "description": "Keep uniqueItems in the prompt",
                    "oneOf": [{"type": "array", "minItems": 1}]}


@pytest.mark.parametrize("message", [{"role": "system", "content": "Keep metadata", "name": "policy"},
                                    {"role": "system", "content": [{"type": "text", "text": "Keep structure"}]}])
def test_v2_does_not_drop_unmergeable_system_message_information(tmp_path, message):
    c, sdk = client(tmp_path, [], request_profile="sdk-defaults-v1", thinking_mode="unspecified",
                  decision_mode="json_decision", compatibility_mode="modelbest-v2")
    with pytest.raises(ProviderError) as caught:
        c.chat([message, {"role": "user", "content": "Question"}], tools=TOOLS)
    assert caught.value.code == "invalid_system_layout" and not sdk.calls
    assert c.usage_summary()["requests"] == 0


def test_v2_http_failure_audit_records_selected_compatibility(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(400, content=b'{"error":{"message":"another explicit rejection"}}')],
        request_profile="sdk-defaults-v1", thinking_mode="unspecified", compatibility_mode="modelbest-v2")
    with pytest.raises(ProviderError):
        c.chat([{"role": "user", "content": "Question"}])
    assert _failure_audits(c)[0][1]["compatibility_mode"] == "modelbest-v2"


@pytest.mark.parametrize("options", [{"compatibility_mode": "unknown"}, {"compatibility_mode": "modelbest-v2"}])
def test_unknown_or_fixed_profile_compatibility_override_is_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        client(tmp_path, [], **options)


def test_real_sdk_400_persists_full_redacted_error_body_and_request_identity(tmp_path):
    import httpx
    from openai import OpenAI
    secret = "synthetic-modelbest-key-do-not-save"
    raw = json.dumps({"error": {"message": "Unsupported schema keyword oneOf; sent " + secret,
        "param": "response_format.json_schema.schema", "code": "invalid_json_schema",
        "api_key": secret, "authorization": "Bearer different-private-token"}}, indent=2)
    requests, metrics = [], []
    def respond(request):
        requests.append(request)
        return httpx.Response(400, text=raw, headers={"content-type": "application/json",
            "x-request-id": "req-offline-400", "set-cookie": "private-cookie",
            "authorization": "Bearer " + secret, "x-private-header": "private-header"})
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key=secret, base_url="https://injected.invalid/v1", http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), request_profile="sdk-defaults-v1",
            thinking_mode="unspecified", ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        with c.capture_calls(metrics.append), pytest.raises(ProviderError) as caught:
            c.chat([{"role": "user", "content": "Return a decision"}], tools=TOOLS, request_id="audit-400")
    assert caught.value.code == "http_400" and len(requests) == 1
    status, audit = _failure_audits(c)[0]
    assert status == "failed" and audit["http_status"] == 400 and audit["exception_type"] == "BadRequestError"
    assert audit["body"]["error"]["code"] == "invalid_json_schema"
    assert audit["body"]["error"]["param"] == "response_format.json_schema.schema"
    assert "Unsupported schema keyword oneOf" in audit["raw_response_text"]
    assert "\n" in audit["raw_response_text"]
    assert audit["headers"] == {"content-type": "application/json", "x-request-id": "req-offline-400"}
    assert audit["request_id"] == metrics[0]["request_id"] and audit["payload_hash"] == metrics[0]["payload_hash"]
    assert audit["request_profile"] == "sdk-defaults-v1" and audit["thinking_mode"] == "unspecified"
    assert audit["attempt"] == 1 and audit["provider"] == "modelbest"
    assert metrics[0]["error_audit"]["body_sha256"] == audit["body_sha256"]
    assert "raw_response_text" not in metrics[0]["error_audit"] and "body" not in metrics[0]["error_audit"]
    saved = json.dumps([audit, metrics])
    for sensitive in (secret, "different-private-token", "private-cookie", "private-header"):
        assert sensitive not in saved
    assert "[REDACTED]" in saved


@pytest.mark.parametrize("status,content", [(429, '{"error":{"message":"slow down"}}'), (404, "")])
def test_real_sdk_retry_keeps_first_rejected_response_audit(tmp_path, status, content):
    import httpx
    from openai import OpenAI
    responses = [httpx.Response(status, text=content, headers={"x-request-id": "first-rejection"})]
    raw = answer()
    raw.update(id="offline-success", object="chat.completion", created=0)
    raw["choices"][0]["index"] = 0
    responses.append(httpx.Response(200, json=raw))
    def respond(request):
        return responses.pop(0)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key="synthetic", base_url="https://injected.invalid/v1", http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), sleep=lambda _: None,
                           ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        c.chat([{"role": "user", "content": "hello"}])
    rejected, audit = _failure_audits(c)[0]
    assert rejected == "failed" and audit["http_status"] == status and audit["attempt"] == 1
    assert audit["raw_response_text"] == content and audit["headers"]["x-request-id"] == "first-rejection"
    assert not responses and c.usage_summary()["requests"] == 2


def test_real_sdk_html_error_keeps_text_and_redacts_key_echo(tmp_path):
    import httpx
    from openai import OpenAI
    secret = "synthetic-html-secret"
    def respond(request):
        return httpx.Response(500, text="<html>upstream failure; Authorization: Bearer " + secret + "</html>",
                              headers={"content-type": "text/html", "request-id": "html-failure"})
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key=secret, base_url="https://injected.invalid/v1", http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        with pytest.raises(RequestUncertain):
            c.chat([{"role": "user", "content": "hello"}])
    status, audit = _failure_audits(c)[0]
    assert status == "uncertain" and audit["http_status"] == 500
    assert audit["exception_type"] == "InternalServerError" and audit["headers"]["content-type"] == "text/html"
    assert audit["raw_response_text"].startswith("<html>upstream failure;")
    assert secret not in audit["raw_response_text"] and "[REDACTED]" in audit["raw_response_text"]
    assert c.usage_summary()["blocking_uncertain_requests"] == 1


def test_real_sdk_timeout_audit_is_retained_and_never_retried(tmp_path):
    import httpx
    from openai import OpenAI
    requests = []
    def respond(request):
        requests.append(request)
        raise httpx.ReadTimeout("synthetic timed-out response", request=request)
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = OpenAI(api_key="synthetic", base_url="https://injected.invalid/v1", http_client=http_client, max_retries=0)
        c = ModelBestClient(sdk=sdk, pricing_evidence=evidence(), request_profile="sdk-defaults-v1",
            thinking_mode="unspecified", ledger=BudgetLedger(tmp_path / "budget.sqlite", limit_cny=600))
        for _ in range(2):
            with pytest.raises(RequestUncertain):
                c.chat([{"role": "user", "content": "hello"}], request_id="timeout")
    status, audit = _failure_audits(c)[0]
    assert status == "uncertain" and audit["exception_type"] == "APITimeoutError"
    assert audit["http_status"] is None and audit["headers"] == {}
    assert audit["raw_response_text"] is audit["body"] is None and audit["body_source"] == "unavailable"
    assert len(requests) == 1


def test_error_redaction_preserves_nonsecret_raw_text_and_masks_nested_credentials():
    from camroll_agent.modelbest import _redact_error_text, _redact_error_value
    raw = '{\n  "error": {"message": "Unsupported oneOf", "param": "response_format", "api_key": {"value": "opaque-secret"}, "X-Api-Key": "opaque-second"}\n}'
    sanitized = _redact_error_text(raw, ())
    assert json.loads(sanitized) == _redact_error_value(json.loads(raw), ())
    assert sanitized.startswith('{\n  "error":') and "Unsupported oneOf" in sanitized
    assert "opaque" not in sanitized
    assert "opaque-third" not in _redact_error_text('<html>api_key=opaque-third; Authorization: Basic dXNlcjpwYXNz</html>', ())
    assert "dXNlcjpwYXNz" not in _redact_error_text('Authorization: Basic dXNlcjpwYXNz', ())


def test_optional_journal_failure_details_preserve_old_calls_and_completed_responses(tmp_path):
    journal = RequestJournal(tmp_path / "journal.sqlite")
    journal.claim("historical-failed", "hash-old", lambda: "old-call")
    journal.fail("historical-failed", "http_400")
    with sqlite3.connect(journal.path) as db:
        old = db.execute("SELECT * FROM cloud_requests WHERE request_id='historical-failed'").fetchone()
    journal.fail("historical-failed", "http_401", failure_details={"body": "must not replace history"})
    journal.claim("completed", "hash-completed", lambda: "complete-call")
    response = {"model": MODEL, "usage": {"prompt_tokens": 1}}
    journal.complete("completed", response)
    journal.fail("completed", "usage_missing", uncertain=True, failure_details={"body": "do not replace raw response"})
    journal.claim("new-failed", "hash-new", lambda: "new-call")
    journal.fail("new-failed", "http_400", failure_details={"body": "new evidence"})
    with sqlite3.connect(journal.path) as db:
        assert db.execute("SELECT * FROM cloud_requests WHERE request_id='historical-failed'").fetchone() == old
        assert json.loads(db.execute("SELECT response FROM cloud_requests WHERE request_id='completed'").fetchone()[0]) == response
        assert json.loads(db.execute("SELECT response FROM cloud_requests WHERE request_id='new-failed'").fetchone()[0]) == {"_failure": {"body": "new evidence"}}


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
@pytest.mark.parametrize("error", [HTTPError(429), HTTPError(404, content=b"")])
def test_only_429_and_empty_404_retry_once(tmp_path, error, options):
    from camroll_eval.minicpm_experiment import minicpm_infrastructure_failure
    c, sdk = client(tmp_path, [error, answer()], **options)
    observed = []
    with c.request_scope() as scope, c.capture_calls(observed.append):
        assert c.chat([{"role": "user", "content": "hi"}])["content"]
    assert len(sdk.calls) == scope.summary()["requests"] == 2
    assert c.usage_summary()["requests"] == 2
    assert observed[0]["http_status"] == error.status_code
    assert minicpm_infrastructure_failure({"model_calls": observed}) is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_explicit_rejections_do_not_retry_or_leak_errors(tmp_path, status):
    c, sdk = client(tmp_path, [HTTPError(status), answer()])
    with pytest.raises(ProviderError) as caught:
        c.chat([{"role": "user", "content": "hi"}])
    assert "private" not in str(caught.value) and len(sdk.calls) == 1
    assert c.usage_summary()["uncertain_requests"] == 0


def test_retry_bound(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(429), HTTPError(429), answer()])
    with pytest.raises(ProviderError):
        c.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 2


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
@pytest.mark.parametrize("outcome", [HTTPError(500), TimeoutError(), answer(usage=False)])
def test_zero_reserved_unknown_still_blocks_new_submissions_and_replay(tmp_path, outcome, options):
    c, sdk = client(tmp_path, [outcome, answer()], **options)
    with pytest.raises(RequestUncertain):
        c.chat([{"role": "user", "content": "hi"}], request_id="first")
    status = c.usage_summary()
    assert status["reserved_cny"] == 0 and status["blocking_uncertain_requests"] == 1
    for request_id in ("first", "new"):
        with pytest.raises(RequestUncertain):
            c.chat([{"role": "user", "content": "hi"}], request_id=request_id)
    assert len(sdk.calls) == 1


def test_request_replay_does_not_repeat_or_double_count(tmp_path):
    c, sdk = client(tmp_path, [answer()])
    for _ in range(2):
        result = c.chat([{"role": "user", "content": "hi"}], request_id="same")
    assert result["_usage"]["replayed"] and len(sdk.calls) == 1
    assert c.usage_summary()["usage"]["total_tokens"] == 150


def test_sdk_defaults_cannot_replay_a_fixed_profile_request_identity(tmp_path):
    fixed, sdk = client(tmp_path, [answer()], request_namespace="shared-identity-fixture")
    fixed.chat([{"role": "user", "content": "hi"}], request_id="same")
    defaults, unused = client(tmp_path, [], request_namespace="shared-identity-fixture",
                              request_profile="sdk-defaults-v1", thinking_mode="unspecified")
    with pytest.raises(ProviderError) as caught:
        defaults.chat([{"role": "user", "content": "hi"}], request_id="same")
    assert caught.value.code == "request_identity_mismatch" and not unused.calls
    assert len(sdk.calls) == 1 and defaults.usage_summary()["requests"] == 1


def test_server_context_error_is_explicit_and_not_unknown(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(400, body={"error": {"code": "context_length_exceeded"}})])
    with pytest.raises(ProviderError) as caught:
        c.chat([{"role": "user", "content": "hi"}])
    assert caught.value.code == "context_limit" and c.usage_summary()["uncertain_requests"] == 0


def test_no_qwen_utf8_byte_context_rejection(tmp_path):
    c, sdk = client(tmp_path, [answer()])
    c.chat([{"role": "user", "content": "长" * 100000}])
    assert len(sdk.calls) == 1


@pytest.mark.parametrize("mode,format_type", [("json_schema", "json_schema"), ("json_object", "json_object"), ("prompt", None)])
def test_fixed_json_modes(tmp_path, mode, format_type):
    c, sdk = client(tmp_path, [answer()], json_mode=mode)
    assert c.generate_json("Describe", [], SCHEMA) == {"caption": "一只猫"}
    assert sdk.calls[0].get("response_format", {}).get("type") == format_type


def test_json_mode_does_not_fallback_after_schema_rejection(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(400, "json_schema not supported"), answer()])
    with pytest.raises(ProviderError):
        c.generate_json("Describe", [], SCHEMA)
    assert len(sdk.calls) == 1


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
@pytest.mark.parametrize("content", ['```json\n{"caption":"cat"}\n```', 'analysis {"caption":"cat"}', '{"caption":NaN}'])
def test_json_whole_content_validation_and_one_final_attempt(tmp_path, content, options):
    c, sdk = client(tmp_path, [answer(content), answer()], **options)
    with pytest.raises(ProviderError) as caught:
        c.generate_json("Describe", [], SCHEMA, role="final", max_attempts=1)
    assert caught.value.code == "invalid_json" and len(sdk.calls) == 1


def test_existing_three_attempt_json_repair_retained(tmp_path):
    c, sdk = client(tmp_path, [answer("bad"), answer('{"caption":3}'), answer()])
    observed = []
    with c.capture_calls(observed.append):
        assert c.generate_json("Describe", [], SCHEMA)["caption"]
    assert len(sdk.calls) == 3
    assert len(observed) == 3
    assert [row["parse_status"] for row in observed] == ["invalid", "invalid", "valid"]
    assert all("reasoning" not in json.dumps(row["messages"]) for row in sdk.calls)


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
def test_native_tools_return_unchanged_for_agent_validation(tmp_path, options):
    calls = [{"id": "call1", "type": "function", "function": {"name": "search", "arguments": '{"query":"cat"}'}}]
    c, sdk = client(tmp_path, [answer(None, tool_calls=calls)], **options)
    result = c.chat([{"role": "user", "content": "find cat"}], tools=TOOLS)
    assert sdk.calls[0]["tools"] == TOOLS and _assistant_envelope(result)["tool_calls"] == calls


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
def test_json_decision_roundtrip_uses_identical_tool_validation(tmp_path, options):
    c, sdk = client(tmp_path, [answer('{"tool_calls":[{"name":"search","arguments":{"query":"cat"}}]}'),
                             answer('{"answer":"A cat","outcome":"answer","selected_image_ids":["img_1"]}')],
                    decision_mode="json_decision", json_mode="prompt", **options)
    history = [{"role": "user", "content": "find cat"}]
    first = _assistant_envelope(c.chat(history, tools=TOOLS))
    call = first["tool_calls"][0]
    assert validate_arguments(call["function"]["name"], json.loads(call["function"]["arguments"])) == {"query": "cat"}
    history.extend([first, {"role": "tool", "tool_call_id": call["id"], "content": '{"image_ids":["img_1"]}'}])
    final = c.chat(history, tools=TOOLS)
    assert json.loads(final["content"])["selected_image_ids"] == ["img_1"]
    assert all("tools" not in row for row in sdk.calls)
    assert not any(m["role"] == "tool" for m in sdk.calls[1]["messages"])


def test_json_decision_cannot_bypass_tool_argument_validation(tmp_path):
    c, sdk = client(tmp_path, [answer('{"tool_calls":[{"name":"get","arguments":{"id":"/etc/passwd"}}]}')],
                    decision_mode="json_decision")
    call = c.chat([{"role": "user", "content": "get"}], tools=TOOLS)["tool_calls"][0]
    with pytest.raises(ToolValidationError):
        validate_arguments(call["function"]["name"], json.loads(call["function"]["arguments"]))


@pytest.mark.parametrize("options", [{}, {"request_profile": "sdk-defaults-v1", "thinking_mode": "unspecified"}])
@pytest.mark.parametrize("actual_model", ["other", None])
def test_model_mismatch_or_absence_has_unknown_cost_and_blocks_despite_zero_reserve(tmp_path, options, actual_model):
    raw = answer(model=actual_model)
    if actual_model is None:
        raw.pop("model")
    c, sdk = client(tmp_path, [raw, answer()], **options)
    metrics = []
    with c.request_scope() as scope, c.capture_calls(metrics.append):
        with pytest.raises(RequestUncertain):
            c.chat([{"role": "user", "content": "hi"}], request_id="mismatch")
    assert len(metrics) == 1 and metrics[0]["error_code"] == "response_model_mismatch"
    assert metrics[0]["status"] == "uncertain" and metrics[0]["estimated_cny"] is None
    assert metrics[0]["completion_tokens"] == 50 and "output_rate" not in metrics[0]
    status = c.usage_summary()
    assert status["reserved_cny"] == 0 and status["blocking_uncertain_requests"] == 1
    assert scope.summary()["requests"] == 1
    with sqlite3.connect(c.journal.path) as db:
        row = db.execute("SELECT status,error_code,response FROM cloud_requests").fetchone()
    assert row[:2] == ("uncertain", "response_model_mismatch") and json.loads(row[2]) == raw
    with sqlite3.connect(c.ledger.path) as db:
        row = db.execute("SELECT status,metadata FROM budget_calls").fetchone()
    assert row[0] == "uncertain"
    metadata = json.loads(row[1])
    assert metadata["error_code"] == "response_model_mismatch" and metadata["completion_tokens"] == 50
    assert "output_rate" not in metadata
    for request_id in ("mismatch", "new"):
        with pytest.raises(RequestUncertain):
            c.chat([{"role": "user", "content": "hi"}], request_id=request_id)
    assert len(sdk.calls) == 1


def test_truncation_never_becomes_final_answer(tmp_path):
    c, _ = client(tmp_path, [answer(finish="length")])
    with pytest.raises(ProviderError) as caught:
        c.chat([{"role": "user", "content": "hi"}])
    assert caught.value.code == "invalid_message"


def test_request_scope_counts_retry(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(429), answer()])
    with c.request_scope(max_requests=1):
        with pytest.raises(RequestLimitExceeded):
            c.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 1


def test_explicit_agent_binding_preserves_snapshot_provenance(tmp_path):
    c, _ = client(tmp_path, [])
    library = SimpleNamespace(config="M4", experiment=SimpleNamespace(mode="M4", deadline_s=600,
                              generation_model="qwen3.8-max"))
    agent = Agent(str(tmp_path), llm=c, vlm=c, library=library, config="M4", inference_model=MODEL)
    assert agent.inference_model == MODEL and library.experiment.generation_model == "qwen3.8-max"
    with pytest.raises(ValueError):
        Agent(str(tmp_path), library=library, config="M4", inference_model=MODEL)
    with pytest.raises(ValueError):
        Agent(str(tmp_path), llm=SimpleNamespace(model=MODEL), library=library, config="M4", inference_model=MODEL)
