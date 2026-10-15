"""Self-hosted integration contracts: no network or credentials."""
import json
import pytest

from test_modelbest import SDK, answer, HTTPError, SCHEMA
from camroll_agent.budget import BudgetLedger, RequestUncertain
from camroll_agent.minicpm_selfhost import MiniCPMSelfHostedClient, MODEL, PROTOCOL
from camroll_agent.qwen import ProviderError
from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig, minicpm_infrastructure_failure


def response(**kwargs):
    result = answer(model=MODEL, **kwargs)
    message = result['choices'][0]['message']
    message['reasoning_content'] = message.pop('reasoning')
    return result


def client(tmp_path, responses, **kwargs):
    sdk = SDK(responses)
    c = MiniCPMSelfHostedClient(sdk=sdk, ledger=BudgetLedger(tmp_path / 'budget.sqlite', limit_cny=600),
        base_url='http://127.0.0.1:18080/v1', pricing_evidence={
            'model': MODEL, 'provider': 'llamacpp-selfhost', 'input_rate': 0, 'output_rate': 0,
            'cost_scope': 'api_only', 'infrastructure_cost_cny': None,
            'observed_at': '2026-10-15T00:00:00Z', 'deployment_manifest_sha256': 'a' * 64},
        **kwargs)
    return c, sdk


def test_selfhost_wire_usage_reasoning_and_replay(tmp_path):
    c, sdk = client(tmp_path, [response()])
    profile = MiniCPMInferenceConfig.selfhost()
    with c.request_scope(deadline_s=profile.deadline_s):
        result = c.chat([{'role': 'user', 'content': 'hello'}], request_id='same')
    assert profile.matches_request(result['_metric']['input'])
    assert sdk.calls[0]['timeout'] == 600
    assert 'extra_body' not in sdk.calls[0]
    assert result['_metric']['reasoning_chars'] > 0
    assert 'Private model reasoning' not in json.dumps(result)
    assert c.chat([{'role': 'user', 'content': 'hello'}], request_id='same')['_usage']['replayed']
    assert len(sdk.calls) == 1
    assert c.capabilities()['context_token_bound'] == 262144
    assert profile.deadline_s == 1200 and profile.protocol_version == PROTOCOL
    assert MiniCPMInferenceConfig(**profile.as_dict()) == profile


@pytest.mark.parametrize('outcome', [TimeoutError(), response(usage=False)])
def test_zero_price_unknown_blocks_new_requests(tmp_path, outcome):
    c, sdk = client(tmp_path, [outcome, response()])
    with pytest.raises(RequestUncertain):
        c.chat([{'role': 'user', 'content': 'hello'}])
    with pytest.raises(Exception):
        c.chat([{'role': 'user', 'content': 'next'}])
    assert len(sdk.calls) == 1
    assert c.ledger.summary()['uncertain_requests'] == 1


def test_llamacpp_explicit_context_rejection_is_known_turn_failure(tmp_path):
    c, sdk = client(tmp_path, [HTTPError(400, body={'type': 'exceed_context_size_error'})])
    with pytest.raises(ProviderError) as exc:
        c.chat([{'role': 'user', 'content': 'long'}])
    assert exc.value.code == 'context_limit'
    assert not c.ledger.summary()['uncertain_requests']
    assert minicpm_infrastructure_failure({'error_code': 'context_limit', 'http_status': 400}) is None


@pytest.mark.parametrize('content', ['{"caption":"a","caption":"b"}', '```json\n{"caption":"a"}\n```', '{"caption":NaN}'])
def test_complete_strict_json_without_repair_disguise(tmp_path, content):
    c, sdk = client(tmp_path, [response(content=content)])
    with pytest.raises(ProviderError):
        c.generate_json('describe', [], SCHEMA, max_attempts=1)
    assert len(sdk.calls) == 1


def test_json_decision_ids_and_final(tmp_path):
    from camroll_agent.schemas import TOOLS
    c, sdk = client(tmp_path, [response(content='{"tool_calls":[{"name":"get","arguments":{"id":"img_test"}}]}')], decision_mode='json_decision')
    result = c.chat([{'role': 'user', 'content': 'get'}], tools=TOOLS)
    assert result['tool_calls'][0]['function']['name'] == 'get'
    assert result['tool_calls'][0]['id']
    assert 'tools' not in sdk.calls[0]


def test_selfhost_agent_binding_and_extended_deadline(tmp_path):
    from types import SimpleNamespace
    from camroll_agent.agent import Agent
    c, _ = client(tmp_path, [])
    library = SimpleNamespace(config='M4', experiment=SimpleNamespace(mode='M4', deadline_s=600, generation_model='qwen3.8-max'))
    a = Agent(str(tmp_path), llm=c, vlm=c, library=library, config='M4', inference_model=MODEL, deadline_s=1200)
    assert a.deadline_s == 1200
    with pytest.raises(ValueError):
        Agent(str(tmp_path), llm=c, library=library, config='M4', inference_model=MODEL, deadline_s=1201)
