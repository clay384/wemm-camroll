import json
import contextlib
from types import SimpleNamespace
import pytest
import minicpm_probe as probe
from minicpm_probe import probe_roundtrip


class Client:
    def __init__(self, *, call_id='call1', call_type='function', target='img_fixture', final_target='img_fixture', answer='A visible stone pillar.'):
        self.calls = []
        self.tool = {'id': call_id, 'type': call_type, 'function': {'name': 'get', 'arguments': json.dumps({'id': target})}}
        self.final = {'answer': answer, 'outcome': 'answer', 'selected_image_ids': [final_target]}

    def chat(self, messages, **kwargs):
        self.calls.append(messages.copy())
        if len(self.calls) == 1:
            return {'role': 'assistant', 'content': None, 'tool_calls': [self.tool], 'reasoning': 'must not enter transcript'}
        return {'role': 'assistant', 'content': json.dumps(self.final)}


def test_complete_roundtrip_uses_same_call_id_and_no_reasoning():
    client = Client()
    result = probe_roundtrip(client, {'id': 'img_fixture', 'kind': 'image', 'caption': 'A stone pillar'})
    assert result['final']['selected_image_ids'] == ['img_fixture']
    assert client.calls[1][-1]['tool_call_id'] == 'call1'
    assert 'reasoning' not in client.calls[1][-2]


@pytest.mark.parametrize('options', [
    {'call_id': 42}, {'call_type': 'invalid'}, {'target': 'img_unknown'},
    {'final_target': 'img_unknown'}, {'answer': 'See img_unknown.'},
])
def test_probe_rejects_native_protocol_or_delivery_violation(options):
    with pytest.raises(ValueError):
        probe_roundtrip(Client(**options), {'id': 'img_fixture', 'kind': 'image'})


def test_final_schema_echo_is_an_interface_error_even_when_facts_are_observational():
    from camroll_agent.agent import FINAL_SCHEMA
    client = Client()
    client.final = FINAL_SCHEMA
    with pytest.raises(ValueError):
        probe_roundtrip(client, {'id': 'img_fixture', 'kind': 'image'})


@pytest.mark.parametrize('color', ['红色', '紅色', 'red'])
def test_new_probe_annotation_preserves_original_glyph_and_color_synonyms(color):
    probe.validate_inscription({'inscription': '三峡壩址基石', 'letter_color': color}, sdk_defaults=True)
    with pytest.raises(ValueError):
        probe.validate_inscription({'inscription': '三峡坝址基石', 'letter_color': color}, sdk_defaults=True)
    probe.validate_inscription({'inscription': 'structural-only legacy'}, sdk_defaults=False)


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('policy,correct', [('facts-and-interface-v1', True),
    ('facts-and-interface-v1', False), ('interface-only-v2', False)])
def test_sdk_probe_runs_only_selected_mode_once_and_freezes_entire_profile(tmp_path, monkeypatch, fails, policy, correct):
    import minicpm_config as config
    import camroll_agent.budget as budget
    import camroll_agent.modelbest as modelbest
    import camroll_agent.library as library_module
    import four_suite
    import gate_continuation
    from camroll_eval.common import atomic_json, read_json, sha256
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    runtime = tmp_path / 'minicpm47-sdk-defaults-m4-off-probe'
    runtime.mkdir()
    profile = MiniCPMInferenceConfig.sdk_defaults(decision_mode='native_tools', json_mode='json_schema', acceptance_policy=policy)
    atomic_json(runtime / 'request-profile.json', profile.as_dict())
    atomic_json(runtime / 'pricing-evidence.json', {'source': 'synthetic'})
    fixtures = [{'id': 'img_' + str(index)} for index in range(7)]
    atomic_json(runtime / 'acceptance/fixture-records.json', fixtures)
    image = tmp_path / 'synthetic-image'
    image.write_bytes(b'synthetic fixture; not a private photo')
    atomic_json(runtime / 'acceptance/annotation-review-sdk-defaults.json', {
        'schema_version': 'minicpm-sdk-defaults-private-annotation-review-v1',
        'new_protocol': profile.protocol_version, 'reviewer': 'synthetic test',
        'formal_qa_gt_accessed': False, 'original_fixtures_modified': False,
        'fixtures': [{'index': index, 'sha256': sha256(image)} for index in range(7)]})
    monkeypatch.setattr(config, 'configure', lambda *args: None)
    monkeypatch.setattr(probe, 'initialize_acknowledgement', lambda *args: object())
    monkeypatch.setattr(budget, 'BudgetLedger', lambda *args, **kwargs: SimpleNamespace(summary=lambda: {}))
    monkeypatch.setattr(gate_continuation, 'ledger_ready', lambda *args: None)
    monkeypatch.setattr(four_suite, 'source_manifest', lambda *args: {'fixture': 'sha'})
    monkeypatch.setattr(library_module, 'Library', lambda *args, **kwargs: SimpleNamespace(version=config.SNAPSHOT,
        image_paths=lambda ids: [str(image)] * len(ids), get=lambda iid: fixtures[6]))
    clients, actions = [], []
    class ProbeClient:
        def __init__(self, **kwargs):
            clients.append(kwargs)
        @contextlib.contextmanager
        def capture_calls(self, callback):
            self.callback = callback
            yield
        @contextlib.contextmanager
        def request_scope(self, **kwargs):
            assert kwargs == {'max_requests': 50, 'deadline_s': 600}
            yield
        def generate_json(self, *args, **kwargs):
            assert kwargs['max_attempts'] == 1
            actions.append('ocr')
            self.callback({'model': profile.generation_model, 'input': {'model': profile.generation_model, 'messages': []}})
            if fails:
                raise ValueError('single_integration_failure')
            return {'inscription': '三峡壩址基石' if correct else '错误答案', 'letter_color': '紅色' if correct else '绿色'}
    def roundtrip(client, record):
        actions.append('tools')
        client.callback({'model': profile.generation_model, 'input': {'model': profile.generation_model, 'messages': []}})
        return {'final': 'synthetic'}
    monkeypatch.setattr(modelbest, 'ModelBestClient', ProbeClient)
    monkeypatch.setattr(probe, 'probe_roundtrip', roundtrip)
    if fails or (not correct and policy == 'facts-and-interface-v1'):
        with pytest.raises(RuntimeError, match='no_structured'):
            probe.execute(runtime)
        assert actions == ['ocr'] and len(clients) == 1
        assert not (runtime / 'acceptance/protocol.json').exists()
    else:
        result = probe.execute(runtime)
        assert actions == ['ocr', 'tools'] and len(clients) == 2
        frozen = read_json(runtime / 'acceptance/protocol.json')
        assert frozen['inference_config'] == profile.as_dict()
        assert frozen['protocol_fingerprint'] == profile.fingerprint
        assert result['purpose'] == 'single_selected_mode_client_integration'
        assert result['fact_observations'][0]['inscription_correct'] is correct
        assert result['fact_observations'][0]['blocks_acceptance'] is (policy != 'interface-only-v2')
    assert all(all(client[key] == value for key, value in profile.client_options().items()) for client in clients)
    with pytest.raises(FileExistsError):
        probe.execute(runtime)
