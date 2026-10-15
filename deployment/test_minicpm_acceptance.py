"""No network: fixed acceptance matrix, exact JSON, assets, and stop behavior."""
import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import minicpm_acceptance as acceptance
from camroll_eval.common import atomic_json, read_json
from camroll_agent.budget import RequestUncertain
from camroll_agent.modelbest import MODEL


def test_fixed_matrix_is_ten_classes_times_three_without_sampling():
    rows = acceptance.planned_cases()
    assert len(rows) == len({row['case_id'] for row in rows}) == 30
    assert all(sum(row['class'] == name for row in rows) == 3 for name in acceptance.CLASSES)
    assert all(row['status'] == 'not_started' for row in rows)


@pytest.mark.parametrize('value', ['```json\n{"text":"a"}\n```', 'prefix {"text":"a"}',
                                  '{"text":"a"} trailing', '{"text":NaN}', '{"text":"a","extra":1}'])
def test_whole_json_and_schema_are_required(value):
    assert not acceptance.strict_valid(value, acceptance.SCHEMAS['ocr'])
    assert acceptance.strict_valid('{"text":"三峡坝址基石"}', acceptance.SCHEMAS['ocr'])


def test_long_history_matches_counts_and_byte_floor_without_benchmark_content():
    for repetition, (byte_count, message_count) in enumerate(acceptance.LONG_SIZES):
        messages, metadata = acceptance.synthetic_history(repetition)
        assert len(messages) == message_count and metadata['actual_serialized_bytes'] >= byte_count
        assert len(json.dumps(messages, ensure_ascii=False).encode()) == metadata['actual_serialized_bytes']
        assert any(row['role'] == 'tool' for row in messages)
        assert 'L47-ACCEPT' in messages[2]['content'] and metadata['truncated'] is False


def test_ocr_annotation_is_exact_and_final_image_ids_are_checked():
    ids = ['img_' + str(index) for index in range(7)]
    assert acceptance.fact_checks('ocr', {'text': '三峡坝址基石'}, ids) == [True]
    assert acceptance.fact_checks('ocr', {'text': '三峡坝址基址'}, ids) == [False]
    assert acceptance.fact_checks('final_schema', {'answer': '三峡坝址基石', 'selected_image_ids': [ids[6]]}, ids) == [True, True]
    assert acceptance.fact_checks('final_schema', {'answer': '三峡坝址基石', 'selected_image_ids': [ids[0]]}, ids) == [True, False]


def test_duplicate_missing_unknown_or_undelivered_ids_are_hard_failures():
    ids = ['img_' + str(index) for index in range(7)]
    valid = {'images': [{'image_id': iid, 'description': 'photo', 'visible_people': 1} for iid in ids[:6]]}
    acceptance.validate_delivery('six_images', valid, ids)
    duplicate = copy.deepcopy(valid)
    duplicate['images'][-1]['image_id'] = ids[0]
    with pytest.raises(ValueError):
        acceptance.validate_delivery('six_images', duplicate, ids)
    for output in ({'answer': 'stone', 'selected_image_ids': [ids[0]]},
                   {'answer': 'img_unknown stone', 'selected_image_ids': [ids[6]]},
                   {'answer': ids[0], 'selected_image_ids': [ids[6]]}):
        with pytest.raises(ValueError):
            acceptance.validate_delivery('final_schema', output, ids)
    acceptance.validate_delivery('long_history', {'answer': 'L47-ACCEPT', 'selected_image_ids': []}, ids)


def test_identity_flags_never_replace_independent_review():
    library = SimpleNamespace(metadata={'people_registry': [{'name': '测试甲', 'aliases': ['小甲']}]})
    result = acceptance.identity_review({'answer': '测试甲和母亲一起站立'}, library)
    assert result['manual_review_required'] is True and result['reviewed'] is False
    assert 'registered_identity_mention:测试甲' in result['auto_flags']
    assert 'relationship_language:母亲' in result['auto_flags']
    assert acceptance.identity_review({'answer': '一名人物站立'}, library)['manual_review_required'] is True


def clean_stage(row):
    return {**row, 'status': 'completed', 'fact_checks': [True],
        'structured_units': [{'attempts': [{'valid': True, 'finish_reason': 'stop'}]}],
        'physical_calls': [{'attempt': 1, 'status': 'success'}]}


def test_thresholds_use_all_thirty_cases_and_preserve_retry_failure():
    rows = [clean_stage(row) for row in acceptance.planned_cases()]
    assert acceptance.passed(acceptance.metrics(rows))
    rows[0]['status'] = 'failed'
    assert not acceptance.passed(acceptance.metrics(rows))
    rows[0]['status'] = 'completed'
    rows[0]['physical_calls'] = [{'attempt': 1, 'status': 'error'}, {'attempt': 2, 'status': 'success'}]
    assert acceptance.metrics(rows)['first_http_success_rate'] == pytest.approx(29 / 30)
    rows[1]['physical_calls'][0]['status'] = 'error'
    assert not acceptance.passed(acceptance.metrics(rows))
    rows[1]['physical_calls'][0]['status'] = 'success'
    rows[2]['structured_units'][0]['attempts'][0]['finish_reason'] = 'length'
    assert not acceptance.passed(acceptance.metrics(rows))


def test_interface_only_policy_records_wrong_facts_but_preserves_all_interface_gates():
    rows = [clean_stage(row) for row in acceptance.planned_cases()]
    for row in rows:
        row['fact_checks'] = [False]
    value = acceptance.metrics(rows)
    assert value['fact_check_rate'] == 0
    assert not acceptance.passed(value)
    assert acceptance.passed(value, acceptance.INTERFACE_ONLY_POLICY)
    for field, bad in [('completed_cases', 29), ('final_structured_rate', .99),
                       ('first_structured_rate', .94), ('first_structured_case_rate', .94),
                       ('first_http_success_rate', .94), ('system_failure_count', 1), ('length_finish_count', 1)]:
        assert not acceptance.passed({**value, field: bad}, acceptance.INTERFACE_ONLY_POLICY)
    assert not acceptance.passed(value, 'unversioned-ignore-errors')


def test_http_200_empty_content_counts_http_success_but_fails_structured_gate():
    rows = [clean_stage(row) for row in acceptance.planned_cases()]
    rows[0].update(status='failed', structured_units=[{'attempts': [{'valid': False}]}],
        physical_calls=[{'attempt': 1, 'status': 'error', 'error_code': 'empty_response', 'http_status': 200}])
    value = acceptance.metrics(rows)
    assert value['first_http_success_count'] == 30 and value['first_http_success_rate'] == 1
    assert value['final_structured_rate'] == pytest.approx(29 / 30)
    assert value['first_structured_rate'] == pytest.approx(29 / 30)
    assert not acceptance.passed(value)
    rows[0]['physical_calls'][0].update(status='success', http_status=429)
    assert acceptance.metrics(rows)['first_http_success_count'] == 29


def test_crops_use_first_eligible_saved_face_and_are_request_local(tmp_path):
    from PIL import Image
    ids, paths = ['img_' + str(index) for index in range(6)], []
    for index in range(6):
        path = tmp_path / (str(index) + '.jpg')
        Image.new('RGB', (20, 20), 'white').save(path)
        paths.append(str(path))
    calls = []
    class Library:
        images = {iid: {'face_instances': [{'face_id': 'face_no', 'eligible': False},
                                            {'face_id': 'face_' + iid, 'eligible': True}]} for iid in ids}
        def image_paths(self, values):
            assert values == ids
            return paths
        def target_regions(self, values, face_id=None):
            calls.append((values, face_id))
            return [{'image_id': values[0], 'face_id': face_id, 'crop_bbox': [1, 2, 10, 13]}]
    with acceptance.twelve_paths(Library(), ids) as (inputs, regions):
        assert len(inputs) == 12 and inputs[:6] == paths
        assert all(Path(path).is_file() for path in inputs)
        crops = inputs[6:]
        assert len(regions) == len(calls) == 6
        assert all(face_id == 'face_' + values[0] for values, face_id in calls)
    assert not any(Path(path).exists() for path in crops)
    assert all(Path(path).exists() for path in paths)


def test_harness_records_invalid_first_json_before_existing_repair():
    class Client:
        outcomes = iter([{'content': '```json\n{"text":"x"}\n```'}, {'content': '{"text":"x"}'}])
        def chat(self, *args, **kwargs):
            return next(self.outcomes)
        def generate_json(self, *args, **kwargs):
            first = self.chat([])
            assert not acceptance.strict_valid(first['content'], acceptance.SCHEMAS['ocr'])
            return json.loads(self.chat([])['content'])
    client = Client()
    original = client.chat
    row = {'structured_units': []}
    with acceptance.Harness(client, row, lambda: None) as harness:
        assert harness.json('read', [], acceptance.SCHEMAS['ocr']) == {'text': 'x'}
    assert client.chat == original
    assert [value['valid'] for value in row['structured_units'][0]['attempts']] == [False, True]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import minicpm_suite
    import four_suite
    import camroll_eval.gate_experiment as gates
    root = tmp_path / 'runtime'
    protocol = {'decision_mode': 'native_tools', 'json_mode': 'prompt', 'pricing_evidence': {'source': 'synthetic'}}
    atomic_json(root / 'acceptance/protocol.json', protocol)
    atomic_json(root / 'pricing-evidence.json', protocol['pricing_evidence'])
    fixtures = [{'id': 'img_' + str(index)} for index in range(7)]
    atomic_json(root / 'acceptance/fixture-records.json', fixtures)
    monkeypatch.setattr(four_suite, 'source_manifest', lambda release: {'file': 'frozen'})
    monkeypatch.setattr(minicpm_suite, 'ledger_ready', lambda ledger: {'limit_cny': 600, 'uncertain_requests': 1, 'remaining_cny': 200})
    binding = SimpleNamespace(as_dict=lambda: {'snapshot_version': acceptance.SNAPSHOT},
                              validate=lambda *args: None, experiment_config={})
    monkeypatch.setattr(gates.IndexBinding, 'capture', lambda *args: binding)
    library = SimpleNamespace(version=acceptance.SNAPSHOT, get=lambda iid: next(row for row in fixtures if row['id'] == iid),
                              image_paths=lambda ids: ['/synthetic/' + iid for iid in ids])
    class Client:
        callback = None
        reasoning_tokens = 1
        payload = {'model': MODEL, 'messages': [], 'temperature': 0,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}, 'stream': False, 'max_tokens': 8192}
        @contextlib.contextmanager
        def request_context(self, value):
            yield
        @contextlib.contextmanager
        def request_scope(self, **limits):
            assert limits == {'max_requests': 50, 'deadline_s': 600}
            yield
        @contextlib.contextmanager
        def capture_calls(self, callback):
            self.callback = callback
            yield
            self.callback = None
        def chat(self, *args, **kwargs):
            self.callback({'status': 'success', 'model': MODEL, 'attempt': 1, 'prompt_tokens': 10,
                'completion_tokens': 10, 'reasoning_tokens': self.reasoning_tokens,
                'input': self.payload})
            return {'content': '{"text":"三峡坝址基石"}', '_finish_reason': 'stop'}
    return root, library, Client()


def test_execution_persists_complete_case_plan_and_exclusive_claim(prepared, monkeypatch):
    root, library, client = prepared
    def run(row, harness, *args):
        harness._unit(schema=acceptance.SCHEMAS['ocr'], max_attempts=1)
        return json.loads(harness.client.chat([])['content'])
    monkeypatch.setattr(acceptance, 'run_case', run)
    monkeypatch.setattr(acceptance, 'fact_checks', lambda *args: [True])
    monkeypatch.setattr(acceptance, 'validate_delivery', lambda *args: None)
    state = acceptance.execute(root, object(), client=client, library=library)
    assert state['status'] == 'awaiting_review' and state['e2e_status'] == 'not_started'
    assert isinstance(state['stages'], dict) and len(state['cases']) == 30
    assert state['metrics']['completed_cases'] == 30 and state['real_wemm_executed'] is False
    assert len(list((root / 'acceptance/cases').glob('*.json'))) == 30
    assert len((root / 'acceptance/model-calls.jsonl').read_text().splitlines()) == 30
    with pytest.raises(FileExistsError):
        acceptance.execute(root, object(), client=client, library=library)
    assert read_json(root / 'acceptance/status.json') == state


def test_thinking_request_flag_without_response_evidence_stops_acceptance(prepared, monkeypatch):
    root, library, client = prepared
    client.reasoning_tokens = 0
    def run(row, harness, *args):
        harness._unit(schema=acceptance.SCHEMAS['ocr'], max_attempts=1)
        return json.loads(harness.client.chat([])['content'])
    monkeypatch.setattr(acceptance, 'run_case', run)
    state = acceptance.execute(root, object(), client=client, library=library)
    assert state['status'] == 'failed' and state['metrics']['system_failure_count'] == 1
    assert state['metrics']['executed_cases'] == 1
    assert state['cases'][0]['physical_calls'][0]['thinking_evidence']['present'] is False


@pytest.mark.parametrize('policy,interface_valid', [('facts-and-interface-v1', True),
    ('interface-only-v2', True), ('interface-only-v2', False)])
def test_sdk_defaults_records_zero_reasoning_without_claiming_thinking_on(prepared, monkeypatch, policy, interface_valid):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    from camroll_eval.common import sha256
    root, library, client = prepared
    sdk_root = root.with_name('minicpm47-sdk-defaults-m4-off-test')
    root.rename(sdk_root)
    profile = MiniCPMInferenceConfig.sdk_defaults(decision_mode='native_tools', json_mode='json_schema', acceptance_policy=policy)
    atomic_json(sdk_root / 'request-profile.json', profile.as_dict())
    protocol = read_json(sdk_root / 'acceptance/protocol.json')
    protocol.update(decision_mode=profile.decision_mode, json_mode=profile.json_mode,
        inference_config=profile.as_dict(), protocol_fingerprint=profile.fingerprint,
        annotation_review={'sha256': 'synthetic-independent-review'})
    atomic_json(sdk_root / 'acceptance/protocol.json', protocol)
    monkeypatch.setattr(acceptance, 'annotation_review', lambda *args, **kwargs: protocol['annotation_review'])
    client.reasoning_tokens, client.payload = 0, {'model': MODEL, 'messages': []}
    def run(row, harness, *args):
        harness._unit(schema=acceptance.SCHEMAS['ocr'], max_attempts=1)
        return json.loads(harness.client.chat([])['content'])
    monkeypatch.setattr(acceptance, 'run_case', run)
    monkeypatch.setattr(acceptance, 'fact_checks', lambda *args: [policy != 'interface-only-v2'])
    monkeypatch.setattr(acceptance, 'validate_delivery', lambda *args: None)
    state = acceptance.execute(sdk_root, object(), client=client, library=library)
    assert state['status'] == 'awaiting_review' and state['metrics']['completed_cases'] == 30
    assert state['inference_config'] == profile.as_dict()
    assert state['annotation_version'] == 'assistant-reviewed-original-壩-v2'
    evidence = state['cases'][0]['physical_calls'][0]['thinking_evidence']
    assert evidence['requested'] == 'unspecified' and evidence['required_for_acceptance'] is False
    assert evidence['present'] is False
    if policy == 'interface-only-v2':
        assert state['metrics']['fact_check_rate'] == 0 and state['facts_block_acceptance'] is False
        decisions = [{'case_id': row['case_id'],
            'case_sha256': sha256(sdk_root / 'acceptance/cases' / (row['case_id'] + '.json')),
            'interface_contract_valid': interface_valid, 'facts_verified': False, 'identity_grounded': False,
            'notes': 'Incorrect factual content retained; interface audited separately.'} for row in state['cases']]
        review = {'schema_version': 'minicpm-independent-interface-review-v2',
            'reviewer': 'assistant-independent-visual-review', 'status_sha256': sha256(sdk_root / 'acceptance/status.json'),
            'cases': decisions}
        review_path = sdk_root / 'review.json'
        atomic_json(review_path, review)
        checked = acceptance.apply_independent_review(sdk_root, review_path)
        assert checked['status'] == ('passed' if interface_valid else 'failed')
        assert checked['independent_review']['facts_verified_cases'] == 0
        assert checked['independent_review']['quality_is_observational'] is True


def test_sdk_annotation_keeps_original_traditional_glyph_without_rewriting_legacy():
    ids = ['img_' + str(index) for index in range(7)]
    new = 'minicpm47-sdk-defaults-m4-off-v1'
    assert acceptance.fact_checks('ocr', {'text': '三峡壩址基石'}, ids, new) == [True]
    assert acceptance.fact_checks('ocr', {'text': '三峡坝址基石'}, ids, new) == [False]
    assert acceptance.fact_checks('ocr', {'text': '三峡坝址基石'}, ids) == [True]
    for name in ('final_schema', 'm4_review', 'six_tools_roundtrip'):
        key = 'evidence_image_ids' if name == 'm4_review' else 'selected_image_ids'
        assert acceptance.fact_checks(name, {'answer': '三峡壩址基石', key: [ids[6]]}, ids, new) == [True, True]
    for color in ('玫红', '品红', '洋紅', 'magenta'):
        output = {'scene': '婚礼', 'background_color': color, 'white_garment': '婚纱'}
        assert acceptance.fact_checks('scene', output, ids, new) == [True, True, True]
        assert acceptance.fact_checks('scene', output, ids) == [True, False, True]
    appearance = {'garment_type': '皮质外观夹克', 'garment_color': '黑色', 'pose': '扶胯', 'background_colors': ['紫', '白']}
    assert acceptance.fact_checks('appearance', appearance, ids, new) == [True] * 5
    assert acceptance.fact_checks('appearance', appearance, ids) == [False, True, False, True, True]
    appearance['garment_color'] = '蓝色'
    assert acceptance.fact_checks('appearance', appearance, ids, new)[1] is False


def test_sdk_six_image_prompt_defines_people_as_live_people_only():
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    prompts = []
    harness = SimpleNamespace(json=lambda prompt, *args, **kwargs: prompts.append(prompt))
    library = SimpleNamespace(image_paths=lambda ids: [])
    fixtures = [{'id': 'img_' + str(index)} for index in range(7)]
    row = {'class': 'six_images', 'repetition': 0}
    acceptance.run_case(row, harness, library, fixtures, MiniCPMInferenceConfig.sdk_defaults())
    assert 'visible_people只计现场真人' in prompts[-1] and '排除墙上照片' in prompts[-1]
    acceptance.run_case(row, harness, library, fixtures, MiniCPMInferenceConfig())
    assert 'visible_people只计现场真人' not in prompts[-1]


@pytest.mark.parametrize('facts_verified', [True, False])
def test_independent_review_requires_all_original_case_hashes(prepared, monkeypatch, facts_verified):
    from camroll_eval.common import sha256
    root, library, client = prepared
    def run(row, harness, *args):
        harness._unit(schema=acceptance.SCHEMAS['ocr'], max_attempts=1)
        return json.loads(harness.client.chat([])['content'])
    monkeypatch.setattr(acceptance, 'run_case', run)
    monkeypatch.setattr(acceptance, 'fact_checks', lambda *args: [True])
    monkeypatch.setattr(acceptance, 'validate_delivery', lambda *args: None)
    state = acceptance.execute(root, object(), client=client, library=library)
    assert state['status'] == 'awaiting_review'
    decisions = [{'case_id': row['case_id'],
        'case_sha256': sha256(root / 'acceptance/cases' / (row['case_id'] + '.json')),
        'approved': True, 'facts_verified': facts_verified, 'identity_grounded': True,
        'notes': 'Synthetic independent review fixture.'} for row in state['cases']]
    review = {'schema_version': 'minicpm-independent-visual-review-v1',
        'reviewer': 'assistant-independent-visual-review', 'status_sha256': sha256(root / 'acceptance/status.json'),
        'cases': decisions}
    path = root / 'review.json'
    atomic_json(path, {**review, 'cases': decisions[:-1]})
    with pytest.raises(ValueError):
        acceptance.apply_independent_review(root, path)
    assert read_json(root / 'acceptance/status.json')['status'] == 'awaiting_review'
    altered = copy.deepcopy(review)
    altered['cases'][0]['case_sha256'] = '0' * 64
    atomic_json(path, altered)
    with pytest.raises(ValueError):
        acceptance.apply_independent_review(root, path)
    atomic_json(path, review)
    checked = acceptance.apply_independent_review(root, path)
    assert checked['status'] == ('passed' if facts_verified else 'failed') and checked['independent_review']['reviewed_cases'] == 30
    assert checked['stages']['independent_review']['status'] == ('passed' if facts_verified else 'failed')
    assert checked['independent_review']['source_status_sha256'] == review['status_sha256']


def test_unknown_keeps_remaining_cases_unexecuted_and_never_restarts(prepared, monkeypatch):
    root, library, client = prepared
    def run(row, harness, *args):
        client.callback({'status': 'uncertain', 'error_code': 'usage_missing', 'attempt': 1})
        raise RequestUncertain('synthetic usage missing')
    monkeypatch.setattr(acceptance, 'run_case', run)
    state = acceptance.execute(root, object(), client=client, library=library)
    assert state['status'] == 'failed' and state['metrics']['executed_cases'] == 1
    assert state['metrics']['system_failure_count'] == 1
    assert sum(row['status'] == 'not_started' for row in state['cases']) == 29
    assert state['reason'].startswith('acceptance_system_failure')
