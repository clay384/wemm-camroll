"""No-network MiniCPM supervisor safety and acceptance tests."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import minicpm_config as config
import minicpm_suite as suite
from camroll_eval.common import atomic_json, read_json
from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig, PROTOCOL
from test_gate_suite import complete_run


def ready_summary():
    return {'limit_cny': 600, 'reserved_cny': 1.268122, 'uncertain_requests': 1,
        'acknowledged_reserved_cny': 1.268122, 'acknowledged_uncertain_requests': 1,
        'blocking_uncertain_requests': 0, 'remaining_cny': 100}


def complete(stage='formal'):
    result = complete_run()
    result.update(mode='minicpm_compare', protocol_label=PROTOCOL, inference_complete=True,
                  inference_config=MiniCPMInferenceConfig().as_dict())
    if stage == 'e2e':
        result.update(formal_protocol=False, items=result['items'][-3:])
        result['qa_plan'] = result['qa_plan'][-3:]
        for row in result['items']:
            row['tool_trace'] = [{'tool': 'final_image_review', 'conversation_turn': 0, 'status': 'ok',
                'args': {'image_ids': ['img_1']}, 'result': {'evidence_image_ids': ['img_1']}}]
            row['model_call_metrics'] = [{'call_type': 'final_review', 'conversation_turn': 0, 'status': 'success',
                'model': 'MiniCPM-V-4.7-35B-A3B', 'input': {'model': 'MiniCPM-V-4.7-35B-A3B', 'messages': [],
                    'stream': False, 'temperature': 0, 'max_tokens': 8192,
                    'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}}}]
        from camroll_agent.schemas import TOOL_NAMES
        result['items'][0]['tool_trace'].extend({'tool': name, 'status': 'success', 'conversation_turn': 0,
            'result': {}, 'cached': False, 'embedding_calls': [{'operation': '/api/internal/embeddings', 'status': 'success'}]
            if name == 'search' else []} for name in TOOL_NAMES)
    return result


def test_runtime_is_required_and_history_cannot_be_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'BASE', tmp_path)
    approved = tmp_path / 'runtime/minicpm47-thinking-m4-off-new'
    assert config.validate_runtime(approved) == approved
    sdk = tmp_path / 'runtime/minicpm47-sdk-defaults-m4-off-new'
    assert config.validate_runtime(sdk) == sdk
    for path in (tmp_path / 'runtime/photobench-flash-gate-old', approved / 'nested', Path('relative')):
        with pytest.raises(ValueError):
            config.validate_runtime(path)
    approved.parent.mkdir()
    link = approved.parent / 'minicpm47-thinking-m4-off-link'
    link.symlink_to(approved)
    with pytest.raises(ValueError):
        config.validate_runtime(link)
    with pytest.raises(SystemExit):
        suite.main([])


def test_validate_fixed_formal_and_e2e_contracts():
    assert suite.validate_run(complete())
    assert suite.validate_run(complete('e2e'), stage='e2e')
    historical = complete_run()
    assert not suite.validate_run(historical)
    formal = complete()
    formal['items'][0]['runtime_turns'][0].update(answer_origin='system_failure', judge={'score': None}, judge_status='skipped_expected')
    assert suite.validate_run(formal)
    smoke = complete('e2e')
    smoke['items'][0]['runtime_turns'][0]['answer_origin'] = 'system_failure'
    assert not suite.validate_run(smoke, stage='e2e')


@pytest.mark.parametrize('policy', ['facts-and-interface-v1', 'interface-only-v2'])
def test_sdk_defaults_e2e_requires_missing_parameters_and_matching_version(policy):
    profile = MiniCPMInferenceConfig.sdk_defaults(acceptance_policy=policy)
    smoke = complete('e2e')
    smoke.update(protocol_label=profile.protocol_version, inference_config=profile.as_dict())
    for row in smoke['items']:
        row['model_call_metrics'][0]['input'] = {'model': profile.generation_model, 'messages': []}
    assert suite.validate_run(smoke, stage='e2e')
    for key in ('temperature', 'stream', 'max_tokens', 'max_completion_tokens', 'chat_template_kwargs', 'extra_body'):
        invalid = copy.deepcopy(smoke)
        invalid['items'][0]['model_call_metrics'][0]['input'][key] = None
        assert not suite.validate_run(invalid, stage='e2e')
    smoke['protocol_label'] = PROTOCOL
    assert not suite.validate_run(smoke, stage='e2e')


def test_e2e_accepts_real_normalized_m4_trace_and_wrong_judged_answers():
    from camroll_eval.metrics import normalize_calls, normalize_tools
    smoke = complete('e2e')
    profile = MiniCPMInferenceConfig.sdk_defaults(acceptance_policy='interface-only-v2')
    smoke.update(protocol_label=profile.protocol_version, inference_config=profile.as_dict())
    for item in smoke['items']:
        call = item['model_call_metrics'][0]
        call.update(call_type='final', input={'model': profile.generation_model, 'messages': []})
        item['model_call_metrics'] = normalize_calls([call], 0)
        for row in item['tool_trace']:
            if row['tool'] == 'final_image_review':
                row.pop('status')
        item['tool_trace'] = normalize_tools(item['tool_trace'], 0)
        turn = item['runtime_turns'][0]
        turn['judge']['score'] = 0
        turn['task_judge']['correct'] = False
    assert suite.validate_run(smoke, stage='e2e')


def test_request_and_protocol_profiles_cannot_silently_revert_or_drift(tmp_path):
    runtime = tmp_path / 'minicpm47-sdk-defaults-m4-off-profile'
    with pytest.raises(ValueError, match='explicit'):
        config.request_profile(runtime)
    profile = MiniCPMInferenceConfig.sdk_defaults()
    atomic_json(runtime / 'request-profile.json', profile.as_dict())
    assert config.request_profile(runtime) == profile
    protocol = {'decision_mode': profile.decision_mode, 'json_mode': profile.json_mode,
        'inference_config': profile.as_dict(), 'protocol_fingerprint': profile.fingerprint}
    assert config.protocol_profile(runtime, protocol) == profile
    altered = MiniCPMInferenceConfig.sdk_defaults(json_mode='prompt')
    with pytest.raises(ValueError):
        config.protocol_profile(runtime, {**protocol, 'json_mode': altered.json_mode,
            'inference_config': altered.as_dict(), 'protocol_fingerprint': altered.fingerprint})
    with pytest.raises(ValueError):
        config.protocol_profile(runtime, {'decision_mode': 'native_tools', 'json_mode': 'json_schema'})
    protocol['protocol_fingerprint'] = 'changed'
    with pytest.raises(ValueError):
        config.protocol_profile(runtime, protocol)


def test_independent_review_binds_actual_images_not_reviewer_local_paths(tmp_path):
    from camroll_eval.common import sha256
    profile = MiniCPMInferenceConfig.sdk_defaults()
    image = tmp_path / 'original-photo'
    image.write_bytes(b'fixed original photo fixture')
    path = tmp_path / 'acceptance/annotation-review-sdk-defaults.json'
    review = {'schema_version': 'minicpm-sdk-defaults-private-annotation-review-v1',
        'new_protocol': profile.protocol_version, 'reviewer': 'independent test',
        'formal_qa_gt_accessed': False, 'original_fixtures_modified': False,
        'fixtures': [{'index': index, 'sha256': sha256(image), 'file': '/other-host/private-image.jpg'} for index in range(7)]}
    atomic_json(path, review)
    binding = config.annotation_review(tmp_path, profile, image_paths=[image] * 7)
    assert binding['sha256'] == sha256(path)
    image.write_bytes(b'changed photo')
    with pytest.raises(ValueError, match='Library assets'):
        config.annotation_review(tmp_path, profile, image_paths=[image] * 7)
    review['formal_qa_gt_accessed'] = True
    atomic_json(path, review)
    with pytest.raises(ValueError, match='identity'):
        config.annotation_review(tmp_path, profile)


@pytest.mark.parametrize('tool', ['search', 'grep', 'list_by_date', 'get', 'view_image', 'query_photo_people'])
def test_e2e_requires_successful_execution_of_all_six_tools(tool):
    smoke = complete('e2e')
    matching = next(row for row in smoke['items'][0]['tool_trace'] if row['tool'] == tool)
    matching['status'] = 'error'
    assert not suite.validate_run(smoke, stage='e2e')
    matching.update(status='success', cached=True)
    assert not suite.validate_run(smoke, stage='e2e')


def test_e2e_requires_actual_embedding_call_and_exactly_one_m4_review_each_turn():
    smoke = complete('e2e')
    search = next(row for row in smoke['items'][0]['tool_trace'] if row['tool'] == 'search')
    search['embedding_calls'] = []
    assert not suite.validate_run(smoke, stage='e2e')
    for index in range(3):
        smoke = complete('e2e')
        trace = smoke['items'][index]['tool_trace']
        review = next(row for row in trace if row['tool'] == 'final_image_review')
        trace.remove(review)
        assert not suite.validate_run(smoke, stage='e2e')
        trace.extend([review, copy.deepcopy(review)])
        assert not suite.validate_run(smoke, stage='e2e')


def test_e2e_requires_pinned_model_thinking_and_successful_final_cloud_call_not_correct_answers():
    smoke = complete('e2e')
    assert all(row['runtime_turns'][0]['judge']['score'] == 0 for row in smoke['items'])
    assert suite.validate_run(smoke, stage='e2e')
    call = smoke['items'][1]['model_call_metrics'][0]
    call['input']['extra_body']['chat_template_kwargs']['enable_thinking'] = False
    assert not suite.validate_run(smoke, stage='e2e')
    call['input']['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    call['status'] = 'error'
    assert not suite.validate_run(smoke, stage='e2e')
    call['status'] = 'success'
    call['model'] = 'qwen3.7-flash'
    assert not suite.validate_run(smoke, stage='e2e')


def test_historical_baseline_requires_exact_original_identity_and_bytes(tmp_path, monkeypatch):
    import camroll_eval.common as common
    monkeypatch.setattr(suite, 'BASELINE_RUNTIME', tmp_path)
    expected_id = 'f37-off-gate-20261006-181014-8bd9a5'
    expected_sha = '8ee4497f779148783241dc0c2add0574d727e8a368e96687b6aca0bdbcc91d37'
    assert suite.BASELINE_RUN_ID == expected_id and suite.BASELINE_RUN_SHA256 == expected_sha
    path = tmp_path / 'evaluation/runs' / expected_id / 'run.json'
    atomic_json(path, {**complete_run(), 'run_id': expected_id})
    suite_path = tmp_path / 'formal-suite.json'
    atomic_json(suite_path, {'groups': [{'name': 'F37-Off', 'run_id': expected_id}]})
    monkeypatch.setattr(common, 'sha256', lambda value: expected_sha if Path(value) == path else 'suite-hash')
    run, baseline_suite = suite.historical_baseline()
    assert run == {'path': str(path), 'sha256': expected_sha}
    assert baseline_suite['path'] == str(suite_path)
    atomic_json(suite_path, {'groups': [{'name': 'F37-Off', 'run_id': 'replacement-valid-run'}]})
    with pytest.raises(RuntimeError, match='identity_changed'):
        suite.historical_baseline()
    atomic_json(suite_path, {'groups': [{'name': 'F37-Off', 'run_id': expected_id}]})
    monkeypatch.setattr(common, 'sha256', lambda value: 'modified-bytes')
    with pytest.raises(RuntimeError, match='hash_changed'):
        suite.historical_baseline()


@pytest.mark.parametrize('change', [{'blocking_uncertain_requests': 1, 'uncertain_requests': 2},
    {'reserved_cny': 1.3}, {'uncertain_requests': 0}, {'acknowledged_uncertain_requests': 0}, {'limit_cny': 601}])
def test_only_old_exact_acknowledged_charge_is_allowed(change):
    class Ledger:
        def check_limit(self): pass
        def summary(self): return {**ready_summary(), **change}
    with pytest.raises(RuntimeError):
        suite.ledger_ready(Ledger())


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    import four_gpu_window
    import four_supplement
    import gate_continuation
    runtime = tmp_path / 'runtime/minicpm47-thinking-m4-off-test'
    runtime.mkdir(parents=True)
    atomic_json(runtime / 'acceptance/status.json', {'status': 'passed', 'e2e_status': 'not_started'})
    events, controls = [], {'new_unknown': False, 'foreign_gpu': False, 'register_error': False, 'cleanup_error': False}
    class Ack:
        def as_dict(self): return {'call_id': 'exact-old-unknown'}
    class Ledger:
        def check_limit(self): pass
        def summary(self):
            return {**ready_summary(), **({'blocking_uncertain_requests': 1, 'uncertain_requests': 2}
                                         if controls.get('blocked') else {})}
    class Window:
        def __init__(self, release, window_id, *, runtime):
            assert runtime == suite.HISTORY_RUNTIME
            self.state_path = tmp_path / 'window.json'
            events.append('shared_registry')
        def enter(self):
            events.append('enter')
            if controls['foreign_gpu']: raise RuntimeError('Unexpected GPU1 owner')
            self.state_path.write_text('{}')
            return {'before': {'fingerprint': 'unchanged'}}
        def switch(self, kind):
            assert kind == 'wemm'
            events.append('switch')
        def register_child(self, pid, expected_command):
            events.append('register')
            if controls['register_error']: raise RuntimeError('registration failed')
        def unregister_child(self, pid): events.append('unregister')
        def restore(self, reason):
            events.append('restore')
            return {'state': 'restored'}
    class Child:
        pid = 112233
        def __init__(self, command, **kwargs):
            events.append('child')
            assert kwargs['cwd'] == suite.RELEASE
            self.returncode, self.polls = (2 if controls['new_unknown'] else 0), 0
            stage = command[command.index('--stage') + 1]
            run_id = command[command.index('--run-id') + 1]
            run = complete(stage)
            if controls['new_unknown']:
                run.update(status='blocked', inference_complete=False)
                controls['blocked'] = True
            atomic_json(runtime / 'evaluation/runs' / run_id / 'run.json', run)
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode
    def cleanup(child):
        events.append('cleanup')
        if controls['cleanup_error']: raise RuntimeError('cleanup failed')
        return {'state': 'absent_or_exited'}
    freeze = {'wemm_fingerprint': 'unchanged', 'model_files': {'models': {}},
              'wemm_weights': {'actual_sha256': 'unchanged', 'file_identity_after': {'inode': 1}}}
    def frozen_inputs(runtime, stage, *args):
        atomic_json(runtime / (stage + '-freeze.json'), freeze)
        return copy.deepcopy(freeze)
    monkeypatch.setattr(config, 'BASE', tmp_path)
    monkeypatch.setattr(suite, 'freeze_inputs', frozen_inputs)
    monkeypatch.setattr(suite, 'verify_frozen', lambda *args: None)
    monkeypatch.setattr(suite.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(suite.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(suite.subprocess, 'Popen', Child)
    monkeypatch.setattr(four_gpu_window, 'GpuWindow', Window)
    monkeypatch.setattr(four_gpu_window, 'verify_model_files', lambda: {'models': {}})
    monkeypatch.setattr(four_supplement, 'stop_owned_child', cleanup)
    monkeypatch.setattr(gate_continuation, 'verify_wemm_weights', lambda: {
        'actual_sha256': 'unchanged', 'file_identity_before': {'inode': 1}})
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', 'minicpm-test')
    return runtime, Ledger(), Ack(), events, controls


@pytest.mark.parametrize('stage', ['formal', 'e2e'])
def test_independent_stage_runs_and_restores(supervisor, stage):
    runtime, ledger, ack, events, controls = supervisor
    assert suite.execute(runtime, ledger, ack, stage=stage) == 0
    state = read_json(runtime / (stage + '-suite.json'))
    assert state['status'] == 'passed' and state['restoration']['state'] == 'restored'
    assert state['suite_id'].startswith('minicpm47-thinking-m4-off-')
    assert state['run_id'].startswith('minicpm47-thinking-m4-off-')
    assert state['budget_after']['reserved_cny'] == 1.268122
    assert state['wemm_weights_unchanged'] and events.count('child') == 1
    if stage == 'e2e':
        accepted = read_json(runtime / 'acceptance/status.json')
        assert accepted['e2e_status'] == 'passed' and set(accepted['e2e_evidence']) == {'run', 'suite', 'freeze'}
    with pytest.raises(FileExistsError):
        suite.execute(runtime, ledger, ack, stage=stage)


def test_foreign_gpu_never_spawns_or_controls_process(supervisor):
    runtime, ledger, ack, events, controls = supervisor
    controls['foreign_gpu'] = True
    assert suite.execute(runtime, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['status'] == 'blocked' and state['restoration']['state'] == 'not_entered'
    assert 'child' not in events and 'switch' not in events and 'restore' not in events


def test_new_unknown_marks_cancel_and_restores(supervisor):
    runtime, ledger, ack, events, controls = supervisor
    controls['new_unknown'] = True
    assert suite.execute(runtime, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['status'] == 'blocked' and state['restoration']['state'] == 'restored'
    assert state['new_submissions_blocked'] == 'new_uncertain_cloud_request_stop'
    assert (runtime / 'evaluation/runs' / state['run_id'] / 'cancel.requested').is_file()


def test_failed_registration_and_cleanup_still_restore(supervisor):
    runtime, ledger, ack, events, controls = supervisor
    controls.update(register_error=True, cleanup_error=True)
    assert suite.execute(runtime, ledger, ack) == 2
    state = read_json(runtime / 'formal-suite.json')
    assert state['child_cleanup']['state'] == 'failed'
    assert state['restoration']['state'] == 'restored'


def test_formal_requires_e2e_before_any_gpu_operation(tmp_path, monkeypatch):
    from four_suite import source_manifest
    source = source_manifest(suite.RELEASE)
    atomic_json(tmp_path / 'validation.json', {'status': 'passed', 'source_files': source})
    atomic_json(tmp_path / 'acceptance/status.json', {'status': 'passed', 'source_files': source, 'e2e_status': 'not_started'})
    with pytest.raises(RuntimeError, match='successful_real_photo_e2e'):
        suite.acceptance_inputs(tmp_path, 'formal')


@pytest.fixture
def bound_e2e(tmp_path, monkeypatch):
    import four_suite
    from camroll_eval.common import sha256
    profile = MiniCPMInferenceConfig.sdk_defaults(acceptance_policy='interface-only-v2')
    source = {'all-source': 'current-frozen-sha'}
    revision, run_id, suite_id = 'minicpm-current-release', 'minicpm-current-e2e-run', 'minicpm-current-e2e'
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION', revision)
    monkeypatch.setattr(four_suite, 'source_manifest', lambda *args: source)
    paths = {'run': tmp_path / 'evaluation/runs' / run_id / 'run.json',
             'suite': tmp_path / 'e2e-suite.json', 'freeze': tmp_path / 'e2e-freeze.json'}
    run = complete('e2e')
    run.update(run_id=run_id, suite_id=suite_id, protocol_label=profile.protocol_version,
        inference_config=profile.as_dict(), inference_fingerprint=profile.fingerprint,
        code_revision=revision, index_version=suite.SNAPSHOT, index_binding={'snapshot': suite.SNAPSHOT})
    for item in run['items']:
        item['model_call_metrics'][0]['input'] = {'model': profile.generation_model, 'messages': []}
    documents = {'run': run, 'suite': {'status': 'passed', 'stage': 'e2e', 'run_id': run_id, 'suite_id': suite_id,
        'source_revision': revision, 'restoration': {'state': 'restored'}},
        'freeze': {'stage': 'e2e', 'source_files': source, 'source_revision': revision,
        'inference_config': profile.as_dict(), 'inference_fingerprint': profile.fingerprint,
        'snapshot': suite.SNAPSHOT, 'index_binding': run['index_binding']}}
    status = {'status': 'passed', 'source_files': source, 'inference_config': profile.as_dict(),
        'e2e_status': 'passed', 'e2e_run_id': run_id, 'e2e_suite_id': suite_id}
    def persist(*, refresh_hashes=True):
        for name, value in documents.items():
            atomic_json(paths[name], value)
        if refresh_hashes:
            status['e2e_evidence'] = {name: sha256(path) for name, path in paths.items()}
        atomic_json(tmp_path / 'acceptance/status.json', status)
    persist()
    atomic_json(tmp_path / 'validation.json', {'status': 'passed', 'source_files': source})
    return tmp_path, profile, source, status, documents, paths, persist


def test_current_e2e_sources_profile_and_identity_are_bound(bound_e2e):
    runtime, profile, source, status, docs, paths, persist = bound_e2e
    assert suite.validate_e2e_evidence(runtime, status, source, profile) == paths


@pytest.mark.parametrize('variant', ['old_protocol', 'different_mode', 'freeze_source', 'freeze_profile',
    'freeze_fingerprint', 'code_revision', 'run_identity', 'suite_identity', 'index_binding', 'run_hash', 'freeze_hash'])
def test_formal_rejects_other_e2e_protocol_mode_sources_or_changed_bytes(bound_e2e, variant):
    runtime, profile, source, status, docs, paths, persist = bound_e2e
    if variant in ('old_protocol', 'different_mode'):
        other = (MiniCPMInferenceConfig.sdk_defaults() if variant == 'old_protocol' else
                 MiniCPMInferenceConfig.sdk_defaults(decision_mode='json_decision', json_mode='prompt', acceptance_policy='interface-only-v2'))
        docs['run'].update(protocol_label=other.protocol_version, inference_config=other.as_dict(), inference_fingerprint=other.fingerprint)
        docs['freeze'].update(inference_config=other.as_dict(), inference_fingerprint=other.fingerprint)
    elif variant == 'freeze_source': docs['freeze']['source_files'] = {'old-source': 'old-sha'}
    elif variant == 'freeze_profile': docs['freeze']['inference_config'] = MiniCPMInferenceConfig.sdk_defaults().as_dict()
    elif variant == 'freeze_fingerprint': docs['freeze']['inference_fingerprint'] = 'different'
    elif variant == 'code_revision': docs['run']['code_revision'] = 'old-release'
    elif variant == 'run_identity': docs['run']['run_id'] = 'different-run'
    elif variant == 'suite_identity': docs['run']['suite_id'] = 'different-suite'
    elif variant == 'index_binding': docs['run']['index_binding'] = {'other': 'index'}
    else: docs['run' if variant == 'run_hash' else 'freeze']['added_after_completion'] = True
    persist(refresh_hashes=variant not in ('run_hash', 'freeze_hash'))
    with pytest.raises(RuntimeError):
        suite.acceptance_inputs(runtime, 'formal')


@pytest.mark.parametrize('change,passes', [
    ({}, True), ({'raw_model': {'is_free': True, 'input_pricing': '0', 'output_pricing': '0'}}, False),
    ({'source': 'https://different.example/prices'}, False), ({'context_limit': 1000000}, False),
    ({'input_rate': 1}, False), ({'raw_model': {'is_free': False, 'input_pricing': '0', 'output_pricing': '1'}}, False),
])
def test_stage_can_refresh_capture_time_but_not_pricing_or_capability_facts(tmp_path, change, passes):
    from four_suite import source_manifest
    source = source_manifest(suite.RELEASE)
    old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()
    evidence = {'model': 'MiniCPM-V-4.7-35B-A3B', 'input_rate': 0, 'output_rate': 0,
        'observed_at': old, 'captured_at': old, 'source': 'https://platform.modelbest.cn/models',
        'context_limit': 256000, 'raw_model': {'is_free': False, 'input_pricing': '0', 'output_pricing': '0'}}
    current = {**evidence, 'observed_at': fresh, 'captured_at': fresh, **change}
    atomic_json(tmp_path / 'validation.json', {'status': 'passed', 'source_files': source})
    atomic_json(tmp_path / 'acceptance/status.json', {'status': 'passed', 'source_files': source,
        'inference_config': MiniCPMInferenceConfig().as_dict(), 'pricing_evidence': evidence})
    atomic_json(tmp_path / 'pricing-evidence.json', current)
    if passes:
        result = suite.acceptance_inputs(tmp_path, 'e2e')
        assert result[-1] == current and result[1]['pricing_evidence'] == evidence
    else:
        with pytest.raises((RuntimeError, ValueError)):
            suite.acceptance_inputs(tmp_path, 'e2e')


def test_interface_only_stage_requires_exact_policy_and_review_but_not_fact_accuracy(tmp_path, monkeypatch):
    import four_suite
    import minicpm_acceptance as acceptance
    from camroll_eval.common import sha256
    runtime = tmp_path / 'minicpm47-sdk-defaults-m4-off-interface'
    profile = MiniCPMInferenceConfig.sdk_defaults(acceptance_policy='interface-only-v2')
    source = {'all-source': 'frozen'}
    monkeypatch.setattr(four_suite, 'source_manifest', lambda *args: source)
    annotation = {'sha256': 'pre-inference-review'}
    monkeypatch.setattr(suite, 'annotation_review', lambda *args: annotation)
    pricing = {'model': profile.generation_model, 'input_rate': 0, 'output_rate': 0,
        'observed_at': datetime.now(timezone.utc).isoformat(), 'source': 'synthetic-test'}
    atomic_json(runtime / 'pricing-evidence.json', pricing)
    atomic_json(runtime / 'request-profile.json', profile.as_dict())
    protocol = {'decision_mode': profile.decision_mode, 'json_mode': profile.json_mode,
        'inference_config': profile.as_dict(), 'protocol_fingerprint': profile.fingerprint,
        'annotation_review': annotation}
    atomic_json(runtime / 'acceptance/protocol.json', protocol)
    atomic_json(runtime / 'validation.json', {'status': 'passed', 'source_files': source})
    review_path = runtime / 'acceptance/independent-review.json'
    rows = [{**row, 'status': 'completed', 'fact_checks': [False],
        'structured_units': [{'attempts': [{'valid': True}]}],
        'physical_calls': [{'attempt': 1, 'status': 'success'}]} for row in acceptance.planned_cases()]
    decisions = []
    for row in rows:
        case_path = runtime / 'acceptance/cases' / (row['case_id'] + '.json')
        atomic_json(case_path, row)
        decisions.append({'case_id': row['case_id'], 'case_sha256': sha256(case_path),
            'interface_contract_valid': True, 'facts_verified': False, 'identity_grounded': False})
    atomic_json(review_path, {'schema_version': 'minicpm-independent-interface-review-v2',
        'reviewer': 'assistant-independent-visual-review', 'cases': decisions})
    status = {'status': 'passed', 'source_files': source, 'inference_config': profile.as_dict(),
        'acceptance_policy': 'interface-only-v2', 'schema_version': 'minicpm-30-case-acceptance-v2',
        'facts_block_acceptance': False, 'manual_review_required': False, 'metrics': acceptance.metrics(rows),
        'annotation_review': annotation, 'protocol_sha256': sha256(runtime / 'acceptance/protocol.json'),
        'pricing_evidence': pricing, 'independent_review': {'path': 'independent-review.json',
            'sha256': sha256(review_path), 'reviewed_cases': 30}}
    atomic_json(runtime / 'acceptance/status.json', status)
    assert suite.acceptance_inputs(runtime, 'e2e')[3] == profile
    atomic_json(runtime / 'acceptance/status.json', {**status, 'acceptance_policy': 'facts-and-interface-v1'})
    with pytest.raises(RuntimeError, match='policy'):
        suite.acceptance_inputs(runtime, 'e2e')
    atomic_json(runtime / 'acceptance/status.json', status)
    atomic_json(review_path, {'changed': True})
    with pytest.raises(RuntimeError, match='hash_bound_review'):
        suite.acceptance_inputs(runtime, 'e2e')


def test_child_identity_mismatch_stops_before_gpu_and_backend(tmp_path):
    atomic_json(tmp_path / 'formal-suite.json', {'suite_id': 'one', 'status': 'running', 'stage': 'formal', 'run_id': 'owned'})
    atomic_json(tmp_path / 'formal-freeze.json', {'stage': 'formal'})
    with pytest.raises(RuntimeError, match='identity'):
        suite.run_child(tmp_path, 'formal', 'one', 'not-owned', object())
