import copy
import json

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json, sha256
import camroll_eval.repository as repository
from camroll_eval.repository import ComparisonScopeError, Repository
import minicpm_report as report
from test_report_ui import evaluate


@pytest.fixture
def pair(tmp_path, monkeypatch):
    runtime = tmp_path / 'minicpm47-thinking-m4-off-synthetic'
    root = runtime / 'evaluation'
    history = tmp_path / 'photobench-flash-gate-20261006f/evaluation'
    plan = [{'qa_id': f'qa-{i}', 'question': f'Question {i}',
             'planned_turn_count': 2 if i < 11 else 1} for i in range(100)]
    common = {'formal_protocol': True, 'status': 'completed_with_errors', 'qa_count': 100,
        'planned_turn_count': 111, 'planned_eligible_qa_count': 81, 'qa_plan': plan,
        'input_integrity': {'qa_sha256': 'qa-fixed', 'dataset_sha256': 'dataset-fixed'},
        'index_binding': {'snapshot_version': 'same-frozen-index'},
        'judge_protocol': {'prompt_hash': 'fixed', 'model': 'doubao'},
        'summary': {'retrieval_metric_count': 81, 'judge_valid_count': 0,
                    'quality_correct_planned_rate': 0, 'quality_mean_fixed_planned': 0},
        'items': [{'qa_id': 'qa-0', 'final_turn_executed': True, 'runtime_turns': [{}]}]}
    old = {**copy.deepcopy(common), 'run_id': repository.MINICPM_HISTORY_ID,
        'mode': 'gate_ab', 'protocol_label': 'camroll-flash-gate-v1',
        'gate_summary_version': 'camroll-flash-gate-v1',
        'inference_config': {'group': 'F37-Off', 'generation_model': 'qwen3.7-flash', 'gate_policy': 'off'}}
    new = {**copy.deepcopy(common), 'run_id': 'minicpm-formal-synthetic',
        'mode': 'minicpm_compare', 'protocol_label': repository.MINICPM_PROTOCOL,
        'gate_summary_version': repository.MINICPM_PROTOCOL,
        'inference_config': {'group': 'MiniCPM47-Thinking-On-Off', 'gate_policy': 'off',
            'generation_model': 'MiniCPM-V-4.7-35B-A3B', 'enable_thinking': True,
            'stream': False, 'max_output_tokens': 8192, 'decision_mode': 'json_decision', 'json_mode': 'prompt'}}
    old_path = history / 'runs' / old['run_id'] / 'run.json'
    new_path = root / 'runs' / new['run_id'] / 'run.json'
    atomic_json(old_path, old)
    atomic_json(new_path, new)
    # Only synthetic test bytes use a substituted pin; production is fixed.
    digest = sha256(old_path)
    monkeypatch.setattr(repository, 'MINICPM_HISTORY_SHA256', digest)
    monkeypatch.setattr(report, 'MINICPM_HISTORY_SHA256', digest)
    monkeypatch.setattr(report, 'history_roots', lambda: [history])
    return runtime, root, history, new, old, new_path, old_path


def test_pinned_manifest_preserves_summaries_bytes_and_only_exact_pair(pair):
    runtime, root, history, new, old, new_path, old_path = pair
    before = new_path.read_bytes(), old_path.read_bytes()
    repo = Repository(root, history_roots=[history])
    with pytest.raises(ComparisonScopeError):
        repo.compare([new['run_id'], old['run_id']])
    manifest = report.write_comparison(runtime, new['run_id'])
    result = repo.compare([old['run_id'], new['run_id']])
    assert result['comparison_scope'] == repository.MINICPM_COMPARISON
    assert result['minicpm_scope']['source_hashes_verified'] is True
    assert result['minicpm_scope']['contemporary'] is False
    assert len(result['rows']) == 100
    assert result['runs'][1]['summary'] == new['summary']
    assert result['runs'][1]['report_progress']['executed_turn_count'] == 1
    assert (new_path.read_bytes(), old_path.read_bytes()) == before
    assert manifest['runs'][1]['run_sha256'] == sha256(old_path)
    with pytest.raises(ComparisonScopeError):
        repo.compare([new['run_id'], new['run_id']])


@pytest.mark.parametrize('field,value', [
    ('formal_protocol', False), ('status', 'evaluating'), ('qa_count', 99),
    ('planned_turn_count', 110), ('planned_eligible_qa_count', 80),
    ('judge_protocol', {'prompt_hash': 'different'}),
    ('index_binding', {'snapshot_version': 'other'}),
    ('input_integrity', {'qa_sha256': 'other', 'dataset_sha256': 'dataset-fixed'}),
    ('qa_plan', []), ('protocol_label', 'other-protocol'),
])
def test_export_rejects_protocol_and_denominator_drift_without_publishing(pair, field, value):
    runtime, root, _, new, _, new_path, old_path = pair
    original = old_path.read_bytes()
    new[field] = value
    atomic_json(new_path, new)
    with pytest.raises(ValueError):
        report.write_comparison(runtime, new['run_id'])
    assert not (root / 'minicpm-comparison.json').exists()
    assert old_path.read_bytes() == original


def test_manifest_cannot_repin_changed_history_or_use_untrusted_paths(pair, tmp_path):
    runtime, root, history, new, old, new_path, old_path = pair
    report.write_comparison(runtime, new['run_id'])
    manifest_path = root / 'minicpm-comparison.json'
    manifest = json.loads(manifest_path.read_text())
    old['summary']['judge_valid_count'] = 99
    atomic_json(old_path, old)
    manifest['runs'][1]['run_sha256'] = sha256(old_path)
    manifest['runs'][1]['path'] = str(old_path)
    atomic_json(manifest_path, manifest)
    with pytest.raises(ComparisonScopeError):
        Repository(root, history_roots=[history]).compare([new['run_id'], old['run_id']])
    # A caller-controlled arbitrary file cannot supply the original bytes.
    manifest_path.unlink()
    outside = tmp_path / 'untrusted.json'
    atomic_json(outside, manifest)
    manifest_path.symlink_to(outside)
    with pytest.raises(ComparisonScopeError):
        Repository(root, history_roots=[history]).compare([new['run_id'], old['run_id']])


def test_changed_new_run_invalidates_even_a_previously_cached_comparison(pair):
    runtime, root, history, new, old, new_path, _ = pair
    report.write_comparison(runtime, new['run_id'])
    repo = Repository(root, history_roots=[history])
    repo.compare([new['run_id'], old['run_id']])
    new['summary']['judge_valid_count'] = 3
    atomic_json(new_path, new)
    with pytest.raises(ComparisonScopeError):
        repo.compare([new['run_id'], old['run_id']])


def test_acceptance_only_has_no_formal_results_and_readonly_routes(pair, monkeypatch):
    runtime, root, history, new, old, new_path, old_path = pair
    new_path.unlink()
    atomic_json(runtime / 'acceptance/status.json', {'status': 'passed', 'reason': '接口验收完成',
        'secret': 'must-not-expose', 'stages': {'image': {'status': 'passed', 'payload': 'private'}}})
    atomic_json(runtime / 'status.json', {'status': 'blocked', 'reason': 'GPU归属待核验'})
    app = report.create_report(runtime)
    before = old_path.read_bytes()
    with TestClient(app) as client:
        state = client.get('/api/minicpm/status').json()
        assert state['formal_status'] == 'not_started'
        assert state['reason'] == 'GPU归属待核验'
        assert state['groups'][0]['run_id'] is None
        assert state['comparison_ready'] is False
        assert 'private' not in json.dumps(state) and 'must-not-expose' not in json.dumps(state)
        assert client.get('/api/health').json()['read_only'] is True
        assert client.get('/api/runs').json()['runs'][0]['run_id'] == old['run_id']
        assert client.post('/api/runs').status_code == 405
        assert client.post('/camroll/api/ask').status_code == 405
        assert not any(getattr(route, 'path', '') == '/camroll' for route in app.routes)
    assert old_path.read_bytes() == before


def test_minicpm_ui_discloses_differences_and_never_fabricates_ttft():
    value = evaluate("""(()=>{
      const run={run_id:'mini',mode:'minicpm_compare',formal_protocol:true,qa_count:100,
        planned_turn_count:111,status:'blocked',report_progress:{finished_question_count:1,executed_turn_count:1},
        inference_config:{decision_mode:'json_decision',json_mode:'prompt',stream:false,enable_thinking:true,max_output_tokens:8192},
        summary:{quality_correct_planned_rate:.01,llm_ttft_ms_mean:42,llm_tokens_per_second_mean:99}};
      state.runs=[run];renderRuns();
      renderProfile({schema_version:'minicpm47-thinking-m4-off-v1',groups:[{name:'MiniCPM',run_id:null},{name:'F37-Off',run_id:'old',reused:true}],
        reason:'GPU <pending>',acceptance:{status:'passed'},comparison_ready:false});
      return {panel:minicpmPanel(run),list:$('#run-list').innerHTML,time:modelTimingLabel(run),
        profile:$('#comparison-profile').innerHTML,disabled:$('#compare-minicpm').disabled};
    })()""")
    assert 'MiniCPM' in value['list'] and '1.0%' in value['list']
    assert '已执行轮次 1 / 111' in value['list']
    assert '正式测评未启动' in value['profile'] and 'GPU &lt;pending&gt;' in value['profile']
    assert '四组已完成' not in value['profile'] and value['disabled'] is True
    assert 'json_decision / prompt' in value['panel']
    assert all(term in value['panel'] for term in ('非同期', '8192', 'context_limit', '81题', '不重评分'))
    assert value['time'] == '未观测（非流式，不以总耗时替代TTFT）'


def sdk_pair(pair):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    runtime, root, history, new, old, new_path, old_path = pair
    cfg = MiniCPMInferenceConfig.sdk_defaults().as_dict()
    new.update(protocol_label=cfg['protocol_version'], gate_summary_version=cfg['protocol_version'],
               inference_config=cfg)
    atomic_json(new_path, new)
    return pair


def test_sdk_defaults_comparison_has_its_own_protocol_and_preserves_history(pair):
    runtime, root, history, new, old, new_path, old_path = sdk_pair(pair)
    before = old_path.read_bytes()
    manifest = report.write_comparison(runtime, new['run_id'])
    result = Repository(root, history_roots=[history]).compare([new['run_id'], old['run_id']])
    assert manifest['protocol'] == repository.MINICPM_SDK_PROTOCOL
    assert result['minicpm_scope']['source_hashes_verified']
    assert '参数均未指定' in result['warning']
    assert old_path.read_bytes() == before


@pytest.mark.parametrize('field,value', [('enable_thinking', True), ('max_output_tokens', 8192),
                                        ('temperature', 0), ('stream', False), ('thinking_mode', 'on')])
def test_sdk_defaults_report_rejects_mixed_protocol_parameters(pair, field, value):
    runtime, root, history, new, old, new_path, old_path = sdk_pair(pair)
    new['inference_config'][field] = value
    atomic_json(new_path, new)
    with pytest.raises(ValueError):
        report.write_comparison(runtime, new['run_id'])
    assert not (root / 'minicpm-comparison.json').exists()


def test_sdk_defaults_pending_report_discloses_mode_and_prior_failures(pair):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    runtime, root, history, new, old, new_path, old_path = pair
    new_path.unlink()
    atomic_json(runtime / 'request-profile.json', MiniCPMInferenceConfig.sdk_defaults().as_dict())
    atomic_json(runtime / 'prior-attempts.json', {'attempts': [{'label': '旧Thinking On探针',
        'status': 'blocked', 'reason': 'content为空', 'private_path': '/private/not-public'}]})
    state = report.profile(runtime, Repository(root, history_roots=[history]))
    assert state['schema_version'] == repository.MINICPM_SDK_PROTOCOL
    assert state['groups'][0]['name'] == 'MiniCPM47-SDK-Defaults-Off'
    assert state['prior_attempts'][0]['reason'] == 'content为空'
    assert 'private_path' not in state['prior_attempts'][0]
    value = evaluate("""(()=>{
      const cfg={request_profile:'sdk-defaults-v1',thinking_mode:'unspecified',enable_thinking:null,
        temperature:null,max_output_tokens:null,stream:null};
      const run={mode:'minicpm_compare',inference_config:cfg};
      renderProfile({schema_version:'minicpm47-sdk-defaults-m4-off-v1',inference_config:cfg,groups:[],
        prior_attempts:[{label:'旧Thinking On探针',status:'blocked',reason:'content为空 <raw>'}]});
      return {label:label(run),panel:minicpmPanel(run),profile:$('#comparison-profile').innerHTML};
    })()""")
    assert 'Thinking 未指定' in value['label'] and 'Thinking On' not in value['label']
    assert '实际上限未知' in value['panel'] or '实际输出上限未知' in value['panel']
    assert '最大输出8192' not in value['panel']
    assert 'content为空 &lt;raw&gt;' in value['profile']


def test_minicpm_three_question_smoke_does_not_claim_formal_denominators():
    value = evaluate("""(()=>{
      const run={run_id:'mini-smoke',mode:'minicpm_compare',formal_protocol:false,
        qa_count:3,planned_turn_count:3,experiment:'W38Max',
        inference_config:{request_profile:'sdk-defaults-v1'},summary:{}};
      state.runs=[run];renderRuns();
      return {list:$('#run-list').innerHTML,panel:minicpmPanel(run)};
    })()""")
    assert '分母3题' in value['list'] and '分母100题' not in value['list']
    assert '不参与历史质量对照' in value['panel']
    assert '100题/111轮' not in value['panel'] and '81题' not in value['panel']


def test_interface_only_v2_comparison_and_status_disclose_policy(pair):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    runtime, root, history, new, old, new_path, old_path = pair
    cfg = MiniCPMInferenceConfig.sdk_defaults(acceptance_policy="interface-only-v2").as_dict()
    new.update(protocol_label=cfg['protocol_version'], gate_summary_version=cfg['protocol_version'], inference_config=cfg)
    atomic_json(new_path, new)
    before = old_path.read_bytes()
    manifest = report.write_comparison(runtime, new['run_id'])
    result = Repository(root, history_roots=[history]).compare([new['run_id'], old['run_id']])
    assert manifest['protocol'] == repository.MINICPM_INTERFACE_PROTOCOL
    assert result['minicpm_scope']['source_hashes_verified']
    assert '参数均未指定' in result['warning']
    assert old_path.read_bytes() == before
    state = report.profile(runtime, Repository(root, history_roots=[history]))
    assert state['schema_version'] == repository.MINICPM_INTERFACE_PROTOCOL
    assert 'Thinking未指定' in state['title']
    value = evaluate("""(()=>{
      const cfg={request_profile:'sdk-defaults-v1',acceptance_policy:'interface-only-v2'};
      return {note:minicpmDifferenceNote(cfg),panel:minicpmPanel({mode:'minicpm_compare',inference_config:cfg}),
        supported:minicpmProtocol('minicpm47-sdk-defaults-m4-off-v2')};
    })()""")
    assert value['supported'] is True
    assert '事实错误保留并计入评分' in value['note']
    assert '接口与结构验收' in value['panel']


def test_q6_comparison_retains_original_history_and_discloses_timing(pair):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    runtime, root, history, new, old, new_path, old_path = pair
    cfg = MiniCPMInferenceConfig.selfhost().as_dict()
    new.update(protocol_label=cfg['protocol_version'], gate_summary_version=cfg['protocol_version'], inference_config=cfg)
    atomic_json(new_path, new)
    before = old_path.read_bytes()
    report.write_comparison(runtime, new['run_id'])
    result = Repository(root, history_roots=[history]).compare([new['run_id'], old['run_id']])
    assert result['minicpm_scope']['protocol'] == cfg['protocol_version']
    assert old_path.read_bytes() == before
    p = report.profile(runtime, Repository(root, history_roots=[history]))
    assert 'Q6' in p['title'] and p['schema_version'] == cfg['protocol_version']
    html = evaluate("minicpmDifferenceNote(" + json.dumps(cfg) + ")")
    assert all(word in html for word in ('Q6_K', '1200', '600', '服务器成本未核算'))
