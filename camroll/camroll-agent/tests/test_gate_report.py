import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json
from camroll_eval.repository import ComparisonScopeError, Repository
from camroll_eval.service import create_app
from test_report_ui import evaluate


def gate_state(name='F37-On'):
    return {'run_id': name, 'mode': 'gate_ab', 'protocol_label': 'camroll-flash-gate-v1',
            'suite_id': 'synthetic-suite',
            'gate_summary_version': 'camroll-flash-gate-v1', 'formal_protocol': True,
            'index_binding': {'snapshot_version': 'frozen', 'album_manifest_sha256': 'unchanged'},
            'inference_config': {'group': name, 'generation_model': 'qwen3.7-flash', 'gate_policy': 'repair-evidence-v1'},
            'qa_count': 100, 'planned_turn_count': 111, 'planned_eligible_qa_count': 81,
            'status': 'completed_with_errors', 'source_unchanged': True,
            'summary': {'judge_valid_count': 1, 'quality_correct_planned_rate': .01,
                        'quality_mean_fixed_planned': .02, 'retrieval_metric_count': 81},
            'qa_plan': [{'qa_id': 'synthetic', 'planned_turn_count': 1}],
            'items': [{'qa_id': 'synthetic', 'answer_origin': 'gate_exhausted',
                       'gate_events': [{'state': 'rejected', 'candidate': {'content': 'synthetic candidate'},
                                        'error': {'code': 'synthetic_invalid'}}]}]}


def test_gate_summary_not_replaced_by_legacy_metrics_and_audit_is_lazy(tmp_path):
    state = gate_state()
    path = tmp_path / 'runs/F37-On/run.json'
    atomic_json(path, state)
    before = path.read_bytes()
    repo = Repository(tmp_path)
    assert repo.get_brief('F37-On')['summary'] == state['summary']
    assert repo.detail('F37-On')['items'][0]['answer_origin'] == 'gate_exhausted'
    assert repo.item('F37-On', 'synthetic', light=True)['trace_counts']['gate'] == 1
    assert repo.item_trace('F37-On', 'synthetic', 'gate', 0) == state['items'][0]['gate_events'][0]
    assert path.read_bytes() == before


def test_gate_comparison_requires_identical_binding_and_protocol(tmp_path):
    a, b = gate_state(), gate_state('F38-On')
    for row in (a, b):
        atomic_json(tmp_path / 'runs' / row['run_id'] / 'run.json', row)
    repo = Repository(tmp_path)
    assert repo.compare(['F37-On', 'F38-On'])['comparison_scope'] == 'frozen_index_flash_gate_comparison'
    b['index_binding']['snapshot_version'] = 'other'
    atomic_json(tmp_path / 'runs/F38-On/run.json', b)
    with pytest.raises(ValueError, match='same immutable'):
        repo.compare(['F37-On', 'F38-On'])
    b.update(mode='full')
    atomic_json(tmp_path / 'runs/F38-On/run.json', b)
    with pytest.raises(ValueError, match='separately'):
        repo.compare(['F37-On', 'F38-On'])


def test_gate_report_execution_progress_is_independent_of_judge_completeness(tmp_path):
    state = gate_state('F37-Off')
    state['summary'].update(completed=99, judge_valid_count=91,
                            quality_unscored_planned_count=9)
    state['items'] = [{'qa_id': f'synthetic-{index}', 'final_turn_executed': True,
                       'judge_status': 'failed' if index == 0 else 'completed',
                       'runtime_turns': [{}] * (2 if index < 11 else 1)}
                      for index in range(100)]
    path = tmp_path / 'runs/F37-Off/run.json'
    atomic_json(path, state)
    before = path.read_bytes()
    app = create_app(tmp_path, tmp_path / 'source', tmp_path / 'albums',
                     read_only=True, mount_agent=False)
    with TestClient(app) as client:
        views = [client.get('/api/runs').json()['runs'][0],
                 client.get('/api/runs/F37-Off').json(),
                 client.get('/api/compare?ids=F37-Off').json()['runs'][0]]
        for view in views:
            assert view['report_progress'] == {'finished_question_count': 100, 'executed_turn_count': 111}
            assert view['summary'] == state['summary']
        assert 'report_progress' not in app.state.repository.load('F37-Off')
        assert client.get('/api/runs/F37-Off/export').content == before == path.read_bytes()

    # A started multi-turn question is not an ended question; expected skips
    # and failed execution still count as executed once recorded.
    state['items'] = [
        {'qa_id': 'partial', 'final_turn_executed': False, 'runtime_turns': [{}]},
        {'qa_id': 'failure', 'final_turn_executed': True, 'judge_status': 'skipped_expected',
         'runtime_turns': [{'answer_origin': 'gate_exhausted'}]},
        {'qa_id': 'unstarted', 'runtime_turns': []}]
    atomic_json(path, state)
    assert app.state.repository.get_brief('F37-Off')['report_progress'] == {
        'finished_question_count': 1, 'executed_turn_count': 2}


@pytest.mark.parametrize('suite_a,suite_b', [
    ('current', 'historical'), ('current', None), (None, None), ('current', ''), (' ', ' ')])
def test_gate_cross_suite_comparison_fails_closed_with_explanation(tmp_path, suite_a, suite_b):
    for name, suite in [('F37-On', suite_a), ('F38-On', suite_b)]:
        state = gate_state(name)
        state['suite_id'] = suite
        atomic_json(tmp_path / 'runs' / name / 'run.json', state)
    repo = Repository(tmp_path)
    with pytest.raises(ComparisonScopeError, match='批次'):
        repo.compare(['F37-On', 'F38-On'])
    app = create_app(tmp_path, tmp_path / 'source', tmp_path / 'albums',
                     read_only=True, mount_agent=False)
    with TestClient(app) as client:
        response = client.get('/api/compare?ids=F37-On,F38-On')
        assert response.status_code == 422
        assert '批次' in response.json()['detail'] and 'run not found' not in response.text
        assert client.get('/api/compare?ids=F37-On').status_code == 200


def test_gate_same_suite_history_remains_comparable(tmp_path):
    current, historical = tmp_path / 'current', tmp_path / 'historical'
    for name in ('F37-Off', 'F37-On'):
        atomic_json(historical / 'runs' / name / 'run.json', gate_state(name))
    app = create_app(current, tmp_path / 'source', tmp_path / 'albums', read_only=True,
                     mount_agent=False, history_roots=[historical])
    with TestClient(app) as client:
        assert client.get('/api/compare?ids=F37-Off,F37-On').status_code == 200


def test_history_root_preserves_old_cost_and_invalidity_overlays(tmp_path):
    new, old = tmp_path / 'new', tmp_path / 'old'
    atomic_json(new / 'runs/F37-On/run.json', gate_state())
    historical = {'run_id': 'old-supplement', 'mode': 'supplement', 'status': 'blocked',
                  'supplement_summary_version': 'http403-supplement-v1', 'summary': {}}
    old_path = old / 'runs/old-supplement/run.json'
    atomic_json(old_path, historical)
    before = old_path.read_bytes()
    atomic_json(old / 'supplement-cost-audit.json', {'runs': {'old-supplement': {'own_estimated_cny': 75}}})
    atomic_json(old / 'supplement-validity-audit.json', {'runs': {'old-supplement': {'valid': False}}})
    app = create_app(new, tmp_path / 'source', tmp_path / 'albums', read_only=True, mount_agent=False,
                     history_roots=[old])
    with TestClient(app) as client:
        listing = {row['run_id']: row for row in client.get('/api/runs').json()['runs']}
        assert set(listing) == {'F37-On', 'old-supplement'}
        assert listing['old-supplement']['cost_audit']['own_estimated_cny'] == 75
        assert listing['old-supplement']['validity_audit']['valid'] is False
        assert client.get('/api/runs/old-supplement/export').content == before
        assert client.post('/api/runs').status_code == 405
        assert client.get('/api/runs/F37-On/items/synthetic/trace/gate/0').status_code == 200
    assert old_path.read_bytes() == before


def test_gate_ui_never_labels_shared_index_as_cold_build():
    result = evaluate("""(()=>{
      const run={run_id:'g',mode:'gate_ab',qa_count:100,inference_config:{group:'F37-On',
        generation_model:'qwen3.7-flash',gate_policy:'repair-evidence-v1'},
        summary:{judge_valid_count:1,quality_correct_planned_rate:.01,quality_mean_fixed_planned:.02}};
      state.runs=[run]; renderRuns();
      renderProfile({schema_version:'camroll-flash-gate-v1',groups:[{name:'F37-On',run_id:'g'}]});
      return {list:$('#run-list').innerHTML,panel:gatePanel(run),description:$('#report-description').textContent,
        trace:traceSection([{state:'rejected'}],'gate'),time:e2eLabel(run)};
    })()""")
    assert 'F37-On' in result['list'] and '1.0%' in result['list']
    assert '独立冷建库' not in result['list']
    assert '共享冻结' in result['description']
    assert '固定100题' in result['panel'] and '同模型关联' in result['panel']
    assert '全部候选' in result['trace']
    assert result['time'] == '不适用（未建库）'


def test_gate_ui_separates_execution_scoring_and_historical_batch():
    result = evaluate("""(async()=>{
      const old={run_id:'old-off',suite_id:'old',mode:'gate_ab',status:'completed_with_errors',
        qa_count:100,planned_turn_count:111,inference_config:{group:'F37-Off'},
        report_progress:{finished_question_count:100,executed_turn_count:111},
        summary:{completed:99,judge_valid_count:91,quality_unscored_planned_count:9,
          quality_correct_planned_rate:.5}};
      const current={...old,run_id:'current-off',suite_id:'current',status:'evaluating',
        report_progress:{finished_question_count:1,executed_turn_count:2}};
      state.runs=[old,current];renderRuns();
      const unknown=gateBatchBadge(old);
      fetch=async()=>({ok:true,json:async()=>({schema_version:'camroll-flash-gate-v1',
        groups:[{name:'F37-Off',run_id:'current-off'}]})});
      await loadProfile();
      return {html:$('#run-list').innerHTML,unknown,
        current:gateBatchBadge(current),old:gateBatchBadge(old)};
    })()""")
    assert '已结束问题 100 / 100' in result['html']
    assert '已执行轮次 111 / 111' in result['html']
    assert '已结束问题 1 / 100' in result['html']
    assert '有效评分 91 / 100' in result['html']
    assert '99 / 100' not in result['html']
    assert '已执行 · 有错误' in result['html'] and '50.0%' in result['html']
    assert '实施无效' not in result['html']
    assert '批次未确认' in result['unknown'] and '历史批次' not in result['unknown']
    assert '当前批次' in result['current'] and '历史批次' not in result['current']
    assert '历史批次' in result['old'] and '历史批次' in result['html']


def test_default_gate_comparison_selects_only_current_profile_run_ids():
    result = evaluate("""(()=>{
      const groups=['F37-Off','F37-On','F38-Off','F38-On'];
      state.runs=['old','current'].flatMap(suite=>groups.map(group=>({
        run_id:suite+'-'+group,suite_id:suite,mode:'gate_ab',inference_config:{group}})));
      state.selected=new Set(['old-F37-Off']);
      const profile={schema_version:'camroll-flash-gate-v1',groups:groups.map(name=>({
        name,run_id:'current-'+name}))};
      let selected;$('#compare').handlers.click=()=>{selected=[...state.selected];};
      renderProfile(profile);$('#compare-four').onclick();
      return selected;
    })()""")
    assert result == [f'current-{group}' for group in ('F37-Off', 'F37-On', 'F38-Off', 'F38-On')]
