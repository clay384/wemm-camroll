import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from camroll_eval.common import atomic_json
from camroll_eval.repository import Repository
from camroll_eval.service import create_app
from test_report_ui import evaluate


def gate_state(name='F37-On'):
    return {'run_id': name, 'mode': 'gate_ab', 'protocol_label': 'camroll-flash-gate-v1',
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
