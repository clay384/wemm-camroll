import json
import sqlite3
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from four_supplement_export import export


@pytest.mark.parametrize('invalid', [False, True])
def test_supplement_has_own_denominators_and_never_bills_parent_build(tmp_path, invalid):
    run_id='new-supplement'; folder=tmp_path/'evaluation/runs'/run_id;folder.mkdir(parents=True)
    plans=[{'qa_id':str(i),'planned_turn_count':2 if i<7 else 1} for i in range(64)]
    item={'qa_id':'0','answerability':'answerable','judge':{'score':2},'runtime_turns':[{'judge':{'score':2},'task_judge':{'actual_action':'answer','correct':True}}],
          'image_retrieval_counts':{'gt':1,'predicted':1,'matched':1},'model_call_metrics':[{'call_id':'fresh','call_type':'agent','model':'qwen3.8-max'}]}
    state={'run_id':run_id,'parent_run_id':'old','mode':'supplement','status':'completed_with_errors','planned_eligible_qa_count':52,'qa_count':64,'planned_turn_count':71,'qa_plan':plans,'items':[item]}
    path=folder/'run.json';path.write_text(json.dumps(state));original=path.read_bytes()
    (folder/'judge_calls.jsonl').write_text('')
    old_cost=tmp_path/'evaluation/cost-audit.json';old_cost.write_text('{"original": true}')
    old_report=tmp_path/'reports/four-models.json';old_report.parent.mkdir();old_report.write_text('{"original": true}')
    if invalid:
        (tmp_path/'evaluation/supplement-validity-audit.json').write_text(json.dumps(
            {'runs': {run_id: {'valid': False, 'reason': 'Asset boundary mismatch'}}}))
    ledger=tmp_path/'budget.sqlite'
    with sqlite3.connect(ledger) as c:
        c.execute('CREATE TABLE budget_calls (id TEXT,status TEXT,reserved_units INTEGER,charged_units INTEGER,metadata TEXT)')
        for identifier,cost in [('parent-build',99000000),('fresh',100000)]:
            c.execute('INSERT INTO budget_calls VALUES (?,?,?,?,?)',(identifier,'settled',0,cost,json.dumps({'model':'qwen3.8-max','response_model':'qwen3.8-max','prompt_tokens':1,'completion_tokens':1})))
    result=export(tmp_path,run_id,ledger_path=ledger)
    assert result['accuracy_fixed_planned']==1/64
    assert result['final_delivery_macro']['f1']==pytest.approx(1/52)
    assert result['quality_distribution']['unscored']==63
    assert len(result['questions'])==64 and result['questions'][-1]['execution_status']=='not_started'
    assert result['cost_audit']['request_ids']==['fresh']
    assert result['cost_audit']['own_estimated_cny']==.1
    assert not result['build_performed'] and result['build_cost_cny']==0
    assert not result['acceptance_complete'] and not result['formal_first_attempt_replaced']
    assert path.read_bytes()==original
    assert old_cost.read_text()==old_report.read_text()=='{"original": true}'
    overlay=json.loads((tmp_path/'evaluation/supplement-cost-audit.json').read_text())
    assert set(overlay['runs'])=={run_id}
    if invalid:
        assert result['validity_audit']['valid'] is False
        assert '不用于模型比较' in result['title']
        assert 'Asset boundary mismatch' in (tmp_path/'reports'/f'{run_id}.md').read_text()
