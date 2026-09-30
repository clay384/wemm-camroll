import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from four_supplement import validate_probe, validate_supplement


def state():
    turn = {'judge': {'score': 0}, 'task_judge': {'actual_action': 'none', 'correct': False}}
    items = [{'qa_id': str(i), 'runtime_turns': [copy.deepcopy(turn)], 'judge_status': 'completed', 'judge': {'score': 0}} for i in range(64)]
    for item in items[:7]:
        item['runtime_turns'].append(copy.deepcopy(turn))
    return {'mode': 'supplement', 'status': 'completed_with_errors', 'items': items,
            'qa_plan': [{'qa_id': item['qa_id'], 'planned_turn_count': len(item['runtime_turns'])} for item in items]}


def test_supplement_preserves_wrong_answers_but_requires_every_turn():
    s = state()
    assert validate_supplement(s)
    s['items'][0]['runtime_turns'].pop()
    s['items'][-1]['runtime_turns'].append(copy.deepcopy(s['items'][-1]['runtime_turns'][0]))
    assert sum(len(i['runtime_turns']) for i in s['items']) == 71
    assert not validate_supplement(s)


def test_supplement_missing_and_invalid_scores_do_not_pass():
    for key in ('missing_item', 'missing_turn_score', 'invalid_score_type', 'missing_action', 'missing_behavior_score', 'invalid_plan'):
        s = state()
        if key == 'missing_item':
            s['items'].pop()
        elif key == 'missing_turn_score':
            s['items'][0]['runtime_turns'][0]['judge']['score'] = None
        elif key == 'invalid_score_type':
            s['items'][0]['judge']['score'] = True
        elif key == 'missing_action':
            s['items'][0]['runtime_turns'][0]['task_judge'] = {}
        elif key == 'missing_behavior_score':
            s['items'][0]['runtime_turns'][0]['task_judge'].pop('correct')
        else:
            s['qa_plan'][0]['planned_turn_count'] = True
        assert not validate_supplement(s)


def test_probe_requires_fresh_successful_max_capabilities():
    probe = {'status': 'passed', 'model': 'qwen3.8-max', 'source': 'synthetic_nonformal', 'finished_at': 'recorded',
             'checks': [{'kind': k, 'passed': True, 'actual_model': 'qwen3.8-max'} for k in ('agent-tool', 'image', 'structured')]}
    assert validate_probe(probe)
    for kind in ('wrong_model', 'missing_tool', 'failed_image'):
        broken = copy.deepcopy(probe)
        if kind == 'wrong_model':
            broken['checks'][0]['actual_model'] = 'qwen3.8-flash'
        elif kind == 'missing_tool':
            broken['checks'].pop(0)
        else:
            broken['checks'][1]['passed'] = False
        assert not validate_probe(broken)


def test_unregistered_owned_child_is_stopped_and_wait_is_bounded():
    import subprocess
    from four_supplement import stop_owned_child
    class Child:
        returncode=None
        def __init__(self): self.actions=[]
        def poll(self): return self.returncode
        def terminate(self): self.actions.append('terminate')
        def kill(self): self.actions.append('kill')
        def wait(self, timeout):
            self.actions.append(('wait', timeout))
            if timeout==15: raise subprocess.TimeoutExpired('owned-child',timeout)
            self.returncode=-9
    child=Child()  # Registration failed before entering the window children list.
    result=stop_owned_child(child)
    assert child.actions==['terminate',('wait',15),'kill',('wait',5)]
    assert result['state']=='killed_after_timeout'
    assert stop_owned_child(child)['state']=='absent_or_exited'


def test_registration_failure_cleans_child_before_restoring_gpu_window(tmp_path, monkeypatch):
    import json
    import pytest
    import four_supplement as target
    import four_gpu_window
    import camroll_agent.budget as budget_module
    runtime=tmp_path/'runtime';(runtime/'supplements').mkdir(parents=True)
    (runtime/'formal-suite.json').write_text(json.dumps({'restoration':{'state':'restored'}}))
    (runtime/'freeze.json').write_text(json.dumps({'wemm_fingerprint':'fp','model_manifest_sha256':'models','model_files':{}}))
    probe={'status':'passed','model':'qwen3.8-max','source':'synthetic_nonformal','finished_at':'recorded',
           'checks':[{'kind':k,'passed':True,'actual_model':'qwen3.8-max'} for k in ('agent-tool','image','structured')]}
    probe_path=runtime/'supplements/probe.json';probe_path.write_text(json.dumps(probe))
    events=[]
    class Ledger:
        def __init__(self,*a,**kw):pass
        def check_limit(self):pass
        def summary(self):return {'reserved_cny':0,'uncertain_requests':0,'limit_cny':600}
    class Window:
        def __init__(self,*a,**kw):self.state_path=runtime/'window.json'
        def enter(self):
            self.state_path.write_text('{}')
            return {'before':{'fingerprint':'fp'},'model_integrity':{'manifest_sha256':'models','models':{}}}
        def switch(self,kind):events.append('switch')
        def register_child(self,*a,**kw):raise RuntimeError('registration_failed')
        def restore(self,*a):events.append('restore');return {'state':'restored'}
    class Child:
        pid=123;returncode=None
        def __init__(self,*a,**kw):events.append('spawn')
        def poll(self):return self.returncode
        def terminate(self):events.append('terminate');self.returncode=-15
        def wait(self,timeout):events.append('wait')
    monkeypatch.setattr(target,'RUNTIME',runtime)
    monkeypatch.setattr(target,'configure',lambda release:None)
    monkeypatch.setattr(budget_module,'BudgetLedger',Ledger)
    monkeypatch.setattr(four_gpu_window,'GpuWindow',Window)
    monkeypatch.setattr(target.subprocess,'Popen',Child)
    monkeypatch.setenv('CAMROLL_BUDGET_DB',str(tmp_path/'unused.sqlite'))
    monkeypatch.setenv('CAMROLL_RELEASE_REVISION','test')
    with pytest.raises(RuntimeError,match='registration_failed'):
        target.main(['--parent-id','parent','--run-id','new','--probe',str(probe_path)])
    assert events==['switch','spawn','terminate','wait','restore']
    result=json.loads((runtime/'supplements/new/supervisor.json').read_text())
    assert result['status']=='blocked' and result['restoration']['state']=='restored'
    assert result['first_suite_unchanged'] is True
