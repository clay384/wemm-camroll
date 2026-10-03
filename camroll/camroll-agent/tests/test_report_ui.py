import json
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "camroll_eval/static/app.js"


def test_invalid_supplement_is_evidence_only_and_explains_asset_failure():
    result = evaluate("""(()=>{
      const run={run_id:'invalid-run',mode:'supplement',status:'blocked',qa_count:64,
        planned_turn_count:71,planned_eligible_qa_count:52,experiment:'W38Max',
        validity_audit:{valid:false,reason:'图片路径边界失败 <unsafe>'},
        summary:{completed:51,judge_valid_count:49,quality_correct_planned_rate:.59}};
      state.runs=[run]; renderRuns();
      return {list:$('#run-list').innerHTML,notice:supplementNotice(run)};
    })()""")
    assert '实施无效' in result['list']
    assert '仅存证' in result['list']
    assert '不用于模型比较' in result['notice']
    assert '图片路径边界失败 &lt;unsafe&gt;' in result['notice']
    assert '59.0%' not in result['list']


def evaluate(expression):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for report browser-state tests")
    program = r"""
const fs = require('node:fs'), vm = require('node:vm');
const nodes = new Map();
const element = selector => {
  if (!nodes.has(selector)) nodes.set(selector, {
    innerHTML: '', textContent: '', value: '', hidden: false, disabled: false,
    style: {}, dataset: {}, handlers: {}, classList: {toggle() {}},
    addEventListener(name, handler) {this.handlers[name] = handler;},
    scrollIntoView() {}, showModal() {this.open = true;}, close() {this.open = false;},
    insertAdjacentHTML(position, html) {this.innerHTML = html + this.innerHTML;},
    click() {return this.handlers.click?.();}
  });
  return nodes.get(selector);
};
const context = vm.createContext({
  document: {querySelector: element, querySelectorAll: () => [], hidden: false},
  window: {confirm: () => {throw Error('Unexpected mutation confirmation');}},
  fetch: () => {throw Error('Unexpected network request');},
  setInterval: () => 1, devicePixelRatio: 1, console
});
let source = fs.readFileSync(process.argv[2], 'utf8');
source = source.replace('enter().catch(err=>notice(err.message,true));', '');
vm.runInContext(source, context);
Promise.resolve(vm.runInContext(process.argv[3], context)).then(result => {
  process.stdout.write(JSON.stringify(result));
}).catch(error => {console.error(error); process.exit(1);});
"""
    result = subprocess.run([node, "-", str(SCRIPT), expression], input=program,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_expected_skips_never_appear_as_pending_grades_during_an_active_run():
    result = evaluate("""({skip:judgeLabel(null,'skipped_expected','evaluating'),
      pending:judgeLabel(null,'pending','evaluating'),scored:judgeLabel(2,'completed','evaluating')})""")
    assert result == {'skip': '未评分（按协议跳过）', 'pending': '待评分', 'scored': '2 分'}


def test_read_only_health_hides_and_guards_mutation_controls():
    result = evaluate("""(async()=>{
      applyHealth({read_only:true}); renderRuns(); await $('#start').handlers.click();
      return {hidden:$('#start').hidden, disabled:$('#start').disabled,
              configHidden:$('#config').hidden, connection:$('#connection').textContent};
    })()""")
    assert result == {"hidden": True, "disabled": True, "configHidden": True,
                      "connection": "只读比较报告 · 8774"}


def test_four_group_profile_keeps_future_groups_and_historical_reference():
    result = evaluate("""(()=>{
      state.runs=[{run_id:'run-q37',experiment:'Q37',status:'building',qa_count:100,summary:{judge_valid_count:0}}];
      renderProfile({suite_status:'blocked',groups:['Q37','Q38','W37','W38Max'].map(name=>({name,run_id:'run-'+name.toLowerCase()}))});
      return {html:$('#comparison-profile').innerHTML,disabled:$('#compare-four').disabled};
    })()""")
    assert all(name in result["html"] for name in ("Q37", "Q38", "W37", "W38Max"))
    assert "等待本组运行" in result["html"] and "已阻断" in result["html"]
    assert "75 正确 / 14 部分 / 11 错误" in result["html"]
    assert result["disabled"] is True


def test_report_preserves_all_planned_questions_before_execution():
    result = evaluate("""(()=>{
      state.currentData={run_id:'run',status:'building',qa_count:100,
        qa_plan:Array.from({length:100},(_,index)=>({qa_id:'qa-'+index,question:'题'+index,planned_turn_count:1})),
        items:[{qa_id:'qa-0',question:'题0',answer:'回答',judge:{score:2},execution_status:'complete'}]};
      state.current='run'; renderQuestions(); openQuestion('run','qa-99');
      return {count:questionRows(state.currentData).length,html:$('#qa-table').innerHTML,dialog:$('#item-content').innerHTML};
    })()""")
    assert result["count"] == 100
    assert "qa-99" in result["html"] and "尚未开始" in result["html"]
    assert "尚无模型回答、交付图片或评分" in result["dialog"]


def test_experiment_summary_renders_fixed_denominators_and_ranking_audit():
    result = evaluate("""(()=>{
      const run={experiment:'Q37',experiment_config:{embedding_backend:'qwen'},qa_count:100,
        summary:{judge_valid_count:1,quality_unscored_planned_count:99,
          retrieval_diagnostics:{pre_top10_recall_macro:0,post_top10_recall_macro:1/81,metric_count:81},
          local_retrieval_performance:{rerank:{calls:2,errors:0,mean_ms:300,p50_ms:200,p95_ms:400}}},
        build_summary:{image_count:355,image_embedding_count:355,caption_count:354,text_embedding_count:354,event_count:10,event_embedding_count:10}};
      return {unscored:runCounts(run).invalidQa,panel:experimentPanel(run),trace:rankingTrace({tool_trace:[{
        tool:'search',args:{query:'<script>'},result:{retrieval_trace:{images:{candidate_count:31,reranked_count:30,
          candidates:[{id:'img_a',recall_rank:31,final_rank:31,reranked:false,rerank_score:null}]}}}}]})};
    })()""")
    assert result["unscored"] == 99
    assert "355 / 355" in result["panel"] and "354 / 355" in result["panel"]
    assert "最终排序 Top-10 Recall" in result["panel"]
    assert "保留召回顺序" in result["trace"] and "&lt;script&gt;" in result["trace"]
    assert "<script>" not in result["trace"]


def test_run_cost_uses_request_attribution_and_keeps_shared_ledger_separate():
    result = evaluate("""(()=>{
      const run={experiment:'Q37',budget_before:{estimated_spent_cny:20},budget_after:{estimated_spent_cny:35},
        cost_audit:{status:'complete',own_estimated_cny:2.5,own_reserved_cny:.25,request_count:10,
          matched_request_count:10,missing_call_ids:[],unidentified_call_records:0,uncertain_requests:1,
          by_model:[{model:'qwen3.7-flash',estimated_cny:2.5,request_count:10}]}};
      return {cost:runCost(run),shared:ledgerDelta(run),label:runCostLabel(run),
              panel:experimentPanel(run),note:costAuditNote(run)};
    })()""")
    assert result["cost"] == 2.5 and result["shared"] == 15
    assert result["label"] == "¥2.5"
    assert "共享账本同期增量（含其他项目）" in result["panel"]
    assert "<td>¥15</td>" in result["panel"]
    assert "本组费用归属审计" in result["panel"] and "qwen3.7-flash" in result["panel"]
    assert "尚未落盘的在途调用不包含" in result["note"]


@pytest.mark.parametrize("audit", [
    {"status": "partial", "own_estimated_cny": 3, "known_estimated_cny": 2,
     "missing_call_ids": ["missing"], "unidentified_call_records": 0},
    {"status": "partial", "own_estimated_cny": 3, "known_estimated_cny": 2,
     "missing_call_ids": [], "unidentified_call_records": 1},
    {"own_estimated_cny": 3},
])
def test_incomplete_cost_audit_never_presents_total_own_cost(audit):
    result = evaluate("""(()=>{
      const run={cost_audit:""" + json.dumps(audit) + """};
      return {cost:runCost(run),label:runCostLabel(run),note:costAuditNote(run)};
    })()""")
    assert result["cost"] is None
    assert result["label"] == "归属审计未完整"
    if "known_estimated_cny" in audit:
        assert "已归属部分 ¥2" in result["note"]


def test_absent_cost_audit_never_falls_back_to_shared_ledger_delta():
    result = evaluate("""(()=>{
      const run={budget_before:{estimated_spent_cny:20},budget_after:{estimated_spent_cny:40}};
      return {cost:runCost(run),shared:ledgerDelta(run),label:runCostLabel(run),
              noCalls:runCostLabel({cost_audit:{status:'no_recorded_calls',own_estimated_cny:0,missing_call_ids:[]}})};
    })()""")
    assert result == {"cost": None, "shared": 20, "label": "待归属审计", "noCalls": "暂无已记录调用"}


def test_item_initial_view_requests_only_light_projection_and_defers_trace_dom():
    result = evaluate("""(async()=>{
      const requests=[];
      fetch=async path=>{requests.push(path);return {ok:true,json:async()=>({
        qa_id:'qa-one',question:'Synthetic',answer:'Answer',judge:{score:2,reason:'valid'},
        runtime_turns:[{message:'turn',answer:'turn answer',error_type:'example_error'}],
        model_call_metrics:[{trace_index:0,call_type:'agent',input:'DEFERRED_LARGE_PAYLOAD'}],
        tool_trace:[{trace_index:0,tool:'search',result:{hidden:'DEFERRED_TOOL_PAYLOAD'}}]
      })};};
      await openItem('run-one','qa-one');
      return {requests,html:$('#item-content').innerHTML,open:$('#item-dialog').open};
    })()""")
    assert result["requests"] == ["/api/runs/run-one/items/qa-one/view"]
    assert result["open"] is True
    assert "Answer" in result["html"] and "example_error" in result["html"]
    assert 'data-trace-kind="model"' in result["html"] and 'data-trace-kind="tool"' in result["html"]
    assert "DEFERRED_LARGE_PAYLOAD" not in result["html"] and "DEFERRED_TOOL_PAYLOAD" not in result["html"]
    assert 'href="/api/runs/run-one/items/qa-one" download=' in result["html"]


def test_trace_fetches_once_when_opened_and_renders_escaped_input_and_search_ranks():
    result = evaluate("""(async()=>{
      const requests=[],body={};
      fetch=async path=>{requests.push(path);return {ok:true,json:async()=>({
        tool:'search',args:{query:'<script>unsafe</script>'},result:{retrieval_trace:{images:{
          candidate_count:1,reranked_count:1,candidates:[{id:'img_one',recall_rank:2,final_rank:1,reranked:true,rerank_score:.9}]
        }}}
      })};};
      const details={open:false,dataset:{traceKind:'tool',traceIndex:'3'},querySelector:()=>body};
      await loadTrace(details,'/api/runs/run/items/qa');
      const closedRequests=requests.length;
      details.open=true;await loadTrace(details,'/api/runs/run/items/qa');
      await loadTrace(details,'/api/runs/run/items/qa');
      return {closedRequests,requests,loaded:details.dataset.loaded,html:body.innerHTML};
    })()""")
    assert result["closedRequests"] == 0
    assert result["requests"] == ["/api/runs/run/items/qa/trace/tool/3"]
    assert result["loaded"] == "true"
    assert "召回排名" in result["html"] and "img_one" in result["html"]
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in result["html"] and "<script>" not in result["html"]


def test_trace_error_does_not_mark_loaded_and_can_be_retried():
    result = evaluate("""(async()=>{
      const body={},details={open:true,dataset:{traceKind:'model',traceIndex:'0'},querySelector:()=>body};
      let calls=0;
      fetch=async()=>{calls++;return {ok:calls>1,json:async()=>calls===1?{detail:'temporary'}:{input:'retried'}};};
      await loadTrace(details,'/base');
      const first={loaded:details.dataset.loaded??null,message:body.textContent};
      await loadTrace(details,'/base');
      return {calls,first,loaded:details.dataset.loaded,html:body.innerHTML};
    })()""")
    assert result["first"]["loaded"] is None and "temporary" in result["first"]["message"]
    assert result["calls"] == 2 and result["loaded"] == "true"
    assert "retried" in result["html"]


def test_recovery_supplement_keeps_64_71_52_denominators_and_has_no_cold_build_label():
    result = evaluate("""(()=>{
      const run={run_id:'supplement-64',parent_run_id:'original-w38max',mode:'supplement',
        experiment:'W38Max',experiment_config:{generation_model:'qwen3.8-max',embedding_backend:'wemm'},
        config:'M4',formal_protocol:false,status:'blocked',stop_reason:'http_403',qa_count:64,
        planned_turn_count:71,planned_eligible_qa_count:52,build_skipped:true,wall_ms:12000,
        index_version:'frozen-snapshot',source_unchanged:true,query_cache_policy:'fresh_clients_no_parent_cache',
        items:[{qa_id:'qa-0',question:'Synthetic question',judge:{score:2},execution_status:'error'}],
        qa_plan:Array.from({length:64},(_,index)=>({qa_id:'qa-'+index,question:'Question '+index,planned_turn_count:1})),
        summary:{judge_distribution:{'2':1,'1':0,'0':0},judge_valid_count:1,quality_unscored_planned_count:63,
          quality_correct_planned_rate:1/64,retrieval_metric_count:52,retrieval_recall_macro:1/52,
          executed_turn_count:1,task_decision_valid_count:1}};
      state.current='supplement-64';state.currentData=run;state.runs=[run];renderRuns();renderDetail(run);
      return {label:label(run),e2e:e2eLabel(run),rows:questionRows(run).length,list:$('#run-list').innerHTML,
        html:$('#detail').innerHTML,counts:runCounts(run)};
    })()""")
    assert "恢复后64题补测·复用快照" in result["label"]
    assert result["e2e"] == "不适用（未建库）" and result["rows"] == 64
    assert result["counts"]["plannedQa"] == 64 and result["counts"]["plannedTurns"] == 71
    assert 'disabled title="补测独立展示"' in result["list"]
    assert "恢复后补测 · 独立展示" in result["list"] and "分母64题" in result["list"]
    assert "smoke" not in result["list"] and "分母100题" not in result["list"]
    for expected in ("固定64题正确率", "64 题 / 71 轮", "52 道可回答题", "12道不可回答题",
                     "补测资源与费用", "本次没有建库阶段", "本次没有建库费用或建库耗时",
                     "http_403", "原四组首跑结果保持独立", "fresh_clients_no_parent_cache"):
        assert expected in result["html"]
    for incorrect in ("固定100题正确率", "19道不可回答题", "独立建库覆盖与资源", "等待阶段记录"):
        assert incorrect not in result["html"]


def test_recovery_supplement_never_replaces_first_run_four_group_card():
    result = evaluate("""(()=>{
      state.runs=[{run_id:'original-w38max',experiment:'W38Max',qa_count:100,
        status:'completed_with_errors',summary:{judge_valid_count:36}},
        {run_id:'supplement-64',parent_run_id:'original-w38max',experiment:'W38Max',mode:'supplement',
        qa_count:64,status:'completed',summary:{judge_valid_count:64}}];
      renderProfile({groups:['Q37','Q38','W37','W38Max'].map(name=>({name,
        run_id:name==='W38Max'?'original-w38max':'original-'+name})),suite_status:'incomplete'});
      return $('#comparison-profile').innerHTML;
    })()""")
    assert result.count('class="group-card secondary"') == 4
    assert "36 / 100" in result and "supplement-64" not in result and "64 / 64" not in result


def test_recovery_supplement_cannot_be_added_to_formal_comparison():
    result = evaluate("""(async()=>{
      state.runs=[{run_id:'supplement-64',mode:'supplement'},{run_id:'original-w38max',mode:'full'}];
      state.selected=new Set(['supplement-64','original-w38max']);
      await $('#compare').handlers.click();
      return {notice:$('#notice').textContent,comparison:$('#comparison').innerHTML};
    })()""")
    assert "补测独立展示" in result["notice"] and result["comparison"] == ""


def test_resource_labels_separate_whole_gpu_from_verified_run_processes():
    result = evaluate("""(()=>{
      const run={run_id:'run',experiment:'Q38',experiment_config:{embedding_backend:'qwen'},
        qa_count:100,planned_turn_count:111,summary:{},items:[],
        resource_summary:{gpu_memory_mib_peak:21377,qwen_process_memory_mib_peak:5020,samples:7767}};
      state.currentData=run;renderDetail(run);renderComparison({runs:[run],rows:[],warning:'test'});
      return {detail:$('#detail').innerHTML,comparison:$('#comparison').innerHTML};
    })()""")
    for html in result.values():
        assert "GPU 1整卡采样峰值（含其他进程）" in html
        assert "本轮已核验模型进程采样峰值" in html
        assert "5,020 MiB" in html
    assert "不能视为本轮独占资源" in result["detail"]
    assert "仅统计本轮身份核验通过的进程" in result["detail"]
