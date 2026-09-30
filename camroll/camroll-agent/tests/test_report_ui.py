import json
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "camroll_eval/static/app.js"


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
