'use strict';
const $=s=>document.querySelector(s), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={runs:[],selected:new Set(),current:null,currentData:null,poll:null,readOnly:true,profile:null};
const pct=v=>typeof v==='number'?(v*100).toFixed(1)+'%':'—';
const num=(v,n=1)=>typeof v==='number'?v.toLocaleString('zh-CN',{maximumFractionDigits:n}):'—';
const duration=v=>typeof v!=='number'?'—':v>=60000?(v/60000).toFixed(1)+' min':(v/1000).toFixed(1)+' s';
const statusName={preparing:'准备中',building:'建库中',evaluating:'问答中',judging:'评分中',completed:'已完成',passed:'四组已完成',completed_with_errors:'已执行 · 有错误',failed:'失败',cancelled:'已取消',cancelling:'取消中',starting:'正在启动',not_started:'尚未开始',scheduled:'待执行',running:'执行中',blocked:'已阻断',incomplete:'尚未完整',interrupted:'已中断',restore_failed:'恢复失败'};
const alive=s=>['preparing','building','evaluating','judging','starting','cancelling'].includes(s);
const badge=s=>`<span class="badge ${['completed','passed'].includes(s)?'good':['failed','blocked','restore_failed'].includes(s)?'bad':alive(s)||['incomplete','interrupted'].includes(s)?'warn':''}">${esc(statusName[s]||s||'未记录')}</span>`;
const label=r=>r.experiment?`${r.experiment} · ${r.experiment_config?.generation_model||''} · ${r.experiment_config?.embedding_backend==='qwen'?'Qwen Embedding + Reranker':'WeMM'} · ${r.config}`:r.system_id==='sentrix'?'Sentrix · '+(r.model_name||r.model_profile||'历史模型')+' · '+(r.mode==='reuse'?'复用测评':'全链路'):`${(r.config||'M1')==='M1'?'历史 M1 参考':'原协议 CamRoll'} · WeMM · `+(r.mode==='composite'?'补齐后合并':r.mode==='repair'?'定点补跑':r.formal_protocol===false?'smoke':'首次结果');
function runCounts(r){
  const s=r.summary||{},plannedQa=r.qa_count??s.planned_count??s.total;
  // The imported full album3 protocol has 111 planned turns even when an old
  // result saved fewer task judgments. Smoke runs retain their own denominator.
  const fullAlbum3=r.qa_set==='100qa-full'&&plannedQa===100&&
    ((r.dataset_id||r.album_id)==='album3-max'||r.formal_protocol===true);
  return {plannedQa,validQa:s.judge_valid_count,invalidQa:s.quality_unscored_planned_count??s.judge_invalid_count,
    plannedTurns:r.planned_turn_count??(fullAlbum3?111:s.task_decision_planned_count??
      (typeof s.executed_turn_count==='number'&&typeof s.unexecuted_turn_count==='number'?s.executed_turn_count+s.unexecuted_turn_count:undefined)),
    executedTurns:s.executed_turn_count,validTask:s.task_decision_valid_count,
    executionErrors:s.execution_error_count};
}
const countRatio=(actual,planned)=>`${num(actual,0)} / ${num(planned,0)}`;
const executionErrorLabel=(r,counts)=>r.system_id==='sentrix'?'未记录（旧字段）':num(counts.executionErrors,0);
function legacyE2EInvalid(r){
  const elapsed=r.summary?.benchmark_e2e_latency_excluding_judge_ms;
  const seconds=r.phases?.pipeline_processing?.total_seconds;
  const pipelineMs=typeof seconds==='number'?seconds*1000:r.phases?.pipeline?.wall_ms;
  return r.system_id==='sentrix'&&typeof elapsed==='number'&&typeof pipelineMs==='number'&&elapsed<pipelineMs;
}
const e2eLabel=r=>['reuse','repair'].includes(r.mode)?'不适用（未建库）':legacyE2EInvalid(r)?'旧估算不适用':duration(r.summary?.benchmark_e2e_latency_excluding_judge_ms);
function judgeLabel(score,judgeStatus,runStatus){
  if([0,1,2].includes(score))return `${score} 分`;
  return alive(runStatus)&&!['failed','incomplete','cancelled','skipped'].includes(judgeStatus)?'待评分':'未评分';
}
async function api(path,options={}){const response=await fetch(path,{credentials:'same-origin',...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});let body;try{body=await response.json();}catch{body={detail:'响应格式异常'};}if(!response.ok)throw Error(body.detail||body.error||'请求失败');return body;}
function notice(text,error=false){$('#notice').hidden=false;$('#notice').textContent=text;$('#notice').classList.toggle('error',error);}
function applyHealth(health){
  state.readOnly=health.read_only!==false;
  for(const selector of ['#start','#config']){$(selector).hidden=state.readOnly;$(selector).disabled=state.readOnly;}
  if(state.readOnly){$('#connection').textContent='只读比较报告 · 8774';$('#report-description').textContent='四组独立冷建库；此页面展示结果与进度，所有缺失和失败记录均保留。';}
}
async function enter(){applyHealth(await api('/api/health'));await refresh();await loadProfile();if(!state.poll)state.poll=setInterval(async()=>{if(document.hidden)return;try{await refresh(false);if(state.current&&state.currentData&&alive(state.currentData.status))await openRun(state.current,false);}catch{}},5000);}
async function loadProfile(){
  try{
    const p=await api('/api/comparison-profile');state.profile=p;renderProfile(p);
  }catch(e){notice(e.message,true);}
}
function renderProfile(p){
  if(p.groups?.length){
    const observed=new Map(state.runs.map(run=>[run.run_id,run]));
    const models={Q37:'Qwen 检索 + qwen3.7-flash',Q38:'Qwen 检索 + qwen3.8-flash',W37:'WeMM + qwen3.7-flash',W38Max:'WeMM + qwen3.8-max'};
    $('#comparison-profile').hidden=false;
    $('#comparison-profile').innerHTML=`<div class="panel-body"><div class="profile-heading"><div><h2>${esc(p.title||'Qwen / WeMM 四组对照')}</h2><p class="notes">M4 · 355 张图 · 100 道题 / 111 轮 · Agent 1 / Judge 4 · Doubao 统一评分</p></div>${badge(p.suite_status||'running')}</div><div class="group-grid">${p.groups.map(group=>{const run=observed.get(group.run_id),counts=run?runCounts(run):{plannedQa:group.qa_count||100,validQa:0};return `<button class="group-card secondary" ${run?`data-profile-run="${esc(group.run_id)}"`:'disabled'}><strong>${esc(group.name)}</strong><span>${esc(models[group.name]||group.name)}</span>${badge(run?.status||'not_started')}<small>有效评分 ${countRatio(counts.validQa??0,counts.plannedQa)} · ${run?'查看运行记录':'等待本组运行'}</small></button>`;}).join('')}</div><div class="actions"><button id="compare-four" class="secondary">比较已生成的四组记录</button></div><p class="notes">${esc(p.note||'四组均为独立冷建库，比较完整系统效果。固定100题正确率；最终交付P/R/F1固定81题，未执行贡献零。')}</p><div class="historical-note">历史 M1：75 正确 / 14 部分 / 11 错误，仅作历史参考。${(p.historical||[]).filter(row=>row.run_id&&observed.has(row.run_id)).map(row=>` <button class="quiet" data-profile-run="${esc(row.run_id)}">${esc(row.label||'查看历史记录')}</button>`).join('')}</div></div>`;
    document.querySelectorAll('[data-profile-run]').forEach(button=>button.onclick=()=>openRun(button.dataset.profileRun));
    const available=p.groups.map(group=>group.run_id).filter(id=>observed.has(id));
    $('#compare-four').disabled=available.length<2;
    $('#compare-four').onclick=()=>{state.selected=new Set(available);renderRuns();$('#compare').click();};
    return;
  }
  if(!p.original_main?.length)return;
    $('#comparison-profile').hidden=false;
    $('#comparison-profile').innerHTML=`<div class="panel-body"><h2>四条系统线</h2><p>我们 / Gemma4-E2B 基线 / Doubao 全链路 / Qwen3.5-4B 复用。Kimi为辅助结果。</p><div class="actions"><button id="original-four" class="secondary">首次结果 · 四线比较</button><button id="repaired-four" ${p.repaired_main?.length?'':'disabled'}>补齐后 · 四线比较</button><label><input type="checkbox" id="include-kimi"> 加入 Kimi 辅助</label></div><p class="notes">${esc(p.caption_note||'')} 补齐后质量与检索来自合并记录；性能保持首次结果，补跑耗时与费用另列。复用测评没有建库时间。</p></div>`;
    for(const [button,key] of [['#original-four','original_main'],['#repaired-four','repaired_main']])$(button).onclick=()=>{
      state.selected=new Set([...(p[key]||[]),...($('#include-kimi').checked?(p.auxiliary||[]):[])]);renderRuns();$('#compare').click();
    };
}
$('#refresh').addEventListener('click',()=>refresh());
$('#close-dialog').addEventListener('click',()=>$('#item-dialog').close());
$('#item-dialog').addEventListener('click',e=>{if(e.target===$('#item-dialog'))$('#item-dialog').close();});
async function refresh(clear=false){try{state.runs=(await api('/api/runs')).runs;renderRuns();if(state.profile)await loadProfile();if(clear)$('#notice').hidden=true;}catch(err){notice(err.message,true);}}
function renderRuns(){const rows=state.runs.map(r=>{const s=r.summary||{},d=s.judge_distribution||{},counts=runCounts(r);return `<tr><td><input type="checkbox" class="compare-check" data-id="${esc(r.run_id)}" ${state.selected.has(r.run_id)?'checked':''} aria-label="选择比较"></td><td><span class="run-name" data-run="${esc(r.run_id)}">${esc(label(r))}<small>${esc(r.run_id)}</small></span></td><td>${badge(r.status)}<div class="subtitle">${r.system_id==='sentrix'?'历史 · 只读':r.mode==='repair'?'定点补跑 · 非全量':r.mode==='composite'?'合并视图 · 保留首次性能':r.formal_protocol===false?'smoke · 非正式':r.experiment?'同期四组 · 独立冷建库':'历史协议'}</div></td><td>${num(s.completed,0)} / ${num(counts.plannedQa,0)}<div class="subtitle">有效评分 ${countRatio(counts.validQa,counts.plannedQa)} · 未评分 ${num(counts.invalidQa,0)}</div></td><td class="count">${pct(s.retrieval_recall_macro??s.image_retrieval_recall_macro)}</td><td><b>${pct(r.experiment?s.quality_correct_planned_rate:s.exact_accuracy)}</b><div class="subtitle">${d['2']??0} 正确 · ${d['1']??0} 部分 · ${d['0']??0} 错误${r.experiment?' · 分母100题':''}</div></td><td>${e2eLabel(r)}</td><td>${r.qa_concurrency??'—'} / ${r.judge_concurrency??'—'}<div class="subtitle">Agent / Judge</div></td></tr>`;}).join('');$('#run-list').innerHTML=rows?`<table><thead><tr><th></th><th>被测系统 / 运行版本</th><th>状态</th><th>进度</th><th>交付 Recall</th><th>回答完全正确</th><th>全链路时间</th><th>并发</th></tr></thead><tbody>${rows}</tbody></table>`:`<div class="empty">暂无运行结果；计划组别与未开始状态保留在上方。</div>`;document.querySelectorAll('[data-run]').forEach(e=>e.addEventListener('click',()=>openRun(e.dataset.run)));document.querySelectorAll('.compare-check').forEach(e=>e.addEventListener('change',()=>{e.checked?state.selected.add(e.dataset.id):state.selected.delete(e.dataset.id);}));$('#start').disabled=state.readOnly||state.runs.some(r=>alive(r.status));}
$('#start').addEventListener('click',async()=>{if(state.readOnly)return;if(!window.confirm('将创建新相册并运行全部355张图片和100道题，使用已有累计预算账本。开始本次独立评测？'))return;try{$('#start').disabled=true;const result=await api('/api/runs',{method:'POST',body:JSON.stringify({config:$('#config').value})});notice('已提交独立运行 '+result.run_id+'。');state.current=result.run_id;await refresh();}catch(err){notice(err.message,true);$('#start').disabled=state.readOnly;}});
function metric(title,value,note){return `<div class="metric"><label>${esc(title)}</label><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`;}
function ledgerDelta(run){const before=run.budget_before?.estimated_spent_cny,after=run.budget_after?.estimated_spent_cny;return typeof before==='number'&&typeof after==='number'?after-before:null;}
function runCost(run){const audit=run.cost_audit;return audit&&(!audit.status||audit.status==='complete')&&Array.isArray(audit.missing_call_ids)&&audit.missing_call_ids.length===0&&!audit.unidentified_call_records&&Number.isFinite(audit.own_estimated_cny)?audit.own_estimated_cny:null;}
function runCostLabel(run){if(!run.cost_audit)return '待归属审计';if(run.cost_audit.status==='no_recorded_calls')return '暂无已记录调用';if(run.cost_audit.status==='unavailable')return '费用审计不可用';const value=runCost(run);return value===null?'归属审计未完整':'¥'+num(value,4);}
function costAuditNote(run){
  const audit=run.cost_audit;
  if(!audit)return '本组费用等待按请求ID归属审计；共享账本同期增量包含其他项目调用。套餐实际账单未接入。';
  const count=Array.isArray(audit.missing_call_ids)?audit.missing_call_ids.length:null;
  const stateNote=count===null?'请求完整性尚未核实。':count||audit.unidentified_call_records?`缺失 ${num(count,0)} 个调用ID，另有 ${num(audit.unidentified_call_records??0,0)} 条调用未识别。已归属部分 ¥${num(audit.known_estimated_cny,4)} 不能作为本组完整费用。`:`按请求ID归属 ${num(audit.request_count,0)} 次调用；结果未知 ${num(audit.uncertain_requests??0,0)} 次。`;
  return `${stateNote} 金额覆盖审计时已写入轨迹的调用，尚未落盘的在途调用不包含。共享账本同期增量包含其他项目调用，单独保留作背景。套餐实际账单未接入。`;
}
function costAuditDetails(run){
  const audit=run.cost_audit;
  if(!audit)return '';
  return `<details class="panel-body"><summary>本组费用归属审计 · ${num(audit.matched_request_count??audit.request_count,0)} / ${num(audit.request_count,0)} 次已匹配</summary><pre>${esc(JSON.stringify(audit,null,2))}</pre></details>`;
}
function roleTotal(groups,names){const values=names.map(name=>groups?.[name]?.total_ms).filter(value=>typeof value==='number');return values.length?values.reduce((total,value)=>total+value,0):null;}
function groupCoverage(run){const build=run.build_summary||{};return {images:countRatio(build.image_embedding_count,build.image_count??355),captions:countRatio(build.caption_count,build.image_count??355),text:countRatio(build.text_embedding_count,build.image_count??355),events:countRatio(build.event_embedding_count,build.event_count)};}
function experimentPanel(run){
  if(!run.experiment)return '';
  const summary=run.summary||{},coverage=groupCoverage(run),diagnostic=summary.retrieval_diagnostics||{},local=summary.local_retrieval_performance||{},resource=run.resource_summary||{},model=run.experiment_config||{};
  const rows=[['图片向量',coverage.images],['图片描述',coverage.captions],['描述文本向量',coverage.text],['事件文本向量',coverage.events],['建库墙钟',duration(run.phases?.pipeline?.wall_ms)],['本组已结算等价费用',runCostLabel(run)],['本组待结算预留',Number.isFinite(run.cost_audit?.own_reserved_cny)?'¥'+num(run.cost_audit.own_reserved_cny,4):'待归属审计'],['共享账本同期增量（含其他项目）',typeof ledgerDelta(run)==='number'?'¥'+num(ledgerDelta(run),4):'待本组结束快照'],['授权GPU 1显存峰值',typeof resource.gpu_memory_mib_peak==='number'?num(resource.gpu_memory_mib_peak/1024,2)+' GiB':'未记录'],['模型进程显存峰值',num(model.embedding_backend==='qwen'?resource.qwen_process_memory_mib_peak:resource.wemm_process_memory_mib_peak,0)+' MiB'],['宿主机RAM峰值',num(resource.host_ram_used_mib_peak,0)+' MiB'],['实测资源样本',num(resource.samples,0)]];
  return `<div class="detail-grid wide"><div class="panel"><div class="section-head">独立建库覆盖与资源</div><table><tbody>${rows.map(([name,value])=>`<tr><td>${esc(name)}</td><td>${esc(value)}</td></tr>`).join('')}</tbody></table><div class="panel-body notes">${esc(model.embedding_model)} · ${esc(model.reranker_model||'无重排')} · ${esc(model.generation_model)}。${esc(costAuditNote(run))}</div>${costAuditDetails(run)}</div><div class="panel"><div class="section-head">召回与重排诊断</div><table><tbody><tr><td>召回顺序 Top-10 Recall</td><td>${pct(diagnostic.pre_top10_recall_macro)}</td></tr><tr><td>最终排序 Top-10 Recall</td><td>${pct(diagnostic.post_top10_recall_macro)}</td></tr><tr><td>完整召回候选池 Recall</td><td>${pct(diagnostic.candidate_pool_recall_macro)}</td></tr><tr><td>有效图片搜索题数 / 指标分母</td><td>${countRatio(diagnostic.questions_with_image_search,diagnostic.metric_count)}</td></tr></tbody></table><div class="panel-body notes">每道题合并各次图片分支搜索的Top-10，去重后计算Recall；无图片搜索或未执行题贡献零。此项诊断与最终交付图片指标分别统计。</div><table><thead><tr><th>本地步骤</th><th>调用 / 错误</th><th>均时</th><th>P50 / P95</th></tr></thead><tbody>${Object.entries(local).map(([name,value])=>`<tr><td>${esc({embedding:'向量编码',rerank:'重排',retrieval_setup:'分块 / 连接'}[name]||name)}</td><td>${value.calls} / ${value.errors}</td><td>${duration(value.mean_ms)}</td><td>${duration(value.p50_ms)} / ${duration(value.p95_ms)}</td></tr>`).join('')}</tbody></table><details class="panel-body"><summary>不可变模型配置与指纹</summary><pre>${esc(JSON.stringify(run.build_summary?.experiment_config||model,null,2))}</pre></details></div></div>`;
}
function questionRows(run){
  const observed=new Map((run.items||[]).map(row=>[row.qa_id,row])),statuses=new Map((run.planned_status||[]).map(row=>[row.qa_id,row]));
  const planned=(run.qa_plan||[]).map(row=>observed.get(row.qa_id)||{...row,...statuses.get(row.qa_id),execution_status:statuses.get(row.qa_id)?.execution_status||'not_started',judge_status:'not_started',_planned_only:true});
  const plannedIds=new Set(planned.map(row=>row.qa_id));
  return [...planned,...(run.items||[]).filter(row=>!plannedIds.has(row.qa_id))];
}
function openQuestion(run,qa){
  const planned=state.currentData?.run_id===run?questionRows(state.currentData).find(row=>row.qa_id===qa):null;
  if(!planned?._planned_only)return openItem(run,qa);
  $('#item-content').innerHTML=`<span class="eyebrow">${esc(qa)}</span><h2>${esc(planned.question)}</h2><div class="notice">本题已列入${state.currentData.qa_count||100}题计划，目前${esc(statusName[planned.execution_status]||planned.execution_status)}。尚无模型回答、交付图片或评分。</div><p class="notes">计划 ${num(planned.planned_turn_count,0)} 轮；结果写入后可查看完整记录。</p>`;
  if(!$('#item-dialog').open)$('#item-dialog').showModal();
}
function rankingTrace(item){
  const calls=(item.tool_trace||[]).filter(tool=>tool.tool==='search'&&tool.result?.retrieval_trace);
  if(!calls.length)return '';
  return `<details><summary>逐次召回与重排排名 · ${calls.length} 次搜索</summary>${calls.map((tool,index)=>`<h3>搜索 ${index+1}：${esc(tool.args?.query||tool.arguments?.query||'')}</h3>${Object.entries(tool.result.retrieval_trace).map(([kind,branch])=>`<p class="notes">${kind==='images'?'图片':'事件'} · 候选 ${branch.candidate_count} · 已重排 ${branch.reranked_count}</p><div class="rank-table"><table><thead><tr><th>候选ID</th><th>召回排名</th><th>重排分数</th><th>最终排名</th><th>状态</th></tr></thead><tbody>${(branch.candidates||[]).map(candidate=>`<tr><td class="qa-id">${esc(candidate.id)}</td><td>${num(candidate.recall_rank,0)}</td><td>${num(candidate.rerank_score,6)}</td><td>${num(candidate.final_rank,0)}</td><td>${candidate.reranked?'已重排':branch.reranked_count?'保留召回顺序':'未启用重排'}</td></tr>`).join('')}</tbody></table></div>`).join('')}`).join('')}</details>`;
}
async function openRun(id,scroll=true){try{const data=await api('/api/runs/'+encodeURIComponent(id));state.current=id;state.currentData=data;renderDetail(data);if(scroll)$('#detail').scrollIntoView({behavior:'smooth',block:'start'});const resources=await api('/api/runs/'+encodeURIComponent(id)+'/resources');drawResources(resources);}catch(err){notice(err.message,true);}}
function renderDetail(r){const counts=runCounts(r),s={...(r.summary||{}),judge_invalid_count:counts.invalidQa},dist=s.judge_distribution||{},valid=r.experiment?counts.plannedQa:s.judge_valid_count||0,stages=r.build_summary?.stages||r.phases?.pipeline?.stages||[],tools=s.tool_performance||{},b=r.budget_after||r.budget_before;$('#detail').hidden=false;$('#detail').innerHTML=`<div class="detail-title"><div><h2>${esc(label(r))} ${badge(r.status)}</h2><p>${esc(r.run_id)} · Judge ${esc(r.judge_model||'未记录')} · ${esc(r.code_revision?String(r.code_revision).slice(0,12):'历史版本')}</p></div><div class="actions">${!state.readOnly&&alive(r.status)&&r.system_id!=='sentrix'?'<button class="secondary" id="cancel-run">停止后续提交</button>':''}<a class="badge" href="/api/runs/${encodeURIComponent(r.run_id)}/export">导出完整 JSON ↗</a></div></div>
${r.mode==='composite'?`<div class="notice">本视图由原运行的97题和3题补跑记录合并；有效补齐 ${r.repaired_valid_count??0}/3题。性能、资源与建库时间沿用首次运行，不代表再次完成全链路。补跑墙钟 ${duration(r.repair_wall_ms)}；费用增量 ¥${num((r.repair_budget_after?.estimated_spent_cny??0)-(r.repair_budget_before?.estimated_spent_cny??0),4)}。计划111轮；含原失败尝试累计 ${r.attempted_turn_count_total??'—'} 轮。<button class="secondary" id="view-original">查看首次结果</button><button class="secondary" id="view-repair">查看补跑记录</button></div>`:''}
${r.fatal_error?`<div class="notice error">${esc(r.fatal_error.reason||r.fatal_error.type||r.fatal_error)}</div>`:''}
${experimentPanel(r)}
<div class="cards">${metric(r.experiment?'固定100题正确率':'完全正确率',pct(r.experiment?s.quality_correct_planned_rate:s.exact_accuracy),`${dist['2']||0} 正确 / ${r.experiment?r.qa_count||100:valid} ${r.experiment?'计划题':'有效评分'}；未评分 ${counts.invalidQa??'—'}`)}${metric('部分正确 / 错误',`${dist['1']||0} / ${dist['0']||0}`,'Judge 1 分 / 0 分，未评分单列')}${metric('最终交付 Recall',pct(s.retrieval_recall_macro),`P ${pct(s.retrieval_precision_macro)} · F1 ${pct(s.retrieval_f1_macro)}；${s.retrieval_metric_count??'—'} 题`)}${metric('全链路时间',e2eLabel(r),legacyE2EInvalid(r)?'旧记录估算短于流水线墙钟，原值保留在导出JSON中':'身份/输入准备 → 最后Agent完成；不含Judge')}${metric('Agent 单题均时',duration(s.agent_task_latency_mean_ms),`P50 ${duration(s.agent_task_latency_p50_ms)} · P95 ${duration(s.agent_task_latency_p95_ms)}；并发 ${r.qa_concurrency??'—'}`)}</div>
<div class="detail-grid"><div class="panel"><div class="section-head">回答与证据质量</div><div class="panel-body"><div class="scorebar">${[2,1,0].map(n=>`<i class="score-${n}" data-share="${valid?(dist[n]||0)/valid*100:0}"></i>`).join('')}</div><div class="legend"><span><b>●</b>正确 ${dist['2']||0}</span><span>部分 ${dist['1']||0}</span><span>错误 ${dist['0']||0}</span><span>无效 / 未评分 ${s.judge_invalid_count??0}</span></div><table class="wide"><tbody><tr><td>核心准确率（1或2分）</td><td>${pct(s.core_accuracy)}</td><td>平均质量</td><td>${num(s.answer_quality_mean,3)} / 2</td></tr><tr><td>任务判断</td><td>${pct(s.task_decision_accuracy)}</td><td>有效行为评分</td><td>${countRatio(counts.validTask,counts.plannedTurns)}</td></tr><tr><td>有效评分 / 计划 QA</td><td>${countRatio(counts.validQa,counts.plannedQa)}</td><td>无效 / 未评分 QA</td><td>${num(counts.invalidQa,0)}</td></tr><tr><td>已执行 / 计划轮次</td><td>${countRatio(counts.executedTurns,counts.plannedTurns)}</td><td>执行错误 QA</td><td>${executionErrorLabel(r,counts)}</td></tr><tr><td>工具候选 Recall</td><td>${pct(s.candidate_retrieval_macro?.recall)}</td><td>步数内完成</td><td>${pct(s.qa_completion_within_steps_rate)}</td></tr><tr><td>${r.system_id==='sentrix'?'JSON动作解析':'原生工具协议'}</td><td>${pct(r.system_id==='sentrix'?s.json_parse_success_rate:s.native_protocol_success_rate)}</td><td>图片证据Judge</td><td>${s.evidence_valid_count?num(s.evidence_mean,2):'未开启 / 未记录'}</td></tr></tbody></table><p class="notes">最终交付和工具候选独立计分。19道不可回答题不进入媒体召回分母，但参与回答质量与行为评分。工具成功不代表答案正确。</p></div></div><div class="panel"><div class="section-head">资源与费用</div><div class="panel-body"><canvas id="resource-chart" aria-label="GPU显存实测趋势"></canvas><p id="resource-note" class="notes">正在读取实测采样…</p><table><tbody><tr><td>主Agent输入 / 输出</td><td>${num(s.prompt_tokens_total,0)} / ${num(s.completion_tokens_total,0)} token</td></tr><tr><td>模型真实TTFT / 速度</td><td>${duration(s.llm_ttft_ms_mean)} / ${num(s.llm_tokens_per_second_mean)} token/s</td></tr><tr><td>累计预算估算 / 剩余</td><td>${b?'¥'+num(b.estimated_spent_cny,3)+' / ¥'+num(b.remaining_cny,3):'未记录'}</td></tr></tbody></table><p class="notes">本地仅采集授权GPU 1；云端Qwen显存不可观测。Doubao套餐调用按公开最高档单价估算等价费用，不是实际账单。</p></div></div></div>
<div class="detail-grid wide"><div class="panel"><div class="section-head">建库 / 流水线阶段 ${r.build_summary?.model_calls_file?`<a class="badge" href="/api/runs/${encodeURIComponent(r.run_id)}/pipeline-calls">全部模型/编码调用 JSONL ↗</a>`:""}</div>${stages.length?`<table><thead><tr><th>阶段</th><th>墙钟</th><th>成功 / 失败</th><th>缓存</th></tr></thead><tbody>${stages.map(x=>`<tr><td>${esc(x.name)}</td><td>${duration(x.wall_ms)}</td><td>${x.succeeded??'—'} / ${x.failed??'—'}</td><td>${x.cache_hits??'—'}</td></tr>`).join('')}</tbody></table>`:`<div class="panel-body">${r.system_id==='sentrix'?'历史阶段指标保留在原始运行记录中。':esc(r.live_progress?.event?.name||r.build_progress?.name||'等待阶段记录')}<details><summary>原始阶段与进度</summary><pre>${esc(JSON.stringify(r.phases||{},null,2))}</pre></details></div>`}</div><div class="panel"><div class="section-head">工具调用性能</div>${Object.keys(tools).length?`<table><thead><tr><th>工具</th><th>调用</th><th>成功率</th><th>P50 / P95</th></tr></thead><tbody>${Object.entries(tools).map(([name,x])=>`<tr><td>${esc(name)}</td><td>${x.calls}</td><td>${pct(x.ok_rate)}</td><td>${duration(x.p50_ms)} / ${duration(x.p95_ms)}</td></tr>`).join('')}</tbody></table>`:'<div class="panel-body">尚无工具记录。</div>'}</div></div>
<div class="panel"><div class="section-head">QA逐题结果 · ${r.items.length} / ${r.qa_count||100}</div><div class="filter-row"><input id="qa-search" placeholder="搜索题号或问题"><select id="qa-score" aria-label="评分筛选"><option value="all">全部评分</option><option value="2">正确 · 2分</option><option value="1">部分正确 · 1分</option><option value="0">错误 · 0分</option><option value="missing">未评分</option></select></div><div id="qa-table"></div></div><details><summary>计分版本、输入完整性与运行条件</summary><pre>${esc(JSON.stringify({input_integrity:{...(r.input_integrity||{}),files:undefined},metric_source:r.metric_source,judge_protocol:r.judge_protocol,hardware_context:r.hardware_context,original_summary:r.original_summary?{judge_distribution:r.original_summary.judge_distribution,retrieval_recall_mean:r.original_summary.retrieval_recall_mean}:undefined,timing_comparable:r.timing_comparable,resumed_at:r.resumed_at},null,2))}</pre></details>`;document.querySelectorAll('[data-share]').forEach(e=>e.style.width=e.dataset.share+'%');$('#qa-search').addEventListener('input',renderQuestions);$('#qa-score').addEventListener('change',renderQuestions);renderQuestions();if($('#view-original'))$('#view-original').onclick=()=>openRun(r.parent_run_id);if($('#view-repair'))$('#view-repair').onclick=()=>openRun(r.repair_run_id);if($('#cancel-run'))$('#cancel-run').addEventListener('click',async()=>{try{await api('/api/runs/'+encodeURIComponent(r.run_id)+'/cancel',{method:'POST',body:'{}'});notice('已请求取消后续提交。正在执行的调用会完成并保存；不会终止其他进程。');}catch(err){notice(err.message,true);}});}
function renderQuestions(){const query=($('#qa-search')?.value||'').toLowerCase(),score=$('#qa-score')?.value||'all';const rows=questionRows(state.currentData).filter(r=>(r.qa_id+' '+r.question).toLowerCase().includes(query)&&(score==='all'||(score==='missing'?r.judge?.score==null:String(r.judge?.score)===score)));$('#qa-table').innerHTML=`<table><thead><tr><th>题号 / 问题</th><th>Judge</th><th>交付 P / R / F1</th><th>状态</th></tr></thead><tbody>${rows.map(r=>`<tr><td class="question" data-qa="${esc(r.qa_id)}"><span class="qa-id">${esc(r.qa_id)}</span><br>${esc(r.question)}</td><td><span class="badge ${r.judge?.score===2?'good':r.judge?.score===0?'bad':r.judge?.score===1?'warn':''}">${r._planned_only?'未评分':judgeLabel(r.judge?.score,r.judge_status,state.currentData.status)}</span></td><td>${pct(r.retrieval_precision)} / ${pct(r.retrieval_recall)} / ${pct(r.retrieval_f1)}</td><td>${esc(statusName[r.execution_status]||r.agent_status||r.execution_status||'未执行')}</td></tr>`).join('')}</tbody></table>`;document.querySelectorAll('[data-qa]').forEach(e=>e.addEventListener('click',()=>openQuestion(state.current,e.dataset.qa)));}
function photos(rows){return `<div class="photos">${(rows||[]).map(r=>{const name=r.file_name||r.media_id;return `<a class="photo" href="/api/media/${encodeURIComponent(name)}?thumbnail=false" target="_blank" rel="noreferrer"><img loading="lazy" alt="${esc(name)}" src="/api/media/${encodeURIComponent(name)}"><span>${esc(name)}</span></a>`;}).join('')||'<span class="notes">无图片 / 未记录</span>'}</div>`;}
async function openItem(run,qa){
  try{
    const item=await api('/api/runs/'+encodeURIComponent(run)+'/items/'+encodeURIComponent(qa));
    $('#item-content').innerHTML=`<span class="eyebrow">${esc(item.qa_id)}</span><h2>${esc(item.question)}</h2>
    ${item.repair_provenance?`<p class="notice">本题为单次补跑：${esc(item.repair_provenance.repair_run_id)} · 原快照 ${esc(item.repair_provenance.snapshot_version)} <button class="secondary" id="original-item">查看原失败记录</button></p>`:''}
    <div class="legend"><span>${esc(item.task_type||'')}</span><span>${esc(item.answerability||'')}</span><span>Judge ${judgeLabel(item.judge?.score,item.judge_status,state.runs.find(row=>row.run_id===run)?.status)}</span><span>交付召回 ${pct(item.retrieval_recall)}</span></div>
    <div class="item-grid"><div><h3>模型回答</h3><div class="answer-block">${esc(item.answer||'（无回答）')}</div></div><div><h3>参考答案</h3><div class="answer-block">${esc(item.reference_answer)}</div><h3>Judge理由</h3><p class="notes">${esc(item.judge?.reason)}</p></div></div>
    <h3>模型最终交付 ${item.predicted_images?.length??0}</h3>${photos(item.predicted_images||item.predicted_media)}<h3>工具候选 ${item.retrieved_candidate_images?.length??'未记录'}</h3>${photos(item.retrieved_candidate_images)}<h3>GT目标</h3>${photos(item.gt_images||item.gt_media)}
    ${rankingTrace(item)}
    <details open><summary>多轮对话 · ${(item.runtime_turns||[]).length} 轮</summary>${(item.runtime_turns||[]).map(turn=>`<div class="panel-body"><b>用户：</b>${esc(turn.message)}<p>${esc(turn.answer)}</p><span class="notes">预期 ${esc(turn.expected_action)} · 判定 ${esc(turn.task_judge?.actual_action)} · ${esc(turn.judge?.reason)}</span></div>`).join('')}</details>
    <details><summary>时延、终态与协议统计</summary><pre>${esc(JSON.stringify({timing:item.timing_breakdown,agent_stability:item.agent_stability,termination_reason:item.termination_reason,agent_status:item.agent_status},null,2))}</pre></details>
    <details><summary>模型逐次调用 · ${(item.model_call_metrics||[]).length}</summary><pre>${esc(JSON.stringify(item.model_call_metrics||[],null,2))}</pre></details>
    <details><summary>完整工具输入 / 输出 · ${(item.tool_trace||[]).length}</summary><pre>${esc(JSON.stringify(item.tool_trace||[],null,2))}</pre></details>
    <details><summary>Judge完整输入 / 输出</summary><pre>${esc(JSON.stringify({answer:item.judge,task:item.task_judges},null,2))}</pre></details>`;
    if(item.repair_provenance&&$('#original-item'))$('#original-item').onclick=()=>openItem(item.repair_provenance.parent_run_id,item.qa_id);
    if(!$('#item-dialog').open)$('#item-dialog').showModal();
  }catch(error){notice(error.message,true);}
}
$('#compare').addEventListener('click',async()=>{
  if(state.selected.size<2){notice('请选择至少两条运行进行比较。');return;}
  try{
    const data=await api('/api/compare?ids='+encodeURIComponent([...state.selected].join(',')));
    // Comparison summaries omit phases. Read the immutable historical detail
    // to validate old E2E estimates against their recorded pipeline wall time.
    await Promise.allSettled(data.runs.filter(r=>r.system_id==='sentrix').map(async r=>{
      const detail=state.currentData?.run_id===r.run_id?state.currentData:await api('/api/runs/'+encodeURIComponent(r.run_id));
      r.phases=detail.phases;
    }));
    renderComparison(data);
  }catch(err){notice(err.message,true);}
});
function renderComparison(data){
  const metrics=[
    ['正确数 / 计划100题','quality_correct_planned_rate',pct],['完全正确率（有效评分）','exact_accuracy',pct],
    ['核心准确率（有效评分）','core_accuracy',pct],['质量均分（有效评分）','answer_quality_mean',value=>num(value,3)],
    ['最终交付 Precision','retrieval_precision_macro',pct],['最终交付 Recall','retrieval_recall_macro',pct],['最终交付 F1','retrieval_f1_macro',pct],
    ['最终交付指标分母','retrieval_metric_count',value=>num(value,0)],['任务判断','task_decision_accuracy',pct],
    ['召回顺序 Top-10 Recall','retrieval_diagnostics',value=>pct(value?.pre_top10_recall_macro)],
    ['重排后 Top-10 Recall','retrieval_diagnostics',value=>pct(value?.post_top10_recall_macro)],
    ['完整召回候选池 Recall','retrieval_diagnostics',value=>pct(value?.candidate_pool_recall_macro)],
    ['返回工具候选 Recall','candidate_retrieval_macro',value=>pct(value?.recall)],
    ['图片向量 / 相册图片','unused',(value,run)=>groupCoverage(run).images],
    ['图片描述 / 相册图片','unused',(value,run)=>groupCoverage(run).captions],
    ['描述向量 / 相册图片','unused',(value,run)=>groupCoverage(run).text],
    ['事件向量 / 事件数量','unused',(value,run)=>groupCoverage(run).events],
    ['建库墙钟','unused',(value,run)=>duration(run.phases?.pipeline?.wall_ms)],
    ['Agent 单题均时','agent_task_latency_mean_ms',duration],['Agent 单题 P50','agent_task_latency_p50_ms',duration],['Agent 单题 P95','agent_task_latency_p95_ms',duration],
    ['向量编码均时','local_retrieval_performance',value=>duration(value?.embedding?.mean_ms)],
    ['向量编码 P50 / P95','local_retrieval_performance',value=>`${duration(value?.embedding?.p50_ms)} / ${duration(value?.embedding?.p95_ms)}`],
    ['重排均时','local_retrieval_performance',value=>duration(value?.rerank?.mean_ms)],
    ['重排 P50 / P95','local_retrieval_performance',value=>`${duration(value?.rerank?.p50_ms)} / ${duration(value?.rerank?.p95_ms)}`],
    ['VLM 看图累计耗时','model_role_performance',value=>duration(roleTotal(value,['view','view_image','visual','image_review','tool_internal']))],
    ['最终复核累计耗时','model_role_performance',value=>duration(roleTotal(value,['final_review','final']))],
    ['主Agent 输入 token','prompt_tokens_total',value=>num(value,0)],['主Agent 输出 token','completion_tokens_total',value=>num(value,0)],
    ['本组已结算等价费用（请求归属）','unused',(value,run)=>runCostLabel(run)],
    ['本组待结算预留','unused',(value,run)=>Number.isFinite(run.cost_audit?.own_reserved_cny)?'¥'+num(run.cost_audit.own_reserved_cny,4):'待归属审计'],
    ['共享账本同期增量（含其他项目）','unused',(value,run)=>typeof ledgerDelta(run)==='number'?'¥'+num(ledgerDelta(run),4):'未记录'],
    ['套餐实际账单','unused',()=> '未接入'],
    ['GPU 1 显存峰值（实测）','unused',(value,run)=>num(run.resource_summary?.gpu_memory_mib_peak,0)+' MiB'],
    ['宿主机 RAM 峰值（实测）','unused',(value,run)=>num(run.resource_summary?.host_ram_used_mib_peak,0)+' MiB'],
    ['资源采样数','unused',(value,run)=>num(run.resource_summary?.samples,0)],
    ['全链路不含 Judge','benchmark_e2e_latency_excluding_judge_ms',(value,run)=>e2eLabel(run)],
    ['历史补跑墙钟','unused',(value,run)=>duration(run.repair_wall_ms)],
  ];
  const counts=data.runs.map(runCounts),runById=new Map(data.runs.map(r=>[r.run_id,r]));
  const countRows=[
    ['正确 / 部分 / 错误 / 未评分 QA',(r,c)=>{const d=r.summary?.judge_distribution||{};return `${num(d['2'],0)} / ${num(d['1'],0)} / ${num(d['0'],0)} / ${num(c.invalidQa,0)}`;}],
    ['有效评分 / 计划 QA',(r,c)=>countRatio(c.validQa,c.plannedQa)],
    ['无效 / 未评分 QA',(r,c)=>num(c.invalidQa,0)],
    ['已执行 / 计划轮次',(r,c)=>countRatio(c.executedTurns,c.plannedTurns)],
    ['有效行为评分 / 计划轮次',(r,c)=>countRatio(c.validTask,c.plannedTurns)],
    ['执行错误 QA',(r,c)=>executionErrorLabel(r,c)],
  ];
  $('#comparison').hidden=false;
  $('#comparison').innerHTML=`<div class="detail-title"><div><h2>运行对比</h2><p>QA哈希 ${data.same_qa_hash?'一致':'不同或缺失'} · 完整输入哈希 ${data.same_dataset_hash?'一致':'不同或缺失'}</p></div></div><div class="notice">${esc(data.warning)}</div><p class="notes">合并记录的性能使用首次运行，补跑单独计时计费；复用记录不含建库。本轮四组正确率使用100道计划题为分母；历史记录沿用各自协议。质量均分使用有效评分分母，无效或未评分题单列。行为评分按实际有效判定计算，已执行轮次与计划轮次同时展示。旧历史缺失执行错误字段时显示“未记录”；若旧端到端估算短于其流水线墙钟，标记为“旧估算不适用”，原始数值仍保留在导出JSON中。</p><div class="panel"><table class="comparison-table"><thead><tr><th>指标</th>${data.runs.map(r=>`<th>${esc(label(r))}<br><small>${esc(r.run_id)}</small><br>${badge(r.status)}</th>`).join('')}</tr></thead><tbody>${countRows.map(([title,format])=>`<tr><td>${title}</td>${data.runs.map((r,i)=>`<td>${format(r,counts[i])}</td>`).join('')}</tr>`).join('')}${metrics.map(([title,key,format])=>`<tr><td>${title}</td>${data.runs.map(r=>`<td>${format(r.summary?.[key],r)}</td>`).join('')}</tr>`).join('')}<tr><td>并发 Agent / Judge</td>${data.runs.map(r=>`<td>${r.qa_concurrency??'—'} / ${r.judge_concurrency??'—'}</td>`).join('')}</tr></tbody></table></div><details><summary>同题回答差异 · ${data.rows.length} 道</summary><table><thead><tr><th>题号 / 问题</th>${data.runs.map(r=>`<th>${esc(label(r))}</th>`).join('')}</tr></thead><tbody>${data.rows.map(row=>`<tr><td>${esc(row.qa_id)}<br>${esc(row.question)}</td>${row.results.map(r=>`<td><a class="comparison-item" data-run="${esc(r.run_id)}" data-item="${esc(row.qa_id)}">${judgeLabel(r.score,null,runById.get(r.run_id)?.status)} · R ${pct(r.recall)} ↗</a></td>`).join('')}</tr>`).join('')}</tbody></table></details>`;
  document.querySelectorAll('.comparison-item').forEach(e=>e.addEventListener('click',()=>openItem(e.dataset.run,e.dataset.item)));
  if(data.runs.some(run=>run.experiment))$('#comparison').insertAdjacentHTML('afterbegin','<div class="notice">本轮正确率固定以100道计划题为分母；最终交付P/R/F1与Top-10诊断固定以81题为分母。缺失题贡献零，未评分另列。Top-10诊断按各次图片搜索候选并集计算；完整系统比较不单独归因于embedding或reranker。</div>');
  $('#comparison').scrollIntoView({behavior:'smooth'});
}
function drawResources(data){const canvas=$('#resource-chart');if(!canvas)return;const samples=data.samples||[];const points=samples.map((r,i)=>{let v=r.gpu_memory_mib;if(typeof v!=='number')v=r.gpu_memory_used_mb??r.memory_used_mb;return {i,v:typeof v==='number'?v/1024:null};}).filter(r=>r.v!==null);$('#resource-note').textContent=points.length?`${data.sample_count} 次实测采样 · 授权GPU 1整卡显存 GiB；模型进程峰值另列，云端显存不可观测。`:'当前记录没有可绘制的GPU采样；保留未记录，不用零填充。';const dpr=devicePixelRatio||1,w=canvas.clientWidth||400,h=190;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);c.strokeStyle='#e5edf2';c.fillStyle='#94a3ae';c.font='10px sans-serif';for(let i=0;i<4;i++){const y=20+i*44;c.beginPath();c.moveTo(35,y);c.lineTo(w-8,y);c.stroke();}if(!points.length){c.fillText('暂无实测曲线',45,86);return;}const hi=Math.max(...points.map(p=>p.v))*1.12||1;for(let i=0;i<4;i++)c.fillText((hi*(1-i/3)).toFixed(1),2,22+i*44);c.strokeStyle='#16878b';c.lineWidth=1.8;c.beginPath();points.forEach((p,index)=>{const x=35+p.i/Math.max(1,samples.length-1)*(w-45),y=152-p.v/hi*132;index?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();c.fillText('开始',35,180);c.fillText('结束',w-34,180);}
enter().catch(err=>notice(err.message,true));
