"""Frozen evaluation; only invokes existing single GPU worker, no model load."""
import json,time,hashlib,math,csv,html,collections,datetime,uuid
from pathlib import Path
import numpy as np
from backend.common import ROOT,db
from backend.retrieval import search

def metric(ranking,relevant):
 ranks=[i+1 for i,x in enumerate(ranking) if x in relevant]
 return {'Hit@1':float(any(x<=1 for x in ranks)),'Hit@5':float(any(x<=5 for x in ranks)),'Hit@10':float(any(x<=10 for x in ranks)),
 'Recall@5':sum(x<=5 for x in ranks)/len(relevant) if relevant else None,'Recall@10':sum(x<=10 for x in ranks)/len(relevant) if relevant else None,'MRR':1/min(ranks) if ranks else 0.0}

def run(worker):
 b=ROOT/'benchmark';freeze=json.loads((b/'FREEZE.json').read_text())
 for name,h in freeze['hashes'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen input hash mismatch: '+name)
 queries=[json.loads(s) for s in (b/'queries.jsonl').read_text().splitlines()];qrels=[json.loads(s) for s in (b/'qrels.jsonl').read_text().splitlines()]
 fingerprint=worker.encoder.fingerprint;rid=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');out=ROOT/'reports/runs'/rid;out.mkdir(parents=True)
 with db() as c:
  units=[dict(x) for x in c.execute("SELECT u.* FROM units u WHERE fingerprint=? AND EXISTS (SELECT 1 FROM members m WHERE m.asset=u.asset AND m.dataset='demo')",(fingerprint,))]
  assets=[dict(x) for x in c.execute("SELECT a.* FROM assets a WHERE EXISTS (SELECT 1 FROM members m WHERE m.asset=a.id AND m.dataset='demo')")]
 if not units:raise ValueError('No indexed units')
 expected={json.loads(line)['sha256'] for line in (ROOT/'data_manifest.jsonl').read_text().splitlines()}
 if {a['id'] for a in assets}!=expected:raise ValueError('Demo asset set differs from frozen manifest; restore the independent demo dataset')
 rows=[];latency=[]
 # Explicit warmup, no query embedding cache.
 for _ in range(2):worker.request({'text':'温暖的阳光照在草地上'}).result(timeout=600)
 for q in queries:
  try:
   kw={}
   if q['text']:kw['text']=q['text']
   if q['image_asset']:
    u=next(x for x in units if x['asset']==q['image_asset'] and x['kind']=='image');kw['image']=ROOT/u['path']
   vec,t=worker.request(kw).result(timeout=600);latency.append(t)
   kind={'text_image':'image','image_image':'image','joint_image':'image','text_pdf':'pdf','text_video':'video','text_text':'text'}[q['type']]
   rels=[x for x in qrels if x['query']==q['id']]
   rel_assets={x['asset'] for x in rels};rel_units={u['id'] for u in units for r in rels if u['asset']==r['asset'] and (r.get('page') is None or json.loads(u['meta']).get('page')==r['page'])}
   for dim in [4096,1024,256]:
    ranked,s=search(vec,fingerprint,'demo',kind,None,dim,q['exclude_assets'],False)
    source=[];seen=set()
    for x in ranked:
     if x['asset'] not in seen:source.append(x);seen.add(x['asset'])
    for level,hits,rel,key in [('unit',ranked,rel_units,'id'),('file',source,rel_assets,'asset')]:
     row={'query':q['id'],'text':q['text'],'query_image':q['image_asset'],'type':q['type'],'split':q['split'],'dimension':dim,'level':level,'no_match':q['no_match'],'status':'OBSERVED','fingerprint':fingerprint,'timings':{**t,**s},'top_k':[{**h,'judgment':'relevant' if h[key] in rel else 'unjudged'} for h in hits[:10]],'unjudged_top10':sum(h[key] not in rel for h in hits[:10]),'judged_relevant_count':len(rel),'metrics':metric([x[key] for x in hits],rel) if rel else None,'top_score':hits[0]['score'] if hits else None}
     rows.append(row)
   np.save(out/(q['id']+'.npy'),vec)
  except Exception as e:rows.append({'query':q['id'],'type':q['type'],'split':q['split'],'status':'FAILED','error':str(e)})
  (out/'per_query_results.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 # Fixed 20 repeated hot encodes per modality; repeats count reported separately.
 hot={}
 samples={'text':{'text':'一只猫躺在沙发上'},'image':{'image':ROOT/next(u['path'] for u in units if u['kind']=='image')},'video':{'video':ROOT/next(u['path'] for u in units if u['kind']=='video')}}
 for kind,kw in samples.items():
  worker.request(kw).result(timeout=600);measure=[]
  for _ in range(20):
   _,stats=worker.request(kw).result(timeout=600);measure.append(stats)
  hot[kind]={'n':20,'cache':False,'input_reuse':True,'warmup':1,'p50_embedding_s':float(np.percentile([x['embedding_s'] for x in measure],50)),'p95_embedding_s':float(np.percentile([x['embedding_s'] for x in measure],95)),'p50_total_s':float(np.percentile([x['total_s'] for x in measure],50)),'p95_total_s':float(np.percentile([x['total_s'] for x in measure],95)),'samples':measure}
 groups=collections.defaultdict(list)
 for x in rows:
  if x.get('metrics'):
   groups[(x['split'],x['type'],x['dimension'],x['level'])].append(x['metrics'])
 summary=[]
 for (split,kind,dim,level),vals in groups.items():summary.append({'split':split,'type':kind,'dimension':dim,'level':level,'n':len(vals),**{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}})
 metrics={'status':'OBSERVED','run_id':rid,'fingerprint':fingerprint,'freeze':freeze,'queries':len(queries),'source_files':len({u['asset'] for u in units}),'units':len(units),'cold_load_s':worker.encoder.load_seconds,'summary':summary,'hot':hot,'query_timings':latency,'failures':[x for x in rows if x['status']=='FAILED'],'NDCG@10':'NOT_COMPUTED: binary/incomplete judgments, no graded relevance','recall_scope':'against annotated evidence only; unjudged items not labeled irrelevant; Hit/MRR likewise measure known evidence at original ranks, not complete human judgments','no_match':'Top-K distribution only; rejection threshold NOT_ENABLED','mrl':'same full query/candidate vectors truncated then re-normalized, no re-encoding and no model memory savings','index_bytes':sum((ROOT/u['vector']).stat().st_size for u in units),'resident_rss_bytes':__import__('psutil').Process().memory_info().rss,'unit_encode_s':sum(json.loads(u['stats'])['embedding_s'] for u in units),'asset_preprocess_s':sum(a['preprocess_s'] for a in assets),'peak_allocated_mib':max(json.loads(u['stats'])['peak_allocated_mib'] for u in units),'peak_reserved_mib':max(json.loads(u['stats'])['peak_reserved_mib'] for u in units)}
 negatives=[x['top_score'] for x in rows if x.get('no_match') and x.get('dimension')==4096 and x.get('level')=='file' and x.get('top_score') is not None]
 metrics['no_match_distribution']={'n':len(negatives),'scores':negatives,'min':min(negatives) if negatives else None,'max':max(negatives) if negatives else None,'p50':float(np.median(negatives)) if negatives else None,'false_accept_rate':'NOT_COMPUTED: no acceptance threshold enabled'}
 metrics['mrl_raw_vector_bytes']={str(d):len(units)*d*4 for d in [4096,1024,256]}
 with db() as c:
  times=[x[0] for x in c.execute("SELECT time FROM events WHERE kind='indexed'")]
 metrics['import_wall_s']=max(times)-min(a['created'] for a in assets) if times else None
 metrics['throughput_units_per_s']=len(units)/metrics['import_wall_s'] if metrics['import_wall_s'] else None
 metrics['scorer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 metrics['retrieval_sha256']=hashlib.sha256((ROOT/'backend/retrieval.py').read_bytes()).hexdigest()
 for name in ['config.yaml','SOURCE_LOCK.json']:(out/name).write_bytes((ROOT/name).read_bytes())
 (out/'FREEZE.json').write_bytes((b/'FREEZE.json').read_bytes())
 (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
 for name in ['metrics.json','per_query_results.jsonl']:(ROOT/'reports'/name).write_bytes((out/name).read_bytes())
 export(metrics,rows)

def export(m,rows):
 fields=['query','type','split','dimension','level','status','top_score','Hit@1','Hit@5','Hit@10','Recall@5','Recall@10','MRR']
 with (ROOT/'reports/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for x in rows:w.writerow({**x,**(x.get('metrics') or {})})
 table='<table><tr><th>集合</th><th>任务</th><th>维度</th><th>级别</th><th>N</th><th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th></tr>'
 for x in m['summary']:table+='<tr>'+''.join('<td>'+html.escape(f'{x[k]:.3f}' if isinstance(x[k],float) else str(x[k]))+'</td>' for k in ['split','type','dimension','level','n','Hit@1','Hit@5','Hit@10','MRR'])+'</tr>'
 table+='</table>'
 details=''
 for x in rows:
  if x.get('dimension')!=4096 or x.get('level')!='file':continue
  details+='<details><summary>'+html.escape(x['query']+' '+x['text'])+'</summary><pre>'+html.escape(json.dumps({'metrics':x['metrics'],'top_k':x['top_k'],'timings':x['timings']},ensure_ascii=False,indent=2))+'</pre></details>'
 content='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>WeMM 独立验证报告</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1100px;color:#1e293b;background:#fff}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}h1{font-size:26px}</style><h1>WeMM-Embedding-9B 独立小规模验证</h1><p>本页为已运行结果的静态报告；新检索需要服务器。来源类别注释和程序标签，不是本轮人工 gold。Recall 仅针对已标注证据。未判定项不等于不相关。合成视频成绩不代表真实动作理解。</p><p>'+html.escape(f"运行 {m['run_id']}；{m['source_files']} 源文件 / {m['units']} 单元 / {m['queries']} 查询；BF16 / 4096 维。")+'</p>'+table+details+'</html>'
 (ROOT/'reports/validation_report.html').write_text(content)
 (ROOT/'reports/validation_report.md').write_text('# WeMM 独立验证\n\nOBSERVED。详见 metrics.json、results.csv 与静态 HTML。\n\n'+json.dumps({k:m[k] for k in ['run_id','source_files','units','queries','cold_load_s','index_bytes','peak_allocated_mib','peak_reserved_mib','failures']},ensure_ascii=False,indent=2)+'\n\n不声称复现官方 MMEB；相关性不完整；无匹配阈值未启用。\n')
