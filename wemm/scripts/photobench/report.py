"""Export a portable report. No model calls and no changes to judgments."""
import collections,csv,json,shutil,sqlite3,time
from pathlib import Path
import numpy as np
from bench import ROOT,OUT,SOURCE,K,sha,dump,lines

def mean(rows):
    return {k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}

def export():
    rows=lines(OUT/'results.jsonl');qs={q['qa_id']:q for q in lines(OUT/'eligible.jsonl')}
    assert len(rows)==4992
    rewrites={r['qa_id']:r for r in lines(OUT/'rewrites.jsonl')}
    summary=[];groups=collections.defaultdict(list)
    for r in rows:
        dimensions={'overall':'all','composition':r['composition'],'task':r['task_type'],'target_count':'single' if r['target_count']==1 else 'multiple'}
        for typ,label in dimensions.items():groups[(r['track'],r['arm'],typ,label)].append(r)
        for source in r['required_evidence_sources']:groups[(r['track'],r['arm'],'evidence',source)].append(r)
    for (track,arm,typ,label),rs in groups.items():
        summary.append({'track':track,'arm':arm,'group_type':typ,'group':label,'n':len(rs),'failed':sum(r['status']!='OK' for r in rs),**mean(rs)})
    dump(OUT/'summary.json',summary)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    deltas=[]
    for track in ['images','mixed','supplement']:
        a={r['qa_id']:r for r in rows if r['track']==track and r['arm']=='A'}
        for arm in ['B','C']:
            b={r['qa_id']:r for r in rows if r['track']==track and r['arm']==arm};assert a.keys()==b.keys()
            ids=sorted(a);metrics=list(a[ids[0]]['metrics']);events=sorted({a[i]['event_id'] for i in ids})
            clusters={e:[] for e in events}
            for i in ids:clusters[a[i]['event_id']].append(i)
            sums=np.array([[sum(b[i]['metrics'][m]-a[i]['metrics'][m] for i in clusters[e]) for m in metrics] for e in events])
            sizes=np.array([len(clusters[e]) for e in events]);rng=np.random.default_rng(20260914)
            draws=rng.integers(0,len(events),size=(2000,len(events)))
            bootstrap=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
            common=[i for i in ids if a[i]['status']==b[i]['status']=='OK']
            for j,m in enumerate(metrics):
                values=[b[i]['metrics'][m]-a[i]['metrics'][m] for i in ids]
                deltas.append({'track':track,'comparison':arm+'-A','metric':m,'n':len(ids),'events':len(events),'delta':float(np.mean(values)),
                    'ci95_low':float(np.percentile(bootstrap[:,j],2.5)),'ci95_high':float(np.percentile(bootstrap[:,j],97.5)),
                    'improved':sum(v>0 for v in values),'regressed':sum(v<0 for v in values),'unchanged':sum(v==0 for v in values),
                    'common_success_n':len(common),'common_success_delta':float(np.mean([b[i]['metrics'][m]-a[i]['metrics'][m] for i in common])) if common else None})
    dump(OUT/'comparisons.json',deltas)
    with (OUT/'comparisons.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
    failures=[{'qa_id':r['qa_id'],'track':r['track'],'arm':r['arm'],'error':r.get('error'),'attempts':r['attempts']} for r in rows if r['status']!='OK']
    dump(OUT/'errors.json',{'query_errors':failures,'rewrite_errors':[r for r in rewrites.values() if r['status']!='OK']})
    timing=[]
    for arm in 'ABC':
        rs=[r for r in rows if r['arm']==arm and r['track'] in ['mixed','supplement'] and r['status']=='OK']
        rec={'arm':arm,'n':len(rs)}
        for field in ['embedding_s','retrieval_s','queue_s','response_total_s']:
            v=[r['timings'][field] for r in rs];rec[field]={'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95))}
        timing.append(rec)
    durations=[r['rewrite_seconds'] for r in rewrites.values()]
    dump(OUT/'timings.json',{'search':timing,'rewrite_pair_generation':{'n':len(durations),'sum_s':sum(durations),'p50_s':float(np.percentile(durations,50)),'p95_s':float(np.percentile(durations,95))},'note':'B/C are generated together; rewrite_seconds is shared pair latency, not independently measured per arm. Search timings include no rewriting.'})
    media=json.loads((OUT/'media.json').read_text());unique={}
    for m in media:unique.setdefault(m['sha256'],m)
    ordered=sorted(unique);numbers={asset:i for i,asset in enumerate(ordered)};assets=[]
    from PIL import Image,ImageOps
    image_dir=OUT/'media';image_dir.mkdir(exist_ok=True)
    database=sqlite3.connect('file:'+str(ROOT/'data/lab.sqlite')+'?mode=ro',uri=True);database.row_factory=sqlite3.Row
    for asset in ordered:
        m=unique[asset];thumb=image_dir/(asset+'.jpg');preview=image_dir/(asset+'-large.jpg')
        if m['kind']=='image':
            with Image.open(SOURCE/m['path']) as raw:
                im=ImageOps.exif_transpose(raw).convert('RGB');im.thumbnail((1000,1000));im.save(preview,quality=85);im.thumbnail((240,240));im.save(thumb,quality=80)
            content='media/'+preview.name
        else:
            u=database.execute('SELECT thumb FROM units WHERE asset=? AND kind=? ORDER BY id LIMIT 1',(asset,'video')).fetchone();assert u
            shutil.copyfile(ROOT/u['thumb'],thumb)
            dst=image_dir/(asset+'.mp4')
            if not dst.exists():shutil.copyfile(SOURCE/m['path'],dst)
            content='media/'+dst.name
        assets.append({'id':m['media_id'],'sha256':asset,'kind':m['kind'],'thumb':'media/'+thumb.name,'content':content})
    database.close()
    cards=[]
    byq=collections.defaultdict(list)
    for r in rows:byq[(r['track'],r['qa_id'])].append(r)
    for (track,qid),rs in byq.items():
        q=qs[qid];arms={}
        for r in rs:
            arms[r['arm']]={'query':r['query'],'status':r['status'],'error':r.get('error'),'metrics':r['metrics'],
               'hits':[[numbers[h['asset']],round(h['score'],6)] for h in r['top50']]}
        cards.append({'id':qid,'track':track,'question':q['question'],'task':q.get('task_type'),'composition':q['composition'],
            'answer':q['answer'],'evidence_sources':q.get('required_evidence_sources',[]),'human_verified':q.get('human_verified'),
            'gold':[numbers[a] for a in q['gold_assets']],'arms':arms})
    payload={'summary':summary,'comparisons':deltas,'assets':assets,'cards':cards}
    (OUT/'report-data.js').write_text('window.BENCH_DATA='+json.dumps(payload,ensure_ascii=False).replace('</','<\/')+';')
    for name in ['report.html','report.js','report.css']:
        shutil.copyfile(Path(__file__).parent/name,OUT/name)
    overall=[r for r in summary if r['group_type']=='overall']
    md=['# PhotoBench WeMM 实际召回结果','', '仅衡量已标注证据；不是完整人工 gold。固定提示词，无结果驱动调参。','', '|集合|查询|N|Hit@1|Hit@5|Hit@10|Recall@10|All@10|MRR@50|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in overall:md.append('|'+ '|'.join([r['track'],r['arm'],str(r['n'])]+[f"{r[k]:.4f}" for k in ['Hit@1','Hit@5','Hit@10','Recall@10','All@10','MRR@50']])+'|')
    md+=['','B/C 一次联合生成；改写延迟是两种输出的共同成本。缺失 expected_action 的五题仅在 supplement 中。','图片题目多数标注依赖事件描述；纯媒体输入不包含日期、GPS、身份或事件文本。未命中不直接等于画面理解错误。','视频为照片生成基准片，不代表真实摄像动作检索质量。','',f'查询失败：{len(failures)}；改写失败：{sum(r["status"]!="OK" for r in rewrites.values())}。']
    (OUT/'RESULTS.md').write_text('\n'.join(md)+'\n')
    dump(OUT/'report_export.json',{'created':time.time(),'report_script_sha256':sha(__file__),'results_sha256':sha(OUT/'results.jsonl'),'cards':len(cards),'assets':len(assets),'scope':'known annotated evidence only'})
    print('REPORT_EXPORTED',len(cards),len(assets),flush=True)

if __name__=='__main__':export()
