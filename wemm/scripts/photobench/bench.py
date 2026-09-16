"""Independent PhotoBench evaluation through the existing WeMM HTTP API."""
import argparse, collections, hashlib, json, os, shutil, sqlite3, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN_ID='20260914-115425'
OUT=ROOT/'reports/photobench'/RUN_ID
SOURCE=ROOT/'import/photobench-20260914/album3'
K=(1,5,10,20)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(tmp,p)

def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write_lines(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def metric(ranking, relevant):
    rel=set(relevant);assert rel
    seen=set();ranking=[x for x in ranking if not(x in seen or seen.add(x))]
    result={}
    for k in K:
        n=len(set(ranking[:k])&rel)
        result.update({f'Hit@{k}':float(n>0),f'Recall@{k}':n/len(rel),f'All@{k}':float(n==len(rel))})
    ranks=[i+1 for i,x in enumerate(ranking[:50]) if x in rel]
    result['MRR@50']=1/min(ranks) if ranks else 0.
    return result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Already prepared; verify/resume instead'
    media=[]
    for folder,kind in [('photos','image'),('videos','video')]:
        for p in sorted((SOURCE/folder).iterdir()):
            assert p.is_file() and not p.is_symlink()
            media.append({'media_id':'album3/'+p.name if kind=='image' else p.stem,'kind':kind,'path':str(p.relative_to(SOURCE)),'sha256':sha(p),'bytes':p.stat().st_size})
    assert collections.Counter(x['kind'] for x in media)=={'image':334,'video':10}
    byid={x['media_id']:x for x in media};raw=lines(SOURCE/'qa/mixed-image-video-1167q.jsonl')
    summary=json.loads((SOURCE/'qa/summary.json').read_text())
    for name,v in summary['collections'].items():assert sha(SOURCE/'qa'/name)==v['sha256']
    imageqa=lines(SOURCE/'qa/image-related-1115q.jsonl');videoqa=lines(SOURCE/'qa/single-video-52q.jsonl')
    assert {r['qa_id']:r for r in imageqa+videoqa}=={r['qa_id']:r for r in raw}
    eligible=[];excluded=[];questions=[]
    for q in raw:
        if q.get('interaction_type')!='direct_question' or q.get('answerability')!='answerable':
            excluded.append({'qa_id':q['qa_id'],'reason':'non-positive interaction/answerability','interaction_type':q.get('interaction_type'),'answerability':q.get('answerability')});continue
        assert q.get('expected_action') in ('answer',None)
        refs=q['retrieval_media_refs'];assert refs==q['answer_evidence_media_refs'] and refs
        for ref in refs:assert ref['media_id'] in byid and byid[ref['media_id']]['kind']==ref['media_type']
        gold=sorted({byid[r['media_id']]['sha256'] for r in refs})
        comp='+'.join(sorted({r['media_type'] for r in refs}))
        eligible.append({**q,'gold_assets':gold,'composition':comp,'cohort':'main' if q.get('expected_action')=='answer' else 'missing_action'})
        questions.append({'qa_id':q['qa_id'],'question':q['question']})
    assert len(eligible)==864 and len(excluded)==303
    assert sum(q['cohort']=='main' and q['composition']=='image' for q in eligible)==800
    dump(OUT/'media.json',media);write_lines(OUT/'eligible.jsonl',eligible);write_lines(OUT/'excluded.jsonl',excluded);write_lines(OUT/'questions_only.jsonl',questions)
    from rewrite import PROMPT
    protocol={'arms':{'A':'question','B':'rewrite','C':'keywords'},'dimension':4096,'k':list(K),'requested_k':50,
       'scoring':'known evidence; canonical SHA256; fixed denominator failures zero; MRR truncated at 50',
       'primary':'800 image-only positive questions against 334 images','mixed':'859 positives against 344 media','supplement':'5 missing expected_action, separate',
       'rewrite_prompt':PROMPT,'generation':{'do_sample':False,'max_new_tokens':512,'batch_size':1,'max_attempts':2},
       'bootstrap':{'seed':20260914,'samples':2000,'cluster':'event_id, fallback qa_id','interval':'percentile 95%'},
       'no_tuning_on_results':True,'no_router':True,'metadata_indexed':False,'source_review_not_human_gold':True}
    dump(OUT/'protocol.json',protocol)
    files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    freeze={'run_id':RUN_ID,'created':time.time(),'source_files':files,'generated_files':{n:sha(OUT/n) for n in ['media.json','eligible.jsonl','excluded.jsonl','questions_only.jsonl','protocol.json']},
            'scripts':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'source_lock_sha256':sha(ROOT/'SOURCE_LOCK.json')}
    dump(OUT/'freeze.json',freeze);print('PREPARED',len(media),len(eligible),len(excluded),flush=True)

def verify():
    f=json.loads((OUT/'freeze.json').read_text())
    for n,h in f['source_files'].items():assert sha(SOURCE/n)==h,n
    for n,h in f['generated_files'].items():assert sha(OUT/n)==h,n
    for n,h in f['scripts'].items():assert sha(Path(__file__).parent/n)==h,n
    assert sha(ROOT/'SOURCE_LOCK.json')==f['source_lock_sha256']
    return f

def client():
    import requests,yaml
    conf=yaml.safe_load((ROOT/'config.yaml').read_text());base='http://127.0.0.1:'+str(conf['port'])
    s=requests.Session();s.trust_env=False
    r=s.post(base+'/api/login',json={'password':(ROOT/'run/access.key').read_text().strip()},timeout=30);r.raise_for_status()
    return s,base

def api(s,base,path,body=None):
    r=s.get(base+path,timeout=600) if body is None else s.post(base+path,json=body,timeout=600)
    r.raise_for_status();return r.json()

def ingest():
    verify();s,base=client();st=api(s,base,'/api/status');assert st['state']=='ready',st
    p=OUT/'dataset.json'
    if p.exists():d=json.loads(p.read_text())
    else:
        d=api(s,base,'/api/datasets',{'name':'PhotoBench album3 · '+RUN_ID});dump(p,d)
    results=[];started=time.time()
    for folder in ['photos','videos']:
        results+=api(s,base,'/api/import-directory',{'dataset':d['id'],'path':str((SOURCE/folder).relative_to(ROOT/'import'))})['files']
    dump(OUT/'import_registration.json',results);assert len(results)==344 and not any(x.get('error') for x in results)
    while True:
        assets=api(s,base,'/api/assets?dataset='+d['id']);counts=collections.Counter(a['status'] for a in assets)
        print('INGEST',dict(counts),flush=True)
        if counts.get('failed') or counts.get('cancelled'):dump(OUT/'import_failures.json',assets);raise RuntimeError('Import failure; do not omit')
        if counts.get('indexed')==len(assets):break
        time.sleep(5)
    expected={m['sha256'] for m in json.loads((OUT/'media.json').read_text())};assert {a['id'] for a in assets}==expected
    dump(OUT/'import_complete.json',{'assets':assets,'wall_s':time.time()-started,'status':api(s,base,'/api/status')})
    validate_index()

def validate_index():
    import numpy as np
    d=json.loads((OUT/'dataset.json').read_text());s,base=client();st=api(s,base,'/api/status');fp=st['fingerprint']
    c=sqlite3.connect('file:'+str(ROOT/'data/lab.sqlite')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute('SELECT u.* FROM units u WHERE fingerprint=? AND EXISTS(SELECT 1 FROM members m WHERE m.asset=u.asset AND m.dataset=?)',(fp,d['id']))];c.close()
    media=json.loads((OUT/'media.json').read_text());assert {r['asset'] for r in rows}=={m['sha256'] for m in media}
    units=[]
    for r in rows:
        v=np.load(ROOT/r['vector'],allow_pickle=False);assert v.shape==(4096,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-5
        assert not json.loads(r['meta']).get('joint_text');assert r['kind'] in ('image','video')
        units.append({'id':r['id'],'asset':r['asset'],'kind':r['kind'],'vector_sha256':sha(ROOT/r['vector'])})
    dump(OUT/'index_verified.json',{'fingerprint':fp,'assets':len(set(r['asset'] for r in rows)),'units':len(rows),'kind_counts':dict(collections.Counter(r['kind'] for r in rows)),'unit_vectors':units,'verified':time.time()})
    print('INDEX_VALID',len(rows),flush=True)

def evaluate():
    verify();rewrites=lines(OUT/'rewrites.jsonl');rc=json.loads((OUT/'rewrite_complete.json').read_text())
    assert sha(OUT/'rewrites.jsonl')==rc['output_sha256'];assert sha(OUT/'rewrite_lock.json')==rc['lock_sha256']
    qs=lines(OUT/'eligible.jsonl');rw={r['qa_id']:r for r in rewrites};assert set(rw)=={q['qa_id'] for q in qs}
    idx=json.loads((OUT/'index_verified.json').read_text());dataset=json.loads((OUT/'dataset.json').read_text())['id']
    s,base=client();st=api(s,base,'/api/status');assert st['fingerprint']==idx['fingerprint'] and st['state']=='ready'
    lock={'fingerprint':idx['fingerprint'],'index_sha256':sha(OUT/'index_verified.json'),'rewrites_sha256':sha(OUT/'rewrites.jsonl'),'scorer_sha256':sha(__file__),'config_sha256':sha(ROOT/'config.yaml'),'retrieval_sha256':sha(ROOT/'backend/retrieval.py'),'dataset':dataset}
    lp=OUT/'evaluation_lock.json'
    if lp.exists():assert json.loads(lp.read_text())==lock
    else:dump(lp,lock)
    dest=OUT/'results.jsonl';done={}
    if dest.exists():
        for r in lines(dest):done[(r['qa_id'],r['track'],r['arm'])]=r
    count=len(done)
    with dest.open('a',buffering=1) as f:
        for q in qs:
            tracks=['mixed' if q['cohort']=='main' else 'supplement']
            if q['cohort']=='main' and q['composition']=='image':tracks.insert(0,'images')
            for arm,field in [('A','question'),('B','rewrite'),('C','keywords')]:
                text=q['question'] if arm=='A' else (rw[q['qa_id']].get('outputs') or {}).get(field)
                for track in tracks:
                    key=(q['qa_id'],track,arm)
                    if key in done:continue
                    row={'qa_id':q['qa_id'],'track':track,'arm':arm,'question':q['question'],'query':text,'event_id':q.get('event_id') or q['qa_id'],
                       'task_type':q.get('task_type'),'composition':q['composition'],'gold_assets':q['gold_assets'],'target_count':len(q['gold_assets']),
                       'required_evidence_sources':q.get('required_evidence_sources',[]),'human_verified':q.get('human_verified'),
                       'rewrite_seconds':0 if arm=='A' else rw[q['qa_id']]['rewrite_seconds'],'status':'OK','fingerprint':idx['fingerprint'],'attempts':[]}
                    hits=[]
                    if not text:row.update(status='FAILED',error='Rewrite generation failed')
                    else:
                        for attempt in range(2):
                            try:
                                response=api(s,base,'/api/search',{'dataset':dataset,'text':text,'kind':'image' if track=='images' else None,'k':50,'dimension':4096,'aggregate':True})
                                assert response['timings']['fingerprint']==idx['fingerprint']
                                hits=response['results'];row['timings']=response['timings'];break
                            except Exception as e:
                                row['attempts'].append(str(e));
                                if attempt==1:row.update(status='FAILED',error=str(e))
                    row['top50']=[{k:h[k] for k in ['asset','id','kind','name','score','meta','rank']} for h in hits]
                    row['metrics']=metric([h['asset'] for h in hits],q['gold_assets'])
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());count+=1
                    if count%50==0:print('EVAL',count,'/4992',flush=True)
    allrows=lines(dest);assert len(allrows)==4992 and len({(r['qa_id'],r['track'],r['arm']) for r in allrows})==4992
    dump(OUT/'evaluation_complete.json',{'rows':len(allrows),'sha256':sha(dest),'failed':sum(r['status']!='OK' for r in allrows),'status':api(s,base,'/api/status')})
    print('EVALUATION_COMPLETE',len(allrows),flush=True)

def selftest():
    assert metric(['x','a'],['a'])['MRR@50']==.5
    m=metric(['a','a','b'],['a','b']);assert m['Recall@1']==.5 and m['All@5']==1
    assert metric(['x'],['a'])['Hit@20']==0
    assert metric(['x']*50+['a'],['a'])['MRR@50']==.5 # duplicate assets do not consume ranks
    assert metric([str(i) for i in range(50)]+['a'],['a'])['MRR@50']==0
    # Exercise actual production aggregation/tie ordering without touching its database.
    import tempfile,types,importlib.util
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);database=root/'test.sqlite';c=sqlite3.connect(database)
        c.executescript('CREATE TABLE units(id TEXT,asset TEXT,kind TEXT,fingerprint TEXT,vector TEXT,meta TEXT,thumb TEXT); CREATE TABLE members(dataset TEXT,asset TEXT,name TEXT);')
        for uid,asset,v,kind in [('b1','b',[1.,0.],'video'),('b2','b',[.8,.6],'video'),('a1','a',[1.,0.],'image')]:
            np.save(root/(uid+'.npy'),np.array(v));c.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?)',(uid,asset,kind,'fp',uid+'.npy','{}',None))
        c.executemany('INSERT INTO members VALUES(?,?,?)',[('d','a','alias1'),('d','a','alias2'),('d','b','video')]);c.commit();c.close()
        def db():
            x=sqlite3.connect(database);x.row_factory=sqlite3.Row;return x
        saved=sys.modules.get('backend.common');sys.modules['backend.common']=types.SimpleNamespace(ROOT=root,db=db)
        try:
            spec=importlib.util.spec_from_file_location('test_retrieval',ROOT/'backend/retrieval.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            hits,_=mod.search(np.array([1.,0.]),'fp','d',k=5)
            assert [h['asset'] for h in hits]==['a','b'] and len(hits[1]['children'])==1
            assert len(mod.search(np.array([1.,0.]),'fp','d',kind='image',k=5)[0])==1
        finally:
            if saved is None:sys.modules.pop('backend.common',None)
            else:sys.modules['backend.common']=saved
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'scorer_tests.json',{'passed':True,'cases':['single target','multiple targets','miss','duplicate assets','MRR cutoff','production video max aggregation','production SHA tie ordering','image filter'],'scorer_sha256':sha(__file__)})
    print('SCORER_TESTS_PASS')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','verify','ingest','evaluate','selftest','validate-index']);args=a.parse_args()
    {'prepare':prepare,'verify':verify,'ingest':ingest,'evaluate':evaluate,'selftest':selftest,'validate-index':validate_index}[args.stage]()
