"""Live verification; only restarts the task-owned WeMM process when requested."""
import argparse,json,subprocess,time,sqlite3
import numpy as np
from bench import ROOT,OUT,sha,dump,lines,client,api

def snapshot():
    s,base=client();status=api(s,base,'/api/status');assert status['state']=='ready'
    idx=json.loads((OUT/'index_verified.json').read_text());assert status['fingerprint']==idx['fingerprint']
    dataset=json.loads((OUT/'dataset.json').read_text())['id'];results=lines(OUT/'results.jsonl')
    # Deterministic selection from each media composition and each arm, no score-based selection.
    selected=[]
    for comp in ['image','video','image+video']:
        for arm in 'ABC':
            eligible=sorted([r for r in results if r['track']=='mixed' and r['composition']==comp and r['arm']==arm and r['status']=='OK'],key=lambda r:r['qa_id'])
            selected.append(eligible[0])
    ranks=[]
    for row in selected:
        response=api(s,base,'/api/search',{'dataset':dataset,'text':row['query'],'k':50,'dimension':4096,'aggregate':True})
        hits=response['results'];assert [h['asset'] for h in hits]==[h['asset'] for h in row['top50']]
        diff=max(abs(h['score']-old['score']) for h,old in zip(hits,row['top50']))
        assert diff<1e-6
        ranks.append({'qa_id':row['qa_id'],'arm':row['arm'],'assets':[h['asset'] for h in hits],'scores':[h['score'] for h in hits],'max_difference_from_evaluation':diff})
    c=sqlite3.connect('file:'+str(ROOT/'data/lab.sqlite')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    unitrows=[dict(r) for r in c.execute('SELECT * FROM units WHERE fingerprint=? AND EXISTS(SELECT 1 FROM members m WHERE m.asset=units.asset AND m.dataset=?)',(idx['fingerprint'],dataset))]
    vectorhash={u['id']:sha(ROOT/u['vector']) for u in unitrows}
    assert vectorhash=={u['id']:u['vector_sha256'] for u in idx['unit_vectors']}
    for u in unitrows:
        v=np.load(ROOT/u['vector'],allow_pickle=False);assert v.shape==(4096,) and np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-5
    contents=[]
    for kind in ['image','video']:
        u=next(u for u in unitrows if u['kind']==kind)
        for mode in ['thumbnail','content']:
            r=s.get(base+'/api/units/'+u['id']+'/'+mode,timeout=30);r.raise_for_status();assert r.content
            contents.append({'kind':kind,'mode':mode,'bytes':len(r.content),'type':r.headers.get('content-type')})
    pid=json.loads((ROOT/'run/service.json').read_text())['pid']
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    owned=[l for l in output.splitlines() if l.split(',')[0].strip()==str(pid)]
    assert len(owned)==1 and 'GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0' in owned[0]
    old=json.loads((OUT/'historical_reports_before.json').read_text())
    assert all(sha(ROOT/name)==h for name,h in old.items()),'Historical report changed'
    return {'time':time.time(),'pid':pid,'gpu_process':owned[0],'status':status,'ranks':ranks,'vector_hashes_unchanged':True,'historical_reports_unchanged':True,'content_checks':contents,'validation_script_sha256':sha(__file__)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--restart',action='store_true');args=p.parse_args()
    before=snapshot();dump(OUT/'live_before_restart.json',before);print('LIVE_PRE_RESTART_PASS',flush=True)
    if args.restart:
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/manage.py'),'restart'],cwd=ROOT,check=True)
        ready=False
        for _ in range(120):
            try:
                s,b=client();st=api(s,b,'/api/status')
                if st['state']=='error':raise RuntimeError(st['error'])
                if st['state']=='ready':ready=True;break
            except Exception:pass
            time.sleep(1)
        assert ready,'Restart did not become ready'
        after=snapshot();assert before['pid']!=after['pid'];assert [r['assets'] for r in before['ranks']]==[r['assets'] for r in after['ranks']]
        dump(OUT/'live_after_restart.json',after);print('LIVE_POST_RESTART_PASS',flush=True)

if __name__=='__main__':main()
