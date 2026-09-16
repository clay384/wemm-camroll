import concurrent.futures,queue,threading,subprocess,time,json,sys,hashlib,os,traceback,signal
from pathlib import Path
import numpy as np
from backend.common import ROOT,CONFIG,db,event

class Worker:
 def __init__(self):
  self.encoder=None;self.state='unloaded';self.error=None;self.q=queue.Queue(maxsize=32);self.stop=False;self.busy_asset=None
  self.thread=threading.Thread(target=self.loop,daemon=True,name='single-wemm-worker');self.thread.start()
 def request(self,kw):
  if self.state=='error':raise RuntimeError(self.error)
  future=concurrent.futures.Future();self.q.put_nowait((future,time.perf_counter(),kw));return future
 def dispatch(self,kw):
  kw=dict(kw)
  if kw.pop('_reference',False):return self.encoder.reference_encode(**kw)
  return self.encoder.encode(**kw)
 def loop(self):
  try:
   self.state='loading'
   from backend.encoder import Encoder
   self.encoder=Encoder();self.state='ready'
   with db() as c:
    c.execute("UPDATE assets SET status='pending',error='Recovered interrupted task' WHERE status IN ('encoding','preprocessing')")
    c.execute("UPDATE assets SET status='pending' WHERE status='indexed' AND id IN (SELECT asset FROM units WHERE fingerprint!=?)",(self.encoder.fingerprint,))
   event('model_loaded',{'load_s':self.encoder.load_seconds,'fingerprint':self.encoder.fingerprint})
  except Exception as e:self.state='error';self.error=str(e);event('model_error',{'error':str(e)});traceback.print_exc();return
  while not self.stop:
   try:
    try:f,queued,kw=self.q.get_nowait()
    except queue.Empty:f=None
    if f is not None:
     try:
      wait=time.perf_counter()-queued;v,s=self.dispatch(kw);s['queue_s']=wait;f.set_result((v,s))
     except Exception as e:f.set_exception(e);event('query_error',{'error':str(e)})
     continue
    with db() as c:
     a=c.execute("SELECT a.* FROM assets a WHERE a.status='pending' AND EXISTS (SELECT 1 FROM members m JOIN datasets d ON d.id=m.dataset WHERE m.asset=a.id AND d.paused=0) ORDER BY a.created LIMIT 1").fetchone()
    if a:self.process(dict(a))
    else:time.sleep(.15)
   except Exception as e:event('worker_error',{'error':str(e)});traceback.print_exc();time.sleep(.5)
 def process(self,a):
  asset=a['id'];self.busy_asset=asset
  try:
   with db() as c:
    changed=c.execute("UPDATE assets SET status='preprocessing',error=NULL WHERE id=? AND status='pending'",(asset,)).rowcount
   if not changed:return
   t=time.perf_counter()
   proc=subprocess.Popen([sys.executable,str(ROOT/'backend/preprocess.py'),asset,a['ext']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
   try:stdout,stderr=proc.communicate(timeout=CONFIG['preprocess_timeout'])
   except subprocess.TimeoutExpired:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.communicate();raise ValueError('预处理超过资源时间上限')
   if proc.returncode:raise ValueError(stderr[-2000:])
   units=json.loads(stdout);expanded=[]
   for u in units:
    if u['kind']=='text':
     text=(ROOT/u['path']).read_text();offset=0;part=0;buffer='';buffer_start=0
     tokenizer=self.encoder.processor.tokenizer
     def emit(value,meta):
      nonlocal part
      part+=1;p=ROOT/'data/derived'/asset/f'text-{part}.txt';p.write_text(value)
      expanded.append({**u,'path':str(p.relative_to(ROOT)),'meta':{**meta,'chunk':part}})
     for para in text.split('\n\n'):
      ids=tokenizer.encode(para,add_special_tokens=False)
      candidate=buffer+'\n\n'+para if buffer else para
      if len(tokenizer.encode(candidate,add_special_tokens=False))<=512:
       if not buffer:buffer_start=offset
       buffer=candidate
      else:
       if buffer:emit(buffer,{'char_start':buffer_start,'char_end':offset-2});buffer=''
       if len(ids)<=512:buffer=para;buffer_start=offset
       else:
        for start in range(0,len(ids),512):
         emit(tokenizer.decode(ids[start:start+512]),{'paragraph_char_start':offset,'paragraph_char_end':offset+len(para),'token_start':start,'token_end':min(start+512,len(ids))})
      offset+=len(para)+2
     if buffer:emit(buffer,{'char_start':buffer_start,'char_end':len(text)})
    else:expanded.append(u)
   with db() as c:c.execute("UPDATE assets SET status='encoding',preprocess_s=? WHERE id=? AND status!='cancelled'",(time.perf_counter()-t,asset))
   for i,u in enumerate(expanded):
    uid=hashlib.sha256(f'{asset}:{self.encoder.fingerprint}:{i}'.encode()).hexdigest()
    with db() as c:
     current=c.execute('SELECT status FROM assets WHERE id=?',(asset,)).fetchone()[0]
     active=c.execute('SELECT count(*) FROM members m JOIN datasets d ON d.id=m.dataset WHERE m.asset=? AND d.paused=0',(asset,)).fetchone()[0]
     old=c.execute('SELECT vector FROM units WHERE id=?',(uid,)).fetchone()
    if current=='cancelled':return
    if not active:
     with db() as c:c.execute("UPDATE assets SET status='pending' WHERE id=?",(asset,))
     return
    if old and (ROOT/old[0]).is_file():continue
    kw={'text':(ROOT/u['path']).read_text()} if u['kind']=='text' else {'video' if u['kind']=='video' else 'image':ROOT/u['path']}
    if u['meta'].get('joint_text'):kw['text']=u['meta']['joint_text']
    v,s=self.dispatch(kw)
    vector=Path('data/indexes')/(uid+'.npy');tmp=ROOT/(str(vector)+'.tmp')
    with tmp.open('wb') as f:np.save(f,v)
    os.replace(tmp,ROOT/vector)
    with db() as c:c.execute('INSERT OR REPLACE INTO units VALUES(?,?,?,?,?,?,?,?,?)',(uid,asset,u['kind'],u['path'],u['thumb'],json.dumps(u['meta'],ensure_ascii=False),str(vector),self.encoder.fingerprint,json.dumps(s)))
    # Service pending searches between safe import units.
    while not self.q.empty():
     f,queued,kw=self.q.get_nowait()
     try:
      wait=time.perf_counter()-queued;v,s=self.dispatch(kw);s['queue_s']=wait;f.set_result((v,s))
     except Exception as e:f.set_exception(e)
   with db() as c:
    c.execute("UPDATE assets SET status='indexed' WHERE id=? AND status!='cancelled'",(asset,))
    c.execute('DELETE FROM units WHERE asset=? AND fingerprint!=?',(asset,self.encoder.fingerprint))
   event('indexed',{'asset':asset,'units':len(expanded)})
  except Exception as e:
   with db() as c:c.execute("UPDATE assets SET status='failed',error=? WHERE id=? AND status!='cancelled'",(str(e)[:2000],asset))
   event('import_failed',{'asset':asset,'error':str(e)})
   if 'out of memory' in str(e).lower():
    import torch;torch.cuda.empty_cache()
  finally:self.busy_asset=None
