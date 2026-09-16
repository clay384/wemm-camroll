import os,json,time,hashlib,hmac,secrets,asyncio,shutil,uuid,subprocess,threading,fcntl,sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.common import ROOT,CONFIG,db,init_db,event,digest,relative_name

def offline_network_guard(event_name,args):
 if event_name=='socket.connect':
  address=args[1]
  if isinstance(address,tuple) and address[0] not in ['127.0.0.1','::1','localhost']:
   raise OSError('External network is disabled inside this project service')
sys.addaudithook(offline_network_guard)

SECRET_PATH=ROOT/'run/access.key'
if not SECRET_PATH.exists():
 fd=os.open(SECRET_PATH,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(24))
SECRET=SECRET_PATH.read_text().strip();worker=None

def session_valid(value):
 try:
  ts,sig=value.split('.');return 0<=time.time()-int(ts)<86400 and hmac.compare_digest(sig,hmac.new(SECRET.encode(),ts.encode(),'sha256').hexdigest())
 except Exception:return False
@asynccontextmanager
async def lifespan(app):
 global worker
 lock=open(ROOT/'run/model.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 init_db()
 from backend.worker import Worker
 worker=Worker()
 yield
 worker.stop=True;lock.close()
app=FastAPI(title='WeMM 检索实验室',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
@app.middleware('http')
async def protect(request,call_next):
 if request.method not in ['GET','HEAD']:
  origin=request.headers.get('origin')
  if origin and origin!=str(request.base_url).rstrip('/'):return JSONResponse({'detail':'跨来源请求已拒绝'},403)
 if request.url.path.startswith('/api/') and request.url.path!='/api/login':
  if not session_valid(request.cookies.get('wemm_session','')):return JSONResponse({'detail':'请先输入项目访问口令'},401)
 try:response=await call_next(request)
 except ValueError as e:return JSONResponse({'detail':str(e)},400)
 response.headers['X-Content-Type-Options']='nosniff';response.headers['X-Frame-Options']='DENY'
 response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
 return response
@app.post('/api/login')
async def login(request:Request):
 b=await request.json()
 if not hmac.compare_digest(str(b.get('password','')),SECRET):await asyncio.sleep(.4);raise HTTPException(401,'访问口令不正确')
 ts=str(int(time.time()));value=ts+'.'+hmac.new(SECRET.encode(),ts.encode(),'sha256').hexdigest()
 r=JSONResponse({'ok':True});r.set_cookie('wemm_session',value,httponly=True,samesite='strict',max_age=86400);return r
@app.get('/api/status')
def status():
 with db() as c:
  counts={x[0]:x[1] for x in c.execute('SELECT status,count(*) FROM assets GROUP BY status')}
  n=c.execute('SELECT count(*) FROM units').fetchone()[0]
 try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()
 except Exception:gpu=[]
 return {'state':worker.encoder.status if worker and worker.encoder else worker.state if worker else 'unloaded','error':worker.error if worker else None,'model':CONFIG['model_id'],'revision':CONFIG['revision'],'precision':'BF16','dimension':4096,'offload':False,'profile':CONFIG['profile'],'limits':CONFIG['profiles'][CONFIG['profile']],'native_video':True,'gpu':gpu,'counts':counts,'units':n,'queue':worker.q.qsize() if worker else 0,'busy_asset':worker.busy_asset if worker else None,'fingerprint':worker.encoder.fingerprint if worker and worker.encoder else None,'cold_load_s':worker.encoder.load_seconds if worker and worker.encoder else None,'encoding_calls':worker.encoder.calls if worker and worker.encoder else 0,'benchmark':BENCH_STATE}
@app.get('/api/datasets')
def datasets():
 with db() as c:return [dict(x) for x in c.execute('SELECT * FROM datasets')]
@app.post('/api/datasets')
async def create_dataset(request:Request):
 b=await request.json();name=str(b.get('name','')).strip()[:100]
 if not name:raise HTTPException(400,'名称不能为空')
 ident=uuid.uuid4().hex
 with db() as c:c.execute('INSERT INTO datasets VALUES(?,?,0)',(ident,name))
 return {'id':ident,'name':name}
def check_dataset(ident):
 with db() as c:
  if not c.execute('SELECT 1 FROM datasets WHERE id=?',(ident,)).fetchone():raise HTTPException(404,'数据集不存在')
def register(path,name,dataset):
 check_dataset(dataset);name=relative_name(name);size=path.stat().st_size
 if size==0:raise ValueError('文件为空')
 if size>CONFIG['upload_max_mb']*1024**2:raise ValueError('单文件超过 128 MB 上限')
 sha=digest(path);dest=ROOT/'data/originals'/sha
 with db() as c:
  old=c.execute('SELECT * FROM assets WHERE id=?',(sha,)).fetchone()
  if not old:
   shutil.copyfile(path,dest)
   c.execute('INSERT INTO assets(id,ext,bytes,status,error,created) VALUES(?,?,?,?,?,?)',(sha,Path(name).suffix.lower(),size,'pending',None,time.time()))
  c.execute('INSERT OR IGNORE INTO members VALUES(?,?,?)',(dataset,sha,name))
 return {'asset':sha,'duplicate':bool(old),'status':old['status'] if old else 'pending','bytes':size,'name':name}
@app.post('/api/upload')
async def upload(request:Request,dataset:str='personal',name:str=''):
 check_dataset(dataset);relative_name(name)
 with db() as c:
  pending=c.execute("SELECT count(*) FROM assets WHERE status IN ('pending','encoding','preprocessing')").fetchone()[0]
 if pending>=CONFIG['queue_limit']:raise HTTPException(429,'任务队列已满，请稍后继续')
 tmp=ROOT/'data/tmp'/uuid.uuid4().hex;n=0
 try:
  with tmp.open('wb') as f:
   async for chunk in request.stream():
    n+=len(chunk)
    if n>CONFIG['upload_max_mb']*1024**2:raise HTTPException(413,'单文件超过 128 MB 上限')
    f.write(chunk)
  return await asyncio.to_thread(register,tmp,name,dataset)
 finally:tmp.unlink(missing_ok=True)
@app.post('/api/import-directory')
async def import_directory(request:Request):
 b=await request.json();dataset=b.get('dataset','personal');check_dataset(dataset)
 rel=b.get('path','');base=(ROOT/CONFIG['import_root']).resolve();path=(base/relative_name(rel or '.')).resolve()
 if not path.is_relative_to(base) or not path.is_dir():raise HTTPException(400,'目录必须位于项目 import 白名单内')
 def scan():
  results=[]
  for p in sorted(path.rglob('*')):
   if p.is_symlink() or not p.resolve().is_relative_to(base):results.append({'name':str(p.relative_to(base)),'error':'拒绝符号链接'});continue
   if not p.is_file():continue
   if len(results)>=CONFIG['queue_limit']:results.append({'error':'本次目录导入达到 500 文件上限'});break
   try:results.append(register(p,str(p.relative_to(base)),dataset))
   except Exception as e:results.append({'name':str(p.relative_to(base)),'error':str(e)})
  return {'files':results}
 return await asyncio.to_thread(scan)
@app.post('/api/joint-candidate')
async def joint_candidate(request:Request):
 b=await request.json();text=str(b.get('text','')).strip()
 if not text or len(text)>2000:raise HTTPException(400,'请输入不超过 2000 字的候选文字')
 payload={'image_asset':str(b.get('image_asset','')),'text':text}
 tmp=ROOT/'data/tmp'/uuid.uuid4().hex
 try:
  tmp.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True))
  return await asyncio.to_thread(register,tmp,'图文联合候选.wemmjoint',b.get('dataset','personal'))
 finally:tmp.unlink(missing_ok=True)
@app.get('/api/assets')
def assets(dataset:str='personal'):
 with db() as c:return [dict(x) for x in c.execute('SELECT a.*,MIN(m.name) AS name,(SELECT count(*) FROM units u WHERE u.asset=a.id) AS units FROM assets a JOIN members m ON a.id=m.asset WHERE m.dataset=? GROUP BY a.id ORDER BY a.created DESC',(dataset,))]
@app.post('/api/control')
async def control(request:Request):
 b=await request.json();action=b.get('action');dataset=b.get('dataset','personal');check_dataset(dataset)
 with db() as c:
  if action in ['pause','resume']:c.execute('UPDATE datasets SET paused=? WHERE id=?',(int(action=='pause'),dataset))
  elif action=='retry':c.execute("UPDATE assets SET status='pending',error=NULL WHERE status IN ('failed','cancelled') AND id IN (SELECT asset FROM members WHERE dataset=?)",(dataset,))
  elif action=='cancel':c.execute("UPDATE assets SET status='cancelled' WHERE status IN ('pending','encoding','preprocessing') AND id IN (SELECT asset FROM members WHERE dataset=?)",(dataset,))
  else:raise HTTPException(400,'未知操作')
 return {'ok':True}
@app.get('/api/units/{uid}/{mode}')
def unit_content(uid:str,mode:str):
 with db() as c:u=c.execute('SELECT * FROM units WHERE id=?',(uid,)).fetchone()
 if not u or mode not in ['thumbnail','content']:raise HTTPException(404)
 p=u['thumb'] if mode=='thumbnail' else u['path']
 if not p:raise HTTPException(404)
 path=(ROOT/p).resolve()
 if not path.is_relative_to(ROOT/'data/derived'):raise HTTPException(403)
 return FileResponse(path,media_type='text/plain; charset=utf-8' if u['kind']=='text' else None)
@app.get('/api/assets/{asset}/original')
def original(asset:str):
 with db() as c:a=c.execute('SELECT * FROM assets WHERE id=?',(asset,)).fetchone()
 if not a:raise HTTPException(404)
 return FileResponse(ROOT/'data/originals'/asset,filename='original'+a['ext'])
@app.post('/api/search')
async def query(request:Request):
 start=time.perf_counter();b=await request.json();dataset=b.get('dataset','demo');check_dataset(dataset);kw={}
 if b.get('text'):kw['text']=str(b['text'])
 image_asset=b.get('image_asset');exclude=[]
 if image_asset:
  with db() as c:u=c.execute("SELECT * FROM units WHERE asset=? AND kind='image' LIMIT 1",(image_asset,)).fetchone()
  if not u:raise HTTPException(400,'查询图尚未处理完成')
  kw['image']=ROOT/u['path']
  if b.get('exclude_self',True):exclude.append(image_asset)
 if not kw:raise HTTPException(400,'请输入文字或选择查询图')
 try:
  v,t=await asyncio.wrap_future(worker.request(kw))
  from backend.retrieval import search
  hits,s=await asyncio.to_thread(search,v,worker.encoder.fingerprint,dataset,b.get('kind') or None,int(b.get('k',10)),int(b.get('dimension',4096)),exclude,b.get('aggregate',True))
 except Exception as e:raise HTTPException(400,str(e))
 qid=uuid.uuid4().hex;t.update(s);t['response_total_s']=time.perf_counter()-start
 event('query',{'id':qid,'fingerprint':worker.encoder.fingerprint,'timings':t})
 return {'id':qid,'results':hits,'timings':t,'notice':'相似度是排序分数，不是正确率；Top-K 不保证存在相关结果。'}
@app.post('/api/annotations')
async def annotation(request:Request):
 b=await request.json()
 if b.get('label') not in ['relevant','irrelevant','unjudged']:raise HTTPException(400)
 with db() as c:c.execute('INSERT INTO annotations VALUES(?,?,?,?)',(str(b.get('query','')),str(b.get('unit','')),b['label'],time.time()))
 return {'ok':True}
BENCH_STATE={'state':'NOT_RUN'}
@app.post('/api/benchmark')
def start_benchmark():
 if BENCH_STATE['state']=='running':raise HTTPException(409,'评测正在运行')
 if not worker or not worker.encoder:raise HTTPException(409,'模型尚未就绪')
 with db() as c:
  n=c.execute("SELECT count(*) FROM assets a JOIN members m ON m.asset=a.id WHERE m.dataset='demo' AND a.status!='indexed'").fetchone()[0]
 if n:raise HTTPException(409,'请先完成 demo 索引')
 def execute():
  BENCH_STATE.update(state='running',error=None)
  try:
   from backend.benchmark import run
   run(worker);BENCH_STATE['state']='complete'
  except Exception as e:BENCH_STATE.update(state='failed',error=str(e));event('benchmark_failed',{'error':str(e)})
 threading.Thread(target=execute,daemon=True).start();return {'started':True}
@app.post('/api/validate-alignment')
async def validate_alignment():
 import numpy as np
 samples=[('text',{'text':'一个红色圆形'}),('image',{'image':ROOT/'data/smoke/a.png'}),('video',{'video':ROOT/'data/smoke/a.mp4'}),('joint',{'image':ROOT/'data/smoke/a.png','text':'红色圆形'})]
 rows=[];vectors={}
 for name,kw in samples:
  actual,stats=await asyncio.wrap_future(worker.request(kw))
  reference,_=await asyncio.wrap_future(worker.request({**kw,'_reference':True}))
  delta=float(np.abs(actual-reference).max());rows.append({'input':name,'max_abs':delta,'cosine':float(actual@reference),'passed':delta<1e-4,'fingerprint':worker.encoder.fingerprint});vectors[name+'_service']=actual;vectors[name+'_reference']=reference
 report={'status':'OBSERVED','route':'authenticated HTTP -> single worker -> encoding/reference','rows':rows,'passed':all(x['passed'] for x in rows)}
 (ROOT/'reports/service_alignment.json').write_text(json.dumps(report,indent=2));np.savez(ROOT/'reports/service_alignment_vectors.npz',**vectors)
 return report
@app.get('/api/benchmark')
def benchmark_status():
 p=ROOT/'reports/metrics.json';return {'job':BENCH_STATE,'metrics':json.loads(p.read_text()) if p.exists() else None}
@app.get('/api/report/{name}')
def report(name:str):
 if name not in ['validation_report.html','validation_report.md','metrics.json','results.csv','per_query_results.jsonl','smoke.json']:raise HTTPException(404)
 p=ROOT/'reports'/name
 if not p.exists():raise HTTPException(404,'未运行')
 return FileResponse(p,filename=name)
from backend.camroll_internal import install as install_camroll_internal
install_camroll_internal(app, lambda: worker, '/home/realmagic/camroll-wemm/runtime', session_validator=session_valid)
app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
