"""Fault injection and offline recovery, ONLY this project's process and config."""
import sys,time,json,subprocess,socket,concurrent.futures
from pathlib import Path
import requests,yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));cfg=R/'config.yaml';original=cfg.read_bytes();config=yaml.safe_load(original);base=f"http://127.0.0.1:{config['port']}";checks=[]
def rec(name,passed,details=None):
 checks.append({'name':name,'passed':bool(passed),'details':details});(R/'reports/recovery_acceptance.json').write_text(json.dumps({'status':'OBSERVED','checks':checks},ensure_ascii=False,indent=2));assert passed,name
def manager(action):subprocess.run([str(R/'.venv/bin/python'),str(R/'scripts/manage.py'),action],check=True,cwd=R)
def session():
 s=requests.Session();s.trust_env=False;s.post(base+'/api/login',json={'password':(R/'run/access.key').read_text().strip()},timeout=10).raise_for_status();return s
def state(expected):
 for _ in range(180):
  try:
   s=session();j=s.get(base+'/api/status',timeout=10).json()
   if j['state'] in expected:return s,j
  except requests.RequestException:pass
  time.sleep(.5)
 raise TimeoutError('Service state '+str(expected))
s,prior=state(['ready']);query={'dataset':'demo','text':'一只猫','k':10,'kind':'image'};before=s.post(base+'/api/search',json=query).json()
with socket.socket() as sock:
 rec('download proxy tunnel closed',sock.connect_ex(('127.0.0.1',17890))!=0)
# Stop owns PID/cwd/command checks; corrupted path exists only in task config temporarily.
try:
 manager('stop');bad=dict(config);bad['model_path']='models/FAULT_INJECTION_MISSING_MODEL';cfg.write_text(yaml.safe_dump(bad,allow_unicode=True,sort_keys=False));manager('start');s,error=state(['error']);rec('model load fault exposed as error, not permanently loading',bool(error['error']),{'state':error['state'],'error':error['error']});manager('stop')
finally:
 manager('stop');cfg.write_bytes(original)
manager('restart');s,after=state(['ready']);rec('offline recovery loads exact fingerprint and persistent units',after['fingerprint']==prior['fingerprint'] and after['units']==prior['units'],{'before_units':prior['units'],'after_units':after['units'],'fingerprint':after['fingerprint'],'cold_load_s':after['cold_load_s']})
new=s.post(base+'/api/search',json=query,timeout=120).json();rec('restart preserves exact ranking and scores',[(x['id'],x['score']) for x in new['results']]==[(x['id'],x['score']) for x in before['results']])
# One request error must not poison subsequent requests.
bad=s.post(base+'/api/search',json={**query,'text':'非常长的查询文本。'*2000},timeout=120);good=s.post(base+'/api/search',json=query,timeout=120);rec('query exception is isolated and next query succeeds',bad.status_code==400 and good.status_code==200,{'rejected_http':bad.status_code,'detail':bad.json(),'next_http':good.status_code})
def parallel(i):
 client=session();r=client.post(base+'/api/search',json={**query,'text':'一辆公共汽车 '+str(i)},timeout=120);r.raise_for_status();return r.json()['timings']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:timings=list(pool.map(parallel,range(4)))
rec('four concurrent queries serialized with queue/model timings',all('queue_s' in t and t['embedding_s']>0 for t in timings),timings)
# New bytes after tunnel removal, not a duplicate or query cache.
d=s.post(base+'/api/datasets',json={'name':'关闭下载代理后离线实测'}).json()['id'];text='独立离线恢复验收：九月十一日下午在实验室检查蓝色望远镜，记录编号 WEMM-OFFLINE-20260911。';r=s.post(base+'/api/upload',params={'dataset':d,'name':'离线 新文件.txt'},data=(text+' 本次运行 '+str(time.time())).encode());r.raise_for_status();asset=r.json()['asset']
for _ in range(180):
 a=s.get(base+'/api/assets',params={'dataset':d}).json()
 if a and a[0]['status'] in ['indexed','failed']:break
 time.sleep(.5)
r=s.post(base+'/api/search',json={'dataset':d,'text':'蓝色望远镜离线恢复验收','k':5},timeout=120);r.raise_for_status();res=r.json();rec('new offline upload is encoded and retrieved',a[0]['status']=='indexed' and res['results'][0]['asset']==asset,{'asset':asset,'timings':res['timings'],'score':res['results'][0]['score']})
rec('project config restored byte-for-byte',cfg.read_bytes()==original)
print(json.dumps({'checks':len(checks),'passed':all(x['passed'] for x in checks)}))
