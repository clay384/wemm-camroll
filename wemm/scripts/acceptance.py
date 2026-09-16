"""Actual running-service API checks, never substitutes for model inference."""
from pathlib import Path
import sys,json,time,subprocess,requests,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from backend.common import CONFIG
base=f"http://127.0.0.1:{CONFIG['port']}";s=requests.Session();s.trust_env=False
s.post(base+'/api/login',json={'password':(R/'run/access.key').read_text().strip()}).raise_for_status()
checks=[]
def record(name,passed,details=None):checks.append({'name':name,'passed':bool(passed),'details':details});(R/'reports/api_acceptance.json').write_text(json.dumps({'status':'OBSERVED','checks':checks},ensure_ascii=False,indent=2))
def post(path,body):
 x=s.post(base+path,json=body,timeout=600);x.raise_for_status();return x.json()
def wait_assets(ds,timeout=600):
 start=time.time()
 while time.time()-start<timeout:
  xs=s.get(base+'/api/assets',params={'dataset':ds}).json()
  if xs and all(x['status'] in ['indexed','failed','cancelled'] for x in xs):return xs
  time.sleep(.5)
 raise TimeoutError(ds)
unauth=requests.Session();unauth.trust_env=False;record('all API status requires authentication',unauth.get(base+'/api/status').status_code==401)
for name in ['../escape.txt','/tmp/escape.txt','a/../../escape.txt']:
 x=s.post(base+'/api/upload',params={'dataset':'personal','name':name},data=b'no',timeout=10);record('reject path '+name,x.status_code==400,{'http':x.status_code})
x=s.post(base+'/api/import-directory',json={'dataset':'personal','path':'../data'},timeout=10);record('reject directory outside whitelist',x.status_code==400)
canary=R/'data/test-canary';canary.mkdir(exist_ok=True);(canary/'private.txt').write_text('project-owned whitelist test canary')
link=R/'import/escape-test'
if not link.exists():link.symlink_to(canary,target_is_directory=True)
x=s.post(base+'/api/import-directory',json={'dataset':'personal','path':'escape-test'},timeout=10);record('reject symlink outside whitelist',x.status_code==400)
link.unlink()
ds=post('/api/datasets',{'name':'API 验收独立数据'})['id']
for name,data in [('empty.txt',b''),('broken.png',b'not a real PNG'),('audio.wav',b'RIFF fake audio')]:
 x=s.post(base+'/api/upload',params={'dataset':ds,'name':name},data=data,timeout=10)
 if not data:record('reject empty file',x.status_code==400)
 else:x.raise_for_status()
xs=wait_assets(ds);record('bad and unsupported files isolated',all(x['status']=='failed' for x in xs),[{'name':x['name'],'error':x['error']} for x in xs])
# Real bounded over-limit transfer; server must reject and remove its temporary file.
x=s.post(base+'/api/upload',params={'dataset':ds,'name':'oversize.txt'},data=(b'x'*1024**2 for _ in range(129)),timeout=60);record('reject actual >128 MiB upload',x.status_code==413,{'http':x.status_code})
photo=sorted((R/'import/demo/photos').glob('*.jpg'))[15]
before=s.get(base+'/api/status').json()['encoding_calls'];x=s.post(base+'/api/upload',params={'dataset':'personal','name':'重复照片.jpg'},data=photo.read_bytes());x.raise_for_status();time.sleep(1);after=s.get(base+'/api/status').json()['encoding_calls']
record('duplicate bytes skip inference',x.json()['duplicate'] and before==after,{'before':before,'after':after,'asset':x.json()['asset']})
cancel_ds=post('/api/datasets',{'name':'暂停取消恢复验收'})['id'];post('/api/control',{'dataset':cancel_ds,'action':'pause'})
for i in range(3):
 content=(f'第 {i} 组远足记录。携带雨衣和饮用水，在山间观察植物。\n\n'*200).encode()
 x=s.post(base+'/api/upload',params={'dataset':cancel_ds,'name':f'长文本 {i}.txt'},data=content);x.raise_for_status()
xs=s.get(base+'/api/assets',params={'dataset':cancel_ds}).json();record('pause retains pending uploads',all(x['status']=='pending' for x in xs))
post('/api/control',{'dataset':cancel_ds,'action':'resume'});time.sleep(.3);post('/api/control',{'dataset':cancel_ds,'action':'cancel'});time.sleep(.5)
xs=s.get(base+'/api/assets',params={'dataset':cancel_ds}).json();record('cancel at safe boundary',any(x['status']=='cancelled' for x in xs))
post('/api/control',{'dataset':cancel_ds,'action':'retry'});xs=wait_assets(cancel_ds,1200);record('resume cancelled work completes',all(x['status']=='indexed' for x in xs))
query={'dataset':'demo','text':'一只猫','k':10,'kind':'image'};result=post('/api/search',query);record('real query returns finite scores',bool(result['results']) and all(-1.01<=x['score']<=1.01 for x in result['results']),result['timings'])
(R/'reports/restart_before.json').write_text(json.dumps(result,ensure_ascii=False))
print(json.dumps({'checks':len(checks),'passed':sum(x['passed'] for x in checks)},ensure_ascii=False));assert all(x['passed'] for x in checks)
