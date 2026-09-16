"""Bounded resumable HTTP ranges through the configured proxy.
Preserves the sequential prefix produced by the official Xet reconstruction.
Final SHA256 must match official LFS metadata before installing the snapshot.
"""
from pathlib import Path
import requests,os,json,time,hashlib,concurrent.futures,shutil,threading
R=Path(__file__).resolve().parents[1];M=R/'models/WeMM-Embedding-9B';D=R/'models/http-parts';D.mkdir(parents=True,exist_ok=True);M.mkdir(parents=True,exist_ok=True)
SIZE=18815757458;SHA='b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4';REV='00c52839de57a6d4fd5b78cf5522ccf0ac8ea482'
URL=f'https://huggingface.co/tencent/WeMM-Embedding-9B/resolve/{REV}/model.safetensors'
verified=D/'VERIFIED.json';existing=M/'model.safetensors'
if verified.exists() and existing.exists():
 proof=json.loads(verified.read_text())
 if proof.get('sha256')==SHA and existing.stat().st_size==SIZE and existing.stat().st_mtime<=proof['verified_at']:
  print('Already verified; no download or repeat hashing needed');raise SystemExit(0)
proxy=os.environ.get('HTTPS_PROXY')
if not proxy:raise SystemExit('A working proxy is required for HF fallback')
P=D/'prefix';meta=D/'resume.json'
if not meta.exists():
 candidates=list((M/'.cache/huggingface/download').glob('*'+SHA+'*.incomplete'))
 if candidates:
  source=max(candidates,key=lambda x:x.stat().st_size)
  with source.open('rb') as f:
   end=os.lseek(f.fileno(),0,os.SEEK_HOLE)
   if end<source.stat().st_size:raise RuntimeError('Sparse prefix cannot safely resume')
  shutil.copyfile(source,P)
 else:P.write_bytes(b'')
 meta.write_text(json.dumps({'prefix_bytes':P.stat().st_size,'size':SIZE,'sha256':SHA,'revision':REV,'reason':'Xet adaptive concurrency collapsed to 1 with no disk progress; preserve sequential complete prefix'}))
PREFIX=json.loads(meta.read_text())['prefix_bytes'];CHUNK=8*1024**2

local=threading.local();url_lock=threading.Lock();signed={'url':None,'time':0}
def session():
 if not hasattr(local,'session'):
  local.session=requests.Session();local.session.trust_env=False;local.session.proxies={'http':proxy,'https':proxy}
 return local.session
def location():
 with url_lock:
  if not signed['url'] or time.time()-signed['time']>2400:
   r=session().head(URL,allow_redirects=False,timeout=30);r.raise_for_status()
   if r.status_code not in [302,307]:raise RuntimeError('Expected official signed download redirect')
   signed.update(url=r.headers['Location'],time=time.time())
  return signed['url']
def fetch(i):
 start=PREFIX+i*CHUNK;end=min(SIZE,start+CHUNK)-1;part=D/f'{i:05d}.part';n=end-start+1
 if part.exists() and part.stat().st_size==n:return
 for retry in range(5):
  try:
   s=session()
   response=s.get(location(),headers={'Range':f'bytes={start}-{end}'},timeout=(20,120),stream=True)
   if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':raise RuntimeError('Invalid range response: '+str(response.status_code))
   tmp=part.with_suffix('.tmp')
   with tmp.open('wb') as f:
    for block in response.iter_content(256*1024):f.write(block)
   if tmp.stat().st_size!=n:raise RuntimeError('Incomplete range')
   tmp.replace(part);print(json.dumps({'part':i,'bytes':n,'time':time.time()}),flush=True);return
  except Exception as e:
   print(json.dumps({'retry':retry,'part':i,'error_type':type(e).__name__}),flush=True)
   if retry==4:raise
   time.sleep(10*2**retry)
count=(SIZE-PREFIX+CHUNK-1)//CHUNK
with concurrent.futures.ThreadPoolExecutor(max_workers=min(16,int(os.environ.get("WEMM_DOWNLOAD_WORKERS","8")))) as pool:list(pool.map(fetch,range(count)))
final=M/'model.safetensors.assembling';h=hashlib.sha256()
with final.open('wb') as out:
 for p in [P]+[D/f'{i:05d}.part' for i in range(count)]:
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);out.write(block)
assert final.stat().st_size==SIZE
assert h.hexdigest()==SHA,'SHA256 mismatch; preserve files and do not load'
final.replace(M/'model.safetensors');(D/'VERIFIED.json').write_text(json.dumps({'sha256':SHA,'bytes':SIZE,'verified_at':time.time()}));print('VERIFIED',SHA,flush=True)
