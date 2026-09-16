from pathlib import Path
import json,hashlib,subprocess,datetime
r=Path(__file__).resolve().parents[1]; p=r/'models/WeMM-Embedding-9B'
meta=json.loads((r/'hf_metadata.json').read_text()); files=[]
for f in meta['files']:
 path=p/f['name']; assert path.is_file(),str(path)
 assert path.stat().st_size==f['size'],str(path)
 proof_path=r/'models/http-parts/VERIFIED.json';lfs=f.get('lfs')
 proof=json.loads(proof_path.read_text()) if f['name']=='model.safetensors' and proof_path.exists() else None
 if proof and proof['bytes']==path.stat().st_size and path.stat().st_mtime<=proof['verified_at']:
  sha=proof['sha256']
 else:
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
  sha=h.hexdigest()
 if lfs and lfs.get('sha256'): assert sha==lfs['sha256'],str(path)
 files.append({'name':f['name'],'size':f['size'],'sha256':sha})
 print(f['name'],sha,flush=True)
if (r/'official-locked/.git').exists():
 code_commit=subprocess.check_output(['git','-C',str(r/'official-locked'),'rev-parse','HEAD'],text=True).strip()
else:
 code_commit=(r/'official-locked/COMMIT').read_text().strip()
 for name,sha in json.loads((r/'official-locked/HASHES.json').read_text()).items():
  assert hashlib.sha256((r/'official-locked'/name).read_bytes()).hexdigest()==sha,'Reference code mismatch'
lock={'platform':'Hugging Face','model_id':meta['id'],'revision':meta['sha'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'code_commit':code_commit,'modelscope_failure':'Official SDK HubApi.get_model_files returned 404 record not found for Tencent-Hunyuan/WeMM-Embedding-9B master on 2026-09-11','network':'Mac HTTP 7890 -> SSH remote loopback 17890','software':subprocess.check_output([str(r/'.venv/bin/pip'),'freeze'],text=True).splitlines()}
(r/'SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2)); (r/'requirements.lock').write_text('\n'.join(lock['software'])+'\n')
