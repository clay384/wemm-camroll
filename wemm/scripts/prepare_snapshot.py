"""Restore only the locked official small files; large weights use resumable ranges."""
import hashlib,json,os,time
from pathlib import Path
import requests
R=Path(__file__).resolve().parents[1];lock=json.loads((R/'SOURCE_LOCK.json').read_text());dest=R/'models/WeMM-Embedding-9B';dest.mkdir(parents=True,exist_ok=True)
proxy=os.environ.get('HTTPS_PROXY')
if not proxy:raise SystemExit('Set verified HTTPS_PROXY for this download process')
s=requests.Session();s.trust_env=False;s.proxies={'http':proxy,'https':proxy}
for f in lock['files']:
 if f['name']=='model.safetensors':continue
 p=dest/f['name'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']:continue
 for retry in range(5):
  try:
   r=s.get(f"https://huggingface.co/{lock['model_id']}/resolve/{lock['revision']}/{f['name']}",timeout=90);r.raise_for_status();assert len(r.content)==f['size'];assert hashlib.sha256(r.content).hexdigest()==f['sha256'];p.write_bytes(r.content);print('Verified',f['name']);break
  except Exception:
   if retry==4:raise
   time.sleep(2**retry)
