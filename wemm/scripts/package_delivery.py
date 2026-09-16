"""Explicit allow-list source package: no weights, environments, access key or private imports."""
from pathlib import Path
import tarfile,hashlib,json
R=Path(__file__).resolve().parents[1];out=R/'deliveries';out.mkdir(exist_ok=True)
files=['AGENTS.md','README.md','CURRENT_STATE.md','WORKLOG.md','TASK.md','SOURCE_LOCK.json','config.yaml','.env.example','.gitignore','requirements.lock','hf_metadata.json','data_manifest.jsonl']
dirs=['backend','frontend/src','frontend/dist','scripts','official-locked','benchmark','reports']
items=[R/x for x in files if (R/x).is_file()]
for d in dirs:
 items.extend(p for p in (R/d).rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','node_modules']) and p.suffix!='.pyc')
secret=(R/'run/access.key').read_bytes().strip() if (R/'run/access.key').exists() else None
for p in items:
 if secret and secret in p.read_bytes():raise RuntimeError('Restricted runtime material found in source package input')
archive=out/'wemm-embedding-lab-source.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(set(items)):
  t.add(p,arcname='wemm-embedding-lab/'+str(p.relative_to(R)),recursive=False)
record={'status':'OBSERVED','archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(set(items)),'excluded':['model weights','environments','runtime key','private imports','database','raw private files','model cache'],'runtime_key_scan':'PASS'}
(out/'source_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
