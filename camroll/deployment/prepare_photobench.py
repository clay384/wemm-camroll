"""Create a photo-only manifest from the frozen source, excluding QA labels."""
import hashlib
import json
import sys
from pathlib import Path

source=Path('/home/realmagic/wemm-embedding-lab/import/photobench-20260914/album3')
frozen=Path('/home/realmagic/wemm-embedding-lab/reports/photobench/20260914-115425')
project=Path('/home/realmagic/camroll-wemm')
out=project/'reports'/sys.argv[1]
spec_path=project/'runtime/photobench-album3.json'
metadata_path=source/'metadata/image_metadata.jsonl'
metadata={r['filename']:r for r in map(json.loads,metadata_path.read_text().splitlines())}
media=[r for r in json.loads((frozen/'media.json').read_text()) if r['kind']=='image']
assert len(media)==334
images=[]
hashes={}
for row in media:
    path=(source/row['path']).resolve()
    assert path.is_relative_to(source/'photos')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==row['sha256'],str(path)
    hashes[row['path']]=digest
    meta=metadata.get(path.name,{})
    date=meta.get('capture_date') if meta.get('date_status')=='metadata_exact' else None
    images.append({'path':str(path),'date':date})
images.sort(key=lambda x:(x['date'] or '9999',x['path']))
spec={'album_id':'photobench-album3','images':images}
if spec_path.exists():
    assert json.loads(spec_path.read_text())==spec,'Manifest changed; use a new run'
else:
    spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2))
audit={'media_count':len(images),'unique_assets':len(set(hashes.values())),
       'dated_count':sum(bool(x['date']) for x in images),'whitelisted_fields':['path','capture_date (metadata_exact only)'],
       'no_qa_or_annotations_in_build':True,'source_hashes':hashes,
       'metadata_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
       'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest()}
(out/'build_input_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'},ensure_ascii=False))
