import json,random,hashlib,time,concurrent.futures
from pathlib import Path
import requests
R=Path(__file__).resolve().parents[1]; S=R/'data/sources'; D=R/'import/demo/photos'; D.mkdir(parents=True,exist_ok=True)
a=json.loads((S/'instances_val2017.json').read_text()); cats={x['id']:x['name'] for x in a['categories']}; lic={x['id']:x for x in a['licenses']}
by={}
for x in a['annotations']: by.setdefault(x['image_id'],set()).add(cats[x['category_id']])
images=[x for x in a['images'] if x['license'] in [4,5,7,8]]
rng=random.Random(20260911); rng.shuffle(images)
selected=[]; counts={}
# Broad coverage first, then seeded sample. No ranking/model outputs involved.
for cat in cats.values():
 for im in images:
  if cat in by.get(im['id'],set()) and im not in selected:
   selected.append(im); break
for im in images:
 if im not in selected:selected.append(im)
 if len(selected)>=120:break

def fetch(pair):
 i,im=pair; path=D/f'{i:04d}.jpg'; url='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+im['file_name']
 for n in range(4):
  try:
   if not path.exists():
    s=requests.Session(); s.trust_env=False; res=s.get(url,timeout=60);res.raise_for_status(); path.write_bytes(res.content)
   sha=hashlib.sha256(path.read_bytes()).hexdigest()
   row={'path':str(path.relative_to(R/'import')),'sha256':sha,'source_url':url,'attribution_url':im.get('flickr_url'),'license':lic[im['license']],'coco_id':im['id'],'obtained':'COCO official validation subset','kind':'real_photo'}
   print('downloaded',i,flush=True);return row,{'sha256':sha,'categories':sorted(by.get(im['id'],set()))}
  except Exception as e:
   if n==3:return {'error':str(e),'source_url':url},None
   time.sleep(2**n)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(fetch,enumerate(selected)))
(R/'data/manifests').mkdir(parents=True,exist_ok=True); (R/'benchmark').mkdir(exist_ok=True)
(R/'data/manifests/photos.jsonl').write_text(''.join(json.dumps(x[0],ensure_ascii=False)+'\n' for x in rows))
(R/'benchmark/photo_evidence.json').write_text(json.dumps([x[1] for x in rows if x[1]],ensure_ascii=False,indent=2))
assert all(x[1] for x in rows), 'Some photos failed; rerun resumes'
