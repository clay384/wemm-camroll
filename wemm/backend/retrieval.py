import time,json,hashlib
import numpy as np
from backend.common import ROOT,db

def search(v,fingerprint,dataset='demo',kind=None,k=10,dimension=4096,exclude=None,aggregate=True):
 t=time.perf_counter()
 if dimension not in [4096,1024,256]:raise ValueError('不支持的维度')
 if k is not None and k not in [5,10,20,50]:raise ValueError('不支持的 Top-K')
 with db() as c:
  rows=c.execute('SELECT u.*,MIN(m.name) AS name FROM units u JOIN members m ON m.asset=u.asset WHERE m.dataset=? AND u.fingerprint=? GROUP BY u.id ORDER BY u.id',(dataset,fingerprint)).fetchall()
 rows=[dict(x) for x in rows if (not kind or x['kind']==kind) and x['asset'] not in (exclude or [])]
 if not rows:return [],{'retrieval_s':time.perf_counter()-t,'units_searched':0}
 mat=np.stack([np.load(ROOT/x['vector'],allow_pickle=False)[:dimension] for x in rows]);mat/=np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12)
 q=v[:dimension].copy();q/=max(np.linalg.norm(q),1e-12);scores=mat@q
 ordered=sorted(range(len(rows)),key=lambda i:(-float(scores[i]),rows[i]['asset'],rows[i]['id']))
 results=[];seen={}
 for i in ordered:
  x=rows[i];hit={'id':x['id'],'asset':x['asset'],'kind':x['kind'],'name':x['name'],'score':float(scores[i]),'meta':json.loads(x['meta']),'thumbnail':f"/api/units/{x['id']}/thumbnail" if x['thumb'] else None,'content':f"/api/units/{x['id']}/content"}
  if aggregate and x['asset'] in seen:
   seen[x['asset']]['children'].append(hit);continue
  hit['children']=[];seen[x['asset']]=hit;results.append(hit)
 for rank,x in enumerate(results):x['rank']=rank+1
 return results[:k],{'retrieval_s':time.perf_counter()-t,'units_searched':len(rows),'aggregation':'max unit score' if aggregate else 'unit','dimension':dimension}
