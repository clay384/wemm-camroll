"""Read only COCO val metadata members using HTTP byte ranges, bounded retries."""
from pathlib import Path
import io,json,zipfile,requests,time,os,signal
ROOT=Path(__file__).resolve().parents[1]; dest=ROOT/'data/sources'; dest.mkdir(parents=True,exist_ok=True)
URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
class Remote(io.RawIOBase):
 def __init__(self): self.pos=0; self.size=252907541; self.s=requests.Session(); self.s.trust_env=False
 def seekable(self): return True
 def seek(self,n,whence=0):
  self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
  return self.pos
 def tell(self): return self.pos
 def read(self,n=-1):
  n=min(self.size-self.pos,n if n>=0 else self.size)
  if not n:return b''
  for a in range(4):
   try:
    r=self.s.get(URL,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=120); r.raise_for_status()
    if r.status_code!=206 or len(r.content)!=n: raise ValueError('Unexpected range response')
    self.pos+=n; return r.content
   except Exception:
    if a==3:raise
    time.sleep(2**a)
f=Remote()
with zipfile.ZipFile(f) as z:
 for name in ['annotations/captions_val2017.json','annotations/instances_val2017.json']:
  out=dest/Path(name).name
  if not out.exists(): out.write_bytes(z.read(name))
  print(out.name,out.stat().st_size,flush=True)
