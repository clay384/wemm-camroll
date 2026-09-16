import sys,json,time,subprocess,importlib.util,hashlib,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.encoder import Encoder,ROOT
import numpy as np
from PIL import Image,ImageDraw
r=ROOT/'reports'; r.mkdir(exist_ok=True)
p=ROOT/'data/smoke'; p.mkdir(parents=True,exist_ok=True)
im=Image.new('RGB',(320,256),'white'); d=ImageDraw.Draw(im); d.ellipse((70,40,240,210),fill='red'); im.save(p/'a.png')
subprocess.run(['ffmpeg','-y','-f','lavfi','-i','testsrc2=size=256x192:rate=4','-t','2','-pix_fmt','yuv420p',str(p/'a.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
e=Encoder(); results={'load_s':e.load_seconds,'load_peak_allocated_mib':e.load_peak_allocated_mib,'load_peak_reserved_mib':e.load_peak_reserved_mib,'fingerprint':e.fingerprint,'status':'OBSERVED','tests':[]}; vs={}
for name,kw in [('text_a',{'text':'一个红色圆形'}),('text_repeat',{'text':'一个红色圆形'}),('image',{'image':p/'a.png'}),('text_after_image',{'text':'一个红色圆形'}),('video',{'video':p/'a.mp4'}),('text_after_video',{'text':'一个红色圆形'}),('joint',{'image':p/'a.png','text':'红色圆形'}),('negative',{'text':'一只蓝色汽车在雪地行驶'})]:
 v,s=e.encode(**kw); vs[name]=v
 results['tests'].append({'name':name,'shape':list(v.shape),'norm':float(np.linalg.norm(v)),'finite':bool(np.isfinite(v).all()),**s})
 (r/'smoke.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
 print(name,s,flush=True)
for name,kw in [('real_cat',{'image':ROOT/'import/demo/photos/0015.jpg'}),('real_bus',{'image':ROOT/'import/demo/photos/0001.jpg'}),('cat_text',{'text':'一只躺着休息的猫'}),('bus_text',{'text':'城市街道上的公交车'}),('public_animation',{'video':ROOT/'import/demo/generated/b00.mp4'})]:
 v,s=e.encode(**kw);vs[name]=v;results['tests'].append({'name':name,'shape':list(v.shape),'norm':float(np.linalg.norm(v)),'finite':bool(np.isfinite(v).all()),**s});print(name,s,flush=True)
pdf_hash=hashlib.sha256((ROOT/'import/demo/generated/d00.pdf').read_bytes()).hexdigest()
v,s=e.encode(image=ROOT/'data/derived'/pdf_hash/'page-1.jpg');vs['pdf_page']=v;results['tests'].append({'name':'pdf_page','shape':list(v.shape),'norm':float(np.linalg.norm(v)),'finite':bool(np.isfinite(v).all()),**s})
results['real_pair_similarities']={im:{txt:float(vs[im]@vs[txt]) for txt in ['cat_text','bus_text']} for im in ['real_cat','real_bus']}
results['parameter_dtypes']=dict(collections.Counter(str(p.dtype) for p in e.model.parameters()))
results['parameter_devices']=dict(collections.Counter(str(p.device) for p in e.model.parameters()))
results['parameter_count']=sum(p.numel() for p in e.model.parameters())
results['stability']={k:{'max_abs':float(np.abs(vs[k]-vs['text_a']).max()),'cosine':float(vs[k]@vs['text_a'])} for k in ['text_repeat','text_after_image','text_after_video']}
results['image_similarities']={k:float(vs['image']@vs[k]) for k in ['text_a','negative']}
# Same loaded model, untouched upstream encode, same explicitly bounded messages.
spec=importlib.util.spec_from_file_location('upstream',ROOT/'official-locked/examples/transformers_inference.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ref=mod.encode(e.processor,e.model,e.messages(text='一个红色圆形'),'cuda',4096).numpy()[0]
results['upstream_alignment']={'max_abs':float(abs(ref-vs['text_a']).max()),'cosine':float(ref@vs['text_a'])}
results['assertions']={'norm':all(abs(t['norm']-1)<1e-5 for t in results['tests']),'stability':all(x['max_abs']<1e-4 for x in results['stability'].values()),'upstream':results['upstream_alignment']['max_abs']<1e-4}
np.savez(r/'smoke_vectors.npz',**vs)
(r/'smoke.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)); print(json.dumps(results,ensure_ascii=False),flush=True)
assert all(results['assertions'].values())
