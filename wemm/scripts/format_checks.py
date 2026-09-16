from pathlib import Path
import sys,json,subprocess,hashlib,shutil,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT
from PIL import Image
import pillow_heif
import pymupdf as fitz
pillow_heif.register_heif_opener()
d=ROOT/'import/formats';d.mkdir(parents=True,exist_ok=True);bad=ROOT/'data/negative-formats';bad.mkdir(exist_ok=True)
image=Image.open(ROOT/'import/demo/photos/0015.jpg').convert('RGB')
files=[]
for ext,fmt in [('jpg','JPEG'),('png','PNG'),('webp','WEBP'),('tiff','TIFF'),('bmp','BMP'),('heic','HEIF')]:
 p=d/('sample.'+ext)
 try:image.save(p,format=fmt);files.append((p,True))
 except Exception as e:print('Optional format generation failed',ext,str(e))
p=d/'orientation.jpg';exif=Image.Exif();exif[274]=6;Image.new('RGB',(80,120),'red').save(p,exif=exif);files.append((p,True))
for ext in ['txt','md']:
 p=d/('sample.'+ext);p.write_text('# 独立格式验证\n\n这是一份 UTF-8 '+ext+' 文件。');files.append((p,True))
p=d/'sample.pdf';shutil.copyfile(ROOT/'import/demo/generated/d00.pdf',p);files.append((p,True))
for ext,codec in [('mp4','libx264'),('mov','libx264'),('webm','libvpx-vp9')]:
 p=d/('sample.'+ext);subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(ROOT/'import/demo/generated/b00.mp4'),'-t','2','-an','-c:v',codec,'-threads','1',str(p)],check=True);files.append((p,True))
for name,data in [('broken.png',b'bad image'),('empty.txt',b''),('audio.wav',b'RIFF not supported'),('invalid.mp4',b'#EXTM3U\nhttp://127.0.0.1/forbidden')]:
 p=bad/name;p.write_bytes(data);files.append((p,False))
p=bad/'too-many-pages.pdf';doc=fitz.open()
for _ in range(101):doc.new_page()
doc.save(p);doc.close();files.append((p,False))
p=bad/'too-large.png';Image.new('RGB',(8000,6000)).save(p);files.append((p,False))
results=[]
for p,expected in files:
 sha=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,ROOT/'data/originals'/sha);t=time.perf_counter()
 proc=subprocess.run([sys.executable,str(ROOT/'backend/preprocess.py'),sha,p.suffix],capture_output=True,text=True,timeout=70)
 row={'file':p.name,'sha256':sha,'expected_success':expected,'returncode':proc.returncode,'passed':(proc.returncode==0)==expected,'elapsed_s':time.perf_counter()-t,'units':json.loads(proc.stdout) if proc.returncode==0 else None,'error':proc.stderr[-1500:] if proc.returncode else None}
 if p.name=='orientation.jpg' and proc.returncode==0:
  row['orientation_size']=list(Image.open(ROOT/row['units'][0]['path']).size);row['passed']=row['orientation_size']==[120,80]
 results.append(row);print(p.name,row['passed'],flush=True)
(ROOT/'reports/format_preprocessing.json').write_text(json.dumps({'status':'OBSERVED','scope':'actual CPU format decoding/preprocessing only; embedding checks follow separately','results':results},ensure_ascii=False,indent=2))
assert all(x['passed'] for x in results)
