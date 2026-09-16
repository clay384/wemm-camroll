"""Runs in a bounded CPU-only child; never imports the model runtime."""
import sys,json,subprocess,resource,time,os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT,CONFIG
from PIL import Image,ImageOps
Image.MAX_IMAGE_PIXELS=CONFIG['image_max_pixels']
try:
 import pillow_heif;pillow_heif.register_heif_opener()
except ImportError:pass

def run(asset,ext):
 p=ROOT/'data/originals'/asset; out=ROOT/'data/derived'/asset;out.mkdir(exist_ok=True)
 units=[]
 def add(kind,path,thumb,meta):units.append({'kind':kind,'path':str(path.relative_to(ROOT)),'thumb':str(thumb.relative_to(ROOT)) if thumb else None,'meta':meta})
 def save(im,name):
  im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((2048,2048)); target=out/name;im.save(target)
  thumb=im.copy();thumb.thumbnail((384,384));tp=out/('thumb-'+name);thumb.save(tp);return target,tp
 if ext in ['.jpg','.jpeg','.png','.webp','.heic','.heif','.tif','.tiff','.bmp']:
  with Image.open(p) as im:
   size=im.size
   if size[0]*size[1]>CONFIG['image_max_pixels']:raise ValueError('图片超过像素上限')
   target,tp=save(im,'image.jpg');add('image',target,tp,{'original_size':size,'exif_transpose':True,'rgb':True})
 elif ext=='.wemmjoint':
  from backend.common import db
  payload=json.loads(p.read_text());text=payload['text']
  if not isinstance(text,str) or not 0<len(text)<=4000:raise ValueError('联合候选文字超限')
  with db() as c:source=c.execute("SELECT path FROM units WHERE asset=? AND kind='image' LIMIT 1",(payload['image_asset'],)).fetchone()
  if not source:raise ValueError('联合候选图尚未索引')
  source_path=(ROOT/source[0]).resolve()
  if not source_path.is_relative_to(ROOT/'data/derived'):raise ValueError('非法图像来源')
  with Image.open(source_path) as im:
   target,tp=save(im,'joint.jpg');add('image',target,tp,{'candidate_mode':'image_text_joint','joint_text':text,'source_image_asset':payload['image_asset']})
 elif ext in ['.txt','.md']:
  if p.stat().st_size>CONFIG['text_max_bytes']:raise ValueError('文本文件超过 2 MB 上限')
  text=p.read_text(encoding='utf-8-sig')
  if len(text)>CONFIG['text_max_chars']:raise ValueError('文本超过 200000 字符上限')
  if not text.strip():raise ValueError('文本为空')
  target=out/'text.txt';target.write_text(text);add('text',target,None,{'characters':len(text)})
 elif ext=='.pdf':
  import pymupdf as fitz
  with fitz.open(p) as doc:
   if doc.needs_pass:raise ValueError('不支持加密 PDF')
   if not 0<len(doc)<=CONFIG['pdf_max_pages']:raise ValueError('PDF 页数超限或为空')
   for i,page in enumerate(doc):
    scale=min(1.5,1536/max(page.rect.width,page.rect.height));pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
    target,tp=save(Image.frombytes('RGB',[pix.width,pix.height],pix.samples),f'page-{i+1}.jpg')
    add('pdf',target,tp,{'page':i+1,'channel':'visual_page','render_scale':scale})
 elif ext in ['.mp4','.mov','.webm']:
  header=p.read_bytes()[:16] if p.stat().st_size<16 else p.open('rb').read(16)
  if not (header[:4]==bytes.fromhex('1a45dfa3') or header[4:8] in [b'ftyp',b'moov',b'wide',b'mdat']):raise ValueError('无效 MP4/MOV/WebM 容器；不接受播放列表')
  info=json.loads(subprocess.check_output(['ffprobe','-threads','1','-protocol_whitelist','file,pipe','-v','error','-show_format','-show_streams','-of','json',str(p)],timeout=15))
  duration=float(info['format']['duration'])
  if duration<=0 or duration>CONFIG['video_max_seconds']:raise ValueError('视频时长超过 300 秒或为空')
  streams=[x for x in info['streams'] if x['codec_type']=='video']
  if not streams:raise ValueError('未发现视频轨道；不支持音频检索')
  seg=CONFIG['video_segment_seconds'];start=0
  while start<duration:
   end=min(start+seg,duration);target=out/f'clip-{int(start)}.mp4';tp=out/f'clip-{int(start)}.jpg'
   subprocess.run(['ffmpeg','-protocol_whitelist','file,pipe','-v','error','-nostdin','-y','-threads','1','-ss',str(start),'-i',str(p),'-t',str(end-start),'-an','-vf','scale=384:384:force_original_aspect_ratio=decrease:force_divisible_by=2','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(target)],timeout=45,check=True)
   # A fractional tail can contain only one decoded frame. Preserve it and
   # duplicate its last frame to meet the fixed native-video sampling budget.
   count=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(target)],timeout=15))
   decoded_frames=int(count['streams'][0]['nb_read_frames']);required_frames=CONFIG['profiles'][CONFIG['profile']]['video_frames']
   if decoded_frames<1:raise ValueError('视频切片没有可解码帧')
   padded_frames=max(0,required_frames-decoded_frames)
   if padded_frames:
    padded=out/f'clip-{int(start)}-padded.mp4'
    subprocess.run(['ffmpeg','-protocol_whitelist','file,pipe','-v','error','-nostdin','-y','-threads','1','-i',str(target),'-an','-vf',f'tpad=stop_mode=clone:stop={padded_frames}','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(padded)],timeout=45,check=True)
    os.replace(padded,target)
   subprocess.run(['ffmpeg','-protocol_whitelist','file,pipe','-v','error','-nostdin','-y','-threads','1','-i',str(target),'-frames:v','1',str(tp)],timeout=15,check=True)
   add('video',target,tp,{'start':start,'end':end,'duration':duration,'mode':'native_video','audio':'ignored','source_codec':streams[0]['codec_name'],'conversion':'H264 bounded derivative, original retained','frames':required_frames,'decoded_frames_before_padding':decoded_frames,'repeated_tail_frames':padded_frames});start=end
 else:raise ValueError('不支持此格式；音频不受支持')
 return units
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(90,90))
 try:print(json.dumps(run(sys.argv[1],sys.argv[2]),ensure_ascii=False))
 except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
