import subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'import/demo/generated';D.mkdir(parents=True,exist_ok=True)
rows=[];evidence=[]
for i,(color,zh) in enumerate([('red','红色'),('blue','蓝色'),('green','绿色'),('yellow','黄色')]):
 p=D/f'v{i:02d}.mp4'
 # Known synthetic motion scene; no claim about real-world video quality.
 subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i',f'color=white:s=320x240:r=8:d=10','-vf',f"drawbox=x=80:y=60:w=100:h=100:color={color}:t=fill",'-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(p)],check=True)
 sha=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'path':str(p.relative_to(R/'import')),'sha256':sha,'license':'CC0-1.0','kind':'synthetic_video','source':'scripts/generate_videos.py','seed':20260911})
 evidence.append({'sha256':sha,'color':zh,'description':f'白色背景中的{zh}方块','limitation':'static synthetic scene encoded as native video; not temporal understanding proof'})
(R/'data/manifests/videos.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));(R/'benchmark/video_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
