import subprocess,json,time,os,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1];(R/'run').mkdir(exist_ok=True);(R/'run/gpu-monitor.pid').write_text(str(os.getpid()))
with (R/'logs/gpu_samples.jsonl').open('a',buffering=1) as f:
 while True:
  try:
   out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3)
   rows=[]
   for line in out.strip().splitlines():
    index,uuid,used,free,util=[s.strip() for s in line.split(',')];rows.append({'index':int(index),'uuid':uuid,'used_mib':int(used),'free_mib':int(free),'util_percent':int(util)})
   f.write(json.dumps({'time':time.time(),'whole_cards_including_other_processes':rows})+'\n')
  except Exception as e:f.write(json.dumps({'time':time.time(),'error':type(e).__name__})+'\n')
  time.sleep(1)
