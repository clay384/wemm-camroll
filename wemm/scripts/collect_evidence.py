import json,time,os,signal,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];samples=[]
for line in (R/'logs/gpu_samples.jsonl').read_text().splitlines():
 try:samples.append(json.loads(line))
 except ValueError:pass
cards={}
for s in samples:
 for c in s.get('whole_cards_including_other_processes',[]):
  v=cards.setdefault(str(c['index']),{'uuid':c['uuid'],'sampled_peak_used_mib':0,'minimum_free_mib':999999,'samples':0});v['sampled_peak_used_mib']=max(v['sampled_peak_used_mib'],c['used_mib']);v['minimum_free_mib']=min(v['minimum_free_mib'],c['free_mib']);v['samples']+=1;v['last']=c
out={'status':'OBSERVED','sampling_interval_s':1,'note':'Whole-card samples include other processes. Sampled peak is not a continuous peak or PyTorch allocation.','cards':cards,'start':samples[0]['time'],'end':samples[-1]['time'],'current_gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True),'disk':subprocess.check_output(['df','-h',str(R)],text=True)}
(R/'reports/resource_final.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if '--stop-monitor' in sys.argv:
 p=int((R/'run/gpu-monitor.pid').read_text());proc=Path('/proc')/str(p)
 if proc.exists() and (proc/'cwd').resolve()==R and b'scripts/gpu_monitor.py' in (proc/'cmdline').read_bytes():os.kill(p,signal.SIGTERM);print('Stopped owned GPU monitor',p)
