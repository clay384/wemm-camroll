"""Local project client; secrets read only from restricted runtime file."""
import sys,json,time
from pathlib import Path
import requests,yaml
R=Path(__file__).resolve().parents[1];c=yaml.safe_load((R/'config.yaml').read_text());base=f"http://127.0.0.1:{c['port']}";s=requests.Session();s.trust_env=False
res=s.post(base+'/api/login',json={'password':(R/'run/access.key').read_text().strip()});res.raise_for_status()
cmd=sys.argv[1]
if cmd=='demo':res=s.post(base+'/api/import-directory',json={'dataset':'demo','path':'demo'},timeout=180)
elif cmd=='benchmark':res=s.post(base+'/api/benchmark',timeout=20)
elif cmd=='status':res=s.get(base+'/api/status')
elif cmd=='export':
 for name in ['metrics.json','results.csv','per_query_results.jsonl','validation_report.html','validation_report.md']:
  res=s.get(base+'/api/report/'+name);res.raise_for_status();print(name,len(res.content))
 sys.exit(0)
else:raise SystemExit('demo | benchmark | status | export')
res.raise_for_status();print(json.dumps(res.json(),ensure_ascii=False,indent=2))
