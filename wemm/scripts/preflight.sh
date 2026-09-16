#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p reports
python3 - <<'PY'
import json,subprocess,datetime,os
from pathlib import Path
checks={'host':['hostname'],'os':['uname','-a'],'gpu':['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.free','--format=csv'],'gpu_processes':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],'memory':['free','-h'],'disk':['df','-h','.'],'ports':['ss','-ltn'],'cpu':['lscpu']}
data={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(Path.cwd()),'status':'OBSERVED'}
for key,cmd in checks.items():
 p=subprocess.run(cmd,capture_output=True,text=True);data[key]={'stdout':p.stdout,'returncode':p.returncode}
Path('reports/environment.json').write_text(json.dumps(data,indent=2));print('reports/environment.json')
PY
