"""Restart only the verified WeMM process bound to 8765, preserving config."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

root = Path('/home/realmagic/wemm-embedding-lab')
project = Path('/home/realmagic/camroll-wemm')
run = project/'reports'/sys.argv[1]
session = requests.Session(); session.trust_env = False
session.post('http://127.0.0.1:8765/api/login',json={'password':(root/'run/access.key').read_text().strip()},timeout=10).raise_for_status()
status = session.get('http://127.0.0.1:8765/api/status',timeout=10).json()
if status.get('queue') or status.get('busy_asset') or status.get('state') != 'ready':
    raise SystemExit('WeMM is busy; defer restart')
matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if (p/'cwd').resolve()==root and b'backend.app:app' in args and b'8765' in args:
            matches.append(int(p.name))
    except (OSError, PermissionError): pass
if len(matches)!=1:
    raise SystemExit(f'Expected one owned WeMM process, found {matches}')
pid=matches[0]
os.kill(pid,signal.SIGTERM)
for _ in range(120):
    if not Path(f'/proc/{pid}/cmdline').exists(): break
    time.sleep(.25)
else:
    raise SystemExit('WeMM did not stop gracefully; refusing second model')
env=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (run/'wemm_service.log').open('ab') as log:
    proc=subprocess.Popen([str(root/'.venv/bin/python'),'-m','uvicorn','backend.app:app','--host','0.0.0.0','--port','8765','--workers','1'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'run/service.json').write_text(json.dumps({'pid':proc.pid,'cwd':str(root),'created':time.time()}))
print(json.dumps({'old_pid':pid,'new_pid':proc.pid}))
