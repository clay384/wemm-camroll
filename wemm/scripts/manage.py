#!/usr/bin/env python3
import os,sys,json,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];run=R/'run';run.mkdir(exist_ok=True);pidfile=run/'service.json'
def owned():
 try:
  rec=json.loads(pidfile.read_text());p=Path('/proc')/str(rec['pid'])
  args=(p/'cmdline').read_bytes()
  return rec['pid'] if (p/'cwd').resolve()==R and b'backend.app:app' in args and str(R/'.venv/bin/python').encode() in args else None
 except (OSError,ValueError,KeyError):return None
cmd=sys.argv[1];pid=owned()
if cmd in ['stop','restart'] and pid:
 os.kill(pid,signal.SIGTERM)
 for _ in range(120):
  if not owned():break
  time.sleep(.5)
 if owned():raise SystemExit('Task process still exiting; not starting a second worker')
 print('Stopped task-owned process',pid);pid=None
if cmd in ['start','restart']:
 if pid:print('Already running',pid);sys.exit(0)
 if not (R/'SOURCE_LOCK.json').exists():raise SystemExit('BLOCKED: verify SOURCE_LOCK first')
 import yaml,socket
 config=yaml.safe_load((R/'config.yaml').read_text());selected=None
 for port in [config['port']]+[p for p in range(8766,8786) if p!=config['port']]:
  s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind((config['host'],port));selected=port;break
  except OSError:pass
  finally:s.close()
 if selected is None:raise SystemExit('BLOCKED: no free port in the authorized range')
 if selected!=config['port']:
  config['port']=selected;(R/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False));print('Selected free project port',selected)
 env=os.environ.copy()
 for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
 env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(R/'.hf'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(config['cpu_threads']),OPENBLAS_NUM_THREADS='1')
 with (R/'logs/service.log').open('ab') as log:
  p=subprocess.Popen([str(R/'.venv/bin/python'),'-m','uvicorn','backend.app:app','--host',config['host'],'--port',str(config['port']),'--workers','1'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 pidfile.write_text(json.dumps({'pid':p.pid,'cwd':str(R),'created':time.time()}));print('Started',p.pid)
elif cmd=='status':print(json.dumps({'pid':pid,'running':bool(pid),'root':str(R)}))
elif cmd=='logs':subprocess.run(['tail','-n','60',str(R/'logs/service.log')])
elif cmd!='stop':raise SystemExit('start | stop | restart | status | logs')
