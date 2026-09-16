"""Read-only evidence of the existing WeMM service; output goes to new run."""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

root = Path('/home/realmagic/wemm-embedding-lab')
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
s = requests.Session()
s.trust_env = False
s.post('http://127.0.0.1:8765/api/login', json={'password': (root/'run/access.key').read_text().strip()}, timeout=10).raise_for_status()
r = s.get('http://127.0.0.1:8765/api/status', timeout=10)
r.raise_for_status()
status = r.json()
conn = sqlite3.connect(f'file:{root}/data/lab.sqlite?mode=ro', uri=True)
conn.row_factory = sqlite3.Row
assets = [dict(x) for x in conn.execute('SELECT id,status,error FROM assets ORDER BY id')]
units = [dict(x) for x in conn.execute('SELECT id,asset,vector,fingerprint FROM units ORDER BY id')]
conn.close()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
vectors = {x['vector']: digest(root/x['vector']) for x in units}
reports = {str(p.relative_to(root)): digest(p) for p in sorted((root/'reports').rglob('*')) if p.is_file()}
sources = {n: digest(root/n) for n in ['backend/app.py','backend/encoder.py','backend/worker.py','backend/preprocess.py','config.yaml']}
record = {'observed_at': datetime.now(timezone.utc).isoformat(), 'status': status, 'assets': assets, 'units': units, 'vector_hashes': vectors, 'report_hashes': reports, 'source_hashes': sources}
out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps({'output':str(out),'state':status['state'],'fingerprint':status['fingerprint'],'assets':len(assets),'units':len(units),'vectors':len(vectors),'reports':len(reports)},ensure_ascii=False))
