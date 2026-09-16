"""Guarded API-only update; never edits model/worker/preprocess/config files."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

project = Path('/home/realmagic/camroll-wemm')
root = Path('/home/realmagic/wemm-embedding-lab')
run = project/'reports'/sys.argv[1]
before = json.loads((run/'wemm_before.json').read_text())
for name, expected in before['source_hashes'].items():
    if name == 'backend/app.py':
        continue  # The API may already contain our precisely known first patch.
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Source changed since preflight: {name}; refusing to overwrite')
path = root/'backend/app.py'
source = path.read_text()
anchor = "app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')"
legacy = "from backend.camroll_internal import install as install_camroll_internal\ninstall_camroll_internal(app, lambda: worker, '/home/realmagic/camroll-wemm/runtime')\n"
extension = "from backend.camroll_internal import install as install_camroll_internal\ninstall_camroll_internal(app, lambda: worker, '/home/realmagic/camroll-wemm/runtime', session_validator=session_valid)\n"
original = source.replace(extension, '').replace(legacy, '')
if source.count(anchor) != 1 or hashlib.sha256(original.encode()).hexdigest() != before['source_hashes']['backend/app.py']:
    raise SystemExit('Unexpected API mount layout')
backup = run/'wemm_app.py.before'
if not backup.exists():
    shutil.copy2(path, backup)
elif hashlib.sha256(backup.read_bytes()).hexdigest() != before['source_hashes']['backend/app.py']:
    raise SystemExit('Original API backup differs; refusing overwrite')
module = root/'backend/camroll_internal.py'
if module.exists():
    digest = hashlib.sha256(module.read_bytes()).hexdigest()[:12]
    archived = run/('wemm_internal.' + digest + '.before.py')
    if not archived.exists():
        shutil.copy2(module, archived)
shutil.copy2(project/'deployment/wemm_internal.py', root/'backend/camroll_internal.py')
updated = original.replace(anchor, extension+anchor)
compile(updated, str(path), 'exec')
path.write_text(updated)
print(json.dumps({'app_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'api_installed':True}))
