"""Execute a module/script with protected runtime config, never print secrets."""
import json
import os
import runpy
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.environ.update(json.loads((root/'runtime/secrets.json').read_text()))
target=sys.argv[1]
sys.argv=sys.argv[1:]
if target.endswith('.py'):
    runpy.run_path(target,run_name='__main__')
else:
    runpy.run_module(target,run_name='__main__',alter_sys=True)
