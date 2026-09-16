"""Execute the independent CLI with shared budget and isolated runtime paths."""
import runpy
import sys
from pathlib import Path

from eval_config import configure

RELEASE = Path(__file__).resolve().parents[1]
configure(RELEASE)
sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
if len(sys.argv) < 2:
    raise SystemExit('Specify a Python module or script')
target = sys.argv[1]
sys.argv = sys.argv[1:]
if target.endswith('.py'):
    runpy.run_path(target, run_name='__main__')
else:
    runpy.run_module(target, run_name='__main__', alter_sys=True)
