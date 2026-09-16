"""Fixed independent PhotoBench server entrypoint; does not manage old services."""
import runpy
import sys
from pathlib import Path

from eval_config import configure

RELEASE = Path(__file__).resolve().parents[1]
configure(RELEASE)
sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
sys.argv = ['camroll_eval', 'serve', '--host', '0.0.0.0', '--port', '8772']
runpy.run_module('camroll_eval', run_name='__main__', alter_sys=True)
