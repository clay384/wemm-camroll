"""Record local/remote offline acceptance for the exact release files."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from gate_config import RUNTIME
from four_suite import source_manifest

RELEASE = Path(__file__).resolve().parents[1]


def main():
    before = source_manifest(RELEASE)
    output = Path(os.getenv('CAMROLL_GATE_VALIDATION_ROOT', str(RUNTIME)))
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    command = [sys.executable, '-m', 'pytest', '-q', 'camroll/camroll-agent/tests', 'camroll/tests', 'deployment']
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(RELEASE / 'camroll/camroll-agent'), str(RELEASE), str(RELEASE / 'deployment')]),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    started = time.time()
    result = subprocess.run(command, cwd=RELEASE, env=env, capture_output=True, text=True)
    log = output / 'validation.log'
    log.write_text(result.stdout + result.stderr)
    log.chmod(0o600)
    after = source_manifest(RELEASE)
    report = {'status': 'passed' if result.returncode == 0 and before == after else 'failed',
              'source_files': after, 'source_unchanged_during_tests': before == after,
              'command': command, 'returncode': result.returncode, 'started_at_epoch': started,
              'finished_at_epoch': time.time(), 'platform': sys.platform, 'python': sys.version,
              'log': str(log), 'paid_calls': False}
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
    from camroll_eval.common import atomic_json
    atomic_json(output / 'validation.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'returncode', 'log')}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
