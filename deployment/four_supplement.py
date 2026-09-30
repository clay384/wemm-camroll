"""Supervise an explicitly requested, separate post-403 snapshot supplement.

This never changes the frozen four-group suite or its comparison profile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from four_config import RUNTIME, SOURCE, PYTHON, configure

RELEASE = Path(__file__).resolve().parents[1]


def validate_probe(probe):
    checks = probe.get('checks') or []
    return (probe.get('status') == 'passed' and probe.get('model') == 'qwen3.8-max'
            and probe.get('source') == 'synthetic_nonformal'
            and {c.get('kind') for c in checks} == {'agent-tool', 'image', 'structured'}
            and all(c.get('passed') is True for c in checks)
            and all(c.get('actual_model') == 'qwen3.8-max' for c in checks if c.get('kind') != 'structured')
            and bool(probe.get('finished_at')))


def validate_supplement(run):
    rows = run.get('items') or []
    plans = run.get('qa_plan') or []
    plan_by_id = {p.get('qa_id'): p.get('planned_turn_count') for p in plans}
    return (run.get('mode') == 'supplement' and run.get('status') in ('completed', 'completed_with_errors')
            and len(plans) == 64 and len(plan_by_id) == 64 and len(rows) == 64
            and all(type(n) is int and n > 0 for n in plan_by_id.values())
            and len({r['qa_id'] for r in rows}) == 64
            and {r['qa_id'] for r in rows} == {r['qa_id'] for r in plans}
            and sum(p['planned_turn_count'] for p in plans) == 71
            and sum(len(r.get('runtime_turns') or []) for r in rows) == 71
            and all(len(r.get('runtime_turns') or []) == plan_by_id[r['qa_id']]
                    and r.get('judge_status') == 'completed' and type((r.get('judge') or {}).get('score')) is int
                    and r['judge']['score'] in (0, 1, 2)
                    and all(type((t.get('judge') or {}).get('score')) is int and t['judge']['score'] in (0, 1, 2)
                            and type((t.get('task_judge') or {}).get('correct')) is bool
                            and (t.get('task_judge') or {}).get('actual_action') in ('answer', 'refuse', 'clarify', 'none')
                            for t in r.get('runtime_turns') or []) for r in rows))



def stop_owned_child(child):
    """Only the Popen child created by this invocation, including registration failure.

    Popen owns/reaps this direct child; no historical PID or process-name lookup
    is used. A bounded termination also leaves uncertain request reservations.
    """
    if child is None or child.poll() is not None:
        return {"state": "absent_or_exited"}
    child.terminate()
    try:
        child.wait(timeout=15)
        return {"state": "terminated", "returncode": child.returncode}
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
        return {"state": "killed_after_timeout", "returncode": child.returncode}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-id', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args(argv)
    configure(RELEASE)
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
    from camroll_agent.budget import BudgetLedger
    from camroll_eval.common import atomic_json, now_iso, read_json, safe_id, sha256
    from camroll_eval.supplement import SupplementRunner
    parent_id, run_id = safe_id(args.parent_id), safe_id(args.run_id)
    freeze_path = RUNTIME / 'freeze.json'
    root = RUNTIME / 'evaluation'
    if args.child:
        result = SupplementRunner(root, SOURCE, RUNTIME / 'albums').run_supplement(
            parent_id, run_id=run_id, freeze_path=freeze_path)
        passed = validate_supplement(result)
        print(json.dumps({'run_id': run_id, 'status': result['status'], 'complete': passed}), flush=True)
        return 0 if passed else 2
    probe_path = args.probe.resolve()
    if probe_path.parent != RUNTIME / 'supplements' or not validate_probe(read_json(probe_path)):
        raise RuntimeError('a_fresh_recorded_successful_recharge_probe_is_required')
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600)
    ledger.check_limit()
    before = ledger.summary()
    if before['reserved_cny'] or before['uncertain_requests']:
        raise RuntimeError('pending_requests_require_reconciliation')
    if (root / 'runs' / run_id).exists():
        raise RuntimeError('supplement_run_exists_no_automatic_repeat')
    directory = RUNTIME / 'supplements' / run_id
    directory.mkdir(mode=0o700)  # Exclusive claim survives any failed preflight.
    state_path = directory / 'supervisor.json'
    first_suite = RUNTIME / 'formal-suite.json'
    first_suite_hash = sha256(first_suite)
    original = read_json(first_suite)
    if original.get('restoration', {}).get('state') != 'restored':
        raise RuntimeError('first_suite_restoration_is_not_complete')
    state = {'schema_version': 'camroll-post-403-supplement-supervisor-v1', 'run_id': run_id,
             'parent_run_id': parent_id, 'window_id': run_id, 'status': 'preparing',
             'started_at': now_iso(), 'source_revision': os.environ['CAMROLL_RELEASE_REVISION'],
             'adapter_sha256': sha256(Path(__file__)), 'first_suite_sha256': first_suite_hash,
             'freeze_sha256': sha256(freeze_path), 'recovery_probe': str(probe_path),
             'recovery_probe_sha256': sha256(probe_path), 'budget_before': before,
             'authorization': 'User explicitly selected reusing the W38Max snapshot and supplementing only the 64 HTTP403 questions, separately reported, after confirming account top-up.',
             'expected_qa_count': 64, 'expected_turn_count': 71, 'expected_eligible_qa_count': 52,
             'first_results_replaced': False, 'build_performed': False}
    def save():
        state['updated_at'] = now_iso()
        atomic_json(state_path, state)
    save()
    from four_gpu_window import GpuWindow
    window = GpuWindow(RELEASE, run_id)
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('supplement_signal_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        entered = window.enter()
        freeze = read_json(freeze_path)
        if (entered['before']['fingerprint'] != freeze['wemm_fingerprint']
                or entered['model_integrity']['manifest_sha256'] != freeze['model_manifest_sha256']
                or entered['model_integrity']['models'] != freeze['model_files']):
            raise RuntimeError('models_or_fingerprint_changed_since_first_suite')
        window.switch('wemm')
        state['status'] = 'running'; save()
        command = [str(PYTHON), str(Path(__file__).resolve()), '--child', '--parent-id', parent_id,
                   '--run-id', run_id, '--probe', str(probe_path)]
        with (directory / 'evaluator.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            window.register_child(child.pid, expected_command=command)
            while child.poll() is None:
                time.sleep(5)
                ledger.check_limit()
            state['returncode'] = child.returncode
            window.unregister_child(child.pid)
            child = None
        run = read_json(root / 'runs' / run_id / 'run.json')
        state.update(status='passed' if validate_supplement(run) else 'incomplete', run_status=run['status'],
                     acceptance_complete=validate_supplement(run))
    except BaseException as exc:
        state.update(status='blocked' if isinstance(exc, Exception) else 'interrupted', error_type=type(exc).__name__)
        raise
    finally:
        try:
            try:
                state['child_cleanup'] = stop_owned_child(child)
            finally:
                state['restoration'] = window.restore('supplement_' + state['status']) if window.state_path.exists() else {'state': 'not_entered'}
            state['first_suite_unchanged'] = sha256(first_suite) == first_suite_hash
            if not state['first_suite_unchanged']:
                state['status'] = 'failed_integrity'
        except BaseException as exc:
            state.update(status='restore_failed', restoration={'state': 'failed', 'error_type': type(exc).__name__})
            raise
        finally:
            state.update(finished_at=now_iso(), budget_after=ledger.summary())
            save()
    return 0 if state['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
