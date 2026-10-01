"""One-shot, supervised Flash Gate suite over the approved frozen WeMM index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from gate_config import (ALBUMS, HISTORY_RUNTIME, MODELS, ORDER, PYTHON, RUNTIME,
                         SNAPSHOT, SOURCE, SOURCE_ALBUM_ID, SOURCE_RUN_ID, configure)

RELEASE = Path(__file__).resolve().parents[1]


def bootstrap():
    configure(RELEASE)
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))


def validate_run(state):
    """Wrong answers/Gate exhaustion are measurements, missing runs are not."""
    plans = {row['qa_id']: row['planned_turn_count'] for row in state.get('qa_plan', [])}
    items = state.get('items', [])
    if (state.get('mode') != 'gate_ab' or state.get('status') not in ('completed', 'completed_with_errors')
            or state.get('gate_complete') is not True or state.get('formal_protocol') is not True
            or state.get('finalization_errors')
            or state.get('planned_eligible_qa_count') != 81
            or len(plans) != 100 or sum(plans.values()) != 111 or len(items) != 100
            or {row['qa_id'] for row in items} != set(plans)
            or state.get('source_unchanged') is not True):
        return False
    for item in items:
        turns = item.get('runtime_turns', [])
        if len(turns) != plans[item['qa_id']]:
            return False
        for turn in turns:
            origin = turn.get('answer_origin')
            if turn.get('execution_status') == 'not_started' or origin not in ('model', 'system_failure', 'gate_exhausted'):
                return False
            if origin == 'model':
                if (type((turn.get('judge') or {}).get('score')) is not int
                        or turn['judge']['score'] not in (0, 1, 2)
                        or type((turn.get('task_judge') or {}).get('correct')) is not bool
                        or (turn.get('task_judge') or {}).get('actual_action') not in {'answer', 'clarify', 'refuse', 'none'}):
                    return False
            elif turn.get('judge_status') != 'skipped_expected' or (turn.get('judge') or {}).get('score') is not None:
                return False
    return True


def ledger_ready(ledger):
    ledger.check_limit()
    value = ledger.summary()
    if value['reserved_cny'] or value['uncertain_requests']:
        raise RuntimeError('pending_requests_require_reconciliation')
    if value['remaining_cny'] <= 0:
        raise RuntimeError('budget_exhausted')
    return value


def freeze_inputs(destination):
    from camroll_eval.common import atomic_json, read_json, now_iso, sha256
    from camroll_eval.dataset import Dataset
    from camroll_eval.judge import Judge
    from four_suite import source_manifest
    from four_gpu_window import verify_files, verify_model_files, _legacy_fingerprint
    source_files = source_manifest(RELEASE)
    checks = {name: read_json(RUNTIME / name) for name in ('validation.json', 'cloud-smoke.json')}
    if any(row.get('status') != 'passed' or row.get('source_files') != source_files for row in checks.values()):
        raise RuntimeError('matching_source_validation_and_cloud_smoke_required')
    if not RELEASE.name.startswith('gate-'):
        raise RuntimeError('formal_runs_require_frozen_gate_release')
    original = HISTORY_RUNTIME / 'evaluation/runs' / SOURCE_RUN_ID / 'run.json'
    parent = read_json(original)
    if parent.get('album_id') != SOURCE_ALBUM_ID or parent.get('index_version') != SNAPSHOT:
        raise RuntimeError('unexpected_source_album_or_snapshot')
    integrity = Dataset(SOURCE).integrity()
    if integrity != parent['input_integrity']:
        raise RuntimeError('frozen_inputs_changed')
    # This is a file-only manifest; paid clients are not constructed here.
    album_manifest = source_manifest(ALBUMS / SOURCE_ALBUM_ID)
    result = {'schema_version': 'camroll-flash-gate-freeze-v1', 'created_at': now_iso(),
              'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'source_files': source_files,
              'source_run_id': SOURCE_RUN_ID, 'source_run_sha256': sha256(original),
              'source_album_id': SOURCE_ALBUM_ID, 'snapshot': SNAPSHOT,
              'album_files': album_manifest, 'input_integrity': integrity,
              'protected_files': verify_files(), 'model_files': verify_model_files(),
              'wemm_fingerprint': _legacy_fingerprint(), 'order': list(ORDER), 'models': MODELS,
              'judge_protocol': Judge(expected_limit_cny=600).protocol(), 'acceptance': checks,
              'budget_limit_cny': 600, 'max_turn_seconds': 600, 'qa_concurrency': 1, 'judge_concurrency': 4,
              'temperature': 0, 'enable_thinking': False, 'config': 'M4',
              'qa_count': 100, 'planned_turn_count': 111, 'retrieval_denominator': 81,
              'gate_max_generations': 4, 'gate_evidence_max_images': 6,
              'gate_judge_same_model': True, 'seen_benchmark': True, 'build_performed': False}
    if destination.exists():
        raise RuntimeError('freeze_already_exists_no_automatic_repeat')
    atomic_json(destination, result)
    return result


def verify_frozen(freeze):
    from camroll_eval.common import sha256
    from camroll_eval.dataset import Dataset
    from four_suite import source_manifest
    if source_manifest(RELEASE) != freeze['source_files']:
        raise RuntimeError('frozen_source_changed')
    if source_manifest(ALBUMS / SOURCE_ALBUM_ID) != freeze['album_files']:
        raise RuntimeError('frozen_album_changed')
    if Dataset(SOURCE).integrity() != freeze['input_integrity']:
        raise RuntimeError('frozen_inputs_changed')
    if sha256(HISTORY_RUNTIME / 'evaluation/runs' / SOURCE_RUN_ID / 'run.json') != freeze['source_run_sha256']:
        raise RuntimeError('source_run_changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--child-group', choices=ORDER)
    parser.add_argument('--run-id')
    parser.add_argument('--suite-id')
    args = parser.parse_args(argv)
    bootstrap()
    from camroll_agent.budget import BudgetLedger, BudgetLimitChanged
    from camroll_eval.common import atomic_json, now_iso, read_json
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600)
    if not args.execute_paid:
        print(json.dumps({'status': 'not_started', 'budget': ledger.summary(), 'order': ORDER}))
        return 0
    if args.child_group:
        from camroll_eval.gate_experiment import GateRunner
        frozen = read_json(RUNTIME / 'freeze.json')
        suite = read_json(RUNTIME / 'formal-suite.json')
        matches = [row for row in suite.get('groups', []) if row.get('name') == args.child_group]
        if (suite.get('suite_id') != args.suite_id or suite.get('status') != 'running'
                or len(matches) != 1 or matches[0].get('run_id') != args.run_id
                or matches[0].get('status') != 'running'):
            raise RuntimeError('child_identity_does_not_match_frozen_suite')
        from four_gpu_window import GpuWindow
        window_state = GpuWindow(RELEASE, args.suite_id, runtime=HISTORY_RUNTIME)._read()
        if window_state.get('state') != 'active' or window_state.get('current_kind') != 'wemm':
            raise RuntimeError('child_requires_active_supervised_wemm_window')
        verify_frozen(frozen)
        result = GateRunner(RUNTIME / 'evaluation', SOURCE, ALBUMS).run_gate(
            run_id=args.run_id, suite_id=args.suite_id, group=args.child_group,
            source_album_id=SOURCE_ALBUM_ID, source_run_id=SOURCE_RUN_ID,
            source_evaluation_root=HISTORY_RUNTIME / 'evaluation', expected_snapshot=SNAPSHOT)
        complete = validate_run(result)
        print(json.dumps({'run_id': args.run_id, 'status': result['status'], 'complete': complete}), flush=True)
        return 0 if complete else 2
    from four_gpu_window import GpuWindow
    from four_supplement import stop_owned_child
    suite_id = 'gate-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    suite_path = RUNTIME / 'formal-suite.json'
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Exclusive claim prevents accidental repeat even if preflight is interrupted.
    (RUNTIME / 'formal-suite.claim').mkdir(mode=0o700)
    before = ledger.summary()
    groups = [{'name': name, 'run_id': name.lower() + '-' + suite_id, 'status': 'not_started',
               'qa_count': 100, 'planned_turn_count': 111} for name in ORDER]
    state = {'schema_version': 'camroll-flash-gate-suite-v1', 'suite_id': suite_id, 'status': 'preparing',
             'started_at': now_iso(), 'groups': groups, 'budget_before': before,
             'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'build_performed': False}
    def save():
        state['updated_at'] = now_iso()
        atomic_json(suite_path, state)
        atomic_json(RUNTIME / 'evaluation/comparison-profile.json', {
            'schema_version': 'camroll-flash-gate-v1', 'title': 'Qwen Flash · Gate 四组对照',
            'groups': groups, 'suite_status': state['status'], 'updated_at': state['updated_at'],
            'snapshot': SNAPSHOT, 'note': '共享冻结 WeMM 索引，不重新建库。固定100题、111轮；交付P/R/F1固定81题。'
                'Doubao同时用于事实Gate与独立最终评分，存在模型关联；通过Gate不保证正确。album3为已见基准。',
            'historical': [{'label': '原四组首跑（历史记录）', 'run_id': SOURCE_RUN_ID}]})
    save()
    # Share the existing GPU ownership registry with all earlier supervisors.
    # The window ID is new; a separate lock directory would allow concurrent owners.
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)
    active, freeze = None, None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('suite_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        from camroll_eval.dataset import Dataset
        dataset = Dataset(SOURCE)
        plans = [{'qa_id': row['qa_id'], 'question': row['question'],
                  'planned_turn_count': len(dataset.turns(row)), 'execution_status': 'not_started',
                  'judge_status': 'not_started'} for row in dataset.rows]
        atomic_json(RUNTIME / 'evaluation/planned-groups.json', {
            'suite_id': suite_id, 'groups': [{**group, 'qa_plan': plans} for group in groups]})
        ledger_ready(ledger)
        freeze = freeze_inputs(RUNTIME / 'freeze.json')
        entered = window.enter()
        state['window_state_path'] = str(window.state_path)
        if entered['before']['fingerprint'] != freeze['wemm_fingerprint']:
            raise RuntimeError('window_fingerprint_differs_from_freeze')
        window.switch('wemm')
        state['status'] = 'running'
        save()
        for group in groups:
            ledger_ready(ledger)
            verify_frozen(freeze)
            group.update(status='running', started_at=now_iso())
            save()
            log_path = RUNTIME / 'logs' / (group['run_id'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            command = [str(PYTHON), str(Path(__file__).resolve()), '--execute-paid', '--child-group', group['name'],
                       '--run-id', group['run_id'], '--suite-id', suite_id]
            with log_path.open('xb') as log:
                active = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                window.register_child(active.pid, expected_command=command)
                stopped_submissions_at = None
                while active.poll() is None:
                    time.sleep(5)
                    reason = None
                    try:
                        ledger.check_limit()
                    except BudgetLimitChanged:
                        reason = 'budget_limit_changed'
                    if ledger.summary()['uncertain_requests']:
                        reason = 'uncertain_cloud_request_stop'
                    if reason:
                        # The ledger itself rejects all new reservations. Ask
                        # the runner to stop, allowing already-sent requests to
                        # settle rather than creating avoidable unknown charges.
                        marker = RUNTIME / 'evaluation/runs' / group['run_id'] / 'cancel.requested'
                        if marker.parent.exists() and not marker.exists():
                            marker.write_text(reason)
                            marker.chmod(0o600)
                        if stopped_submissions_at is None:
                            stopped_submissions_at = time.monotonic()
                            group['new_submissions_blocked'] = reason
                            save()
                    if stopped_submissions_at is not None and time.monotonic() - stopped_submissions_at >= 600:
                        raise RuntimeError('blocked_child_did_not_exit_within_turn_deadline')
                group['returncode'] = active.returncode
                window.unregister_child(active.pid)
                active = None
            path = RUNTIME / 'evaluation/runs' / group['run_id'] / 'run.json'
            run = read_json(path) if path.exists() else {}
            group.update(status=run.get('status', 'failed'), finished_at=now_iso(),
                         acceptance_complete=group['returncode'] == 0 and validate_run(run), summary=run.get('summary'))
            verify_frozen(freeze)
            save()
            if run.get('status') not in ('completed', 'completed_with_errors'):
                raise RuntimeError('group_blocked_' + group['name'])
        state['status'] = 'passed' if all(row['acceptance_complete'] for row in groups) else 'incomplete'
    except BaseException as exc:
        state.update(status='blocked' if isinstance(exc, Exception) else 'interrupted',
                     error_type=type(exc).__name__, error_code=getattr(exc, 'code', None),
                     error=str(exc) if isinstance(exc, (RuntimeError, ValueError, KeyboardInterrupt)) else type(exc).__name__)
        for group in groups:
            if group['status'] == 'running':
                group.update(status=state['status'], finished_at=now_iso())
    finally:
        try:
            try:
                state['child_cleanup'] = stop_owned_child(active)
            except BaseException as exc:
                state['child_cleanup'] = {'state': 'failed', 'error_type': type(exc).__name__}
            finally:
                # Even failure to terminate/reap the direct child must enter
                # identity-checked restoration of all registered processes.
                state['restoration'] = window.restore('gate_suite_' + state['status']) if window.state_path.exists() else {'state': 'not_entered'}
            if freeze:
                from four_gpu_window import verify_model_files
                models = verify_model_files()
                if models['models'] != freeze['model_files']['models']:
                    raise RuntimeError('protected_model_files_changed')
                state['model_files_unchanged'] = True
                verify_frozen(freeze)
                state['frozen_inputs_unchanged'] = True
        except BaseException as exc:
            state.update(status='restore_failed', restoration_error=type(exc).__name__)
        try:
            state['budget_after'] = ledger.summary()
        except Exception as exc:
            state.update(status='incomplete' if state['status'] == 'passed' else state['status'],
                         budget_after=None, budget_audit_error=type(exc).__name__)
        state['finished_at'] = now_iso()
        save()
    return 0 if state['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
