"""One-shot execution of the never-started F38-On group; prior runs stay read-only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

import gate_continuation as continuation
from gate_config import ALBUMS, BASE, HISTORY_RUNTIME, ORDER, PYTHON, SNAPSHOT, SOURCE, SOURCE_ALBUM_ID, SOURCE_RUN_ID
from gate_suite import validate_run

RELEASE = Path(__file__).resolve().parents[1]
RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007h'
PARENT_RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007g'
PARENT_SUITE = 'gate-continuation-20261007-102950-966c02'
PARENT_FREEZE_SHA256 = '3432907b7b82964a261fbd9793aeb8472526026f0fd02d0b74a9e9ea204d0162'
PARENT_SUITE_SHA256 = '9f3410780d22fb812916e097aea02da449efbc6f313a408ec35d382943e91d1a'
PARENT_RUN_SHA256 = {
    'F37-Off': 'd1952f607d1dff6487971a800efe5f16de777da097d543dd1dadabf8d77f6e00',
    'F37-On': '6365deb09cb6fbcb61f7bba725cd8e2f2b67314d62a512cf8fef255ae8630a62',
    'F38-Off': '09eebf3b23723b4e54647a2d272cc573354b7bac91f13c52a1b38e4d1289ad38'}
PARENT_RELEASE = BASE / 'releases/gate-b433657'
GROUP = 'F38-On'
MIN_FREE_BYTES = 2 * 1024 ** 3
# Explicit report-only changes authorized for h; experiment/runner/model code
# and all other files that already existed in b433657 remain byte-identical.
REPORT_ONLY_CHANGES = frozenset({'camroll/camroll-agent/camroll_eval/repository.py',
                               'camroll/camroll-agent/camroll_eval/static/app.js'})


def bootstrap(runtime):
    from gate_config import configure
    configure(RELEASE)
    os.environ['CAMROLL_EVAL_ROOT'] = str(Path(runtime) / 'evaluation')
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))


def no_links(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('remaining_paths_must_not_use_symlinks')
    return path


def independent_runtime(runtime, parent):
    # Check the original path before resolve so links cannot hide its boundary.
    no_links(runtime)
    no_links(parent)
    runtime, parent = Path(runtime).resolve(), Path(parent).resolve()
    if (runtime == parent or runtime.is_relative_to(parent) or parent.is_relative_to(runtime)
            or runtime == RELEASE or runtime.is_relative_to(RELEASE)):
        raise ValueError('remaining_requires_independent_runtime')
    return runtime, parent


def disk_ready(runtime):
    path = Path(runtime)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError('remaining_requires_2GiB_free_disk')
    return {'free_bytes': free, 'minimum_free_bytes': MIN_FREE_BYTES}


def _run_files(directory):
    from four_suite import source_manifest
    # A read-only SQLite reader may recreate SHM coordination state; committed
    # database, WAL, raw attempts and all measurement files remain exact.
    directory = Path(directory)
    if any(path.is_symlink() for path in directory.rglob('*')):
        raise ValueError('parent_run_symlink')
    return {name: value for name, value in source_manifest(directory).items() if not name.endswith('-shm')}


def _journal_terminal(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('parent_journal_missing_or_linked')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT request_id,payload_hash,call_id,status FROM cloud_requests').fetchall()
    if not rows or any(row[3] not in ('completed', 'failed') for row in rows):
        raise RuntimeError('parent_has_new_pending_or_unknown_requests')
    if len({row[2] for row in rows}) != len(rows):
        raise RuntimeError('parent_journal_duplicate_call_identity')
    return len(rows)


def checked_parent(parent_runtime, acknowledgement):
    """No paid clients: require the exact restored g experiment and fixed cause."""
    from camroll_eval.common import read_json, sha256
    from camroll_eval.dataset import Dataset
    from camroll_eval.gate_experiment import infrastructure_failure
    from four_suite import source_manifest
    parent = no_links(parent_runtime).resolve()
    if parent != Path(PARENT_RUNTIME).resolve():
        raise RuntimeError('fixed_g_parent_runtime_required')
    freeze_path, suite_path = parent / 'freeze.json', parent / 'formal-suite.json'
    no_links(freeze_path)
    no_links(suite_path)
    if sha256(freeze_path) != PARENT_FREEZE_SHA256:
        raise RuntimeError('parent_g_freeze_changed')
    if sha256(suite_path) != PARENT_SUITE_SHA256:
        raise RuntimeError('parent_g_suite_changed')
    frozen, suite = read_json(freeze_path), read_json(suite_path)
    if (suite.get('suite_id') != PARENT_SUITE or suite.get('status') != 'blocked'
            or suite.get('error') != 'group_blocked_F38-Off' or not suite.get('finished_at')
            or suite.get('restoration', {}).get('state') != 'restored'
            or suite.get('restoration', {}).get('no_experiment_gpu_residual') is not True
            or any(suite.get(key) is not True for key in
                   ('model_files_unchanged', 'wemm_weights_unchanged', 'frozen_inputs_unchanged'))):
        raise RuntimeError('restored_context_blocked_g_parent_required')
    if tuple(row['name'] for row in suite['groups']) != ORDER:
        raise RuntimeError('parent_group_plan_changed')
    ack = acknowledgement.as_dict()
    if ack != frozen['acknowledgement'] or ack != suite['acknowledgement']:
        raise RuntimeError('only_original_acknowledgement_scope_is_allowed')
    completion = read_json(parent / 'completion-audit.json')
    if (completion.get('suite_id') != PARENT_SUITE or completion.get('suite_status') != 'blocked'
            or completion.get('export_returncode') != 0
            or completion.get('report_sha256') != sha256(parent / 'reports/flash-gate.json')
            or completion.get('ledger_snapshot_sha256') != sha256(parent / 'budget-final-snapshot.sqlite')):
        raise RuntimeError('parent_finalizer_not_complete')
    source = source_manifest(RELEASE)
    if source_manifest(PARENT_RELEASE) != frozen['source_files']:
        raise RuntimeError('parent_g_source_changed')
    if any(source.get(name) != digest for name, digest in frozen['source_files'].items()
           if name not in REPORT_ONLY_CHANGES):
        raise RuntimeError('existing_frozen_source_must_be_byte_identical')
    smoke = read_json(parent / 'cloud-smoke.json')
    if (smoke.get('status') != 'passed' or smoke.get('restoration', {}).get('state') != 'restored'
            or smoke.get('source_files') != frozen['source_files']
            or continuation.protocol_manifest(RELEASE) != frozen['protocol_files']):
        raise RuntimeError('unchanged_passed_parent_smoke_required')
    refs, files, counts = {}, {}, {}
    for group in suite['groups'][:3]:
        directory = parent / 'evaluation/runs' / group['run_id']
        path = directory / 'run.json'
        no_links(path)
        if sha256(path) != PARENT_RUN_SHA256[group['name']]:
            raise RuntimeError('parent_g_run_changed')
        run = read_json(path)
        if run.get('suite_id') != PARENT_SUITE or run.get('inference_config', {}).get('group') != group['name']:
            raise RuntimeError('parent_run_identity_changed')
        if group['name'] != 'F38-Off':
            if not validate_run(run) or group.get('acceptance_complete') is not True:
                raise RuntimeError('completed_flash37_parents_required')
        else:
            turns = [turn for item in run.get('items', []) for turn in item.get('runtime_turns', [])]
            if (run.get('status') != 'blocked' or run.get('stop_reason') != 'context_limit'
                    or run.get('source_unchanged') is not True or run.get('finalization_errors')
                    or len(run.get('items', [])) != 1 or len(turns) != 1 or turns[0].get('index') != 0
                    or turns[0].get('answer_origin') != 'system_failure'
                    or turns[0].get('judge_status') != 'skipped_expected'):
                raise RuntimeError('only_first_turn_local_context_limit_parent_is_allowed')
            relative = Path(turns[0]['raw_attempt_path'])
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise RuntimeError('parent_raw_attempt_path_invalid')
            raw_path = no_links(directory / relative).resolve()
            if not raw_path.is_relative_to(directory.resolve()):
                raise RuntimeError('parent_raw_attempt_path_invalid')
            if infrastructure_failure(read_json(raw_path).get('response')) != 'context_limit':
                raise RuntimeError('parent_context_limit_structured_evidence_missing')
        if group['name'] != 'F37-Off':
            counts[group['name']] = _journal_terminal(directory / 'request-journal.sqlite')
        refs[group['name']] = {'path': str(path), 'sha256': sha256(path)}
        files[group['name']] = _run_files(directory)
    pending = suite['groups'][3]
    if (pending['status'] != 'not_started'
            or (parent / 'evaluation/runs' / pending['run_id']).exists()
            or (parent / 'evaluation/gate_claims' / PARENT_SUITE / (GROUP + '.json')).exists()):
        raise RuntimeError('F38_On_must_never_have_started')
    # Verify g's full immutable ancestry and the original source/album inputs.
    ancestor = read_json(frozen['parent_freeze']['path'])
    paths = [frozen['parent_freeze'], frozen['parent_suite'], *frozen['parent_runs'].values()]
    if any(sha256(row['path']) != row['sha256'] for row in paths):
        raise RuntimeError('original_parent_records_changed')
    if source_manifest(Path(frozen['parent_release'])) != ancestor['source_files']:
        raise RuntimeError('original_parent_source_changed')
    if (source_manifest(ALBUMS / SOURCE_ALBUM_ID) != ancestor['album_files']
            or Dataset(SOURCE).integrity() != ancestor['input_integrity']
            or sha256(HISTORY_RUNTIME / 'evaluation/runs' / SOURCE_RUN_ID / 'run.json') != ancestor['source_run_sha256']
            or frozen['snapshot'] != SNAPSHOT):
        raise RuntimeError('original_snapshot_or_inputs_changed')
    return {'parent_suite': {'path': str(suite_path), 'sha256': sha256(suite_path)},
        'parent_freeze': {'path': str(freeze_path), 'sha256': sha256(freeze_path)},
        'parent_runs': refs, 'parent_run_files': files, 'parent_journal_counts': counts,
        'parent_smoke': {'path': str(parent / 'cloud-smoke.json'), 'sha256': sha256(parent / 'cloud-smoke.json')},
        'parent_completion': {'path': str(parent / 'completion-audit.json'), 'sha256': sha256(parent / 'completion-audit.json')},
        'parent_release': str(PARENT_RELEASE), 'protocol_files': frozen['protocol_files'],
        'wemm_fingerprint': frozen['wemm_fingerprint']}


def verify_frozen(freeze, runtime, acknowledgement):
    from camroll_eval.common import read_json
    from four_suite import source_manifest
    if source_manifest(RELEASE) != freeze['source_files']:
        raise RuntimeError('remaining_source_changed')
    if checked_parent(freeze['parent_runtime'], acknowledgement) != freeze['parent_inputs']:
        raise RuntimeError('remaining_parent_inputs_changed')
    validation = read_json(Path(runtime) / 'validation.json')
    if validation.get('status') != 'passed' or validation.get('source_files') != freeze['source_files']:
        raise RuntimeError('matching_remaining_offline_validation_required')


def freeze_inputs(runtime, parent_runtime, acknowledgement):
    from camroll_eval.common import atomic_json, now_iso
    from four_suite import source_manifest
    from four_gpu_window import verify_files, verify_model_files
    inputs = checked_parent(parent_runtime, acknowledgement)
    result = {'schema_version': 'camroll-flash-gate-remaining-freeze-v1', 'created_at': now_iso(),
        'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'source_files': source_manifest(RELEASE),
        'parent_runtime': str(parent_runtime), 'parent_inputs': inputs,
        'parent_suite': inputs['parent_suite'], 'parent_freeze': inputs['parent_freeze'],
        'parent_runs': inputs['parent_runs'], 'acknowledgement': acknowledgement.as_dict(),
        'snapshot': SNAPSHOT, 'build_performed': False, 'execution_order': [GROUP],
        'new_question_count': 100, 'new_turn_count': 111, 'reused_paid_smoke': inputs['parent_smoke'],
        'protected_files': verify_files(), 'model_files': verify_model_files(),
        'wemm_weights': continuation.verify_wemm_weights(), 'wemm_fingerprint': inputs['wemm_fingerprint']}
    verify_frozen(result, runtime, acknowledgement)
    path = Path(runtime) / 'freeze.json'
    if path.exists():
        raise RuntimeError('remaining_freeze_already_exists')
    atomic_json(path, result)
    return result


def run_child(runtime, suite_id, run_id, ledger, acknowledgement):
    from camroll_eval.common import read_json
    from camroll_eval.gate_experiment import GateRunner
    from four_gpu_window import GpuWindow
    runtime = Path(runtime)
    state, freeze = read_json(runtime / 'formal-suite.json'), read_json(runtime / 'freeze.json')
    group = state['groups'][-1]
    if (state.get('suite_id') != suite_id or state.get('status') != 'running'
            or state.get('execution_order') != [GROUP] or group.get('name') != GROUP
            or group.get('run_id') != run_id or group.get('status') != 'running'):
        raise RuntimeError('remaining_child_identity_mismatch')
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)._read()
    if window.get('state') != 'active' or window.get('current_kind') != 'wemm':
        raise RuntimeError('remaining_child_requires_supervised_wemm')
    continuation.ledger_ready(ledger)
    verify_frozen(freeze, runtime, acknowledgement)
    return GateRunner(runtime / 'evaluation', SOURCE, ALBUMS, budget_ledger=ledger).run_gate(
        run_id=run_id, suite_id=suite_id, group=GROUP, source_album_id=SOURCE_ALBUM_ID,
        source_run_id=SOURCE_RUN_ID, source_evaluation_root=HISTORY_RUNTIME / 'evaluation', expected_snapshot=SNAPSHOT)


def execute(runtime, parent_runtime, ledger, acknowledgement):
    from camroll_eval.common import atomic_json, read_json, now_iso
    from four_gpu_window import GpuWindow, verify_model_files
    from four_supplement import stop_owned_child
    runtime, parent_runtime = independent_runtime(runtime, parent_runtime)
    space = disk_ready(runtime)
    continuation.ledger_ready(ledger)
    parent_inputs = checked_parent(parent_runtime, acknowledgement)
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    (runtime / 'formal-suite.claim').mkdir(mode=0o700)
    suite_id = 'gate-remaining-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    old = read_json(parent_runtime / 'formal-suite.json')
    groups = [{**row, 'reused': True, 'parent_run_path': parent_inputs['parent_runs'][row['name']]['path'],
               'parent_run_sha256': parent_inputs['parent_runs'][row['name']]['sha256']} for row in old['groups'][:3]]
    group = {'name': GROUP, 'run_id': 'f38-on-' + suite_id, 'status': 'not_started', 'qa_count': 100, 'planned_turn_count': 111}
    groups.append(group)
    state = {'schema_version': 'camroll-flash-gate-remaining-suite-v1', 'suite_id': suite_id,
        'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'status': 'preparing', 'started_at': now_iso(),
        'execution_order': [GROUP], 'execution_scope': 'Never-started F38-On only; prior F38-Off remains blocked.',
        'groups': groups, 'parent_runtime': str(parent_runtime), 'acknowledgement': acknowledgement.as_dict(),
        'remaining_group_complete': False, 'build_performed': False, 'budget_before': ledger.summary(), 'disk_before': space}
    def save():
        state['updated_at'] = now_iso()
        atomic_json(runtime / 'formal-suite.json', state)
        atomic_json(runtime / 'evaluation/comparison-profile.json', {
            'schema_version': 'camroll-flash-gate-v1', 'title': 'Qwen Flash · Gate 剩余组独立执行',
            'groups': groups, 'suite_status': state['status'], 'comparison_status': 'incomplete',
            'execution_order': state['execution_order'],
            'remaining_group_status': state['status'], 'updated_at': state['updated_at'],
            'snapshot': SNAPSHOT, 'note': '只新增从未执行的F38-On。前三组按原批次只读引用；'
            'F38-Off首轮context_limit及其余未执行状态原样保留，完整四组对照仍未完成。'})
    save()
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)
    active = freeze = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('remaining_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        freeze = freeze_inputs(runtime, parent_runtime, acknowledgement)
        from camroll_eval.dataset import Dataset
        dataset = Dataset(SOURCE)
        plan = [{'qa_id': row['qa_id'], 'question': row['question'], 'planned_turn_count': len(dataset.turns(row)),
                 'execution_status': 'not_started', 'judge_status': 'not_started'} for row in dataset.rows]
        atomic_json(runtime / 'evaluation/planned-groups.json', {'suite_id': suite_id,
            'groups': [{**row, 'qa_plan': plan} for row in groups]})
        entered = window.enter()
        state['window_state_path'] = str(window.state_path)
        if entered['before']['fingerprint'] != freeze['wemm_fingerprint']:
            raise RuntimeError('remaining_window_fingerprint_changed')
        window.switch('wemm')
        continuation.ledger_ready(ledger)
        verify_frozen(freeze, runtime, acknowledgement)
        state['status'] = 'running'
        group.update(status='running', started_at=now_iso())
        save()
        command = [str(PYTHON), str(Path(__file__).resolve()), '--execute-paid', '--runtime', str(runtime),
                   '--child', '--suite-id', suite_id, '--run-id', group['run_id']]
        log = runtime / 'logs' / (group['run_id'] + '.log')
        log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with log.open('xb') as stream:
            active = subprocess.Popen(command, cwd=RELEASE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            window.register_child(active.pid, expected_command=command)
            stopped_at = None
            while active.poll() is None:
                time.sleep(5)
                reason = None
                try:
                    ledger.check_limit()
                    value = ledger.summary()
                    if value.get('blocking_uncertain_requests', value['uncertain_requests']):
                        reason = 'new_uncertain_cloud_request_stop'
                except Exception as exc:
                    reason = getattr(exc, 'code', None) or 'budget_guard_failed'
                if reason:
                    marker = runtime / 'evaluation/runs' / group['run_id'] / 'cancel.requested'
                    if marker.parent.exists() and not marker.exists():
                        marker.write_text(reason); marker.chmod(0o600)
                    stopped_at = stopped_at or time.monotonic()
                    group['new_submissions_blocked'] = reason
                    save()
                if stopped_at is not None and time.monotonic() - stopped_at >= 600:
                    raise RuntimeError('remaining_child_not_stopped_within_turn_deadline')
            group['returncode'] = active.returncode
            window.unregister_child(active.pid)
            active = None
        path = runtime / 'evaluation/runs' / group['run_id'] / 'run.json'
        run = read_json(path) if path.exists() else {}
        complete = group['returncode'] == 0 and validate_run(run)
        group.update(status=run.get('status', 'failed'), acceptance_complete=complete, finished_at=now_iso(), summary=run.get('summary'))
        state.update(status='passed' if complete else 'blocked', remaining_group_complete=complete)
    except BaseException as exc:
        state.update(status='blocked' if isinstance(exc, Exception) else 'interrupted',
            error_type=type(exc).__name__, error_code=getattr(exc, 'code', None),
            error=str(exc) if isinstance(exc, (RuntimeError, ValueError, KeyboardInterrupt)) else type(exc).__name__)
        if group['status'] == 'running':
            group.update(status=state['status'], finished_at=now_iso())
    finally:
        try:
            try:
                state['child_cleanup'] = stop_owned_child(active)
            except BaseException as exc:
                state['child_cleanup'] = {'state': 'failed', 'error_type': type(exc).__name__}
            finally:
                state['restoration'] = window.restore('gate_remaining_' + state['status']) if window.state_path.exists() else {'state': 'not_entered'}
            if window.state_path.exists() and (state['restoration'].get('state') != 'restored'
                    or state['restoration'].get('no_experiment_gpu_residual') is not True):
                raise RuntimeError('remaining_window_restoration_not_complete')
            if state['child_cleanup'].get('state') == 'failed':
                raise RuntimeError('remaining_child_cleanup_failed')
            if freeze:
                if verify_model_files()['models'] != freeze['model_files']['models']:
                    raise RuntimeError('remaining_model_files_changed')
                weights = continuation.verify_wemm_weights()
                state['wemm_weights_after'] = weights
                if (weights['actual_sha256'] != freeze['wemm_weights']['actual_sha256']
                        or weights['file_identity_before'] != freeze['wemm_weights']['file_identity_after']):
                    raise RuntimeError('remaining_wemm_weight_changed')
                state['model_files_unchanged'] = state['wemm_weights_unchanged'] = True
                verify_frozen(freeze, runtime, acknowledgement)
                state['frozen_inputs_unchanged'] = True
        except BaseException as exc:
            state.update(status='restore_failed', restoration_error=type(exc).__name__,
                         restoration_error_code=str(exc) if isinstance(exc, RuntimeError) else None)
        try:
            ledger.check_limit()
            state['budget_after'] = ledger.summary()
            ending = state['budget_after']
            if (ending.get('blocking_uncertain_requests', ending['uncertain_requests'])
                    or ending.get('acknowledged_uncertain_requests') != 1
                    or ending['uncertain_requests'] != 1
                    or round(ending['reserved_cny'] * 1e6) != round(ending.get('acknowledged_reserved_cny', 0) * 1e6)):
                state['budget_audit_error'] = 'unexpected_pending_at_completion'
                if state['status'] == 'passed': state['status'] = 'incomplete'
        except Exception as exc:
            state.update(status='incomplete' if state['status'] == 'passed' else state['status'],
                         budget_after=None, budget_audit_error=type(exc).__name__)
        state['finished_at'] = now_iso()
        save()
    return 0 if state['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--parent-runtime', type=Path, default=PARENT_RUNTIME)
    parser.add_argument('--acknowledgement', type=Path)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--suite-id')
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    if not args.execute_paid:
        print(json.dumps({'status': 'not_started', 'execution_order': [GROUP], 'new_turn_count': 111}))
        return 0
    bootstrap(args.runtime)
    from camroll_agent.budget import BudgetLedger, UncertainRequestAcknowledgement
    from camroll_eval.common import read_json
    data = (read_json(args.runtime / 'freeze.json')['acknowledgement'] if args.child
            else read_json(args.acknowledgement) if args.acknowledgement else None)
    if data is None:
        parser.error('--acknowledgement required for the existing explicitly scoped unknown')
    ack = UncertainRequestAcknowledgement.from_dict(data)
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600, acknowledged_uncertain=(ack,))
    if args.child:
        run = run_child(args.runtime, args.suite_id, args.run_id, ledger, ack)
        return 0 if validate_run(run) else 2
    return execute(args.runtime, args.parent_runtime, ledger, ack)


if __name__ == '__main__':
    raise SystemExit(main())
