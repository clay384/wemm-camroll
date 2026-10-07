"""One-shot continuation with an explicitly acknowledged, still-reserved request."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

from gate_config import (ALBUMS, BASE, HISTORY_RUNTIME, ORDER, PYTHON, SNAPSHOT,
                         SOURCE, SOURCE_ALBUM_ID, SOURCE_RUN_ID, configure)
from gate_suite import validate_run

RELEASE = Path(__file__).resolve().parents[1]
PARENT_RUNTIME = BASE / 'runtime/photobench-flash-gate-20261006f'
RUNTIME = BASE / 'runtime/photobench-flash-gate-20261007g'
EXECUTION_ORDER = ('F37-On', 'F38-Off', 'F38-On')
WEMM_WEIGHT = Path('/home/realmagic/wemm-embedding-lab/models/WeMM-Embedding-9B/model.safetensors')
# Same immutable weight recorded by the legacy fingerprint and source lock.
WEMM_WEIGHT_SHA256 = 'b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4'
WEMM_WEIGHT_SIZE = 18_815_757_458


def bootstrap(runtime):
    configure(RELEASE)
    os.environ['CAMROLL_EVAL_ROOT'] = str(Path(runtime) / 'evaluation')
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))


def ledger_ready(ledger):
    """Only the exact acknowledged reservation may predate this execution."""
    ledger.check_limit()
    value = ledger.summary()
    if (value.get('blocking_uncertain_requests', value['uncertain_requests'])
            or value.get('acknowledged_uncertain_requests') != 1
            or value['uncertain_requests'] != 1
            or round(value['reserved_cny'] * 1e6) != round(value.get('acknowledged_reserved_cny', 0) * 1e6)):
        raise RuntimeError('unexpected_pending_requests_require_reconciliation')
    if value['remaining_cny'] <= 0:
        raise RuntimeError('budget_exhausted')
    return value


def verify_wemm_weights(path=WEMM_WEIGHT, *, expected_sha256=WEMM_WEIGHT_SHA256,
                        expected_size=WEMM_WEIGHT_SIZE):
    """Hash every byte without importing a backend or loading GPU weights."""
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError('wemm_weight_must_be_a_regular_file')
    def identity(value):
        return {'device': value.st_dev, 'inode': value.st_ino, 'size': value.st_size,
                'mtime_ns': str(value.st_mtime_ns), 'ctime_ns': str(value.st_ctime_ns)}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size:
            raise RuntimeError('wemm_weight_size_or_type_changed')
        before = identity(opened)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        after = identity(os.fstat(stream.fileno()))
    if before != after or after != identity(path.stat()):
        raise RuntimeError('wemm_weight_identity_changed_during_hash')
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError('wemm_weight_hash_changed')
    return {'path': str(path.resolve()), 'expected_sha256': expected_sha256,
            'actual_sha256': actual, 'file_identity_before': before, 'file_identity_after': after,
            'size': after['size'], 'identity_stable': True, 'verified_at_epoch': time.time()}


def protocol_manifest(release):
    """Freeze substantive protocol while allowing explicit accounting wiring."""
    from camroll_eval.common import sha256
    agent = Path(release) / 'camroll/camroll-agent'
    exact = ('camroll_agent/agent.py', 'camroll_agent/gate.py', 'camroll_agent/prompts.py',
             'camroll_eval/judge_prompts.json', 'camroll_eval/task_actions.json', 'camroll_eval/gate_metrics.py')
    result = {name: sha256(agent / name) for name in exact}
    class AccountingClone(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            owner = bound = None
            if (isinstance(node.func, ast.Name) and node.func.id == 'BudgetLedger'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Attribute)
                    and node.args[0].attr == 'path' and len(node.keywords) == 1
                    and node.keywords[0].arg == 'expected_limit_cny'):
                owner, bound = node.args[0].value, node.keywords[0].value
            elif (isinstance(node.func, ast.Attribute) and node.func.attr == 'with_expected_limit'
                    and len(node.args) == 1 and not node.keywords):
                owner, bound = node.func.value, node.args[0]
            if isinstance(owner, ast.Name) and owner.id in {'ledger', 'source_ledger'}:
                return ast.Call(func=ast.Name(id='__scope_preserving_ledger_clone__', ctx=ast.Load()),
                                args=[owner, bound], keywords=[])
            return node
    # The only normalized change is the exact scope-preserving ledger clone;
    # constructor sampling/limits/endpoint checks remain part of the hash.
    for name in ('camroll_agent/qwen.py', 'camroll_agent/evidence_gate.py', 'camroll_eval/judge.py'):
        tree = ast.parse((agent / name).read_text())
        tree = AccountingClone().visit(tree)
        result[name + ':scope_preserving_budget_clone'] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    tree = ast.parse((agent / 'camroll_eval/gate_experiment.py').read_text())
    selected = [node for node in tree.body if getattr(node, 'name', None) in
                {'InferenceConfig', 'inference_config', 'IndexBinding', 'gate_summary', 'valid_judgment'}]
    result['inference_binding_and_scoring'] = hashlib.sha256(
        '\n'.join(ast.dump(node) for node in selected).encode()).hexdigest()
    return result


def verify_frozen(freeze, runtime):
    from camroll_eval.common import read_json, sha256
    from four_suite import source_manifest
    if source_manifest(RELEASE) != freeze['source_files']:
        raise RuntimeError('continuation_source_changed')
    parent_freeze = read_json(freeze['parent_freeze']['path'])
    if sha256(freeze['parent_freeze']['path']) != freeze['parent_freeze']['sha256']:
        raise RuntimeError('parent_freeze_changed')
    parent_release = Path(freeze['parent_release'])
    if source_manifest(parent_release) != parent_freeze['source_files']:
        raise RuntimeError('parent_frozen_source_changed')
    if protocol_manifest(RELEASE) != freeze['protocol_files'] or protocol_manifest(parent_release) != freeze['protocol_files']:
        raise RuntimeError('continuation_protocol_changed')
    for record in freeze['parent_runs'].values():
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError('parent_run_changed')
    if sha256(freeze['parent_suite']['path']) != freeze['parent_suite']['sha256']:
        raise RuntimeError('parent_suite_changed')
    if source_manifest(ALBUMS / SOURCE_ALBUM_ID) != parent_freeze['album_files']:
        raise RuntimeError('frozen_album_changed')
    from camroll_eval.dataset import Dataset
    if Dataset(SOURCE).integrity() != parent_freeze['input_integrity']:
        raise RuntimeError('frozen_inputs_changed')
    original = HISTORY_RUNTIME / 'evaluation/runs' / SOURCE_RUN_ID / 'run.json'
    if sha256(original) != parent_freeze['source_run_sha256']:
        raise RuntimeError('source_run_changed')
    for filename in ('validation.json', 'cloud-smoke.json'):
        record = read_json(Path(runtime) / filename)
        if record.get('status') != 'passed' or record.get('source_files') != freeze['source_files']:
            raise RuntimeError('matching_continuation_validation_and_smoke_required')


def freeze_inputs(runtime, parent_runtime, acknowledgement):
    from camroll_eval.common import atomic_json, read_json, now_iso, sha256
    from four_suite import source_manifest
    from four_gpu_window import verify_files, verify_model_files, _legacy_fingerprint
    runtime, parent_runtime = Path(runtime).resolve(), Path(parent_runtime).resolve()
    if runtime == parent_runtime or runtime.is_relative_to(parent_runtime) or parent_runtime.is_relative_to(runtime):
        raise ValueError('continuation_requires_independent_runtime')
    parent_path = parent_runtime / 'formal-suite.json'
    parent = read_json(parent_path)
    if (parent.get('status') != 'blocked' or parent.get('restoration', {}).get('state') != 'restored'
            or parent.get('model_files_unchanged') is not True or parent.get('frozen_inputs_unchanged') is not True):
        raise RuntimeError('restored_frozen_parent_required')
    groups = {row['name']: row for row in parent['groups']}
    if tuple(row['name'] for row in parent['groups']) != ORDER:
        raise RuntimeError('unexpected_parent_groups')
    parents = {}
    for name in ('F37-Off', 'F37-On'):
        path = parent_runtime / 'evaluation/runs' / groups[name]['run_id'] / 'run.json'
        run = read_json(path)
        if run.get('suite_id') != parent['suite_id'] or run.get('inference_config', {}).get('group') != name:
            raise RuntimeError('parent_run_identity_mismatch')
        if name == 'F37-Off' and not validate_run(run):
            raise RuntimeError('completed_off_parent_required')
        if name == 'F37-On' and (run.get('status') != 'blocked' or run.get('stop_reason') != 'request_uncertain'):
            raise RuntimeError('uncertain_on_parent_required')
        parents[name] = {'path': str(path), 'sha256': sha256(path)}
    if any(groups[name]['status'] != 'not_started' for name in ('F38-Off', 'F38-On')):
        raise RuntimeError('unstarted_flash38_parents_required')
    ack = acknowledgement.as_dict()
    # Identity, payload and row hashes are additionally enforced atomically by
    # BudgetLedger; the exception must belong to this exact parent journal.
    journal = Path(parents['F37-On']['path']).parent / 'request-journal.sqlite'
    if Path(ack['journal_path']).resolve() != journal.resolve():
        raise RuntimeError('acknowledgement_not_from_parent_on_journal')
    parent_freeze_path = parent_runtime / 'freeze.json'
    parent_freeze = read_json(parent_freeze_path)
    result = {'schema_version': 'camroll-flash-gate-continuation-freeze-v1', 'created_at': now_iso(),
        'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'source_files': source_manifest(RELEASE),
        'parent_runtime': str(parent_runtime), 'parent_suite_id': parent['suite_id'],
        'parent_suite': {'path': str(parent_path), 'sha256': sha256(parent_path)},
        'parent_freeze': {'path': str(parent_freeze_path), 'sha256': sha256(parent_freeze_path)},
        'parent_release': parent['restoration']['release'], 'parent_runs': parents,
        'protocol_files': protocol_manifest(RELEASE), 'acknowledgement': ack,
        'execution_order': list(EXECUTION_ORDER), 'display_order': list(ORDER),
        'snapshot': SNAPSHOT, 'build_performed': False,
        'protected_files': verify_files(), 'model_files': verify_model_files(),
        'wemm_weights': verify_wemm_weights(),
        'wemm_fingerprint': _legacy_fingerprint()}
    if (parent_freeze.get('snapshot') != SNAPSHOT
            or result['wemm_fingerprint'] != parent_freeze['wemm_fingerprint']):
        raise RuntimeError('parent_index_or_fingerprint_changed')
    verify_frozen(result, runtime)
    path = runtime / 'freeze.json'
    if path.exists():
        raise RuntimeError('continuation_freeze_already_exists')
    atomic_json(path, result)
    return result


def run_child(runtime, suite_id, run_id, group, ledger):
    from camroll_eval.common import read_json
    from camroll_eval.gate_experiment import GateRunner
    from four_gpu_window import GpuWindow
    runtime = Path(runtime)
    suite, freeze = read_json(runtime / 'formal-suite.json'), read_json(runtime / 'freeze.json')
    matching = [row for row in suite['groups'] if row['name'] == group]
    if (group not in EXECUTION_ORDER or suite['suite_id'] != suite_id or suite['status'] != 'running'
            or len(matching) != 1 or matching[0]['run_id'] != run_id or matching[0]['status'] != 'running'):
        raise RuntimeError('child_identity_does_not_match_continuation')
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)._read()
    if window.get('state') != 'active' or window.get('current_kind') != 'wemm':
        raise RuntimeError('child_requires_supervised_wemm_window')
    ledger_ready(ledger)
    verify_frozen(freeze, runtime)
    extra = ({'continuation_parent': freeze['parent_runs']['F37-On'],
              'acknowledged_call_ids': [freeze['acknowledgement']['call_id']]} if group == 'F37-On' else {})
    return GateRunner(runtime / 'evaluation', SOURCE, ALBUMS, budget_ledger=ledger).run_gate(
        run_id=run_id, suite_id=suite_id, group=group, source_album_id=SOURCE_ALBUM_ID,
        source_run_id=SOURCE_RUN_ID, source_evaluation_root=HISTORY_RUNTIME / 'evaluation',
        expected_snapshot=SNAPSHOT, **extra)


def execute(runtime, parent_runtime, ledger, acknowledgement):
    from camroll_eval.common import atomic_json, read_json, now_iso
    from camroll_eval.dataset import Dataset
    from four_gpu_window import GpuWindow, verify_model_files
    from four_supplement import stop_owned_child
    from gate_continuation_export import create_reference_run
    runtime = Path(runtime).resolve()
    parent_runtime = Path(parent_runtime).resolve()
    if runtime == parent_runtime or runtime.is_relative_to(parent_runtime) or parent_runtime.is_relative_to(runtime):
        raise ValueError('continuation_requires_independent_runtime')
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    (runtime / 'formal-suite.claim').mkdir(mode=0o700)
    suite_id = 'gate-continuation-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    groups = [{'name': name, 'run_id': name.lower() + '-' + suite_id, 'status': 'not_started',
               'qa_count': 100, 'planned_turn_count': 111} for name in ORDER]
    state = {'schema_version': 'camroll-flash-gate-continuation-suite-v1', 'suite_id': suite_id,
             'status': 'preparing', 'started_at': now_iso(), 'groups': groups,
             'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'build_performed': False,
             'execution_order': list(EXECUTION_ORDER), 'acknowledgement': acknowledgement.as_dict(),
             'parent_runtime': str(Path(parent_runtime).resolve()), 'budget_before': ledger.summary()}
    def save():
        state['updated_at'] = now_iso()
        atomic_json(runtime / 'formal-suite.json', state)
        atomic_json(runtime / 'evaluation/comparison-profile.json', {
            'schema_version': 'camroll-flash-gate-v1', 'title': 'Qwen Flash · Gate 独立续测',
            'groups': groups, 'suite_status': state['status'], 'updated_at': state['updated_at'],
            'snapshot': SNAPSHOT, 'note': 'F37-Off为完整旧结果只读引用；F37-On保留既有轮次，只执行剩余轮次。'
            '旧未知请求未重发且预留未释放。各组固定100题/111轮/81题交付分母；独立请求费用与继承费用分列。'})
    save()
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)
    active = freeze = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('suite_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        ledger_ready(ledger)
        freeze = freeze_inputs(runtime, parent_runtime, acknowledgement)
        dataset = Dataset(SOURCE)
        plans = [{'qa_id': row['qa_id'], 'question': row['question'],
                  'planned_turn_count': len(dataset.turns(row)), 'execution_status': 'not_started',
                  'judge_status': 'not_started'} for row in dataset.rows]
        atomic_json(runtime / 'evaluation/planned-groups.json', {
            'suite_id': suite_id, 'groups': [{**group, 'qa_plan': plans} for group in groups]})
        parent = freeze['parent_runs']['F37-Off']
        reference = create_reference_run(parent['path'], runtime / 'evaluation', groups[0]['run_id'],
                                         suite_id, parent['sha256'])
        groups[0].update(status=reference['status'], reused=True, returncode=0, acceptance_complete=True,
                         parent_run_path=parent['path'], parent_run_sha256=parent['sha256'], summary=reference['summary'])
        entered = window.enter()
        state['window_state_path'] = str(window.state_path)
        if entered['before']['fingerprint'] != freeze['wemm_fingerprint']:
            raise RuntimeError('window_fingerprint_differs_from_freeze')
        window.switch('wemm')
        state['status'] = 'running'
        save()
        for group in groups[1:]:
            ledger_ready(ledger)
            verify_frozen(freeze, runtime)
            group.update(status='running', started_at=now_iso())
            save()
            log_path = runtime / 'logs' / (group['run_id'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            command = [str(PYTHON), str(Path(__file__).resolve()), '--execute-paid', '--runtime', str(runtime),
                       '--child-group', group['name'], '--run-id', group['run_id'], '--suite-id', suite_id]
            with log_path.open('xb') as log:
                active = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                        if stopped_at is None:
                            stopped_at = time.monotonic()
                            group['new_submissions_blocked'] = reason
                            save()
                    if stopped_at is not None and time.monotonic() - stopped_at >= 600:
                        raise RuntimeError('blocked_child_did_not_exit_within_turn_deadline')
                group['returncode'] = active.returncode
                window.unregister_child(active.pid)
                active = None
            run_path = runtime / 'evaluation/runs' / group['run_id'] / 'run.json'
            run = read_json(run_path) if run_path.exists() else {}
            group.update(status=run.get('status', 'failed'), finished_at=now_iso(),
                         acceptance_complete=group['returncode'] == 0 and validate_run(run), summary=run.get('summary'))
            verify_frozen(freeze, runtime)
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
                state['restoration'] = window.restore('gate_continuation_' + state['status']) if window.state_path.exists() else {'state': 'not_entered'}
            if freeze:
                if verify_model_files()['models'] != freeze['model_files']['models']:
                    raise RuntimeError('protected_model_files_changed')
                weights = verify_wemm_weights()
                state['wemm_weights_after'] = weights
                if (weights['actual_sha256'] != freeze['wemm_weights']['actual_sha256']
                        or weights['file_identity_before'] != freeze['wemm_weights']['file_identity_after']):
                    raise RuntimeError('wemm_weight_changed_since_freeze')
                state['wemm_weights_unchanged'] = True
                state['model_files_unchanged'] = True
                verify_frozen(freeze, runtime)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--parent-runtime', type=Path, default=PARENT_RUNTIME)
    parser.add_argument('--acknowledgement', type=Path, help='Private JSON from an explicit acknowledgement capture')
    parser.add_argument('--child-group', choices=EXECUTION_ORDER)
    parser.add_argument('--run-id')
    parser.add_argument('--suite-id')
    args = parser.parse_args(argv)
    bootstrap(args.runtime)
    from camroll_agent.budget import BudgetLedger, UncertainRequestAcknowledgement
    from camroll_eval.common import read_json
    if not args.execute_paid:
        print(json.dumps({'status': 'not_started', 'execution_order': EXECUTION_ORDER}))
        return 0
    data = (read_json(args.runtime / 'freeze.json')['acknowledgement'] if args.child_group
            else read_json(args.acknowledgement) if args.acknowledgement else None)
    if data is None:
        parser.error('--acknowledgement is required for paid continuation')
    ack = UncertainRequestAcknowledgement.from_dict(data)
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600, acknowledged_uncertain=(ack,))
    if args.child_group:
        result = run_child(args.runtime, args.suite_id, args.run_id, args.child_group, ledger)
        return 0 if validate_run(result) else 2
    return execute(args.runtime, args.parent_runtime, ledger, ack)


if __name__ == '__main__':
    raise SystemExit(main())
