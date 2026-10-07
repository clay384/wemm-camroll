"""Paid synthetic connectivity checks for an explicitly scoped continuation."""
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

import gate_continuation as continuation
from gate_cloud_smoke import fixtures, text_probe_status
from gate_config import HISTORY_RUNTIME, MODELS, PYTHON

RELEASE = continuation.RELEASE
RUNTIME = continuation.RUNTIME
PARENT_RUNTIME = continuation.PARENT_RUNTIME


def independent_runtime(runtime, parent_runtime):
    runtime, parent = Path(runtime).resolve(), Path(parent_runtime).resolve()
    if (runtime == parent or runtime.is_relative_to(parent) or parent.is_relative_to(runtime)
            or runtime == RELEASE or runtime.is_relative_to(RELEASE)):
        raise ValueError('continuation_smoke_requires_independent_runtime')
    return runtime, parent


def checked_inputs(runtime, parent_runtime):
    """Reuse the completed synthetic chain only with its exact original source."""
    from camroll_eval.common import read_json, sha256
    from four_suite import source_manifest
    runtime, parent = independent_runtime(runtime, parent_runtime)
    source = source_manifest(RELEASE)
    validation_path = runtime / 'validation.json'
    validation = read_json(validation_path)
    if validation.get('status') != 'passed' or validation.get('source_files') != source:
        raise RuntimeError('matching_offline_validation_required_before_paid_smoke')
    prior_path, parent_freeze_path = parent / 'cloud-smoke.json', parent / 'freeze.json'
    prior, parent_freeze = read_json(prior_path), read_json(parent_freeze_path)
    restoration = prior.get('restoration') or {}
    if prior.get('status') != 'passed' or restoration.get('state') != 'restored':
        raise RuntimeError('passed_restored_parent_cloud_smoke_required')
    if not isinstance(restoration.get('release'), str) or not Path(restoration['release']).is_absolute():
        raise RuntimeError('parent_smoke_release_identity_required')
    parent_release = Path(restoration['release']).resolve()
    if (not prior.get('source_files') or prior['source_files'] != parent_freeze.get('source_files')
            or source_manifest(parent_release) != prior['source_files']):
        raise RuntimeError('parent_smoke_frozen_source_changed')
    protocol = continuation.protocol_manifest(RELEASE)
    if continuation.protocol_manifest(parent_release) != protocol:
        raise RuntimeError('continuation_smoke_core_protocol_changed')
    return {'source_files': source, 'protocol_files': protocol,
            'validation': {'path': str(validation_path), 'sha256': sha256(validation_path)},
            'parent_smoke': {'path': str(prior_path), 'sha256': sha256(prior_path),
                             'status': 'passed', 'restoration_state': 'restored',
                             'release': str(parent_release), 'source_files': prior['source_files']},
            'parent_freeze': {'path': str(parent_freeze_path), 'sha256': sha256(parent_freeze_path)}}


def verify_inputs(freeze, runtime):
    if checked_inputs(runtime, freeze['parent_runtime']) != freeze['inputs']:
        raise RuntimeError('continuation_smoke_inputs_changed')


def base_report(freeze):
    return {'schema_version': 'camroll-flash-gate-continuation-smoke-v1',
            'source': 'synthetic_nonformal', 'smoke_id': freeze['smoke_id'], 'status': 'running',
            'started_at': freeze['created_at'], 'models': {},
            'source_files': freeze['inputs']['source_files'],
            'protocol_files': freeze['inputs']['protocol_files'],
            'parent_cloud_smoke': freeze['inputs']['parent_smoke'],
            'acknowledgement': freeze['acknowledgement'], 'build_performed': False,
            'formal_questions_submitted': False, 'local_model_loaded': False,
            'scope': 'Continuation budget scope and Flash/Doubao connectivity only. '
                     'Cold build and Backend/Library/Agent/view/final-review chains inherit '
                     'the passed, restored parent synthetic smoke with an unchanged core protocol.'}


def run_checks(runtime, smoke_id, ledger):
    from camroll_agent.qwen import QwenClient, image_messages
    from camroll_agent.evidence_gate import EvidenceGateVerifier
    from camroll_eval.common import append_jsonl, atomic_json, now_iso, read_json
    from camroll_eval.judge import Judge
    from four_gpu_window import GpuWindow
    runtime = Path(runtime).resolve()
    freeze = read_json(runtime / 'smoke-freeze.json')
    if (freeze['smoke_id'] != smoke_id or not (runtime / 'smoke.claim').is_dir()
            or [ack.as_dict() for ack in ledger.acknowledged_uncertain] != [freeze['acknowledgement']]):
        raise RuntimeError('child_identity_does_not_match_continuation_smoke')
    window = GpuWindow(RELEASE, smoke_id, runtime=HISTORY_RUNTIME)._read()
    if window.get('state') != 'active' or window.get('current_kind') is not None:
        raise RuntimeError('child_requires_supervised_cloud_only_window')
    report, deadline = base_report(freeze), time.monotonic() + 600
    destination = runtime / 'cloud-smoke.json'
    def save():
        atomic_json(destination, report)
    def guard():
        if (runtime / 'smoke.cancel').exists():
            raise RuntimeError('continuation_smoke_cancelled')
        if time.monotonic() >= deadline:
            raise RuntimeError('continuation_smoke_deadline_exceeded')
        return continuation.ledger_ready(ledger)
    save()
    try:
        verify_inputs(freeze, runtime)
        report['before'] = guard()
        source = fixtures(runtime=runtime)
        journal, calls = runtime / 'smoke-request-journal.sqlite', runtime / 'smoke-model-calls.jsonl'
        schema = {'type': 'object', 'properties': {'color': {'type': 'string', 'enum': ['red']}},
                  'required': ['color'], 'additionalProperties': False}
        tool = {'type': 'function', 'function': {'name': 'record_color', 'description': 'Record a color', 'parameters': schema}}
        clients = {}
        for model in dict.fromkeys(MODELS.values()):
            guard()
            client = QwenClient(model=model, strict_response_model=True, ledger=ledger, expected_limit_cny=600,
                request_namespace=smoke_id + ':' + model, request_journal_path=journal)
            clients[model] = client
            result = {'checks': {}, 'status': 'running'}
            report['models'][model] = result
            save()
            with client.capture_calls(lambda event: append_jsonl(calls, event)), client.request_scope(
                    deadline_s=max(.001, deadline - time.monotonic()), max_requests=50):
                guard()
                response = client.chat([{'role': 'user', 'content': 'Reply exactly OK.'}], role='probe',
                                       max_tokens=64, request_id='text')
                result['checks']['text'] = {**text_probe_status(response.get('content')), 'usage': response['_usage']}
                save(); guard()
                response = client.chat(image_messages('Name the colored shape in this picture.', [source / 'photos/red.jpg']),
                                       role='probe', max_tokens=128, request_id='image')
                content = (response.get('content') or '').lower()
                result['checks']['image'] = {'passed': 'red' in content and 'square' in content, 'usage': response['_usage']}
                save(); guard()
                response = client.chat([{'role': 'user', 'content': 'Call record_color with color red.'}], [tool],
                    tool_choice={'type': 'function', 'function': {'name': 'record_color'}}, role='probe', max_tokens=128, request_id='tool')
                result['checks']['tool'] = {'passed': any(row.get('function', {}).get('name') == 'record_color'
                    and json.loads(row['function']['arguments']) == {'color': 'red'} for row in response.get('tool_calls', [])),
                    'usage': response['_usage']}
                save(); guard()
                with client.request_context('structured'):
                    output = client.generate_json('Return the color red.', [], schema, role='probe', max_attempts=1)
                result['checks']['structured'] = {'passed': output == {'color': 'red'}}
            guard()
            result['status'] = 'passed' if all(row['passed'] for row in result['checks'].values()) else 'failed'
            save()
            if result['status'] != 'passed':
                raise RuntimeError('flash_capability_smoke_failed')
        verifier = EvidenceGateVerifier(client=clients['qwen3.7-flash'], ledger=ledger,
            on_call=lambda event: append_jsonl(runtime / 'smoke-verifier-calls.jsonl', event))
        guard()
        with clients['qwen3.7-flash'].request_scope(deadline_s=max(.001, deadline - time.monotonic()), max_requests=50):
            verdict = verifier.verify(question='Describe the colored shape in each supplied image.',
                candidate_answer='img-red shows a red square; img-blue shows a blue circle.', outcome='answer',
                images=[{'image_id': 'img-red', 'path': source / 'photos/red.jpg'},
                        {'image_id': 'img-blue', 'path': source / 'photos/blue.jpg'}],
                delivery_image_ids=['img-red', 'img-blue'], trusted_metadata=[], derived_observations=[],
                request_id=smoke_id + ':two-images', deadline=deadline,
                request_budget_hook=clients['qwen3.7-flash']._gate_request_budget_hook)
        report['verifier'] = {'status': 'passed' if verdict['status'] == 'supported' else 'failed',
                              'verdict': verdict, 'protocol': verifier.protocol()}
        save(); guard()
        if report['verifier']['status'] != 'passed':
            raise RuntimeError('doubao_fact_capability_smoke_failed')
        judge = Judge(ledger=ledger, expected_limit_cny=600, request_namespace=smoke_id + ':judge',
            request_journal_path=journal, on_call=lambda event: append_jsonl(runtime / 'smoke-judge-calls.jsonl', event))
        with judge.request_context('arithmetic-quality'):
            quality = judge._judge('2+2等于几？', '4', '4', expected_action='answer')
        report['judge'] = {'status': 'running', 'quality': quality, 'protocol': judge.protocol()}
        save(); guard()
        if quality.get('score') != 2 or quality.get('judge_status') != 'completed':
            raise RuntimeError('doubao_quality_capability_smoke_failed')
        with judge.request_context('arithmetic-task'):
            behavior = judge._judge_task_action('2+2等于几？', '4', 'answer', 'complete', 'ok', reference='4')
        report['judge']['behavior'] = behavior
        report['judge']['status'] = 'passed' if (behavior.get('correct') is True
            and behavior.get('actual_action') == 'answer' and behavior.get('judge_status') == 'completed') else 'failed'
        save(); guard()
        if report['judge']['status'] != 'passed':
            raise RuntimeError('doubao_behavior_capability_smoke_failed')
        verify_inputs(freeze, runtime)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='blocked', error_type=type(exc).__name__, error_code=getattr(exc, 'code', None),
                      reason=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)
    finally:
        try:
            report['after'] = ledger.summary()
        except Exception as exc:
            report.update(status='blocked', budget_audit_error=type(exc).__name__)
        report['finished_at'] = now_iso()
        save()
    return report['status'] == 'passed'


def execute(runtime, parent_runtime, ledger, acknowledgement):
    from camroll_eval.common import atomic_json, read_json, now_iso
    from four_gpu_window import GpuWindow, verify_model_files
    from four_supplement import stop_owned_child
    runtime, parent_runtime = independent_runtime(runtime, parent_runtime)
    before = continuation.ledger_ready(ledger)
    if [ack.as_dict() for ack in ledger.acknowledged_uncertain] != [acknowledgement.as_dict()]:
        raise RuntimeError('continuation_smoke_budget_scope_mismatch')
    inputs = checked_inputs(runtime, parent_runtime)
    if (runtime / 'cloud-smoke.json').exists() or (runtime / 'smoke-freeze.json').exists():
        raise RuntimeError('smoke_already_recorded_inspect_before_repeat')
    (runtime / 'smoke.claim').mkdir(mode=0o700)
    smoke_id = 'gate-continuation-smoke-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    freeze = {'schema_version': 'camroll-flash-gate-continuation-smoke-freeze-v1', 'smoke_id': smoke_id,
              'created_at': now_iso(), 'inputs': inputs, 'parent_runtime': str(parent_runtime),
              'acknowledgement': acknowledgement.as_dict()}
    atomic_json(runtime / 'smoke-freeze.json', freeze)
    initial = base_report(freeze)
    initial['before'] = before
    atomic_json(runtime / 'cloud-smoke.json', initial)
    window = GpuWindow(RELEASE, smoke_id, runtime=HISTORY_RUNTIME)
    child, entered, code, failure = None, None, 2, None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('smoke_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        entered = window.enter()
        command = [str(PYTHON), str(Path(__file__).resolve()), '--execute-paid', '--runtime', str(runtime),
                   '--child', '--smoke-id', smoke_id]
        with (runtime / 'cloud-smoke.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            window.register_child(child.pid, expected_command=command)
            deadline = time.monotonic() + 600
            while child.poll() is None:
                time.sleep(1)
                ledger.check_limit()
                value = ledger.summary()
                if value.get('blocking_uncertain_requests', value['uncertain_requests']):
                    marker = runtime / 'smoke.cancel'
                    if not marker.exists():
                        marker.write_text('new_uncertain_cloud_request_stop'); marker.chmod(0o600)
                if time.monotonic() >= deadline:
                    raise RuntimeError('continuation_smoke_child_deadline_exceeded')
            code = child.returncode
            window.unregister_child(child.pid)
    except BaseException as exc:
        failure = {'error_type': type(exc).__name__, 'error_code': getattr(exc, 'code', None),
                   'reason': str(exc) if isinstance(exc, (RuntimeError, KeyboardInterrupt)) else type(exc).__name__}
    finally:
        cleanup, restoration = None, {'state': 'not_entered'}
        try:
            try:
                cleanup = stop_owned_child(child)
            except BaseException as exc:
                cleanup = {'state': 'failed', 'error_type': type(exc).__name__}
            finally:
                if window.state_path.exists():
                    restoration = window.restore('gate_continuation_smoke_exit')
        except BaseException as exc:
            restoration = {'state': 'restore_failed', 'error_type': type(exc).__name__}
        atomic_json(runtime / 'smoke-restoration.json', restoration)
        path = runtime / 'cloud-smoke.json'
        report = read_json(path) if path.exists() else {**initial, 'status': 'blocked', 'reason': 'child_did_not_write_report'}
        report.update(restoration=restoration, child_cleanup=cleanup, child_returncode=code, finished_at=now_iso())
        if failure:
            report.update(failure, status='blocked')
        if (code != 0 or restoration.get('state') != 'restored'
                or restoration.get('no_experiment_gpu_residual') is not True or not cleanup or cleanup.get('state') == 'failed'):
            report['status'] = 'blocked'
        try:
            verify_inputs(freeze, runtime)
            if entered and verify_model_files()['models'] != entered['model_integrity']['models']:
                raise RuntimeError('protected_model_files_changed')
            report['source_unchanged'] = True
            report['model_files_unchanged'] = bool(entered)
            report['after'] = continuation.ledger_ready(ledger)
        except Exception as exc:
            report.update(status='blocked', finalization_error=getattr(exc, 'code', None) or type(exc).__name__)
            try:
                report['after'] = ledger.summary()
            except Exception as audit_exc:
                report['budget_audit_error'] = type(audit_exc).__name__
        atomic_json(path, report)
    return 0 if report['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--parent-runtime', type=Path, default=PARENT_RUNTIME)
    parser.add_argument('--acknowledgement', type=Path)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--smoke-id')
    args = parser.parse_args(argv)
    continuation.bootstrap(args.runtime)
    if not args.execute_paid:
        print(json.dumps({'status': 'not_started', 'scope': 'continuation_connectivity_smoke'}))
        return 0
    from camroll_agent.budget import BudgetLedger, UncertainRequestAcknowledgement
    from camroll_eval.common import read_json
    data = (read_json(args.runtime / 'smoke-freeze.json')['acknowledgement'] if args.child
            else read_json(args.acknowledgement) if args.acknowledgement else None)
    if data is None:
        parser.error('--acknowledgement is required for paid continuation smoke')
    ack = UncertainRequestAcknowledgement.from_dict(data)
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600, acknowledged_uncertain=(ack,))
    if args.child:
        return 0 if run_checks(args.runtime, args.smoke_id, ledger) else 2
    return execute(args.runtime, args.parent_runtime, ledger, ack)


if __name__ == '__main__':
    raise SystemExit(main())
