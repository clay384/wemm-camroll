"""Supervised real-photo E2E and formal MiniCPM runs with immutable ancestry."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from minicpm_config import (ALBUMS, BASELINE_RUNTIME, HISTORY_RUNTIME, PYTHON, SNAPSHOT,
                            SOURCE, SOURCE_ALBUM_ID, SOURCE_RUN_ID, configure, validate_runtime,
                            SDK_RUNTIME_PREFIX, request_matches_profile, protocol_profile, annotation_review,
                            acceptance_policy, FACTS_AND_INTERFACE_POLICY, INTERFACE_ONLY_POLICY)

RELEASE = Path(__file__).resolve().parents[1]
BASELINE_RUN_ID = 'f37-off-gate-20261006-181014-8bd9a5'
BASELINE_RUN_SHA256 = '8ee4497f779148783241dc0c2add0574d727e8a368e96687b6aca0bdbcc91d37'
RUN_PREFIX = 'minicpm47-thinking-m4-off-'
PRICING_CAPTURE_TIMESTAMPS = frozenset({'observed_at', 'collected_at', 'captured_at', 'retrieved_at'})


def bootstrap(runtime):
    configure(RELEASE, runtime)
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))


def ledger_ready(ledger):
    """Retain exactly the old acknowledged unknown, including its reservation."""
    from gate_continuation import ledger_ready as existing_check
    value = existing_check(ledger)
    if value.get('limit_cny') != 600:
        raise RuntimeError('budget_limit_changed')
    return value


def e2e_trace_complete(items, profile):
    """Exercise real retrieval, every tool contract, and one M4 review per turn."""
    from camroll_agent.schemas import TOOL_NAMES
    from camroll_eval.minicpm_experiment import MODEL
    successful_tools, embedded = set(), False
    for item in items:
        traces = item.get('tool_trace') or []
        calls = item.get('model_call_metrics') or []
        if not calls:
            return False
        for call in calls:
            body = call.get('input') or {}
            if call.get('model') != profile.generation_model or not request_matches_profile(profile, body):
                return False
        for turn in item.get('runtime_turns') or []:
            index = turn.get('index', 0)
            current = [row for row in traces if row.get('conversation_turn') == index]
            reviews = [row for row in current if row.get('tool') == 'final_image_review']
            if (len(reviews) != 1 or reviews[0].get('status') not in ('ok', 'success')
                    or (reviews[0].get('result') or {}).get('error')
                    or not 1 <= len((reviews[0].get('args') or {}).get('image_ids') or []) <= 6):
                return False
            if not any(call.get('conversation_turn') == index and call.get('call_type') == 'final_review'
                       and call.get('status') == 'success' for call in calls):
                return False
            for row in current:
                if (row.get('tool') in TOOL_NAMES and row.get('status') in ('ok', 'success')
                        and not (row.get('result') or {}).get('error') and row.get('cached') is not True):
                    successful_tools.add(row['tool'])
                    if row['tool'] == 'search':
                        embedded = embedded or any(call.get('operation') == '/api/internal/embeddings'
                            and call.get('status') == 'success' for call in row.get('embedding_calls') or [])
    return successful_tools == set(TOOL_NAMES) and embedded


def validate_run(state, *, stage='formal'):
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig, SUPPORTED_PROTOCOLS
    if (state.get('mode') != 'minicpm_compare' or state.get('protocol_label') not in SUPPORTED_PROTOCOLS
            or state.get('inference_complete') is not True):
        return False
    try:
        profile = MiniCPMInferenceConfig(**state['inference_config'])
        if state['protocol_label'] != profile.protocol_version:
            return False
    except (TypeError, ValueError, KeyError):
        return False
    if stage == 'formal':
        from gate_suite import validate_run as historical_validator
        return historical_validator({**state, 'mode': 'gate_ab'})
    items = state.get('items') or []
    turns = [turn for row in items for turn in row.get('runtime_turns') or []]
    plans = state.get('qa_plan') or []
    return (state.get('status') in ('completed', 'completed_with_errors') and state.get('formal_protocol') is False
        and state.get('source_unchanged') is True and not state.get('finalization_errors')
        and len(items) == 3 and len(turns) == 3 and len(plans) == 3
        and len({row.get('qa_id') for row in plans}) == 3
        and {row.get('qa_id') for row in items} == {row.get('qa_id') for row in plans}
        and all(row.get('planned_turn_count') == 1 for row in plans)
        and all(len(row.get('runtime_turns') or []) == 1 for row in items)
        and e2e_trace_complete(items, profile)
        and all(turn.get('answer_origin') == 'model' and turn.get('judge_status') == 'completed'
                and type((turn.get('judge') or {}).get('score')) is int
                and turn['judge']['score'] in (0, 1, 2)
                and type((turn.get('task_judge') or {}).get('correct')) is bool
                and (turn.get('task_judge') or {}).get('actual_action') in ('answer', 'clarify', 'refuse', 'none')
                for turn in turns))


def e2e_evidence_paths(runtime, run_id):
    from camroll_eval.common import safe_id
    root = Path(runtime).resolve()
    paths = {'run': root / 'evaluation/runs' / safe_id(run_id) / 'run.json',
             'suite': root / 'e2e-suite.json', 'freeze': root / 'e2e-freeze.json'}
    if any(not path.is_file() or path.is_symlink() or path.resolve() != path.absolute() for path in paths.values()):
        raise RuntimeError('regular_local_e2e_evidence_required')
    return paths


def validate_e2e_evidence(runtime, status, source, profile):
    from camroll_eval.common import read_json, sha256
    paths = e2e_evidence_paths(runtime, status.get('e2e_run_id'))
    expected = status.get('e2e_evidence') or {}
    if set(expected) != set(paths) or any(expected[name] != sha256(path) for name, path in paths.items()):
        raise RuntimeError('successful_e2e_evidence_hashes_changed_or_missing')
    run, suite, freeze = (read_json(paths[name]) for name in ('run', 'suite', 'freeze'))
    revision = os.getenv('CAMROLL_RELEASE_REVISION', RELEASE.name)
    if (suite.get('status') != 'passed' or suite.get('stage') != 'e2e'
            or suite.get('run_id') != status.get('e2e_run_id')
            or suite.get('suite_id') != status.get('e2e_suite_id')
            or suite.get('restoration', {}).get('state') != 'restored'
            or run.get('run_id') != status.get('e2e_run_id') or run.get('suite_id') != status.get('e2e_suite_id')
            or not validate_run(run, stage='e2e')):
        raise RuntimeError('successful_restored_e2e_evidence_required')
    if (run.get('inference_config') != profile.as_dict() or run.get('protocol_label') != profile.protocol_version
            or run.get('inference_fingerprint') != profile.fingerprint
            or freeze.get('stage') != 'e2e' or freeze.get('source_files') != source
            or freeze.get('inference_config') != profile.as_dict()
            or freeze.get('inference_fingerprint') != profile.fingerprint
            or freeze.get('source_revision') != revision or suite.get('source_revision') != revision
            or run.get('code_revision') != revision
            or freeze.get('snapshot') != SNAPSHOT or run.get('index_version') != SNAPSHOT
            or not freeze.get('index_binding') or run.get('index_binding') != freeze['index_binding']):
        raise RuntimeError('e2e_source_profile_or_index_differs_from_current_acceptance')
    return paths


def acceptance_inputs(runtime, stage):
    from camroll_eval.common import read_json
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    from camroll_agent.modelbest import _pricing_evidence
    from four_suite import source_manifest
    runtime = Path(runtime)
    source = source_manifest(RELEASE)
    status = read_json(runtime / 'acceptance/status.json')
    validation = read_json(runtime / 'validation.json')
    if (status.get('status') != 'passed' or validation.get('status') != 'passed'
            or status.get('source_files') != source or validation.get('source_files') != source):
        raise RuntimeError('matching_source_validation_and_cloud_acceptance_required')
    if stage == 'formal' and status.get('e2e_status') != 'passed':
        raise RuntimeError('successful_real_photo_e2e_required_before_formal')
    profile = MiniCPMInferenceConfig(**status['inference_config'])
    if stage == 'formal':
        validate_e2e_evidence(runtime, status, source, profile)
    policy = acceptance_policy(profile)
    if status.get('acceptance_policy', FACTS_AND_INTERFACE_POLICY) != policy:
        raise RuntimeError('accepted_policy_differs_from_inference_protocol')
    if policy == INTERFACE_ONLY_POLICY:
        from minicpm_acceptance import passed, planned_cases
        from camroll_eval.common import sha256
        review = status.get('independent_review') or {}
        review_path = runtime / 'acceptance/independent-review.json'
        if (status.get('schema_version') != 'minicpm-30-case-acceptance-v2'
                or status.get('facts_block_acceptance') is not False
                or not passed(status.get('metrics') or {}, policy)
                or status.get('manual_review_required') is not False
                or review.get('reviewed_cases') != 30 or review.get('path') != 'independent-review.json'
                or not review_path.is_file() or review_path.is_symlink()
                or review.get('sha256') != sha256(review_path)):
            raise RuntimeError('complete_interface_acceptance_and_hash_bound_review_required')
        document = read_json(review_path)
        decisions = document.get('cases') or []
        expected_ids = {row['case_id'] for row in planned_cases()}
        if (document.get('schema_version') != 'minicpm-independent-interface-review-v2'
                or document.get('reviewer') != 'assistant-independent-visual-review'
                or len(decisions) != 30 or {row.get('case_id') for row in decisions} != expected_ids
                or any(row.get('interface_contract_valid') is not True
                       or any(type(row.get(field)) is not bool for field in ('facts_verified', 'identity_grounded'))
                       for row in decisions)):
            raise RuntimeError('complete_interface_review_decisions_required')
        if any(row.get('case_sha256') != sha256(runtime / 'acceptance/cases' / (row['case_id'] + '.json'))
               for row in decisions):
            raise RuntimeError('reviewed_interface_case_evidence_changed')
    protocol_path = runtime / 'acceptance/protocol.json'
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol_profile(runtime, protocol) != profile:
            raise RuntimeError('acceptance_differs_from_frozen_protocol')
        if getattr(profile, 'request_profile', None) in ('sdk-defaults-v1', 'selfhost-q6-v1'):
            from camroll_eval.common import sha256
            if (status.get('protocol_sha256') != sha256(protocol_path)
                    or protocol.get('annotation_review') != status.get('annotation_review')):
                raise RuntimeError('accepted_protocol_evidence_changed')
    elif getattr(profile, 'request_profile', None) in ('sdk-defaults-v1', 'selfhost-q6-v1'):
        raise RuntimeError('sdk_defaults_frozen_protocol_required')
    if annotation_review(runtime, profile) != status.get('annotation_review'):
        raise RuntimeError('independent_annotation_review_changed')
    if profile.provider == 'llamacpp-selfhost':
        from camroll_agent.minicpm_selfhost import pricing_evidence as _pricing_evidence
        from minicpm_selfhost_deployment import verify_service
        verify_service(runtime)
    pricing = _pricing_evidence(read_json(runtime / 'pricing-evidence.json'))
    accepted_pricing = status.get('pricing_evidence')
    # Every stage refreshes the account observation. Only top-level collection
    # timestamps may change; nested model timestamps/capabilities, raw tariff,
    # units, source, and all other evidence remain comparison facts.
    def facts(value):
        return {key: item for key, item in value.items() if key not in PRICING_CAPTURE_TIMESTAMPS}
    if not isinstance(accepted_pricing, dict) or facts(accepted_pricing) != facts(pricing):
        raise RuntimeError('pricing_evidence_differs_from_accepted_cloud_protocol')
    return source, status, validation, profile, pricing


def historical_baseline():
    """Bind the exact original completed F37-Off bytes, never a replacement run."""
    from camroll_eval.common import read_json, sha256
    from gate_suite import validate_run as validate_baseline
    suite_path = BASELINE_RUNTIME / 'formal-suite.json'
    baseline_suite = read_json(suite_path)
    matches = [row for row in baseline_suite.get('groups', []) if row.get('name') == 'F37-Off']
    if len(matches) != 1 or matches[0].get('run_id') != BASELINE_RUN_ID:
        raise RuntimeError('historical_f37_off_baseline_identity_changed')
    run_path = BASELINE_RUNTIME / 'evaluation/runs' / BASELINE_RUN_ID / 'run.json'
    if sha256(run_path) != BASELINE_RUN_SHA256:
        raise RuntimeError('historical_f37_off_baseline_hash_changed')
    run = read_json(run_path)
    if run.get('run_id') != BASELINE_RUN_ID or not validate_baseline(run):
        raise RuntimeError('complete_historical_f37_off_baseline_required')
    return {'path': str(run_path), 'sha256': BASELINE_RUN_SHA256}, {'path': str(suite_path), 'sha256': sha256(suite_path)}


def freeze_inputs(runtime, stage, acknowledgement, smoke_questions=None):
    from camroll_eval.common import atomic_json, read_json, now_iso, sha256, object_hash
    from camroll_eval.dataset import Dataset
    from camroll_eval.gate_experiment import IndexBinding
    from camroll_eval.judge import Judge
    from four_suite import source_manifest
    from four_gpu_window import verify_files, verify_model_files, _legacy_fingerprint
    from gate_continuation import verify_wemm_weights
    runtime = Path(runtime)
    source, status, validation, profile, pricing = acceptance_inputs(runtime, stage)
    if not RELEASE.name.startswith('minicpm-'):
        raise RuntimeError('formal_and_e2e_require_frozen_minicpm_release')
    original = HISTORY_RUNTIME / 'evaluation/runs' / SOURCE_RUN_ID / 'run.json'
    binding = IndexBinding.capture(ALBUMS / SOURCE_ALBUM_ID, original)
    integrity = Dataset(SOURCE).integrity()
    if integrity != read_json(original)['input_integrity']:
        raise RuntimeError('original_inputs_changed')
    baseline_run, baseline_suite = historical_baseline()
    if stage == 'e2e':
        from camroll_eval.minicpm_experiment import _SmokeDataset
        if smoke_questions is None:
            raise ValueError('E2E requires explicit independent smoke questions')
        smoke = _SmokeDataset(SOURCE, smoke_questions)
        smoke_record = {'path': str(smoke.plan_path), 'sha256': smoke.plan_sha256}
    else:
        if smoke_questions is not None:
            raise ValueError('Formal runs cannot use smoke questions')
        smoke_record = None
    files = {name: {'path': str(runtime / name), 'sha256': sha256(runtime / name)}
             for name in ('validation.json', 'acceptance/status.json', 'pricing-evidence.json')}
    for name in ('request-profile.json', 'acceptance/protocol.json', 'acceptance/annotation-review-sdk-defaults.json',
                 'acceptance/independent-review.json', 'deployment-manifest.json', 'connection.json'):
        if (runtime / name).exists():
            files[name] = {'path': str(runtime / name), 'sha256': sha256(runtime / name)}
    if stage == 'formal':
        for name in ('e2e-suite.json', 'e2e-freeze.json', 'evaluation/runs/' + status['e2e_run_id'] + '/run.json'):
            files[name] = {'path': str(runtime / name), 'sha256': sha256(runtime / name)}
    result = {'schema_version': 'minicpm-supervised-freeze-v1', 'created_at': now_iso(), 'stage': stage,
        'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'source_files': source,
        'source_run': {'path': str(original), 'sha256': sha256(original)}, 'index_binding': binding.as_dict(),
        'album_files': source_manifest(ALBUMS / SOURCE_ALBUM_ID), 'input_integrity': integrity,
        'snapshot': SNAPSHOT, 'source_album_id': SOURCE_ALBUM_ID, 'source_run_id': SOURCE_RUN_ID,
        'baseline_run': baseline_run, 'baseline_suite': baseline_suite,
        'inference_config': profile.as_dict(), 'inference_fingerprint': profile.fingerprint,
        'pricing_evidence': pricing, 'acceptance_files': files, 'smoke_questions': smoke_record,
        'accepted_pricing_evidence_sha256': object_hash(status['pricing_evidence']),
        'stage_pricing_evidence_sha256': object_hash(pricing),
        'acknowledgement': acknowledgement.as_dict(), 'protected_files': verify_files(),
        'model_files': verify_model_files(), 'wemm_weights': verify_wemm_weights(),
        'wemm_fingerprint': _legacy_fingerprint(), 'judge_protocol': Judge(expected_limit_cny=600).protocol(),
        'budget_limit_cny': 600, 'qa_count': 100 if stage == 'formal' else 3,
        'planned_turn_count': 111 if stage == 'formal' else 3, 'build_performed': False, 'seen_benchmark': True}
    destination = runtime / (stage + '-freeze.json')
    if destination.exists():
        raise RuntimeError('stage_freeze_already_exists_no_automatic_repeat')
    atomic_json(destination, result)
    return result


def verify_frozen(freeze):
    from camroll_eval.common import sha256
    from camroll_eval.dataset import Dataset
    from four_suite import source_manifest
    expected_baseline = BASELINE_RUNTIME / 'evaluation/runs' / BASELINE_RUN_ID / 'run.json'
    if (freeze.get('baseline_run', {}).get('path') != str(expected_baseline)
            or freeze.get('baseline_run', {}).get('sha256') != BASELINE_RUN_SHA256):
        raise RuntimeError('frozen_historical_baseline_identity_changed')
    if source_manifest(RELEASE) != freeze['source_files']:
        raise RuntimeError('frozen_source_changed')
    if source_manifest(ALBUMS / SOURCE_ALBUM_ID) != freeze['album_files']:
        raise RuntimeError('original_album_changed')
    if Dataset(SOURCE).integrity() != freeze['input_integrity']:
        raise RuntimeError('frozen_input_changed')
    records = [freeze['source_run'], freeze['baseline_run'], freeze['baseline_suite']]
    records.extend(freeze['acceptance_files'].values())
    if freeze.get('smoke_questions'):
        records.append(freeze['smoke_questions'])
    if any(sha256(row['path']) != row['sha256'] for row in records):
        raise RuntimeError('frozen_evidence_changed')


def run_child(runtime, stage, suite_id, run_id, ledger):
    from camroll_eval.common import read_json
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig, MiniCPMRunner
    from four_gpu_window import GpuWindow
    runtime = Path(runtime)
    suite = read_json(runtime / (stage + '-suite.json'))
    freeze = read_json(runtime / (stage + '-freeze.json'))
    if (suite.get('suite_id') != suite_id or suite.get('status') != 'running'
            or suite.get('run_id') != run_id or suite.get('stage') != stage or freeze.get('stage') != stage):
        raise RuntimeError('child_identity_does_not_match_frozen_suite')
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)._read()
    if window.get('state') != 'active' or window.get('current_kind') != 'wemm':
        raise RuntimeError('child_requires_active_supervised_wemm_window')
    ledger_ready(ledger)
    verify_frozen(freeze)
    questions = freeze['smoke_questions']['path'] if stage == 'e2e' else None
    return MiniCPMRunner(runtime / 'evaluation', SOURCE, ALBUMS, budget_ledger=ledger,
        inference=MiniCPMInferenceConfig(**freeze['inference_config']), pricing_evidence=freeze['pricing_evidence'],
        sample_resources=True).run_minicpm(run_id=run_id, suite_id=suite_id, source_album_id=SOURCE_ALBUM_ID,
        source_run_id=SOURCE_RUN_ID, source_evaluation_root=HISTORY_RUNTIME / 'evaluation', smoke_questions=questions)


def execute(runtime, ledger, acknowledgement, *, stage='formal', smoke_questions=None):
    from camroll_eval.common import atomic_json, read_json, now_iso
    from four_gpu_window import GpuWindow, verify_model_files
    from four_supplement import stop_owned_child
    from gate_continuation import verify_wemm_weights
    runtime = validate_runtime(runtime)
    if stage not in ('e2e', 'formal'):
        raise ValueError('Unknown MiniCPM stage')
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    (runtime / (stage + '-suite.claim')).mkdir(mode=0o700)
    prefix = 'minicpm47-q6-thinking-m4-off-' if runtime.name.startswith('minicpm47-q6-thinking-m4-off-') else SDK_RUNTIME_PREFIX if runtime.name.startswith(SDK_RUNTIME_PREFIX) else RUN_PREFIX
    suite_id = prefix + stage + '-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run_id = suite_id + '-run'
    state = {'schema_version': 'minicpm-supervised-suite-v1', 'suite_id': suite_id, 'stage': stage,
        'run_id': run_id, 'status': 'preparing', 'started_at': now_iso(), 'budget_before': ledger.summary(),
        'source_revision': os.environ['CAMROLL_RELEASE_REVISION'], 'build_performed': False,
        'acknowledgement': acknowledgement.as_dict()}
    suite_path = runtime / (stage + '-suite.json')
    def save():
        state['updated_at'] = now_iso()
        atomic_json(suite_path, state)
    save()
    window = GpuWindow(RELEASE, suite_id, runtime=HISTORY_RUNTIME)
    active, freeze = None, None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('suite_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        ledger_ready(ledger)
        freeze = freeze_inputs(runtime, stage, acknowledgement, smoke_questions)
        entered = window.enter()
        state['window_state_path'] = str(window.state_path)
        if entered['before']['fingerprint'] != freeze['wemm_fingerprint']:
            raise RuntimeError('window_fingerprint_differs_from_freeze')
        window.switch('wemm')
        verify_frozen(freeze)
        ledger_ready(ledger)
        state['status'] = 'running'
        save()
        command = [str(PYTHON), str(Path(__file__).resolve()), '--runtime', str(runtime), '--stage', stage,
                   '--execute-paid', '--child', '--run-id', run_id, '--suite-id', suite_id]
        log_path = runtime / 'logs' / (run_id + '.log')
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with log_path.open('xb') as log:
            log_path.chmod(0o600)
            active = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            window.register_child(active.pid, expected_command=command)
            stopped_at = None
            while active.poll() is None:
                time.sleep(5)
                reason = None
                try:
                    ledger.check_limit()
                    if ledger.summary().get('blocking_uncertain_requests', 1):
                        reason = 'new_uncertain_cloud_request_stop'
                except Exception as exc:
                    reason = getattr(exc, 'code', 'budget_check_failed')
                if reason:
                    marker = runtime / 'evaluation/runs' / run_id / 'cancel.requested'
                    if marker.parent.exists() and not marker.exists():
                        marker.write_text(reason)
                        marker.chmod(0o600)
                    if stopped_at is None:
                        stopped_at = time.monotonic()
                        state['new_submissions_blocked'] = reason
                        save()
                if stopped_at is not None and time.monotonic() - stopped_at >= freeze["inference_config"]["deadline_s"]:
                    raise RuntimeError('blocked_child_did_not_exit_within_turn_deadline')
            state['returncode'] = active.returncode
            window.unregister_child(active.pid)
            active = None
        run_path = runtime / 'evaluation/runs' / run_id / 'run.json'
        run = read_json(run_path) if run_path.exists() else {}
        state.update(run_status=run.get('status', 'failed'), summary=run.get('summary'),
            acceptance_complete=state['returncode'] == 0 and validate_run(run, stage=stage))
        ledger_ready(ledger)
        verify_frozen(freeze)
        state['status'] = 'passed' if state['acceptance_complete'] else 'incomplete'
    except BaseException as exc:
        state.update(status='blocked' if isinstance(exc, Exception) else 'interrupted',
            error_type=type(exc).__name__, error_code=getattr(exc, 'code', None),
            error=str(exc) if isinstance(exc, (RuntimeError, ValueError, KeyboardInterrupt)) else type(exc).__name__)
    finally:
        try:
            try:
                state['child_cleanup'] = stop_owned_child(active)
            except BaseException as exc:
                state['child_cleanup'] = {'state': 'failed', 'error_type': type(exc).__name__}
            finally:
                state['restoration'] = (window.restore('minicpm_' + stage + '_' + state['status'])
                    if window.state_path.exists() else {'state': 'not_entered'})
            if freeze:
                if verify_model_files()['models'] != freeze['model_files']['models']:
                    raise RuntimeError('protected_model_files_changed')
                weights = verify_wemm_weights()
                if (weights['actual_sha256'] != freeze['wemm_weights']['actual_sha256']
                        or weights['file_identity_before'] != freeze['wemm_weights']['file_identity_after']):
                    raise RuntimeError('wemm_weights_changed')
                verify_frozen(freeze)
                state.update(model_files_unchanged=True, wemm_weights_unchanged=True, frozen_inputs_unchanged=True)
            if state['status'] == 'passed' and state['restoration'].get('state') != 'restored':
                raise RuntimeError('restoration_not_verified')
        except BaseException as exc:
            state.update(status='restore_failed', restoration_error=type(exc).__name__)
        try:
            state['budget_after'] = ledger.summary()
        except Exception as exc:
            state.update(status='incomplete' if state['status'] == 'passed' else state['status'], budget_audit_error=type(exc).__name__)
        state['finished_at'] = now_iso()
        save()
    if state['status'] == 'passed' and stage == 'e2e':
        from camroll_eval.common import sha256
        acceptance = read_json(runtime / 'acceptance/status.json')
        acceptance.update(e2e_status='passed', e2e_run_id=run_id, e2e_suite_id=suite_id, e2e_finished_at=state['finished_at'],
            e2e_evidence={name: sha256(path) for name, path in e2e_evidence_paths(runtime, run_id).items()})
        atomic_json(runtime / 'acceptance/status.json', acceptance)
    return 0 if state['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--stage', choices=('e2e', 'formal'), default='formal')
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--acknowledgement', type=Path)
    parser.add_argument('--smoke-questions', type=Path)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--suite-id')
    args = parser.parse_args(argv)
    runtime = validate_runtime(args.runtime)
    if not args.execute_paid:
        print(json.dumps({'status': 'not_started', 'runtime': str(runtime), 'stage': args.stage}))
        return 0
    bootstrap(runtime)
    from camroll_agent.budget import BudgetLedger, UncertainRequestAcknowledgement
    from camroll_eval.common import read_json
    data = (read_json(runtime / (args.stage + '-freeze.json'))['acknowledgement'] if args.child
            else read_json(args.acknowledgement) if args.acknowledgement else None)
    if data is None:
        parser.error('--acknowledgement is required for this execution')
    ack = UncertainRequestAcknowledgement.from_dict(data)
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600, acknowledged_uncertain=(ack,))
    if args.child:
        result = run_child(runtime, args.stage, args.suite_id, args.run_id, ledger)
        return 0 if validate_run(result, stage=args.stage) else 2
    return execute(runtime, ledger, ack, stage=args.stage, smoke_questions=args.smoke_questions)


if __name__ == '__main__':
    raise SystemExit(main())
