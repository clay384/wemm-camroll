"""One-shot, accounted MiniCPM capability selection using a nonformal photo.

This command does not acquire GPU ownership or launch an evaluation. Failed
capability probes are retained and never silently retried as a fresh batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

RELEASE = Path(__file__).resolve().parents[1]


def initialize_acknowledgement(runtime):
    from camroll_agent.budget import UncertainRequestAcknowledgement
    from camroll_eval.common import atomic_json, now_iso
    from minicpm_config import BASE, BASELINE_RUNTIME
    ack = UncertainRequestAcknowledgement.capture(
        BASE / 'runtime/budget.sqlite',
        BASELINE_RUNTIME / 'evaluation/runs/f37-on-gate-20261006-181014-8bd9a5/request-journal.sqlite',
        call_id='011a8df1906847719e30c959cfaec785',
        request_id='924265cfa3c1cd9e9a3477e3580c89a243cb8bd8db40eaa3fe3ebfd1d0faf367:0',
        payload_hash='548aae2ffd16ff81261eb8b2107b3caad0d51db306986c6e79dafc82421298bf',
        reserved_units=1268122)
    # The complete captured rows are checked by the ledger on every reserve;
    # retain them under this explicit new authorization, never reconcile them.
    if ack.ledger_row_sha256 != 'cf8dd6ceee70e4d1f59f677907482d8bf5db0994501b639f30f523dd0acfed20':
        raise RuntimeError('original_uncertain_ledger_row_changed')
    if ack.journal_row_sha256 != '8d299b765700b107235638e3e8bc8208056469f29dad9b60d992d2e7d159e3b3':
        raise RuntimeError('original_uncertain_journal_row_changed')
    target = Path(runtime) / 'acknowledgement.json'
    if target.exists():
        if json.loads(target.read_text()) != ack.as_dict():
            raise RuntimeError('acknowledgement_changed')
    else:
        atomic_json(target, ack.as_dict())
        atomic_json(Path(runtime) / 'authorization.json', {
            'observed_at': now_iso(), 'scope': Path(runtime).name,
            'source': 'User approved MiniCPM implementation plan, 2026-10-14',
            'acknowledged_call_id': ack.call_id, 'reservation_retained_units': 1268122,
            'replay_authorized': False, 'new_unknown_requests_block': True})
    return ack


def probe_roundtrip(client, record):
    from camroll_agent.agent import FINAL_SCHEMA, _assistant_envelope, _ID_RE
    from camroll_agent.schemas import TOOLS, validate_arguments, validate_schema
    from camroll_agent.prompts import public_evidence
    from camroll_agent.modelbest import _strict_json
    iid = record['id']
    messages = [{'role': 'system', 'content': 'Use tools when requested. After a tool result return only the final JSON object, no Markdown. Do not infer identities.'},
        {'role': 'user', 'content': 'Compatibility exercise, unrelated to formal QA. First call only get for this image ID: ' + iid +
         '. After its result summarize the visible record and deliver only this image. Final JSON schema: ' + json.dumps(FINAL_SCHEMA)}]
    first = _assistant_envelope(client.chat(messages, tools=TOOLS, request_id='decision'))
    calls = first.get('tool_calls') or []
    if len(calls) != 1 or calls[0].get('function', {}).get('name') != 'get':
        raise ValueError('expected_one_get_tool_call')
    arguments = _strict_json(calls[0]['function']['arguments'])
    validate_arguments('get', arguments)
    if arguments.get('id') != iid or not calls[0].get('id'):
        raise ValueError('invalid_tool_identity')
    messages.append({k: first[k] for k in ('role', 'content', 'tool_calls') if k in first})
    messages.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'name': 'get',
                     'content': json.dumps(public_evidence([record]), ensure_ascii=False)})
    final = _assistant_envelope(client.chat(messages, tools=TOOLS, request_id='final'))
    if final.get('tool_calls'):
        raise ValueError('unexpected_extra_tool_call')
    parsed = _strict_json(final.get('content'))
    validate_schema(parsed, FINAL_SCHEMA, 'final')
    if parsed['selected_image_ids'] != [iid] or parsed['outcome'] != 'answer':
        raise ValueError('final_delivery_mismatch')
    if set(_ID_RE.findall(parsed['answer'])) - {iid}:
        raise ValueError('unknown_final_citation')
    return {'tool_call': calls[0], 'final': parsed}


def execute(runtime):
    from minicpm_config import (configure, BASE, request_profile, request_matches_profile, annotation_review,
                               acceptance_policy, INTERFACE_ONLY_POLICY)
    configure(RELEASE, runtime)
    from camroll_agent.budget import BudgetLedger, RequestUncertain
    from camroll_agent.modelbest import ModelBestClient
    from camroll_agent.library import Library
    from camroll_eval.common import atomic_json, read_json, now_iso
    from minicpm_config import ALBUMS, SOURCE_ALBUM_ID, SNAPSHOT
    from four_suite import source_manifest
    from gate_continuation import ledger_ready
    from camroll_eval.minicpm_experiment import client_class, MiniCPMInferenceConfig

    requested_profile = request_profile(runtime)
    sdk_defaults = getattr(requested_profile, 'request_profile', None) == 'sdk-defaults-v1'
    policy = acceptance_policy(requested_profile)
    if requested_profile.provider == "llamacpp-selfhost":
        from minicpm_selfhost_deployment import verify_service
        verify_service(runtime)

    folder = Path(runtime) / 'acceptance'
    folder.mkdir(exist_ok=True)
    with (folder / 'probe-claim.json').open('x') as stream:
        json.dump({'started_at': now_iso(), 'source_files': source_manifest(RELEASE)}, stream)
    acknowledgement = initialize_acknowledgement(runtime)
    ledger = BudgetLedger(BASE / 'runtime/budget.sqlite', limit_cny=600,
        expected_limit_cny=600, acknowledged_uncertain=(acknowledgement,))
    ledger_ready(ledger)
    pricing = read_json(Path(runtime) / 'pricing-evidence.json')
    fixture = read_json(folder / 'fixture-records.json')
    library = Library(ALBUMS / SOURCE_ALBUM_ID, config='M4')
    if library.version != SNAPSHOT:
        raise ValueError('snapshot_changed')
    record = fixture[6]
    path = library.image_paths([record['id']])[0]
    if library.get(record['id']) != record:
        raise ValueError('fixture_record_changed')
    reviewed = annotation_review(runtime, requested_profile,
        image_paths=library.image_paths([row['id'] for row in fixture]) if sdk_defaults else None)
    results = {'status': 'running', 'started_at': now_iso(), 'probes': [], 'pricing_evidence': pricing,
               'requested_inference_config': requested_profile.as_dict(),
               'annotation_review': reviewed,
               'acceptance_policy': policy, 'fact_observations': [],
               'purpose': 'single_selected_mode_client_integration' if sdk_defaults else 'capability_selection',
               'source_files': source_manifest(RELEASE), 'snapshot': SNAPSHOT}
    atomic_json(folder / 'probe-results.json', results)
    calls_file = folder / 'probe-calls.jsonl'

    def client_for(mode, json_mode, label):
        profile = MiniCPMInferenceConfig(**{**requested_profile.as_dict(), 'decision_mode': mode, 'json_mode': json_mode})
        return client_class(profile)(ledger=ledger, pricing_evidence=pricing,
            **profile.client_options(),
            request_namespace=Path(runtime).name + ':probe:' + label,
            request_journal_path=folder / 'probe-journal.sqlite')

    def run(label, client, action):
        row = {'name': label, 'started_at': now_iso(), 'calls': []}
        def capture(call):
            row['calls'].append(call)
            with calls_file.open('a') as stream:
                stream.write(json.dumps({'probe': label, **call}, ensure_ascii=False) + '\n')
        try:
            with client.capture_calls(capture), client.request_scope(max_requests=50, deadline_s=requested_profile.deadline_s):
                row['result'] = action()
            if any(not request_matches_profile(requested_profile, call.get('input'))
                   for call in row['calls']):
                raise RuntimeError('probe_request_contract_invalid')
            row['status'] = 'passed'
        except RequestUncertain:
            row.update(status='blocked', error='request_uncertain')
            raise
        except Exception as exc:
            row.update(status='failed', error=getattr(exc, 'code', type(exc).__name__))
            if row['error'] not in ('invalid_json', 'invalid_message', 'empty_response', 'http_400', 'http_422', 'ValueError', 'ToolValidationError'):
                raise
        finally:
            row['finished_at'] = now_iso()
            results['probes'].append(row)
            atomic_json(folder / 'probe-results.json', results)
        ledger_ready(ledger)
        return row['status'] == 'passed'

    try:
        schema = {'type': 'object', 'additionalProperties': False,
                  'properties': {'inscription': {'type': 'string'}, 'letter_color': {'type': 'string'}},
                  'required': ['inscription', 'letter_color']}
        chosen_json = None
        for mode in ((requested_profile.json_mode,) if sdk_defaults else ('json_schema', 'json_object', 'prompt')):
            client = client_for(requested_profile.decision_mode if sdk_defaults else 'native_tools', mode, mode)
            def read_inscription():
                result = client.generate_json(
                    '读取石柱中央竖排文字，原样写入inscription；letter_color使用中文描述字的颜色。不要猜测不可见信息。',
                    [path], schema, role='view', max_attempts=1)
                observation = inscription_observation(result)
                results['fact_observations'].append({'probe': mode, **observation,
                    'blocks_acceptance': sdk_defaults and policy != INTERFACE_ONLY_POLICY})
                if policy != INTERFACE_ONLY_POLICY:
                    validate_inscription(result, sdk_defaults=sdk_defaults)
                return result
            if run(mode, client, read_inscription):
                chosen_json = mode
                break
        if chosen_json is None:
            raise RuntimeError('no_structured_output_mode_passed')
        chosen_tools = None
        for mode in ((requested_profile.decision_mode,) if sdk_defaults else ('native_tools', 'json_decision')):
            client = client_for(mode, chosen_json, mode)
            if run(mode, client, lambda: probe_roundtrip(client, record)):
                chosen_tools = mode
                break
        if chosen_tools is None:
            raise RuntimeError('no_tool_roundtrip_passed')
        profile = MiniCPMInferenceConfig(**{**requested_profile.as_dict(), 'decision_mode': chosen_tools, 'json_mode': chosen_json})
        protocol = {'decision_mode': chosen_tools, 'json_mode': chosen_json,
                    'pricing_evidence': pricing, 'inference_config': profile.as_dict(),
                    'annotation_review': reviewed,
                    'protocol_fingerprint': profile.fingerprint, 'frozen_at': now_iso()}
        if annotation_review(runtime, profile) != reviewed:
            raise RuntimeError('annotation_review_changed_during_probe')
        atomic_json(folder / 'protocol.json', protocol)
        results.update(status='passed', protocol=protocol)
    except Exception as exc:
        results.update(status='blocked', reason=getattr(exc, 'code', type(exc).__name__))
        raise
    finally:
        results.update(finished_at=now_iso(), budget=ledger.summary())
        atomic_json(folder / 'probe-results.json', results)
        atomic_json(folder / 'status.json', {'status': 'capability_passed' if results['status'] == 'passed' else 'blocked',
            'reason': results.get('reason', 'Frozen 30-session image acceptance is pending'),
            'updated_at': now_iso(), 'stages': {'capability': results['status'], 'cloud_images': 'not_started', 'e2e': 'not_started'}})
    return results


def validate_inscription(result, *, sdk_defaults):
    # The old capability probe only validated structure. Its recorded behavior
    # remains unchanged. New independently reviewed annotation preserves 壩.
    if sdk_defaults and not all(inscription_observation(result).values()):
        raise ValueError('preannotated_inscription_or_color_mismatch')


def inscription_observation(result):
    return {'inscription_correct': result.get('inscription', '').strip() == '三峡壩址基石',
        'letter_color_correct': any(word in str(result.get('letter_color', '')).lower() for word in ('红', '紅', 'red'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not args.execute_paid:
        parser.error('--execute-paid is required for explicitly authorized cloud calls')
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
    from minicpm_config import validate_runtime
    result = execute(validate_runtime(args.runtime))
    print(json.dumps({'status': result['status'], 'probe_count': len(result['probes']), 'protocol': result.get('protocol', {}).get('inference_config')}))


if __name__ == '__main__':
    main()
