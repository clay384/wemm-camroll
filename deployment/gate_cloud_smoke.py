"""Paid synthetic acceptance for Flash Gate; never submits formal album3 questions."""
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

from gate_config import HISTORY_RUNTIME, MODELS, ORDER, PYTHON, RUNTIME, configure

RELEASE = Path(__file__).resolve().parents[1]


def fixtures():
    from PIL import Image, ImageDraw
    from camroll_eval.common import atomic_json
    root = RUNTIME / 'fixtures/synthetic-flash-gate-v1'
    if root.exists():
        raise RuntimeError('synthetic_fixture_exists_inspect_before_repeat')
    (root / 'photos').mkdir(parents=True, mode=0o700)
    for name, color in (('red', 'red'), ('blue', 'blue')):
        picture = Image.new('RGB', (320, 240), 'white')
        draw = ImageDraw.Draw(picture)
        if name == 'red':
            draw.rectangle((70, 30, 250, 210), fill=color)
        else:
            draw.ellipse((70, 30, 250, 210), fill=color)
        picture.save(root / 'photos' / (name + '.jpg'))
    rows = [{'qa_id': 'synthetic-red', 'question': '请找到红色正方形照片，查看原图确认颜色和形状，交付图片。',
             'answer': '红色正方形照片 red.jpg。', 'retrieval_image_ids': ['red.jpg'],
             'answerability': 'answerable', 'expected_action': 'answer'},
            {'qa_id': 'synthetic-blue', 'question': '请找到蓝色圆形照片，查看原图确认并交付。',
             'answer': '蓝色圆形照片 blue.jpg。', 'retrieval_image_ids': ['blue.jpg'],
             'answerability': 'answerable', 'expected_action': 'answer'},
            {'qa_id': 'synthetic-missing', 'question': '这些图片中哪张是黄色汽车？不能确定时请明确说明。',
             'answer': '没有找到黄色汽车的照片，无法提供。', 'retrieval_image_ids': [],
             'answerability': 'unanswerable', 'expected_action': 'refuse'}]
    (root / 'questions.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n')
    atomic_json(root / 'manifest.json', {'album_id': 'synthetic-flash-gate-v1',
        'photos': ['photos/red.jpg', 'photos/blue.jpg'], 'faces': [], 'family_relationships': [],
        'qa_sets': {'100qa-full': 'questions.jsonl'}})
    return root


def run_checks(smoke_id):
    from camroll_agent.budget import BudgetLedger
    from camroll_agent.qwen import QwenClient, image_messages
    from camroll_agent.evidence_gate import EvidenceGateVerifier
    from camroll_eval.common import append_jsonl, atomic_json, now_iso
    from camroll_eval.judge import Judge
    from camroll_eval.runner import Runner, CamrollBackend
    from camroll_eval.gate_experiment import GateRunner
    from four_suite import source_manifest
    from gate_suite import ledger_ready
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600)
    report = {'schema_version': 'camroll-flash-gate-smoke-v1', 'source': 'synthetic_nonformal',
              'started_at': now_iso(), 'status': 'running', 'before': ledger_ready(ledger),
              'models': {}, 'groups': [], 'smoke_id': smoke_id}
    destination = RUNTIME / 'cloud-smoke.json'
    def save():
        atomic_json(destination, report)
    save()
    try:
        source = fixtures()
        journal = RUNTIME / 'smoke-request-journal.sqlite'
        calls = RUNTIME / 'smoke-model-calls.jsonl'
        schema = {'type': 'object', 'properties': {'color': {'type': 'string', 'enum': ['red']}},
                  'required': ['color'], 'additionalProperties': False}
        tool = {'type': 'function', 'function': {'name': 'record_color', 'description': 'Record a color', 'parameters': schema}}
        clients = {}
        for model in dict.fromkeys(MODELS.values()):
            ledger_ready(ledger)
            client = QwenClient(model=model, strict_response_model=True, expected_limit_cny=600,
                                request_namespace=smoke_id + ':' + model, request_journal_path=journal)
            clients[model] = client
            result = {'checks': {}, 'status': 'running'}
            report['models'][model] = result
            save()
            with client.capture_calls(lambda event: append_jsonl(calls, event)):
                response = client.chat([{'role': 'user', 'content': 'Reply exactly OK.'}], role='probe',
                                       max_tokens=64, request_id='text')
                result['checks']['text'] = {'passed': (response.get('content') or '').strip() == 'OK', 'usage': response['_usage']}
                response = client.chat(image_messages('Name the colored shape in this picture.', [source / 'photos/red.jpg']),
                                       role='probe', max_tokens=128, request_id='image')
                content = (response.get('content') or '').lower()
                result['checks']['image'] = {'passed': 'red' in content and 'square' in content, 'usage': response['_usage']}
                response = client.chat([{'role': 'user', 'content': 'Call record_color with color red.'}], [tool],
                    tool_choice={'type': 'function', 'function': {'name': 'record_color'}}, role='probe', max_tokens=128, request_id='tool')
                result['checks']['tool'] = {'passed': any(row.get('function', {}).get('name') == 'record_color'
                    and json.loads(row['function']['arguments']) == {'color': 'red'} for row in response.get('tool_calls', [])), 'usage': response['_usage']}
                with client.request_context('structured'):
                    output = client.generate_json('Return the color red.', [], schema, role='probe', max_attempts=1)
                result['checks']['structured'] = {'passed': output == {'color': 'red'}}
            result['status'] = 'passed' if all(row['passed'] for row in result['checks'].values()) else 'failed'
            save()
            if result['status'] != 'passed':
                raise RuntimeError('flash_capability_smoke_failed')
        verifier = EvidenceGateVerifier(client=clients['qwen3.7-flash'],
            on_call=lambda event: append_jsonl(RUNTIME / 'smoke-verifier-calls.jsonl', event))
        with clients['qwen3.7-flash'].request_scope(deadline_s=600, max_requests=50):
            verdict = verifier.verify(question='Describe the colored shape in each supplied image.',
                candidate_answer='img-red shows a red square; img-blue shows a blue circle.', outcome='answer',
                images=[{'image_id': 'img-red', 'path': source / 'photos/red.jpg'},
                        {'image_id': 'img-blue', 'path': source / 'photos/blue.jpg'}],
                delivery_image_ids=['img-red', 'img-blue'],
                trusted_metadata=[], derived_observations=[], request_id=smoke_id + ':two-images',
                deadline=time.monotonic() + 600, request_budget_hook=clients['qwen3.7-flash']._gate_request_budget_hook)
        report['verifier'] = {'status': 'passed' if verdict['status'] == 'supported' else 'failed', 'verdict': verdict}
        judge = Judge(expected_limit_cny=600, request_namespace=smoke_id + ':judge', request_journal_path=journal,
                      on_call=lambda event: append_jsonl(RUNTIME / 'smoke-judge-calls.jsonl', event))
        with judge.request_context('arithmetic'):
            quality = judge._judge('2+2等于几？', '4', '4', expected_action='answer')
        report['judge'] = {'status': 'passed' if quality.get('score') == 2 else 'failed', 'protocol': judge.protocol()}
        save()
        if report['verifier']['status'] != 'passed' or report['judge']['status'] != 'passed':
            raise RuntimeError('doubao_capability_smoke_failed')
        class SyntheticRunner(Runner):
            def _save(self):
                self._state['synthetic_nonformal'] = True
                return super()._save()
        class SyntheticBackend(CamrollBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.qwen.strict_response_model = True
        evaluation, albums = RUNTIME / 'smoke-evaluation', RUNTIME / 'smoke-albums'
        parent = SyntheticRunner(evaluation, source, albums, backend_factory=SyntheticBackend).run(run_id=smoke_id + '-build', experiment='W37',
            config='M4', qa_concurrency=1, judge_concurrency=4, strict=False)
        report['synthetic_build'] = {'run_id': parent['run_id'], 'status': parent['status'], 'snapshot': parent.get('index_version')}
        save()
        if parent['status'] not in ('completed', 'completed_with_errors') or not parent.get('index_version'):
            raise RuntimeError('synthetic_build_failed')
        for group in ORDER:
            ledger_ready(ledger)
            run = GateRunner(evaluation, source, albums).run_gate(run_id=smoke_id + '-' + group.lower(),
                suite_id=smoke_id, group=group, source_album_id=parent['album_id'], source_run_id=parent['run_id'],
                expected_snapshot=parent['index_version'], strict=False)
            calls = [call for item in run['items'] for call in item.get('model_call_metrics', [])]
            roles = {call.get('original_call_type') or call.get('call_type') for call in calls if call.get('status') in ('ok', 'success', 'completed')}
            valid = (bool(run.get('gate_complete')) and {'view', 'final_review'} <= roles
                     and all(turn.get('answer_origin') == 'model' for item in run['items'] for turn in item.get('runtime_turns', [])))
            report['groups'].append({'name': group, 'run_id': run['run_id'], 'status': 'passed' if valid else 'failed',
                                     'roles': sorted(str(role) for role in roles), 'summary': run['summary']})
            save()
            if not valid:
                raise RuntimeError('real_backend_library_view_final_chain_failed_' + group)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='blocked', error_type=type(exc).__name__, error_code=getattr(exc, 'code', None),
                      reason=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)
    finally:
        report.update(finished_at=now_iso(), after=ledger.summary(), source_files=source_manifest(RELEASE))
        save()
    return report['status'] == 'passed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--smoke-id')
    args = parser.parse_args(argv)
    configure(RELEASE)
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))
    if not args.execute_paid:
        raise SystemExit('Supply --execute-paid to run the approved synthetic checks within the existing 600 CNY cap.')
    if args.child:
        return 0 if run_checks(args.smoke_id) else 2
    from camroll_eval.common import atomic_json, read_json
    from camroll_agent.budget import BudgetLedger
    from four_gpu_window import GpuWindow
    from four_supplement import stop_owned_child
    from four_suite import source_manifest
    from gate_suite import ledger_ready
    ledger_ready(BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600))
    validation = read_json(RUNTIME / 'validation.json')
    if validation.get('status') != 'passed' or validation.get('source_files') != source_manifest(RELEASE):
        raise RuntimeError('matching_offline_validation_required_before_paid_smoke')
    smoke_id = 'gate-smoke-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    if (RUNTIME / 'cloud-smoke.json').exists():
        raise RuntimeError('smoke_already_recorded_inspect_before_repeat')
    (RUNTIME / 'smoke.claim').mkdir(parents=True, mode=0o700)
    window = GpuWindow(RELEASE, smoke_id, runtime=HISTORY_RUNTIME)
    child, code = None, 2
    def interrupted(signum, frame):
        raise KeyboardInterrupt('smoke_signal_' + str(signum))
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        window.enter()
        window.switch('wemm')
        command = [str(PYTHON), str(Path(__file__).resolve()), '--execute-paid', '--child', '--smoke-id', smoke_id]
        with (RUNTIME / 'cloud-smoke.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            window.register_child(child.pid, expected_command=command)
            code = child.wait()
            window.unregister_child(child.pid)
    finally:
        cleanup, restoration = None, {'state': 'not_entered'}
        try:
            try:
                cleanup = stop_owned_child(child)
            except BaseException as exc:
                cleanup = {'state': 'failed', 'error_type': type(exc).__name__}
            finally:
                if window.state_path.exists():
                    restoration = window.restore('gate_smoke_exit')
        except BaseException as exc:
            restoration = {'state': 'restore_failed', 'error_type': type(exc).__name__}
            code = 2
        finally:
            atomic_json(RUNTIME / 'smoke-restoration.json', restoration)
            path = RUNTIME / 'cloud-smoke.json'
            report = read_json(path) if path.exists() else {'status': 'blocked', 'reason': 'child_did_not_write_report'}
            report.update(restoration=restoration, child_cleanup=cleanup)
            if restoration.get('state') != 'restored':
                report['status'] = 'blocked'
                code = 2
            atomic_json(path, report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
