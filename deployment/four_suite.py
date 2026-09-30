"""Supervised, sequential cold runs; never retries or repairs formal answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from four_config import RUNTIME, SOURCE, PYTHON, configure

RELEASE = Path(__file__).resolve().parents[1]
ORDER = ('Q37', 'Q38', 'W37', 'W38Max')


def bootstrap():
    configure(RELEASE)
    sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))


def source_manifest(root):
    rows = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(part in {'.git', '__pycache__', '.pytest_cache', '.venv'} for part in path.relative_to(root).parts) or path.name == '.DS_Store':
            continue
        rows[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def synthetic_source():
    from PIL import Image, ImageDraw
    from camroll_eval.common import atomic_json
    root = RUNTIME / 'fixtures' / 'full-chain-v1'
    if (root / 'manifest.json').exists():
        return root
    (root / 'photos').mkdir(parents=True, exist_ok=True)
    for color, shape in (('red', 'square'), ('blue', 'circle'), ('green', 'triangle')):
        picture = Image.new('RGB', (320, 240), 'white')
        draw = ImageDraw.Draw(picture)
        if shape == 'square':
            draw.rectangle((70, 30, 250, 210), fill=color)
        elif shape == 'circle':
            draw.ellipse((70, 30, 250, 210), fill=color)
        else:
            draw.polygon(((160, 30), (250, 210), (70, 210)), fill=color)
        picture.save(root / 'photos' / (color + '.jpg'))
    rows = [
        {'qa_id': 'red', 'question': '请找出红色正方形的照片。', 'answer': '红色正方形照片 red.jpg。', 'retrieval_image_ids': ['red.jpg'], 'answerability': 'answerable', 'expected_action': 'answer'},
        {'qa_id': 'blue', 'question': '请找蓝色圆形。', 'answer': '蓝色圆形照片 blue.jpg。', 'retrieval_image_ids': ['blue.jpg'], 'answerability': 'answerable', 'expected_action': 'answer', 'conversation': [
            {'message': '请找蓝色圆形。', 'expected_action': 'answer', 'reference_answer': '蓝色圆形照片 blue.jpg。'},
            {'message': '它是什么颜色的？请再提供这张图片。', 'expected_action': 'answer', 'reference_answer': '蓝色，blue.jpg。'}]},
        {'qa_id': 'missing', 'question': '相册里有黄色汽车吗？', 'answer': '没有黄色汽车的照片，无法提供。', 'retrieval_image_ids': [], 'answerability': 'unanswerable', 'expected_action': 'refuse'},
    ]
    (root / 'questions.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n')
    atomic_json(root / 'manifest.json', {'album_id': 'synthetic-full-chain-v1', 'photos': ['photos/red.jpg', 'photos/blue.jpg', 'photos/green.jpg'], 'faces': [], 'family_relationships': [], 'qa_sets': {'100qa-full': 'questions.jsonl'}})
    return root


def validate_run(state, *, formal):
    expected_questions, expected_turns, expected_images = (100, 111, 355) if formal else (3, 4, 3)
    if state.get('status') not in {'completed', 'completed_with_errors'}:
        return False
    items = state.get('items', [])
    plan = {row['qa_id']: row['planned_turn_count'] for row in state.get('qa_plan', [])}
    if len(plan) != expected_questions or {row.get('qa_id') for row in items} != set(plan):
        return False
    # Wrong answers are valid measurements; missing turns or scores are incomplete.
    return (len(items) == expected_questions and sum(len(row.get('runtime_turns', [])) for row in items) == expected_turns
            and len(state.get('media_mapping', {})) == expected_images
            and all(row.get('judge_status') == 'completed' and (row.get('judge') or {}).get('score') in (0, 1, 2)
                    and len(row.get('runtime_turns', [])) == plan[row['qa_id']]
                    and all((turn.get('judge') or {}).get('score') in (0, 1, 2)
                            and (turn.get('task_judge') or {}).get('actual_action') in {'answer', 'refuse', 'clarify', 'none'}
                            for turn in row.get('runtime_turns', [])) for row in items))


def freeze_inputs(destination):
    from camroll_agent.experiment import PROFILES
    from camroll_eval.common import atomic_json, read_json, now_iso
    from camroll_eval.dataset import Dataset
    from camroll_eval.judge import Judge
    from four_gpu_window import MODEL_ROOT, MODEL_MANIFEST_SHA256, _legacy_fingerprint
    gates = {name: read_json(RUNTIME / name) for name in ('cloud-smoke.json', 'gpu-acceptance.json', 'smoke-suite.json', 'validation.json')}
    if any(gate.get('status') != 'passed' for gate in gates.values()):
        raise RuntimeError('formal_acceptance_gate_not_passed')
    if not RELEASE.name.startswith('four-'):
        raise RuntimeError('formal_runs_require_frozen_four_release')
    manifest_path = MODEL_ROOT / 'MODEL_WORKER_DOWNLOAD_MANIFEST.json'
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MODEL_MANIFEST_SHA256:
        raise RuntimeError('frozen_model_manifest_changed')
    model_manifest = json.loads(manifest_bytes)
    model_files = {name: {item['path']: {'sha256': item['sha256'], 'size': item['size']}
                         for item in model_manifest['models']['Qwen/' + name]['files']}
                   for name in ('Qwen3-VL-Embedding-2B', 'Qwen3-VL-Reranker-2B')}
    result = {'created_at': now_iso(), 'source_revision': os.environ['CAMROLL_RELEASE_REVISION'],
              'source_files': source_manifest(RELEASE), 'input_integrity': Dataset(SOURCE).integrity(),
              'model_manifest_sha256': MODEL_MANIFEST_SHA256, 'model_files': model_files,
              'wemm_fingerprint': _legacy_fingerprint(),
              'qwen_worker_fingerprints': gates['gpu-acceptance.json']['worker_fingerprints'],
              'profiles': {name: profile.as_dict() for name, profile in PROFILES.items()},
              'judge_protocol': Judge(expected_limit_cny=600).protocol(), 'acceptance': gates,
              'order': ORDER, 'report_port': 8774, 'budget_limit_cny': 600,
              'interpretation': 'contemporary complete systems; not isolated embedding/reranker attribution'}
    if destination.exists():
        raise RuntimeError('freeze_already_exists_no_automatic_rerun')
    atomic_json(destination, result)
    return result


def verify_frozen_inputs(freeze, run=None):
    from camroll_eval.dataset import Dataset
    if Dataset(SOURCE).integrity() != freeze['input_integrity']:
        raise RuntimeError('frozen_album_inputs_changed')
    if run is not None and run.get('input_integrity') != freeze['input_integrity']:
        raise RuntimeError('run_input_integrity_differs_from_freeze')


def child_run(args):
    from camroll_eval.runner import Runner
    root = RUNTIME / ('evaluation' if args.formal else 'smoke-evaluation')
    source = SOURCE if args.formal else synthetic_source()
    result = Runner(root, source, RUNTIME / 'albums').run(run_id=args.run_id, experiment=args.child_group,
              config='M4', qa_concurrency=1, judge_concurrency=4, strict=args.formal)
    print(json.dumps({'run_id': result['run_id'], 'status': result['status'], 'complete': validate_run(result, formal=args.formal)}), flush=True)
    return 0 if validate_run(result, formal=args.formal) else 2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--child-group', choices=ORDER)
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    bootstrap()
    if args.child_group:
        return child_run(args)
    from camroll_agent.budget import BudgetLedger
    from camroll_eval.common import atomic_json, read_json, now_iso
    from four_gpu_window import GpuWindow
    ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'], expected_limit_cny=600)
    ledger.check_limit()
    if ledger.summary()['uncertain_requests'] or ledger.summary()['reserved_cny']:
        raise RuntimeError('pending_requests_require_reconciliation')
    run_stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    suite_id = ('formal-' if args.formal else 'smoke-') + run_stamp
    formal = args.formal
    suite_path = RUNTIME / ('formal-suite.json' if formal else 'smoke-suite.json')
    if suite_path.exists():
        raise RuntimeError('suite_already_exists_no_automatic_repeat')
    freeze = freeze_inputs(RUNTIME / 'freeze.json') if formal else None
    groups = [{'name': name, 'run_id': name.lower() + '-' + suite_id, 'status': 'not_started',
               'qa_count': 100 if formal else 3, 'planned_turn_count': 111 if formal else 4} for name in ORDER]
    state = {'suite_id': suite_id, 'started_at': now_iso(), 'status': 'running', 'formal': formal,
             'groups': groups, 'budget_before': ledger.summary(), 'source_revision': RELEASE.name}
    profile_path = RUNTIME / 'evaluation/comparison-profile.json'
    def save():
        state['updated_at'] = now_iso()
        atomic_json(suite_path, state)
        if formal:
            atomic_json(profile_path, {'schema_version': 'camroll-four-models-v1', 'title': 'Qwen / WeMM 四组对照',
                'groups': groups, 'historical': [{'label': '历史 M1 · 75/14/11（仅历史参考）'}],
                'suite_status': state['status'], 'updated_at': state['updated_at'],
                'note': '固定100题、111轮；完整系统效果比较，不单独归因于 embedding 或 reranker。'})
    save()
    window = GpuWindow(RELEASE, suite_id)
    active = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('suite_signal_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        entered = window.enter()
        if freeze:
            if (entered['model_integrity']['manifest_sha256'] != freeze['model_manifest_sha256']
                    or entered['model_integrity']['models'] != freeze['model_files']
                    or entered['before']['fingerprint'] != freeze['wemm_fingerprint']):
                raise RuntimeError('gpu_window_models_differ_from_freeze')
        if not formal:
            window.switch('qwen')
            from four_gpu_acceptance import run_acceptance
            if run_acceptance(RUNTIME)['status'] != 'passed':
                raise RuntimeError('local_qwen_acceptance_failed')
        for group in groups:
            ledger.check_limit()
            if ledger.summary()['uncertain_requests']:
                raise RuntimeError('uncertain_cloud_request_stop')
            if freeze and source_manifest(RELEASE) != freeze['source_files']:
                raise RuntimeError('frozen_source_changed')
            if freeze:
                verify_frozen_inputs(freeze)
            window.switch('qwen' if group['name'].startswith('Q') else 'wemm')
            group.update(status='running', started_at=now_iso())
            save()
            log_path = RUNTIME / 'logs' / (group['run_id'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [str(PYTHON), str(Path(__file__).resolve()), '--child-group', group['name'], '--run-id', group['run_id']]
            if formal:
                command.append('--formal')
            with log_path.open('xb') as log:
                active = subprocess.Popen(command, cwd=RELEASE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                window.register_child(active.pid, expected_command=command)
                while active.poll() is None:
                    time.sleep(5)
                    ledger.check_limit()
                code = active.returncode
                window.unregister_child(active.pid)
                active = None
            run_path = RUNTIME / ('evaluation' if formal else 'smoke-evaluation') / 'runs' / group['run_id'] / 'run.json'
            run = read_json(run_path) if run_path.exists() else {}
            if freeze:
                verify_frozen_inputs(freeze, run)
            group.update(status=run.get('status', 'failed'), returncode=code, finished_at=now_iso(),
                         acceptance_complete=validate_run(run, formal=formal), summary=run.get('summary'))
            save()
            if run.get('status') in {'failed', 'cancelled'} or ledger.summary()['uncertain_requests']:
                raise RuntimeError('group_incomplete_stop_' + group['name'])
        state['status'] = 'passed' if all(g.get('acceptance_complete') for g in groups) else 'incomplete'
    except BaseException as exc:
        state.update(status='blocked' if isinstance(exc, Exception) else 'interrupted', error_type=type(exc).__name__,
                     error=str(exc) if isinstance(exc, (RuntimeError, ValueError, KeyboardInterrupt)) else type(exc).__name__)
        for group in groups:
            if group['status'] == 'running':
                group.update(status=state['status'], finished_at=now_iso(), error_type=type(exc).__name__)
        raise
    finally:
        try:
            state['restoration'] = window.restore('suite_' + state['status']) if window.state_path.exists() else {'state': 'not_entered'}
        except BaseException as exc:
            state.update(status='restore_failed', restoration={'state': 'failed', 'error_type': type(exc).__name__})
            raise
        finally:
            state.update(finished_at=now_iso(), budget_after=ledger.summary())
            save()
    return 0 if state['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
