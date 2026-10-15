"""Independent read-only MiniCPM report; preserve every historical report root."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'camroll/camroll-agent'))

from camroll_eval.common import atomic_json, object_hash, read_json, safe_id, sha256
from camroll_eval.repository import (MINICPM_COMPARISON, MINICPM_HISTORY_ID,
    MINICPM_HISTORY_SHA256, MINICPM_MANIFEST, MINICPM_PROTOCOL, MINICPM_SDK_PROTOCOLS, Repository)


def history_roots():
    from gate_config import BASE, HISTORY_RUNTIME
    return [BASE / 'runtime' / name / 'evaluation' for name in (
        'photobench-flash-gate-20261007h', 'photobench-flash-gate-20261007g',
        'photobench-flash-gate-20261006f', 'photobench-flash-gate-20261002d')
    ] + [HISTORY_RUNTIME / 'evaluation']


def _fixed_json(path):
    """Only a fixed sibling path, never a manifest-supplied filename."""
    if not path.is_file():
        return {}
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError('unsafe report status path')
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError('report status must be an object')
    return value


def _public_status(path):
    try:
        value = _fixed_json(path)
    except (OSError, ValueError):
        return {'status': 'unavailable', 'reason': '状态文件不可读取或未通过路径校验'}
    result = {key: value[key] for key in ('status', 'reason', 'updated_at', 'observed_at')
              if isinstance(value.get(key), str)}
    if isinstance(value.get('stages'), dict):
        # Stage details can include private payloads. Expose status/reason only.
        result['stages'] = {str(name): {key: row[key] for key in ('status', 'reason')
                            if isinstance(row.get(key), str)}
                            for name, row in value['stages'].items() if isinstance(row, dict)}
    try:
        result['run_id'] = safe_id(value['run_id'])
    except (KeyError, ValueError):
        pass
    return result


def profile(runtime, repository):
    runtime = Path(runtime).resolve()
    acceptance = _public_status(runtime / 'acceptance/status.json')
    execution = _public_status(runtime / 'status.json')
    candidates = [run for run in repository.list()
        if run.get('mode') == 'minicpm_compare' and run.get('formal_protocol') is True
        and repository.path(run['run_id']).parent.parent.parent == repository.root]
    current = next((run for run in candidates if run['run_id'] == execution.get('run_id')), None)
    if current is None and len(candidates) == 1:
        current = candidates[0]
    from camroll_eval.minicpm_experiment import inference_config
    frozen = _fixed_json(runtime / 'acceptance/protocol.json').get('inference_config')
    configured = frozen or _fixed_json(runtime / 'request-profile.json')
    cfg = inference_config(configured).as_dict() if configured else {}
    if current:
        cfg = current.get('inference_config') or cfg
    sdk_defaults = cfg.get('protocol_version') in MINICPM_SDK_PROTOCOLS
    prior = _fixed_json(runtime / 'prior-attempts.json').get('attempts', [])
    prior = [{key: row[key] for key in ('label', 'status', 'reason', 'observed_at', 'evidence_sha256')
              if isinstance(row.get(key), str)} for row in prior if isinstance(row, dict)] if isinstance(prior, list) else []
    reason = execution.get('reason') or acceptance.get('reason') or '尚无新正式运行记录；验收结果不等于100题正式测评。'
    groups = [{'name': cfg.get('group') or ('MiniCPM47-SDK-Defaults-Off' if sdk_defaults else 'MiniCPM47-Thinking-On-Off'), 'run_id': current['run_id'] if current else None,
               'status': current.get('status') if current else 'not_started', 'reused': False},
              {'name': 'F37-Off', 'run_id': MINICPM_HISTORY_ID, 'reused': True}]
    scope = None
    if current:
        try:
            scope = repository.compare([current['run_id'], MINICPM_HISTORY_ID]).get('minicpm_scope')
        except (OSError, ValueError):
            pass
    return {'schema_version': cfg.get('protocol_version', MINICPM_PROTOCOL), 'comparison_scope': MINICPM_COMPARISON,
        'title': '自建 MiniCPM Q6 Thinking · M4 Gate-Off 历史对照' if cfg.get('provider') == 'llamacpp-selfhost' else 'MiniCPM · Thinking未指定 · M4 Gate-Off 历史对照' if sdk_defaults else 'MiniCPM Thinking · M4 Gate-Off 与原 F37-Off 历史对照',
        'inference_config': cfg, 'prior_attempts': prior,
        'formal_status': current.get('status') if current else 'not_started',
        'reason': reason, 'acceptance': acceptance, 'execution': execution, 'groups': groups,
        'comparison_ready': scope is not None, 'source_hashes_verified': scope is not None,
        'history_note': '保留10月 h/g/f/d与9月30日全部历史结果；旧组不重评分、不改写。'}


def write_comparison(runtime, run_id):
    """Publish a terminal pair only after the report's complete scope check.

    It never edits either run. Validation happens before publishing, so readers
    never observe an unverified candidate and an older manifest stays intact.
    """
    root = Path(runtime).resolve() / 'evaluation'
    repo = Repository(root, history_roots=history_roots())
    current_path, old_path = repo.path(safe_id(run_id)), repo.path(MINICPM_HISTORY_ID)
    current, old = repo.load(run_id), repo.load(MINICPM_HISTORY_ID)
    if sha256(old_path) != MINICPM_HISTORY_SHA256:
        raise ValueError('original F37-Off hash mismatch')
    manifest = {'schema_version': MINICPM_MANIFEST, 'protocol': current.get('protocol_label'),
        'comparison_scope': MINICPM_COMPARISON, 'input_integrity': current.get('input_integrity'),
        'index_binding': current.get('index_binding'), 'qa_plan_sha256': object_hash(current.get('qa_plan')),
        'judge_protocol_sha256': object_hash(current.get('judge_protocol')),
        'runs': [{'source': 'current', 'run_id': run_id, 'run_sha256': sha256(current_path)},
                 {'source': 'history', 'run_id': MINICPM_HISTORY_ID, 'run_sha256': MINICPM_HISTORY_SHA256}]}
    destination = root / 'minicpm-comparison.json'
    if destination.exists():
        _fixed_json(destination)
    if repo._minicpm_comparison_scope([current, old], manifest=manifest) is None:
        raise ValueError('MiniCPM historical comparison validation failed')
    atomic_json(destination, manifest)
    return manifest


def create_report(runtime):
    from gate_config import ALBUMS, SOURCE
    from camroll_eval.service import create_app
    runtime = Path(runtime).resolve()
    app = create_app(runtime / 'evaluation', SOURCE, ALBUMS, read_only=True,
                     mount_agent=False, history_roots=history_roots())

    @app.get('/api/minicpm/status')
    def status():
        return profile(runtime, app.state.repository)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args(argv)
    from gate_config import BASE
    runtime = args.runtime.resolve()
    if (runtime.parent != (BASE / 'runtime').resolve()
            or not runtime.name.startswith(('minicpm47-thinking-m4-off-', 'minicpm47-sdk-defaults-m4-off-', 'minicpm47-q6-thinking-m4-off-'))):
        parser.error('runtime must be a new MiniCPM protocol directory under BASE/runtime')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Fail before application construction if somebody already owns 8774.
    sock.bind(('0.0.0.0', 8774))
    sock.listen(128)
    import uvicorn
    app = create_report(runtime)
    uvicorn.Server(uvicorn.Config(app, access_log=False)).run(sockets=[sock])


if __name__ == '__main__':
    main()
