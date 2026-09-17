"""Read protected deployment configuration without exposing its values."""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE = Path('/home/realmagic/camroll-wemm')
RUNTIME = BASE / 'runtime/photobench-v2'
PYTHON = BASE / '.venv-photobench/bin/python'


def _protected_json(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError('Missing or unsafe protected configuration file')
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise RuntimeError('Cannot read protected configuration') from None
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in value.items()):
        raise RuntimeError('Protected configuration must contain string values')
    return value


def configure(release: Path) -> dict[str, str]:
    """Apply old credentials, independent Judge credentials, then fixed isolation."""
    values = _protected_json(BASE / 'runtime/secrets.json')
    values.update(_protected_json(RUNTIME / 'secrets.json'))
    values.update({
        'CAMROLL_BUDGET_DB': str(BASE / 'runtime/budget.sqlite'),
        'CAMROLL_ALBUM_ROOT': str(RUNTIME / 'albums'),
        'CAMROLL_EVAL_ROOT': str(RUNTIME / 'evaluation'),
        'CAMROLL_EVAL_SOURCE_ROOT': str(RUNTIME / 'source/album3-max'),
        'CAMROLL_EVAL_ALBUM_ROOT': str(RUNTIME / 'albums'),
        'CAMROLL_EVAL_JUDGE_API_KEY_FILE': str(RUNTIME / 'judge.key'),
        'WEMM_BASE_URL': 'http://127.0.0.1:8765',
        'WEMM_KEY_PATH': '/home/realmagic/wemm-embedding-lab/run/access.key',
        'CAMROLL_FACE_MODEL_DIR': '/home/realmagic/.insightface/models/buffalo_l',
        'NO_ALBUMENTATIONS_UPDATE': '1',
        'PYTHONPATH': os.pathsep.join([str(release / 'camroll/camroll-agent'), str(release)]),
        'PYTHONUNBUFFERED': '1',
        'CAMROLL_RELEASE_REVISION': release.name,
    })
    for name in ('judge.key',):
        p = RUNTIME / name
        if p.is_symlink() or not p.is_file() or p.stat().st_mode & 0o077:
            raise RuntimeError('Missing or unsafe independent Judge key file')
    os.environ.update(values)
    return values
