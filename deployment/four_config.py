"""Configuration for the isolated four-group experiment and read-only report."""
import os
from pathlib import Path

from eval_config import BASE, configure as legacy_configure

RUNTIME = BASE / 'runtime/photobench-qwen-20260930'
SOURCE = BASE / 'runtime/photobench-v2/source/album3-max'
PYTHON = BASE / '.venv-photobench/bin/python'
WORKER_PYTHON = Path('/home/realmagic/weknora-album-lab/.venv-model-worker/bin/python')
GPU_UUID = 'GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0'


def configure(release):
    import json
    legacy_configure(release)
    values = {
        'CAMROLL_ALBUM_ROOT': str(RUNTIME / 'albums'),
        'CAMROLL_EVAL_ROOT': str(RUNTIME / 'evaluation'),
        'CAMROLL_EVAL_ALBUM_ROOT': str(RUNTIME / 'albums'),
        'CAMROLL_EVAL_SOURCE_ROOT': str(SOURCE),
        'CAMROLL_BUDGET_CNY': '600',
        'CAMROLL_EXPECTED_BUDGET_CNY': '600',
        'CAMROLL_QWEN_RUNTIME': str(RUNTIME / 'qwen-worker'),
        'CAMROLL_QWEN_IMAGE_ROOTS': json.dumps([str(SOURCE), str(RUNTIME / 'albums'), str(RUNTIME / 'fixtures')]),
        'CAMROLL_QWEN_GPU_ENABLED': '1',
        'CAMROLL_QWEN_BASE_URL': 'http://127.0.0.1:18778',
        'CAMROLL_EVAL_QWEN_RELEASE': str(release),
        'CUDA_VISIBLE_DEVICES': GPU_UUID,
        'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HOME': str(RUNTIME / 'model-cache'),
        'TORCH_HOME': str(RUNTIME / 'torch-cache'),
        'NO_PROXY': 'localhost,127.0.0.1,192.168.0.100',
        'no_proxy': 'localhost,127.0.0.1,192.168.0.100',
    }
    os.environ.update(values)
    return values
