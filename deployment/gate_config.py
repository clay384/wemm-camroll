"""Flash Gate experiment: new records, unchanged album and shared budget."""
import os

from four_config import (BASE, GPU_UUID, PYTHON, RUNTIME as HISTORY_RUNTIME,
                         SOURCE, configure as previous_configure)

RUNTIME = BASE / 'runtime/photobench-flash-gate-20261001'
ALBUMS = HISTORY_RUNTIME / 'albums'
SOURCE_RUN_ID = 'w38max-formal-20260930-122300-813923'
SOURCE_ALBUM_ID = 'eval-' + SOURCE_RUN_ID
SNAPSHOT = 'bcc1cd1176ab39697c789a4f44c7426e'
ORDER = ('F37-Off', 'F37-On', 'F38-Off', 'F38-On')
MODELS = {'F37-Off': 'qwen3.7-flash', 'F37-On': 'qwen3.7-flash',
          'F38-Off': 'qwen3.8-flash', 'F38-On': 'qwen3.8-flash'}


def configure(release):
    previous_configure(release)
    values = {
        'CAMROLL_ALBUM_ROOT': str(ALBUMS),
        'CAMROLL_EVAL_ALBUM_ROOT': str(ALBUMS),
        'CAMROLL_EVAL_ROOT': str(RUNTIME / 'evaluation'),
        'CAMROLL_EXPECTED_BUDGET_CNY': '600',
        'CAMROLL_BUDGET_CNY': '600',
        'CUDA_VISIBLE_DEVICES': GPU_UUID,
    }
    os.environ.update(values)
    return values
