"""Fixed entrypoint for the isolated CamRoll chat UI on port 8773."""
from pathlib import Path
import sys

import uvicorn


RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'camroll/camroll-chat'))

from camroll_chat.app import create_app
from camroll_chat.config import ChatConfig, DEFAULT_ALBUM_ROOT, DEFAULT_UPSTREAM_URL


APP = create_app(ChatConfig(
    album_root=DEFAULT_ALBUM_ROOT,
    upstream_url=DEFAULT_UPSTREAM_URL,
    static_dir=RELEASE / 'camroll/camroll-chat/static',
))

uvicorn.run(
    APP,
    host='0.0.0.0',
    port=8773,
    workers=1,
    access_log=False,
)
