"""Read-only 8774 continuation report, retaining the original immutable history."""
from pathlib import Path
import socket
import sys


def main():
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release / 'camroll/camroll-agent'))
    from gate_config import ALBUMS, BASE, HISTORY_RUNTIME, SOURCE
    from gate_continuation_export import PARENT_RUNTIME, RUNTIME
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 8774))  # A conflict stops deployment; no listener is killed.
    sock.listen(128)
    from camroll_eval.service import create_app
    import uvicorn
    app = create_app(RUNTIME / 'evaluation', SOURCE, ALBUMS, read_only=True, mount_agent=False,
        history_roots=[PARENT_RUNTIME / 'evaluation',
                       BASE / 'runtime/photobench-flash-gate-20261002d/evaluation',
                       HISTORY_RUNTIME / 'evaluation'])
    uvicorn.Server(uvicorn.Config(app, access_log=False)).run(sockets=[sock])


if __name__ == '__main__':
    main()
