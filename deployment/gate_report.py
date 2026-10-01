"""Read-only Gate report on 8774, with prior experiments accessible as history."""
from pathlib import Path
import socket
import sys


def main():
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release / 'camroll/camroll-agent'))
    from gate_config import ALBUMS, HISTORY_RUNTIME, RUNTIME, SOURCE
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # A conflict is an error. Never terminate a listener here.
    sock.bind(('0.0.0.0', 8774))
    sock.listen(128)
    from camroll_eval.service import create_app
    import uvicorn
    app = create_app(RUNTIME / 'evaluation', SOURCE, ALBUMS, read_only=True, mount_agent=False,
                     history_roots=[HISTORY_RUNTIME / 'evaluation'])
    uvicorn.Server(uvicorn.Config(app, access_log=False)).run(sockets=[sock])


if __name__ == '__main__':
    main()
