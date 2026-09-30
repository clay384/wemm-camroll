"""Read-only port8774 report server: no credentials, evaluation or Agent routes."""
import os
from pathlib import Path
import socket
import sys


def main():
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release / 'camroll/camroll-agent'))
    from four_config import RUNTIME, SOURCE
    # Bind before constructing anything; a conflict must fail, never displace it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 8774))
    sock.listen(128)
    from camroll_eval.service import create_app
    import uvicorn
    app = create_app(RUNTIME / 'evaluation', SOURCE, RUNTIME / 'albums', read_only=True, mount_agent=False)
    uvicorn.Server(uvicorn.Config(app, access_log=False)).run(sockets=[sock])


if __name__ == '__main__':
    main()
