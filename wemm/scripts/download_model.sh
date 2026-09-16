#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
: "${HTTPS_PROXY:?Set the verified task proxy first (server loopback 17890)}"
.venv/bin/python scripts/prepare_snapshot.py
exec .venv/bin/python -u scripts/download_model.py
