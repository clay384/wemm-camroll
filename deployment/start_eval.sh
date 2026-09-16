#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /home/realmagic/camroll-wemm/.venv-photobench/bin/python "$SCRIPT_DIR/manage_eval.py" start
