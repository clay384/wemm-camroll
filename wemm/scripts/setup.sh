#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p run logs reports frontend/dist import data/originals data/derived data/indexes data/tmp models
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock
.venv/bin/python -m pip check
