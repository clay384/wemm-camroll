#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/client.py demo
