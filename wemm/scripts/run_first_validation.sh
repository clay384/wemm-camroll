#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p run reports logs
printf '%s\n' "$$" > run/first-validation.pid
while [ ! -f models/http-parts/VERIFIED.json ]; do sleep 5; done
.venv/bin/python scripts/lock_sources.py > logs/source-lock.log 2>&1
if [ ! -f reports/environment-setup.json ]; then cp reports/environment.json reports/environment-setup.json; fi
scripts/preflight.sh
cp reports/environment.json reports/environment-before-model.json
set +e
flock -n run/model.lock .venv/bin/python -u scripts/smoke.py > logs/smoke.log 2>&1
code=$?
printf '%s\n' "$code" > reports/smoke.exit
exit "$code"
