#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p frontend/dist
cp frontend/src/index.html frontend/src/style.css frontend/src/app.js frontend/dist/
