"""Load protected config before importing the API app."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for key, value in json.loads((root/'runtime/secrets.json').read_text()).items():
    os.environ[key] = value
import uvicorn

uvicorn.run('camroll_agent.service:app', host='127.0.0.1', port=8770, workers=1)
