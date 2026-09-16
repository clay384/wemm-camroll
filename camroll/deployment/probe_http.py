"""Small live sync/SSE API checks; all generation is billed by the service ledger."""
import json
import sys
from pathlib import Path

import requests

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
if any((out / name).exists() for name in ['sync.json', 'stream.json', 'summary.json']):
    raise SystemExit('Existing probe output; use a new output directory before charged requests')
client = requests.Session()
client.trust_env = False
question = '请先检索相册中的动物照片，用view_image查看最相关的一张，简短描述可观察内容并引用图片ID。'
payload = {'album_id': 'photobench-album3', 'question': question, 'config': 'M1'}
response = client.post('http://127.0.0.1:8770/api/ask', json=payload, timeout=200)
response.raise_for_status()
sync = response.json()
(out / 'sync.json').write_text(json.dumps(sync, ensure_ascii=False, indent=2))
events = []
with client.post('http://127.0.0.1:8770/api/ask', json={**payload, 'config': 'M4', 'stream': True}, stream=True, timeout=200) as response:
    response.raise_for_status()
    response.encoding = 'utf-8'
    kind = None
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith('event: '):
            kind = line[7:]
        elif line.startswith('data: '):
            events.append({'event': kind, 'data': json.loads(line[6:])})
(out / 'stream.json').write_text(json.dumps(events, ensure_ascii=False, indent=2))
final = next((item['data'] for item in reversed(events) if item['event'] == 'answer'), {})
checks = {'sync_answer': bool(sync.get('answer')), 'sync_normal': sync.get('stopped_reason') == 'ok',
          'sync_refs': bool(sync.get('evidence_image_ids')), 'stream_answer': bool(final.get('final_text')),
          'stream_normal': final.get('stopped_reason') == 'ok', 'stream_refs': bool(final.get('evidence_image_ids')),
          'same_index': bool(sync.get('index_version')) and sync.get('index_version') == final.get('index_version'),
          'stream_final_images': any(row['tool'] == 'final_image_review' and 'error' not in row['result'] for row in final.get('tool_trace', []))}
summary = {'checks': checks, 'passed': all(checks.values()),
           'sync_usage': sync.get('usage'), 'stream_usage': final.get('usage'),
           'purpose': 'HTTP/SSE and evidence transmission; not answer-quality scoring'}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
if not summary['passed']:
    raise SystemExit(2)
