"""Export sanitized usage evidence; never exports configuration or messages."""
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

from camroll_agent.budget import BudgetLedger

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
ledger = BudgetLedger(os.environ['CAMROLL_BUDGET_DB'])
with sqlite3.connect('file:' + str(ledger.path) + '?mode=ro', uri=True) as connection:
    rows = connection.execute('SELECT id,created,status,reserved_units,charged_units,metadata FROM budget_calls ORDER BY created,id').fetchall()
calls = [{'id': row[0], 'created': row[1], 'status': row[2],
          'reserved_cny': row[3] / 1_000_000, 'estimated_charged_cny': row[4] / 1_000_000,
          'metadata': json.loads(row[5])} for row in rows]
report = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'pricing_source': 'https://help.aliyun.com/zh/model-studio/model-pricing',
          'pricing_checked_on': '2026-09-15',
          'estimate_not_invoice': True, 'summary': ledger.summary()}
(output / 'usage_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
(output / 'usage_calls.jsonl').write_text(''.join(json.dumps(call, ensure_ascii=False) + '\n' for call in calls))
print(json.dumps(report, ensure_ascii=False))
