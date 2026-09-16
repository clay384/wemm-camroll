"""Process-safe cloud budget accounting; never stores prompts or credentials.

Amounts are integer micro-yuan. A reservation is committed before the network
call, and uncertain calls remain charged at their reservation until reconciled.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    code = "budget_exhausted"


class BudgetLedger:
    def __init__(self, path: str | Path, limit_cny: float = 300.0):
        if not 0 < limit_cny <= 300:
            raise ValueError("The authorized budget must be within (0, 300] CNY")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit_units = round(limit_cny * 1_000_000)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS budget_calls (
                    id TEXT PRIMARY KEY, created REAL NOT NULL,
                    status TEXT NOT NULL, reserved_units INTEGER NOT NULL,
                    charged_units INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS budget_settings (
                    key TEXT PRIMARY KEY, value INTEGER NOT NULL
                );
            """)
            conn.execute("INSERT OR IGNORE INTO budget_settings VALUES ('limit_units', ?)",
                         (self.limit_units,))
            # Re-opening a ledger must never replenish its allowance.
            conn.execute("UPDATE budget_settings SET value = MIN(value, ?) WHERE key='limit_units'",
                         (self.limit_units,))
            self.limit_units = conn.execute(
                "SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
        os.chmod(self.path, 0o600)

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=30000")
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _safe_metadata(metadata: dict | None) -> str:
        allowed = {
            "role", "model", "response_model", "prompt_version", "temperature",
            "max_tokens", "enable_thinking", "prompt_tokens", "completion_tokens",
            "total_tokens", "cached_tokens", "error_code", "elapsed_s", "stream",
            "pricing_version", "input_rate", "output_rate", "usage_missing",
        }
        clean = {k: v for k, v in (metadata or {}).items()
                 if k in allowed and (v is None or isinstance(v, (str, int, float, bool)))}
        # Do not accept arbitrarily large values even in an approved field.
        clean = {k: v[:180] if isinstance(v, str) else v for k, v in clean.items()}
        return json.dumps(clean, ensure_ascii=False)

    def reserve(self, amount_cny: float, metadata: dict | None = None) -> str:
        if amount_cny < 0 or not math.isfinite(amount_cny):
            raise ValueError("Invalid reservation")
        units = math.ceil(amount_cny * 1_000_000)
        call_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            limit = conn.execute("SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
            committed = conn.execute("""SELECT COALESCE(SUM(CASE
                WHEN status IN ('reserved','uncertain') THEN reserved_units
                ELSE charged_units END),0) FROM budget_calls""").fetchone()[0]
            if committed + units > limit:
                raise BudgetExceeded("Cloud budget has insufficient unreserved funds")
            conn.execute("INSERT INTO budget_calls (id,created,status,reserved_units,metadata) VALUES (?,?,'reserved',?,?)",
                         (call_id, time.time(), units, self._safe_metadata(metadata)))
        return call_id

    def finish(self, call_id: str, amount_cny: float | None, metadata: dict | None = None,
               *, rejected: bool = False) -> None:
        """Settle known usage, release explicit rejection, or retain uncertain cost."""
        if amount_cny is not None and (amount_cny < 0 or not math.isfinite(amount_cny)):
            raise ValueError("Invalid charge")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status, metadata FROM budget_calls WHERE id=?", (call_id,)).fetchone()
            if row is None:
                raise KeyError(call_id)
            if row[0] not in ("reserved", "uncertain"):
                return  # Idempotent completion, never double charge.
            merged = json.loads(row[1])
            merged.update(json.loads(self._safe_metadata(metadata)))
            status = "released" if rejected else ("uncertain" if amount_cny is None else "settled")
            charge = 0 if amount_cny is None or rejected else math.ceil(amount_cny * 1_000_000)
            conn.execute("UPDATE budget_calls SET status=?,charged_units=?,metadata=? WHERE id=?",
                         (status, charge, self._safe_metadata(merged), call_id))

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status,reserved_units,charged_units,metadata FROM budget_calls").fetchall()
            limit = conn.execute("SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
        spent = sum(row[2] for row in rows)
        pending = sum(row[1] for row in rows if row[0] in ("reserved", "uncertain"))
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        roles: dict[str, dict] = {}
        for status, reserved, charged, raw in rows:
            metadata = json.loads(raw)
            role = str(metadata.get("role", "unknown"))
            item = roles.setdefault(role, {"requests": 0, "estimated_cny": 0.0, "reserved_cny": 0.0})
            item["requests"] += 1
            item["estimated_cny"] += charged / 1_000_000
            if status in ("reserved", "uncertain"):
                item["reserved_cny"] += reserved / 1_000_000
            for key in usage:
                usage[key] += int(metadata.get(key, 0) or 0)
        return {"limit_cny": limit / 1_000_000, "estimated_spent_cny": spent / 1_000_000,
                "reserved_cny": pending / 1_000_000,
                "remaining_cny": max(0, limit - spent - pending) / 1_000_000,
                "requests": len(rows), "uncertain_requests": sum(row[0] == "uncertain" for row in rows),
                "usage": usage, "roles": roles,
                "note": "Usage-priced estimate; uncertain calls retain their full reservation."}

    def capability(self, key: str) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM capabilities WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_capability(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO capabilities VALUES (?,?)", (key, json.dumps(value)))
