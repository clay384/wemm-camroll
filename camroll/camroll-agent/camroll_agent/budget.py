"""Process-safe cloud budget accounting; never stores prompts or credentials.

Amounts are integer micro-yuan. A reservation is committed before the network
call, and uncertain calls remain charged at their reservation until reconciled.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, fields
import hashlib
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


class BudgetLimitChanged(BudgetExceeded):
    code = "budget_limit_changed"


class RequestUncertain(RuntimeError):
    code = "request_uncertain"


class RequestReplayError(RuntimeError):
    def __init__(self, code: str):
        super().__init__("Persisted request failed ({})".format(code))
        self.code = code


def _row_sha256(row) -> str:
    return hashlib.sha256(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _read_only_row(path, query, values):
    connection = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        return connection.execute(query, values).fetchone()
    finally:
        connection.close()


_LEDGER_ACK_ROW = "SELECT id,created,status,reserved_units,charged_units,metadata FROM budget_calls WHERE id=?"
_JOURNAL_ACK_ROW = "SELECT request_id,payload_hash,call_id,status,created,response,error_code FROM cloud_requests WHERE request_id=?"


@dataclass(frozen=True)
class UncertainRequestAcknowledgement:
    """Explicit permission to continue elsewhere while one old charge stays held.

    This is a run-scoped allowlist entry, not reconciliation and not permission
    to replay the old request. Paths, file identities and complete rows are
    frozen so a different ledger, altered request, or changed reservation fails
    closed. Callers must supply the specifically authorized request identity.
    """
    ledger_path: str
    journal_path: str
    ledger_device: int
    ledger_inode: int
    journal_device: int
    journal_inode: int
    call_id: str
    request_id: str
    payload_hash: str
    reserved_units: int
    ledger_row_sha256: str
    journal_row_sha256: str

    def __post_init__(self):
        for name in ("ledger_path", "journal_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
                raise ValueError("Acknowledgement paths must be canonical absolute paths")
        if self.ledger_path == self.journal_path:
            raise ValueError("Acknowledgement requires separate ledger and journal files")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in
               ("ledger_device", "ledger_inode", "journal_device", "journal_inode")):
            raise ValueError("Invalid acknowledged file identity")
        if type(self.reserved_units) is not int or self.reserved_units <= 0:
            raise ValueError("Acknowledgement must retain the positive original reservation")
        if any(not isinstance(value, str) or not value or len(value) > 1000 for value in (self.call_id, self.request_id)):
            raise ValueError("Explicit acknowledged request identity is required")
        for value in (self.payload_hash, self.ledger_row_sha256, self.journal_row_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Acknowledgement hashes must be SHA256 values")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("Acknowledgement fields differ from the frozen contract")
        return cls(**value)

    @staticmethod
    def _check_rows(ledger_row, journal_row, *, call_id, request_id, payload_hash, reserved_units):
        if (ledger_row is None or journal_row is None
                or ledger_row[0] != call_id or ledger_row[2] != "uncertain"
                or ledger_row[3] != reserved_units or ledger_row[4] != 0
                or journal_row[0] != request_id or journal_row[1] != payload_hash
                or journal_row[2] != call_id or journal_row[3] != "uncertain"
                or journal_row[5] is not None):
            raise RequestUncertain("Acknowledged request state or identity changed; new submissions stop")
        try:
            metadata = json.loads(ledger_row[5])
        except (TypeError, ValueError):
            raise RequestUncertain("Acknowledged ledger metadata is invalid") from None
        if (not isinstance(metadata, dict) or metadata.get("request_id") != request_id or not journal_row[6]
                or metadata.get("error_code") != journal_row[6]):
            raise RequestUncertain("Acknowledged ledger and journal do not identify the same unknown request")

    @classmethod
    def capture(cls, ledger_path, journal_path, *, call_id, request_id, payload_hash, reserved_units):
        """Read a caller-selected identity; never discovers or authorizes unknowns."""
        ledger, journal = Path(ledger_path).expanduser().resolve(), Path(journal_path).expanduser().resolve()
        try:
            ledger_stat, journal_stat = ledger.stat(), journal.stat()
            ledger_row = _read_only_row(ledger, _LEDGER_ACK_ROW, (call_id,))
            journal_row = _read_only_row(journal, _JOURNAL_ACK_ROW, (request_id,))
        except (OSError, sqlite3.Error):
            raise RequestUncertain("Acknowledged request evidence is unavailable") from None
        cls._check_rows(ledger_row, journal_row, call_id=call_id, request_id=request_id,
                        payload_hash=payload_hash, reserved_units=reserved_units)
        return cls(str(ledger), str(journal), ledger_stat.st_dev, ledger_stat.st_ino,
                   journal_stat.st_dev, journal_stat.st_ino, call_id, request_id, payload_hash, reserved_units,
                   _row_sha256(ledger_row), _row_sha256(journal_row))

    def validate(self, ledger_path, connection):
        try:
            ledger, journal = Path(ledger_path).resolve(), Path(self.journal_path)
            ledger_stat, journal_stat = ledger.stat(), journal.stat()
            if (str(ledger) != self.ledger_path or (ledger_stat.st_dev, ledger_stat.st_ino) != (self.ledger_device, self.ledger_inode)
                    or (journal_stat.st_dev, journal_stat.st_ino) != (self.journal_device, self.journal_inode)):
                raise RequestUncertain("Acknowledged ledger or journal file identity changed")
            ledger_row = connection.execute(_LEDGER_ACK_ROW, (self.call_id,)).fetchone()
            journal_row = _read_only_row(journal, _JOURNAL_ACK_ROW, (self.request_id,))
            self._check_rows(ledger_row, journal_row, call_id=self.call_id, request_id=self.request_id,
                             payload_hash=self.payload_hash, reserved_units=self.reserved_units)
            if _row_sha256(ledger_row) != self.ledger_row_sha256 or _row_sha256(journal_row) != self.journal_row_sha256:
                raise RequestUncertain("Acknowledged request evidence changed")
        except (OSError, sqlite3.Error):
            raise RequestUncertain("Acknowledged request evidence is unavailable") from None


class BudgetLedger:
    def __init__(self, path: str | Path, limit_cny: float = 300.0, *,
                 expected_limit_cny: float | None = None,
                 acknowledged_uncertain=()):
        if not math.isfinite(limit_cny) or not 0 < limit_cny <= 600:
            raise ValueError("The authorized budget must be within (0, 600] CNY")
        if expected_limit_cny is not None and (not math.isfinite(expected_limit_cny) or not 0 < expected_limit_cny <= 600):
            raise ValueError("Invalid expected budget")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit_units = round(limit_cny * 1_000_000)
        self.expected_limit_units = None if expected_limit_cny is None else round(expected_limit_cny * 1_000_000)
        self.acknowledged_uncertain = tuple(acknowledged_uncertain)
        if (any(not isinstance(ack, UncertainRequestAcknowledgement) for ack in self.acknowledged_uncertain)
                or len({ack.call_id for ack in self.acknowledged_uncertain}) != len(self.acknowledged_uncertain)):
            raise ValueError("Uncertain request acknowledgements must be distinct frozen identities")
        if self.acknowledged_uncertain and self.expected_limit_units != 600_000_000:
            raise ValueError("Acknowledged continuation requires the unchanged 600 CNY guard")
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
                CREATE TABLE IF NOT EXISTS budget_limit_audit (
                    id TEXT PRIMARY KEY, created REAL NOT NULL,
                    previous_units INTEGER NOT NULL, limit_units INTEGER NOT NULL,
                    reason TEXT NOT NULL
                );
            """)
            conn.execute("INSERT OR IGNORE INTO budget_settings VALUES ('limit_units', ?)",
                         (self.limit_units,))
            self.limit_units = conn.execute(
                "SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
            self._acknowledged_ids(conn)
        os.chmod(self.path, 0o600)

    def with_expected_limit(self, expected_limit_cny):
        return BudgetLedger(self.path, expected_limit_cny=expected_limit_cny,
                            acknowledged_uncertain=self.acknowledged_uncertain)

    def _acknowledged_ids(self, connection):
        for ack in self.acknowledged_uncertain:
            ack.validate(self.path, connection)
        return {ack.call_id for ack in self.acknowledged_uncertain}

    def raise_limit(self, limit_cny: float, *, expected_previous_cny: float, reason: str) -> dict:
        if not math.isfinite(limit_cny) or not 0 < limit_cny <= 600 or not reason.strip():
            raise ValueError("An authorized limit within (0, 600] and audit reason are required")
        units = round(limit_cny * 1_000_000)
        previous_units = round(expected_previous_cny * 1_000_000)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
            if current != previous_units:
                raise BudgetLimitChanged("Budget changed before the explicit limit adjustment")
            if units <= current:
                raise ValueError("The explicit adjustment must raise the current limit")
            audit_id = uuid.uuid4().hex
            conn.execute("INSERT INTO budget_limit_audit VALUES (?,?,?,?,?)",
                         (audit_id, time.time(), current, units, reason[:1000]))
            conn.execute("UPDATE budget_settings SET value=? WHERE key='limit_units'", (units,))
        self.limit_units = units
        return {"audit_id": audit_id, "previous_cny": current / 1_000_000, "limit_cny": limit_cny}

    def check_limit(self) -> None:
        with self._connect() as conn:
            actual = conn.execute("SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
        if self.expected_limit_units is not None and actual != self.expected_limit_units:
            raise BudgetLimitChanged("Cumulative budget ceiling changed; new submissions are stopped")

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
            "request_id",
            "provider", "reasoning_tokens", "image_tokens", "created_cache_tokens",
            "decision_mode", "json_mode",
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
            if self.expected_limit_units is not None and limit != self.expected_limit_units:
                raise BudgetLimitChanged("Cumulative budget ceiling changed; new submissions are stopped")
            acknowledged = self._acknowledged_ids(conn)
            unknown = {row[0] for row in conn.execute("SELECT id FROM budget_calls WHERE status='uncertain'")}
            if self.expected_limit_units is not None and unknown - acknowledged:
                raise RequestUncertain("An earlier cloud charge is uncertain; new submissions are stopped")
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
        if any(ack.call_id == call_id for ack in self.acknowledged_uncertain):
            raise RequestUncertain("Acknowledgement does not authorize settling or releasing the held request")
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
            acknowledged = self._acknowledged_ids(conn)
            rows = conn.execute("SELECT status,reserved_units,charged_units,metadata,id FROM budget_calls").fetchall()
            limit = conn.execute("SELECT value FROM budget_settings WHERE key='limit_units'").fetchone()[0]
        spent = sum(row[2] for row in rows)
        pending = sum(row[1] for row in rows if row[0] in ("reserved", "uncertain"))
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        roles: dict[str, dict] = {}
        for status, reserved, charged, raw, call_id in rows:
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
                "acknowledged_uncertain_requests": len(acknowledged),
                "blocking_uncertain_requests": sum(row[0] == "uncertain" and row[4] not in acknowledged for row in rows),
                "acknowledged_reserved_cny": sum(row[1] for row in rows if row[4] in acknowledged) / 1_000_000,
                "usage": usage, "roles": roles,
                "note": "Usage-priced estimate; uncertain calls retain their full reservation."}

    def capability(self, key: str) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM capabilities WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_capability(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO capabilities VALUES (?,?)", (key, json.dumps(value)))


class RequestJournal:
    """Persist private cloud responses separately from the shared cost ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS cloud_requests (
                request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL,
                call_id TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                response TEXT, error_code TEXT
            )""")
            conn.execute("CREATE TABLE IF NOT EXISTS request_settings (key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                yield conn
        finally:
            conn.close()

    def claim(self, request_id: str, payload_hash: str, reserve_callback) -> dict:
        if not request_id or not payload_hash:
            raise ValueError("Request identity and payload hash are required")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_hash,call_id,status,response,error_code FROM cloud_requests WHERE request_id=?",
                               (request_id,)).fetchone()
            if row:
                if row[0] != payload_hash:
                    raise RequestReplayError("request_identity_mismatch")
                if row[2] == "completed":
                    return {"call_id": row[1], "response": json.loads(row[3]), "replayed": True}
                if row[2] == "failed":
                    raise RequestReplayError(row[4] or "request_failed")
                raise RequestUncertain("Persisted request is pending or uncertain; automatic resend is disabled")
            call_id = reserve_callback()
            conn.execute("INSERT INTO cloud_requests (request_id,payload_hash,call_id,status,created) VALUES (?,?,?,'pending',?)",
                         (request_id, payload_hash, call_id, time.time()))
            return {"call_id": call_id, "response": None, "replayed": False}

    def remember(self, key: str, value: Any) -> Any:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO request_settings VALUES (?,?)", (key, json.dumps(value)))
            stored = conn.execute("SELECT value FROM request_settings WHERE key=?", (key,)).fetchone()[0]
        return json.loads(stored)

    def complete(self, request_id: str, response: dict) -> None:
        payload = json.dumps(response, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute("UPDATE cloud_requests SET status='completed',response=?,error_code=NULL WHERE request_id=? AND status='pending'",
                                  (payload, request_id))
            if cursor.rowcount != 1:
                raise RequestUncertain("Only a pending request can be completed")

    def fail(self, request_id: str, error_code: str, *, uncertain: bool = False,
             failure_details: dict | None = None) -> None:
        """Optionally retain caller-redacted failure evidence for a new request.

        Existing callers keep their original behavior. A completed response is
        never replaced, and already failed/uncertain historical rows stay fixed.
        """
        if failure_details is not None and not isinstance(failure_details, dict):
            raise TypeError("failure_details must be a redacted dictionary")
        response = json.dumps({"_failure": failure_details}, ensure_ascii=False) if failure_details is not None else None
        with self._connect() as conn:
            conn.execute("UPDATE cloud_requests SET status=?,error_code=?,response=COALESCE(response,?) WHERE request_id=? AND (status='pending' OR (status='completed' AND ?))",
                         ("uncertain" if uncertain else "failed", error_code[:100], response, request_id, uncertain))
