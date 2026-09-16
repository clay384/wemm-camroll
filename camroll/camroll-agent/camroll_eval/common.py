"""Small, bounded filesystem helpers for the independent evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.chmod(path, 0o600)


def read_secret(name):
    """FILE has priority. Never put credential values in errors or logs."""
    filename = os.environ.get(name + "_FILE")
    if filename:
        path = Path(filename).expanduser()
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError(name + "_FILE must be an owner-only regular file")
        value = path.read_text().strip()
    else:
        value = os.environ.get(name, "").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(name + " is not configured")
    return value


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path leaves the configured root")
    return path


def safe_id(value):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError("invalid identifier")
    return value


def timer_record(start_epoch, start_clock):
    return {"started_at": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
            "finished_at": now_iso(), "wall_ms": round((time.monotonic() - start_clock) * 1000, 3)}
