import hashlib
import json
from pathlib import Path

import pytest

from materialize_host153 import materialize


def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    (root / "current").mkdir(parents=True)
    (root / "base").mkdir()
    rows = []
    for name, before, after in [("a.py", b"before", b"after"), ("new.py", None, b"new")]:
        row = {"path": name, "mode": "644"}
        for state, data in [("base", before), ("current", after)]:
            row[state + "_sha256"] = hashlib.sha256(data).hexdigest() if data else None
            if data:
                (root / state / (name + ".snapshot")).write_bytes(data)
        rows.append(row)
    (root / "manifest.json").write_text(json.dumps({"base_commit": "base", "files": rows}))
    return root


def test_both_directions_and_no_overwrite(tmp_path):
    source = snapshot(tmp_path)
    for state in ("base", "current"):
        out = tmp_path / state
        receipt = materialize(source, out, state)
        assert (out / "a.py").read_bytes() == (b"before" if state == "base" else b"after")
        assert (out / "new.py").exists() == (state == "current")
        assert receipt["live_changes"] is False
        with pytest.raises(FileExistsError):
            materialize(source, out, state)


@pytest.mark.parametrize("problem", ["corrupt", "escape", "symlink", "duplicate"])
def test_invalid_source_rejected_before_writes(tmp_path, problem):
    source = snapshot(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text())
    if problem == "corrupt":
        (source / "current/a.py.snapshot").write_bytes(b"wrong")
    elif problem == "escape":
        manifest["files"][0]["path"] = "../escape"
    elif problem == "duplicate":
        manifest["files"].append(manifest["files"][0])
    else:
        target = source / "current/a.py.snapshot"
        target.unlink()
        target.symlink_to(source / "base/a.py.snapshot")
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        materialize(source, tmp_path / "output", "current")
    assert not (tmp_path / "output").exists()
