"""Verify and materialize an archived source overlay into a NEW offline directory.

Never connects to a server, applies a live patch, starts services or touches data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def materialize(snapshot, destination, state):
    if state not in {"base", "current"}:
        raise ValueError("state must be base or current")
    snapshot, destination = Path(snapshot).resolve(), Path(destination)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    prepared, absent, seen = [], [], set()
    for row in manifest["files"]:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or str(rel) in seen:
            raise ValueError("unsafe or duplicate source path")
        seen.add(str(rel))
        expected = row[state + "_sha256"]
        if expected is None:
            absent.append(str(rel))
            continue
        source = snapshot / state / (str(rel) + ".snapshot")
        if source.resolve() != source or not source.is_file():
            raise ValueError("missing or symlinked snapshot source")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("snapshot hash mismatch: " + str(rel))
        mode = int(row["mode"], 8)
        if mode & ~0o777:
            raise ValueError("unsafe file mode")
        prepared.append((rel, data, mode))
    # All validation precedes the first write. Never overwrite an existing tree.
    destination.mkdir(parents=True, exist_ok=False)
    for rel, data, mode in prepared:
        target = destination / str(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        os.chmod(target, mode)
    return {"state": state, "files": len(prepared), "absent_in_base": absent,
            "base_commit": manifest["base_commit"], "live_changes": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[1] / "integrations/host153/20261015")
    parser.add_argument("--state", choices=("base", "current"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.snapshot, args.destination, args.state), ensure_ascii=False, indent=2))
