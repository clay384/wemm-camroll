"""Read-only access to one immutable CamRoll album snapshot."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

from .config import ChatConfig


IMAGE_ID_RE = re.compile(r"^img_[0-9a-f]{64}$")


class AlbumUnavailable(RuntimeError):
    pass


class IndexVersionChanged(RuntimeError):
    pass


class SnapshotIntegrityChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class AlbumState:
    status: str
    current_index_version: str | None
    image_count: int | None


class FrozenAlbum:
    """Resolve public image IDs without importing the CamRoll backend."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.root = config.album_root.resolve()
        self.directory = (self.root / config.album_id).resolve()
        if not self.directory.is_relative_to(self.root):
            raise ValueError("album directory escapes configured root")
        self.assets = (self.directory / "assets").resolve()
        if not self.assets.is_relative_to(self.directory):
            raise ValueError("album assets directory escapes the frozen album")
        self._cache_lock = threading.Lock()
        self._records_signature: tuple[int, int] | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._integrity_signatures: dict[str, tuple[int, int, int]] = {}
        self._asset_signatures: dict[str, tuple[int, int, int]] = {}

    def _current_version(self) -> str:
        current = self.directory / "CURRENT"
        try:
            value = current.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AlbumUnavailable("album publication is unavailable") from exc
        if value != self.config.index_version:
            raise IndexVersionChanged("published album index changed")
        return value

    def _snapshot_root(self) -> Path:
        version = self._current_version()
        snapshot_root = (self.directory / "snapshots" / version).resolve()
        if not snapshot_root.is_relative_to(self.directory):
            raise AlbumUnavailable("snapshot is unavailable")
        return snapshot_root

    def _snapshot_path(self, filename: str) -> Path:
        snapshot_root = self._snapshot_root()
        path = (snapshot_root / filename).resolve()
        if not path.is_relative_to(snapshot_root):
            raise AlbumUnavailable("snapshot file is unavailable")
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise AlbumUnavailable("snapshot file is unavailable") from exc
        return digest.hexdigest()

    def _verify_snapshot_file(self, filename: str) -> Path:
        path = self._snapshot_path(filename)
        try:
            stat = path.stat()
        except OSError as exc:
            raise AlbumUnavailable("snapshot file is unavailable") from exc
        if not path.is_file():
            raise AlbumUnavailable("snapshot file is unavailable")
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self._cache_lock:
            if self._integrity_signatures.get(filename) == signature:
                return path
        if self._sha256(path) != self.config.snapshot_hashes[filename]:
            raise SnapshotIntegrityChanged("frozen snapshot hash changed")
        with self._cache_lock:
            self._integrity_signatures[filename] = signature
        return path

    def verify_integrity(self) -> None:
        for filename in ("records.json", "search.sqlite", "vectors.npz"):
            self._verify_snapshot_file(filename)

    def records(self) -> dict[str, dict[str, Any]]:
        path = self._verify_snapshot_file("records.json")
        try:
            stat = path.stat()
        except OSError as exc:
            raise AlbumUnavailable("snapshot records are unavailable") from exc
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            if self._records_signature == signature:
                return self._records
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise AlbumUnavailable("snapshot records are invalid") from exc
            rows = value.get("images") if isinstance(value, dict) else None
            if not isinstance(rows, list):
                raise AlbumUnavailable("snapshot image records are invalid")
            result: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                image_id = row.get("id") or row.get("image_id")
                if not isinstance(image_id, str) or not IMAGE_ID_RE.fullmatch(image_id):
                    continue
                if image_id in result:
                    raise AlbumUnavailable("snapshot contains duplicate image IDs")
                result[image_id] = row
            if len(result) != self.config.photo_count:
                raise AlbumUnavailable("snapshot image count does not match the frozen album")
            self._records = result
            self._records_signature = signature
            return self._records

    def inspect(self) -> AlbumState:
        try:
            self.verify_integrity()
            rows = self.records()
            return AlbumState("ready", self.config.index_version, len(rows))
        except IndexVersionChanged:
            try:
                current = (self.directory / "CURRENT").read_text(encoding="utf-8").strip()
            except OSError:
                current = None
            return AlbumState("index_changed", current, None)
        except AlbumUnavailable:
            return AlbumState("unavailable", None, None)
        except SnapshotIntegrityChanged:
            return AlbumState("integrity_changed", self.config.index_version, None)

    def ensure_ready(self) -> None:
        self.verify_integrity()
        self.records()

    def media_path(self, image_id: str) -> Path:
        # A media response must belong to the exact published snapshot, not
        # merely to a still-valid records.json beside changed search assets.
        self.verify_integrity()
        if not IMAGE_ID_RE.fullmatch(image_id):
            raise KeyError(image_id)
        row = self.records().get(image_id)
        if row is None:
            raise KeyError(image_id)
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or image_id != "img_" + asset_id:
            raise KeyError(image_id)
        # Only published, per-album assets are eligible. source_path and other
        # metadata paths deliberately never become web-readable.
        candidates = (row.get("original_path"), row.get("path"))
        for value in candidates:
            if not isinstance(value, str):
                continue
            try:
                path = Path(value).resolve(strict=True)
            except OSError:
                continue
            expected_asset_root = (self.assets / asset_id).resolve()
            if (path.is_file() and path.is_relative_to(self.assets)
                    and path.is_relative_to(expected_asset_root)):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                with self._cache_lock:
                    verified = self._asset_signatures.get(asset_id) == signature
                if not verified:
                    if self._sha256(path) != asset_id:
                        raise SnapshotIntegrityChanged("frozen media content hash changed")
                    try:
                        after = path.stat()
                    except OSError:
                        continue
                    after_signature = (after.st_ino, after.st_size, after.st_mtime_ns)
                    if after_signature != signature:
                        raise SnapshotIntegrityChanged("frozen media changed during verification")
                    with self._cache_lock:
                        self._asset_signatures[asset_id] = signature
                return path
        raise KeyError(image_id)

    def public_metadata(self) -> dict[str, Any]:
        state = self.inspect()
        return {
            "id": self.config.album_id,
            "label": self.config.album_label,
            "source": self.config.source_label,
            "photo_count": self.config.photo_count,
            "identity_count": self.config.identity_count,
            "relationship_count": self.config.relationship_count,
            "index_version": self.config.index_version,
            "current_index_version": state.current_index_version,
            "index_status": state.status,
            "config": self.config.agent_config,
            "model": self.config.model,
            "is_default": True,
        }
