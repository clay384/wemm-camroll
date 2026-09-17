"""Frozen deployment configuration for the independent chat service."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_ALBUM_ID = "eval-20260916-141627-camroll-e5a5fe2f"
DEFAULT_INDEX_VERSION = "bd64fa387f0c5d507ef845fb2493eed6"
DEFAULT_ALBUM_ROOT = Path("/home/realmagic/camroll-wemm/runtime/photobench-v2/albums")
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8772/camroll"
DEFAULT_SNAPSHOT_HASHES = {
    "records.json": "bd64fa387f0c5d507ef845fb2493eed6c7c7d23695480001314db4f278a89e3e",
    "search.sqlite": "700fa075a5d1e0addd807f5313d1d8dd9e22abd90ac33a780af4f44fd1b58586",
    "vectors.npz": "a5c80764b5bd8b8d21190df748c5b63c12f1e8004c08741320631ab092dd6584",
}


@dataclass(frozen=True)
class ChatConfig:
    """Configuration with intentionally fixed model and retrieval policy.

    Production construction ignores environment overrides. Tests may inject a
    temporary album root, static directory, Host allowlist and matching file
    hashes; the album identity, index, upstream, CamRoll configuration and
    Qwen model remain frozen.
    """

    album_root: Path = DEFAULT_ALBUM_ROOT
    upstream_url: str = DEFAULT_UPSTREAM_URL
    static_dir: Path | None = None
    album_id: str = DEFAULT_ALBUM_ID
    index_version: str = DEFAULT_INDEX_VERSION
    album_label: str = "PhotoBench album3 完整版（355图）"
    source_label: str = ".153 · album3-max"
    photo_count: int = 355
    identity_count: int = 8
    relationship_count: int = 6
    agent_config: str = "M4"
    model: str = "qwen3.7-flash"
    upstream_connect_timeout_s: float = 5.0
    upstream_write_timeout_s: float = 15.0
    upstream_pool_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 210.0
    allowed_hosts: tuple[str, ...] = (
        "192.168.0.100:8773", "127.0.0.1:8773", "localhost:8773", "[::1]:8773",
    )
    snapshot_hashes: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SNAPSHOT_HASHES)
    )

    def __post_init__(self) -> None:
        parsed = urlsplit(self.upstream_url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.port != 8772 or parsed.path.rstrip("/") != "/camroll"):
            raise ValueError("CamRoll chat upstream must be http://127.0.0.1:8772/camroll")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("CamRoll chat upstream must not contain credentials or parameters")
        if self.album_id != DEFAULT_ALBUM_ID or self.index_version != DEFAULT_INDEX_VERSION:
            raise ValueError("CamRoll chat only serves the frozen PhotoBench album")
        if self.agent_config != "M4" or self.model != "qwen3.7-flash":
            raise ValueError("CamRoll chat has a frozen M4 / qwen3.7-flash policy")
        if not self.allowed_hosts or any(not isinstance(value, str) or not value for value in self.allowed_hosts):
            raise ValueError("CamRoll chat requires an explicit Host allowlist")
        if set(self.snapshot_hashes) != {"records.json", "search.sqlite", "vectors.npz"}:
            raise ValueError("CamRoll chat requires all frozen snapshot hashes")
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
               for value in self.snapshot_hashes.values()):
            raise ValueError("CamRoll chat snapshot hashes must be lowercase SHA256 values")

    @classmethod
    def from_env(cls) -> "ChatConfig":
        """Return the fixed production policy; shell variables cannot redirect it."""
        package_root = Path(__file__).resolve().parents[1]
        return cls(
            album_root=DEFAULT_ALBUM_ROOT,
            upstream_url=DEFAULT_UPSTREAM_URL,
            static_dir=package_root / "static",
        )
