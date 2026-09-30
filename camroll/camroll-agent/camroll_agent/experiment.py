"""Immutable model and protocol settings for the four comparison groups."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from types import MappingProxyType


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    generation_model: str
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int
    reranker_model: str | None = None
    chunk_tokens: int = 512
    preprocessing: str = "wemm-original"
    embedding_fingerprint: str = ""
    reranker_fingerprint: str = ""
    protocol_version: str = "camroll-four-models-v1"
    mode: str = "M4"
    agent_concurrency: int = 1
    judge_concurrency: int = 4
    caption_concurrency: int = 4
    deadline_s: int = 600
    recall_top_k: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 30
    query_role: str = "query"
    document_role: str = "document"
    temperature: float = 0.0
    enable_thinking: bool = False
    judge_model: str = "doubao-seed-2.0-lite"
    image_evidence_judge: bool = False

    @property
    def embedding_provider(self) -> str:
        return self.embedding_backend

    @property
    def dimension(self) -> int:
        return self.embedding_dimension

    @property
    def reranker(self) -> str | None:
        return self.reranker_model

    def as_dict(self) -> dict:
        return asdict(self)

    to_dict = as_dict

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def profile_hash(self) -> str:
        return self.fingerprint

    def with_fingerprints(self, *, embedding_fingerprint: str,
                          reranker_fingerprint: str = "") -> ExperimentProfile:
        if not embedding_fingerprint or (self.reranker_model and not reranker_fingerprint):
            raise ValueError("Runtime model fingerprints must be nonempty")
        return replace(self, embedding_fingerprint=embedding_fingerprint,
                       reranker_fingerprint=reranker_fingerprint)


Q37 = ExperimentProfile("Q37", "qwen3.7-flash", "qwen", "Qwen3-VL-Embedding-2B", 2048,
                        "Qwen3-VL-Reranker-2B", 2048, "qwen-vl-vendor-locked-rope-deltas")
Q38 = replace(Q37, name="Q38", generation_model="qwen3.8-flash")
W37 = ExperimentProfile("W37", "qwen3.7-flash", "wemm", "wemm", 4096)
W38Max = replace(W37, name="W38Max", generation_model="qwen3.8-max")
PROFILES = MappingProxyType({profile.name: profile for profile in (Q37, Q38, W37, W38Max)})
EXPERIMENT_ORDER = tuple(PROFILES)


def get_experiment_profile(value: str | dict | ExperimentProfile) -> ExperimentProfile:
    if isinstance(value, ExperimentProfile):
        return get_experiment_profile(value.as_dict())
    if isinstance(value, dict):
        profile = ExperimentProfile(**value)
        expected = PROFILES.get(profile.name)
        if expected is None or replace(profile, embedding_fingerprint="", reranker_fingerprint="") != expected:
            raise ValueError("Experiment configuration differs from the frozen protocol")
        return profile
    try:
        return PROFILES[value]
    except (KeyError, TypeError):
        raise ValueError("Unknown experiment profile: {!r}".format(value)) from None


def get_experiment(value: str | dict | ExperimentProfile | None = None) -> ExperimentProfile | None:
    return None if value is None else get_experiment_profile(value)


ExperimentConfig = ExperimentProfile
