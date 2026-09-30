"""Select and validate retrieval providers for an immutable experiment."""
from __future__ import annotations

import hashlib
import json

from .experiment import get_experiment


def resolve_experiment(experiment_config=None, cache_namespace=None):
    if isinstance(experiment_config, dict):
        experiment_config = dict(experiment_config)
        stored_namespace = experiment_config.pop("cache_namespace", None)
        if cache_namespace and stored_namespace and cache_namespace != stored_namespace:
            raise ValueError("Experiment cache namespace differs from the requested namespace")
        cache_namespace = cache_namespace or stored_namespace
    profile = get_experiment(experiment_config)
    if cache_namespace is not None and (not isinstance(cache_namespace, str) or not cache_namespace.strip()):
        raise ValueError("Experiment cache namespace must be a nonempty string")
    return profile, cache_namespace


def experiment_metadata(profile, cache_namespace):
    if profile is None:
        return {}
    return {"experiment_config": profile.as_dict(), "experiment_fingerprint": profile.fingerprint,
            "cache_namespace": cache_namespace, "embedding_provider": profile.embedding_backend,
            "embedding_model": profile.embedding_model, "embedding_dimension": profile.embedding_dimension,
            "reranker_model": profile.reranker_model, "reranker_fingerprint": profile.reranker_fingerprint,
            "text_chunk_tokens": profile.chunk_tokens}


def cache_key(values, profile=None, cache_namespace=None):
    if profile is not None:
        configuration = profile.as_dict()
        configuration.pop("embedding_fingerprint", None)
        configuration.pop("reranker_fingerprint", None)
        values = [configuration, cache_namespace, values]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_embedding(embedding, profile, *, require_fingerprint=True):
    if profile is None:
        if getattr(embedding, "provider", "wemm") != "wemm":
            raise ValueError("An explicit experiment configuration is required for this embedding provider")
        return None
    if getattr(embedding, "provider", None) != profile.embedding_backend:
        raise ValueError("Embedding provider differs from the experiment configuration")
    if getattr(embedding, "dimension", None) != profile.embedding_dimension:
        raise ValueError("Embedding dimension differs from the experiment configuration")
    if not require_fingerprint:
        return profile
    fingerprint = embedding.fingerprint
    if profile.embedding_fingerprint and profile.embedding_fingerprint != fingerprint:
        raise ValueError("Embedding fingerprint differs from the experiment configuration")
    reranker_fingerprint = ""
    if profile.reranker_model:
        reranker_fingerprint = embedding.reranker_fingerprint
        if profile.reranker_fingerprint and profile.reranker_fingerprint != reranker_fingerprint:
            raise ValueError("Reranker fingerprint differs from the experiment configuration")
    return profile.with_fingerprints(embedding_fingerprint=fingerprint,
                                     reranker_fingerprint=reranker_fingerprint)


def split_document(embedding, text):
    chunks = embedding.split_text(text)
    if (not isinstance(chunks, list) or not chunks
            or any(not isinstance(chunk, str) or not chunk for chunk in chunks)
            or "".join(chunks) != text):
        raise ValueError("Tokenizer split lost input text")
    return chunks


def make_retrieval_clients(experiment_config=None, *, cache_namespace=None):
    profile, cache_namespace = resolve_experiment(experiment_config, cache_namespace)
    if profile is None or profile.embedding_backend == "wemm":
        from .wemm import EmbeddingClient
        return EmbeddingClient(), None
    if not cache_namespace:
        raise ValueError("Qwen retrieval requires an isolated experiment cache namespace")
    from .qwen_retrieval import QwenRetrievalClient
    client = QwenRetrievalClient(cache_namespace=cache_namespace,
                                 expected_fingerprint=profile.embedding_fingerprint or None,
                                 expected_reranker_fingerprint=profile.reranker_fingerprint or None)
    return client, client
