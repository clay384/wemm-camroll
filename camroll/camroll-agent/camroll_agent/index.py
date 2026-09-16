"""Publish a new WeMM index from cached captions, with no cloud generation."""
DEFAULT_EMBEDDING_MODEL="wemm"


def run(memory_dir, *, embedding_model="wemm", embedding_client=None):
    if embedding_model not in ("wemm","tencent/WeMM-Embedding-9B"):
        raise ValueError("This deployment uses WeMM embeddings exclusively")
    from .pipeline import reindex
    from .wemm import EmbeddingClient
    return reindex(memory_dir,embedding_client or EmbeddingClient())
