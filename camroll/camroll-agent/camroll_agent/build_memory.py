"""Compatibility entry point for the resumable Qwen/WeMM pipeline."""
from .pipeline import inspect


def run(spec_path, output_dir, *, vlm=None, backend="qwen", model=None,
        max_images=None, resume=True, checkpoint_every=50, embedding_client=None):
    if backend != "qwen" or model not in (None,"qwen3.7-flash"):
        raise ValueError("This deployment uses qwen3.7-flash exclusively")
    from .pipeline import build
    from .qwen import QwenClient
    from .wemm import EmbeddingClient
    return build(spec_path,output_dir,vlm or QwenClient(),embedding_client or EmbeddingClient(),max_images=max_images,resume=resume)
