"""Compatibility entry point for the resumable Qwen/WeMM pipeline."""
from .pipeline import inspect


def run(spec_path, output_dir, *, vlm=None, backend="qwen", model=None,
        max_images=None, resume=True, checkpoint_every=50, embedding_client=None, experiment=None):
    if backend != "qwen":
        raise ValueError("This deployment uses the Qwen cloud provider")
    from .pipeline import build
    from .runtime_clients import build_clients
    cloud, embedding, reranker, config = build_clients(experiment, output_dir)
    if model is not None and model != cloud.model:
        raise ValueError('Generation model differs from the selected experiment')
    return build(spec_path, output_dir, vlm or cloud, embedding_client or embedding,
                 max_images=max_images, resume=resume, experiment_config=config)
