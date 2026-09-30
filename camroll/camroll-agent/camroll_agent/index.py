"""Publish a new WeMM index from cached captions, with no cloud generation."""
DEFAULT_EMBEDDING_MODEL="wemm"


def run(memory_dir, *, embedding_model=None, embedding_client=None, experiment=None):
    from pathlib import Path
    from .pipeline import reindex
    from .library import Library
    from .experiment import get_experiment_profile
    from .retrieval import make_retrieval_clients
    directory = Path(memory_dir).expanduser().resolve()
    library = Library(directory) if (directory / 'CURRENT').is_file() else None
    profile = get_experiment_profile(experiment) if experiment else library.experiment if library else None
    config = profile.as_dict() if profile else None
    if config:
        config['cache_namespace'] = library.cache_namespace if library else directory.name
    if embedding_model and embedding_model not in (
            profile.embedding_backend if profile else 'wemm', profile.embedding_model if profile else 'tencent/WeMM-Embedding-9B'):
        raise ValueError('Embedding model differs from the published snapshot')
    client = embedding_client or make_retrieval_clients(config)[0]
    return reindex(memory_dir, client, experiment_config=config)
