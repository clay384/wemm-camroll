"""Construct clients bound to a selected experiment or published snapshot."""
from pathlib import Path

from .experiment import get_experiment_profile
from .retrieval import make_retrieval_clients


def build_clients(experiment=None, memory=None, snapshot=None):
    from .qwen import QwenClient
    directory = Path(memory).resolve() if memory else None
    profile = get_experiment_profile(experiment) if experiment else None
    namespace = directory.name if directory else None
    if directory and (directory / 'CURRENT').is_file():
        from .library import Library
        saved = snapshot if snapshot is not None else Library(directory)
        saved_profile = getattr(saved, 'experiment', None)
        if profile and saved_profile and profile.name != saved_profile.name:
            raise ValueError('Requested experiment differs from the published snapshot')
        profile = saved_profile or profile
        namespace = getattr(saved, 'cache_namespace', None) or namespace
    config = profile.as_dict() if profile else None
    if config:
        config['cache_namespace'] = namespace
    embedding, reranker = make_retrieval_clients(config)
    options = dict(model=profile.generation_model, expected_limit_cny=600, request_namespace=namespace,
                   request_journal_path=directory / '.request-journal.sqlite') if profile else {}
    return QwenClient(**options), embedding, reranker, config
