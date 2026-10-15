"""Explicit isolated MiniCPM runtime; original album and GPU registry are shared."""
import os
from pathlib import Path

from four_config import BASE, GPU_UUID, PYTHON, RUNTIME as HISTORY_RUNTIME, SOURCE
from four_config import configure as previous_configure

ALBUMS = HISTORY_RUNTIME / 'albums'
SOURCE_RUN_ID = 'w38max-formal-20260930-122300-813923'
SOURCE_ALBUM_ID = 'eval-' + SOURCE_RUN_ID
SNAPSHOT = 'bcc1cd1176ab39697c789a4f44c7426e'
BASELINE_RUNTIME = BASE / 'runtime/photobench-flash-gate-20261006f'
RUNTIME_PREFIX = 'minicpm47-thinking-m4-off-'
SDK_RUNTIME_PREFIX = 'minicpm47-sdk-defaults-m4-off-'
SELFHOST_RUNTIME_PREFIX = "minicpm47-q6-thinking-m4-off-"
RUNTIME_PREFIXES = (RUNTIME_PREFIX, SDK_RUNTIME_PREFIX, SELFHOST_RUNTIME_PREFIX)
FACTS_AND_INTERFACE_POLICY = 'facts-and-interface-v1'
INTERFACE_ONLY_POLICY = 'interface-only-v2'


def acceptance_policy(profile):
    policy = getattr(profile, 'acceptance_policy', FACTS_AND_INTERFACE_POLICY)
    if (policy not in (FACTS_AND_INTERFACE_POLICY, INTERFACE_ONLY_POLICY)
            or ((policy == INTERFACE_ONLY_POLICY)
                != (profile.protocol_version in ('minicpm47-sdk-defaults-m4-off-v2', 'minicpm47-q6-thinking-m4-off-v1')))):
        raise ValueError('Acceptance policy differs from its versioned inference protocol')
    return policy


def validate_runtime(runtime):
    path = Path(runtime).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('MiniCPM runtime must be an absolute regular path')
    resolved = path.resolve()
    if resolved.parent != (BASE / 'runtime').resolve() or not resolved.name.startswith(RUNTIME_PREFIXES):
        raise ValueError('MiniCPM requires its independent runtime prefix under the project runtime directory')
    if resolved == BASELINE_RUNTIME.resolve() or resolved == HISTORY_RUNTIME.resolve():
        raise ValueError('Historical runtime cannot be reused')
    return resolved


def request_profile(runtime):
    """Read the explicit new request contract; absence retains only legacy defaults."""
    from camroll_eval.common import read_json
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    path = Path(runtime) / 'request-profile.json'
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Request profile must be a regular file')
        profile = MiniCPMInferenceConfig(**read_json(path))
    else:
        if Path(runtime).name.startswith((SDK_RUNTIME_PREFIX, SELFHOST_RUNTIME_PREFIX)):
            raise ValueError('SDK-defaults runtime requires an explicit request-profile.json')
        profile = MiniCPMInferenceConfig()
    if (Path(runtime).name.startswith(SDK_RUNTIME_PREFIX)
            != (getattr(profile, 'request_profile', None) == 'sdk-defaults-v1')):
        raise ValueError('Runtime prefix differs from frozen request profile')
    if Path(runtime).name.startswith(SELFHOST_RUNTIME_PREFIX) != (profile.provider == "llamacpp-selfhost"):
        raise ValueError("Self-hosted runtime prefix differs from profile")
    return profile


def protocol_profile(runtime, protocol):
    """Restore the complete frozen profile, never silently adopt new defaults."""
    from camroll_eval.minicpm_experiment import MiniCPMInferenceConfig
    if 'inference_config' in protocol:
        profile = MiniCPMInferenceConfig(**protocol['inference_config'])
    else:
        # Old private fixtures may predate the full profile field. This branch
        # cannot construct the new SDK-defaults protocol.
        profile = MiniCPMInferenceConfig(decision_mode=protocol['decision_mode'], json_mode=protocol['json_mode'])
    if (profile.decision_mode != protocol['decision_mode'] or profile.json_mode != protocol['json_mode']
            or ('protocol_fingerprint' in protocol and protocol['protocol_fingerprint'] != profile.fingerprint)):
        raise ValueError('Frozen protocol profile or fingerprint differs')
    if (Path(runtime) / 'request-profile.json').exists() or Path(runtime).name.startswith(SDK_RUNTIME_PREFIX):
        requested = request_profile(runtime).as_dict()
        selected = profile.as_dict()
        if getattr(profile, 'request_profile', None) == 'sdk-defaults-v1':
            if protocol.get('protocol_fingerprint') != profile.fingerprint:
                raise ValueError('SDK-defaults protocol requires its exact fingerprint')
        else:
            for name in ('decision_mode', 'json_mode'):
                requested.pop(name)
                selected.pop(name)
        if requested != selected:
            raise ValueError('Selected protocol changed the requested inference contract')
    return profile


def request_matches_profile(profile, payload):
    """Check transmitted fields separately from observed response behavior."""
    return profile.matches_request(payload)


def annotation_review(runtime, profile, *, image_paths=None):
    """Bind the independent seven-image review, using Library paths on this host."""
    if getattr(profile, 'request_profile', None) not in ('sdk-defaults-v1', 'selfhost-q6-v1'):
        return None
    from camroll_eval.common import read_json, sha256
    relative = 'acceptance/annotation-review-sdk-defaults.json'
    path = Path(runtime) / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError('Independent SDK-defaults annotation review is required')
    review = read_json(path)
    fixtures = review.get('fixtures') or []
    if (review.get('schema_version') != 'minicpm-sdk-defaults-private-annotation-review-v1'
            or review.get('new_protocol') != profile.protocol_version
            or review.get('formal_qa_gt_accessed') is not False
            or review.get('original_fixtures_modified') is not False
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
            or len(fixtures) != 7 or [row.get('index') for row in fixtures] != list(range(7))
            or any(not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64 for row in fixtures)):
        raise ValueError('Independent annotation review has invalid identity or image bindings')
    if image_paths is not None and (len(image_paths) != 7 or
            any(sha256(image) != row['sha256'] for image, row in zip(image_paths, fixtures))):
        raise ValueError('Independently reviewed photos differ from original Library assets')
    return {'path': relative, 'sha256': sha256(path),
            'fixture_sha256': [row['sha256'] for row in fixtures]}


def configure(release, runtime):
    from eval_config import _protected_json
    import sys
    sys.path.insert(0, str(Path(release) / "camroll/camroll-agent"))
    runtime = validate_runtime(runtime)
    previous_configure(release)
    selfhost = request_profile(runtime).provider == 'llamacpp-selfhost'
    secrets = _protected_json(runtime / 'secrets.json') if not selfhost else {}
    key = secrets.get('MODELBEST_API_KEY')
    if not selfhost and (not key or not key.strip() or '\n' in key or '\r' in key):
        raise RuntimeError('Protected MiniCPM key is missing or invalid')
    values = {'MODELBEST_API_KEY': key, 'CAMROLL_ALBUM_ROOT': str(ALBUMS),
        'CAMROLL_EVAL_ALBUM_ROOT': str(ALBUMS), 'CAMROLL_EVAL_ROOT': str(runtime / 'evaluation'),
        'CAMROLL_MINICPM_PRICING_EVIDENCE': str(runtime / 'pricing-evidence.json'),
        'CAMROLL_BUDGET_CNY': '600', 'CAMROLL_EXPECTED_BUDGET_CNY': '600',
        'CUDA_VISIBLE_DEVICES': GPU_UUID}
    if selfhost:
        values.pop("MODELBEST_API_KEY")
        from camroll_eval.common import read_json
        values["CAMROLL_MINICPM_BASE_URL"] = read_json(runtime / "connection.json")["base_url"]
    os.environ.update(values)
    # Return operational paths only, never the secret value.
    return {key: value for key, value in values.items() if key != 'MODELBEST_API_KEY'}
