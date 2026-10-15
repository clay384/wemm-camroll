"""Read-only service verification for the frozen self-hosted experiment."""
import hashlib
import json
from pathlib import Path
import urllib.request


def verify_service(runtime):
    root = Path(runtime)
    connection = json.loads((root / 'connection.json').read_text())
    evidence = json.loads((root / 'pricing-evidence.json').read_text())
    manifest = root / 'deployment-manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != evidence['deployment_manifest_sha256']:
        raise RuntimeError('selfhost_deployment_manifest_changed')
    expected = json.loads(manifest.read_text())
    base = connection['base_url']
    from camroll_agent.minicpm_selfhost import MODEL
    if base != 'http://127.0.0.1:18080/v1':
        raise ValueError('Unexpected self-hosted tunnel endpoint')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def read(path):
        with opener.open(base[:-3] + path, timeout=10) as response:
            return json.load(response)
    health, props, slots, models = (read(path) for path in ('/health', '/props', '/slots', '/v1/models'))
    if (health.get('status') != 'ok' or props.get('model_alias') != MODEL
            or props.get('build_info') != expected['build_info'] or props.get('total_slots') != 1
            or props.get('default_generation_settings', {}).get('n_ctx') != 262144
            or props.get('modalities', {}).get('vision') is not True
            or len(slots) != 1 or slots[0].get('is_processing') is not False
            or not any(row.get('id') == MODEL for row in models.get('data', []))):
        raise RuntimeError('selfhost_service_busy_or_configuration_changed')
    template_hash = hashlib.sha256(json.dumps(props.get('chat_template'), sort_keys=True).encode()).hexdigest()
    if template_hash != expected['chat_template_sha256']:
        raise RuntimeError('selfhost_chat_template_changed')
    return {'status': 'passed', 'model': MODEL, 'build_info': props['build_info'],
            'context_limit': 262144, 'idle': True, 'deployment_manifest_sha256': evidence['deployment_manifest_sha256']}
