import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from four_suite import validate_run, source_manifest


def test_frozen_profile_rejects_modified_instances():
    from dataclasses import replace
    import pytest
    from camroll_agent.experiment import Q37, get_experiment_profile
    with pytest.raises(ValueError, match='frozen protocol'):
        get_experiment_profile(replace(Q37, embedding_dimension=4096))


def test_formal_delivery_requires_all_questions_turns_and_scores():
    turn = {'judge': {'score': 0}, 'task_judge': {'actual_action': 'none'}}
    items = [{'qa_id': str(i), 'runtime_turns': [copy.deepcopy(turn)], 'judge_status': 'completed', 'judge': {'score': 0}} for i in range(100)]
    for item in items[:11]:
        item['runtime_turns'].append(copy.deepcopy(turn))
    state = {'status': 'completed_with_errors', 'items': items, 'media_mapping': {str(i): {} for i in range(355)}}
    state['qa_plan'] = [{'qa_id': item['qa_id'], 'planned_turn_count': len(item['runtime_turns'])} for item in items]
    assert validate_run(state, formal=True)  # Incorrect responses remain legitimate observations.
    for mutation in ('missing_question', 'missing_turn', 'missing_score', 'missing_image'):
        broken = copy.deepcopy(state)
        if mutation == 'missing_question':
            broken['items'].pop()
        elif mutation == 'missing_turn':
            broken['items'][0]['runtime_turns'].pop()
        elif mutation == 'missing_score':
            broken['items'][0]['judge']['score'] = None
        else:
            broken['media_mapping'].pop('0')
        assert not validate_run(broken, formal=True)


def test_source_freeze_ignores_generated_bytecode_and_detects_source_change(tmp_path):
    source = tmp_path / 'a.py'
    source.write_text('first')
    before = source_manifest(tmp_path)
    (tmp_path / '__pycache__').mkdir()
    (tmp_path / '__pycache__/a.pyc').write_bytes(b'generated')
    assert source_manifest(tmp_path) == before
    source.write_text('second')
    assert source_manifest(tmp_path) != before


def test_formal_freeze_checks_current_media_and_recorded_run_inputs(monkeypatch):
    import pytest
    from types import SimpleNamespace
    import camroll_eval.dataset as dataset
    from four_suite import verify_frozen_inputs
    frozen = {'dataset_sha256': 'original-photo-bytes', 'files': [{'sha256': 'original'}]}
    current = copy.deepcopy(frozen)
    monkeypatch.setattr(dataset, 'Dataset', lambda source: SimpleNamespace(integrity=lambda: current))
    verify_frozen_inputs({'input_integrity': frozen}, {'input_integrity': frozen})
    with pytest.raises(RuntimeError, match='run_input_integrity_differs'):
        verify_frozen_inputs({'input_integrity': frozen}, {'input_integrity': {'dataset_sha256': 'other'}})
    current['files'][0]['sha256'] = 'modified-photo'
    with pytest.raises(RuntimeError, match='frozen_album_inputs_changed'):
        verify_frozen_inputs({'input_integrity': frozen})


def test_readonly_report_exposes_no_mutation_routes(tmp_path):
    from fastapi.testclient import TestClient
    from camroll_eval.service import create_app
    app = create_app(tmp_path / 'evaluation', tmp_path / 'source', tmp_path / 'albums', read_only=True, mount_agent=False)
    assert all(not getattr(route, 'methods', set()) - {'GET', 'HEAD', 'OPTIONS'} for route in app.routes)
    assert not any(getattr(route, 'path', '').startswith('/api/ask') for route in app.routes)
    client = TestClient(app)
    assert client.get('/api/health').json()['read_only'] is True
    assert client.post('/api/runs', json={'config': 'M4'}).status_code == 405
    assert client.post('/agent/api/ask', json={}).status_code == 405
