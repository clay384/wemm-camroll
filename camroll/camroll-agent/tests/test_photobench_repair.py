"""One-shot repair safety, provenance and partial-result accounting; no paid calls."""
import copy
import json
import pytest
from test_photobench_eval import source, Backend, FakeJudge
from camroll_eval.common import atomic_json, read_json, sha256, object_hash
from camroll_eval.runner import Runner
from camroll_eval.repair import RepairRunner, compose, validate_selection
from camroll_eval.dataset import Dataset
from camroll_eval.metrics import summary
from camroll_eval.repository import Repository


def parent_fixture(tmp_path, source):
    root = tmp_path / 'eval'
    parent = Runner(root, source, backend_factory=Backend, judge_factory=FakeJudge, sample_resources=False).run(run_id='original', strict=False)
    row = parent['items'][1]
    row.update(execution_status='error', judge={'score': None}, judge_status='incomplete')
    row['runtime_turns'] = row['runtime_turns'][:1]
    parent['status'] = 'completed_with_errors'
    parent['summary'] = summary(parent)
    atomic_json(root / 'runs/original/run.json', parent)
    return root, parent


def test_repair_only_selected_question_full_dialogue_and_no_second_attempt(tmp_path, source, monkeypatch):
    root, parent = parent_fixture(tmp_path, source)
    original = root / 'runs/original/run.json'
    digest = sha256(original)
    backend = Backend()
    monkeypatch.setattr('camroll_eval.repair.Dataset', lambda p: Dataset(p, strict=False))
    runner = RepairRunner(root, source, backend_factory=lambda: backend, judge_factory=FakeJudge, sample_resources=False)
    merged = runner.run_repair('original', ['two'], run_id='repair-one', verify_frozen=False)
    assert len(backend.questions) == 2
    assert backend.questions[0][1] == []
    assert backend.questions[1][1]  # full conversation, not a naked final turn
    assert sha256(original) == digest
    assert merged['items'][0] == parent['items'][0]
    assert merged['repaired_valid_count'] == 1
    assert merged['summary']['judge_valid_count'] == 2
    assert merged['summary']['executed_turn_count'] == 3
    assert merged['attempted_turn_count_total'] == 4
    assert merged['summary']['benchmark_e2e_latency_excluding_judge_ms'] == parent['summary']['benchmark_e2e_latency_excluding_judge_ms']
    assert merged['unchanged_item_hashes']['one'] == object_hash(parent['items'][0])
    with pytest.raises(ValueError, match='already has a repair'):
        RepairRunner(root, source, backend_factory=lambda: backend, judge_factory=FakeJudge, sample_resources=False).run_repair(
            'original', ['two'], run_id='different-id', verify_frozen=False)
    assert len(backend.questions) == 2


def test_selection_refuses_valid_unknown_duplicate(tmp_path, source):
    _, parent = parent_fixture(tmp_path, source)
    for ids in [['one'], ['absent'], ['two', 'two'], []]:
        with pytest.raises(ValueError): validate_selection(parent, ids)


def test_composite_retains_failed_attempt_and_missing_score(tmp_path, source):
    _, parent = parent_fixture(tmp_path, source)
    repaired = copy.deepcopy(parent['items'][1])
    repair = {'qa_ids': ['two'], 'items': [repaired], 'run_id': 'failed-repair', 'status': 'completed_with_errors',
              'parent_sha256': 'hash', 'index_version': parent['index_version']}
    merged = compose(parent, repair)
    assert merged['repaired_valid_count'] == 0
    assert merged['status'] == 'completed_with_errors'
    assert merged['summary']['judge_invalid_count'] == 1
    assert merged['summary']['unexecuted_turn_count'] == 1
    repaired['mapping_error_count'] = 1
    with pytest.raises(ValueError, match='unknown delivered'): compose(parent, repair)
    repaired['mapping_error_count'] = 0
    repair['index_version'] = 'other'
    with pytest.raises(ValueError, match='snapshot'): compose(parent, repair)


def test_invalid_judge_and_nonterminal_repairs_cannot_be_success(tmp_path, source):
    _, parent = parent_fixture(tmp_path, source)
    row = copy.deepcopy(parent['items'][1]); row['execution_status'] = 'complete'; row['judge_status'] = 'failed'
    repair = {'qa_ids': ['two'], 'items': [row], 'run_id': 'repair', 'status': 'completed_with_errors',
              'parent_sha256': 'hash', 'index_version': parent['index_version']}
    assert compose(parent, repair)['repaired_valid_count'] == 0
    repair['status'] = 'evaluating'
    with pytest.raises(ValueError, match='terminal'): compose(parent, repair)


def test_reuse_import_keeps_bytes_and_omits_full_chain_time(tmp_path):
    origin = tmp_path / 'input'; origin.mkdir()
    state = {'run_id': 'reuse', 'album_id': 'album3-max', 'qa_set': '100qa-full', 'mode': 'reuse',
        'items': [{'qa_id': str(i), 'judge': {'score': 2}} for i in range(100)],
        'input_integrity': {'qa_sha256': 'f625ae9e26b4666c277b424bd472b6790baf0d6773e9020b20eff985d4cc2eb0',
          'dataset_sha256': '120e2e7273e731ed24fd24cdf477ce81e7c6b71e43554b0366aa8677b8098627'},
        'summary': {'benchmark_e2e_latency_excluding_judge_ms': 999}}
    atomic_json(origin / 'run.json', state)
    repo = Repository(tmp_path / 'eval'); repo.import_history(origin)
    assert sha256(repo.path('reuse')) == sha256(origin / 'run.json')
    result = repo.load('reuse')
    assert result['original_summary']['benchmark_e2e_latency_excluding_judge_ms'] == 999
    assert result['summary']['benchmark_e2e_latency_excluding_judge_ms'] is None
    assert result['summary']['execution_error_count'] is None
    state['input_integrity']['dataset_sha256'] = 'different'; atomic_json(origin / 'run.json', state)
    with pytest.raises(ValueError, match='approved'): repo.import_history(origin)


def test_caption_only_adds_one_text_vector_and_keeps_events_and_original(tmp_path, monkeypatch):
    import contextlib
    import hashlib
    from camroll_agent.pipeline import _publish
    from camroll_eval.repair import repair_caption, file_hashes
    root, albums = tmp_path / 'eval', tmp_path / 'albums'
    photo = tmp_path / 'one.jpg'; photo.write_bytes(b'picture')
    image_id = 'img_' + sha256(photo)
    images = [{'id': image_id, 'asset_id': sha256(photo), 'path': str(photo), 'kind': 'image',
               'caption': '', 'visible_text': '', 'event_id': None},
              {'id': 'two', 'kind': 'image', 'caption': 'other', 'visible_text': '', 'event_id': 'event'}]
    events = [{'id': 'event', 'kind': 'event', 'name': 'old event', 'image_ids': ['two']}]
    metadata = {'embedding_fingerprint': 'frozen', 'caption_count': 1, 'text_embedding_count': 1, 'image_count': 2}
    version = _publish(albums / 'original', images, events,
        {'image': [image_id, 'two'], 'caption': ['two'], 'event': ['event']},
        {'image': [[1, 0], [0, 1]], 'caption': [[0, 1]], 'event': [[0, 1]]}, metadata)
    parent = {'album_id': 'original', 'index_version': version, 'build_summary': {'errors': [{'stage': 'caption', 'image_id': image_id}]}}
    parent_path = root / 'runs/original/run.json'; atomic_json(parent_path, parent)
    before = file_hashes(albums / 'original')
    calls = []
    class Embedding:
        fingerprint = 'frozen'
        def capture_calls(self, callback): return contextlib.nullcontext()
        def split_text(self, text): return [text]
        def encode(self, **kwargs): calls.append('encode'); return [1, 0]
    class Qwen:
        def capture_calls(self, callback): return contextlib.nullcontext()
    class CaptionBackend:
        qwen = Qwen(); embedding = Embedding()
        def budget(self): return {'estimated_spent_cny': 0}
    def caption(*args):
        calls.append('caption')
        return {'caption': 'repaired', 'visible_text': '', 'location': '', 'people': []}
    monkeypatch.setattr('camroll_eval.repair.verify_core', lambda p: {})
    monkeypatch.setattr('camroll_eval.runner.CamrollBackend', CaptionBackend)
    monkeypatch.setattr('camroll_agent.pipeline._caption', caption)
    monkeypatch.setattr('camroll_agent.pipeline._manifest', lambda p: (None, None, None))
    result = repair_caption(root, albums, 'original', repair_id='single-photo')
    assert calls == ['caption', 'encode']
    assert result['caption_count'] == result['text_embedding_count'] == 2
    assert result['event_membership_repaired'] is False
    assert file_hashes(albums / 'original') == before
    derived = read_json(root / 'caption_repairs/original/derived_memory/snapshots' / result['derived_index_version'] / 'records.json')
    assert derived['images'][0]['event_id'] is None
    assert derived['events'] == events
    assert derived['metadata']['text_coverage'] == 1.0
    with pytest.raises(FileExistsError): repair_caption(root, albums, 'original', repair_id='second')
    assert calls == ['caption', 'encode']


def test_original_asset_hash_is_not_the_derived_view_hash(tmp_path):
    from PIL import Image
    from camroll_eval.repair import validate_photo
    raw = tmp_path / 'original.jpg'; view = tmp_path / 'view.jpg'
    Image.new('RGB', (12, 16), 'red').save(raw, quality=80)
    with Image.open(raw) as image:
        image.convert('RGB').save(view, format='JPEG', quality=95, subsampling=0)
    row = {'asset_id': sha256(raw), 'path': str(view), 'sources': [{'source_path': str(raw)}]}
    assert sha256(raw) != sha256(view)
    assert validate_photo(row)['derivation_verified'] is True
    view.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='derived'): validate_photo(row)
