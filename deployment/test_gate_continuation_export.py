import copy
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate_continuation_export as module
from gate_config import ORDER, SNAPSHOT
from test_gate_suite import complete_run
from camroll_agent.budget import BudgetLedger, RequestJournal
from camroll_eval.common import atomic_json, read_json, sha256


def formal(group, run_id, suite_id):
    run = complete_run()
    run.update(run_id=run_id, suite_id=suite_id, qa_count=100, planned_turn_count=111,
        protocol_label=module.PROTOCOL, gate_summary_version=module.PROTOCOL,
        index_version=SNAPSHOT, index_binding={'snapshot_version': SNAPSHOT},
        inference_config={'group': group}, summary={'judge_valid_count': 100,
        'quality_distribution_planned': {'correct': 0, 'partial': 0, 'incorrect': 100, 'unscored': 0},
        'quality_correct_planned_rate': 0., 'quality_mean_fixed_planned': 0.,
        'quality_mean_scored': 0., 'retrieval_metric_count': 81}, wall_ms=1000)
    for item in run['items']:
        item['final_turn_executed'] = True
        for index, turn in enumerate(item['runtime_turns']):
            turn.update(index=index, raw_attempt_path='turn_attempts/' + item['qa_id'] + '/' + str(index) + '.json')
    return run


def add_call(directory, ledger, key, uncertain=False):
    journal = RequestJournal(directory / 'request-journal.sqlite')
    request_id = key + ':request'
    claim = journal.claim(request_id, key + '-payload', lambda: ledger.reserve(.25, {'model': 'qwen3.7-flash', 'role': 'agent'}))
    if uncertain:
        journal.fail(request_id, 'transport_error', uncertain=True)
        ledger.finish(claim['call_id'], None)
    else:
        journal.complete(request_id, {'synthetic': True})
        ledger.finish(claim['call_id'], .1)
    with (directory / 'model_calls.jsonl').open('a') as stream:
        stream.write(json.dumps({'call_id': claim['call_id'], 'request_id': request_id,
            'prompt_tokens': 10, 'completion_tokens': 1, 'call_type': 'agent', 'duration_ms': 7}) + '\n')
    return claim['call_id'], request_id


@pytest.fixture
def setup(tmp_path, monkeypatch):
    parent = tmp_path / 'f'
    child = tmp_path / 'g'
    monkeypatch.setattr(module, 'PARENT_RUNTIME', parent)
    ledger = BudgetLedger(tmp_path / 'budget.sqlite', 600)
    old_off = parent / 'evaluation/runs/old-off/run.json'
    off = formal('F37-Off', 'old-off', 'old-suite')
    atomic_json(old_off, off)
    add_call(old_off.parent, ledger, 'old-off')
    return parent, child, ledger, old_off


def test_reference_is_immutable_ancestry_only_and_cannot_overwrite(setup):
    parent, child, ledger, original = setup
    before = {path: path.read_bytes() for path in parent.rglob('*') if path.is_file()}
    result = module.create_reference_run(original, child / 'evaluation', 'new-off', 'new-suite', sha256(original))
    target = child / 'evaluation/runs/new-off'
    assert set(path.name for path in target.iterdir()) == {'run.json'}
    assert result['reference']['parent_run_sha256'] == sha256(original)
    assert result['reference']['inherited_turn_count'] == 111
    assert result['reference']['new_cloud_requests'] == 0
    assert result['items'] == read_json(original)['items']
    assert result['run_id'] == 'new-off' and result['suite_id'] == 'new-suite'
    with pytest.raises(FileExistsError):
        module.create_reference_run(original, child / 'evaluation', 'new-off', 'new-suite', sha256(original))
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('mutation', ['hash', 'group', 'incomplete', 'protocol', 'path', 'symlink', 'same_root'])
def test_reference_rejects_unverified_or_wrong_source(setup, mutation, tmp_path):
    parent, child, ledger, original = setup
    run = read_json(original)
    path = original
    root = child / 'evaluation'
    if mutation == 'group':
        run['inference_config']['group'] = 'F38-Off'
    elif mutation == 'incomplete':
        run['items'].pop()
    elif mutation == 'protocol':
        run['protocol_label'] = 'other'
    elif mutation == 'same_root':
        root = parent / 'evaluation'
    atomic_json(original, run)
    digest = sha256(original)
    if mutation == 'hash':
        digest = '0' * 64
    elif mutation == 'path':
        path = tmp_path / 'arbitrary/run.json'
        atomic_json(path, run)
    elif mutation == 'symlink':
        real = original.with_name('real.json')
        original.rename(real)
        original.symlink_to(real)
    with pytest.raises(ValueError):
        module.create_reference_run(path, root, 'new-off', 'new-suite', digest)


def build_continuation(setup):
    parent, child, ledger, old_off = setup
    module.create_reference_run(old_off, child / 'evaluation', 'new-off', 'new-suite', sha256(old_off))
    old_on = parent / 'evaluation/runs/old-on/run.json'
    original = formal('F37-On', 'old-on', 'old-suite')
    original.update(status='blocked', stop_reason='request_uncertain', gate_complete=False)
    original['items'] = original['items'][:1]
    original['items'][0]['runtime_turns'] = original['items'][0]['runtime_turns'][:1]
    original['items'][0]['runtime_turns'][0].update(answer_origin='system_failure',
        judge={'score': None}, judge_status='skipped_expected', error_type='request_uncertain')
    original['items'][0]['final_turn_executed'] = False
    atomic_json(old_on, original)
    raw = {'context': {'qa_id': '0', 'turn_index': 0}, 'synthetic': 'original immutable attempt'}
    atomic_json(old_on.parent / 'turn_attempts/0/0.json', raw)
    unknown, request_id = add_call(old_on.parent, ledger, 'old-on', uncertain=True)
    on = formal('F37-On', 'new-on', 'new-suite')
    turn = copy.deepcopy(original['items'][0]['runtime_turns'][0])
    turn['raw_attempt_path'] = 'inherited_turn_attempts/0/0.json'
    turn['record_provenance'] = {'kind': 'inherited', 'run_id': 'old-on', 'run_sha256': sha256(old_on)}
    on['items'][0]['runtime_turns'][0] = turn
    on['continuation'] = {'parent_run_path': str(old_on), 'parent_run_id': 'old-on',
        'parent_run_sha256': sha256(old_on), 'source_suite_id': 'old-suite',
        'preserved_turns': [{'qa_id': '0', 'turn_index': 0}], 'inherited_turn_count': 1,
        'new_turn_count': 110, 'acknowledged_call_ids': [unknown]}
    atomic_json(child / 'evaluation/runs/new-on/run.json', on)
    atomic_json(child / 'evaluation/runs/new-on/inherited_turn_attempts/0/0.json', raw)
    add_call(child / 'evaluation/runs/new-on', ledger, 'new-on')
    groups = [{'name': name, 'run_id': identifier, 'status': 'completed', 'returncode': 0}
              for name, identifier in zip(ORDER, ('new-off', 'new-on', 'f38-off', 'f38-on'))]
    for group in groups[2:]:
        directory = child / 'evaluation/runs' / group['run_id']
        atomic_json(directory / 'run.json', formal(group['name'], group['run_id'], 'new-suite'))
        add_call(directory, ledger, group['run_id'])
    suite = {'schema_version': 'camroll-flash-gate-continuation-suite-v1', 'suite_id': 'new-suite',
        'status': 'passed', 'groups': groups, 'restoration': {'state': 'restored'},
        'model_files_unchanged': True, 'frozen_inputs_unchanged': True, 'wemm_weights_unchanged': True,
        'acknowledgement': {'call_id': unknown, 'request_id': request_id, 'payload_hash': 'old-on-payload'}}
    atomic_json(child / 'formal-suite.json', suite)
    return child, ledger, old_on, unknown


def test_complete_work_keeps_original_unknown_and_separates_cost_segments(setup):
    child, ledger, old_on, unknown = build_continuation(setup)
    before = old_on.read_bytes()
    ledger_before = ledger.summary()
    result = module.export(child, ledger_path=ledger.path)
    assert result['work_complete'] is result['execution_complete'] is result['acceptance_complete'] is True
    assert result['financial_settlement_complete'] is False
    assert result['status'] == 'work_complete_with_outstanding_settlement'
    assert all(row['complete_pair'] for row in result['paired_comparisons'])
    assert result['acknowledged_outstanding'][0]['call_id'] == unknown
    assert result['acknowledged_outstanding'][0]['reserved_cny'] == .25
    on = result['groups'][1]
    assert on['cost_audit']['request_count'] == 2
    assert on['cost_audit']['own_estimated_cny'] == .1
    assert on['cost_audit']['own_reserved_cny'] == .25
    assert on['cost_audit']['settlement_complete'] is False
    assert on['cost_audit']['unacknowledged_outstanding_call_ids'] == []
    inherited, new = on['cost_audit']['cost_segments']
    assert inherited['known_reserved_cny'] == .25 and inherited['known_estimated_cny'] == 0
    assert new['known_estimated_cny'] == .1 and new['known_reserved_cny'] == 0
    assert on['calls_and_tokens']['calls'] == 2
    assert result['groups'][0]['cost_audit']['cost_segments'][1]['request_count'] == 0
    assert result['groups'][0]['new_run_wall_ms'] == 0
    assert old_on.read_bytes() == before and ledger.summary() == ledger_before
    assert read_json(child / 'evaluation/cost-audit.json')['runs']['new-on']['financial_settlement_complete'] is False


def test_journal_union_and_tokens_deduplicate_same_call_id(setup):
    child, ledger, old_on, unknown = build_continuation(setup)
    source = sqlite3.connect(old_on.parent / 'request-journal.sqlite')
    target = sqlite3.connect(child / 'evaluation/runs/new-on/request-journal.sqlite')
    row = source.execute('SELECT * FROM cloud_requests').fetchone()
    target.execute('INSERT INTO cloud_requests VALUES (' + ','.join('?' for _ in row) + ')', row)
    target.commit()
    source.close()
    target.close()
    with (child / 'evaluation/runs/new-on/model_calls.jsonl').open('a') as stream:
        stream.write((old_on.parent / 'model_calls.jsonl').read_text())
    result = module.export(child, ledger_path=ledger.path)['groups'][1]
    assert result['cost_audit']['request_count'] == 2
    assert result['cost_audit']['known_reserved_cny'] == .25
    assert result['calls_and_tokens']['calls'] == 2
    assert result['calls_and_tokens']['token_totals']['prompt_tokens'] == 20
    assert [row['request_count'] for row in result['cost_audit']['cost_segments']] == [1, 1]


@pytest.mark.parametrize('mutation', ['parent_hash', 'parent_path', 'inherited_answer', 'missing_turn', 'duplicate_plan', 'new_ack', 'source_suite'])
def test_ancestry_rejects_mutations_without_rewriting_source(setup, mutation):
    child, ledger, old_on, unknown = build_continuation(setup)
    before = old_on.read_bytes()
    path = child / 'evaluation/runs/new-on/run.json'
    run = read_json(path)
    if mutation == 'parent_hash':
        run['continuation']['parent_run_sha256'] = '0' * 64
    elif mutation == 'parent_path':
        run['continuation']['parent_run_path'] = '/unrelated/private/run.json'
    elif mutation == 'inherited_answer':
        run['items'][0]['runtime_turns'][0]['answer'] = 'fabricated'
    elif mutation == 'missing_turn':
        run['items'][0]['runtime_turns'].pop(0)
    elif mutation == 'duplicate_plan':
        run['continuation']['preserved_turns'] *= 2
    elif mutation == 'new_ack':
        run['continuation']['acknowledged_call_ids'] = ['fresh-unknown']
    elif mutation == 'source_suite':
        run['continuation']['source_suite_id'] = 'other-suite'
    atomic_json(path, run)
    with pytest.raises(ValueError):
        module.export(child, ledger_path=ledger.path)
    assert old_on.read_bytes() == before


def test_incomplete_continuation_retains_full_plans_and_old_unknown(setup):
    child, ledger, old_on, unknown = build_continuation(setup)
    path = child / 'evaluation/runs/new-on/run.json'
    run = read_json(path)
    run.update(status='blocked', gate_complete=False)
    run['items'] = run['items'][:1]
    atomic_json(path, run)
    missing = child / 'evaluation/runs/f38-on/run.json'
    missing.unlink()
    result = module.export(child, ledger_path=ledger.path)
    assert result['work_complete'] is False and result['status'] == 'incomplete'
    assert result['groups'][3]['summary']['unexecuted_turn_count'] == 111
    assert result['groups'][3]['summary']['quality_distribution_planned']['unscored'] == 100
    assert result['groups'][3]['summary']['retrieval_metric_count'] == 81
    assert result['acknowledged_outstanding'][0]['call_id'] == unknown


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'symlink', 'directory_symlink', 'absolute_path', 'other_path', 'parent_symlink'])
def test_inherited_raw_attempt_copy_must_be_managed_and_identical(setup, mutation, tmp_path):
    child, ledger, old_on, unknown = build_continuation(setup)
    path = child / 'evaluation/runs/new-on/run.json'
    run = read_json(path)
    raw = path.parent / 'inherited_turn_attempts/0/0.json'
    if mutation == 'missing':
        raw.unlink()
    elif mutation == 'changed':
        raw.write_text('{}')
    elif mutation in {'symlink', 'parent_symlink'}:
        target = raw if mutation == 'symlink' else old_on.parent / 'turn_attempts/0/0.json'
        source = tmp_path / 'elsewhere.json'
        source.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(source)
    elif mutation == 'directory_symlink':
        source = tmp_path / 'elsewhere'
        raw.parent.rename(source)
        raw.parent.symlink_to(source, target_is_directory=True)
    elif mutation == 'absolute_path':
        run['items'][0]['runtime_turns'][0]['raw_attempt_path'] = str(raw)
    elif mutation == 'other_path':
        run['items'][0]['runtime_turns'][0]['raw_attempt_path'] = 'turn_attempts/0/0.json'
    atomic_json(path, run)
    with pytest.raises(ValueError):
        module.export(child, ledger_path=ledger.path)


def test_fresh_unknown_is_not_implicitly_acknowledged(setup):
    child, ledger, old_on, old_unknown = build_continuation(setup)
    new_unknown, _ = add_call(child / 'evaluation/runs/new-on', ledger, 'fresh', uncertain=True)
    report = module.export(child, ledger_path=ledger.path)
    result = report['groups'][1]['cost_audit']
    assert report['work_complete'] is False
    assert result['unacknowledged_outstanding_call_ids'] == [new_unknown]
    assert [row['call_id'] for row in result['acknowledged_outstanding']] == [old_unknown]
    assert result['known_reserved_cny'] == .5 and result['settlement_complete'] is False


@pytest.mark.parametrize('failure', ['ledger_missing', 'trace_corrupt', 'weights_unverified'])
def test_complete_execution_does_not_hide_incomplete_delivery_audit(setup, failure):
    child, ledger, old_on, unknown = build_continuation(setup)
    ledger_path = ledger.path
    if failure == 'ledger_missing':
        ledger_path = child / 'absent.sqlite'
    elif failure == 'trace_corrupt':
        with (child / 'evaluation/runs/new-on/model_calls.jsonl').open('a') as stream:
            stream.write('{truncated')
    elif failure == 'weights_unverified':
        suite = read_json(child / 'formal-suite.json')
        del suite['wemm_weights_unchanged']
        atomic_json(child / 'formal-suite.json', suite)
    result = module.export(child, ledger_path=ledger_path)
    assert result['execution_complete'] is True
    assert result['work_complete'] is result['acceptance_complete'] is False
    assert result['status'] == 'incomplete'
