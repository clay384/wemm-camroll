import json

from camroll_eval.common import atomic_json
from camroll_eval.repository import Repository
from test_report_ui import evaluate


def test_repository_projects_only_safe_ancestry_fields_without_following_paths(tmp_path):
    path = tmp_path / 'runs/derived/run.json'
    metadata = {'parent_run_path': '/unrelated/private/run.json', 'parent_run_id': 'old-run',
        'source_suite_id': 'old-suite', 'parent_run_sha256': 'a' * 64,
        'inherited_turn_count': 10, 'new_turn_count': 101,
        'preserved_turns': [{'qa_id': 'qa', 'turn_index': 0}], 'acknowledged_call_ids': ['private-call']}
    atomic_json(path, {'run_id': 'derived', 'mode': 'gate_ab', 'gate_summary_version': 'camroll-flash-gate-v1',
        'summary': {}, 'continuation': metadata, 'items': []})
    before = path.read_bytes()
    repo = Repository(tmp_path)
    brief, detail = repo.get_brief('derived'), repo.detail('derived')
    assert brief['report_provenance'] == detail['report_provenance'] == {
        'kind': 'continuation', 'parent_run_id': 'old-run', 'source_suite_id': 'old-suite',
        'parent_run_sha256': 'a' * 64, 'inherited_turn_count': 10, 'new_turn_count': 101}
    assert '/unrelated' not in json.dumps(brief) + json.dumps(detail)
    assert 'private-call' not in json.dumps(brief) + json.dumps(detail)
    # Immutable export retains the complete original private provenance.
    assert repo.load('derived')['continuation'] == metadata
    detail['report_provenance']['new_turn_count'] = 0
    assert repo.detail('derived')['report_provenance']['new_turn_count'] == 101
    assert path.read_bytes() == before


def test_public_provenance_rejects_paths_and_unbounded_nonidentity_values():
    result = Repository.brief({'mode': 'gate_ab', 'items': [], 'reference': {
        'parent_run_id': '/private/source', 'source_suite_id': '../source',
        'parent_run_sha256': '/private/hash', 'inherited_turn_count': True, 'new_turn_count': 900}})
    assert result['report_provenance'] == {'kind': 'reference'}


def test_ui_distinguishes_completed_work_from_retained_unknown_reservation():
    result = evaluate("""(()=>{
      const run={mode:'gate_ab',report_provenance:{kind:'continuation',parent_run_id:'old-run',
        parent_run_sha256:'abc',inherited_turn_count:10,new_turn_count:101},cost_audit:{work_complete:true,
        financial_settlement_complete:false,acknowledged_outstanding:[{reserved_cny:1.268122}]}};
      return {notice:continuationNotice(run),inherited:inheritedTurnNote({record_provenance:{kind:'inherited',run_id:'old-run'}}),fresh:inheritedTurnNote({})};
    })()""")
    assert '10 轮' in result['notice'] and '101 轮' in result['notice']
    assert '执行及必要评分：已完成；费用结算：尚未完成' in result['notice']
    assert '¥1.268122' in result['notice'] and '没有记为已结算或已消费' in result['notice']
    assert '/api/runs/old-run/export' in result['notice']
    assert '原答案、失败与评分保留' in result['inherited'] and result['fresh'] == ''


def test_reference_and_legacy_ui_remain_distinct_and_escape_source_text():
    result = evaluate("""({legacy:continuationNotice({mode:'gate_ab'}),reference:continuationNotice({mode:'gate_ab',
      report_provenance:{kind:'reference',parent_run_id:'<unsafe>',parent_run_sha256:'<digest>'}})})""")
    assert result['legacy'] == ''
    assert '没有重新生成或评分' in result['reference']
    assert '&lt;unsafe&gt;' in result['reference'] and '&lt;digest&gt;' in result['reference']
    assert '/api/runs/' not in result['reference'] and '<unsafe>' not in result['reference']
