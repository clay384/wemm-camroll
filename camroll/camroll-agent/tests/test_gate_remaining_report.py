from test_report_ui import evaluate


def test_remaining_group_passed_never_renders_four_groups_completed():
    result = evaluate("""(()=>{
      const groups=['F37-Off','F37-On','F38-Off','F38-On'].map((name,i)=>({name,run_id:'run-'+i,reused:i<3,qa_count:100}));
      state.runs=groups.map((g,i)=>({run_id:g.run_id,mode:'gate_ab',qa_count:100,
        status:i===2?'blocked':'completed_with_errors',summary:{judge_valid_count:i===2?0:80}}));
      const profile={schema_version:'camroll-flash-gate-v1',groups,suite_status:'passed',comparison_status:'incomplete',
        remaining_group_status:'passed',execution_order:['F38-On']};
      state.profile=profile;renderProfile(profile);
      return {html:$('#comparison-profile').innerHTML,old:gateBatchBadge(state.runs[0]),fresh:gateBatchBadge(state.runs[3])};
    })()""")
    assert '尚未完整' in result['html'] and '四组已完成' not in result['html']
    assert '本次F38-On已执行及完成必要评分' in result['html']
    assert '只读引用此前封存的结果' in result['html'] and '本次独立新增' in result['html']
    assert '引用封存结果' in result['old'] and '独立新增' in result['fresh']


def test_active_remaining_scope_explains_terminal_hash_gate_and_legacy_is_unchanged():
    result = evaluate("""({active:remainingProfileNote({comparison_status:'incomplete',execution_order:['F38-On'],remaining_group_status:'running'}),
      legacy:remainingProfileNote({suite_status:'passed'})})""")
    assert '终态导出并核对全部来源哈希后开放跨批比较' in result['active']
    assert '不重发、不释放' in result['active']
    assert result['legacy'] == ''
