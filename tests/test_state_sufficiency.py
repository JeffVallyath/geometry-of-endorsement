"""CPU consistency and fail-closed guards for the compact current evidence."""
import copy
import json
import re

import pytest

from repro import state_sufficiency as ss
from repro.common import ROOT
from repro.evidence import check


@pytest.fixture(scope='module')
def summaries():
    return ss.replay()


def test_eight_primary_cells_and_all_condition_recurrence(summaries):
    assert len(summaries['v10']) == len(summaries['strong']) == 8
    assert all(r['lower'] > 0 for r in summaries['v10'])
    assert [r['roots'] for r in summaries['v10']] == [50,41,30,25,24,25,38,40]
    assert [r['count'] for r in summaries['strong']] == [49,24,20,22,32,7,36,33]


@pytest.mark.parametrize('mutation', ['missing_seed','duplicate','denominator','root_identity','witness_subset','bootstrap','v9_view','v6_interface','coherence','example'])
def test_compact_reconstruction_rejects_corruption(monkeypatch, mutation):
    original=ss.load_json
    def altered(path):
        data=copy.deepcopy(original(path))
        if path.name=='root_counts.json':
            if mutation=='missing_seed': data=[r for r in data if r['condition']!='INV_PAIR_NLL_s1']
            elif mutation=='duplicate': data[-1]=copy.deepcopy(data[0])
            elif mutation=='denominator': data[0]['questions']=data[0]['AC']
            elif mutation=='root_identity': data[0]['root_id']='unobserved-root'
            elif mutation=='witness_subset': data[0]['W']=19
        if path.name=='specification_summary.json' and mutation=='bootstrap': data['bootstrap_seed']+=1
        if path.name=='primary_summary.json' and mutation=='v9_view': data[0]['view']='STRONG_REFERENCE'
        if path.name=='fixed_certificate_summary.json' and mutation=='v6_interface': data[0]['interface']='d0'
        if path.name=='coherence_summary.json' and mutation=='coherence': data[0]['mean']='0.99'
        if path.name=='selected_repeat_checked_example.json' and mutation=='example': data['half_probability_range']=0.1
        return data
    monkeypatch.setattr(ss,'load_json',altered)
    with pytest.raises(ValueError): ss.reconstruct()


LABELS={'INV_PAIR_NLL':'Constrained learned editor','FREE_PAIR_CONSISTENCY':'Unrestricted consistency-trained editor',
        'EXISTING_CORRECTION':'Existing textual correction','LATEST_SAME_WORDING':'Latest-value wording'}


def test_headline_rows_are_bound_by_model_and_method_not_just_number_inventory(summaries):
    text=(ROOT/'docs/RESULTS_AND_CLAIMS.md').read_text(encoding='utf8')
    for r in summaries['v10']:
        expected=f"| {r['model'].capitalize()} | {LABELS[r['group']]} | {100*r['rate']:.3f}% | [{100*r['lower']:.3f}, {100*r['upper']:.3f}]% | {r['roots']}/64 |"
        assert expected in text
    for r in summaries['strong']:
        assert f"| {r['model'].capitalize()} | {LABELS[r['group']]} | {r['count']}/{r['opportunities']} | {r['roots']}/64 |" in text
    for r in summaries['v7']:
        assert f"| {r['model'].capitalize()} | {r['seed']} | {100*r['changed']:.2f}% | {100*r['harm']:.2f}% |" in text
    v9labels={'INV_PAIR_NLL_s0':'Constrained paired NLL seed 0','INV_PAIR_NLL_s1':'Constrained paired NLL seed 1',
              'FREE_PAIR_CONSISTENCY_s0':'Unrestricted paired consistency seed 0','FREE_PAIR_CONSISTENCY_s1':'Unrestricted paired consistency seed 1',
              'EXISTING_CORRECTION':'Existing textual correction','LATEST_SAME_WORDING':'Latest-value wording'}
    for condition,label in v9labels.items():
        g=next(r for r in summaries['v9'] if r['model']=='gemma' and r['condition']==condition)
        q=next(r for r in summaries['v9'] if r['model']=='qwen' and r['condition']==condition)
        assert f"| {label} | {g['count']}/{g['eligible']} | {g['roots']}/64 | {q['count']}/{q['eligible']} | {q['roots']}/64 |" in text
    for condition,label in [('INV_COMPLETE_SINGLE_s0','Constrained seed 0'),('INV_COMPLETE_SINGLE_s1','Constrained seed 1'),('FREE_COMPLETE_SINGLE_s0','Unrestricted seed 0'),('FREE_COMPLETE_SINGLE_s1','Unrestricted seed 1'),('TEXT_CORRECTION','Textual correction')]:
        g=next(r for r in summaries['v6'] if r['model']=='gemma' and r['condition']==condition)
        q=next(r for r in summaries['v6'] if r['model']=='qwen' and r['condition']==condition)
        assert f"| {label} | {g['roots']}/64 | {q['roots']}/32 |" in text


@pytest.mark.parametrize('old,new', [
    ('including ineligible questions in the denominator','excluding ineligible questions from the denominator'),
    ('Textual-correction procedures were not independently repeated','Textual-correction procedures were independently repeated'),
    ('not a fully independent replication or a prospective result','a fully independent prospective replication'),
    ('Every updating condition\'s unfiltered mean was below its native-final mean','Every updating condition\'s unfiltered mean was above its native-final mean'),
    ('produced no scientific outcome','produced a negative scientific outcome'),
])
def test_claim_boundaries_fail_closed(old,new):
    text=(ROOT/'docs/RESULTS_AND_CLAIMS.md').read_text(encoding='utf8')
    # Wrap-independent mutation, preserving the rest of the scientific document.
    pattern=r'\s+'.join(re.escape(w) for w in old.split())
    bad,n=re.subn(pattern,new,text)
    assert n==1
    report=check(document='results',overrides={'results':bad})
    assert 'material-qualification-missing' in {r['rule'] for r in report.errors}


def test_current_status_and_historical_failure_are_separate():
    status=json.loads((ROOT/'artifacts/project/status.json').read_text())
    stages={r['stage']:r['status'] for r in status['stages']}
    assert status['as_of']=='2026-09-15'
    assert stages['Source-history sufficiency: V10 confirmation']=='Complete; prospective controlled generated-case confirmation'
    assert stages['External public-method validation']=='Pending; no scientific outcome'
    assert stages['Human-audited confirmation']=='Not certified by this public package'
    gates=ss.load_json(ss.BASE/'v7/constructive_summary.json')['gates']
    assert not any(r['H3'] for r in gates)


def test_public_story_and_style_contract():
    for name in ('README.md','PROJECT_STRATEGY.md','docs/RESULTS_AND_CLAIMS.md'):
        text=(ROOT/name).read_text(encoding='utf8')
        assert '\u2014' not in text
        assert 'source-history sufficiency' in text
        assert 'prospective confirmation' in text
        assert not re.search(r'\bmy own experiments\b|strongest unresolved test|unfinished successor is not a completed result',text,re.I)
    captions=(ROOT/'figures/CAPTIONS.md').read_text(encoding='utf8')
    assert 'earlier V6 descriptive source-history figure, not the V10 prospective' in captions


def test_export_refuses_existing_output(tmp_path):
    ss.export(tmp_path)
    with pytest.raises(FileExistsError): ss.export(tmp_path)
