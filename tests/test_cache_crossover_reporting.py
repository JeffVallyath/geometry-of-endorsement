"""Publication-only regressions: semantic labels, exact cells, no panel pooling."""
import csv
import json
import math
from pathlib import Path
import gzip
import importlib.util
from types import SimpleNamespace

import pytest

from repro.cache_crossover_extension import BASE, CORE, DESIGN, analyze, export, replay
from repro.common import load_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/cache_crossover'


def rows(name):
    with (DATA/name).open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def test_semantic_mapping_and_figure_values():
    queries = {q['query_id']:q for q in json.loads((DATA/'query_mappings.json').read_text())}
    joint = rows('extension_joint.csv')
    assert len(joint)==8
    for row in joint:
        assert [row['label_no'],row['label_yes']]==queries[row['query_id']]['labels']
        lp = [float(row[k]) for k in ('logp_no','logp_yes')]
        weights = [math.exp(x-max(lp)) for x in lp]
        assert weights[1]/sum(weights)==float(row['p_yes'])
    direct = rows('extension_direct.csv')
    assert len(direct)==8
    for row in direct:
        assert [row['label_no'],row['label_yes']]==queries[row['query_id']]['labels']
        assert row['passed']=='True' and row['valid']=='True' and row['correct']=='True'
        assert float(row['p_correct'])>=.95 and float(row['mass'])>=.90
        assert abs(float(row['p_correct'])-float(row['baseline_p_correct']))==float(row['recipient_probability_change'])<=.02
    coded = {r['query_id']:r for r in direct if r['code']=='coded_A_B'}
    assert coded['SSC1-FINAL-0035-d1-03']['label_no']=='A'
    assert coded['SSC1-FINAL-0035-d1-extra0']['label_no']=='B'


def test_all_original_outcomes_remain_separate_and_unchanged():
    original = rows('original_panel.csv')
    saved = json.loads((ROOT/'reproducibility/cache_crossover/saved_summaries.json').read_text(encoding='utf8'))
    assert len(original)==len(saved)==24
    assert {r['pair_id'] for r in original}=={'gemma-pair-'+p for p in ('00','02','04','05','06','07')}
    for row,s in zip(original,saved):
        assert row['label']==s['label'] and row['root_id']==s['root_id']
        assert row['code']==('primary' if s['primary'] else 'alternate')
        assert {k:float(row[k]) for k in ('AA','AB','BA','BB')}==s['probability_effects']['cells']
    appendix=(ROOT/'docs/CACHE_CROSSOVER_APPENDIX.md').read_text(encoding='utf8')
    assert sum(line.startswith('| ') for line in appendix.splitlines())==25
    assert 'MIXED/INCONCLUSIVE' in appendix and 'LOW-DISCRIMINATION' in appendix
    # Regression: PowerShell's legacy default decoding must not corrupt UTF-8.
    for name in ('docs/CACHE_CROSSOVER_APPENDIX.md','reproducibility/cache_crossover/README.md'):
        text=(ROOT/name).read_text(encoding='utf8')
        assert '\u00e2\u20ac' not in text and '\ufffd' not in text


def test_independent_extension_replay_and_portable_export(tmp_path):
    actual = replay()
    assert actual == load_json(BASE / 'replay.json')
    assert actual['cases'] == 1 and actual['formats'] == 2
    assert [r['label'] for r in actual['results']] == ['LATER-FOLLOWING'] * 2
    assert actual['strict_direct_checks'] == 8 and actual['all_direct_passed']
    assert actual['replay_rows'] == 108
    assert actual['sham_checks'] + actual['full_copy_checks'] == 24
    assert actual['recorded_forwards'] == 166 and actual['neural_forwards_this_replay'] == 0
    export(tmp_path)
    assert load_json(tmp_path / 'cache_crossover_extension/replay.json') == actual
    for name in ('extension_joint.csv','extension_direct.csv','original_panel.csv'):
        with (tmp_path / 'cache_crossover_extension' / name).open(newline='', encoding='utf8') as f:
            assert list(csv.DictReader(f)) == rows(name)
    with pytest.raises(ValueError, match='fresh'):
        export(tmp_path)


@pytest.mark.parametrize('mutation', ['missing_cell','score','older_score','prefix','changed_facts','direct_truth','layer'])
def test_independent_replay_rejects_bad_inputs(mutation):
    design = load_json(ROOT / DESIGN)
    core = json.loads(gzip.decompress((ROOT / CORE).read_bytes()))
    candidate = load_json(BASE / 'candidate.json')
    raw = [json.loads(line) for line in gzip.decompress((BASE / 'raw.jsonl.gz').read_bytes()).splitlines()]
    terminal = load_json(BASE / 'terminal.json')
    if mutation == 'missing_cell':
        raw.remove(next(r for r in raw if r['stage']=='cell'))
    elif mutation == 'score':
        next(r for r in raw if r.get('state')=='AB')['result']['logps'][0] += 1
    elif mutation == 'older_score':
        first=raw[0]
        next(r for r in core['physical'] if (r['context'],r['query_id'])==(first['context'],first['query_id']))['logps'][0] += 1
    elif mutation == 'prefix':
        core['contexts']['o3/INV_PAIR_NLL_s1']['prefix_ids'].pop()
    elif mutation == 'changed_facts':
        candidate['histories'][0]['actual_changed_facts'] += 1
    elif mutation == 'direct_truth':
        design['root']['terminal']['values'][3][0] ^= 1
    elif mutation == 'layer':
        next(r for r in raw if 'provenance' in r)['provenance']['cut'] = 16
    with pytest.raises(ValueError):
        analyze(design,core,candidate,raw,terminal)


def test_every_extension_figure_bar_and_annotation_matches_sources(tmp_path):
    import matplotlib.pyplot as plt
    spec=importlib.util.spec_from_file_location('extension_figure',ROOT/'scripts/plot_cache_crossover.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    figures=[]
    module.plot(SimpleNamespace(OUT=tmp_path,save=lambda fig,name:figures.append((fig,name))))
    fig,name=figures[0]
    try:
        assert name=='fig09_cache_crossover_extension'
        for ax,code in zip(fig.axes,('literal_No_Yes','coded_A_B')):
            selected=[r for r in rows('extension_joint.csv') if r['code']==code]
            assert len(ax.patches)==len(selected)==4
            for bar,row in zip(ax.patches,selected):
                assert bar.get_width()==float(row['p_yes'])
                assert f"{float(row['p_yes']):.6f}" in [t.get_text() for t in ax.texts]
                assert ('-d0-' if code=='literal_No_Yes' else '-d1-') in row['query_id']
        visible=' '.join(t.get_text() for ax in fig.axes for t in ax.texts)+' '+' '.join(t.get_text() for t in fig.texts)
        assert all(label in visible for label in module.LABELS)
        for jargon in ('AA','AB','BA','BB','donor','recipient','LATER-FOLLOWING','robust witness'):
            assert jargon not in visible
        actual=replay()
        assert f"{actual['minimum_direct_probability']:.6f}" in visible
        assert f"{actual['maximum_direct_change']:.8f}" in visible
        assert 'all 8 strict checks' in visible
        assert load_json(tmp_path/'cache_crossover_extension.json')==load_json(ROOT/'artifacts/figures/cache_crossover_extension.json')
    finally:
        plt.close(fig)


@pytest.mark.parametrize('old,new', [
    ('One retrospectively selected Gemma history pair','Two independently selected Gemma history pairs'),
    ('not independent cases','independent cases'),
    ('not parameter editing','parameter editing'),
    ('mechanism branch is closed for the current paper','mechanism branch needs more root searches'),
])
def test_extension_scope_guards(old,new):
    from repro.evidence import check
    import re
    text=(ROOT/'docs/RESULTS_AND_CLAIMS.md').read_text(encoding='utf8')
    pattern=r'\s+'.join(re.escape(word) for word in old.split())
    changed,count=re.subn(pattern,new,text)
    assert count == 1
    result=check(document='results',overrides={'results':changed})
    assert any(e['unit']=='cache_crossover_extension' for e in result.errors)


def test_no_stale_pending_followup_and_no_new_em_dashes():
    for name in ('README.md','PROJECT_STRATEGY.md','docs/RESULTS_AND_CLAIMS.md','reproducibility/cache_crossover/README.md'):
        text=(ROOT/name).read_text(encoding='utf8')
        assert '\u2014' not in text
        assert 'have not yet been demonstrated together' not in text
        assert 'remains optional and has not run' not in text
    strategy=(ROOT/'PROJECT_STRATEGY.md').read_text(encoding='utf8')
    assert 'follow-up is now completed' in strategy
    assert 'mechanism branch is closed for the current paper' in strategy
