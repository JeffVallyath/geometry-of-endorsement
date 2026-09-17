"""Crossover input integrity, independent replay, and every displayed figure value."""
import copy
import csv
import gzip
import importlib.util
import json
import math
from types import SimpleNamespace

import pytest

from repro.cache_crossover import BASE, analyze, classify, export, replay
from repro.cache_crossover_figure import figure_data, plot
from repro.common import ROOT, load_json, sha256


def inputs():
    return (load_json(BASE / 'case_plan.json'), load_json(BASE / 'terminal_worlds.json'),
            [json.loads(line) for line in gzip.decompress((BASE / 'raw.jsonl.gz').read_bytes()).splitlines()])


def test_replay_matches_committed_values_and_six_case_table(tmp_path):
    result = replay()
    assert result == load_json(BASE / 'results.json')
    assert result['counts']['witness']['main'] == dict(clear_later=5, mixed=1, low_separation=0, clear_early=0, total=6)
    assert result['counts']['witness']['alternate'] == dict(clear_later=1, mixed=2, low_separation=3, clear_early=0, total=6)
    assert result['counts']['control']['main']['low_separation'] == 6
    assert result['counts']['control']['alternate']['low_separation'] == 5
    assert result['counts']['control']['alternate']['mixed'] == 1
    assert result['controls']['total_checks'] == 288
    assert result['controls']['maximum_probability_change'] == 0
    assert all(r['direct_preservation'] for r in result['classifications'])
    export(tmp_path)
    assert (tmp_path / 'cache_crossover/six_cases.csv').read_bytes() == (BASE / 'six_cases.csv').read_bytes()
    with (BASE / 'six_cases.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with pytest.raises(FileExistsError):
        export(tmp_path)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'primary', 'label_order', 'operand', 'gold', 'score', 'layer'])
def test_bad_scientific_inputs_fail_closed(mutation):
    plan, worlds, rows = inputs()
    cell = next(r for r in rows if r['stage'] == 'cell')
    if mutation == 'missing':
        rows.remove(cell)
    elif mutation == 'duplicate':
        rows.append(copy.deepcopy(cell))
    elif mutation == 'primary':
        plan['records'][0]['primary_query_id'] = 'unplanned'
    elif mutation == 'label_order':
        plan['records'][0]['questions'][0]['answer_label_token_ids'].reverse()
    elif mutation == 'operand':
        plan['records'][0]['questions'][1]['spec']['a'] = 999
    elif mutation == 'gold':
        q = next(q for q in plan['records'][0]['questions'] if not q['is_joint'])['spec']
        worlds[cell['root_id']]['values'][q['a']][q['p']] ^= 1
    elif mutation == 'score':
        cell['metrics']['raw_logps_no_yes'][0] += 1
    elif mutation == 'layer':
        next(r for r in rows if 'provenance' in r)['provenance']['cut'] = 14
    with pytest.raises((ValueError, IndexError)):
        analyze(plan, worlds, rows)


def test_classification_requires_separation_direction_and_preservation():
    joint = {s: dict(p_yes=p, prediction=int(p > .5), mass=.999, valid=True)
             for s, p in dict(AA=.96, BB=.02, AB=.03, BA=.97).items()}
    assert classify(joint, True)['label'] == 'LATER-FOLLOWING'
    assert classify(joint, False)['label'] == 'MIXED/INCONCLUSIVE'
    joint['AB']['mass'] = .5
    assert classify(joint, True)['label'] == 'MIXED/INCONCLUSIVE'
    joint['BB']['p_yes'] = .95
    assert classify(joint, True)['label'] == 'LOW-DISCRIMINATION'


def test_every_figure_value_recomputes_from_the_identified_raw_record(tmp_path):
    import matplotlib.pyplot as plt

    data = figure_data()
    assert data == load_json(ROOT / 'artifacts/figures/cache_crossover.json')
    for name, digest in data['sources'].items():
        assert sha256(BASE / name) == digest
    plan, worlds, rows = inputs()
    figures = []
    plot(SimpleNamespace(OUT=tmp_path, save=lambda fig, name: figures.append((fig, name))))
    fig, name = figures[0]
    try:
        assert name == 'fig08_cache_crossover'
        bars = fig.axes[0].patches
        assert len(bars) == 4
        assert data['example']['query_id'] == 'SSC1-FINAL-0026-d1-extra2'
        for bar, state in zip(bars, data['example']['states']):
            assert bar.get_width() == state['joint']['p_yes']
            for value in [state['joint'], *state['direct']]:
                raw = rows[value['raw_line'] - 1]
                assert raw['root_id'] == data['example']['root_id'] and raw['state'] == state['state']
                lp = raw['metrics']['raw_logps_no_yes']
                independent = 1 / (1 + math.exp(lp[0] - lp[1]))
                assert abs(independent - value['p_yes']) < 1e-15
            for direct in state['direct']:
                assert direct['correct']
                # Both selected facts are Yes; assert their actual artist text, not just plot JSON.
                assert f"{direct['p_yes']:.6f}" in [t.get_text() for t in fig.axes[1].texts]
        assert all(f['correct_semantic_answer'] == 1 for f in data['example']['direct_facts'])
        assert [(f['actor'], f['project']) for f in data['example']['direct_facts']] == [('Dion', 'Bridge'), ('Orla', 'Bridge')]
        assert load_json(tmp_path / 'cache_crossover.json') == data
    finally:
        plt.close(fig)


@pytest.mark.parametrize('document,path', [
    ('captions', 'figures/CAPTIONS.md'),
    ('caption_details', 'figures/CAPTION_DETAILS.md'),
])
def test_caption_and_claim_guards(document, path):
    from repro.evidence import check
    text = (ROOT / path).read_text(encoding='utf8')
    mutations = [('not a prevalence estimate', 'a prevalence estimate'),
                 ('model weights stay fixed', 'model weights change'),
                 ('two correct direct answers', 'two incorrect direct answers')]
    if document == 'captions':
        mutations = [('selected Gemma example', 'representative Gemma example'),
                     ('Model weights stay fixed', 'Model weights change'),
                     ('both direct fact answers remain correct', 'both direct fact answers become incorrect')]
    for old, new in mutations:
        assert old in text
        report = check(document=document, overrides={document: text.replace(old, new)})
        assert any(r['unit'] == 'figure8' for r in report.errors)


def test_extractor_rejects_unsafe_paths_and_wrong_archive(tmp_path):
    spec = importlib.util.spec_from_file_location('crossover_extract', BASE / 'extract_sources.py')
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    for name in ('../escape', '/absolute', 'C:/drive', 'x\\y', 'a/../b'):
        with pytest.raises(ValueError):
            extractor.safe(name)
    archive = tmp_path / 'wrong.zip'
    archive.write_bytes(b'not the completed archive')
    with pytest.raises(AssertionError):
        extractor.extract(archive, BASE / 'delivery_receipt.json', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
