"""Guard the V10 primary estimand and the unchanged V6 archival rendering."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from repro.common import ROOT
from repro.source_history_figure import BASE, LABELS, PRIMARY, SPECIFICATION, figure_data, plot


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf8'))


def test_all_eight_points_and_corrected_intervals_copy_primary_output_exactly():
    data = figure_data()
    primary = read(PRIMARY)
    assert len(data['rows']) == len(primary) == 8
    assert [(r['actor'], r['group']) for r in data['rows']] == [
        (a, g) for a in ('gemma', 'qwen') for g in LABELS]
    for plotted, saved in zip(data['rows'], primary):
        for key in ('actor', 'group', 'roots', 'mean', 'corrected_interval',
                    'corrected_level', 'draws', 'seed', 'seeds_averaged_within_root'):
            assert plotted[key] == saved[key]
        assert plotted['corrected_interval'][0] > 0
        assert plotted['mean'] != saved['root_witness_prevalence']
        assert plotted['corrected_interval'] != saved['descriptive_interval']
        assert plotted['fixed_opportunities_per_seed_or_textual_condition'] == 1152
    for name, digest in data['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert data == read('artifacts/figures/v10_source_history.json')


def test_fixed_denominators_and_seed_averaging_agree_with_saved_root_counts():
    counts = read(BASE + 'root_counts.json')
    for plotted in figure_data()['rows']:
        conditions = ([plotted['group'] + '_s0', plotted['group'] + '_s1']
                      if plotted['seeds_averaged_within_root'] else [plotted['group']])
        rows = [r for r in counts if r['actor'] == plotted['actor'] and r['condition'] in conditions]
        assert len(rows) == 64 * len(conditions)
        roots = {r['root_id'] for r in rows}
        assert len(roots) == 64
        scores = []
        for root in sorted(roots):
            selected = [r for r in rows if r['root_id'] == root]
            assert {r['condition'] for r in selected} == set(conditions)
            assert all(r['questions'] == plotted['joint_questions_per_root'] == 18 for r in selected)
            # W/18, never W/AC, and learned seeds averaged before the root mean.
            scores.append(np.mean([r['W'] / r['questions'] for r in selected]))
        assert np.mean(scores) == pytest.approx(plotted['mean'], abs=1e-15)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'roots', 'level', 'seed', 'averaging'])
def test_plot_projection_rejects_changed_primary_inventory(tmp_path, mutation):
    primary = read(PRIMARY)
    if mutation == 'missing':
        primary.pop()
    elif mutation == 'duplicate':
        primary[-1] = primary[0]
    else:
        field, value = {'roots': ('roots', 63), 'level': ('corrected_level', .95),
                        'seed': ('seed', 0), 'averaging': ('seeds_averaged_within_root', False)}[mutation]
        primary[0][field] = value
    (tmp_path / BASE).mkdir(parents=True)
    (tmp_path / PRIMARY).write_text(json.dumps(primary), encoding='utf8')
    (tmp_path / SPECIFICATION).write_text(json.dumps(read(SPECIFICATION)), encoding='utf8')
    with pytest.raises(ValueError, match='V10'):
        figure_data(tmp_path)


def test_renderer_draws_the_primary_values_and_zero_reference(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    figures = []
    plot(SimpleNamespace(OUT=tmp_path, save=lambda fig, name: figures.append((fig, name))))
    fig, name = figures[0]
    try:
        assert name == 'fig07_source_history'
        axis = fig.axes[1]
        bars = [c for c in axis.containers if isinstance(c, ErrorbarContainer)]
        assert len(bars) == 8
        for bar, row in zip(bars, read(PRIMARY)):
            point, _, intervals = bar.lines
            assert point.get_xdata()[0] == 100 * row['mean']
            endpoints = intervals[0].get_segments()[0][:, 0]
            np.testing.assert_allclose(endpoints, np.array(row['corrected_interval']) * 100, rtol=0, atol=1e-14)
            assert point.get_marker() == ('o' if row['seeds_averaged_within_root'] else 's')
        assert any(list(line.get_xdata()) == [0, 0] for line in axis.lines)
        assert json.loads((tmp_path / 'v10_source_history.json').read_text()) == figure_data()
    finally:
        plt.close(fig)


def test_v6_archive_retains_original_figure_bytes():
    # SHA-256 of Figure 7 before the V10 replacement, not freshly generated expectations.
    for extension, digest in {
        'pdf': 'abad3414e477f96012d62f81b80a71605e41f456f6fb0d9a31deeb7e070d8572',
        'png': '7d83c4913314cbafa397a9b3e243a6a56f32058f44fceb7ad6843e1d5fb6082d',
    }.items():
        path = ROOT / f'figures/supplementary/figS3_v6_source_history.{extension}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_main_and_archival_captions_distinguish_evidence_and_witness():
    captions = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    main = captions.split('## Figure 7')[1].split('## Supplementary Figure S1')[0]
    archive = captions.split('## Supplementary Figure S3')[1]
    assert '99.375%' in main and '1,152' in main
    assert 'separate physical old/new stores' in main
    assert 'illustrative repeat-checked example' in main and 'not a prevalence estimate' in main
    assert 'earlier shared-state editing study (`V6`)' in archive and 'not the prospective confirmation' in archive
    results = (ROOT / 'docs/RESULTS_AND_CLAIMS.md').read_text(encoding='utf8')
    assert 'fig07_source_history.png' not in results.split('### Does the effect recur in a study planned before its outcomes?')[0]
    assert 'figS3_source_history_readable.png' in results


@pytest.mark.parametrize('old,new', [
    ('Ineligible questions remain in the denominator', 'Ineligible questions are excluded'),
    ('two learned-seed scores are averaged within each benchmark case', 'learned-seed scores are pooled across benchmark cases'),
    ('All eight intervals lie above zero', 'Some intervals cross zero'),
    ('not a prevalence estimate', 'a prevalence estimate'),
])
def test_v10_caption_contract_rejects_estimand_and_scope_drift(old, new):
    from repro.evidence import check
    captions = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    assert old in captions
    report = check(document='captions', overrides={'captions': captions.replace(old, new)})
    assert any(r['rule'] == 'material-qualification-missing' and r['unit'] == 'figure7'
               for r in report.errors)
