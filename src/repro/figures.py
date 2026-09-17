"""Render representation figures and matched editing comparisons from saved data."""
from __future__ import annotations

import importlib.util
import csv
import gzip
import json
from pathlib import Path

from .common import ROOT, new_output


def editing_comparisons(root=ROOT):
    """Select declared within-study comparisons, including both original seeds."""
    source = root / 'reproducibility/relational_editing'
    with (source / 'v4/expected/T10_coverage_contrasts.csv').open(newline='', encoding='utf8') as stream:
        coverage = list(csv.DictReader(stream))
    panels = []
    for actor in ('gemma', 'qwen'):
        for architecture, label in (('INV', 'Constrained'), ('FREE', 'Free overwrite')):
            contrast = f'{architecture}_COMPLETE_SINGLE_minus_{architecture}_SPARSE_CONTINUE'
            rows = [r for r in coverage if r['actor'] == actor and r['contrast'] == contrast]
            if len(rows) != 3 or {r['seed'] for r in rows} != {'s0', 's1', 'avg'}:
                raise ValueError('Coverage comparison requires both seeds and their paired average')
            rows = {r['seed']: r for r in rows}
            main = rows['avg']
            if any(float(r['level']) != .9875 or int(r['scenes']) != {'gemma': 64, 'qwen': 32}[actor] for r in rows.values()):
                raise ValueError('Coverage interval or scene population changed')
            panels.append(dict(actor=actor, editor=label, scenes=int(main['scenes']),
                effect=float(main['effect']), lower=float(main['lower']), upper=float(main['upper']),
                level=float(main['level']), seeds=[float(rows[s]['effect']) for s in ('s0', 's1')]))
    reader_rows = json.loads(gzip.decompress((source / 'v6/scores/reader_paired_contrasts.json.gz').read_bytes()))
    expected = {(a, c, r) for a in ('gemma', 'qwen') for c in ('INV_COMPLETE_SINGLE', 'FREE_COMPLETE_SINGLE') for r in ('R1', 'R2')}
    if len(reader_rows) != len(expected) or {(r['actor'], r['architecture'], r['reader']) for r in reader_rows} != expected:
        raise ValueError('Fixed-reader comparison inventory changed')
    readers = []
    for actor in ('gemma', 'qwen'):
        for architecture, label in (('INV_COMPLETE_SINGLE', 'Constrained'), ('FREE_COMPLETE_SINGLE', 'Free overwrite')):
            for reader, reader_label in (('R1', 'Paraphrase'), ('R2', 'Explicit rule')):
                row = next(r for r in reader_rows if (r['actor'], r['architecture'], r['reader']) == (actor, architecture, reader))
                if row['baseline'] != 'R0' or row['program'] != 'ABC' or row['family'] != 'same' or row['seeds'] != [0, 1] or set(row['seed_specific']) != {'0', '1'}:
                    raise ValueError('Reader comparison conditions or original seeds changed')
                if row['level'] != .99375 or row['unit'] != 'whole scene' or row['scenes'] != {'gemma': 64, 'qwen': 32}[actor]:
                    raise ValueError('Reader interval or scene population changed')
                readers.append(dict(actor=actor, editor=label, reader=reader_label,
                    **{key: row[key] for key in ('effect', 'lower', 'upper', 'level', 'scenes')},
                    seeds=[row['seed_specific'][str(s)]['effect'] for s in (0, 1)]))
    return dict(coverage=panels, readers=readers)


def plot_editing(module):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data = editing_comparisons()
    (module.OUT / 'supplementary').mkdir(exist_ok=True)
    legends = [Line2D([], [], marker='o', color='#333333', linestyle='none', label='Paired mean and interval'),
               Line2D([], [], marker='v', color='#777777', linestyle='none', markersize=4, label='Seed 0'),
               Line2D([], [], marker='^', color='#aaaaaa', linestyle='none', markersize=4, label='Seed 1')]
    for name, key, title, xlabel, footer, limits in (
        ('supplementary/figS1_editing_coverage', 'coverage', 'Does broader single-edit training help later sequences?',
         'Broader minus narrower training (percentage points)',
         'All-question success after two / three edits; 98.75% paired situation intervals.\nRelative gains do not establish reliable joint control.', (-20, 50)),
        ('supplementary/figS2_fixed_readers', 'readers', 'Does changing the question improve answers from the same edited state?',
         'Alternative minus original question (percentage points)',
         'Same-side question accuracy after three edits; 99.375% paired situation intervals.\nExplicit-rule questions add instructions. Crossing zero does not establish equivalence.', (-36, 14)),
    ):
        rows = data[key]
        fig, ax = plt.subplots(figsize=(8, 3.6 if key == 'coverage' else 5.2))
        labels = []
        for index, row in enumerate(rows):
            y = len(rows) - 1 - index
            color = '#2166ac' if row['actor'] == 'gemma' else '#b2182b'
            low, effect, high = [100 * row[k] for k in ('lower', 'effect', 'upper')]
            ax.errorbar(effect, y, xerr=[[effect-low], [high-effect]], fmt='o', color=color, capsize=3, markersize=5)
            ax.scatter([100*row['seeds'][0]], [y-.13], marker='v', color='#777777', s=15, zorder=3)
            ax.scatter([100*row['seeds'][1]], [y+.13], marker='^', color='#aaaaaa', s=15, zorder=3)
            label = f"{row['actor'].title()} · {row['editor']}"
            if key == 'readers':
                label += ' · ' + row['reader']
            labels.append(label)
        ax.set_yticks(list(reversed(range(len(rows)))), labels)
        ax.set_ylim(-.6, len(rows)-.4)
        ax.set_xlim(*limits)
        ax.axvline(0, color='#555555', lw=.8)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', pad=12, fontsize=10)
        ax.legend(handles=legends, loc='upper center', bbox_to_anchor=(.5, -.21 if key == 'coverage' else -.13), ncol=3)
        fig.text(.5, .015, footer + '\nGemma: 64 synthetic situations; Qwen: 32. Models and studies are not pooled.', ha='center', va='bottom', fontsize=7)
        fig.tight_layout(rect=(0, .18 if key == 'coverage' else .14, 1, 1))
        module.save(fig, name)
    (module.OUT / 'editing_comparisons.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')


def reproduce(output: Path) -> dict:
    # Reuse the scientific plotting code instead of implementing a second renderer.
    spec = importlib.util.spec_from_file_location("public_paper_figures", ROOT / "scripts/make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = new_output(output / "figures")
    if any(destination.iterdir()):
        raise FileExistsError("Choose a fresh figure output directory; existing files are never overwritten")
    module.OUT = destination
    module.main()
    plot_editing(module)
    from .consequence_figures import plot
    plot(module)
    from .source_history_figure import plot as plot_source_history
    plot_source_history(module)
    from .cache_crossover_figure import plot as plot_cache_crossover
    plot_cache_crossover(module)
    extension_spec = importlib.util.spec_from_file_location('cache_crossover_extension_plot', ROOT / 'scripts/plot_cache_crossover.py')
    extension = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(extension)
    extension.plot(module)
    from .update_methods_figure import plot as plot_update_methods
    plot_update_methods(module)
    return {"status": "figures_regenerated", "output": str(destination),
            "scope": "committed figure source data; no model inference"}
