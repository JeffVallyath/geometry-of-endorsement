"""Figure 7: display the frozen V10 primary estimates, without fitting an analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import ROOT

BASE = 'reproducibility/state_sufficiency/v10/'
PRIMARY = BASE + 'primary_results.json'
SPECIFICATION = BASE + 'specification_summary.json'
LABELS = {
    'INV_PAIR_NLL': 'Constrained learned editor',
    'FREE_PAIR_CONSISTENCY': 'Unrestricted consistency-trained editor',
    'EXISTING_CORRECTION': 'Existing textual correction',
    'LATEST_SAME_WORDING': 'Latest-value wording',
}
LEARNED = {'INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY'}


def figure_data(root: Path = ROOT) -> dict:
    """Copy full-precision primary means/CIs; do not substitute root prevalence."""
    primary = json.loads((root / PRIMARY).read_text(encoding='utf8'))
    spec = json.loads((root / SPECIFICATION).read_text(encoding='utf8'))
    expected = {(actor, group) for actor in ('gemma', 'qwen') for group in LABELS}
    if len(primary) != 8 or {(r['actor'], r['group']) for r in primary} != expected:
        raise ValueError('V10 requires exactly eight unique model/update cells')
    if (spec['root_denominator'], spec['joint_questions_per_root'], spec['inventory']['origins'],
        spec['interval_level'], spec['bootstrap_draws'], spec['bootstrap_seed'],
        spec['learned_seeds_averaged_within_root']) != (64, 18, 8, .99375, 10000, 2609141002, True):
        raise ValueError('V10 frozen design or primary interval specification changed')
    rows = []
    for actor in ('gemma', 'qwen'):
        for group, label in LABELS.items():
            source = next(r for r in primary if (r['actor'], r['group']) == (actor, group))
            if (source['roots'] != spec['root_denominator'] or
                len(source['root_ids']) != 64 or len(set(source['root_ids'])) != 64 or
                len(source['root_scores']) != 64 or not source['full_coverage'] or
                source['seeds_averaged_within_root'] != (group in LEARNED) or
                source['corrected_level'] != spec['interval_level'] or
                source['draws'] != spec['bootstrap_draws'] or source['seed'] != spec['bootstrap_seed']):
                raise ValueError('V10 primary coverage, seed averaging or interval changed')
            low, high = source['corrected_interval']
            if not 0 <= low <= source['mean'] <= high <= 1:
                raise ValueError('Invalid V10 primary interval')
            rows.append({key: source[key] for key in (
                'actor', 'group', 'roots', 'mean', 'corrected_interval', 'corrected_level',
                'draws', 'seed', 'seeds_averaged_within_root')})
            rows[-1].update(label=label, update_type='Learned' if group in LEARNED else 'Textual',
                           joint_questions_per_root=spec['joint_questions_per_root'],
                           fixed_opportunities_per_seed_or_textual_condition=64 * 18)
    return dict(
        study='V10 prospective confirmation', rows=rows,
        statistic='Mean root witness count / 18 fixed joint-question opportunities; learned seeds averaged within root',
        witness=spec['witness'], origins=spec['inventory']['origins'],
        interpretation='Behavioral source-history dependence despite protected factual readout; not root prevalence or individual-answer error rate',
        sources={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in (PRIMARY, SPECIFICATION)},
    )


def plot(module) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = figure_data()
    fig = plt.figure(figsize=(12.4, 8.6))
    fig.suptitle('Correct current facts can still leave downstream answers\n'
                 'dependent on source history.', x=.035, y=.98, ha='left', fontsize=16)
    fig.text(.035, .865, 'A  Matched-history design', fontsize=11, weight='bold')
    design = fig.add_axes([.035, .70, .93, .135])
    design.set_xlim(0, 1)
    design.set_ylim(0, 1)
    design.axis('off')
    stages = [
        (.075, 'History 1\nHistory 2\n… History n'),
        (.285, 'Same update\ncommands'),
        (.505, 'Same intended\ncurrent facts'),
        (.755, 'Same fresh\ndownstream question'),
    ]
    for x, label in stages:
        design.text(x, .63, label, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=.55', facecolor='#f1f4f7', edgecolor='#b9c4ce'))
    for start, end in ((.15, .215), (.355, .425), (.59, .655)):
        design.annotate('', xy=(end, .63), xytext=(start, .63),
                        arrowprops=dict(arrowstyle='->', lw=1.3, color='#52606d'))
    design.text(.94, .63, 'Compare\nanswers', ha='center', va='center', fontsize=10)
    design.annotate('', xy=(.892, .63), xytext=(.849, .63),
                    arrowprops=dict(arrowstyle='->', lw=1.3, color='#52606d'))
    design.text(.075, -.02, '8 starting\nhistories', ha='center', fontsize=9)
    design.text(.505, -.02, 'Check relevant direct facts\nafter updating', ha='center', fontsize=9, color='#245b70')
    design.text(.79, -.02, 'Question and answer interface\nheld fixed across histories', ha='center', fontsize=9)
    fig.text(.035, .64, 'Behavioral test: identical intended facts do not imply identical hidden states or erased history.',
             fontsize=9, color='#52606d')

    fig.text(.035, .593, 'B  Prospective matched-history study', fontsize=11, weight='bold')
    level = 100 * data['rows'][0]['corrected_level']
    fig.text(.035, .564, f'Qualifying-question rates and multiplicity-adjusted {level:g}% case-bootstrap intervals', fontsize=10)
    axis = fig.add_axes([.49, .19, .285, .34])
    ys = [8, 7, 6, 5, 3, 2, 1, 0]
    colors = {'Learned': '#276882', 'Textual': '#806296'}
    markers = {'Learned': 'o', 'Textual': 's'}
    for y, row in zip(ys, data['rows']):
        mean = 100 * row['mean']
        low, high = [100 * value for value in row['corrected_interval']]
        kind = row['update_type']
        axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=markers[kind],
                      color=colors[kind], capsize=3, markersize=6)
        label = row['label'].replace('Unrestricted consistency-trained editor',
                                     'Unrestricted consistency-trained\neditor')
        axis.text(-1.30, y, label, transform=axis.get_yaxis_transform(), va='center', fontsize=9)
        axis.text(1.08, y, f'{mean:.3f}  [{low:.3f}, {high:.3f}]',
                  transform=axis.get_yaxis_transform(), va='center', fontsize=9)
    for y, actor in ((8, 'Gemma'), (3, 'Qwen')):
        axis.text(-1.60, y, actor, transform=axis.get_yaxis_transform(), va='center', fontsize=10, weight='bold')
    axis.text(1.08, 9.05, 'Rate [interval], %', transform=axis.get_yaxis_transform(), fontsize=9)
    axis.axvline(0, color='#525b64', lw=1)
    axis.axhline(4, color='#d9dfe5', lw=.8)
    axis.set_xlim(-.35, 10)
    axis.set_ylim(-.65, 8.65)
    axis.set_xticks([0, 2, 4, 6, 8, 10])
    axis.set_yticks([])
    axis.spines['left'].set_visible(False)
    axis.set_xlabel('Qualifying-question rate (%)', fontsize=10)
    fig.legend(handles=[Line2D([], [], marker=markers[kind], color=colors[kind], linestyle='none',
                              label=f'{kind} updates') for kind in colors],
               loc='center left', bbox_to_anchor=(.032, .138), ncol=2, fontsize=9)
    fig.text(.49, .122, 'All eight adjusted intervals are above zero.', fontsize=10, weight='bold')
    fig.text(.035, .075,
             'Each row: 64 benchmark cases × 18 questions combining facts = 1,152 per seed or textual condition.\n'
             'Ineligible questions remain in the denominator; two learned seeds are averaged within each case.', fontsize=9)
    fig.text(.035, .02,
             'Qualifies: needed facts correct in every history; references correct; different answers to a question combining facts.\n'
             'References: current facts from the start, and the same update on already-correct facts. Not a case-prevalence or answer-error rate.',
             fontsize=9, color='#52606d')
    module.save(fig, 'fig07_source_history')
    (module.OUT / 'v10_source_history.json').write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')
