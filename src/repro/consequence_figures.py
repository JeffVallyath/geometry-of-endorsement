"""Direct-answer and source-history figures from committed measurements only."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .common import ROOT

COUNTERFACTUAL = 'reproducibility/representation/counterfactual_fidelity/results.json'
SOURCE_ROOTS = 'reproducibility/relational_editing/v6/scores/source_root_statistics.json.gz'
POPULATION = 'reproducibility/relational_editing/v6/data/gemma_source_roots.jsonl'


def comparison_data(root: Path = ROOT) -> dict:
    results = json.loads((root / COUNTERFACTUAL).read_text(encoding='utf8'))
    saved = results['headline']['gemma']
    if saved['worlds'] != 96 or results['headline']['llama']['status'] != 'ASSAY_NOT_QUALIFIED':
        raise ValueError('Counterfactual population or reference qualification changed')
    arms = []
    for key, label in (
        ('ORIGINAL_FROZEN_DIM@0.75', 'Rank-one relation steering'),
        ('NATURAL_FULL_STATE_INTERPOLATION@0.75', 'Whole-state interpolation'),
        ('FULL_STATE_SOURCE_PATCH', 'Natural changed-state patch'),
    ):
        estimate = saved['csr'][key]['csr']
        if estimate['n'] != saved['worlds']:
            raise ValueError('Counterfactual arms must use the same final worlds')
        arm = dict(key=key, label=label, recovery=estimate['mean'], lower=estimate['ci_low'],
                   upper=estimate['ci_high'], worlds=estimate['n'])
        if '@' in key:
            name, tau = key.split('@')
            dose = saved['dose'][name][tau]
            if dose['n'] != estimate['n'] or dose['fraction_applied'] != 1.0:
                raise ValueError('Measured margin comparison requires complete applied coverage')
            arm.update(margin_hit=dose['criteria']['within_tolerance'],
                       hard_answer_correct=dose['criteria']['hard_answer_counterfactual'],
                       strong_target_pass=dose['dose_valid'])
        else:
            arm.update(margin_hit=None, hard_answer_correct=None, strong_target_pass=None)
        arms.append(arm)

    ledger = json.loads(gzip.decompress((root / SOURCE_ROOTS).read_bytes()))['per_root']
    groups = []
    for actor, expected_roots in (('gemma', 64), ('qwen', 32)):
        for editor, label in (('INV', 'Constrained'), ('FREE', 'Free overwrite')):
            for seed in (0, 1):
                condition = f'{editor}_COMPLETE_SINGLE_s{seed}'
                rows = [r for r in ledger if (r['actor'], r['condition'], r['reader']) == (actor, condition, 'R0')]
                if len(rows) != expected_roots or len({r['root_id'] for r in rows}) != expected_roots:
                    raise ValueError('Source-history panel requires every unique root in each model/editor/seed')
                eligible, disagree, bundle = [], [], []
                for row in rows:
                    if row['seed'] != seed or row['all34']['questions'] != 34 or row['all34']['origins'] != 4:
                        raise ValueError('Source-history seed, question bundle, or origin inventory changed')
                    if row['by_family']['direct']['questions'] != 12 or row['by_family']['opposes']['questions'] != 4 or row['joint_only']['questions'] != 18:
                        raise ValueError('Atomic and joint question partitions changed')
                    atomic_correct = all(row['by_family'][family]['all_origin_bundle_correct'] == 1.0
                                         for family in ('direct', 'opposes'))
                    if atomic_correct:
                        eligible.append(row['root_id'])
                        if row['joint_only']['origin_disagreement'] > 0:
                            disagree.append(row['root_id'])
                    if row['all34']['all_origin_bundle_correct'] == 1.0:
                        bundle.append(row['root_id'])
                groups.append(dict(actor=actor, editor=label, condition=condition, reader='R0', seed=seed,
                    roots=expected_roots, atomic_perfect=len(eligible), atomic_perfect_with_joint_disagreement=len(disagree),
                    full_bundle_correct=len(bundle), eligible_root_ids=eligible, disagreement_root_ids=disagree))

    # A real committed design example, not fabricated model answers.
    example = json.loads((root / POPULATION).read_text(encoding='utf8').splitlines()[0])
    commands = example['commands']
    initial = [[source['values'][e['actor']][e['project']] for e in commands] for source in example['sources']]
    terminal = [e['value'] for e in commands]
    if len(initial) != 4 or len(terminal) != 3:
        raise ValueError('Source example requires four starts and three assignments')
    return dict(counterfactual=dict(model='gemma', worlds=96, interval_level=0.95, target_fraction=0.75,
                    bootstrap_draws=10000, arms=arms, source=COUNTERFACTUAL,
                    qualification='Measured transport-arm margins; full strong-target requirements failed. Llama reference unqualified.'),
                source_history=dict(groups=groups, source=SOURCE_ROOTS, reader='R0', origins=4,
                    atomic_questions=16, joint_questions=18, bundle_questions=34,
                    interpretation='Descriptive root counts, separately by model/editor/seed; not a new primary test or pooled estimate.',
                    example=dict(root_id=example['root_id'], source=POPULATION,
                        actor_names=[example['sources'][0]['actors'][e['actor']] for e in commands],
                        project=example['sources'][0]['projects'][commands[0]['project']],
                        initial_values=initial, requested_final_values=terminal)))


def plot(module) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = comparison_data()
    arms = data['counterfactual']['arms']
    colors = ['#b34c3e', '#267f9b', '#487d4b']
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={'width_ratios': [1, 1.3]})
    for index, (row, color) in enumerate(zip(arms, colors)):
        y = 2 - index
        if row['margin_hit'] is not None:
            left.barh(y, 100 * row['margin_hit'], height=.42, color=color)
            left.text(103, y, f"{100 * row['margin_hit']:.0f}%", va='center', fontsize=9)
        else:
            left.text(3, y, 'Reference patch; not margin-tuned', va='center', fontsize=8)
        right.errorbar(row['recovery'], y,
            xerr=[[row['recovery'] - row['lower']], [row['upper'] - row['recovery']]],
            fmt='o', color=color, capsize=4, markersize=6)
        right.text(row['recovery'], y + .25, f"{row['recovery']:.3f}", ha='center', fontsize=9)
    left.set_yticks([2, 1, 0], [r['label'] for r in arms])
    right.set_yticks([2, 1, 0], [])
    for axis in (left, right):
        axis.set_ylim(-.55, 2.55)
        axis.grid(axis='y', visible=False)
    left.set_xlim(0, 120)
    left.set_xticks([0, 50, 100])
    left.set_title('Direct margin reached', loc='left', fontsize=11)
    left.set_xlabel('Final worlds within target tolerance (%)')
    right.set_xlim(-1.2, 1.2)
    right.axvline(0, color='#777777', linewidth=.8)
    right.axvline(1, color='#aaaaaa', linewidth=.8, linestyle=':')
    right.set_title('Broader consequence recovery', loc='left', fontsize=11)
    right.set_xlabel('Recovery score (not answer accuracy)')
    fig.suptitle('Reaching the requested answer does not recover its consequences', x=.02, ha='left', fontsize=13)
    fig.text(.5, .025,
        'Gemma · same 96 final worlds · target fraction 0.75 · saved 95% world-bootstrap intervals\n'
        'Recovery: 0 = unchanged starting state; 1 = natural change. Negative = farther from the natural pattern.\n'
        'Measured margins are not a full strong-target pass. Llama failed reference qualification; no cross-model comparison.',
        ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .20, 1, .92), w_pad=2.5)
    module.save(fig, 'fig6_answer_consequences')

    history = data['source_history']
    fig = plt.figure(figsize=(10.7, 8.0))
    design = fig.add_axes([.07, .74, .88, .17])
    design.axis('off')
    example = history['example']
    symbols = {0: 'Oppose', 1: 'Support'}
    starts = [' / '.join(symbols[b] for b in bits) for bits in example['initial_values']]
    design.text(0, 1, 'Different starting relations', fontsize=10, fontweight='bold', va='top')
    design.text(0, .76, '\n'.join(starts), fontsize=10, linespacing=1.5, va='top')
    design.annotate('', xy=(.57, .47), xytext=(.34, .47), arrowprops={'arrowstyle': '->', 'lw': 1.5})
    design.text(.455, .66, 'Same requested\nassignments', ha='center', fontsize=9)
    design.text(.60, .78, ' / '.join(symbols[b] for b in example['requested_final_values']), fontsize=11, va='top')
    design.text(.60, .49, 'Identical requested final relations\nFresh questions are asked afterward', fontsize=9, va='top')
    design.text(0, -.08, 'Saved design example: ' + ', '.join(example['actor_names']) + ' about ' + example['project'] + '. Outcomes are summarized below.', fontsize=8)
    axis = fig.add_axes([.22, .21, .55, .45])
    rows = history['groups']
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        disagreement = row['atomic_perfect_with_joint_disagreement']
        stable = row['atomic_perfect'] - disagreement
        atomic_error = row['roots'] - row['atomic_perfect']
        offset = 0
        for count, color in ((disagreement, '#bd5a4c'), (stable, '#408f9c'), (atomic_error, '#d9dce0')):
            width = 100 * count / row['roots']
            axis.barh(y, width, left=offset, height=.58, color=color)
            offset += width
        axis.text(103, y, f"{disagreement}/{row['atomic_perfect']} atomic-perfect roots", va='center', fontsize=9)
    axis.set_yticks(list(reversed(range(len(rows)))),
        [f"{r['actor'].title()} · {r['editor']} · seed {r['seed']}" for r in rows], fontsize=9)
    axis.set_xlim(0, 100)
    axis.set_ylim(-.65, len(rows)-.35)
    axis.set_xlabel('Share of terminal roots (%)')
    axis.grid(axis='y', visible=False)
    axis.text(103, len(rows)-.05, 'Joint answer disagreement in', fontsize=8, va='bottom')
    fig.suptitle('Correct atomic relations can still leave joint answers dependent on history',
                 x=.035, y=.97, ha='left', fontsize=13)
    fig.legend(handles=[Patch(color='#bd5a4c', label='All atomic answers correct; joint answers vary with starting state'),
                        Patch(color='#408f9c', label='All atomic answers correct; joint answers do not vary (may still be wrong)'),
                        Patch(color='#d9dce0', label='At least one atomic answer incorrect')],
               loc='lower left', bbox_to_anchor=(.04, .075), fontsize=8, frameon=False)
    fig.text(.5, .014,
        'Original reader · four starting states per root · Gemma: 64 roots; Qwen: 32 roots · both training seeds shown separately\n'
        'Atomic = all 16 direct/opposes questions correct across all starts; joint = 18 both/either/same questions.\n'
        'Descriptive counts from saved root statistics, not a new primary test. No pooling across models, editors, or seeds.',
        ha='center', va='bottom', fontsize=8)
    module.save(fig, 'fig7_source_history')
    (module.OUT / 'consequence_comparisons.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')
