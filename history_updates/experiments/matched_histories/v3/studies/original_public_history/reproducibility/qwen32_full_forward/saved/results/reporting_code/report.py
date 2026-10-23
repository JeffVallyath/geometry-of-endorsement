"""Render the retained 32B tables and plots from saved results."""
import argparse


import csv


import json


from pathlib import Path


METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')


COMPLETE = 'COMPLETE_FIXED_NEURAL_WORK_AND_CPU_REPLAY'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tables(output, cases):
    primary, semantic = [], []
    for rid, case in cases.items():
        for method in METHODS:
            primary.append(dict(root_id=rid, method=method, **{
                k:v for k,v in case[method]['metrics'].items()
                if not isinstance(v, (list, dict)) and k != 'root_id'}))
            for row in case[method]['semantic']:
                semantic.append({k:json.dumps(v, separators=(',', ':'))
                                 if isinstance(v, (list, dict)) else v for k,v in row.items()})
    for name, rows in (('PER_CASE_PRIMARY.csv', primary), ('PER_CASE_SEMANTIC_GATES.csv', semantic)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output/name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)


def figures(runtime, output, summary, example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ('#197278', '#ce6c47')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    names = ['Direct accuracy', 'Native accuracy', 'No-op accuracy', 'Certified A & C',
             'All-history targets', 'Primary W / 18']
    for ax, method, color in zip(axes, METHODS, colors):
        p = summary[method]['primary']
        rows = [p[k] for k in ('direct_accuracy', 'native_accuracy', 'noop_accuracy',
                                'E_coverage', 'all_origin_joint_accuracy')]
        rows += [dict(estimate=p['witness_rate'], interval95=p['interval95'])]
        estimates = np.array([r['estimate'] for r in rows])
        errors = np.array([[max(0, r['estimate']-r['interval95'][0]),
                            max(0, r['interval95'][1]-r['estimate'])] for r in rows]).T
        ax.barh(names, estimates*100, xerr=errors*100, color=color, capsize=3)
        for n, row in enumerate(rows):
            ax.text(100*row['interval95'][1]+2, n, f'{100*row["estimate"]:.2f}%', va='center', fontsize=8)
        ax.set(xlim=(0, 112), xticks=(0,25,50,75,100), xlabel='Percent (descriptive 95% case intervals)', title=method)
    axes[0].invert_yaxis()
    fig.suptitle('Current-fact/reference coverage and all-history target behavior')
    fig.tight_layout()
    fig.savefig(output/'FIGURE_1_COVERAGE.png', dpi=160)
    plt.close(fig)
    if example is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for ax, method, color in zip(axes, METHODS, colors):
            flow = summary[method]['ordered_attrition_flow']
            ax.barh([r['gate'].replace('_', ' ') for r in flow], [r['count'] for r in flow], color=color)
            ax.invert_yaxis()
            ax.set(title=method, xlabel='Semantic opportunities (fixed denominator 576)')
        fig.suptitle('No repeat-verified strongest example: ordered gate attrition')
        fig.tight_layout()
        fig.savefig(output/'FIGURE_2_ATTRITION.png', dpi=160)
        plt.close(fig)
        return
    # Display the exact fixed pair across both formats and all logical controls.
    from qwen32_full_forward.replay import decode, truth
    rid, method, row = example['root_id'], example['method'], example['row']
    design = read(runtime/'cpu/designs'/(rid+'.json'))
    questions = {q['query_id']:q for q in design['questions']}
    values = design['root']['terminal']['values']
    def load_index(phase):
        index = {}
        with (runtime/'results/qwen32_t4'/phase/rid/'RAW_SCORES.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                index[item['context'], item['query_id']] = decode(item)
        return index
    raw, repeated_raw = load_index('final'), load_index('repeat_full')
    def score(context, q, index=None):
        index = raw if index is None else index
        return index[context, q]['prob'][truth(questions[q]['spec'], values)]
    pair = example['pair']
    target_spec = questions[row['target_ids'][0]]['spec']
    target_gold = truth(target_spec, values)
    assert row['changed_fact_counts'][pair[0]] == row['changed_fact_counts'][pair[1]]
    assert row['prefix_token_lengths'][pair[0]] == row['prefix_token_lengths'][pair[1]]
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    target_axes = [fig.add_subplot(grid[0, 0])]
    target_axes.append(fig.add_subplot(grid[0, 1], sharey=target_axes[0]))
    controls = fig.add_subplot(grid[1, :])
    records, repeat_records = [], []
    for fmt, q in enumerate(row['target_ids']):
        ax = target_axes[fmt]
        y = [score(f'o{o}/{method}', q) for o in range(8)]
        ry = [score(f'o{o}/{method}', q, repeated_raw) for o in range(8)]
        ax.plot(range(8), y, 'o-', markersize=4, label='FINAL', color=colors[fmt])
        ax.plot(range(8), ry, 's--', markersize=7, markerfacecolor='none',
                linewidth=1, label='Complete repeat', color=colors[fmt])
        invalid = [o for o in range(8) if not raw[f'o{o}/{method}', q]['valid'] and y[o] is not None]
        if invalid:
            ax.scatter(invalid, [y[o] for o in invalid], color='black', marker='x', s=60,
                       label='Invalid target read (retained)', zorder=5)
        ax.scatter(example['pair'], [y[o] for o in example['pair']], s=130,
                   facecolors='none', edgecolors=colors[fmt], linewidths=2)
        for origin in range(8):
            s = raw[f'o{origin}/{method}', q]
            records.append(dict(query_id=q, history=origin, correct_score=y[origin], **s))
            repeat_records.append(dict(query_id=q, history=origin, correct_score=ry[origin],
                                       **repeated_raw[f'o{origin}/{method}', q]))
    for fmt, ax in enumerate(target_axes):
        ax.set(xticks=range(8), ylim=(-.02, 1.02), xlabel='Starting history',
               ylabel='Two-code-normalized correct-answer score' if fmt == 0 else '',
               title=f'Format {fmt}: same matched histories {pair[0]}/{pair[1]}')
        if fmt: ax.tick_params(axis='y', labelleft=False)
        ax.legend(loc='best', fontsize=8)
    minima = [min(score(f'o{o}/{method}', q) for o in range(8)) for q in row['logical_ids']]
    repeat_minima = [min(score(f'o{o}/{method}', q, repeated_raw) for o in range(8)) for q in row['logical_ids']]
    x = np.arange(len(minima))
    controls.bar(x-.2, minima, width=.4, color='#526d82', label='FINAL')
    controls.bar(x+.2, repeat_minima, width=.4, facecolor='white', edgecolor='#526d82',
                 hatch='//', label='Complete repeat')
    controls.set(ylim=(0, 1.02), xticks=range(len(minima)),
                 xticklabels=[q.replace(rid+'-', '') for q in row['logical_ids']],
                 ylabel='Minimum correct score', xlabel='Every required other logical question; minimum over all 8 histories')
    controls.tick_params(axis='x', labelrotation=45)
    controls.legend(fontsize=8, loc='lower right')
    pair_mass = min(index[f'o{o}/{method}', q]['mass'] for index in (raw, repeated_raw)
                    for o in pair for q in row['target_ids'])
    matching = (f'Matched histories {pair[0]}/{pair[1]}: changed-fact count '
                f'{row["changed_fact_counts"][pair[0]]} each; '
                f'{row["prefix_token_lengths"][pair[0]]} prefix tokens each. '
                f'Minimum paired target candidate mass: {pair_mass:.3f}.')
    refs = ('Native and same-method no-op targets, all 12 direct checks, and all plotted logical controls '
            'are valid/correct in FINAL and complete repeat.\n'
            'Two-code-normalized scores are descriptive, not calibrated beliefs; logical correctness is not a high-confidence claim.')
    fig.suptitle('Complete-case repeat-verified '+('high-margin strongest' if example['level']=='high_margin_pairs' else 'strongest')+' example\n'
                 f'{rid} / {method}\n'
                 f'Target: {target_spec["kind"]}({target_spec["a"]}, {target_spec["b"]}; {target_spec["p"]}); '
                 f'correct semantic answer: {"yes" if target_gold else "no"}', fontsize=12)
    fig.text(.5, .045, matching+'\n'+refs, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .96))
    fig.savefig(output/'FIGURE_2_REPEATED_EXAMPLE.png', dpi=160)
    plt.close(fig)
    example_data = dict(**example, target_semantic_gold=target_gold,
        target_score_records=records, repeat_target_score_records=repeat_records,
        exact_question_texts={q:questions[q]['text'] for q in row['target_ids']+row['logical_ids']+row['direct_ids']},
        native_target_correct_scores=[score('NATIVE_FINAL', q) for q in row['target_ids']],
        noop_target_correct_scores=[score(f'o0/{method}', q) for q in row['target_ids']],
        logical_minimum_correct_scores=minima,
        repeat_logical_minimum_correct_scores=repeat_minima,
        minimum_paired_target_candidate_mass=pair_mass,
        direct_minimum_correct_scores=[min(score(f'o{o}/{method}', q) for o in range(8)) for q in row['direct_ids']],
        full_case_repeat_evidence='analysis/full_repeats/'+rid+'.json',
        normalized_scores_are_not_calibrated_beliefs=True)
    write(output/'REPEATED_EXAMPLE.json', json.dumps(example_data, indent=2))


