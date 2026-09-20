"""Local presentation only; consumes verified outputs, never runs a model.

This file is deliberately outside the frozen execution capsule. It cannot
change classifications, repeat selection, admission or scientific denominators.
"""
import argparse
import csv
import json
from pathlib import Path

METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')
COMPLETE = 'COMPLETE_FIXED_NEURAL_WORK_AND_CPU_REPLAY'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def percent(value):
    return 'undefined (no eligible denominator)' if value is None else f'{100*value:.2f}%'


def interval(value):
    return 'undefined' if value is None else f'[{percent(value[0])}, {percent(value[1])}]'


def validate_complete(runtime):
    terminal = read(runtime/'PIPELINE_TERMINAL.json')
    if terminal['status'] != COMPLETE:
        return False
    freeze = read(runtime/'ANALYSIS_FREEZE.json')
    order = freeze['case_order']
    assert len(order) == len(set(order)) == 64
    expected = {'final': (order, 578),
                'repeat0': (freeze['fixed_repeat_roots'], 136),
                'repeat1': (freeze['fixed_repeat_roots'], 136),
                'repeat_full': (read(runtime/'REPEAT_SELECTION.json')['roots'], 578)}
    counts = {}
    pids = []
    for phase, (roots, per_case) in expected.items():
        folder = runtime/'results/qwen32_t4'/phase
        pids.append(read(folder/'PHASE_DONE.json')['scorer_pid'])
        counts[phase] = 0
        for rid in roots:
            done = read(folder/rid/'DONE.json')
            assert done['logical_scores'] == per_case
            counts[phase] += done['logical_scores']
    assert len(set(pids)) == 4
    assert counts == dict(final=36992, repeat0=1088, repeat1=1088, repeat_full=2312)
    replay = read(runtime/'analysis/INDEPENDENT_REPLAY.json')
    assert replay['status'] == 'PASS' and replay['raw_final_rows'] == 36992
    assert replay['imports_experiment_classification'] is False
    assert len(replay['full_repeat_reconstruction']) == 4
    repeats = read(runtime/'REPEAT_CHECKS.json')
    assert repeats['compared_reads'] == 4488
    assert repeats['fixed_panel_reads'] == 2176 and repeats['complete_case_reads'] == 2312
    assert len(repeats['complete_case_classification']) == 4
    return True


def selected_example(runtime, cases, repeats):
    # A repeat-verified label is withheld on any unresolved repeat discrepancy.
    if repeats['status'] != 'PASS':
        return None
    verified = {r['root_id'] for r in repeats['complete_case_classification']
                if r['classification_identical']}
    order = read(runtime/'ANALYSIS_FREEZE.json')['case_order']
    for key in ('high_margin_pairs', 'strongest_pairs'):
        for rid in order:
            if rid not in verified:
                continue
            repeated = read(runtime/'analysis/full_repeats'/(rid+'.json'))
            for method in METHODS:
                for row in cases[rid][method]['semantic']:
                    twin = next(r for r in repeated[method]['semantic']
                                if r['target_ids'] == row['target_ids'])
                    for pair in row[key]:
                        if pair in twin[key]:
                            return dict(root_id=rid, method=method, level=key, pair=pair, row=row)
    return None


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


def complete_report(runtime, output):
    summary = read(runtime/'analysis/SUMMARY.json')
    repeats = read(runtime/'REPEAT_CHECKS.json')
    cases = {rid:read(runtime/'analysis/cases'/(rid+'.json'))
             for rid in read(runtime/'ANALYSIS_FREEZE.json')['case_order']}
    example = selected_example(runtime, cases, repeats)
    tables(output, cases)
    figures(runtime, output, summary, example)
    lines = ['# Qwen3-32B: full-forward current-fact recovery', '',
             'The fixed neural workload and independent raw-score reconstruction completed. '
             f'Repeat disposition: **{repeats["status"]}**.', '',
             'This is a new measurement on the 64 original designs, not 64 new semantic cases. '
             'All 36,992 FINAL reads recompute the full prompt at batch 1 with no cache. '
             'The BF16 backbone, FP32 full-vocabulary scoring and original invalid-output denominators remain fixed.', '',
             '## Distinct evidence levels', '',
             '| Method | Primary W / 1,152 | Mean W/18; 99.375% interval | Certified A&C | Strongest / 576 | High-margin / 576 |',
             '|---|---:|---|---:|---:|---:|']
    for method in METHODS:
        s = summary[method]; p = s['primary']; c = s['semantic_counts']
        lines.append(f'| {method} | {p["W"]} | {percent(p["witness_rate"])}; '
                     f'{interval(p["primary_family8_interval99375"])} | {percent(p["E_coverage"]["estimate"])} | '
                     f'{c["strongest_pairs"]} | {c["high_margin_pairs"]} |')
    lines += ['', 'Primary intervals retain the inherited eight-cell 99.375% coverage; descriptive intervals use 95%. '
              'Both resample whole designs (10,000 draws; seed 2609141002). Repeats do not enter population estimates.', '',
              'Strongest examples require all 12 direct checks, native/no-op targets, every required same-pair/cross-pair '
              'logical control, and the SAME changed-count/prefix-length-matched pair with the SAME semantic direction '
              'in both formats. High-margin evidence uses that same pair. These analyses are model-held-out confirmation '
              'informed by previous models, not blind metric discovery.', '']
    for method in METHODS:
        s = summary[method]; p = s['primary']
        lines += ['## '+method, '', '| Diagnostic | Estimate | Descriptive 95% case interval |', '|---|---:|---|']
        for k in ('A_coverage', 'C_coverage', 'E_coverage', 'conditional_witness', 'direct_accuracy', 'joint_accuracy',
                  'native_accuracy', 'noop_accuracy', 'all_origin_joint_accuracy', 'battery_accuracy', 'complete_case_success'):
            lines.append(f'| {k} | {percent(p[k]["estimate"])} | {interval(p[k]["interval95"])} |')
        lines += ['', f'Invalid responses: {p["invalid_responses"]} / 17,408. Candidate-mass quantiles: '
                  f'`{json.dumps(p["candidate_mass"]["quantiles"], sort_keys=True)}`. '
                  f'Inherited strong-margin coded witnesses: {p["strong"]} / 1,152.', '',
                  'Distinct designs with semantic survivors: `'+json.dumps(s['designs_with_survivor'], sort_keys=True)+'`.', '',
                  '| Ordered exclusion or survivor | Entering | Count | Remaining |', '|---|---:|---:|---:|']
        for row in s['ordered_attrition_flow']:
            lines.append(f'| {row["gate"]} | {row["entering"]} | {row["count"]} | {row["remaining"]} |')
        lines += ['', 'Overlapping failures (not additive): `'+json.dumps(s['overlapping_failures'], sort_keys=True)+'`.', '',
                  'Changed inspected-question verdicts: `'+json.dumps(s['evaluation_decision_change'], sort_keys=True)+'`.', '',
                  'Separate negated/control accuracy: `'+json.dumps(s['negated_control_accuracy'], sort_keys=True)+'`.', '',
                  'Certification-conditioned zero-event bounds: `'+json.dumps(s['zero_event_bounds'], sort_keys=True)+'`.', '']
    lines += ['## Repetition and claim limits', '',
              f'Two original fixed-panel fresh-process repeats: 2,176 reads. Four complete-case repeats: 2,312 reads. '
              f'All {repeats["compared_reads"]} comparisons are retained; {len(repeats["discrepancies"])} discrepancies. '
              'Complete-case selection followed the frozen positive-case reproducibility priority and seeded fill; it is not random population estimation.', '',
              ('A complete repeat-verified strongest example is shown in figure 2 and REPEATED_EXAMPLE.json.' if example else
               'No strongest example earns a repeat-verified label here; figure 2 shows attrition instead.'), '',
              ('Any unresolved repeat discrepancy suspends definitive inference; all observed values remain reported.'
               if repeats['status'] != 'PASS' else 'The prescribed repeat checks passed; no favorable/stability rerun was performed.'), '',
              'A small witness count with poor certification is limited assay coverage, not good updating. '
              'A zero-width bootstrap does not prove absence. One-sided zero-event bounds assume independent/exchangeable '
              'generator-drawn case designs and use the eligible case count, never correlated question counts.', '',
              'The original Qwen32 NOT_MEASURED disposition, failed cached/full comparison and failed historical bridges remain intact. '
              'No paired size comparison, quantitative scaling law, mechanism, repair, moral-decision result or unseen-error prediction follows.', '',
              '## Reproduction and custody', '',
              'See EXECUTION_AMENDMENT.json and RUN_CAPSULE.json for exact execution provenance; '
              'analysis/SUMMARY.json for coded logical-control sensitivities and semantic intervals; '
              'PER_CASE_PRIMARY.csv and PER_CASE_SEMANTIC_GATES.csv for all case-level tables; '
              'REPEAT_CHECKS.json for every discrepancy; analysis/INDEPENDENT_REPLAY.json for the independent reconstruction; '
              'operator/ for exact commands, allocation, custody and termination receipts.', '']
    write(output/'RESULTS.md', '\n'.join(lines))
    delta = ['# Story delta', '', 'Qwen3-32B is now measured through the separately validated stateless full-forward path.', '']
    for method in METHODS:
        p, c = summary[method]['primary'], summary[method]['semantic_counts']
        delta.append(f'- {method}: basic recurrence {p["W"]}/1,152 coded targets; strongest semantic conjunction '
                     f'{c["strongest_pairs"]}/576; same-pair high-margin conjunction {c["high_margin_pairs"]}/576; '
                     f'current-fact/reference certification {percent(p["E_coverage"]["estimate"])}.')
    delta += ['', f'Repeat status: {repeats["status"]}. Interpret recurrence alongside competence, certification and attrition. '
              'The stronger story is earned only by qualifying same-output examples, and repeated evidence only by passing complete-case repeats.', '',
              'Unchanged: historical failed bridges and the original technical omission. No quantitative size-scaling claim, '
              'new mechanism, intervention, repair claim, paper edit or automatic successor.', '']
    write(output/'STORY_DELTA.md', '\n'.join(delta))


def incomplete_report(runtime, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    terminal = read(runtime/'PIPELINE_TERMINAL.json') if (runtime/'PIPELINE_TERMINAL.json').exists() else read(runtime/'SETUP_FAILURE.json')
    evidence = {n:read(runtime/n) for n in ('READINESS.json', 'PIPELINE_ERROR.json', 'SETUP_FAILURE.json', 'technical/TECHNICAL_CHECKS.json')
                if (runtime/n).exists()}
    complete_cases = list((runtime/'results/qwen32_t4/final').glob('*/DONE.json'))
    text = ['# Qwen3-32B recovery: technical or incomplete boundary', '',
            f'Execution status: **{terminal["status"]}**. Completed FINAL cases: {len(complete_cases)}/64.', '',
            'The full planned experiment did not complete. No missing measurement is interpreted as zero recurrence, '
            'correct updating, or a successful stronger-control result. Partial scores, if any, are retained but not '
            'silently substituted for the fixed 64-case population.', '',
            'The original NOT_MEASURED row and failed historical comparisons remain unchanged.', '',
            '## Technical evidence', '', '```json', json.dumps(evidence, indent=2), '```', '']
    write(output/'RESULTS.md', '\n'.join(text))
    write(output/'STORY_DELTA.md', '# Story delta\n\nNo new complete scientific row. The recovery supplies a technical boundary, '
          'not evidence for or against history dependence. Original omissions and bridge failures remain intact.\n')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.text(.03, .9, 'Scientific coverage: not admitted or incomplete', fontsize=17, va='top')
    ax.text(.03, .65, f'Status: {terminal["status"]}\nCompleted FINAL cases: {len(complete_cases)} / 64\n'
            'Missing outcomes are not zeros. No population claim is made.', fontsize=12, va='top')
    fig.savefig(output/'FIGURE_1_TECHNICAL_BOUNDARY.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    timing = runtime/'technical/LATENCY.json'
    if timing.exists():
        rows = read(timing)['timings']
        ax.scatter([r['request']['bin'] for r in rows], [r['seconds'] for r in rows], color='#197278')
        ax.set(xlabel='Full-prompt length bin (64 tokens)', ylabel='Measured seconds per full forward',
               title='DEV-only full-forward latency; no efficacy admission gate')
    else:
        ax.axis('off')
        ax.text(.03, .8, 'No complete DEV latency benchmark', fontsize=17)
        ax.text(.03, .5, 'Setup / technical evidence is retained in RESULTS.md.\n'
                'Logical-control attrition cannot be measured without the planned outcomes.', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'FIGURE_2_TECHNICAL_BOUNDARY.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    terminal = a.runtime/'PIPELINE_TERMINAL.json'
    if terminal.exists() and validate_complete(a.runtime):
        complete_report(a.runtime, a.output)
    else:
        incomplete_report(a.runtime, a.output)
    print(json.dumps(dict(status='LOCAL_REPORT_WRITTEN_NOT_FINAL_ACCEPTANCE', output=str(a.output))))


if __name__ == '__main__':
    main()
