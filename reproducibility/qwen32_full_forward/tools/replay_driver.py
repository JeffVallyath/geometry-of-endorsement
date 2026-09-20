"""Adapt paths and presentation only; execute the supplied CPU algorithms unchanged."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(path.read_bytes())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def readable_figures(report, runtime, output, summary):
    """Use the original plot values; translate display labels only."""
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    labels = {
        'EXISTING_CORRECTION': 'Correction sentence',
        'LATEST_SAME_WORDING': 'Restated current values',
        'Direct accuracy': 'Individual-fact accuracy',
        'Native accuracy': 'Current-facts-only reference',
        'No-op accuracy': 'Already-correct update reference',
        'Certified A & C': 'Facts and references correct',
        'All-history targets': 'Joint answer correct in every history',
        'Primary W / 18': 'History-dependent targets',
        'invalid answer': 'Invalid answer',
        'failed direct readout': 'Incorrect individual fact',
        'failed native noop target': 'Incorrect reference answer',
        'failed other logical question': 'Incorrect other logical answer',
        'unavailable matched pair': 'No count/length-matched histories',
        'no target disagreement': 'No matched cross-format disagreement',
        'survivor': 'Qualifying comparison',
        'No repeat-verified strongest example: ordered gate attrition': 'No comparison passes the full two-format test',
        'Current-fact/reference coverage and all-history target behavior': 'Correct factual reports can coexist with history-dependent joint answers',
        'Semantic opportunities (fixed denominator 576)': 'Distinct joint questions (576 per procedure)',
    }
    original = Figure.savefig
    plotted = {}
    def save(fig, filename, **kwargs):
        plotted[Path(filename).name] = [[float(patch.get_width()) for patch in ax.patches] for ax in fig.axes]
        for ax in fig.axes:
            ticks = ax.get_yticklabels()
            if ticks:
                ax.set_yticks(ax.get_yticks(), labels=[labels.get(t.get_text(), t.get_text()) for t in ticks])
        for text in fig.findobj(Text):
            if text.get_text() in labels:
                text.set_text(labels[text.get_text()])
        fig.set_size_inches(14, 5)
        fig.tight_layout()
        original(fig, filename, **kwargs)
    Figure.savefig = save
    try:
        report.figures(runtime, output, summary, None)
    finally:
        Figure.savefig = original
    (output / 'PLOTTED_VALUES.json').write_text(json.dumps(plotted, indent=2) + '\n', encoding='utf8')


def worked_example(runtime, independent):
    rid, method, target = 'SSC1-FINAL-0003', 'LATEST_SAME_WORDING', 'SSC1-FINAL-0003-d0-09'
    check = next(row for row in independent['single_format_full_control_examples'] if row['root'] == rid)
    assert check['method'] == method and check['query_id'] == target and check['repeated_full']
    assert [1, 2] in check['matched_pairs'] and [1, 2] in check['high_margin_pairs']
    twin = check['twin_id']
    # This module is the supplied checker, with no study-classification imports.
    helper = module('qwen32_independent_example', runtime.parents[1] / 'handoff/independent_check.py')
    raw = helper.raw_load(runtime / f'results/qwen32_t4/final/{rid}/RAW_SCORES.jsonl')
    repeat = helper.raw_load(runtime / f'results/qwen32_t4/repeat_full/{rid}/RAW_SCORES.jsonl')
    design = load(runtime / f'cpu/designs/{rid}.json')
    questions = {q['query_id']: q for q in design['questions']}
    terminal = design['root']['terminal']['values']
    direct = [q for q in questions if questions[q]['spec']['kind'] == 'direct']
    logical = [q for q in questions if questions[q]['spec']['kind'] in ('same', 'both', 'either')
               and helper.code(q) == '0' and q != target]
    def correct(context, q):
        answer = raw[context, q]
        return answer['valid'] and answer['pred'] == helper.gold(questions[q], terminal)
    assert len(direct) == 12 and all(correct(f'o{o}/{method}', q) for o in range(8) for q in direct)
    assert len(logical) == 8 and all(correct(f'o{o}/{method}', q) for o in range(8) for q in logical)
    assert all(correct(context, target) for context in ('NATIVE_FINAL', f'o0/{method}'))
    assert not correct(f'o0/{method}', twin)
    assert [raw[f'o{o}/{method}', target]['pred'] for o in (1, 2)] == [0, 1]
    assert [raw[f'o{o}/{method}', twin]['pred'] for o in (1, 2)] == [1, 1]
    scientific_score = lambda row: {k: v for k, v in row['raw']['score'].items() if k != 'seconds'}
    assert all(scientific_score(row) == scientific_score(repeat[key]) and row['raw']['tokens'] == repeat[key]['raw']['tokens']
               for key, row in raw.items())
    target_spec = questions[target]['spec']
    pair, project = helper.sem(questions[target])[:2]
    actors = design['root']['terminal']['actors']
    projects = design['root']['terminal']['projects']
    assert projects[project] == 'Gallery' and {actors[a] for a in pair} == {'Mina', 'Wren'}
    assert all(terminal[actor][project] == 0 for actor in pair)
    sources = design['root']['sources']
    assert sources[1]['values'][actors.index('Mina')][project] == 1
    assert sources[2]['values'][actors.index('Ivo')][project] == 1
    assert all(helper.gold(questions[target], sources[o]['values']) == 0 for o in (1, 2))
    seals = load(runtime / f'results/qwen32_t4/final/{rid}/CONTEXT_SEAL.json')['contexts']
    lengths = [len(seals[f'o{o}/{method}']['prefix_ids']) for o in (1, 2)]
    changes = [sum(sources[o]['values'][c['actor']][c['project']] != c['value'] for c in design['root']['commands']) for o in (1, 2)]
    assert lengths == [834, 834] and changes == [1, 1]
    displayed = [q for q in direct if questions[q]['spec']['a'] in pair and questions[q]['spec']['p'] == project and helper.code(q) == '0']
    displayed += [q for q in logical if helper.sem(questions[q])[:2] == (pair, project)] + [target]
    assert len(displayed) == 5
    rows = [dict(query_id=q, question=questions[q]['text'], correct=helper.gold(questions[q], terminal),
                 answers=[raw[f'o{o}/{method}', q]['pred'] for o in (1, 2)]) for q in displayed]
    scores = [raw[f'o{o}/{method}', target]['prob'][0] for o in (1, 2)]
    masses = [raw[f'o{o}/{method}', target]['mass'] for o in (1, 2)]
    assert abs(scores[0] - .9983893724) < 5e-11 and abs(scores[1] - .04365346482) < 5e-12
    assert abs(masses[0] - .9999828285) < 5e-11 and abs(masses[1] - .9999966971) < 5e-11
    return dict(root_id=rid, method=method, target=target, other_format_target=twin, origins=[1, 2],
                prefix_tokens=lengths, changed_facts=changes, table=rows, correct_answer_scores=scores,
                candidate_masses=masses, direct_checks=12, direct_histories=8, logical_checks=8,
                target_references_correct=True, other_format_noop_correct=False,
                other_format_selected_answers=[1, 1], repeated_records=len(repeat),
                repeated_scores_and_token_bindings_identical=True, old_target_truth=[0, 0],
                selection_for_exposition='post_hoc', strongest_cross_format=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.source / 'results/runtime'
    args.output.mkdir(parents=True, exist_ok=False)
    # Run in this already offline-audited process; no child can evade that guard.
    sys.path.insert(0, str(runtime / 'code'))
    analyzer = module('qwen32_source_analyzer', runtime / 'code/qwen32_full_forward/analyze.py')
    analysis = args.output / 'analysis'
    analyzer.run(runtime, analysis)
    import runpy
    def command(path, argv):
        saved = sys.argv
        try:
            sys.argv = [str(path), *map(str, argv)]
            runpy.run_path(str(path), run_name='__main__')
        finally:
            sys.argv = saved
    command(runtime / 'code/qwen32_full_forward/replay.py', ['--runtime', runtime, '--analysis', analysis])
    repeats = module('qwen32_source_repeat_checks', runtime / 'code/qwen32_full_forward/repeat_checks.py')
    original_save = repeats.save
    def save_repeat_output(path, value):
        # The original CLI writes below its input runtime. Redirect only those
        # derived outputs; never compare a new serialization by overwriting it.
        if path == runtime / 'REPEAT_CHECKS.json':
            destination = args.output / 'REPEAT_CHECKS.json'
        elif path.is_relative_to(runtime / 'analysis/full_repeats'):
            destination = analysis / 'full_repeats' / path.name
        else:
            raise ValueError('Unexpected repeat output: ' + str(path))
        original_save(destination, value)
    repeats.save = save_repeat_output
    saved_argv = sys.argv
    try:
        sys.argv = ['repeat_checks.py', '--runtime', str(runtime)]
        repeats.main()
    finally:
        sys.argv = saved_argv
    command(args.source / 'handoff/independent_check.py', ['--source', args.source / 'results', '--output', args.output / 'independent'])
    independent = load(args.output / 'independent/INDEPENDENT_CHECK.json')
    example = worked_example(runtime, independent)
    (args.output / 'WORKED_EXAMPLE.json').write_text(json.dumps(example, indent=2) + '\n', encoding='utf8')
    report = module('qwen32_original_reporting', args.source / 'results/reporting_code/report.py')
    cases = {p.stem: load(p) for p in sorted((analysis / 'cases').glob('*.json'))}
    summary = load(analysis / 'SUMMARY.json')
    presentation = args.output / 'presentation'
    presentation.mkdir()
    report.tables(presentation, cases)
    readable_figures(report, runtime, presentation, summary)
    (presentation / 'FIGURE_DATA.json').write_text(json.dumps({method: dict(
        primary=summary[method]['primary'], ordered_attrition_flow=summary[method]['ordered_attrition_flow'])
        for method in report.METHODS}, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
