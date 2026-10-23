"""Build scientific table data and every public presentation from replay outputs."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

MODELS = ('Qwen2.5-7B', 'Qwen2.5-32B', 'Gemma-2-27B')
STUDIES = ('repair_qwen7', 'repair_qwen32', 'repair_gemma27')
SEEDS = ('STATE_0', 'STATE_1')
NATIVE = dict(zip(MODELS, ('HISTORY', 'NATIVE32_SAVED', 'NATIVE_GEMMA')))
CONDITIONS = ('NATIVE', 'REFLECT', 'CURRENT_TEXT', 'CURRENT_JSON', 'STATE_BASE', *SEEDS)
TABLE_DIRECTORY = Path('paper/tables')
BEGIN = '<!-- BEGIN GENERATED REPLAY TABLES -->'
END = '<!-- END GENERATED REPLAY TABLES -->'


def load(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def legacy_result(values):
    """Preserve all v1 scientific fields and their original list ordering."""
    native = []
    repair = []
    for model, title in [('qwen', 'Qwen2.5-7B'), ('gemma', 'Gemma-2-9B')]:
        r = values['original_native'][model]['primary']
        native.append({'measurement': 'Original', 'model': title, 'subjects': r['subjects'],
            'targets': r['target_phrasings'], 'qualified': r['qualified'], 'guards': r['common_guard'],
            'qualified_interval_98_75': r['qualified_interval'], 'relevance_difference': r['relevance_difference'],
            'relevance_interval_98_75': r['relevance_interval'], 'source_replay': 'original_native'})
    for model, title in [('small', 'Qwen2.5-7B'), ('large', 'Qwen2.5-32B')]:
        r = values['paired_native']['models'][model]['primary']
        native.append({'measurement': 'Fresh paired', 'model': title, 'subjects': r['subjects'],
            'targets': r['target_phrasings'], 'qualified': r['qualified'], 'guards': r['common_guard'],
            'qualified_interval_98_75': r['qualified_interval'], 'relevance_difference': r['relevance_difference'],
            'relevance_interval_98_75': r['relevance_interval'], 'source_replay': 'paired_native'})
    q7 = values['repair_qwen7']
    for arm, r in q7['arms'].items():
        family = r['family']
        repair.append({'model': MODELS[0], 'arm': arm, 'complete_subjects': r['complete_subjects'],
            'subjects': q7['subjects'], 'correct_targets': r['complete_original_target_phrasings'], 'targets': q7['original_targets'],
            'correct_controls': sum(family[k]['correct_all_four_histories'] for k in ('F', 'Fdecoy', 'K')),
            'controls': q7['retained_controls'], 'source_replay': STUDIES[0]})
    for study, title in zip(STUDIES[1:], MODELS[1:]):
        summary = values[study]
        den = summary['denominators']
        for arm, r in summary['endpoints'].items():
            family = r['robust_by_family']
            repair.append({'model': title, 'arm': arm, 'complete_subjects': r['complete_subjects'],
                'subjects': den['subjects'], 'correct_targets': family['J'], 'targets': den['targets'],
                'correct_controls': sum(family[k] for k in ('F', 'Fdecoy', 'K')),
                'controls': den['controls'], 'source_replay': study})
    return {'schema': 'CURRENT_RESULT_TABLES_V1', 'origin': 'completed saved-observation reconstructions',
        'native': native, 'repair': repair,
        'contrasts': {'qwen7': q7['primary_and_descriptive_contrasts'],
            'qwen32': values['repair_qwen32']['primary_paired_contrasts'],
            'gemma27': values['repair_gemma27']['primary_paired_contrasts']},
        'historical_qwen7_absolute_gate': q7['both_seed_absolute_capability'],
        'qwen32_own_witness_repairs': values['repair_qwen32']['known_9_repairs'],
        'gemma27_own_witness_repairs': values['repair_gemma27']['additional']['own_witness_repairs'],
        'gemma27_qwen9_challenge': values['repair_gemma27']['qwen9_challenge_correct_with_controls'],
        'independent_sampling_unit': 'subject, not phrasing or repeated current view'}


def current_error_counts(records, expected_queries, *, qwen7=False):
    result = {arm: {'queries': 0, 'refusals': 0, 'wrong_accepted': 0} for arm in SEEDS}
    seen = set()
    for row in records:
        if row['arm'] not in SEEDS or row['history'] != 'H0' or row['phase'] != 'post':
            continue
        key = (row['arm'], row['case_id'], row['query_id'])
        if key in seen:
            raise ValueError('Duplicate current query in presentation source')
        seen.add(key)
        status = row['outcome']['status'] if qwen7 else row['status']
        if status not in ('VALID', 'REFUSAL') or type(row['correct']) is not bool:
            raise ValueError('Unsupported trained-query outcome; do not silently drop it')
        value = result[row['arm']]
        value['queries'] += 1
        value['refusals'] += status == 'REFUSAL'
        value['wrong_accepted'] += status == 'VALID' and not row['correct']
    if any(value['queries'] != expected_queries for value in result.values()):
        raise ValueError('Missing current query in presentation source')
    return result


def harm_counts(records, baselines, expected_queries, *, qwen7=False):
    result = {arm: {base: 0 for base in baselines} for arm in SEEDS}
    counts = collections.Counter()
    seen = set()
    for row in records:
        arm = row['candidate']
        if arm not in SEEDS:
            continue
        base = row['baseline']
        if base not in baselines:
            raise ValueError('Unexpected comparator in original harm ledger')
        key = (arm, base, row['case_id'], row['query_id'])
        if key in seen:
            raise ValueError('Duplicate comparison in harm ledger')
        seen.add(key)
        old = row['baseline_robust_correct' if qwen7 else 'baseline_correct']
        new = row['candidate_robust_correct' if qwen7 else 'candidate_correct']
        harmed = row['introduced_harm' if qwen7 else 'harmed']
        if any(type(value) is not bool for value in (old, new, harmed)) or harmed != (old and not new):
            raise ValueError('Harm flag disagrees with recorded correctness')
        counts[arm, base] += 1
        result[arm][base] += harmed
    if any(counts[arm, base] != expected_queries for arm in SEEDS for base in baselines):
        raise ValueError('Missing comparison in harm ledger')
    return result


def validate_paired_gain(contrast, seed_counts, baseline_count, subjects):
    if len(seed_counts) != 2 or subjects <= 0 or contrast['subjects'] != subjects:
        raise ValueError('Paired gain requires both seeds and the original subject denominator')
    expected = sum(count / subjects for count in seed_counts) / 2 - baseline_count / subjects
    if contrast['difference'] != expected:
        raise ValueError('Paired gain differs from both seeds and its declared baseline')


def add_presentation(result, values, output, sources):
    by_arm = {(row['model'], row['arm']): row for row in result['repair']}
    witness = []
    gains = []
    errors = []
    q7 = values[STUDIES[0]]
    for model, study in zip(MODELS, STUDIES):
        summary = values[study]
        for arm in SEEDS:
            controls = by_arm[model, arm]
            populations = []
            if study == STUDIES[0]:
                for name, label in [('all_original_94', 'Original Qwen diagnosis'),
                                    ('all_new_HISTORY_witnesses', 'New HISTORY wrapper')]:
                    population = summary['historical_repairs'][name]
                    populations.append((label, population['arms'][arm]['repaired'], population['denominator']))
            elif study == STUDIES[1]:
                populations.append(('Own native witnesses', summary['known_9_repairs'][arm], summary['denominators']['known_witnesses']))
            else:
                population = summary['additional']['own_witness_repairs'][arm]
                if population['status'] != 'MEASURED':
                    raise ValueError('Own-family witness repairs were not measured')
                populations.append(('Own native witnesses', population['numerator'], population['denominator']))
            for label, numerator, denominator in populations:
                if not 0 <= numerator <= denominator or denominator <= 0:
                    raise ValueError('Invalid witness-repair denominator')
                witness.append({'model': model, 'population': label, 'arm': arm,
                    'repaired': numerator, 'witness_queries': denominator,
                    'correct_controls': controls['correct_controls'], 'controls': controls['controls'],
                    'source_replay': study})
        contrasts = summary['primary_and_descriptive_contrasts'] if study == STUDIES[0] else summary['primary_paired_contrasts']
        n = by_arm[model, SEEDS[0]]['subjects']
        baseline_order = (NATIVE[model], *CONDITIONS[1:])
        for baseline in sorted(contrasts, key=baseline_order.index):
            contrast = contrasts[baseline]
            validate_paired_gain(contrast, [by_arm[model, arm]['complete_subjects'] for arm in SEEDS],
                                 by_arm[model, baseline]['complete_subjects'], n)
            gains.append({'model': model, 'baseline': baseline, **contrast,
                          'primary': contrast['primary'] if study == STUDIES[0] else True,
                          'seed_aggregation': 'within-subject mean of STATE_0 and STATE_1',
                          'source_replay': study})
        q7_mode = study == STUDIES[0]
        outcome_name = 'SCORED_LOGICAL.jsonl' if q7_mode else 'ALL_OUTCOMES.jsonl.gz'
        harm_name = 'PAIRED_FIXES_HARMS.jsonl' if q7_mode else 'FIXES_HARMS.jsonl.gz'
        outcomes = output / study / 'analysis' / outcome_name
        harms = output / study / 'analysis' / harm_name
        sources[study + '_outcomes'] = outcomes
        sources[study + '_harms'] = harms
        total = by_arm[model, SEEDS[0]]['targets'] + by_arm[model, SEEDS[0]]['controls']
        error = current_error_counts(rows(outcomes), total, qwen7=q7_mode)
        baselines = (NATIVE[model], 'REFLECT', 'CURRENT_TEXT', 'CURRENT_JSON') if q7_mode else (NATIVE[model], 'CURRENT_TEXT', 'CURRENT_JSON', 'STATE_BASE')
        harm = harm_counts(rows(harms), baselines, total, qwen7=q7_mode)
        for arm in SEEDS:
            if not q7_mode:
                endpoint = summary['endpoints'][arm]
                if error[arm]['refusals'] != endpoint['post_refusals_H0'] or error[arm]['wrong_accepted'] != endpoint['post_wrong_returns_H0']:
                    raise ValueError('Current-error counts disagree with reconstructed summary')
            errors.append({'model': model, 'arm': arm, **error[arm], 'harms_by_baseline': harm[arm],
                           'harm_unit': 'baseline-correct to candidate-incorrect robust query across four histories',
                           'source_replay': study})
    result['schema'] = 'CURRENT_RESULT_TABLES_V2'
    result['qwen7_historical_repairs'] = q7['historical_repairs']
    result['qwen7_history_wrapper_is_new'] = q7['history_wrapper_is_new']
    result['gemma27_diagnostic_units'] = values[STUDIES[2]]['additional']
    result['gemma27_qwen_challenge_denominator'] = values[STUDIES[2]]['denominators']['known_witnesses']
    population_order = {'Original Qwen diagnosis': 0, 'New HISTORY wrapper': 1, 'Own native witnesses': 0}
    witness.sort(key=lambda row: (MODELS.index(row['model']), population_order[row['population']], SEEDS.index(row['arm'])))
    result['presentation'] = {'condition_order': list(CONDITIONS), 'witness_controls': witness,
        'paired_gains': gains, 'trained_current_outcomes': errors,
        'error_count_unit': 'distinct current queries in H0; repeated equal views are not additional queries'}
    return result


def ordered_repair_rows(result):
    rows_by_key = {(row['model'], row['arm']): row for row in result['repair']}
    if len(rows_by_key) != len(result['repair']):
        raise ValueError('Duplicate measured endpoint')
    ordered = []
    for model in MODELS:
        for condition in CONDITIONS:
            arm = NATIVE[model] if condition == 'NATIVE' else condition
            measured = rows_by_key.get((model, arm))
            if measured is None:
                # Only REFLECT is absent in the two later frozen designs.
                if condition != 'REFLECT' or model == MODELS[0]:
                    raise ValueError('Missing required measured endpoint')
                ordered.append({'model': model, 'arm': arm, 'measurement_status': 'UNMEASURED'})
            else:
                ordered.append({**measured, 'measurement_status': 'MEASURED'})
    displayed = {(row['model'], row['arm']) for row in ordered if row['measurement_status'] == 'MEASURED'}
    if displayed != set(rows_by_key):
        raise ValueError('An unrecognized measured endpoint would be omitted')
    return ordered


def ratio(numerator, denominator):
    return str(numerator) + '/' + str(denominator)


def presentation_tables(result):
    native = [[r['measurement'], r['model'], ratio(r['qualified'], r['targets']), ratio(r['guards'], r['targets'])] for r in result['native']]
    repair = []
    for row in ordered_repair_rows(result):
        cells = ['unmeasured'] * 3 if row['measurement_status'] == 'UNMEASURED' else [
            ratio(row['complete_subjects'], row['subjects']), ratio(row['correct_targets'], row['targets']),
            ratio(row['correct_controls'], row['controls'])]
        repair.append([row['model'], row['arm'], *cells])
    witness = [[r['model'], r['population'], r['arm'], ratio(r['repaired'], r['witness_queries']),
                ratio(r['correct_controls'], r['controls'])] for r in result['presentation']['witness_controls']]
    gains = [[r['model'], r['baseline'], f"{100*r['difference']:.2f}",
              f"[{100*r['interval'][0]:.2f}, {100*r['interval'][1]:.2f}]",
              f"{100*r['confidence']:g}%", 'primary' if r['primary'] else 'descriptive']
             for r in result['presentation']['paired_gains']]
    errors = []
    for row in result['presentation']['trained_current_outcomes']:
        nonzero = '; '.join(base + ': ' + str(n) for base, n in row['harms_by_baseline'].items() if n)
        errors.append([row['model'], row['arm'], ratio(row['refusals'], row['queries']),
                       ratio(row['wrong_accepted'], row['queries']), nonzero or 'none in this ledger'])
    return {
        'native_results': (['Measurement', 'Model', 'Qualified targets', 'Guard-passing targets'], native),
        'repair_results': (['Model', 'Endpoint', 'Complete subjects', 'Robust targets', 'Robust controls'], repair),
        'witness_controls': (['Model', 'Witness population', 'Seed', 'Preserved repairs', 'Robust workload controls'], witness),
        'paired_gains': (['Model', 'Comparison baseline', 'Gain (pp)', 'Paired interval (pp)', 'Confidence', 'Status'], gains),
        'residual_errors': (['Model', 'Seed', 'Refusals', 'Wrong accepted answers', 'Nonzero robust-query harms by baseline'], errors),
    }


LEGEND = '''## Reading the conditions

- Native answering reads the update history and answers directly (`HISTORY`, `NATIVE32_SAVED`, or `NATIVE_GEMMA`).
- Reflection writes a note before seeing the question, then answers from the history and that note (`REFLECT`).
- Current-text answering reads only a text rendering of the selected current facts (`CURRENT_TEXT`).
- Current-JSON answering reads those selected facts in a structured representation (`CURRENT_JSON`).
- The base model selects an executable query instead of answering directly (`STATE_BASE`). It has no task-specific selector training.
- A trained selector chooses the query under training seed 0 (`STATE_0`) or seed 1 (`STATE_1`). Code executes the chosen query in both conditions.

Qwen7's `HISTORY` uses a new repair-study wrapper, not the original diagnosis wrapper.
Qwen32's `NATIVE32_SAVED` recodes saved SCALE outputs with the common whole-response codec.
Gemma's `NATIVE_GEMMA` uses its own model template and the repair answer instructions.
These labels do not imply a common wrapper across models.
'''


def markdown_table(table):
    header, records = table
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
        '|' + '|'.join('---' for _ in header) + '|',
        *['| ' + ' | '.join(str(value) for value in row) + ' |' for row in records]])


def render_markdown(result):
    tables = presentation_tables(result)
    q7 = result['qwen7_historical_repairs']
    gemma = result['gemma27_diagnostic_units']
    challenge = [['Gemma-2-27B', arm, ratio(result['gemma27_qwen9_challenge'][arm], result['gemma27_qwen_challenge_denominator'])] for arm in SEEDS]
    gate = 'passed' if result['historical_qwen7_absolute_gate'] else 'failed'
    sampling = {(r['subjects'], r['draws']) for r in result['presentation']['paired_gains']}
    if len(sampling) != 1:
        raise ValueError('Different comparison populations require explicit per-row presentation')
    subjects, draws = next(iter(sampling))
    return '\n\n'.join([
        LEGEND.strip(), '## Native history dependence', markdown_table(tables['native_results']),
        'Qualified failures retain the required current-fact checks. Original and fresh native measurements use different wrappers and remain separate.\nThe fresh 32B relevance interval includes zero.',
        '## Complete repair workload',
        'A complete subject has every target and control correct under all four histories.\nThe six equal current views are repeated conditions, not independent subjects.\nUnmeasured conditions are shown explicitly and are never assigned zero.',
        markdown_table(tables['repair_results']),
        '## Own-family repairs and required controls', markdown_table(tables['witness_controls']),
        'Witness denominators count target queries. The controls column covers the full workload, with each control correct across all four histories.\nPreserved witness repairs also satisfy their original control and canonical-reference checks.\n'
        f"Qwen7's original and new-wrapper witness sets overlap in {q7['reproduced_original_subset']['denominator']} queries; they are not pooled.\n"
        f"Gemma's {gemma['own_native_witnesses']} own-family witnesses belong to {gemma['own_native_witness_subjects']} subjects.",
        '### Separate Gemma challenge on Qwen witnesses',
        markdown_table((['Model', 'Seed', 'Preserved challenge repairs'], challenge)),
        f"This fixed Qwen challenge is not Gemma's own failure population. It overlaps that population in {gemma['Qwen9_intersection']} query.",
        '## Declared paired gains', markdown_table(tables['paired_gains']),
        'Each gain is the within-subject mean of both trained seeds minus the named baseline.\n'
        f'Every listed comparison pairs {subjects} subjects and uses {draws:,} bootstrap draws.\n'
        'Intervals resample whole subjects using each study\'s frozen procedure. Both seeds stay together within each subject.\nThe displayed gains and interval endpoints are percentage points. Full-precision values remain in the canonical JSON. Primary and descriptive comparisons remain distinct.',
        '## Errors and harms remain', markdown_table(tables['residual_errors']),
        'Refusals and wrong accepted answers count distinct current queries in H0. Repeated views are not added to those denominators.\nEach harm count compares robust-query correctness against the named baseline. The table shows nonzero counts; the canonical data also retain the zero counts.\nCounts against different baselines are separate comparisons, not independent failures.\n'
        f"Gemma separately records {gemma['read_harms_total']} harmed read-comparisons across repeated conditions and baselines.\n"
        f"The original Qwen7 strict joint capability gate remains {gate}.\n"
        'Qwen32\'s remaining current-question errors are refusals. Gemma\'s are wrong accepted query selections.\nThere is no combined zero-error or universal no-harm result.',
    ]) + '\n'


def tex_escape(value):
    return ''.join({'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
                    '#': r'\#', '{': r'\{', '}': r'\}'}.get(char, char) for char in str(value))


def render_latex(result):
    rendered = {}
    for name, (header, records) in presentation_tables(result).items():
        lines = ['% Generated from recorded-observation replay outputs.',
                 r'\begin{tabular}{' + 'l' * len(header) + '}', r'\hline',
                 ' & '.join(map(tex_escape, header)) + r' \\', r'\hline']
        lines += [' & '.join(map(tex_escape, row)) + r' \\' for row in records]
        lines += [r'\hline', r'\end{tabular}']
        if name == 'repair_results':
            legend_lines = [r'\noindent\textbf{Reading the conditions.}']
            for line in LEGEND.splitlines():
                if line and not line.startswith('## '):
                    legend_lines.append(tex_escape(line.removeprefix('- ').replace('`', '')) + r'\par')
            lines = [*legend_lines, r'\smallskip', *lines]
        if name == 'paired_gains':
            sampling = {(r['subjects'], r['draws']) for r in result['presentation']['paired_gains']}
            if len(sampling) != 1:
                raise ValueError('Different comparison populations need explicit LaTeX labels')
            subjects, draws = next(iter(sampling))
            lines += [r'\par\smallskip\noindent ' + tex_escape(
                f'Within-subject mean of both trained seeds minus the named baseline; {subjects} paired subjects and {draws:,} whole-subject bootstrap draws. Gains and intervals are percentage points.')]
        rendered[name + '.tex'] = '\n'.join(lines) + '\n'
    return rendered


def document_block(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('Missing or duplicate generated-table markers')
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip() + '\n'


def reconstruct(output, repository):
    output = Path(output)
    repository = Path(repository)
    sources = {'original_native': output / 'original_public_history/analysis/summary.json',
               'paired_native': output / 'paired_native_scale/analysis/SUMMARY.json',
               **{study: output / study / 'analysis/SUMMARY.json' for study in STUDIES}}
    values = {key: load(path) for key, path in sources.items()}
    result = add_presentation(legacy_result(values), values, output, sources)
    if result != load(repository / 'data/results/core_tables.json'):
        raise ValueError('Reconstructed canonical table differs from included expected values')
    markdown = render_markdown(result)
    latex = render_latex(result)
    if markdown != (repository / 'data/results/core_tables.md').read_text(encoding='utf8'):
        raise ValueError('Reconstructed readable table differs')
    if document_block((repository / 'docs/results/core_results.md').read_text(encoding='utf8')) != markdown:
        raise ValueError('Results documentation is not bound to reconstructed tables')
    for name, text in latex.items():
        if text != (repository / TABLE_DIRECTORY / name).read_text(encoding='utf8'):
            raise ValueError('Reconstructed LaTeX table differs: ' + name)
    destination = output / 'tables'
    destination.mkdir()
    products = {'core_tables.json': json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                'core_tables.md': markdown, **latex}
    for name, text in products.items():
        with (destination / name).open('x', encoding='utf8', newline='\n') as handle:
            handle.write(text)
    from .example import verify as verify_example
    example = verify_example(repository, output)
    return {'status': 'PASS', 'worked_example': example, 'native_rows': len(result['native']), 'repair_rows': len(result['repair']),
        'displayed_unmeasured_rows': sum(r['measurement_status'] == 'UNMEASURED' for r in ordered_repair_rows(result)),
        'compared_to': 'data/results/core_tables.json', 'generated_file': 'tables/core_tables.json',
        'generated_files': {name: hashlib.sha256(text.encode('utf8')).hexdigest() for name, text in products.items()},
        'inputs': {key: {'file': path.relative_to(output).as_posix(),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in sources.items()}}
