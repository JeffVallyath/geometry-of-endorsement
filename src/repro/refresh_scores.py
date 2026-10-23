"""New portability wrapper over the original refresh score decoder and gates.

Only public score/token/alias projections are replayed. Native cache receipts and
the original raw journal are not reconstructed from those projections.
"""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import runpy

from .common import load_json, safe_path
from .release_inputs import payload
from .question_blind_repair import require, rows, yes


def records(folder, record):
    binding = dict(record, encoding='gzip')
    decoded = payload(folder, binding)
    values = [json.loads(line) for line in decoded.splitlines()]
    require(len(values) == record['records'], 'Recovered score record count differs')
    return values


def select_panel(panel, reverse):
    """Apply the original forward/reverse filter, without discarded custody fields."""
    require(set(panel) == {'root_id', 'seed', 'actor', 'physical', 'tokens', 'aliases'},
            'Unexpected score projection fields')
    allowed = lambda key: reverse or not key.startswith(('RB/', 'RC/'))
    return dict(panel,
                physical=sorted((r for r in panel['physical'] if allowed(r['context'])),
                                key=lambda r: (r['context'], r['query_id'])),
                tokens={k: v for k, v in panel['tokens'].items() if allowed(k)},
                aliases={k: v for k, v in panel['aliases'].items() if allowed(k)})


def validate_panel(panel, questions, reader):
    """Check delivered score rows; this is not a native-cache or raw-journal audit."""
    question_ids = {q['query_id'] for q in questions}
    require(len(question_ids) == len(questions) == 34, 'Frozen question inventory differs')
    inventory = {}
    for row in panel['physical']:
        reader['decode'](row)
        require(row['master_unchanged'] and row['batch_size'] == 1 and row['physical_slot'] == 0,
                'Single-query score isolation binding differs')
        context, query = row['context'], row['query_id']
        seen = inventory.setdefault(context, {})
        require(query not in seen, 'Duplicate score row')
        seen[query] = row['physical_batch']
    for context, seen in inventory.items():
        require(set(seen) == question_ids and sorted(seen.values()) == list(range(34)),
                'Score/query/batch inventory differs')
        require(context in panel['tokens'] and set(panel['tokens'][context]) == question_ids,
                'Score/token inventory differs')
    require(set(inventory) == set(panel['tokens']), 'Unexpected token context')
    return len(panel['physical'])


def accumulate_primary(computed, rid, seed, target_table, totals, treated_cases, repaired_cases):
    """Original primary recount arithmetic, separated from its private archive I/O."""
    for row in computed['targets']:
        recorded = target_table[rid, seed, row['query_id']]
        require(row['W'] == yes(recorded['both_matched']), 'Original baseline classification differs')
        for arm in ('B', 'C'):
            require(row[arm + '_repair'] == yes(recorded[arm + '_repair']), 'Original repair classification differs')
        totals['baseline_W'] += row['W']
        totals['B_repairs'] += row['B_repair']
        totals['C_repairs'] += row['C_repair']
        totals['treated_qualifying_witnesses'] += row['B_qualifying_witness']
        totals['baseline_witnesses_retained'] += row['W'] and row['B_qualifying_witness']
        totals['new_qualifying_witnesses'] += not row['W'] and row['B_qualifying_witness']
        totals['baseline_ceased_qualifying_without_repair'] += row['W'] and not row['B_qualifying_witness'] and not row['B_repair']
        if row['B_qualifying_witness']:
            treated_cases.add(rid)
        if row['B_repair']:
            repaired_cases.add(rid)
    for row in computed['families']:
        if row['family'] == 'joint':
            totals[row['arm'] + '_joint_errors'] += row['scheduled'] - row['correct']
        if row['arm'] == 'B' and row['family'] in ('joint', 'direct'):
            totals['B_' + row['family'] + '_damage'] += row['harm']
            totals['B_' + row['family'] + '_corrections'] += row['benefit']


def selected_summary(all_rows):
    """Original selected-case summary; repeats never enter the primary population."""
    summary = {}
    for dataset in ('archived', 'fresh'):
        target = [r for r in all_rows['targets'] if r['dataset'] == dataset]
        families = [r for r in all_rows['families'] if r['dataset'] == dataset]
        reverse = [r for r in all_rows['reverse_targets'] if r['dataset'] == dataset]
        summary[dataset] = dict(selected_cases=6, case_seed_blocks=12, target_rows=len(target),
            baseline_W=sum(r['W'] for r in target), B_repairs=sum(r['B_repair'] for r in target),
            C_repairs=sum(r['C_repair'] for r in target), B_remaining_qualifying_witnesses=sum(r['B_qualifying_witness'] for r in target),
            eligible_RB_targets=sum(r['eligible_history_dependent_error'] for r in reverse if r['arm'] == 'RB'),
            eligible_RC_targets=sum(r['eligible_history_dependent_error'] for r in reverse if r['arm'] == 'RC'),
            response_counts=[dict(family=fam, arm=arm, **{key: sum(r[key] for r in families if r['family'] == fam and r['arm'] == arm)
                for key in ('scheduled', 'correct', 'baseline_correct', 'harm', 'benefit', 'invalid')}) for fam in ('direct', 'joint', 'opposes', 'all34') for arm in ('A', 'B', 'C', 'D')])
        require(len({r['root_id'] for r in target}) == 6 and
                len({(r['root_id'], r['seed']) for r in target}) == 12, 'Selected population changed')
    return summary


def table_bytes(values):
    """Original table serialization, using an in-memory stream for optional output."""
    columns = list(dict.fromkeys(k for row in values for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in values:
        writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode('utf-8')


def replay(source, expected, output=None):
    """Reconstruct primary gates and selected-repeat statistics from saved scores."""
    source = Path(source)
    folder = source / 'scores'
    manifest = load_json(folder / 'DATA.json')
    require(manifest['schema'] == 'ANONYMOUS_REFRESH_SCORES_V1', 'Unknown score schema')
    bindings = {r['path']: r for r in manifest['files']}
    require(len(bindings) == len(manifest['files']), 'Duplicate recovered score file')
    require({p.name for p in folder.iterdir()} == set(bindings) | {'DATA.json'},
            'Missing or undeclared recovered score file')
    for name in bindings:
        safe_path(name, folder)
    designs = records(folder, bindings[manifest['designs']])
    design_by_id = {r['root']['root_id']: r for r in designs}
    require(len(design_by_id) == len(designs) == manifest['primary_cases'] == 256, 'Design population differs')
    selection_record = bindings[manifest['repeat_selection']]
    selection = json.loads(payload(folder, dict(selection_record, encoding='verbatim',
                                               decoded_bytes=selection_record['bytes'],
                                               decoded_sha256=selection_record['sha256'])))
    reader = runpy.run_path(str(source / 'supplied/verification/tools/replay.py'))
    target_table = {(r['root_id'], int(r['seed']), r['query_id']): r for r in
                    rows(source / 'supplied/primary/replay/ALL_SEED_FORMAT_TARGETS.csv')}
    require(len(target_table) == 9216 and {k[0] for k in target_table} == set(design_by_id),
            'Primary target coverage differs')
    totals, treated_cases, repaired_cases, seen = Counter(), set(), set(), set()
    selected_old = {}
    selected_roots = set(selection['roots']) | {selection['anchor']}
    physical_rows = 0
    root_order = sorted(design_by_id)
    for part in manifest['primary']:
        panels = records(folder, bindings[part['path']])
        require(len(panels) == part['cases'] == 32 and
                [p['root_id'] for p in panels] == root_order[part['first_case']:part['first_case'] + 32],
                'Primary chunk membership differs')
        for panel in panels:
            rid, seed = panel['root_id'], panel['seed']
            require(seed == part['seed'] and seed in (0, 1) and (rid, seed) not in seen,
                    'Duplicate or unexpected primary case/seed')
            seen.add((rid, seed))
            design = design_by_id[rid]
            physical_rows += validate_panel(panel, design['questions'], reader)
            forward = select_panel(panel, False)
            totals['raw_rows'] += len(forward['physical'])
            computed = reader['metrics'](design['root'], design['questions'], design['plan'], forward, False)
            accumulate_primary(computed, rid, seed, target_table, totals, treated_cases, repaired_cases)
            if rid in selected_roots:
                selected_old[rid, seed] = panel
    require(seen == {(rid, seed) for rid in design_by_id for seed in (0, 1)} and
            len(seen) == manifest['primary_case_seed_blocks'] == 512, 'Primary population incomplete')
    require(physical_rows == 470016 and totals['raw_rows'] == 435200, 'Physical score coverage differs')
    totals.update(cases=len(designs), treated_witness_cases=len(treated_cases), repaired_designs=len(repaired_cases))
    for key, value in totals.items():
        if key in expected['primary']:
            require(value == expected['primary'][key], 'Full primary recount differs: ' + key)

    all_rows = {name: [] for name in ('targets', 'families', 'reverse_targets', 'donor_agreement')}
    comparisons, repeat_seen = [], set()
    repeat = records(folder, bindings[manifest['selected_repeat']])
    expected_repeat_order = [(phase, rid, seed) for phase, roots in
                             [('anchor', [selection['anchor']]), ('repeat', selection['roots'])]
                             for seed in (0, 1) for rid in roots]
    require([(r['phase'], r['panel']['root_id'], r['panel']['seed']) for r in repeat] == expected_repeat_order,
            'Fixed repeat roster or order differs')
    for item in repeat:
        phase, new = item['phase'], item['panel']
        rid, seed = new['root_id'], new['seed']
        reverse = phase == 'repeat' and rid in selection['reverse_roots']
        require(item['reverse'] is reverse and (phase, rid, seed) not in repeat_seen,
                'Repeat reverse scope or multiplicity differs')
        repeat_seen.add((phase, rid, seed))
        design = design_by_id[rid]
        validate_panel(new, design['questions'], reader)
        old, new = select_panel(selected_old[rid, seed], reverse), select_panel(new, reverse)
        comparison = reader['compare'](old, new)
        require(comparison['score_binding_exact_equal'] and comparison['tolerance_pass'],
                'Retained repeat score/token/alias fields differ')
        comparison.update(phase=phase, root_id=rid, seed=seed, rows=len(new['physical']))
        comparisons.append(comparison)
        if phase == 'repeat':
            for label, panel in [('archived', old), ('fresh', new)]:
                computed = reader['metrics'](design['root'], design['questions'], design['plan'], panel, reverse)
                for kind, values in computed.items():
                    all_rows[kind].extend(dict(dataset=label, **row) for row in values)
    require(len(comparisons) == manifest['selected_repeat_blocks'] == 14 and
            sum(r['rows'] for r in comparisons) == 14076, 'Repeat score coverage differs')
    summary = selected_summary(all_rows)
    require(summary == load_json(source / 'supplied/verification/SELECTED_RECOUNT.json'),
            'Reconstructed selected-case statistics differ')
    products = []
    for kind, values in all_rows.items():
        name = kind.upper() + '.csv'
        raw = table_bytes(values)
        require(raw == (source / 'supplied/verification' / name).read_bytes(),
                'Reconstructed selected table differs: ' + name)
        if output is not None:
            with (Path(output) / name).open('xb') as stream:
                stream.write(raw)
        products.append(dict(name=name, rows=len(values), byte_exact=True))
    result = dict(status='PASS', primary=dict(totals), primary_case_seed_blocks=len(seen),
                  physical_rows_including_original_reverse=physical_rows,
                  selected_repeat_blocks=len(comparisons), selected_tables=products,
                  repeat_comparisons=comparisons,
                  projected_score_token_alias_bindings_exact=True,
                  native_cache_and_raw_journal_reverified=False,
                  original_intervals='retained and reconciled, not regenerated',
                  new_independent_observations=0, new_model_measurements=False)
    if output is not None:
        for name, value in [('PRIMARY_RECOUNT.json', dict(totals)), ('SELECTED_RECOUNT.json', summary),
                            ('REPEAT_SCORE_COMPARISON.json', comparisons)]:
            with (Path(output) / name).open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return result
