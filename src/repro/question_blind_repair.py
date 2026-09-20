"""Check the repair closeout; optionally replay both external saved-score archives.

The delivered independent, standard-library replay is reused unchanged. This
adapter verifies provenance, checks its products, and keeps selected repeats
separate from the prospective population. It never executes a neural model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import runpy
import stat
import subprocess
import sys
import zipfile

from .common import ROOT, load_json, new_output, safe_path, sha256

SOURCE = ROOT / 'reproducibility/question_blind_repair'
ARCHIVES = {
    'primary': 'f95a0a20daca0bbc51fa8942b1d2ce03c275e9fca8d72abebd094454c10c7608',
    'verification': '590772eb9f4148a714bd7c22ce769925beb9890bdab0f81654a75934427ed188',
    'handoff': '645483a5353e9f72598cb4c7c86dc6b2e3f23cc59fce0cc79578458b3b6376d8',
}
CAPSULES = {
    'primary': '22bc13ee9b3cc7369741339d4f8f2f74b1577db6bf513e49e1dcb5cecd14da49',
    'verification': 'c31b5602355fe4cbbc7a79450d8847c7e91f2319501c7c9279b0bb0468e15afb',
}
PRODUCTS = ('REPEAT_COMPARISON.json', 'SELECTED_RECOUNT.json', 'TARGETS.csv',
            'FAMILIES.csv', 'REVERSE_TARGETS.csv', 'DONOR_AGREEMENT.csv', 'WORKED_EXAMPLE.csv')
TIMING = sorted(['batch_seconds', 'qualification_seconds', 'preprocessing_seconds', 'compile_seconds'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def yes(value):
    require(value in ('True', 'False'), 'Invalid saved Boolean')
    return value == 'True'


def verify_source(source=SOURCE):
    source = Path(source)
    manifest = load_json(source / 'SOURCE.json')
    require(manifest['schema'] == 'QUESTION_BLIND_REPAIR_COMPACT_V1', 'Wrong compact schema')
    require({k: v['sha256'] for k, v in manifest['archives'].items()} == ARCHIVES,
            'External archive bindings changed')
    originals = {}
    for key, binding in manifest['archives'].items():
        path = source / 'supplied' / key / 'MANIFEST.json'
        require(sha256(path) == binding['manifest_sha256'], 'Frozen manifest changed: ' + key)
        originals[key] = load_json(path)
    actual = {p.relative_to(source).as_posix() for p in (source / 'supplied').rglob('*') if p.is_file()}
    require(actual == set(manifest['files']), 'Missing or undeclared compact member')
    for name, record in manifest['files'].items():
        path = safe_path(name, source)
        require(path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'],
                'Changed supplied evidence: ' + name)
        original = originals[record['archive']]
        if record['archive_member'] != 'MANIFEST.json':
            require(original.get('files', original)[record['archive_member']] ==
                    {k: record[k] for k in ('bytes', 'sha256')}, 'Source manifest disagreement: ' + name)
    bindings = load_json(source / 'supplied/handoff/SOURCE_BINDINGS.json')
    require(bindings['original_results']['sha256'] == ARCHIVES['primary'] and
            bindings['successful_case_verification']['sha256'] == ARCHIVES['verification'], 'Handoff archive binding')
    require(bindings['original_scientific_capsule_sha256'] == CAPSULES['primary'] and
            bindings['extension_scientific_capsule_sha256'] == CAPSULES['verification'], 'Handoff capsule binding')
    return len(actual)


def projection(source=SOURCE):
    """Reconcile compact tables and frozen intervals, without inventing new ones."""
    source = Path(source)
    verify_source(source)
    primary_dir = source / 'supplied/primary'
    repeat_dir = source / 'supplied/verification'
    summary = load_json(primary_dir / 'replay/SUMMARY.json')
    ledger = load_json(repeat_dir / 'CLAIM_LEDGER.json')
    primary = dict(ledger['original_prospective_result'])
    target = rows(primary_dir / 'replay/ALL_SEED_FORMAT_TARGETS.csv')
    require(len(target) == len({(r['root_id'], r['seed'], r['query_id']) for r in target}) == 9216,
            'Primary target inventory')
    require(len({r['root_id'] for r in target}) == primary['cases'] == summary['cases'] == 256,
            'Prospective population changed')
    require(sum(yes(r['both_matched']) for r in target) == primary['baseline_W'] == summary['W_rows'] == 83,
            'Baseline cohort changed')
    for arm, expected in (('B', 36), ('C', 2)):
        require(sum(yes(r[arm + '_repair']) for r in target) == primary[arm + '_repairs'] == expected,
                'Primary repair count changed')
    repaired_cases = len({r['root_id'] for r in target if yes(r['B_repair'])})
    require(repaired_cases == summary['B_repaired_distinct_cases'] == 28, 'Repaired designs changed')
    require(primary['B_repair_rate'] == summary['B_repaired_fraction'] == 36 / 83, 'Repair fraction changed')
    require(primary['strong_repair'] is summary['strong_repair'] is False and
            summary['thresholds']['repaired_at_least_half'] is False, 'Strong-repair failure changed')
    totals = Counter()
    family = rows(primary_dir / 'tables/ALL_SEED_FORMAT_FAMILIES.csv')
    for row in family:
        for name in ('scheduled', 'correct', 'benefit', 'harm'):
            totals[row['arm'], row['family'], name] += int(row[name])
    for arm in ('A', 'B', 'D'):
        require(totals[arm, 'joint', 'scheduled'] == primary['joint_response_denominator'] == 73728,
                'Joint denominator changed')
        require(totals[arm, 'joint', 'scheduled'] - totals[arm, 'joint', 'correct'] == primary[arm + '_joint_errors'],
                'Joint error accounting changed')
    for kind in ('joint', 'direct'):
        require(totals['B', kind, 'benefit'] == primary['B_' + kind + '_corrections'] and
                totals['B', kind, 'harm'] == primary['B_' + kind + '_damage'], 'Benefit/damage accounting changed')
    endpoints = {r['endpoint']: {'estimate': float(r['estimate']),
                 'interval9875': [float(r['interval9875_low']), float(r['interval9875_high'])]}
                 for r in rows(primary_dir / 'tables/FOUR_ENDPOINTS.csv')}
    require(endpoints == summary['family'], 'Original four-endpoint uncertainty family changed')
    require(primary['baseline_W'] == primary['B_repairs'] + primary['baseline_witnesses_retained'] +
            primary['baseline_ceased_qualifying_without_repair'], 'Baseline cohort partition')
    require(primary['treated_qualifying_witnesses'] == primary['baseline_witnesses_retained'] +
            primary['new_qualifying_witnesses'] == 75, 'Treated witness total is not the repair fraction')
    primary.update(repaired_designs=repaired_cases, target_observations=len(target),
                   strong_repair_threshold=0.5, uncertainty_family=endpoints,
                   joint_error_reduction=(primary['A_joint_errors'] - primary['B_joint_errors']) / 73728)

    comparison = load_json(repeat_dir / 'REPEAT_COMPARISON.json')
    recount = load_json(repeat_dir / 'SELECTED_RECOUNT.json')
    require(recount['fresh'] == recount['archived'], 'Selected recount differs')
    require(comparison['post_hoc_selection'] is True and comparison['primary_bootstrap_unchanged'] is True and
            comparison['new_independent_observations'] == 0, 'Selected repeats pooled into primary study')
    require(comparison['blocks'] == comparison['score_binding_exact_blocks'] == 14 and
            comparison['physical_rows'] == 14076 and comparison['max_score_delta'] == 0, 'Repeat coverage or scores differ')
    for block in comparison['comparisons']:
        for evidence in (block, block['seal']):
            require(evidence['excluded_wall_clock_fields'] == TIMING and evidence['tolerance'] == 1e-6,
                    'Frozen comparison rule changed')
            require(evidence['score_binding_exact_equal'] and not evidence['discrepancies'] and not evidence['failures'],
                    'Retained binding discrepancy')
    fresh = [r for r in rows(repeat_dir / 'TARGETS.csv') if r['dataset'] == 'fresh']
    reverse = [r for r in rows(repeat_dir / 'REVERSE_TARGETS.csv') if r['dataset'] == 'fresh']
    require(len(fresh) == 216 and len({(r['root_id'], r['seed'], r['query_id']) for r in fresh}) == 216,
            'Selected target inventory')
    counted = dict(selected_cases=len({r['root_id'] for r in fresh}),
                   case_seed_blocks=len({(r['root_id'], r['seed']) for r in fresh}),
                   baseline_W=sum(yes(r['W']) for r in fresh),
                   B_repairs=sum(yes(r['B_repair']) for r in fresh),
                   C_repairs=sum(yes(r['C_repair']) for r in fresh))
    for arm in ('RB', 'RC'):
        counted['eligible_' + arm + '_targets'] = sum(yes(r['eligible_history_dependent_error']) for r in reverse if r['arm'] == arm)
    require(all(recount['fresh'][k] == v for k, v in counted.items()), 'Selected table accounting differs')
    require(counted == dict(selected_cases=6, case_seed_blocks=12, baseline_W=9, B_repairs=8,
                           C_repairs=0, eligible_RB_targets=3, eligible_RC_targets=0), 'Selected scope changed')
    example = {r['draw']: r for r in fresh if r['root_id'] == 'QBR1-FINAL-0127' and
               r['seed'] == '1' and r['query_id'].endswith('-02')}
    require(yes(example['0']['B_repair']) and not yes(example['1']['B_repair']) and
            not yes(example['1']['B_target_all_correct']), 'Answer-format boundary changed')
    worked = rows(repeat_dir / 'WORKED_EXAMPLE.csv')
    same = next(r for r in worked if r['query_id'].endswith('-02'))
    columns = ['A_history1', 'A_history2', 'B16_23_history2', 'C24_31_history2',
               'D_reference', 'RB16_23_history2', 'RC24_31_history2']
    require([same[k] for k in columns] == ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'], 'Worked SAME answers changed')
    require(len(worked) == 5 and all(r[k] == 'Yes' for r in worked if r is not same for k in columns),
            'Independent direct/other logical answers changed')
    donors = [r for r in rows(repeat_dir / 'DONOR_AGREEMENT.csv') if r['dataset'] == 'fresh' and
              r['root_id'] == 'QBR1-FINAL-0146' and r['seed'] == '0' and r['arm'] == 'RB' and
              r['query_id'] == 'QBR1-FINAL-0146-d0-extra1' and not yes(r['reverse_correct'])]
    require({int(r['origin']) for r in donors} == {6, 7} and all(yes(r['donor_correct']) for r in donors),
            'Correct-donor reverse-error boundary changed')
    return dict(primary=primary,
                post_hoc=dict(status='POST_HOC_SUCCESSFUL_CASE_REPRODUCIBILITY', **counted,
                              anchor_designs=1, seeds=2, blocks=14, reads=14076, new_independent_observations=0,
                              all_retained_score_bindings_exact=True, max_score_delta=0,
                              score_tolerance=1e-6, excluded_timing_fields=TIMING,
                              delivery_status_is_strong_repair=False),
                example=dict(yes_no_full_repair=True, ab_full_repair=False,
                             independent_controls_correct=True, same_answers=[same[k] for k in columns],
                             correct_donor_reverse_error_origins=sorted(int(r['origin']) for r in donors)),
                method=dict(target_layers=[16, 23], comparison_layers=[24, 31], histories=8,
                            complete_current_record_reference_required=True, before_questions=True,
                            keys_other_values_weights_and_recipient_prompt_fixed=True))


def check(source=SOURCE):
    result = projection(source)
    require(result == load_json(Path(source) / 'CLAIMS.json'), 'Generated claim projection differs')
    return dict(status='PASS', scope='Compact source hashes and table/claim consistency; raw replay needs both external archives',
                result=result)


def verify_zip(path, expected, destination=None, manifest_name='MANIFEST.json'):
    """Closed-manifest validation before any extraction; no source mutation."""
    path = Path(path)
    require(sha256(path) == expected, 'Outer archive hash mismatch: ' + path.name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len({n.casefold() for n in names}), 'Duplicate archive member')
        manifest = json.loads(archive.read(manifest_name))
        declared = manifest.get('files', manifest)
        require(set(names) == set(declared) | {manifest_name}, 'Missing or undeclared archive member')
        for info in archive.infolist():
            safe_path(info.filename, Path.cwd())
            require(not stat.S_ISLNK(info.external_attr >> 16), 'Symlink archive member')
            if info.filename != manifest_name:
                expected_member = declared[info.filename]
                with archive.open(info) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                digest = expected_member if isinstance(expected_member, str) else expected_member['sha256']
                require(actual == digest, 'Archive member hash mismatch: ' + info.filename)
                if isinstance(expected_member, dict):
                    require(info.file_size == expected_member['bytes'], 'Archive member size mismatch')
        if destination is not None:
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=False)
            archive.extractall(destination)
    return manifest


def compare_products(delivery, replay):
    result = {}
    for name in PRODUCTS:
        expected = sha256(Path(delivery) / name)
        require(sha256(Path(replay) / name) == expected, 'Scientific replay product differs: ' + name)
        result[name] = expected
    return result


def primary_recount(reader, capsule, primary_roots, source):
    """Recheck already-reported full-sample counts with the delivered fixed gates.

    Intervals are copied from the primary archive, never refit using repeats.
    """
    totals = Counter()
    treated_cases = set()
    repaired_cases = set()
    target_table = {(r['root_id'], int(r['seed']), r['query_id']): r for r in
                    rows(Path(source) / 'supplied/primary/replay/ALL_SEED_FORMAT_TARGETS.csv')}
    roots = sorted({key[0] for key in target_table})
    for rid in roots:
        design = capsule / 'original/generated' / rid
        inputs = [load_json(design / name) for name in ('root.json', 'questions.json', 'PLAN.json')]
        for seed in (0, 1):
            panel, _, audit = reader['read_case'](primary_roots[seed] / 'scientific/final' / rid, False)
            computed = reader['metrics'](*inputs, panel, False)
            totals['raw_rows'] += audit['raw_rows']
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
    totals.update(cases=len(roots), treated_witness_cases=len(treated_cases), repaired_designs=len(repaired_cases))
    expected = projection(source)['primary']
    for key, value in totals.items():
        if key in expected:
            require(value == expected[key], 'Full primary recount differs: ' + key)
    return dict(totals)


def full_replay(verification_archive, original_archive, output, source=SOURCE):
    source, output = Path(source), Path(output).resolve()
    compact = check(source)
    require(not output.exists(), 'Use a fresh output directory; preserved outputs are never overwritten')
    new_output(output)
    receipt = {'scope': 'Offline saved-evidence verification; no model execution or live-provider attestation'}
    delivery, original = output / 'verification', output / 'primary'
    vm = verify_zip(verification_archive, ARCHIVES['verification'], delivery)
    pm = verify_zip(original_archive, ARCHIVES['primary'], original)
    require(vm['extension_capsule_sha256'] == CAPSULES['verification'] and
            pm['scientific_capsule_sha256'] == CAPSULES['primary'], 'Outer capsule binding')
    original_capsule = verify_zip(original / 'capsules/qbr1-r1.zip', CAPSULES['primary'], manifest_name='RUN_CAPSULE.json')
    verify_zip(delivery / 'capsules/qbrv-r2.zip', CAPSULES['verification'], manifest_name='RUN_CAPSULE.json')
    commands = [
        ['tools/run_capsule.py', 'extract', '--archive', 'capsules/qbrv-r2.zip', '--out', str(output / 'capsule'),
         '--sha256', CAPSULES['verification']],
        ['tools/artifact_export.py', '--verify', '--out', 'raw/final_export.zip', '--extract', str(output / 'fresh'),
         '--capsule-hash', CAPSULES['verification']],
        ['tools/replay.py', '--capsule', str(output / 'capsule'), '--seed0', 'reference/seed0/scientific',
         '--seed1', 'reference/seed1/scientific', '--fresh', str(output / 'fresh/scientific'), '--output', str(output / 'replay')],
    ]
    command_receipts = []
    for command in commands:
        completed = subprocess.run([sys.executable, '-B', *command], cwd=delivery, capture_output=True, text=True, check=True)
        command_receipts.append(dict(argv=command, returncode=completed.returncode))
    receipt['commands'] = command_receipts
    receipt['products'] = compare_products(delivery, output / 'replay')
    fresh_manifest = load_json(output / 'fresh/MANIFEST.json')
    terminal = load_json(output / 'fresh/scientific/TERMINAL.json')
    require(fresh_manifest['quiescent_snapshot'] and fresh_manifest['scientific_terminal_present'] and
            terminal['status'] == 'GPU_COMPLETE_ANALYSIS_LOCAL' and terminal['scored_reads'] == 14076, 'Extension terminal contract')
    with zipfile.ZipFile(original / 'capsules/qbr1-r1.zip') as archive:
        for name in archive.namelist():
            require(archive.read(name) == (output / 'capsule/original' / name).read_bytes(), 'Original capsule changed in extension: ' + name)
    primary_roots, reference_members = {}, 0
    for seed in (0, 1):
        member = f'raw/seed{seed}_final_export.zip'
        destination = output / f'primary-seed{seed}'
        nested = verify_zip(original / member, pm['files'][member]['sha256'], destination)
        require(nested['capsule_sha256'] == CAPSULES['primary'] and nested['quiescent_snapshot'] and
                nested['scientific_terminal_present'], 'Primary terminal/capsule contract')
        primary_roots[seed] = destination
        for path in (delivery / f'reference/seed{seed}').rglob('*'):
            if path.is_file():
                rel = path.relative_to(delivery / f'reference/seed{seed}')
                require(path.read_bytes() == (destination / rel).read_bytes(), 'Original reference member differs: ' + str(rel))
                reference_members += 1
    reader = runpy.run_path(str(delivery / 'tools/replay.py'))
    require(sorted(reader['TIMING']) == TIMING, 'Documented timing exclusions differ')
    receipt['primary_recount'] = primary_recount(reader, output / 'capsule', primary_roots, source)
    receipt.update(status='PASS', scientific_products_exact=len(PRODUCTS), original_reference_members_exact=reference_members,
                   original_capsule_members_exact=len(original_capsule['files']) + 1,
                   primary_outer_members=len(pm['files']), verification_outer_members=len(vm['files']),
                   original_uncertainty_family=compact['result']['primary']['uncertainty_family'],
                   archive_sha256=ARCHIVES, capsule_sha256=CAPSULES,
                   repeat_comparison={k: v for k, v in load_json(output / 'replay/REPEAT_COMPARISON.json').items() if k != 'comparisons'})
    with (output / 'VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--verification-archive', type=Path)
    parser.add_argument('--original-archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    supplied = [args.verification_archive, args.original_archive, args.output]
    if any(p is not None for p in supplied):
        if not all(p is not None for p in supplied):
            parser.error('Full replay requires both archives and a fresh --output')
        result = full_replay(args.verification_archive, args.original_archive, args.output, args.source)
    else:
        result = check(args.source)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
