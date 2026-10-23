"""Reconstruct refresh population, harms and selected repeats from saved scores."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from .common import ROOT, load_json, new_output, sha256
from .release_inputs import verify
SOURCE = ROOT / 'reproducibility/question_blind_repair'
PRODUCTS = ('REPEAT_COMPARISON.json', 'SELECTED_RECOUNT.json', 'TARGETS.csv',
            'FAMILIES.csv', 'REVERSE_TARGETS.csv', 'DONOR_AGREEMENT.csv', 'WORKED_EXAMPLE.csv')
TIMING = sorted(['batch_seconds', 'qualification_seconds', 'preprocessing_seconds', 'compile_seconds'])

def verify_source(source=SOURCE):
    return len(verify(source, 'question_blind_repair')['files'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def yes(value):
    require(value in ('True', 'False'), 'Invalid saved Boolean')
    return value == 'True'


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
    return dict(status='PASS', scope='Public input hashes and saved-table reconciliation; the CLI additionally reconstructs saved-score statistics',
                result=result)


def compare_products(delivery, replay):
    result = {}
    for name in PRODUCTS:
        expected = sha256(Path(delivery) / name)
        require(sha256(Path(replay) / name) == expected, 'Scientific replay product differs: ' + name)
        result[name] = expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.source)
    destination = None
    if args.output is not None:
        if args.output.exists():
            raise ValueError('Use a fresh output directory')
        destination = new_output(args.output)
    from .refresh_scores import replay
    result['saved_score_replay'] = replay(args.source, result['result'], destination)
    result['scope'] = 'Saved-score gate/statistic reconstruction and retained-interval reconciliation; no new model measurements'
    if destination is not None:
        (destination / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
