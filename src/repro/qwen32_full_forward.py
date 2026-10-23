"""Portable Qwen3-32B saved-score replay. Never loads a model or contacts a service."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from .common import ROOT, load_json, new_output, safe_path, sha256
from .fresh_history import equivalent, float_differences, run_offline
from .release_inputs import payload as _payload, verify as verify_inputs, restore as restore_inputs

SOURCE = ROOT / 'reproducibility/qwen32_full_forward'
METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')


def require(condition, message):
    if not condition:
        raise ValueError(message)





def verify(source=SOURCE):
    return verify_inputs(Path(source), 'qwen32')


def restore(source, destination, binding):
    return restore_inputs(source, binding, destination)


def claims_from_outputs(summary, cases, independent, example):
    """A small source-derived claim map, with cases as the resampling unit."""
    methods = {}
    for method in METHODS:
        primary = summary[method]['primary']
        rows = [row for case in cases.values() for row in case[method]['semantic']]
        certified = lambda row: row['all_direct'] and row['reference'] and row['all_logical']
        examples = [row for row in independent['single_format_full_control_examples'] if row['method'] == method]
        methods[method] = dict(
            cases=len(cases), primary=primary['W'], denominator=1152,
            designs_with_primary=sum(bool(case[method]['metrics']['W']) for case in cases.values()),
            primary_rate=primary['witness_rate'], interval99375=primary['primary_family8_interval99375'],
            coverage=primary['E_coverage']['estimate'], direct_accuracy=primary['direct_accuracy']['estimate'],
            joint_accuracy=primary['joint_accuracy']['estimate'],
            strongest=summary[method]['semantic_counts']['strongest_pairs'], semantic_denominator=len(rows),
            fully_certified_opportunities=sum(certified(row) for row in rows),
            fully_certified_designs=sum(any(certified(row) for row in case[method]['semantic']) for case in cases.values()),
            matched_single_format=len(examples), single_format_designs=sorted({row['root'] for row in examples}),
        )
    return dict(model='Qwen/Qwen3-32B', revision='9216db5781bf21249d130ec9da846c4624c16137',
                methods=methods, original_designs_newly_measured=True, newly_sampled_designs=False,
                repeat_rows=sum(independent['repeat_rows'].values()),
                repeat_max_logp_delta=independent['repeat_max_logp_delta'],
                single_format_distinct_designs=len({row['root'] for row in independent['single_format_full_control_examples']}),
                strongest_scope='same matched history pair and direction in both answer formats',
                worked_example=example, scaling_comparison_supported=False, causal_repair_measured=False,
                moral_decision_effect_measured=False, saved_score_replay_not_neural_replication=True)


def validate_receipts(replay, independent):
    require(replay['status'] == independent['status'] == 'PASS', 'CPU checker did not pass')
    require(replay['raw_final_rows'] == independent['raw_final_checked'] == 36992, 'Incomplete final records')
    require(replay['classification_fields_checked'] == 27136, 'Incomplete classification check')
    repeats = replay['full_repeat_reconstruction']
    require(len(repeats) == 4 and all(row['classification_identical'] for row in repeats), 'Full-repeat classification drift')
    require(independent['repeat_rows'] == {'repeat0': 1088, 'repeat1': 1088, 'repeat_full': 2312}, 'Incomplete repeat records')
    require(not independent['repeat_discrepancies'] and independent['repeat_max_logp_delta'] == 0,
            'Saved repeat score/token discrepancy')


def check_projection(source=SOURCE, *, integrity=True):
    """Bind the reader projection to frozen case outputs; full raw replay is export()."""
    source = Path(source)
    binding = verify(source) if integrity else load_json(source / 'MANIFEST.json')
    records = {row['member']: row for row in binding['files']}
    def saved(member):
        return json.loads(_payload(source, records['results/' + member]))
    summary = saved('runtime/analysis/SUMMARY.json')
    cases = {Path(name).stem: saved(name.removeprefix('results/')) for name in records
             if name.startswith('results/runtime/analysis/cases/')}
    repeat = saved('runtime/REPEAT_CHECKS.json')
    require(repeat['status'] == 'PASS' and not repeat['discrepancies'], 'Frozen repeat disposition differs')
    examples = [dict(root=rid, method=method) for rid, case in cases.items() for method in METHODS
                for row in case[method]['inherited_coded'] if row['G2ab_M_witness']]
    independent = dict(single_format_full_control_examples=examples,
                       repeat_rows=dict(fixed=repeat['fixed_panel_reads'], full=repeat['complete_case_reads']),
                       repeat_max_logp_delta=0.0)
    derived = claims_from_outputs(summary, cases, independent, load_json(source / 'WORKED_EXAMPLE.json'))
    require(derived == load_json(source / 'CLAIMS.json'), 'Qwen32 claim projection differs from frozen case outputs')
    return derived


def check_portfolio(source=SOURCE, root=ROOT):
    """Count checkpoints from the retained completed rows, preserving failed bridges."""
    table = portfolio_table(root)
    source = Path(root) / 'reproducibility/text_scope'
    binding = verify_inputs(source, 'text_scope', root)
    record = next(r for r in binding['files'] if r['member'] == 'runtime/MODEL_ROSTER_FROZEN.json')
    roster = json.loads(_payload(source, record, root))
    models = {row['id']: row['model'] for row in roster['rows']}
    complete = [row for row in table if row['status'] == 'COMPLETE_FINAL_ROW']
    require(len(complete) == 6 and [row['W'] for row in complete] == [104, 93, 61, 68, 115, 110],
            'Earlier portfolio reconstruction is incomplete')
    original_qwen = [row for row in load_json(Path(root) / 'reproducibility/state_sufficiency/v10/primary_results.json')
                     if row['actor'] == 'qwen' and row['group'] in METHODS]
    require(len(original_qwen) == 2 and all(row['roots'] == 64 and row['mean'] > 0 for row in original_qwen),
            'Original Qwen3-8B record is missing')
    distinct = {models[row['row_id']] for row in complete} | {'Qwen/Qwen3-8B', 'Qwen/Qwen3-32B'}
    require(len(distinct) == 5 and all(row['status'] == 'NOT_MEASURED' for row in table if row['row_id'] == 'qwen32_t4'),
            'Checkpoint duplication or historical disposition drift')
    return dict(checkpoints=5, families=3, minimum_parameters_b=4, maximum_parameters_b=32,
                numerical_bridges_repaired=False, distinct_models=sorted(distinct))


def portfolio_table(root=ROOT):
    source = Path(root) / 'reproducibility/text_scope'
    binding = verify_inputs(source, 'text_scope', root)
    record = next(r for r in binding['files'] if r['member'] == 'runtime/analysis_final/MAIN_TABLE.json')
    return json.loads(_payload(source, record, root))


def export(output, source=SOURCE):
    require(__debug__, 'Replay requires assertions; do not use python -O')
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(source) and not source.is_relative_to(output),
            'Use a fresh output directory outside scientific inputs')
    binding = verify(source)
    output = new_output(output)
    restored = output / 'restored'
    restore(source, restored, binding)
    runtime = restored / 'results/runtime'
    driver = source / 'tools/replay_driver.py'
    try:
        run_offline(driver, ['--source', restored, '--output', output / 'computed'], output)
    except subprocess.CalledProcessError as error:
        raise ValueError('Offline source replay failed:\n' + error.stderr) from None
    computed = output / 'computed'
    replay = load_json(computed / 'analysis/INDEPENDENT_REPLAY.json')
    independent = load_json(computed / 'independent/INDEPENDENT_CHECK.json')
    validate_receipts(replay, independent)
    summary = load_json(computed / 'analysis/SUMMARY.json')
    expected_summary = load_json(runtime / 'analysis/SUMMARY.json')
    require(equivalent(summary, expected_summary), 'Full source summary differs')
    differences = float_differences(summary, expected_summary)
    cases = {}
    for path in sorted((computed / 'analysis/cases').glob('*.json')):
        cases[path.stem] = load_json(path)
        require(equivalent(cases[path.stem], load_json(runtime / 'analysis/cases' / path.name)),
                'Case classification differs: ' + path.name)
    require(len(cases) == 64, 'Missing reconstructed case')
    require(load_json(computed / 'REPEAT_CHECKS.json') == load_json(runtime / 'REPEAT_CHECKS.json'),
            'Reconstructed repeat disposition differs')
    repeat_differences = []
    repeat_paths = sorted((computed / 'analysis/full_repeats').glob('*.json'))
    require(len(repeat_paths) == 4, 'Missing full-repeat reconstruction')
    for path in repeat_paths:
        actual_repeat = load_json(path)
        expected_repeat = load_json(runtime / 'analysis/full_repeats' / path.name)
        require(equivalent(actual_repeat, expected_repeat),
                'Full-repeat reconstruction differs: ' + path.name)
        repeat_differences.extend(float_differences(actual_repeat, expected_repeat))
    for name in ('PER_CASE_PRIMARY.csv', 'PER_CASE_SEMANTIC_GATES.csv'):
        require((computed / 'presentation' / name).read_bytes() == (restored / 'results' / name).read_bytes(),
                'Source table differs: ' + name)
    example = load_json(computed / 'WORKED_EXAMPLE.json')
    claims = claims_from_outputs(summary, cases, independent, example)
    require([claims['methods'][m]['primary'] for m in METHODS] == [128, 123], 'Primary counts differ')
    require(all(claims['methods'][m]['strongest'] == 0 for m in METHODS), 'Strongest result differs')
    with (output / 'CLAIMS.json').open('x', encoding='utf8') as stream:
        json.dump(claims, stream, indent=2)
        stream.write('\n')
    report = dict(status='PASS', compact_members_checked=len(binding['files']),
                  raw_final_rows=36992, classification_fields_checked=27136,
                  repeats_checked=4488, repeats_strictly_equal=True, full_case_classifications_identical=4,
                  all_case_outputs_and_full_summary_reconstructed=True,
                  original_csv_tables_byte_identical=True, original_figure_data_reconstructed=True,
                  summary_max_float_difference=max(differences, default=0), float_atol=1e-12, float_rtol=0,
                  full_repeat_derived_max_float_difference=max(repeat_differences, default=0),
                  worked_example_checked=True, public_inputs_checked=True,
                  inference=False, socket_and_model_imports_blocked=True)
    with (output / 'VERIFICATION.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        binding = verify(args.source)
        result = dict(status='PASS', compact_members_checked=len(binding['files']), public_inputs_checked=True,
                      scope='Source/member integrity only; use --output for full saved-score reconstruction')
    else:
        result = export(args.output, args.source)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
