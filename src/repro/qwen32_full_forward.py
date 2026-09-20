"""Portable Qwen3-32B saved-score replay. Never loads a model or contacts a service."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from .common import ROOT, load_json, new_output, safe_path, sha256
from .fresh_history import equivalent, float_differences, run_offline

SOURCE = ROOT / 'reproducibility/qwen32_full_forward'
ARCHIVE = '5777db17c80cb0145207c447a0303d395f309e045f17f4b2e6d84157a294fca3'
CAPSULE = 'b8b523be57e3058868ea23539c61778024022a903a192d942e58a8bc9a9e70e4'
MANIFESTS = {
    'results': 'e0861d9df4473e11b22764d26185e4b5fe83284d2f1ac12d6554e3c481d8c62e',
    'handoff': '598ec0669aa8d5da0bf294cf084b858f81c3d1ffe259ddfca5a70a0b026c9f71',
}
METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _payload(source, record):
    raw = safe_path(record['path'], source).read_bytes()
    require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'],
            'Stored payload hash mismatch: ' + record['path'])
    require(record['encoding'] in ('gzip', 'verbatim'), 'Unknown payload encoding')
    raw = gzip.decompress(raw) if record['encoding'] == 'gzip' else raw
    require(len(raw) == record['original_bytes'] and hashlib.sha256(raw).hexdigest() == record['original_sha256'],
            'Original member hash mismatch: ' + record['member'])
    return raw


def verify(source=SOURCE, archive=None):
    source = Path(source).resolve()
    binding = load_json(source / 'SOURCE.json')
    require(binding['schema'] == 'QWEN32_COMPACT_REPLAY_V1', 'Wrong source schema')
    require(binding['archives']['results']['sha256'] == ARCHIVE and binding['capsule_sha256'] == CAPSULE,
            'Unexpected source archive or capsule')
    paths, members, manifests = set(), set(), {}
    for label, expected in MANIFESTS.items():
        matches = [row for row in binding['files'] if row['archive'] == label and row['member'] == 'MANIFEST.json']
        require(len(matches) == 1, 'Missing or duplicate original manifest')
        raw = _payload(source, matches[0])
        require(hashlib.sha256(raw).hexdigest() == expected, 'Original manifest changed')
        manifests[label] = json.loads(raw)
    for record in binding['files']:
        safe_path(record['member'], source)
        member = record['archive'], record['member'].casefold()
        require(record['path'].casefold() not in paths and member not in members, 'Duplicate compact binding')
        paths.add(record['path'].casefold())
        members.add(member)
        raw = _payload(source, record)
        if record['member'] != 'MANIFEST.json':
            original = manifests[record['archive']]['files'][record['member']]
            require(len(raw) == original['bytes'] and hashlib.sha256(raw).hexdigest() == original['sha256'],
                    'Payload differs from original archive manifest')
    observed = {p.relative_to(source).as_posix().casefold() for p in (source / 'saved').rglob('*') if p.is_file()}
    require(observed == paths, 'Undeclared or missing compact payload')
    required_inputs = {('results', name.casefold()) for name in manifests['results']['files']
                       if name.startswith(('runtime/results/', 'runtime/cpu/designs/', 'runtime/analysis/'))}
    require(required_inputs <= members, 'Incomplete final, repeat, design or comparison inputs')
    if archive is not None:
        archive = Path(archive)
        require(archive.stat().st_size == binding['archives']['results']['bytes'] and sha256(archive) == ARCHIVE,
                'External archive hash mismatch')
        with zipfile.ZipFile(archive) as z:
            expected = manifests['results']['files']
            require(len(z.namelist()) == len(expected) + 1 and set(z.namelist()) == set(expected) | {'MANIFEST.json'},
                    'External archive member inventory mismatch')
            for name, row in expected.items():
                safe_path(name, source)
                raw = z.read(name)
                require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                        'External archive member changed: ' + name)
    return binding


def restore(source, destination, binding):
    require(not destination.exists(), 'Use a fresh restored-source directory')
    for record in binding['files']:
        path = safe_path(record['archive'] + '/' + record['member'], destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(_payload(source, record))


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
    binding = verify(source) if integrity else load_json(source / 'SOURCE.json')
    records = {(row['archive'], row['member']): row for row in binding['files']}
    def saved(member):
        return json.loads(_payload(source, records['results', member]))
    summary = saved('runtime/analysis/SUMMARY.json')
    cases = {Path(name).stem: saved(name) for label, name in records
             if label == 'results' and name.startswith('runtime/analysis/cases/')}
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
    """Require actual source-bound smaller-model rows before counting five checkpoints."""
    source, root = Path(source), Path(root)
    receipt = load_json(source / 'portfolio/PORTFOLIO_CHECK.json')
    require(receipt['status'] == 'PASS' and receipt['raw_score_records_decoded'] == 110976,
            'Earlier portfolio reconstruction is incomplete')
    require(receipt['source_archive_sha256'] == 'eb5bee9d247bdfbea59ad894053a4b2054a0d605a898e008b889517c614b7441',
            'Unexpected earlier portfolio source')
    for member, record in receipt['original_member_bindings'].items():
        path = safe_path(Path(member).name, source / 'portfolio')
        require(path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'],
                'Earlier portfolio member changed')
    table = load_json(source / 'portfolio/MAIN_TABLE.json')
    for row in receipt['primary_rows']:
        original = next(r for r in table if r['row_id'] == row['row_id'] and r['method'] == row['method'])
        require(original['status'] == 'COMPLETE_FINAL_ROW' and original['W'] == row['witnesses'] > 0,
                'Unsupported completed checkpoint')
    original_qwen = [row for row in load_json(root / 'reproducibility/state_sufficiency/v10/primary_results.json')
                     if row['actor'] == 'qwen' and row['group'] in METHODS]
    require(len(original_qwen) == 2 and all(row['roots'] == 64 and row['mean'] > 0 for row in original_qwen),
            'Original Qwen3-8B record is missing')
    models = {row['model'] for row in receipt['primary_rows']} | {'Qwen/Qwen3-8B', 'Qwen/Qwen3-32B'}
    require(len(models) == 5 and all(row['status'] == 'NOT_MEASURED' for row in table if row['row_id'] == 'qwen32_t4'),
            'Checkpoint duplication or historical disposition drift')
    result = dict(checkpoints=5, families=3, minimum_parameters_b=4, maximum_parameters_b=32,
                  numerical_bridges_repaired=False, distinct_models=sorted(models))
    require(result == load_json(source / 'portfolio/PORTFOLIO.json'), 'Portfolio claim projection differs')
    return result


def export(output, source=SOURCE, archive=None):
    require(__debug__, 'Replay requires assertions; do not use python -O')
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(source) and not source.is_relative_to(output),
            'Use a fresh output directory outside scientific inputs')
    binding = verify(source, archive)
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
                  worked_example_checked=True, external_archive_checked=archive is not None,
                  inference=False, socket_and_model_imports_blocked=True)
    with (output / 'VERIFICATION.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        binding = verify(args.source, args.archive)
        result = dict(status='PASS', compact_members_checked=len(binding['files']), external_archive_checked=args.archive is not None,
                      scope='Source/member integrity only; use --output for full saved-score reconstruction')
    else:
        result = export(args.output, args.source, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
