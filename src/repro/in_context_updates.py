"""CPU replay of the completed Qwen/Ripple study, using both frozen parsers."""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from .common import ROOT, load_json, new_output, safe_path, sha256

BASE = ROOT / 'reproducibility/in_context_updates'
PHASES = ('development', 'qualification', 'final', 'repeat_670', 'repeat_618')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_modules(base, version):
    """Separate package namespaces preserve original relative imports and parsers."""
    name = '_qwen_replay_' + version
    folder = base / 'supplied' / version / 'icmh'
    spec = importlib.util.spec_from_file_location(name, folder / '__init__.py',
                                                submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + '.analyze')


def verify_inputs(base=BASE):
    supplied = base / 'supplied'
    manifest = load_json(supplied / 'MANIFEST.json')
    for row in manifest['files']:
        path = safe_path(row['path'], supplied)
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'Qwen source changed: ' + row['path'])
        if row['transformation'] == 'gzip-lossless':
            import hashlib
            require(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == row['source_sha256'],
                    'Lossless source binding failed: ' + row['path'])
    freeze = load_json(supplied / 'freezes/PARSER_FREEZE.json')
    for name, digest in freeze['inputs_sha256'].items():
        require(sha256(supplied / 'inputs/built' / name) == digest, 'Frozen input changed: ' + name)
    for name, digest in freeze['source_sha256'].items():
        require(sha256(supplied / 'source/icmh' / name) == digest, 'Frozen code changed: ' + name)
    require(sha256(supplied / 'source_v2/icmh/parse.py') == freeze['parser']['v2_parse_py_sha256'],
            'Original parser changed')
    return manifest


def reconstruct_phase(base, phase, analyzer, registry, journal):
    inputs = base / 'supplied/inputs/built'
    plan, scoring, _, generations, starts, completions = analyzer.load_rows(inputs, phase, [journal])
    ids = [u['unit_id'] for u in plan['units']]
    require(len(ids) == len(set(ids)) == plan['unit_count'], 'Duplicate or missing plan units')
    require(set(ids) == set(generations) == set(scoring), 'Missing or extra generation/scoring units')
    require(len(starts) == len(completions) == 1, 'Missing phase start/completion')
    require(starts[0]['plan_sha256'] == sha256(inputs / ('PLAN_' + phase + '.json')), 'Wrong journal plan')
    require(starts[0]['model_receipt_sha256'] == sha256(base / 'supplied/freezes/MODEL_RECEIPT.json'),
            'Wrong journal model receipt')
    rows = [analyzer.score_unit(unit, scoring, registry, generations) for unit in plan['units']]
    roots = {r['root_id']: r for r in load_json(inputs / 'ROOTS.json')['roots']}
    per_root, per_question = analyzer.flags(plan, analyzer.index(rows), roots)
    result = dict(planned=len(ids), measured=len(generations), per_root=per_root, per_question=per_question)
    if phase == 'qualification':
        # Resource availability is historical metadata, never a fresh authorization.
        resource = load_json(base / 'supplied/freezes/PHASE_GATE.json')['gates']['resource']
        gates = analyzer.qualification_gates(plan, rows, per_root, per_question,
                                             analyzer.throughput(rows, starts, completions), resource)
        result['scientific_gates'] = {k: gates[k] for k in ('G1', 'G2', 'G4', 'G5', 'technical')}
    if phase == 'final' or phase.startswith('repeat'):
        result['results'] = analyzer.final_results(per_root, per_question, rows)
    return result, rows, generations


def replay(base=BASE, *, compare=True):
    verify_inputs(base)
    lp1 = frozen_modules(base, 'source')
    union = frozen_modules(base, 'source_v2')
    registries = {key: load_json(base / 'supplied/inputs/built' / name)['registry']
                  for key, name in (('primary', 'REGISTRY.json'), ('alternative', 'REGISTRY_UNION.json'))}
    phases, scored, generations = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='qwen-saved-replay-') as temp:
        for phase in PHASES:
            journal = Path(temp) / (phase + '.jsonl')
            journal.write_bytes(gzip.decompress((base / 'supplied/journals' / phase / 'journal.jsonl.gz').read_bytes()))
            phases[phase], scored[phase], generations[phase] = {}, {}, {}
            for rule, analyzer in (('primary', lp1), ('alternative', union)):
                result, rows, raw = reconstruct_phase(base, phase, analyzer, registries[rule], journal)
                phases[phase][rule], scored[phase][rule], generations[phase][rule] = result, rows, raw
    repeats = {}
    for phase in PHASES[3:]:
        comparisons = {rule: lp1.repeat_comparison(scored['final'][rule], scored[phase][rule])
                       for rule in registries}
        for rule, comparison in comparisons.items():
            require(not comparison['repeat_only'], 'Repeat units absent from final')
            require(comparison['compared_units'] == phases[phase][rule]['measured'], 'Incomplete repeat coverage')
        repeats[phase] = dict(roots=list(phases[phase]['primary']['per_root']),
            compared_units=comparisons['primary']['compared_units'],
            identical=all(c['identical'] for c in comparisons.values()),
            differences={k: v['differences'] for k, v in comparisons.items()},
            witnesses=phases[phase]['primary']['results']['counts']['W_questions'])
    primary = phases['final']['primary']['results']
    alternative = phases['final']['alternative']['results']
    roots_info = {r['root_id']: r for r in load_json(base / 'supplied/inputs/built/ROOTS.json')['roots']}
    obsolete = []
    for witness in primary['witnesses']:
        root = roots_info[witness['root_id']]
        wrong = [h for h in ('B', 'D') if not witness['chronological_' + h]['strict_correct']]
        require(len(wrong) == 1, 'Witness does not have exactly one wrong answer')
        h = wrong[0]
        obsolete.append(witness['chronological_' + h]['extracted_id'] == root['changed_slot_ids'][h])
    result = dict(primary=primary, alternative=alternative, repeats=repeats,
        repeat_roots=len(repeats), repeat_units=sum(r['compared_units'] for r in repeats.values()),
        repeat_witnesses=sum(r['witnesses'] for r in repeats.values()),
        all_repeat_answers_identical=all(r['identical'] for r in repeats.values()),
        all_wrong_witness_answers_are_obsolete=bool(obsolete) and all(obsolete),
        locality_distinct_questions=len({r['reference'] for r in scored['final']['primary'] if r['kind'] == 'locality'}),
        phase_coverage={p: {k: phases[p]['primary'][k] for k in ('planned', 'measured')} for p in PHASES},
        qualification={r: phases['qualification'][r]['scientific_gates'] for r in registries},
        parser_changed_questions=sum(any(phases['final']['primary']['per_question'][q][k] != phases['final']['alternative']['per_question'][q][k]
            for k in ('A', 'C', 'D', 'W', 'A_and_C')) for q in phases['final']['primary']['per_question']),
        scope='Frozen analysis of saved responses; no neural inference or historical clock-time attestation.')
    if compare:
        # Expected results are opened only after all classifications are calculated.
        for phase in PHASES:
            for rule, name in (('primary', 'PRIMARY_LP1'), ('alternative', 'SENSITIVITY_UNION_V2')):
                expected = json.loads(gzip.decompress((base / 'supplied/expected' / phase / (name + '.json.gz')).read_bytes()))
                actual = phases[phase][rule]
                for key in ('planned', 'measured', 'per_root', 'per_question', 'results'):
                    if key in actual:
                        require(actual[key] == expected[key], f'Frozen replay differs: {phase}/{rule}/{key}')
                require(scored[phase][rule] == expected['units'], f'Scored units differ: {phase}/{rule}')
                if phase == 'qualification':
                    require(actual['scientific_gates'] == {k: expected['gates'][k] for k in actual['scientific_gates']},
                            'Qualification replay differs')
    return result


def export(output, base=BASE):
    result = replay(base)
    destination = new_output(Path(output))
    with (destination / 'results.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    with (destination / 'per_root_results.csv').open('x', newline='', encoding='utf8') as f:
        fields = ['parser', 'root_id', 'A', 'records_correct_both', 'records', 'questions', 'C', 'D', 'W']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rule in ('primary', 'alternative'):
            for root, row in result[rule]['per_root'].items():
                writer.writerow(dict(parser=rule, root_id=root, **row))
    with (destination / 'public_data_table.md').open('x', encoding='utf8') as f:
        f.write('| Saved-response measure | Primary label-first parser | Alternative union parser |\n|---|---:|---:|\n')
        for label, field in [('Qualifying history-dependent questions', 'W_questions'), ('Questions passing factual and reference checks', 'A_and_C_questions'), ('Benchmark cases with a qualifying question', 'roots_with_any_W')]:
            denominator = 'roots' if field == 'roots_with_any_W' else 'questions'
            values = [f"{result[r]['counts'][field]}/{result[r]['denominators'][denominator]}" for r in ('primary', 'alternative')]
            f.write(f'| {label} | {values[0]} | {values[1]} |\n')
        f.write('\nSame recorded answers under both rules; clustered questions, not a deployment prevalence estimate.\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('reproduced/qwen'))
    a = p.parse_args()
    r = export(a.output)
    print(json.dumps({k: r[k] for k in ('phase_coverage', 'repeats', 'all_wrong_witness_answers_are_obsolete')} |
        {'primary': r['primary']['counts'], 'alternative': r['alternative']['counts'],
         'locality': r['primary']['locality_panel_diagnostic']}, indent=2))


if __name__ == '__main__':
    main()
