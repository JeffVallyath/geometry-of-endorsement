"""CPU saved-score replay adapter; original scientific functions unchanged.

One study per fresh Python subprocess avoids the original V4 shared-seed mutation.
No model imports, training, generation, witness inspection or provider calls.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import importlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, new_output

STAGE = ROOT / 'reproducibility/relational_editing'
LATEST = {'v1': 'r4', 'v2': 'r4', 'v3': 'r2', 'v4': 'r1'}
MODULE = {'v1': 'qbrc_analyze', 'v2': 'srs2_analyze', 'v3': 'rso3_analyze', 'v4': 'rcc4_analyze'}
FAMILIES = {'QBRC': 'question_blind_relation_control_v1', 'SRS2': 'shared_relation_state_v2', 'RSO3': 'relational_set_operations_v3', 'RCC4': 'relational_composition_coverage_v4'}


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise ImportError('CPU replay forbids model/provider dependency: ' + fullname)
        return None


def load_rows(path):
    path = Path(path)
    if not path.is_file():
        path = Path(str(path) + '.gz')
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_metadata(path):
    path = Path(path)
    # Evidence identity/cache propagation is NOT reverified by saved-score replay.
    # Explicitly exclude those branches; all program identity rows remain read.
    if path.name in {'WITNESS_INDEX.json', 'FORECAST.json', 'MEASUREMENTS_TERMINAL.json'}:
        return None
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else None


def parse_cell(value):
    if value == '':
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def compare_value(expected, actual, path, errors, counts):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            errors.append(path)
    elif isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        counts['numeric_values'] += 1
        if not math.isclose(expected, actual, rel_tol=1e-11, abs_tol=1e-12):
            errors.append(path)
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            errors.append(path + ':keys')
        for key in expected.keys() & actual.keys():
            compare_value(expected[key], actual[key], f'{path}.{key}', errors, counts)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            errors.append(path + ':length')
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare_value(x, y, f'{path}[{i}]', errors, counts)
    elif expected != actual:
        errors.append(path)


def compare_tables(version, output, expected_directory=None):
    results = []
    expected_directory = expected_directory or STAGE / version / 'expected'
    for expected_path in sorted(expected_directory.glob('T*.csv')):
        actual_path = output / expected_path.name
        errors = []
        counts = {'numeric_values': 0}
        if not actual_path.is_file():
            results.append({'table': expected_path.name, 'status': 'MISSING'})
            continue
        with expected_path.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            expected_headers = reader.fieldnames
            expected_rows = list(reader)
        with actual_path.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            actual_headers = reader.fieldnames
            actual_rows = list(reader)
        if set(expected_headers or []) != set(actual_headers or []):
            errors.append('headers')
        if len(expected_rows) != len(actual_rows):
            errors.append('row_count')
        # Canonical V1 composition emits order-disagreement rows by set order.
        # Align on scientific identity, never on values being verified. This
        # preserves every duplicate row and requires unchanged row counts.
        identity = [k for k in ('actor', 'model', 'task', 'panel', 'condition',
                               'candidate', 'seed', 'epoch', 'checkpoint',
                               'strength', 'lr', 'program', 'world', 'procedure',
                               'order', 'phrasing', 'family')
                    if k in (expected_headers or []) and k in (actual_headers or [])]
        if identity:
            key = lambda row: tuple(row[k] for k in identity)
            expected_rows.sort(key=key)
            actual_rows.sort(key=key)
        for i, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
            for key in expected.keys() & actual.keys():
                compare_value(parse_cell(expected[key]), parse_cell(actual[key]), f'row[{i}].{key}', errors, counts)
        results.append({'table': expected_path.name, 'expected_rows': len(expected_rows), 'actual_rows': len(actual_rows), **counts, 'mismatch_count': len(errors), 'mismatches': errors[:30], 'status': 'PASS' if not errors else 'FAIL'})
    return results


def run(version, output, editing_root=None):
    if version not in LATEST:
        raise ValueError('Unknown completed editing version')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty replay directory; existing evidence is immutable')
    new_output(output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    editing_root = STAGE
    # Public scientific layout: version/source contains module files directly;
    # version/reference and version/data are the original supplied interfaces.
    # Keep each version's helper snapshots before fallback dependencies.
    for key, dependency in zip(FAMILIES, ('v1', 'v2', 'v3', 'v4')):
        os.environ[key + '_SUPPLIED'] = str(editing_root / dependency)
    dependencies = [version] + [v for v in ('v3', 'v2', 'v1') if v != version]
    sys.path[:0] = [str(editing_root / v / 'source') for v in dependencies]
    run_root = editing_root / version / 'scores'
    qualification = run_root if version == 'v4' else editing_root / version / 'qualification'
    expected = editing_root / version / 'expected'
    sys.meta_path.insert(0, NoModelImports())
    analyzer = importlib.import_module(MODULE[version])
    # Adapt I/O only, retaining the canonical original metric/threshold/bootstrap
    # functions. All metadata and raw scores are the committed immutable selection.
    for name in ('qbrc_common', 'srs2_common', 'rso3_common', 'rcc4_common'):
        if name in sys.modules:
            sys.modules[name].read_rows = load_rows
    for name in ('qbrc_analyze', 'srs2_analyze', 'rso3_analyze', 'rcc4_analyze'):
        if name in sys.modules:
            sys.modules[name].rows = load_rows
            sys.modules[name].load = load_metadata
    analyzer.analyze(run_root, qualification, output)
    if any(name in sys.modules for name in ('torch', 'transformers', 'huggingface_hub')):
        raise RuntimeError('Unexpected model dependency imported')
    comparisons = compare_tables(version, output, expected)
    report = {
        'version': version,
        'status': 'PASS' if comparisons and all(r['status'] == 'PASS' for r in comparisons) else 'FAIL',
        'tables': comparisons,
        'scientific_functions': 'Original canonical analyzer and executed common/reference code; only saved-row/metadata I/O functions replaced.',
        'boundaries': [
            'T1 uses frozen qualification summaries; no raw generation re-scoring.',
            'T2 uses persisted development/checkpoint summaries; no training rerun.',
            'Fresh single/program/control/energy/workflow tables are recomputed from saved query/episode rows with original bootstrap functions.',
            'Physical witness inspection, runtime forecasts and custody receipts excluded. Original reported interface validity is historical metadata, not newly verified.',
            'V2 numerical criteria remain under amended qualification, not an original-protocol pass.',
            'No models loaded, inference performed, or GPU/provider accessed.',
        ],
    }
    (output / 'REPLAY_CHECK.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'version': version, 'status': report['status'], 'tables': len(comparisons), 'numeric_values': sum(r.get('numeric_values', 0) for r in comparisons), 'mismatches': sum(r.get('mismatch_count', 0) for r in comparisons), 'report': str(output / 'REPLAY_CHECK.json')}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=LATEST, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return run(args.version, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
