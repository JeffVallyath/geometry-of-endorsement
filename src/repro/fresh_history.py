"""Replay fresh Gemma confirmation and validation utility from saved scores only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from .common import ROOT, load_json, new_output, safe_path, sha256

SOURCE = ROOT / 'reproducibility/fresh_history_validation'
ARCHIVE_SHA256 = '0f52f3057797c3d07c9e945c1f920834bab37364fce11f53ed13e404c29431c4'

# The saved-score child cannot import a model/provider client or open a socket.
OFFLINE_BOOTSTRAP = '''
import importlib.abc, runpy, sys
from pathlib import Path
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','huggingface_hub','lambda_api','operator_adapter'}:
            raise ImportError('Saved-score replay forbids model and private-operator imports: '+fullname)
def offline(event, args):
    if event.startswith('socket.') and event != 'socket.gethostname':
        raise RuntimeError('Saved-score replay forbids network access')
sys.meta_path.insert(0, NoModels())
sys.addaudithook(offline)
script=Path(sys.argv[1]).resolve()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(script.parent))
runpy.run_path(str(script),run_name='__main__')
'''


def equivalent(actual, expected):
    """Exact identities/classifications; only finite floats allow absolute 1e-12."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if type(actual) is float and type(expected) is float:
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= 1e-12
    return type(actual) is type(expected) and actual == expected


def verify(source=SOURCE):
    source = Path(source).resolve()
    manifest = load_json(source / 'MANIFEST.json')
    if manifest.get('schema') != 'FHV1_COMPACT_SCIENTIFIC_V1':
        raise ValueError('Wrong compact scientific manifest schema')
    if manifest['source_archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Wrong immutable study source')
    seen = set()
    for row in manifest['files']:
        name = row['path']
        if name.casefold() in seen:
            raise ValueError('Duplicate scientific artifact')
        seen.add(name.casefold())
        path = safe_path(name, source)
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Scientific artifact changed: ' + name)
    provenance = load_json(source / 'SOURCE.json')
    if provenance['source_archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Source provenance changed')
    freeze = load_json(source / 'generated/DATA_FREEZE.json')
    frozen_names = set()
    for recorded_name, expected in freeze['files'].items():
        name = recorded_name.replace('\\', '/')
        if name.casefold() in frozen_names:
            raise ValueError('Duplicate normalized frozen input')
        frozen_names.add(name.casefold())
        if sha256(safe_path(name, source / 'generated')) != expected:
            raise ValueError('Frozen pre-outcome input changed: ' + name)
    actual_names = {p.relative_to(source).as_posix().casefold() for p in source.rglob('*')
                    if p.is_file() and p.relative_to(source).as_posix() != 'MANIFEST.json' and '__pycache__' not in p.parts}
    if actual_names != seen:
        raise ValueError('Compact scientific inventory has missing or undeclared files')
    return len(seen)


def float_differences(actual, expected):
    if isinstance(actual, dict):
        return [delta for key in actual for delta in float_differences(actual[key], expected[key])]
    if isinstance(actual, list):
        return [delta for a, b in zip(actual, expected) for delta in float_differences(a, b)]
    return [abs(actual - expected)] if type(actual) is float and actual != expected else []


def equivalent_csv(actual, expected):
    def read(path):
        with path.open(encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    actual_fields, actual_rows = read(actual)
    expected_fields, expected_rows = read(expected)
    if actual_fields != expected_fields or len(actual_rows) != len(expected_rows):
        return False
    identities = {'seed', 'draw', 'root_id', 'query_id', 'rule', 'procedure'}
    for left, right in zip(actual_rows, expected_rows):
        for key, value in right.items():
            if left[key] == value:
                continue
            if key in identities or not any(c in value.lower() for c in '.e'):
                return False
            try:
                if not equivalent(float(left[key]), float(value)):
                    return False
            except ValueError:
                return False
    return True


def coverage_newline_equivalent(actual, expected, relative_name):
    """One presentation JSON permits CRLF/LF only, never altered values/spacing."""
    return (relative_name == 'tables/TABLE_COVERAGE.json'
            and actual.read_bytes().replace(b'\r\n', b'\n')
            == expected.read_bytes().replace(b'\r\n', b'\n'))


def svg_comparison(actual, expected):
    """Exact structure/text/style; scalar plot coordinates allow absolute 1e-12."""
    left, right = ET.parse(actual).getroot(), ET.parse(expected).getroot()
    deltas = []
    coordinates = {'x', 'y', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy', 'r', 'width', 'height'}
    def same(a, b):
        if (a.tag != b.tag or a.text != b.text or a.tail != b.tail
                or a.attrib.keys() != b.attrib.keys() or len(a) != len(b)):
            return False
        for key, value in b.attrib.items():
            if a.attrib[key] == value:
                continue
            if key not in coordinates:
                return False
            try:
                av, bv = float(a.attrib[key]), float(value)
            except ValueError:
                return False
            if not equivalent(av, bv):
                return False
            deltas.append(abs(av - bv))
        return all(same(x, y) for x, y in zip(a, b))
    return dict(equivalent=same(left, right), coordinate_atol=1e-12,
                changed_coordinates=len(deltas), max_coordinate_difference=max(deltas, default=0))


def run_offline(script, arguments, cwd):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'LAMBDA_API_KEY'):
        env.pop(key, None)
    return subprocess.run([sys.executable, '-B', '-c', OFFLINE_BOOTSTRAP, str(script), *map(str, arguments)],
                          cwd=cwd, env=env, check=True, capture_output=True, text=True)


def export(output, source=SOURCE):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a fresh output directory outside the scientific inputs')
    checked = verify(source)
    output = new_output(output)
    run_offline(source / 'source/replay.py', ['--source', source, '--run', source / 'scientific',
                                           '--output', output / 'replay'], output)
    run_offline(source / 'source/diagnostics.py', ['--source', source, '--output', output / 'diagnostics'], output)
    run_offline(source / 'source/result_tables.py', ['--source', output / 'replay', '--run', source / 'scientific',
                                                   '--output', output / 'tables'], output)
    run_offline(source / 'source/result_figures.py', ['--source', output / 'replay', '--output', output / 'figures'], output)
    comparisons = []
    for expected in sorted((source / 'expected').glob('*.json')):
        actual = output / 'replay' / expected.name
        actual_data, expected_data = load_json(actual), load_json(expected)
        if not equivalent(actual_data, expected_data):
            raise ValueError('Frozen scientific reconstruction differs: ' + expected.name)
        deltas = float_differences(actual_data, expected_data)
        comparisons.append(dict(file=expected.name, byte_identical=sha256(actual) == sha256(expected),
                                changed_float_leaves=len(deltas), max_absolute_float_difference=max(deltas, default=0)))
    repeats = load_json(output / 'replay/REPEATS.json')
    presentation = []
    for folder in ('tables', 'figures'):
        expected_names = {p.name for p in (source / folder).iterdir() if p.is_file()}
        actual_names = {p.name for p in (output / folder).iterdir() if p.is_file()}
        if expected_names != actual_names:
            raise ValueError('Presentation inventory differs: ' + folder)
        for name in sorted(expected_names):
            actual, expected = output / folder / name, source / folder / name
            identical = sha256(actual) == sha256(expected)
            newline_only = not identical and coverage_newline_equivalent(actual, expected, folder + '/' + name)
            svg = svg_comparison(actual, expected) if expected.suffix == '.svg' else None
            matches = identical or newline_only or (expected.suffix == '.csv' and equivalent_csv(actual, expected)) or (svg and svg['equivalent'])
            if not matches:
                raise ValueError('Reconstructed presentation differs: ' + folder + '/' + name)
            presentation.append(dict(file=folder + '/' + name, byte_identical=identical,
                                     newline_only_difference=newline_only, svg_comparison=svg))
    report = dict(status='PASS', checked_artifacts=checked, final_cases=64, seeds=[0, 1],
        repeated_logical_records=sum(r['rows'] for r in repeats),
        repeats_strictly_equal=all(r['max_logp_delta'] == 0 and not r['all_discrepancies'] for r in repeats),
        frozen_repeat_acceptance_tolerance=1e-6, saved_score_float_atol=1e-12,
        saved_score_float_rtol=0, identity_and_classification_equality='exact',
        comparisons=comparisons, presentation=presentation, post_hoc_diagnostics_separate=True,
        inference=False, model_and_operator_imports_blocked=True, socket_access_blocked=True)
    with (output / 'verification.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.source), indent=2))


if __name__ == '__main__':
    main()
