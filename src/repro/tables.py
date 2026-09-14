"""CPU-only saved-evidence replay and deterministic table export."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from .common import ROOT, load_json, new_output
from .relational import replay_v6
from .representation import replay as replay_representation
from .headlines import from_replay
from .relational import equal


def original_representation():
    data = load_json(ROOT / 'artifacts/figures/figure_data.json')
    result = {}
    for model in ('llama', 'gemma'):
        source = data['models'][model]
        result[model] = {'layer': source['selected_layer'],
                         'direction': source['evaluations']['difference_in_means']['I_b'],
                         'logistic': source['evaluations']['logistic']['I_b'],
                         'text': source['evaluations']['sbert_interaction']['I_b']}
    result['llama']['truth'] = data['truth_control']['llama']['primary_T']
    result['gemma']['truth'] = data['models']['gemma']['truth_control_T']
    return result


def compute():
    return {'representation': original_representation(), 'source_independence': replay_v6(),
            'representation_studies': replay_representation()}


def compare_canonical(actual, expected, path='headlines'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Canonical field inventory changed: ' + path)
        for key in expected:
            compare_canonical(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Canonical list inventory changed: ' + path)
        for i, value in enumerate(expected):
            compare_canonical(actual[i], value, path + '.' + str(i))
    else:
        equal(actual, expected, path)


def reproduce(output: Path) -> dict:
    result = compute()
    headlines = from_replay(result)
    compare_canonical(headlines, load_json(ROOT / 'reproducibility/headline_values.json'))
    destination = new_output(output / 'tables')
    for version in ('v1', 'v2', 'v3', 'v4'):
        subprocess.run([sys.executable, '-B', '-m', 'repro.editing', '--version', version,
                        '--output', str(destination / 'relational_editing' / version)], check=True)
    json_path = destination / 'results.json'
    with json_path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (destination / 'headline_values.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(headlines, stream, indent=2, allow_nan=False)
        stream.write('\n')
    programs = result['source_independence']['program_tables']
    with (destination / 'source_independence_programs.csv').open('x', encoding='utf8', newline='') as stream:
        keys = [k for k in programs[0] if k != 'changed_by_direction']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(programs)
    return {'status': 'saved_evidence_replayed', 'output': str(destination),
            'source_independence_program_rows': len(programs),
            'scope': 'original aggregates and scene/root sufficient-statistic replay; no model inference'}
