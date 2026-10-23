"""Scientific saved-score checks, independent of private workers and providers."""
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import zipfile

import pytest

from repro import fresh_history as replay


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


@pytest.fixture(scope='module')
def analyzer():
    spec = importlib.util.spec_from_file_location('fresh_frozen_analyzer', replay.SOURCE / 'source/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def example(analyzer):
    folder = replay.SOURCE / 'generated/FHV1-FINAL-0017'
    design = [load(folder / name) for name in ('root.json', 'questions.json', 'PLAN.json')]
    panel = load(replay.SOURCE / 'scientific/final/FHV1-FINAL-0017/SCORES.json.gz')
    physical = {(r['context'], r['query_id']): r for r in panel['physical']}
    index = {(c, q): analyzer.decode(physical[panel['aliases'].get(c, c), q])
             for c in panel['contexts'] for q in {q for c, q in physical}}
    return design, index


def test_compact_inventory_and_frozen_calculation_bytes():
    assert replay.verify() > 600
    source = load(replay.SOURCE / 'CODE_MANIFEST.json')
    for name in ('source/analyze.py', 'source/decision.py', 'source/replay.py'):
        record = next(r for r in source['files'] if r['path'] == name)
        assert hashlib.sha256((replay.SOURCE / name).read_bytes()).hexdigest() == record['sha256']
    trees = ast.parse((replay.SOURCE / 'source/panels.py').read_text(encoding='utf8'))
    for name, expected in source['function_ast_sha256'].items():
        node = next(n for n in trees.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected


def test_fresh_diagnostics_are_separate_and_keep_all_observations():
    data = load(replay.SOURCE / 'diagnostics/DIAGNOSTICS.json')
    assert data['classification'] == 'POST_HOC_DIAGNOSTICS_NOT_FROZEN_ENDPOINTS'
    assert (data['primary_observations'], data['primary_cases']) == (22, 14)
    assert (data['withheld_target_observations'], data['withheld_target_both_accept']) == (13, 13)
    assert data['withheld_target_cases'] == 7
    assert data['withheld_target_selected_pair_disagreements'] == 9
    assert (data['exact_token_inventory_observations'], data['exact_token_inventory_cases']) == (22, 14)
    assert len(data['witnesses']) == 22


def test_four_repeats_are_the_first_positive_cases_in_frozen_source_order():
    diagnostics = load(replay.SOURCE / 'diagnostics/DIAGNOSTICS.json')
    witnessed = {row['root_id'] for row in diagnostics['witnesses']}
    order = [row['root_id'] for row in load(replay.SOURCE / 'generated/INDEX.json') if row['split'] == 'FINAL']
    selection = load(replay.SOURCE / 'scientific/REPEAT_SELECTION.json')['root_ids']
    assert selection == [rid for rid in order if rid in witnessed][:4]
    comparisons = [load(replay.SOURCE / 'scientific/repeat' / rid / 'COMPARISON.json') for rid in selection]
    assert sum(row['rows'] for row in comparisons) == 2312
    assert all(row['max_logp_delta'] == 0 and row['all_discrepancies'] == [] for row in comparisons)


def test_table_comparison_preserves_exact_identity_and_integer_fields(tmp_path):
    expected, actual = tmp_path / 'expected.csv', tmp_path / 'actual.csv'
    expected.write_text('seed,count,rate\n0,22,0.5\n', encoding='utf8')
    actual.write_text('seed,count,rate\n0,22,0.5000000000001\n', encoding='utf8')
    assert replay.equivalent_csv(actual, expected)
    for row in ('0.0,22,0.5', '0,23,0.5', '0,22,0.500000000002'):
        actual.write_text('seed,count,rate\n'+row+'\n', encoding='utf8')
        assert not replay.equivalent_csv(actual, expected)


def test_extracted_coverage_artifact_allows_only_newlines(tmp_path):
    # Actual archive-derived presentation bytes, not a synthetic scientific value.
    name = 'tables/TABLE_COVERAGE.json'
    original = (replay.SOURCE / name).read_bytes()
    archive = tmp_path / 'coverage.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(name, original)
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(tmp_path / 'extracted')
    expected = tmp_path / 'extracted' / name
    actual = tmp_path / 'actual.json'
    for payload in (original.replace(b'\r\n', b'\n'),
                    original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')):
        actual.write_bytes(payload)
        assert replay.coverage_newline_equivalent(actual, expected, name)
        assert not replay.coverage_newline_equivalent(actual, expected, 'expected/TABLE_COVERAGE.json')
    data = json.loads(original)
    data['rows']['PRIMARY_WITNESSES.csv'] += 1
    actual.write_text(json.dumps(data, indent=2), encoding='utf8')
    assert not replay.coverage_newline_equivalent(actual, expected, name)
    actual.write_bytes(original.replace(b'22', b'23'))
    assert not replay.coverage_newline_equivalent(actual, expected, name)
    actual.write_bytes(original.replace(b'  ', b' ', 1))
    assert not replay.coverage_newline_equivalent(actual, expected, name)
    assert expected.read_bytes() == original


def test_svg_roundoff_never_changes_text_structure_or_style(tmp_path):
    expected, actual = tmp_path / 'expected.svg', tmp_path / 'actual.svg'
    template = '<svg><circle cy="{}" fill="blue"/><text>uncertain</text></svg>'
    expected.write_text(template.format('200.0'), encoding='utf8')
    actual.write_text(template.format('200.00000000000006'), encoding='utf8')
    result = replay.svg_comparison(actual, expected)
    assert result['equivalent'] and 0 < result['max_coordinate_difference'] < 1e-12
    for content in (template.format('200.000000000002'), template.format('200.0').replace('uncertain', 'established'),
                    template.format('200.0').replace('blue', 'red'), '<svg/>'):
        actual.write_text(content, encoding='utf8')
        assert not replay.svg_comparison(actual, expected)['equivalent']


def test_target_answers_do_not_enter_other_question_controls(analyzer, example):
    design, index = example
    query = 'FHV1-FINAL-0017-d0-08'
    original = next(r for r in analyzer.primary(*design, index) if r['seed'] == 0 and r['query_id'] == query)
    assert original['both_matched']
    modified = copy.deepcopy(index)
    for origin in range(1, 8):
        modified[f'o{origin}/INV_PAIR_NLL_s0', query].update(prediction=0, valid=True)
    result = next(r for r in analyzer.primary(*design, modified) if r['seed'] == 0 and r['query_id'] == query)
    assert result['both_eligible'] and not result['both_matched']


@pytest.mark.parametrize('field,failing,surviving', [('same_pair', 'G2a', 'G2b'), ('cross_pair', 'G2b', 'G2a')])
def test_both_logical_control_sets_are_required(analyzer, example, field, failing, surviving):
    design, index = example
    query = 'FHV1-FINAL-0017-d0-08'
    control = design[2]['gates']['targets'][query][field][0]
    modified = copy.deepcopy(index)
    modified['o1/INV_PAIR_NLL_s0', control]['valid'] = False
    result = next(r for r in analyzer.primary(*design, modified) if r['seed'] == 0 and r['query_id'] == query)
    assert not result[failing + '_eligible'] and result[surviving + '_eligible']
    assert not result['both_matched']


def test_exact_classification_is_distinct_from_numerical_tolerance():
    assert replay.equivalent({'p': .5}, {'p': math.nextafter(.5, 1)})
    assert not replay.equivalent({'p': .5}, {'p': .5 + 2e-12})
    assert not replay.equivalent([True, 22], [False, 22])
    assert not replay.equivalent([True, 22], [True, 23])
    assert not replay.equivalent(1, 1.0)
    assert not replay.equivalent(float('nan'), float('nan'))


def test_subthreshold_repeat_difference_is_recorded_not_called_exact():
    tree = ast.parse((replay.SOURCE / 'source/panels.py').read_text(encoding='utf8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('current_rows', 'compare')]
    namespace = {'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-repeat-functions>', 'exec'), namespace)
    row = dict(context='c', query_id='q', logps=[-.5, -1.5], prediction=0, valid=True,
               tokens_sha256='fixed', finite=True, tie=False, argmax_in_labels=True)
    original = dict(physical=[row], aliases={}, contexts={'c': {}})
    changed = copy.deepcopy(original)
    changed['physical'][0]['logps'][0] += 1e-7
    result = namespace['compare'](original, changed)
    assert result['status'] == 'PASS' and result['max_logp_delta'] > 0
    assert len(result['all_discrepancies']) == 1 and not result['failed_rows']


@pytest.mark.parametrize('body', ['import torch', 'import transformers', 'import operator_adapter',
                                 'import socket; socket.socket()',
                                 "import sys; sys.audit('socket.getaddrinfo', 'example.invalid', 80, 0, 0, 0)"])
def test_offline_child_has_no_model_private_operator_or_socket_access(tmp_path, body):
    script = tmp_path / 'forbidden.py'
    script.write_text(body, encoding='utf8')
    with pytest.raises(subprocess.CalledProcessError):
        replay.run_offline(script, [], tmp_path)


def test_offline_child_allows_local_platform_metadata(tmp_path):
    script = tmp_path / 'local_metadata.py'
    script.write_text("import platform; assert isinstance(platform.uname().node, str); print('LOCAL_METADATA_OK')", encoding='utf8')
    result = replay.run_offline(script, [], tmp_path)
    assert result.stdout.strip() == 'LOCAL_METADATA_OK'


def test_changed_compact_artifact_fails_before_replay(tmp_path):
    (tmp_path / 'value.json').write_text('{}', encoding='utf8')
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(dict(schema='REVIEW_FRESH_HISTORY_V1',
        files=[dict(path='value.json', bytes=2, sha256='0' * 64)])), encoding='utf8')
    with pytest.raises(ValueError, match='Scientific artifact changed'):
        replay.verify(tmp_path)


def test_existing_output_is_never_replaced(tmp_path):
    with pytest.raises(ValueError, match='fresh output'):
        replay.export(tmp_path)


def test_nested_manifest_cannot_hide_an_undeclared_file(tmp_path):
    generated = tmp_path / 'generated'
    generated.mkdir()
    payload = b'{"files": {}}'
    (generated / 'DATA_FREEZE.json').write_bytes(payload)
    (tmp_path / 'MANIFEST.json').write_text(json.dumps({
        'schema': 'REVIEW_FRESH_HISTORY_V1', 'files': [{
            'path': 'generated/DATA_FREEZE.json', 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}]}), encoding='utf8')
    assert replay.verify(tmp_path) == 1
    (generated / 'MANIFEST.json').write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='missing or undeclared'):
        replay.verify(tmp_path)
