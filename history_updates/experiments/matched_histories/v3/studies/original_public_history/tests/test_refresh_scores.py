"""Guards for the recovered score projection and its original scientific decoder."""
import copy
import gzip
import hashlib
import io
import json
import math
import runpy

import pytest

from repro import question_blind_repair as qbr
from repro import refresh_scores as scores


@pytest.fixture
def panel_and_reader():
    questions = [{'query_id': 'q' + str(i)} for i in range(34)]
    physical = [dict(context='NATIVE_FINAL', query_id=q['query_id'], physical_batch=i,
                     logps=[math.log(.7), math.log(.2)], finite=True, tie=False,
                     argmax_in_labels=True, valid=True, prediction=0, answer_mass=.9,
                     master_unchanged=True, batch_size=1, physical_slot=0)
                for i, q in enumerate(questions)]
    panel = dict(root_id='synthetic', seed=0, actor='synthetic', physical=physical,
                 tokens={'NATIVE_FINAL': {q['query_id']: [1, 2] for q in questions}}, aliases={})
    reader = runpy.run_path(str(qbr.SOURCE / 'supplied/verification/tools/replay.py'))
    return panel, questions, reader


def test_projection_validates_all_synthetic_score_rows(panel_and_reader):
    assert scores.validate_panel(*panel_and_reader) == 34


@pytest.mark.parametrize('change, message', [
    ('duplicate', 'Duplicate score row'), ('missing', 'Score/query/batch'),
    ('mass', 'Answer-mass binding'), ('prediction', 'Hard-answer binding'),
    ('token', 'Score/token inventory'), ('master', 'Single-query score isolation'),
    ('batch', 'Score/query/batch'),
])
def test_projection_rejects_broken_bindings(panel_and_reader, change, message):
    panel, questions, reader = panel_and_reader
    if change == 'duplicate':
        panel['physical'].append(copy.deepcopy(panel['physical'][0]))
    elif change == 'missing':
        panel['physical'].pop()
    elif change == 'mass':
        panel['physical'][0]['answer_mass'] += 1e-8
    elif change == 'prediction':
        panel['physical'][0]['prediction'] = 1
    elif change == 'token':
        panel['tokens']['NATIVE_FINAL'].pop('q0')
    elif change == 'master':
        panel['physical'][0]['master_unchanged'] = False
    elif change == 'batch':
        panel['physical'][0]['physical_batch'] = 1
    with pytest.raises(ValueError, match=message):
        scores.validate_panel(panel, questions, reader)


def test_forward_filter_preserves_reverse_data_and_input_object(panel_and_reader):
    panel, _, _ = panel_and_reader
    panel['physical'].append(dict(panel['physical'][0], context='RB/o0'))
    panel['tokens']['RB/o0'] = {'q0': [2, 1]}
    panel['aliases']['RC/o0'] = 'RB/o0'
    original = copy.deepcopy(panel)
    assert len(scores.select_panel(panel, False)['physical']) == 34
    assert 'RB/o0' not in scores.select_panel(panel, False)['tokens']
    assert scores.select_panel(panel, False)['aliases'] == {}
    assert len(scores.select_panel(panel, True)['physical']) == 35
    assert scores.select_panel(panel, True)['tokens']['RB/o0'] == {'q0': [2, 1]}
    assert panel == original


def test_projection_rejects_undeclared_native_custody(panel_and_reader):
    panel, _, _ = panel_and_reader
    panel['contexts'] = {}
    with pytest.raises(ValueError, match='Unexpected score projection'):
        scores.select_panel(panel, False)


def test_record_loader_checks_decoded_hash_count_and_stored_bytes(tmp_path):
    decoded = b'{"value":1}\n'
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, filename='', mode='wb', mtime=0) as zipped:
        zipped.write(decoded)
    raw = stream.getvalue()
    path = tmp_path / 'part.jsonl.gz'
    path.write_bytes(raw)
    binding = dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   decoded_bytes=len(decoded), decoded_sha256=hashlib.sha256(decoded).hexdigest(), records=1)
    assert scores.records(tmp_path, binding) == [{'value': 1}]
    with pytest.raises(ValueError, match='record count'):
        scores.records(tmp_path, dict(binding, records=2))
    with pytest.raises(ValueError, match='Decoded payload hash'):
        scores.records(tmp_path, dict(binding, decoded_sha256='0' * 64))
    path.write_bytes(raw + b'x')
    with pytest.raises(ValueError, match='Stored payload hash'):
        scores.records(tmp_path, binding)


def test_original_primary_gate_cannot_be_overridden():
    computed = {'targets': [{'query_id': 'q0', 'W': True}], 'families': []}
    with pytest.raises(ValueError, match='Original baseline classification'):
        scores.accumulate_primary(computed, 'r0', 0, {('r0', 0, 'q0'): {'both_matched': 'False'}},
                                  {}, set(), set())


def test_table_serializer_preserves_original_csv_bytes(tmp_path):
    reader = runpy.run_path(str(qbr.SOURCE / 'supplied/verification/tools/replay.py'))
    values = [dict(flag=True, value=3, histories=[1, 7], text='a,b'),
              dict(flag=False, value=-2, histories=[], text='line\nnext')]
    path = tmp_path / 'table.csv'
    reader['table'](path, values)
    assert scores.table_bytes(values) == path.read_bytes()


def test_gemma_provenance_scope_and_activation_membership():
    base = qbr.ROOT / 'artifacts/probe_weights/m1_relation/provenance'
    provenance = qbr.load_json(base / 'training_provenance.json')
    assert set(provenance['models']) == {'gemma'}
    record = provenance['models']['gemma']
    assert record['trained_rows'] == 107368
    assert record['encoder_revision'] == '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    assert not record['row_level_training_membership_available']
    manifest = qbr.load_json(base / 'probe_membership.json')
    assert manifest['model_binding'] == ['gemma']
    assert not manifest['minilm_common_training_membership_recovered']
    assert sum(r['rows'] for r in manifest['files']) == 2300
    for binding in manifest['files']:
        path = base / binding['path']
        assert qbr.sha256(path) == binding['sha256']
        rows = qbr.rows(path)
        assert len(rows) == len({r['item_id'] for r in rows}) == binding['rows']
        assert all('label' not in row and 'cell_role' not in row for row in rows)
