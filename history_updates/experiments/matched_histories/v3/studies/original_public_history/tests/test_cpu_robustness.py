"""Saved-output audit replay, distinct controls, and exact scientific decisions."""
import copy
import json
import math

import pytest

from repro import cpu_robustness as audit


@pytest.fixture(scope='module')
def replay():
    return audit.reconstruct()


def test_every_saved_opportunity_and_headline_reconstructs(replay):
    assert replay['verification']['independent_boolean_flag_checks'] == 608256
    assert replay['verification']['independent_question_mapping_checks'] == 13824
    summary = replay['summary']
    assert summary == audit.load(audit.BASE / 'summary.json')
    assert summary['coded_opportunities'] == 13824
    assert summary['shared_designs'] == 64
    assert summary['original_witnesses'] == 653
    controls = summary['controls']
    assert controls['G0']['matched'] == 430
    for gate, matched, designs, learned, textual, cross, cross_designs in [
        ('G2a', 188, 53, 140, 48, 37, 25),
        ('G2b', 82, 34, 76, 6, 16, 15),
        ('G2ab', 64, 29, 60, 4, 13, 12),
    ]:
        assert controls[gate] == dict(matched=matched, designs=designs,
            learned=learned, textual=textual,
            crosscode_semantic_matched=cross, crosscode_designs=cross_designs)


def test_probability_roundoff_does_not_relax_scientific_equality():
    audit.compare_value([0.5], [math.nextafter(0.5, 1)], probabilities=True)
    with pytest.raises(ValueError, match='Probability mismatch'):
        audit.compare_value([0.5], [0.5 + 2e-12], probabilities=True)
    with pytest.raises(ValueError, match='Exact classification'):
        audit.compare_value([True, 188], [False, 188])
    with pytest.raises(ValueError, match='Exact classification'):
        audit.compare_value([True, 188], [True, 189])


def test_same_pair_and_cross_pair_controls_are_distinct_and_target_blind():
    condition, target = 'INV_PAIR_NLL_s0', 'target'
    questions = [target, 'alternate_target', 'direct', 'same1', 'same2', 'cross']
    decoded = {(context, q): (1, True, True, 0.99, 0.999)
               for context in ['NATIVE_FINAL', *[f'o{i}/{condition}' for i in range(8)]]
               for q in questions}
    def certify(values):
        return audit.certification(values, condition, target, ['direct'], ['direct'],
                                   ['same1', 'same2'], ['cross'])
    original = certify(decoded)
    assert all(original.values())
    for q in (target, 'alternate_target'):
        for i in range(1, 8):
            decoded[f'o{i}/{condition}', q] = (0, False, False, 0.0, 0.0)
    assert certify(decoded) == original
    for q, fails, survives in [('same1', 'G2a', 'G2b'), ('cross', 'G2b', 'G2a')]:
        changed = copy.deepcopy(decoded)
        changed[f'o7/{condition}', q] = (0, True, False, 0.01, 0.999)
        result = certify(changed)
        assert result['G1'] and result[survives] and not result[fails] and not result['G2ab']


def test_crosscode_requires_the_same_pair_and_direction():
    left = {'matched': [[0, 1]], 'joint': [(0,), (1,), (0,)]}
    assert audit.common_pairs(left, left) == [[0, 1]]
    assert audit.common_pairs(left, {'matched': [[0, 1]], 'joint': [(1,), (0,), (0,)]}) == []
    assert audit.common_pairs(left, {'matched': [[1, 2]], 'joint': left['joint']}) == []


def test_corrupt_audit_bytes_fail_before_scoring(tmp_path):
    (tmp_path / 'changed.json').write_text('{}', encoding='utf8')
    (tmp_path / 'MANIFEST.json').write_text(json.dumps({'files': [
        {'path': 'changed.json', 'bytes': 2, 'sha256': '0' * 64}
    ]}), encoding='utf8')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        audit.check_integrity(tmp_path)


def test_replay_never_reads_reported_original_results(monkeypatch):
    original = audit.load
    def guarded(path):
        assert 'expected' not in path.parts
        return original(path)
    monkeypatch.setattr(audit, 'load', guarded)
    assert audit.reconstruct()['verification']['status'] == 'PASS'


def test_export_refuses_existing_output(tmp_path):
    with pytest.raises(ValueError, match='fresh directory'):
        audit.export(tmp_path)
