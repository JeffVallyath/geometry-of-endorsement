"""Preserve selected-case limitations, harms, exact gates and denominators."""
import copy

import pytest

from repro import completed_followups as mechanism


def test_all_score_rows_direct_denominators_and_adverse_effects():
    result = mechanism.check()
    carrier = result['studies']['carrier']
    mediation = result['studies']['mediation']
    assert carrier['measured_rows'] == 2172
    assert carrier['all_direct_rows_checked'] == 1448
    assert carrier['reported_direct_preservation_rows'] == 1416
    assert carrier['baseline_reference_rows_outside_reported_denominator'] == 32
    assert carrier['new_joint_errors'] == 179
    assert mediation['measured_rows'] == 1422
    assert mediation['reported_direct_preservation_rows'] == 948
    assert mediation['new_joint_errors'] == 61
    assert all(row['direct_failures'] == row['invalid_rows'] == 0 for row in result['studies'].values())
    assert not result['inference'] and not result['active_outcomes_read']


@pytest.mark.parametrize('field,value', [('valid', 'False'), ('correct', 'False'), ('p_correct', '.949'),
                                         ('candidate_mass', '.899'), ('direct_recipient_probability_drift', '.021')])
def test_direct_preservation_keeps_every_original_requirement(field, value):
    row = dict(valid='True', correct='True', p_correct='.95', candidate_mass='.90',
               direct_recipient_probability_drift='.02')
    assert mechanism.direct_preservation(row)
    changed = copy.deepcopy(row)
    changed[field] = value
    assert not mechanism.direct_preservation(changed)


def test_compact_carrier_search_and_reserved_pair_failures():
    _, saved = mechanism.read_records(mechanism.SOURCE, 'carrier')
    rows = saved('ALL_CONDITION_SUMMARIES.json')
    discovery = [row for row in rows if row['root'] == 'SSC1-FINAL-0035']
    p1 = next(row for row in discovery if row['stage'] == 'A' and row['name'] == 'P1')
    assert p1['hard_transfers'] == 4 and p1['direct_passes'] == 8 and p1['complete_transfer']
    sites = [row for row in discovery if row['stage'] == 'B_scan' and row['name'].endswith('_ONLY')]
    assert len(sites) == 64 and not any(row['complete_transfer'] for row in sites)
    reserved = {row['root']: row for row in rows if row['root'] != 'SSC1-FINAL-0035' and row['name'] == 'C'}
    assert {root: row['hard_transfers'] for root, row in reserved.items()} == {'SSC1-FINAL-0026': 2, 'SSC1-FINAL-0054': 1}
    assert all(row['hard_opportunities'] == 4 and row['direct_passes'] == 8 and not row['complete_transfer'] for row in reserved.values())


def test_reader_threshold_and_selection_failure_stay_visible():
    _, saved = mechanism.read_records(mechanism.SOURCE, 'mediation')
    audit = saved('NUMERICAL_AUDIT.json')
    writer = audit['selected_writer']
    reader = audit['selected_reader']
    assert writer['name'] == 'L16_ATTN_ONLY' and writer['direct_passes'] == 8
    assert reader['selected'] == 'TOP8' and reader['status'] == 'NO_QUALIFIED_READER_SET'
    assert reader['training_outcomes']['TOP8']['mediation_score'] == 0.7777574880208111
    assert reader['training_outcomes']['TOP8']['mediation_score'] < .8
    assert reader['frozen_before_coded_and_reverse']
    assert not any(row['qualified_training'] for row in reader['training_outcomes'].values())
