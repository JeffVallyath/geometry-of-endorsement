"""Saved-row reconstruction, semantic corruption guards and paired endpoints."""
import copy
import pytest
from repro import update_methods as m
from repro.common import load_json


@pytest.fixture(scope='module')
def result():
    return m.reconstructed()[0]


def test_saved_primary_contrasts_and_absolute_performance(result):
    assert result == load_json(m.BASE / 'results.json')
    assert result['score_rows'] == 208896 and result['telemetry_rows'] == 6144
    assert [(r['effect'], r['lower'], r['upper']) for r in result['contrasts']] == [
        (.15625, 0., .3046875), (.1328125, -.0078125, .2734375),
        (.3125, .203125, .4375), (.3984375, .2890625, .5078125)]
    for row in result['contrasts']:
        assert len(row['per_root']) == 64 and set(row['per_seed']) == {'0', '1'}
        assert row['effect'] == sum(r['effect'] for r in row['per_seed'].values())/2
    assert all(r['rate'] < .5 for r in result['absolute_rates'])
    assert [r['fallback'] for r in result['fallback']] == [192, 192]


@pytest.mark.parametrize('field', ['correct', 'gold', 'source_gold', 'prediction', 'query_id', 'master_unchanged'])
def test_raw_score_corruption_rejected(field):
    row=next(m.rows(m.BASE/'supplied/completed/raw/gemma__ORIGIN__SOURCE.jsonl.gz'))
    reference=next(r for r in m.rows(m.INPUTS/'cases/fixed/evaluation.jsonl')
                   if r['actor']=='gemma' and r['input_id']==row['input_id'] and r['query_id']==row['query_id'])
    origin=next(r for r in m.rows(m.INPUTS/'cases/fixed/origin.jsonl') if r['root_id']==row['root_id'])
    m.validate_score(row,reference,origin,{})
    bad=copy.deepcopy(row)
    bad[field]='not-a-query' if field=='query_id' else not row[field] if isinstance(row[field],bool) else 1-row[field]
    with pytest.raises(ValueError):
        m.validate_score(bad,reference,origin,{})


def test_bootstrap_retains_half_root_effects_and_zero_boundaries():
    assert m.interval([0.]*64)['lower']==0.
    assert m.interval([.5]*64)['effect']==.5
