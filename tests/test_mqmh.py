"""Regression checks for the public multi-hop saved-response interface."""
import copy
from decimal import Decimal, ROUND_HALF_UP
import json

import pytest

from repro import mqmh
from repro.release_inputs import payload, verify


@pytest.fixture(scope='module')
def saved():
    binding = verify(mqmh.SOURCE, 'mqmh')
    data = {r['member']: json.loads(payload(mqmh.SOURCE, r)) for r in binding['files']}
    return data['cohort.json'], {m: data[m+'_responses.json'] for m in mqmh.MODELS}


@pytest.fixture(scope='module')
def reconstructed(saved):
    return mqmh.analyze(*saved)


def test_original_primary_and_strict_results_reconstruct(saved, reconstructed):
    summary, targets, example = reconstructed
    expected = json.loads((mqmh.SOURCE/'RESULTS.json').read_text(encoding='utf-8'))
    mqmh.assert_equivalent(summary, expected)
    assert len(targets) == 768
    for model, guard, qualified, strong, cross in [('qwen', 242, 94, 32, 11), ('gemma', 279, 101, 65, 21)]:
        row = summary[model]['primary']
        assert (row['subjects'], row['target_phrasings'], row['long_chain_phrasings']) == (128, 384, 195)
        assert (row['common_guard'], row['qualified'], row['long_chain_qualified']) == (guard, qualified, strong)
        assert row['cross_phrasing_long_chain_subjects'] == cross
        assert summary[model]['repeats']['cases'] == summary[model]['repeats']['stable'] == 12
        assert summary[model]['repeats']['new_subjects'] == 0
    assert summary['gemma']['primary']['invalid_contribution'] == 0


def test_same_qualified_question_and_all_controls_in_worked_example(reconstructed):
    _, targets, example = reconstructed
    assert example['case']['current_terminal'] == 'Q23436'
    assert example['case']['old_terminal'] == 'Q83241'
    assert next(q['question'] for q in example['case']['queries'] if q['query_id']=='J2') == (
        "What place did the person who founded William Levada's religion work in?")
    for model in mqmh.MODELS:
        case = example['models'][model]
        assert case['target']['qualified'] and case['target']['long_chain_qualified'] and case['repeat_stable']
        assert all(r['correct'] for r in case['responses'] if r['query_id'] in ('F0','F1','F2','Fdecoy','K'))
        assert all(r['correct'] for r in case['responses'] if r['query_id']=='J2' and r['arm'] in ('C','C+'))
    qwen_j0 = next(q for q in targets if q['model']=='qwen' and q['case_id']==1617 and q['query_id']=='J0')
    assert not qwen_j0['references'] and not qwen_j0['qualified']
    assert qwen_j0['relevance_difference'] == 0


def test_cli_outputs_are_reproducible_and_non_overwriting(tmp_path):
    output = tmp_path/'new output'
    mqmh.run(output)
    assert (output/'target_counts.csv').read_text(encoding='utf-8').count('\n') == 769
    assert json.loads((output/'CHECKS.json').read_text())['status'] == 'PASS'
    with pytest.raises(ValueError, match='new output'):
        mqmh.run(output)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'parser'])
def test_incomplete_duplicate_and_changed_scoring_fail(saved, mutation):
    cohort, responses = saved
    cases = {c['case_id']:c for c in cohort['cases']}
    first = cohort['case_ids'][0]
    rows = copy.deepcopy([r for r in responses['qwen']['primary'] if r['case_id']==first])
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(rows[0])
    else:
        rows[0]['entity_id'] = 'Q_invalid'
    with pytest.raises(ValueError):
        mqmh.endpoints(cases[first], mqmh.score(rows, cases))


def test_models_cannot_be_pooled(saved):
    cohort, responses = saved
    with pytest.raises(ValueError, match='two models'):
        mqmh.analyze(cohort, dict(responses, pooled=responses['qwen']))


def test_aliases_are_case_sensitive_and_ambiguity_is_invalid():
    case = dict(registry={'BE':'Q1', 'be':'Q2', 'Answer: BE':'Q3'},
                canonical_labels={'Q1':'BE','Q2':'be'}, decoy_id='Q4', decoy_label='decoy')
    row = dict(text='BE', truncated=False, control_tokens_valid=True)
    assert mqmh.parse_response(row, case) == ('Q1','Q1')
    assert mqmh.parse_response(dict(row, text='be'), case) == ('Q2','Q2')
    assert mqmh.parse_response(dict(row, text='Answer: BE'), case) == (None,None)
    assert mqmh.parse_response(dict(row, truncated=True), case) == (None,None)
    assert mqmh.parse_response(dict(row, control_tokens_valid=False), case) == (None,None)


def test_displayed_primary_rows_match_reconstructed_values(reconstructed):
    summary, _, _ = reconstructed
    readme = (mqmh.ROOT/'README.md').read_text(encoding='utf-8')
    guide = (mqmh.ROOT/'docs/06_mqmh_public_multihop.md').read_text(encoding='utf-8')
    percent = lambda x: format((Decimal(str(x))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f')
    for model, name in [('qwen','Qwen2.5-7B-Instruct'), ('gemma','Gemma-2-9B-it')]:
        row = summary[model]['primary']
        assert (f"| {name} | {row['qualified']}/384 ({row['qualified_rate']*100:.1f}%) | "
                f"+{row['relevance_difference']*100:.1f} pp | {row['long_chain_qualified']}/195 |") in readme
        lo, hi = map(percent, row['qualified_interval'])
        dlo, dhi = map(percent, row['relevance_interval'])
        assert (f"| {name} | {row['common_guard']}/384 | {row['qualified']}/384 = "
                f"{percent(row['qualified_rate'])}% [{lo}, {hi}]% | +{percent(row['relevance_difference'])} pp "
                f"[+{dlo}, +{dhi}] pp | {row['long_chain_qualified']}/195 |") in guide
