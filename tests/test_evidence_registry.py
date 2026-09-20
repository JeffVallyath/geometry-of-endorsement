"""Regression checks on the saved scientific registry and actual document text."""
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import pytest

# Environment overrides permit testing an isolated integration candidate.
ROOT = Path(os.environ.get('EVIDENCE_REPOSITORY', Path(__file__).resolve().parents[1]))
REGISTRY = Path(os.environ.get('EVIDENCE_REGISTRY', ROOT / 'reproducibility/evidence.json'))
if os.environ.get('EVIDENCE_MODULE'):
    spec = importlib.util.spec_from_file_location('registry_evidence', os.environ['EVIDENCE_MODULE'])
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev
    spec.loader.exec_module(ev)
else:
    from repro import evidence as ev


def document(identity):
    registry = ev.load(REGISTRY)
    doc = next(d for d in registry['documents'] if d['id'] == identity)
    return (ROOT / doc['path']).read_text(encoding='utf8')


def changed(identity, old, new):
    original = document(identity)
    pattern = r'\s+'.join(re.escape(word) for word in old.split())
    mutated, count = re.subn(pattern, lambda _: new, original, count=1)
    assert count == 1, f'Update the corruption target: {old!r}'
    assert mutated != original
    return original, mutated


def rules(identity, text):
    return {f['rule'] for f in ev.check(ROOT, REGISTRY, document=identity, overrides={identity: text}).errors}


def test_actual_registry_passes():
    report = ev.check(ROOT, REGISTRY)
    assert report.ok(), report.errors
    assert sum(c['bound'] for c in report.coverage) == 509
    assert sum(c['bound'] for c in report.coverage if c['document'] == 'captions') == 21
    assert sum(c['bound'] for c in report.coverage if c['document'] == 'results') == 393
    captions = {c['unit']: c for c in report.coverage if c['document'] == 'caption_details'}
    assert set(captions) >= {'figure6', 'figure7', 'supplementary_s1', 'supplementary_s2', 'supplementary_s3'}
    assert captions['figure7']['bound'] == 7  # V10 design and primary interval bindings.
    assert captions['figure8']['bound'] == 2  # Crossover layer counts; prose counts have source predicates.
    assert captions['figure9']['bound'] == 7  # Selected extension matching and strict direct-readout checks.
    assert captions['fresh_confirmation']['bound'] == 7
    assert captions['fresh_utility']['bound'] == 5
    assert captions['supplementary_s3']['bound'] == 5  # Preserved V6 census bindings.
    assert not report.pending


def test_consequence_caption_requires_failed_strong_target_qualification():
    _, bad = changed('caption_details', 'not perfect hard-answer accuracy or a full strong-target pass',
                     'perfect hard-answer accuracy and a full strong-target pass')
    assert 'material-qualification-missing' in rules('caption_details', bad)


def test_source_history_caption_requires_conditional_denominator():
    _, bad = changed('caption_details', 'counts\nuse atomic-perfect roots as their denominator; the bars use all roots',
                     'counts\nuse all roots as their denominator; the bars use all roots')
    assert 'material-qualification-missing' in rules('caption_details', bad)


def test_changed_estimate_is_rejected():
    _, bad = changed('results', '0.9926', '0.9927')
    assert 'number-undeclared' in rules('results', bad)


@pytest.mark.parametrize('old,new', [
    ('65/98 questions', '65/65 questions'),
    ('primary label-first parser found 3/98', 'primary label-first parser found 2/98'),
    ('union-parser sensitivity, 2/98', 'union-parser sensitivity, 3/98'),
    ('counterfactual benchmark\nassignments, not real-world biographical claims',
     'verified real-world biographical claims'),
    ('Neither repeated\nbenchmark case contained a qualifying history-dependent question', 'Both repeated benchmark cases contained a qualifying history-dependent question'),
    ('Without chronological context, 170/272 answers were correct, versus 94/272 with',
     'Without chronological context, 94/272 answers were correct, versus 170/272 with'),
    ('not 272 independent facts', '272 independent facts'),
    ('obsolete record remained explicitly present earlier', 'obsolete record was absent'),
    ('Conventional parameter-editor validation\nis pending', 'Conventional parameter-editor validation is complete'),
])
def test_external_in_context_claim_boundaries(old, new):
    _, bad = changed('results', old, new)
    assert 'material-qualification-missing' in rules('results', bad)


def test_external_counts_are_bound_to_saved_response_replay():
    registry = ev.load(REGISTRY)
    results = next(d for d in registry['documents'] if d['id'] == 'results')
    external = next(u for u in results['units'] if u['id'] == 'external_in_context')
    assert len(external['numbers']) == 10
    assert external['literals'] == []
    assert all(n['path'].startswith('qwen_public.') for n in external['numbers'])
    assert 'saved responses and both frozen parsers' in external['provenance']


def test_wrong_interval_type_is_rejected():
    _, bad = changed('caption_details', 'are not bootstrap intervals.', 'are bootstrap intervals.')
    assert 'material-qualification-missing' in rules('caption_details', bad)


def test_random_item_score_is_not_random_activation_direction():
    text = document('caption_details')
    bad, count = re.subn(r'Its implementation hashes each\s+item identifier to a scalar in the range 0 to 1 and scores the item with that\s+number\. No vector in activation space is ever sampled',
                        'Its implementation samples a random activation direction and rescales scores to the range 0 to 1. A vector in activation space is sampled', text)
    assert count == 1
    assert Counter(ev.number_tokens(text)) == Counter(ev.number_tokens(bad))
    assert 'material-qualification-missing' in rules('caption_details', bad)


@pytest.mark.parametrize('old,new', [
    ('selection-split standard deviations', 'training-split standard deviations'),
    ('(k+1)/(B+1)', '(k+1)/(k+1)'),
])
def test_same_number_counts_do_not_hide_wrong_denominator(old, new):
    original, bad = changed('caption_details', old, new)
    assert Counter(ev.number_tokens(original)) == Counter(ev.number_tokens(bad))
    assert {'material-qualification-missing', 'unsupported-scientific-assertion'} & rules('caption_details', bad)


@pytest.mark.parametrize('assertion', ev.load(REGISTRY)['source_checks'])
def test_background_scientific_predicates_remain_bound(assertion):
    registry = ev.load(REGISTRY)
    artifacts = {key: ev.load(ROOT / value) for key, value in registry['sources'].items()}
    assert ev.check_source(assertion, artifacts), assertion['id']


def test_actual_selector_status_is_bound_to_saved_evidence():
    registry = ev.load(REGISTRY)
    artifacts = {key: ev.load(ROOT / value) for key, value in registry['sources'].items()}
    assertion = next(c for c in registry['source_checks'] if c['id'] == 'selector-relative-not-absolute-0')
    artifacts['selector']['records'][0]['selected_values']['/M1']['status'] = 'ESTABLISHED'
    assert not ev.check_source(assertion, artifacts)


def test_missing_background_source_field_is_reported(tmp_path):
    registry = ev.load(REGISTRY)
    registry['source_checks'][0]['path'] = ['missing_source', 'missing_field']
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(registry), encoding='utf8')
    report = ev.check(ROOT, path)
    assert 'binding-malformed' in {row['rule'] for row in report.errors}


def test_literal_source_key_path_and_percent_display():
    assert ev.resolve({'a': [{'x.y': {'/z': {'': 4}}}]}, ['a', 0, 'x.y', '/z', '']) == 4
    assert ev.render(.99375, {'decimals': 3, 'percent': True}) == '99.375'


def test_generic_cross_study_model_scope_is_rejected():
    original = document('caption_details')
    bad, count = re.subn(r'Later causal and\s+reliability results do not replicate symmetrically across studies\. Qualification\s+failures and bounded positive outcomes differ by experiment;',
                        'Later causal and reliability tests give a narrower Llama finding; Gemma misses the required conjunction;', original)
    assert count == 1
    assert 'material-qualification-missing' in rules('caption_details', bad)


def test_relative_coverage_is_not_absolute_success():
    _, bad = changed('caption_details', 'This relative improvement does not\nestablish a pass of the full absolute joint-control requirements.',
                     'This relative improvement establishes a pass of the full absolute joint-control requirements.')
    assert 'material-qualification-missing' in rules('caption_details', bad)


def test_reader_crossing_zero_is_not_equivalence():
    _, bad = changed('caption_details', 'which is not evidence of equivalence', 'which establishes equivalence')
    assert 'material-qualification-missing' in rules('caption_details', bad)


@pytest.mark.parametrize('old,new', [
    ('0.9549%', '0.9550%'),
    ('-1.7447 percentage points', '1.7447 percentage points'),
    ('At equal 50% retention', 'At equal 75% retention'),
    ('V10 remains the original primary study;', 'The fresh study replaces the original primary study;'),
    ('The practical advantage remains uncertain', 'The practical advantage is established'),
    ('Thirteen full-criterion observations have a withheld target', 'Twelve full-criterion observations have a withheld target'),
    ('Both validators accept the corresponding case/seed/format', 'Both validators reject the corresponding case/seed/format'),
    ('These are post-hoc diagnostics, separate from', 'These are prospective primary criteria, replacing'),
])
def test_fresh_confirmation_utility_and_post_hoc_scope_remain_bound(old, new):
    _, bad = changed('results', old, new)
    assert {'number-undeclared', 'material-qualification-missing'} & rules('results', bad)


def test_fresh_claim_guards_allow_equivalent_readable_wording():
    text = document('results')
    text = text.replace('fixed history matching', 'specified history matching')
    text = text.replace('advantage remains uncertain because its interval includes zero',
                        'advantage remains inconclusive because its interval spans zero')
    report = ev.check(ROOT, REGISTRY, document='results', overrides={'results': text})
    assert report.ok(), report.errors


@pytest.mark.parametrize('old,new', [
    ('| Deletion | 129 | 38 |', '| Deletion | 128 | 38 |'),
    ('75 | 0 |', '76 | 0 |'),
    ('failed 4 of 5 admission', 'failed 3 of 5 admission'),
    ('128 fresh final cases', '127 fresh final cases'),
    ('other upstream influences unresolved', 'no upstream influence possible'),
])
def test_preupdate_counts_and_fixed_procedure_scope(old, new):
    _, bad = changed('results', old, new)
    assert {'number-undeclared', 'number-missing', 'material-qualification-missing'} & rules('results', bad)



@pytest.mark.parametrize('old,new', [
    ('430/653', '431/653'),
    ('| Same-pair logical consistency | 188 | 53/64 |',
     '| Same-pair logical consistency | 189 | 53/64 |'),
    ('other\ntwo logically sufficient compositions of the same relation pair',
     'two arbitrary questions about unrelated relations'),
    ('concentrated in learned-update conditions', 'equally broad under all update conditions'),
    ('Lexical content, clause\nposition and which relation was previously different remain unmatched',
     'All possible confounds have been eliminated'),
    ('earlier analysis of these same outputs', 'independent replication'),
])
def test_post_hoc_audit_claims_and_limits_remain_checked(old, new):
    _, bad = changed('results', old, new)
    assert {'number-undeclared', 'material-qualification-missing'} & rules('results', bad)
