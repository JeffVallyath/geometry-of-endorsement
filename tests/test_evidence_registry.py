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
    assert old in original, f'Update the corruption target: {old!r}'
    mutated = original.replace(old, new, 1)
    assert mutated != original
    return original, mutated


def rules(identity, text):
    return {f['rule'] for f in ev.check(ROOT, REGISTRY, document=identity, overrides={identity: text}).errors}


def test_actual_registry_passes():
    report = ev.check(ROOT, REGISTRY)
    assert report.ok(), report.errors
    assert sum(c['bound'] for c in report.coverage) == 340
    assert sum(c['bound'] for c in report.coverage if c['document'] == 'results') == 278
    assert {c['unit'] for c in report.coverage if c['document'] == 'captions'} >= {'figure6', 'figure7', 'supplementary_s1', 'supplementary_s2'}
    assert not report.pending


def test_consequence_caption_requires_failed_strong_target_qualification():
    _, bad = changed('captions', 'not perfect hard-answer accuracy or a full strong-target pass',
                     'perfect hard-answer accuracy and a full strong-target pass')
    assert 'material-qualification-missing' in rules('captions', bad)


def test_source_history_caption_requires_conditional_denominator():
    _, bad = changed('captions', 'counts\nuse atomic-perfect roots as their denominator; the bars use all roots',
                     'counts\nuse all roots as their denominator; the bars use all roots')
    assert 'material-qualification-missing' in rules('captions', bad)


def test_changed_estimate_is_rejected():
    _, bad = changed('results', '0.9926', '0.9927')
    assert 'number-undeclared' in rules('results', bad)


def test_wrong_interval_type_is_rejected():
    _, bad = changed('captions', 'are not bootstrap intervals.', 'are bootstrap intervals.')
    assert 'material-qualification-missing' in rules('captions', bad)


def test_random_item_score_is_not_random_activation_direction():
    text = document('captions')
    bad, count = re.subn(r'Its implementation hashes each\s+item identifier to a scalar in the range 0 to 1 and scores the item with that\s+number\. No vector in activation space is ever sampled',
                        'Its implementation samples a random activation direction and rescales scores to the range 0 to 1. A vector in activation space is sampled', text)
    assert count == 1
    assert Counter(ev.number_tokens(text)) == Counter(ev.number_tokens(bad))
    assert 'material-qualification-missing' in rules('captions', bad)


@pytest.mark.parametrize('old,new', [
    ('selection-split standard deviations', 'training-split standard deviations'),
    ('(k+1)/(B+1)', '(k+1)/(k+1)'),
])
def test_same_number_counts_do_not_hide_wrong_denominator(old, new):
    original, bad = changed('captions', old, new)
    assert Counter(ev.number_tokens(original)) == Counter(ev.number_tokens(bad))
    assert {'material-qualification-missing', 'unsupported-scientific-assertion'} & rules('captions', bad)


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
    original = document('captions')
    bad, count = re.subn(r'Later causal and\s+reliability results do not replicate symmetrically across studies\. Qualification\s+failures and bounded positive outcomes differ by experiment;',
                        'Later causal and reliability tests give a narrower Llama finding; Gemma misses the required conjunction;', original)
    assert count == 1
    assert 'material-qualification-missing' in rules('captions', bad)


def test_relative_coverage_is_not_absolute_success():
    _, bad = changed('captions', 'This relative improvement does not\nestablish a pass of the full absolute joint-control requirements.',
                     'This relative improvement establishes a pass of the full absolute joint-control requirements.')
    assert 'material-qualification-missing' in rules('captions', bad)


def test_reader_crossing_zero_is_not_equivalence():
    _, bad = changed('captions', 'which is not evidence of equivalence', 'which establishes equivalence')
    assert 'material-qualification-missing' in rules('captions', bad)
