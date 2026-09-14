"""Synthetic scientific evidence regressions; no model or network dependency."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from repro import evidence as ev


def rules(report):
    return {f['rule'] for f in report.findings}


def unit(**changes):
    value = {'id': 'estimate', 'heading_pattern': '^Estimate$',
             'numbers': [{'path': 'scores.mean', 'decimals': 2, 'headline': True}],
             'facts': [{'id': 'development-scope',
                        'description': 'Development data, not independent confirmation.',
                        'all_of': ['development', r'(?:not|rather than).{0,30}confirmatory'],
                        'source_evidence': [{'path': 'scores.confirmatory', 'equals': False}]}]}
    value.update(changes)
    return value


def inspect(text='Development estimate 0.25, not a confirmatory result.', contract=None, data=None, strict=True):
    report = ev.Report()
    ev.check_unit(text, contract or unit(), data or {'scores': {'mean': .25, 'confirmatory': False}}, report, strict=strict)
    return report


def test_bound_number_and_qualification():
    report = inspect()
    assert report.ok()
    assert report.coverage[0]['bound'] == 1


@pytest.mark.parametrize('text', [
    'Development estimate 0.25, not a confirmatory result.',
    'This development measurement is 0.25 rather than a confirmatory estimate.',
    'The 0.25 development result is not an independent confirmatory measurement.',
])
def test_equivalent_qualification_statements(text):
    assert inspect(text).ok()


def test_dropped_qualification_fails():
    assert 'material-qualification-missing' in rules(inspect('Development estimate 0.25.'))


def test_evidence_cannot_be_repaired_by_rewording():
    report = inspect(data={'scores': {'mean': .25, 'confirmatory': True}})
    assert 'scientific-evidence-mismatch' in rules(report)


@pytest.mark.parametrize('text', ['Development estimate 0.35, not confirmatory.',
                                 'Development estimate 0.25 and 0.99, not confirmatory.'])
def test_unbound_or_changed_numbers_fail(text):
    assert 'number-undeclared' in rules(inspect(text))


def test_required_number_cannot_disappear():
    assert 'number-missing' in rules(inspect('Development data, not confirmatory.'))


@pytest.mark.parametrize('entry', [
    {'status': 'verified'},
    {'path': 'scores.absent'},
    {'path': 'scores.mean', 'status': 'invented'},
    {'path': 'scores.confirmatory'},
])
def test_malformed_bindings_fail(entry):
    assert 'binding-malformed' in rules(inspect(contract=unit(numbers=[entry])))


def test_contract_cannot_override_artifact_value():
    report = inspect(contract=unit(numbers=[{'path': 'scores.mean', 'text': '0.50', 'decimals': 2}]))
    assert 'number-contract-mismatch' in rules(report)


def test_pending_evidence_is_strict_by_default_and_explicit_otherwise():
    contract = unit(numbers=[{'status': 'pending', 'value': '0.25',
                              'claimed_source_family': 'synthetic-measurement',
                              'reason': 'original score not recovered'}])
    assert not inspect(contract=contract).ok()
    exploratory = inspect(contract=contract, strict=False)
    assert exploratory.ok() and exploratory.pending
    assert exploratory.coverage[0]['pending'] == 1


def test_pending_does_not_excuse_a_missing_value():
    contract = unit(numbers=[{'status': 'pending', 'value': '0.25'}], facts=[])
    assert 'number-missing' in rules(inspect('No estimate supplied.', contract, strict=False))


def test_wrong_statistical_denominator_fails():
    contract = unit(facts=[{'id': 'denominator',
        'description': 'Selection-data standard deviation is the denominator.',
        'all_of': [r'(?:selection|validation).{0,15}standard deviation'],
        'source_evidence': [{'path': 'scores.denominator', 'equals': 'selection'}]}])
    data = {'scores': {'mean': .25, 'denominator': 'selection'}}
    assert inspect('Estimate 0.25 in selection-data standard deviations.', contract, data).ok()
    assert not inspect('Estimate 0.25 in training-data standard deviations.', contract, data).ok()


def test_wrong_permutation_denominator_fails():
    contract = unit(facts=[{'id': 'permutation-estimator', 'description': 'Use the plus-one denominator.',
                           'any_of': [r'\(k\+1\)/\(B\+1\)', r'\\frac\{k\+1\}\{B\+1\}']}],
                    literals=[{'text': '1'}])
    assert inspect('Estimate 0.25 with (k+1)/(B+1).', contract).ok()
    assert not inspect('Estimate 0.25 with (k+1)/B.', contract).ok()


def test_unsupported_causal_mechanism_fails():
    contract = unit(facts=[], unsupported=[{'id': 'mechanism',
        'pattern': r'regularization (?:causes|explains) the difference',
        'reason': 'The intervention did not isolate regularization.'}])
    assert not inspect('Estimate 0.25; regularization causes the difference.', contract).ok()
    assert inspect('Estimate 0.25; the mechanism was not isolated.', contract).ok()


def test_explicit_denial_does_not_assert_a_mechanism():
    assert not ev.asserted_match('regularization causes the difference',
                                'We do not claim regularization causes the difference.')
    assert ev.asserted_match('regularization causes the difference',
                            'The effect is not noise, but regularization causes the difference.')


def test_all_original_seeds_are_quantified():
    assertion = {'path': 'scores.seeds.*.passed', 'equals': False}
    assert ev.check_source(assertion, {'scores': {'seeds': [{'passed': False}, {'passed': False}]}})
    assert not ev.check_source(assertion, {'scores': {'seeds': [{'passed': False}, {'passed': True}]}})
    with pytest.raises(ValueError):
        ev.check_source(assertion, {'scores': {'seeds': []}})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '0.25'])
def test_numerical_binding_requires_finite_number(value):
    report = inspect(data={'scores': {'mean': value, 'confirmatory': False}})
    assert 'binding-malformed' in rules(report)


def test_changed_source_schema_fails_closed():
    report = inspect(data={'scores': {'mean': .25}})
    assert 'scientific-evidence-missing' in rules(report)


def test_relocated_qualification_requires_destination():
    contract = unit(facts=[{'id': 'scope', 'description': 'The scope is stated in Methods.',
                            'in_section': '^Methods$', 'any_of': ['development']}])
    report = ev.Report()
    ev.check_unit('0.25', contract, {'scores': {'mean': .25}}, report,
                  all_sections=[('Methods', 'Development population.')])
    assert report.ok()
    absent = ev.Report()
    ev.check_unit('0.25', contract, {'scores': {'mean': .25}}, absent, all_sections=[])
    assert 'fact-destination-missing' in rules(absent)


def safety(text, tmp_path):
    report = ev.Report()
    ev.check_document_safety(text, tmp_path / 'document.md', tmp_path, report, 'synthetic')
    return report


@pytest.mark.parametrize('text', ['```math\nx=1', '~~~python\nx = 1', '$x = a\n+b$', '$$x = a'])
def test_unclosed_document_delimiters_fail(text, tmp_path):
    assert not safety(text, tmp_path).ok()


def test_ordinary_math_code_and_currency_escaping(tmp_path):
    text = r'An escaped \$ symbol and $x$.' + '\n```math\ny=\\operatorname{mean}(x)\n```\n'
    assert safety(text, tmp_path).ok()


@pytest.mark.parametrize('target', ['javascript:alert(1)', 'file:///tmp/item', '../../outside'])
def test_unsafe_links_fail(target, tmp_path):
    assert 'unsafe-link' in rules(safety(f'[link]({target})', tmp_path))


def test_local_and_external_links(tmp_path):
    (tmp_path / 'Methods.md').write_text('# Methods\n')
    assert safety('[Methods](Methods.md) [source](https://example.org/paper)', tmp_path).ok()
    assert 'broken-link' in rules(safety('[missing](absent.md)', tmp_path))


def fixture_repository(tmp_path, text='## Estimate\nDevelopment estimate 0.25, not confirmatory.\n'):
    (tmp_path / 'document.md').write_text(text, encoding='utf8')
    (tmp_path / 'scores.json').write_text(json.dumps({'mean': .25, 'confirmatory': False}))
    registry = {'schema_version': 1, 'sources': {'scores': 'scores.json'},
                'documents': [{'id': 'study', 'path': 'document.md', 'units': [unit()]}]}
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(registry))
    return path


def test_whole_document_detects_uncontracted_sections(tmp_path):
    registry = fixture_repository(tmp_path, '## Estimate\nDevelopment estimate 0.25, not confirmatory.\n## Additional measurement\n0.83\n')
    assert 'uncontracted-numbers' in rules(ev.check(tmp_path, registry))


def test_duplicate_section_is_ambiguous(tmp_path):
    registry = fixture_repository(tmp_path, '## Estimate\n0.25\n## Estimate\n0.25\n')
    assert 'unit-missing-or-ambiguous' in rules(ev.check(tmp_path, registry))


def test_document_introduction_is_not_a_numeric_bypass(tmp_path):
    registry = fixture_repository(tmp_path, 'Unexplained result 0.86.\n## Estimate\nDevelopment estimate 0.25, not confirmatory.\n')
    assert 'uncontracted-numbers' in rules(ev.check(tmp_path, registry))


def test_unknown_document_is_not_a_vacuous_pass(tmp_path):
    registry = fixture_repository(tmp_path)
    assert not ev.check(tmp_path, registry, document='absent').ok()


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'a\\b', 'a/../b', 'a//b', 'x:y'])
def test_evidence_paths_are_confined(tmp_path, path):
    with pytest.raises(ValueError):
        ev.rooted(tmp_path, path)


def test_subscripts_and_model_names_are_not_measurements():
    assert ev.number_tokens(r'$s_{11}-s_{12}$') == []
    assert ev.number_tokens('Llama-3.1-8B-Instruct and Gemma-2-9B-it') == []


def test_registry_uses_actual_evidence_on_current_repository():
    root = Path(os.environ.get('EVIDENCE_TEST_ROOT', str(ev.ROOT)))
    registry = os.environ.get('EVIDENCE_TEST_REGISTRY')
    report = ev.check(root, Path(registry) if registry else None)
    assert report.ok(), report.errors
    assert not report.pending
    assert sum(row['bound'] for row in report.coverage) > 0


def test_public_evidence_cli_is_strict(tmp_path):
    registry = fixture_repository(tmp_path)
    (tmp_path / 'reproducibility').mkdir()
    (tmp_path / 'reproducibility/evidence.json').write_bytes(registry.read_bytes())
    package = tmp_path / 'src/repro'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    source = Path(ev.__file__).parent
    for name in ('evidence.py', '__main__.py'):
        (package / name).write_bytes((source / name).read_bytes())
    env = dict(os.environ, PYTHONPATH=str(tmp_path / 'src'))
    command = [sys.executable, '-B', '-m', 'repro', 'evidence']
    good = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)['ok']
    (tmp_path / 'document.md').write_text('## Estimate\nDevelopment estimate 0.35, not confirmatory.\n')
    bad = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert bad.returncode == 1
    assert not json.loads(bad.stdout)['ok']
