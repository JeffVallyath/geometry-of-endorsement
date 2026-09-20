"""Numerical, source-integrity and scientific-scope guards for the QBR closeout."""
import copy
import hashlib
import json
import runpy
import shutil
import zipfile

import pytest

from repro import question_blind_repair as qbr
from repro import evidence


def test_original_population_and_selected_extension_stay_separate():
    result = qbr.check()['result']
    primary, selected = result['primary'], result['post_hoc']
    assert (primary['cases'], primary['baseline_W'], primary['B_repairs'], primary['C_repairs']) == (256, 83, 36, 2)
    assert primary['B_repair_rate'] == 36 / 83 and primary['repaired_designs'] == 28
    assert primary['strong_repair'] is False and primary['strong_repair_threshold'] == 0.5
    assert primary['treated_qualifying_witnesses'] == 35 + 40 == 75
    assert primary['baseline_W'] == 36 + 35 + 12
    assert (primary['B_joint_corrections'], primary['B_joint_damage'], primary['B_direct_damage']) == (944, 347, 76)
    assert primary['D_joint_errors'] < primary['B_joint_errors'] < primary['A_joint_errors']
    assert (selected['selected_cases'], selected['seeds'], selected['anchor_designs']) == (6, 2, 1)
    assert (selected['blocks'], selected['reads'], selected['B_repairs'], selected['C_repairs']) == (14, 14076, 8, 0)
    assert (selected['eligible_RB_targets'], selected['eligible_RC_targets']) == (3, 0)
    assert selected['new_independent_observations'] == 0
    assert selected['delivery_status_is_strong_repair'] is False


def test_example_scope_and_correct_donor_boundary():
    result = qbr.check()['result']
    assert result['example'] == dict(yes_no_full_repair=True, ab_full_repair=False,
        independent_controls_correct=True, same_answers=['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        correct_donor_reverse_error_origins=[6, 7])
    assert result['method']['complete_current_record_reference_required']
    assert result['method']['before_questions']


def test_original_uncertainty_family_is_retained_exactly():
    summary = qbr.load_json(qbr.SOURCE / 'supplied/primary/replay/SUMMARY.json')
    assert qbr.check()['result']['primary']['uncertainty_family'] == summary['family']
    assert summary['bootstrap_unit'] == 'case design preserving all seed/history/question/arm dependence'
    assert summary['bootstrap_draws'] == 10000


@pytest.fixture
def source_copy(tmp_path):
    return shutil.copytree(qbr.SOURCE, tmp_path / 'source')


def test_changed_source_byte_is_rejected(source_copy):
    path = source_copy / 'supplied/verification/WORKED_EXAMPLE.csv'
    path.write_bytes(path.read_bytes().replace(b'Yes', b'No', 1))
    with pytest.raises(ValueError, match='Changed supplied evidence'):
        qbr.check(source_copy)


def test_undeclared_source_is_rejected(source_copy):
    (source_copy / 'supplied/extra.json').write_text('{}')
    with pytest.raises(ValueError, match='undeclared'):
        qbr.check(source_copy)


@pytest.mark.parametrize('section,key,value', [
    ('primary', 'cases', 262),
    ('primary', 'strong_repair', True),
    ('primary', 'B_repair_rate', 8 / 9),
    ('post_hoc', 'new_independent_observations', 6),
    ('post_hoc', 'delivery_status_is_strong_repair', True),
    ('example', 'ab_full_repair', True),
])
def test_generated_scope_cannot_override_source(source_copy, section, key, value):
    path = source_copy / 'CLAIMS.json'
    changed = qbr.load_json(path)
    changed[section][key] = value
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='claim projection'):
        qbr.check(source_copy)


def test_scientific_products_require_raw_byte_equality(tmp_path):
    delivered, rebuilt = tmp_path / 'delivered', tmp_path / 'rebuilt'
    delivered.mkdir()
    rebuilt.mkdir()
    for name in qbr.PRODUCTS:
        (delivered / name).write_bytes(b'one\r\ntwo\r\n')
        (rebuilt / name).write_bytes(b'one\r\ntwo\r\n')
    assert len(qbr.compare_products(delivered, rebuilt)) == 7
    (rebuilt / 'WORKED_EXAMPLE.csv').write_bytes(b'one\ntwo\n')
    with pytest.raises(ValueError, match='Scientific replay product differs'):
        qbr.compare_products(delivered, rebuilt)


def test_delivered_comparator_excludes_only_four_timing_fields():
    reader = runpy.run_path(str(qbr.SOURCE / 'supplied/verification/tools/replay.py'))
    assert sorted(reader['TIMING']) == qbr.TIMING
    original = {'logps': [-0.25, -1.5], 'prediction': 0, 'tokens': [12, 13],
                **{field: 1.0 for field in qbr.TIMING}}
    timed = dict(original, **{field: 3.0 for field in qbr.TIMING})
    report = reader['compare'](original, timed)
    assert report['score_binding_exact_equal'] and not report['raw_record_exact_equal']
    for field in ('prediction', 'tokens', 'arbitrary_timestamp'):
        bad = dict(timed, **{field: 77})
        assert not reader['compare'](original, bad)['tolerance_pass']
    almost = copy.deepcopy(timed)
    almost['logps'][0] += 5e-7
    assert reader['compare'](original, almost)['tolerance_pass']
    assert not reader['compare'](original, almost)['score_binding_exact_equal']
    almost['logps'][0] += 2e-6
    assert not reader['compare'](original, almost)['tolerance_pass']


def make_archive(path, members, declared=None):
    records = {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
               for name, raw in members.items()} if declared is None else declared
    with zipfile.ZipFile(path, 'x') as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('MANIFEST.json', json.dumps({'files': records}))
    return qbr.sha256(path)


@pytest.mark.parametrize('members,declared', [
    ({'../escape': b'x'}, None),
    ({'extra': b'x'}, {}),
    ({'score': b'x'}, {'score': {'bytes': 1, 'sha256': '0' * 64}}),
    ({'Score': b'x', 'score': b'y'}, None),
])
def test_archive_validation_fails_before_extraction(tmp_path, members, declared):
    path = tmp_path / 'bad.zip'
    digest = make_archive(path, members, declared)
    destination = tmp_path / 'extracted'
    with pytest.raises(ValueError):
        qbr.verify_zip(path, digest, destination)
    assert not destination.exists()
    assert not (tmp_path / 'escape').exists()


def test_outer_archive_digest_is_required(tmp_path):
    path = tmp_path / 'valid.zip'
    make_archive(path, {'score': b'x'})
    with pytest.raises(ValueError, match='Outer archive hash'):
        qbr.verify_zip(path, '0' * 64)


def test_replay_does_not_overwrite_a_previous_output(tmp_path):
    with pytest.raises(ValueError, match='fresh output directory'):
        qbr.full_replay(tmp_path / 'missing-verification.zip', tmp_path / 'missing-primary.zip', tmp_path)


def test_current_results_numbers_are_bound_to_the_existing_registry():
    path = qbr.ROOT / 'docs/RESULTS_AND_CLAIMS.md'
    original = path.read_text(encoding='utf-8')
    changed = original.replace('repaired 36/83', 'repaired 37/83', 1)
    assert changed != original
    report = evidence.check(document='results', overrides={'results': changed})
    assert any(row['unit'] == 'question_blind_repair' and row['rule'] == 'number-undeclared' for row in report.errors)


@pytest.mark.parametrize('name', ['README.md', 'WORKED_EXAMPLE.md', 'INTEGRATION_REPORT.md'])
def test_new_reader_docs_have_no_em_dashes(name):
    assert '\u2014' not in (qbr.SOURCE / name).read_text(encoding='utf-8')
