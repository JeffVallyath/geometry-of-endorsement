"""Scientific integrity and portable complete replay, not prescribed caveat wording."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from repro import qwen32_full_forward as qwen
from repro.fresh_history import equivalent


def load(path):
    return json.loads(path.read_bytes())


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    output = tmp_path_factory.mktemp('qwen32') / 'full-replay'
    report = qwen.export(output)
    return output, report


def test_complete_raw_replay_and_frozen_source_tables(replay):
    output, report = replay
    assert report['status'] == 'PASS'
    assert report['compact_members_checked'] == len(load(qwen.SOURCE / 'SOURCE.json')['files'])
    assert report['raw_final_rows'] == 36992 and report['classification_fields_checked'] == 27136
    assert report['repeats_checked'] == 4488 and report['full_case_classifications_identical'] == 4
    assert report['repeats_strictly_equal'] and report['original_csv_tables_byte_identical']
    assert report['summary_max_float_difference'] <= 1e-12
    assert report['full_repeat_derived_max_float_difference'] <= 1e-12
    assert report['inference'] is False and report['socket_and_model_imports_blocked']
    assert load(output / 'CLAIMS.json') == load(qwen.SOURCE / 'CLAIMS.json')


def test_positive_primary_null_strongest_and_nonzero_certification(replay):
    output, _ = replay
    data = load(output / 'CLAIMS.json')
    assert data['original_designs_newly_measured'] and not data['newly_sampled_designs']
    for method, count, designs, single in zip(qwen.METHODS, (128, 123), (43, 49), (1, 2)):
        row = data['methods'][method]
        assert (row['primary'], row['denominator'], row['designs_with_primary']) == (count, 1152, designs)
        assert (row['strongest'], row['semantic_denominator']) == (0, 576)
        assert (row['fully_certified_opportunities'], row['fully_certified_designs']) == (80, 9)
        assert row['matched_single_format'] == single
    assert data['single_format_distinct_designs'] == 2
    assert not any(data[k] for k in ('scaling_comparison_supported', 'causal_repair_measured', 'moral_decision_effect_measured'))


def test_worked_example_all_controls_raw_scores_and_failed_other_format(replay):
    output, _ = replay
    example = load(output / 'computed/WORKED_EXAMPLE.json')
    assert example == load(qwen.SOURCE / 'WORKED_EXAMPLE.json')
    assert example['root_id'] == 'SSC1-FINAL-0003' and example['origins'] == [1, 2]
    assert example['prefix_tokens'] == [834, 834] and example['changed_facts'] == [1, 1]
    assert example['direct_checks'] == 12 and example['logical_checks'] == 8
    assert [row['answers'] for row in example['table']] == [[0, 0], [0, 0], [1, 1], [0, 0], [0, 1]]
    assert not example['other_format_noop_correct'] and example['other_format_selected_answers'] == [1, 1]
    assert example['old_target_truth'] == [0, 0]
    assert example['selection_for_exposition'] == 'post_hoc' and not example['strongest_cross_format']
    assert example['repeated_records'] == 578 and example['repeated_scores_and_token_bindings_identical']


def test_all_plotted_values_match_the_reconstructed_evidence(replay):
    output, _ = replay
    presentation = output / 'computed/presentation'
    plotted = load(presentation / 'PLOTTED_VALUES.json')
    assert plotted == load(qwen.SOURCE / 'figures/PLOTTED_VALUES.json')
    assert load(presentation / 'FIGURE_DATA.json') == load(qwen.SOURCE / 'figures/FIGURE_DATA.json')
    claims = load(output / 'CLAIMS.json')
    for i, method in enumerate(qwen.METHODS):
        bars = plotted['FIGURE_1_COVERAGE.png'][i]
        assert bars[3] == 100 * claims['methods'][method]['coverage']
        assert bars[5] == 100 * claims['methods'][method]['primary_rate']
        excluded = plotted['FIGURE_2_ATTRITION.png'][i]
        assert sum(excluded) == 576 and excluded[-2:] == [80, 0]
    for name in ('FIGURE_1_COVERAGE.png', 'FIGURE_2_ATTRITION.png'):
        assert (presentation / name).stat().st_size > 10000


@pytest.mark.parametrize('mutation', ['repeat_missing', 'repeat_discrepancy', 'classification', 'raw_count'])
def test_source_pass_label_cannot_mask_incomplete_or_changed_repeats(replay, mutation):
    output, _ = replay
    source = load(output / 'computed/analysis/INDEPENDENT_REPLAY.json')
    independent = load(output / 'computed/independent/INDEPENDENT_CHECK.json')
    if mutation == 'repeat_missing':
        independent['repeat_rows']['repeat1'] -= 1
    elif mutation == 'repeat_discrepancy':
        independent['repeat_discrepancies'].append({'changed': True})
    elif mutation == 'classification':
        source['full_repeat_reconstruction'][0]['classification_identical'] = False
    else:
        independent['raw_final_checked'] -= 1
    with pytest.raises(ValueError):
        qwen.validate_receipts(source, independent)


def test_derived_roundoff_never_permits_changed_classifications():
    assert equivalent({'mass': .5}, {'mass': .5 + 1e-16})
    assert not equivalent({'mass': .5}, {'mass': .5 + 2e-12})
    assert not equivalent({'qualified': True}, {'qualified': False})
    assert not equivalent({'count': 80}, {'count': 81})


def test_compact_raw_corruption_fails_before_replay(tmp_path):
    mutant = tmp_path / 'compact'
    shutil.copytree(qwen.SOURCE, mutant)
    binding = load(mutant / 'SOURCE.json')
    row = next(r for r in binding['files'] if r['member'].endswith('RAW_SCORES.jsonl'))
    path = mutant / row['path']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        qwen.verify(mutant)


def test_changed_claim_projection_is_rejected(tmp_path):
    mutant = tmp_path / 'compact'
    shutil.copytree(qwen.SOURCE, mutant)
    claims = load(mutant / 'CLAIMS.json')
    claims['methods']['EXISTING_CORRECTION']['strongest'] = 1
    (mutant / 'CLAIMS.json').write_text(json.dumps(claims), encoding='utf8')
    with pytest.raises(ValueError, match='projection differs'):
        qwen.check_projection(mutant, integrity=False)


def test_existing_output_and_escaping_member_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='fresh output'):
        qwen.export(tmp_path)
    with pytest.raises(ValueError, match='normalized repository-relative'):
        qwen._payload(tmp_path, {'path': '../outside'})


def test_portfolio_uses_five_distinct_checkpoints_and_preserves_old_failures():
    portfolio = qwen.check_portfolio()
    assert portfolio['checkpoints'] == len(set(portfolio['distinct_models'])) == 5
    assert portfolio['families'] == 3 and not portfolio['numerical_bridges_repaired']
    table = load(qwen.SOURCE / 'portfolio/MAIN_TABLE.json')
    old = [row for row in table if row['row_id'] == 'qwen32_t4']
    assert len(old) == 2 and all(row['status'] == 'NOT_MEASURED' and row['witness_rate'] is None for row in old)
