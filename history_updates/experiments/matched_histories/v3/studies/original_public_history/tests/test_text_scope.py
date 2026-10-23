"""Smaller-model measurements retain failed bridges, missing rows and repeats."""
import json

import pytest

from repro import text_scope
from repro.release_inputs import payload, verify


@pytest.fixture(scope='module')
def inputs():
    binding = verify(text_scope.SOURCE, 'text_scope')
    return {row['member']: row for row in binding['files']}


def saved(inputs, name):
    return json.loads(payload(text_scope.SOURCE, inputs[name]))


def test_completed_cells_and_missing_row_are_not_zero_results(inputs):
    table = saved(inputs, 'runtime/analysis_final/MAIN_TABLE.json')
    complete = [row for row in table if row['status'] == 'COMPLETE_FINAL_ROW']
    assert len(complete) == 6
    assert [row['W'] for row in complete] == [104, 93, 61, 68, 115, 110]
    assert all(row['opportunities'] == 1152 for row in complete)
    absent = [row for row in table if row['row_id'] == 'qwen32_t4']
    assert len(absent) == 2 and all(row['status'] == 'NOT_MEASURED' and row['witness_rate'] is None for row in absent)
    contrasts = saved(inputs, 'runtime/analysis_final/PLANNED_COMPARISONS.json')
    assert len(contrasts) == 6 and all(row['status'] == 'NOT_ASSESSABLE' for row in contrasts)


def test_model_variants_and_failed_bridges_are_retained(inputs):
    roster = saved(inputs, 'runtime/MODEL_ROSTER_FROZEN.json')
    rows = {row['id']: row for row in roster['rows']}
    assert rows['gemma9_t4']['model'] == 'google/gemma-2-9b-it'
    assert rows['llama8_t4']['model'] == 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    assert rows['qwen4_t4']['model'] == 'Qwen/Qwen3-4B'
    assert all(len(rows[name]['revision']) == 40 for name in ('gemma9_t4', 'llama8_t4', 'qwen4_t4'))
    for name in ('gemma9_c1', 'qwen8_t4'):
        assert saved(inputs, 'runtime/checks/BRIDGE_' + name + '.json')['status'] != 'PASS'
    for name in ('gemma9_t4', 'llama8_t4', 'qwen4_t4'):
        assert saved(inputs, 'runtime/checks/DEV_' + name + '.json')['status'] != 'PASS'
        assert saved(inputs, 'runtime/checks/DEV_E0_R1_' + name + '.json')['status'] == 'PASS'
    assert saved(inputs, 'runtime/checks/DEV_E0_R1_qwen32_t4.json')['status'] != 'PASS'


def test_repeat_process_distinctness_scores_and_fixed_population(inputs):
    for name in ('gemma9_t4', 'llama8_t4', 'qwen4_t4'):
        result = saved(inputs, 'runtime/checks/REPEATS_' + name + '.json')
        processes = result['fresh_processes']
        assert [row[0] for row in processes] == ['final', 'repeat0', 'repeat1']
        assert len({row[1] for row in processes}) == 3
        for phase, identity in processes:
            assert saved(inputs, f'runtime/results/{name}/{phase}/PHASE_DONE.json')['scorer_pid'] == identity
        assert result['comparisons'] == 3264 and result['max_logp_delta'] == 0
        assert not result['discrepancies'] and not result['cache_identity_discrepancies']
        assert not result['primary_scores_averaged_with_repeats']


def test_full_saved_score_replay(tmp_path):
    report = text_scope.export(tmp_path / 'breadth with spaces')
    assert report['status'] == 'PASS'
    assert report['new_final_score_records'] == 110976
    assert report['failed_planned_contrasts'] == 6
    assert len(report['measured_method_rows']) == 6
    assert report['cached_qwen32_not_measured'] and not report['new_design_sampling']
    assert report['integer_classification_equality'] == 'exact'
    assert report['finite_float_atol'] == 1e-12 and not report['inference']
