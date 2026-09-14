import gzip
import json
import shutil

import pytest

from repro.common import ROOT
from repro.consequence_figures import comparison_data, COUNTERFACTUAL, SOURCE_ROOTS


def test_counterfactual_arms_share_worlds_and_preserve_failed_qualification():
    data = comparison_data()['counterfactual']
    saved = json.loads((ROOT / COUNTERFACTUAL).read_text(encoding='utf8'))['headline']['gemma']
    for row in data['arms']:
        estimate = saved['csr'][row['key']]['csr']
        assert (row['recovery'], row['lower'], row['upper'], row['worlds']) == (
            estimate['mean'], estimate['ci_low'], estimate['ci_high'], estimate['n'])
        if row['margin_hit'] is not None:
            assert row['margin_hit'] == 1.0
            assert row['strong_target_pass'] is False
            assert row['hard_answer_correct'] == 95 / 96
    assert data['arms'][2]['margin_hit'] is None  # A patch is not a tuned margin test.


def test_source_history_counts_include_all_roots_and_both_seeds():
    history = comparison_data()['source_history']
    source = json.loads(gzip.decompress((ROOT / SOURCE_ROOTS).read_bytes()))['per_root']
    assert len(history['groups']) == 8
    for group in history['groups']:
        rows = [r for r in source if (r['actor'], r['condition'], r['reader']) ==
                (group['actor'], group['condition'], 'R0')]
        atomic = [r for r in rows if all(r['by_family'][f]['all_origin_bundle_correct'] == 1 for f in ('direct', 'opposes'))]
        changed = [r for r in atomic if r['joint_only']['origin_disagreement'] > 0]
        assert len(rows) == group['roots']
        assert len(atomic) == group['atomic_perfect']
        assert {r['root_id'] for r in changed} == set(group['disagreement_root_ids'])
        assert len(changed) == group['atomic_perfect_with_joint_disagreement']
    constrained = [r for r in history['groups'] if r['editor'] == 'Constrained']
    assert [(r['atomic_perfect_with_joint_disagreement'], r['atomic_perfect']) for r in constrained] == [(39, 57), (38, 56), (11, 21), (11, 22)]


def test_plot_rejects_missing_source_root(tmp_path):
    (tmp_path / COUNTERFACTUAL).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / COUNTERFACTUAL, tmp_path / COUNTERFACTUAL)
    data = json.loads(gzip.decompress((ROOT / SOURCE_ROOTS).read_bytes()))
    index = next(i for i, r in enumerate(data['per_root']) if r['condition'] == 'INV_COMPLETE_SINGLE_s0' and r['reader'] == 'R0')
    data['per_root'].pop(index)
    (tmp_path / SOURCE_ROOTS).parent.mkdir(parents=True)
    (tmp_path / SOURCE_ROOTS).write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError, match='every unique root'):
        comparison_data(tmp_path)


def test_design_example_is_saved_stimulus_not_fabricated_answers():
    example = comparison_data()['source_history']['example']
    source = json.loads((ROOT / example['source']).read_text(encoding='utf8').splitlines()[0])
    assert example['root_id'] == source['root_id']
    for initial, scene in zip(example['initial_values'], source['sources']):
        assert initial == [scene['values'][e['actor']][e['project']] for e in source['commands']]
    assert example['requested_final_values'] == [e['value'] for e in source['commands']]


def test_committed_plot_data_match_source_projection():
    saved = json.loads((ROOT / 'artifacts/figures/consequence_comparisons.json').read_text(encoding='utf8'))
    assert comparison_data() == saved
