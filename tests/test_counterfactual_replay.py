"""Saved-score equivalence, qualification failures and unchanged relation vectors."""
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from repro import counterfactual as replay
from repro.common import ROOT

BASE = ROOT / 'reproducibility/representation/counterfactual_fidelity'


@pytest.fixture(scope='module')
def computed(tmp_path_factory):
    output = tmp_path_factory.mktemp('consequences') / 'new replay'
    report = replay.export(output)
    return report, json.loads((output / 'RESULTS.json').read_bytes())


def test_all_145_original_comparisons_and_applied_worlds(computed):
    report, results = computed
    assert len(report['comparisons']) == 145
    assert all(row['passed'] and row['tolerance'] == 2e-10 for row in report['comparisons'])
    assert max(row['max_absolute_error'] for row in report['comparisons']) <= 2e-10
    assert not report['inference']
    for arm in ('ORIGINAL_FROZEN_DIM@0.75', 'REGIME_MATCHED_DIM@0.75',
                'NATURAL_FULL_STATE_INTERPOLATION@0.75', 'FULL_STATE_SOURCE_PATCH'):
        assert results['primary/' + arm]['n'] == 96
    assert results['primary/ORIGINAL_FROZEN_DIM@0.75']['csr']['mean'] < 0
    assert results['primary/NATURAL_FULL_STATE_INTERPOLATION@0.75']['csr']['mean'] > .9


def test_direct_flips_never_override_failed_dose_and_assay_qualification(computed):
    report, results = computed
    for arm in ('ORIGINAL_FROZEN_DIM', 'REGIME_MATCHED_DIM'):
        assert results['hits'][arm]['within_tolerance'] == 1
        assert results['hits'][arm]['hard_answer_counterfactual'] == 95 / 96
    analysis = json.loads(gzip.decompress((BASE / 'gemma_analysis.json.gz').read_bytes()))
    for arm in ('ORIGINAL_FROZEN_DIM', 'REGIME_MATCHED_DIM'):
        dose = analysis['dose'][arm]['0.75']
        assert dose['fraction_dose_valid'] == 80 / 96
        assert dose['norm_budget_ok'] and not dose['dose_valid'] and not dose['reach_ok']
    assert report['formal_classifications']['gemma']['class'] == 'DIRECTION_NOT_QUALIFIED'
    assert report['formal_classifications']['gemma']['transport_disposition'] == 'STRONG_TARGET_UNREACHABLE'
    assert report['formal_classifications']['llama']['class'] == 'ASSAY_NOT_QUALIFIED'


def test_retained_corpus_rebuild_and_split_membership_are_exact():
    source = BASE / 'source'
    sys.path.insert(0, str(source))
    try:
        import stcrf_v2_build_corpus as builder
        worlds = builder.build_worlds()
        saved = [json.loads(line) for line in gzip.decompress((BASE / 'corpus.jsonl.gz').read_bytes()).splitlines()]
        assert worlds == saved
        assert collections.Counter(row['split'] for row in worlds) == {'selection': 32, 'fit': 64, 'final': 96}
        assert len({row['world_id'] for row in worlds}) == 192
        assert builder.balance_audit(worlds)['all_factors_balanced_within_each_split']
    finally:
        sys.path.remove(str(source))


def test_available_original_directions_have_the_exact_frozen_array_identity():
    inputs = json.loads((BASE / 'input_manifest.json').read_bytes())
    for model, layer, dimension in (('gemma', 27, 3584), ('llama', 19, 4096)):
        path = ROOT / f'artifacts/probe_weights/m1_relation/{model}_layer{layer}.npz'
        with np.load(path, allow_pickle=False) as arrays:
            direction = arrays['difference_in_means_direction']
            assert direction.shape == (dimension,)
            assert hashlib.sha256(direction.tobytes()).hexdigest() == inputs['original_frozen_dim_sha256'][model]
            assert 'repository_commit' not in arrays
