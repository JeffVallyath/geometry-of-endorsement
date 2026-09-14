import csv
import gzip
import json

import pytest

from repro.common import ROOT
from repro.figures import editing_comparisons


def test_matched_coverage_preserves_all_models_editors_and_seeds():
    selected = editing_comparisons()['coverage']
    with (ROOT / 'reproducibility/relational_editing/v4/expected/T10_coverage_contrasts.csv').open(newline='') as stream:
        source = list(csv.DictReader(stream))
    assert len(selected) == 4
    for row in selected:
        arm = 'INV' if row['editor'] == 'Constrained' else 'FREE'
        rows = {r['seed']: r for r in source if r['actor'] == row['actor'] and r['contrast'] == f'{arm}_COMPLETE_SINGLE_minus_{arm}_SPARSE_CONTINUE'}
        for key in ('effect', 'lower', 'upper', 'level'):
            assert row[key] == float(rows['avg'][key])
        assert row['seeds'] == [float(rows[s]['effect']) for s in ('s0', 's1')]


def test_fixed_reader_plot_preserves_original_paired_intervals():
    selected = editing_comparisons()['readers']
    path = ROOT / 'reproducibility/relational_editing/v6/scores/reader_paired_contrasts.json.gz'
    source = json.loads(gzip.decompress(path.read_bytes()))
    assert len(selected) == len(source) == 8
    for row in selected:
        architecture = 'INV_COMPLETE_SINGLE' if row['editor'] == 'Constrained' else 'FREE_COMPLETE_SINGLE'
        reader = 'R1' if row['reader'] == 'Paraphrase' else 'R2'
        original = next(r for r in source if (r['actor'], r['architecture'], r['reader']) == (row['actor'], architecture, reader))
        for key in ('effect', 'lower', 'upper', 'level', 'scenes'):
            assert row[key] == original[key]
        assert row['seeds'] == [original['seed_specific'][str(s)]['effect'] for s in (0, 1)]


def test_coverage_plot_rejects_a_missing_training_seed(tmp_path):
    folder = tmp_path / 'reproducibility/relational_editing/v4/expected'
    folder.mkdir(parents=True)
    (folder / 'T10_coverage_contrasts.csv').write_text('actor,contrast,seed\n')
    with pytest.raises(ValueError, match='both seeds'):
        editing_comparisons(tmp_path)
