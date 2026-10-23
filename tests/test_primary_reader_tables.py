"""Bind reader-table cells to model/procedure/control identities, not loose numbers."""
import csv
from pathlib import Path

from repro import state_sufficiency


ROOT = Path(__file__).resolve().parents[1]


def test_primary_reader_table_matches_each_recomputed_cell():
    rows = state_sufficiency.replay()['v10']
    labels = {'INV_PAIR_NLL': 'Constrained learned editor',
              'FREE_PAIR_CONSISTENCY': 'Consistency-trained editor',
              'EXISTING_CORRECTION': 'Existing textual correction',
              'LATEST_SAME_WORDING': 'Latest-value wording'}
    page = (ROOT / 'docs/04_matched_histories.md').read_text(encoding='utf-8')
    for row in rows:
        expected = (f"| {row['model'].capitalize()} | {labels[row['group']]} | "
                    f"{100*row['rate']:.3f}% | [{100*row['lower']:.3f}, {100*row['upper']:.3f}]% | "
                    f"{row['roots']}/64 |")
        assert expected in page


def test_control_table_counts_qualifying_rows_and_case_unions_separately():
    path = ROOT / 'reproducibility/state_sufficiency/cpu_audit/RESULTS.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels = {'G0_witness': 'Original witnesses',
              'G0_M_witness': 'Matching with original eligibility',
              'G2a_M_witness': 'Matching and same-pair controls',
              'G2b_M_witness': 'Matching and cross-pair controls',
              'G2ab_M_witness': 'Matching and both controls'}
    page = (ROOT / 'docs/appendix/A06_stronger_controls.md').read_text(encoding='utf-8')
    for field, label in labels.items():
        selected = [row for row in rows if row[field] == 'True']
        cases = {row['root_id'] for row in selected}
        assert f'| {label} | {len(selected)} | {len(cases)}/64 |' in page
