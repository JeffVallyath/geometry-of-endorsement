"""Bounded integration guards for actual completed evidence and its reader routes."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from repro.common import ROOT
from repro.compact_delivery import payload, verify
from repro.completed_followups import check
from repro import text_scope


def test_completed_deliveries_and_missing_row_are_not_zero_results():
    binding = verify(text_scope.SOURCE, 'text_scope', text_scope.ARCHIVE)
    records = {r['member']: r for r in binding['files']}
    table = json.loads(payload(text_scope.SOURCE, records['runtime/analysis_final/MAIN_TABLE.json']))
    assert len(binding['files']) == 1791
    assert len([r for r in table if r['status'] == 'COMPLETE_FINAL_ROW']) == 6
    assert [r['W'] for r in table if r['status'] == 'COMPLETE_FINAL_ROW'] == [104, 93, 61, 68, 115, 110]
    absent = [r for r in table if r['row_id'] == 'qwen32_t4']
    assert len(absent) == 2 and all(r['status'] == 'NOT_MEASURED' and r['witness_rate'] is None for r in absent)
    contrasts = json.loads(payload(text_scope.SOURCE, records['runtime/analysis_final/PLANNED_COMPARISONS.json']))
    assert len(contrasts) == 6 and all(r['status'] == 'NOT_ASSESSABLE' for r in contrasts)
    assert check()['deliveries'] == dict(carrier_compact=9, mediation=13, method_selection=8, valueprism_status=2)


def test_privacy_exception_cannot_relabel_changed_scientific_scores(tmp_path):
    raw = b'{"score": 0.2}\n'
    (tmp_path / 'score.json').write_bytes(raw)
    record = dict(path='score.json', member='runtime/results/SCORES.json', encoding='verbatim',
                  bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), original_bytes=len(raw),
                  original_sha256='0' * 64, privacy_derivative=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match='not permitted for scientific inputs'):
        payload(tmp_path, record)
    del record['privacy_derivative']
    with pytest.raises(ValueError, match='Original member changed'):
        payload(tmp_path, record)


def test_private_fields_removed_with_distinct_original_identities():
    records = json.loads((ROOT / 'reproducibility/privacy_derivatives.json').read_bytes())['files']
    assert records
    for row in records:
        raw = (ROOT / row['path']).read_bytes()
        assert len(raw) == row['derivative_bytes']
        assert hashlib.sha256(raw).hexdigest() == row['derivative_sha256']
        assert row['original_sha256'] != row['derivative_sha256']
    gates = json.loads((ROOT / 'reproducibility/state_sufficiency/cpu_audit/GATE_SPEC.json').read_bytes())
    assert 'user_prompt_path' not in gates
    assert len(gates['mappings']) == 64


def test_gallery_repeated_answers_and_failed_format_match_the_existing_example():
    gallery = (ROOT / 'figures/CAPTIONS.md').read_text(encoding='utf8')
    source = (ROOT / 'reproducibility/question_blind_repair/WORKED_EXAMPLE.md').read_text(encoding='utf8')
    def answers(text):
        return [line.split('|')[2:-1] for line in text.splitlines()
                if line.startswith('| ') and line.count('Yes') + line.count('No') == 7]
    assert answers(gallery) == answers(source)
    assert 'post-hoc-selected' in gallery
    assert 'fails in counterbalanced A/B' in gallery
    assert 'full current-record reference' in gallery
    assert 'strongest cross-format result remains zero' in gallery
    for name in ('fig01_layer_sweep.png', 'fig02_scorer_comparison.png', 'fig03_permutation_null.png',
                 'fig04_truth_control.png', 'fig05_specificity.png', 'repair_benefit_damage.png',
                 'fresh_confirmation_utility.png'):
        assert name in gallery


def test_current_navigation_labels_status_as_historical_and_retains_units():
    guide = (ROOT / 'PAPER_GUIDE.md').read_text(encoding='utf8')
    assert 'September 15 project snapshot' in guide
    assert '36/83' in guide and '28 of 256' in guide
    assert '0/576 for each method' in guide
    assert 'not newly sampled' in guide
    assert 'technical stop before inference' in guide
    assert 'anonymous review copy' not in guide
