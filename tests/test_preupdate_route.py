"""Negative-evidence accounting and the isolated SVG portability fix."""
import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from repro import preupdate_route as replay
from repro.common import ROOT, load_json


@pytest.fixture(scope='module')
def portable():
    path = replay.SOURCE / 'tools/replay_bundle_portable_v1.py'
    spec = importlib.util.spec_from_file_location('preupdate_portable_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_tables_reconstruct_the_closed_negative_branch():
    result = replay.check()['result']
    assert result['joint_errors'] == {'N': 75, 'K': 129, 'SCTRL': 72}
    assert result['joint_responses'] == 1152
    assert (result['direct_new_errors'], result['direct_responses']) == (38, 768)
    assert (result['new_joint_errors'], result['corrected_joint_errors']) == (67, 13)
    assert result['joint_errors']['N'] + result['new_joint_errors'] - result['corrected_joint_errors'] == 129
    assert (result['failed_gates'], result['total_gates']) == (4, 5)
    assert (result['historical_cases'], result['historical_positive_cases']) == (16, 14)
    assert result['seeds'] == [0, 1]
    assert (result['final_cases_run'], result['final_cases_planned']) == (0, 128)
    assert result['corrected_by_seed']['0'] == []
    assert len(result['corrected_by_seed']['1']) == 3


@pytest.mark.parametrize('target', ['supplied/SCREEN_DAMAGE.csv', 'supplied/replay/SCREEN.json',
                                  'supplied/MANIFEST.json', 'supplied/figures/SCREEN_DAMAGE.svg'])
def test_frozen_evidence_integrity_is_byte_exact_even_for_newlines(tmp_path, target):
    source = tmp_path / 'package'
    shutil.copytree(replay.SOURCE, source)
    path = source / target
    raw = path.read_bytes()
    changed = raw.replace(b'\r\n', b'\n') if b'\r\n' in raw else raw.replace(b'\n', b'\r\n')
    assert changed != raw
    path.write_bytes(changed)
    with pytest.raises(ValueError, match='Changed supplied evidence'):
        replay.verify(source)


def test_changed_external_archive_is_rejected(tmp_path):
    archive = tmp_path / 'results.zip'
    archive.write_bytes(b'not the bound external result archive')
    with pytest.raises(ValueError, match='Changed full external archive'):
        replay.verify(archive=archive)


def test_undeclared_supplied_evidence_is_rejected(tmp_path):
    source = tmp_path / 'package'
    shutil.copytree(replay.SOURCE, source)
    (source / 'supplied/unexpected.txt').write_text('extra', encoding='utf8')
    with pytest.raises(ValueError, match='Undeclared or missing'):
        replay.verify(source)


def test_duplicate_case_rows_cannot_inflate_the_statistical_unit(tmp_path):
    source = tmp_path / 'package'
    shutil.copytree(replay.SOURCE, source)
    path = source / 'supplied/SCREEN_DAMAGE.csv'
    lines = path.read_text(encoding='utf8').splitlines()
    path.write_text('\n'.join(lines + [lines[1]]) + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='Missing or duplicate'):
        replay.reconstruct(source)


def test_svg_fix_accepts_only_crlf_serialization_and_preserves_source_bytes(portable):
    for name in ('SCREEN_CORRECTIONS.svg', 'SCREEN_DAMAGE.svg'):
        path = replay.SOURCE / 'supplied/figures' / name
        original = path.read_bytes()
        lf = original.replace(b'\r\n', b'\n')
        crlf = lf.replace(b'\n', b'\r\n')
        assert hashlib.sha256(lf).digest() != hashlib.sha256(crlf).digest()
        assert portable.same_regenerated_svg(lf, crlf)
        assert portable.same_regenerated_svg(crlf, lf)
        assert not portable.same_regenerated_svg(lf, lf.replace(b'width="1100"', b'width="1101"'))
        assert not portable.same_regenerated_svg(lf, lf.replace(b'>3<', b'>4<') if b'>3<' in lf
                                                else lf.replace(b'129', b'128'))
        assert not portable.same_regenerated_svg(lf, lf.replace(b'\n', b'\r'))
        assert path.read_bytes() == original


def test_portable_helper_preserves_original_integrity_and_science_code():
    original = (replay.SOURCE / 'supplied/tools/replay_bundle.py').read_text(encoding='utf8')
    expected = original.replace(
        "for name,digest in regenerated['figures'].items():\n"
        "        if sha(bundle/'figures'/name)!=digest:raise ValueError('Figure regeneration differs')",
        "for name in regenerated['figures']:\n"
        "        if not same_regenerated_svg((bundle/'figures'/name).read_bytes(), (work/'figures'/name).read_bytes()):\n"
        "            raise ValueError('Figure regeneration differs')"
    ).replace('figures_identical=True', 'figures_canonical_lf_identical=True')
    actual = (replay.SOURCE / 'tools/replay_bundle_portable_v1.py').read_text(encoding='utf8')
    def functions(text):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    preserved, patched = functions(expected), functions(actual)
    assert patched.keys() - preserved.keys() == {'same_regenerated_svg'}
    assert all(patched[name] == value for name, value in preserved.items())


def test_audit_is_hash_bound_internal_provenance_only():
    root = ROOT / 'reproducibility/audits/preupdate_evidence_audit_20260919'
    manifest = load_json(root / 'MANIFEST.json')
    assert manifest['new_neural_observations'] == 0
    assert len(manifest['files']) == 4
    for name, record in manifest['files'].items():
        raw = (root / name).read_bytes()
        assert len(raw) == record['bytes']
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
