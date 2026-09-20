"""Historical Windows manifests remain byte-identical and safe when relocated."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from repro.common import ROOT
from repro import cpu_robustness


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture(params=['supplied', 'independent'])
def verifier(request):
    if request.param == 'supplied':
        return module(ROOT / 'reproducibility/fresh_inference/run.py', 'portable_supplied').verify
    return module(ROOT / 'reproducibility/state_sufficiency/independent_verification/reconstruct.py',
                  'portable_independent').check_manifest


def manifest(tmp_path, names):
    payload = b'{"saved": true}\n'
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs/value.json').write_bytes(payload)
    data = dict(files=[dict(path=name, role='input', bytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest()) for name in names])
    raw = json.dumps(data, indent=3).encode()
    (tmp_path / 'MANIFEST.json').write_bytes(raw)
    return raw


def test_windows_separator_normalization_preserves_manifest_bytes(tmp_path, verifier):
    original = manifest(tmp_path, ['inputs\\value.json'])
    verifier(tmp_path)
    assert (tmp_path / 'MANIFEST.json').read_bytes() == original


@pytest.mark.parametrize('name', ['../value.json', '..\\value.json', 'C:\\value.json',
                                 '/value.json', '\\server\\share\\value.json', 'inputs/../value.json'])
def test_recorded_paths_never_expand_authority(tmp_path, verifier, name):
    manifest(tmp_path, [name])
    with pytest.raises(ValueError):
        verifier(tmp_path)


def test_normalized_duplicate_is_rejected(tmp_path, verifier):
    manifest(tmp_path, ['inputs/value.json', 'inputs\\value.json'])
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        verifier(tmp_path)


def test_corruption_still_fails_after_normalization(tmp_path, verifier):
    manifest(tmp_path, ['inputs\\value.json'])
    (tmp_path / 'inputs/value.json').write_text('changed', encoding='utf8')
    with pytest.raises(ValueError, match='changed|mismatch'):
        verifier(tmp_path)


def test_audit_export_forwards_explicit_source_paths(tmp_path, monkeypatch):
    seen = []
    def stop(audit, package):
        seen.append((audit, package))
        raise RuntimeError('argument forwarding reached')
    monkeypatch.setattr(cpu_robustness, 'reconstruct', stop)
    with pytest.raises(RuntimeError, match='argument forwarding'):
        cpu_robustness.export(tmp_path / 'output', tmp_path / 'audit', tmp_path / 'inputs')
    assert seen == [(tmp_path / 'audit', tmp_path / 'inputs')]
