"""Regression tests for the release wrapper, not replacements for scientific tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from repro import paper_results
from repro.release_manifest import verify


def sample_release(tmp_path):
    (tmp_path / 'data.txt').write_bytes(b'fixed scientific input\n')
    value = {'schema': 'ANONYMOUS_REVIEW_RELEASE_V1', 'files': [
        {'path': 'data.txt', 'bytes': 23, 'sha256': hashlib.sha256(b'fixed scientific input\n').hexdigest()}]}
    # Derive the byte length from the fixture, never bypass the size check.
    value['files'][0]['bytes'] = (tmp_path / 'data.txt').stat().st_size
    (tmp_path / 'release_manifest.json').write_text(json.dumps(value), encoding='utf-8')
    return value


def test_release_integrity_and_undeclared_files(tmp_path):
    sample_release(tmp_path)
    assert verify(tmp_path, strict=True)['verified_files'] == 1
    (tmp_path / 'extra.txt').write_text('unexpected', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        verify(tmp_path)


def test_changed_evidence_fails(tmp_path):
    sample_release(tmp_path)
    (tmp_path / 'data.txt').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        verify(tmp_path)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/absolute', 'data.txt/../data.txt'])
def test_unsafe_manifest_paths(tmp_path, name):
    value = sample_release(tmp_path)
    value['files'][0]['path'] = name
    (tmp_path / 'release_manifest.json').write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_only_explicit_runtime_products_are_exempt(tmp_path):
    sample_release(tmp_path)
    product = tmp_path / 'src/anonymous_review.egg-info/PKG-INFO'
    product.parent.mkdir(parents=True)
    product.write_text('local install', encoding='utf-8')
    assert verify(tmp_path)['status'] == 'PASS'
    with pytest.raises(ValueError, match='inventory'):
        verify(tmp_path, strict=True)


def test_existing_output_preserved(tmp_path):
    (tmp_path / 'keep.txt').write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='new output'):
        paper_results.run(tmp_path)
    assert (tmp_path / 'keep.txt').read_text() == 'keep'


def test_offline_guard_blocks_network_and_models(tmp_path):
    environment = paper_results.offline_environment(tmp_path)
    for code in ('import socket; socket.socket()', 'import transformers', 'import sentence_transformers'):
        result = subprocess.run([sys.executable, '-B', '-c', code], env=environment,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert 'Offline review forbids' in result.stderr


def test_paper_only_registry_and_log_paths(tmp_path):
    names = [name for name, _ in paper_results.jobs(tmp_path)]
    assert len(names) == len(set(names)) == 16
    assert names.index('independent-analyze') < names.index('independent-compare')
    assert 'editors' in names and 'mechanism' in names
    assert paper_results.portable_log(str(tmp_path), tmp_path) == '<output>'


def test_successful_test_wrapper_preserves_integrity(tmp_path, monkeypatch):
    sample_release(tmp_path)
    output = tmp_path / 'reproduced/tests'
    alias = output / '.runtime/pytest/examplecurrent'
    target = alias.with_name('example0')

    def fake_tests(command, **kwargs):
        assert '--basetemp' in command
        assert Path(command[-1]) == alias.parent
        target.mkdir(parents=True)
        (target / 'evidence.txt').write_text('preserve', encoding='utf-8')
        alias.symlink_to(target, target_is_directory=True)
        return subprocess.CompletedProcess(command, 0, 'synthetic wrapper check passed\n', '')

    monkeypatch.setattr(paper_results, 'verify', lambda: verify(tmp_path))
    monkeypatch.setattr(paper_results.subprocess, 'run', fake_tests)
    try:
        result = paper_results.run(output, test=True)
        assert result['status'] == 'PASS'
        assert verify(tmp_path)['status'] == 'PASS'
        assert (target / 'evidence.txt').read_text(encoding='utf-8') == 'preserve'
        assert not alias.is_symlink()
        assert result['pytest_current_links_removed'] == 1
    finally:
        if alias.is_symlink():
            alias.unlink()


@pytest.mark.parametrize('kind', ['outside', 'missing', 'wrong_name', 'indirect'])
def test_pytest_alias_cleanup_rejects_unsafe_links_atomically(tmp_path, kind):
    output = tmp_path / 'output'
    base = output / '.runtime/pytest'
    base.mkdir(parents=True)
    target = base / 'example0'
    target.mkdir()
    good = base / 'examplecurrent'
    good.symlink_to(target, target_is_directory=True)
    bad = base / 'badcurrent'
    extra = None
    if kind == 'outside':
        wrong = tmp_path / 'bad0'
        wrong.mkdir()
    elif kind == 'missing':
        wrong = base / 'bad0'
    elif kind == 'wrong_name':
        wrong = target
    else:
        wrong = base / 'bad0'
        wrong.symlink_to(target, target_is_directory=True)
        extra = wrong
    bad.symlink_to(wrong, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match='pytest runtime alias'):
            paper_results.remove_pytest_current_links(output)
        assert good.is_symlink() and bad.is_symlink()
        assert target.is_dir()
    finally:
        for path in (good, bad, extra):
            if path is not None and path.is_symlink():
                path.unlink()


def test_pytest_alias_cleanup_rejects_linked_runtime(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    other = tmp_path / 'other'
    (other / 'pytest').mkdir(parents=True)
    runtime = output / '.runtime'
    runtime.symlink_to(other, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match='linked pytest runtime'):
            paper_results.remove_pytest_current_links(output)
        assert runtime.is_symlink() and (other / 'pytest').is_dir()
    finally:
        runtime.unlink()


def test_arbitrary_runtime_symlink_still_fails_integrity(tmp_path):
    sample_release(tmp_path)
    runtime = tmp_path / 'reproduced'
    runtime.mkdir()
    alias = runtime / 'unexpected-link'
    alias.symlink_to(tmp_path / 'data.txt')
    try:
        with pytest.raises(ValueError, match='symbolic link'):
            verify(tmp_path)
    finally:
        alias.unlink()


def test_portable_log_redacts_json_escaped_paths(tmp_path):
    paths = {'output': str(tmp_path), 'root': str(paper_results.ROOT),
             'environment': str(Path(sys.prefix))}
    text = json.dumps(paths, indent=2)
    result = json.loads(paper_results.portable_log(text, tmp_path))
    assert result == {'output': '<output>', 'root': '<review-root>',
                      'environment': '<python-env>'}
