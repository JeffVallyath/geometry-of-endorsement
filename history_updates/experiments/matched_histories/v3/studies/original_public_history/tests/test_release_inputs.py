"""Public-byte verification rejects integrity bypasses and ambiguous inventories."""
import copy
import gzip
import hashlib
import json

import pytest

from repro.release_inputs import payload, restore, verify


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def compact(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    raw = b'{"count": 8, "qualified": false}\n'
    stored = gzip.compress(raw, mtime=0)
    stored = stored[:9] + b'\xff' + stored[10:]
    (source / 'input.json.gz').write_bytes(stored)
    record = dict(path='input.json.gz', member='runtime/input.json', encoding='gzip',
                  bytes=len(stored), sha256=digest(stored), decoded_bytes=len(raw), decoded_sha256=digest(raw))
    binding = dict(schema='ANONYMOUS_COMPACT_INPUTS_V1', study='fixture', files=[record], support_files=[])
    (source / 'MANIFEST.json').write_text(json.dumps(binding), encoding='utf-8')
    return source, binding, raw


def test_verified_restore_is_exact_and_refuses_existing_destination(compact, tmp_path):
    source, binding, raw = compact
    assert verify(source, 'fixture') == binding
    output = tmp_path / 'restored with spaces'
    restore(source, binding, output)
    assert (output / 'runtime/input.json').read_bytes() == raw
    with pytest.raises(ValueError, match='fresh'):
        restore(source, binding, output)


@pytest.mark.parametrize('mutation', ['stored_hash', 'decoded_hash', 'decoded_size', 'encoding', 'exception', 'two_paths'])
def test_no_integrity_exception_for_changed_scientific_inputs(compact, mutation):
    source, binding, _ = compact
    record = copy.deepcopy(binding['files'][0])
    if mutation == 'stored_hash':
        record['sha256'] = '0' * 64
    elif mutation == 'decoded_hash':
        record['decoded_sha256'] = '0' * 64
    elif mutation == 'decoded_size':
        record['decoded_bytes'] += 1
    elif mutation == 'encoding':
        record['encoding'] = 'unknown'
    elif mutation == 'exception':
        record['privacy_derivative'] = dict(sha256=record['sha256'])
    else:
        record['repository_path'] = 'input.json.gz'
    with pytest.raises(ValueError):
        payload(source, record)


@pytest.mark.parametrize('mutation', ['extra', 'nested_manifest', 'missing', 'duplicate', 'escape'])
def test_closed_inventory_and_member_paths(compact, mutation):
    source, binding, _ = compact
    if mutation == 'extra':
        (source / 'extra.json').write_text('{}', encoding='utf-8')
    elif mutation == 'nested_manifest':
        (source / 'nested').mkdir()
        (source / 'nested/MANIFEST.json').write_text('{}', encoding='utf-8')
    elif mutation == 'missing':
        (source / 'input.json.gz').unlink()
    elif mutation == 'duplicate':
        binding['files'].append(copy.deepcopy(binding['files'][0]))
    else:
        binding['files'][0]['member'] = '../outside'
    (source / 'MANIFEST.json').write_text(json.dumps(binding), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(source, 'fixture')


def test_normalized_header_is_checked_independently_of_payload_hash(compact):
    source, binding, _ = compact
    record = copy.deepcopy(binding['files'][0])
    path = source / record['path']
    raw = bytearray(path.read_bytes())
    raw[4] = 1
    path.write_bytes(raw)
    record['sha256'] = digest(raw)
    with pytest.raises(ValueError, match='header'):
        payload(source, record)
