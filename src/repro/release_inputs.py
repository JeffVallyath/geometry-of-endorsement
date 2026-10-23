"""Verify and restore compact inputs using only the delivered reviewer files."""
import gzip
import hashlib
from pathlib import Path

from .common import ROOT, load_json, safe_path


def payload(source, record, root=ROOT):
    """Check stored and decoded public bytes; no original-custody exception exists."""
    if 'privacy_derivative' in record:
        raise ValueError('Integrity exceptions are not permitted for scientific inputs')
    if ('path' in record) == ('repository_path' in record):
        raise ValueError('Specify exactly one public storage path')
    path = (safe_path(record['path'], Path(source)) if 'path' in record
            else safe_path(record['repository_path'], Path(root)))
    raw = path.read_bytes()
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Stored payload hash mismatch: ' + path.name)
    if record['encoding'] not in ('gzip', 'verbatim'):
        raise ValueError('Unknown public input encoding')
    if record['encoding'] == 'gzip':
        if len(raw) < 10 or raw[:4] != b'\x1f\x8b\x08\x00' or raw[4:8] != bytes(4) or raw[9] != 255:
            raise ValueError('Unnormalized gzip header')
        raw = gzip.decompress(raw)
    if len(raw) != record['decoded_bytes'] or hashlib.sha256(raw).hexdigest() != record['decoded_sha256']:
        raise ValueError('Decoded payload hash mismatch: ' + path.name)
    return raw


def verify(source, study, root=ROOT):
    """Reject changed, missing, duplicate or undeclared public input files."""
    source = Path(source)
    binding = load_json(source / 'MANIFEST.json')
    if binding['schema'] != 'ANONYMOUS_COMPACT_INPUTS_V1' or binding['study'] != study:
        raise ValueError('Unexpected public input manifest')
    members, local = set(), set()
    for record in binding['files']:
        member = record['member']
        safe_path(member, source)
        if member.casefold() in members:
            raise ValueError('Duplicate restored member')
        members.add(member.casefold())
        payload(source, record, root)
        if 'path' in record:
            if record['path'].casefold() in local:
                raise ValueError('Duplicate stored member')
            local.add(record['path'].casefold())
    for record in binding['support_files']:
        payload(source, record, root)
        if 'path' not in record or record['path'].casefold() in local:
            raise ValueError('Invalid or duplicate support file')
        local.add(record['path'].casefold())
    observed = set()
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('Linked inputs are not permitted')
        if path.is_file():
            name = path.relative_to(source).as_posix()
            if name != 'MANIFEST.json':
                if name.casefold() in observed:
                    raise ValueError('Case-ambiguous input inventory')
                observed.add(name.casefold())
    if observed != local:
        raise ValueError('Missing or undeclared public input member')
    return binding


def restore(source, binding, destination, root=ROOT):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a fresh restored-source directory')
    for record in binding['files']:
        path = safe_path(record['member'], destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload(source, record, root))
