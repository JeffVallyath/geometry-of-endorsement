"""Exact member checks for compact, completed deliveries; no execution services."""
import gzip
import hashlib
import json
from pathlib import Path

from .common import load_json, safe_path


def payload(source, record):
    raw = safe_path(record['path'], source).read_bytes()
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Stored compact member changed: ' + record['path'])
    if record['encoding'] not in ('verbatim', 'gzip'):
        raise ValueError('Unknown compact encoding')
    raw = gzip.decompress(raw) if record['encoding'] == 'gzip' else raw
    derivative = record.get('privacy_derivative')
    if derivative:
        # Only operational prose/delivery metadata may be projected. Raw scores,
        # scientific analyses, model pins and executable source stay exact.
        if record['member'] not in {'STATUS.md', 'COMPACT_DELIVERY.json', 'runtime/INTAKE.json',
                                   'code/text_scope/DEV_GATE_AMENDMENT_20260919.md'}:
            raise ValueError('Privacy derivative is not permitted for scientific inputs')
        if len(raw) != derivative['bytes'] or hashlib.sha256(raw).hexdigest() != derivative['sha256']:
            raise ValueError('Sanitized derivative changed: ' + record['member'])
    elif len(raw) != record['original_bytes'] or hashlib.sha256(raw).hexdigest() != record['original_sha256']:
        raise ValueError('Original member changed: ' + record['member'])
    return raw


def manifest_index(data):
    entries = data.get('files', data) if isinstance(data, dict) else data
    if isinstance(entries, list):
        return {r.get('path', r.get('name')): r for r in entries}
    return entries


def verify(source, label, expected_archive):
    source = Path(source)
    binding = load_json(source / (label + '_SOURCE.json'))
    if binding['schema'] != 'COMPLETED_DELIVERY_COMPACT_V1' or binding['archive_sha256'] != expected_archive:
        raise ValueError('Unexpected completed source delivery')
    records = {}
    for record in binding['files']:
        safe_path(record['member'], source)
        if record['member'].casefold() in {n.casefold() for n in records}:
            raise ValueError('Duplicate compact source member')
        records[record['member']] = record
        payload(source, record)
    names = {p.relative_to(source).as_posix() for p in (source / 'saved' / label).rglob('*') if p.is_file()}
    if names != {r['path'] for r in records.values()}:
        raise ValueError('Missing or undeclared compact member')
    manifests = [n for n in ('MANIFEST.json', 'SHA256_MANIFEST.json', 'PACKAGE_MANIFEST.json') if n in records]
    if len(manifests) != 1:
        raise ValueError('Missing original member manifest')
    manifest = manifest_index(json.loads(payload(source, records[manifests[0]])))
    for name, record in records.items():
        if name == manifests[0]:
            continue
        original = manifest[name]
        digest = original if isinstance(original, str) else original['sha256']
        if digest != record['original_sha256']:
            raise ValueError('Compact binding differs from original manifest: ' + name)
        if isinstance(original, dict) and 'bytes' in original and original['bytes'] != record['original_bytes']:
            raise ValueError('Original member size differs: ' + name)
    return binding


def restore(source, binding, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a new restore destination')
    for record in binding['files']:
        path = safe_path(record['member'], destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload(source, record))
