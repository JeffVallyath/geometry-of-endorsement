"""Recognize operational linkage fields without changing scientific identifiers.

The release audit uses this detector on the text extracted by its typed scanner.
The small container helper supports reproducible synthetic regression tests.
Neither function deserializes pickle objects or returns identifier values.
"""
import gzip
import hashlib
import io
from pathlib import PurePosixPath
import re
import zipfile

ANONYMOUS_VALUES = {
    'owned_id': 'ANONYMIZED_RESOURCE',
    'thread_id': 'ANONYMIZED_THREAD',
    'threadid': 'ANONYMIZED_THREAD',
    'task_id': 'ANONYMIZED_TASK',
    'resource_id': 'ANONYMIZED_RESOURCE',
    'contract_id': 'ANONYMIZED_CONTRACT',
    'pod_id': 'ANONYMIZED_RESOURCE',
    'instance_id': 'ANONYMIZED_RESOURCE',
    'ssh_host': 'ANONYMIZED_HOST',
}
# Backslashes are permitted around quotes so JSON embedded in a JSON string is
# covered too. Resource/thread values are literal strings, not arbitrary id keys.
OPERATIONAL_FIELD = re.compile(
    r'''(?i)\\*["'](?P<field>owned_id|thread_id|threadId|task_id|resource_id|contract_id|pod_id|instance_id|ssh_host)\\*["']\s*[:=]\s*\\*["'](?P<value>[^"'\r\n\\]*)\\*["']''')


def find_operational_fields(text):
    """Return only field names and fingerprints, never original values."""
    findings = []
    for match in OPERATIONAL_FIELD.finditer(text):
        field = match.group('field').lower()
        value = match.group('value')
        if not value or value == ANONYMOUS_VALUES[field]:
            continue
        findings.append({'kind': 'operational_linkage_field', 'field': field,
                         'value_sha256': hashlib.sha256(value.encode('utf8')).hexdigest()})
    return findings


def _safe_member(name):
    value = name[:-1] if name.endswith('/') else name
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value or
            '..' in path.parts or '\\' in value or ':' in value):
        raise ValueError('Unsafe container member')


def inspect_text_container(raw, name, *, forbidden_values=(), depth=0):
    """Inspect text and gzip/ZIP members, including names and ZIP metadata.

PDF/image/NumPy extraction belongs to the typed release-audit scanner, which
passes those extracted surfaces to find_operational_fields. This helper makes
no claim of inspecting arbitrary binary formats or image pixels.
"""
    if depth > 8:
        raise ValueError('Container nesting exceeds the inspected bound')
    findings = []
    def text_surface(text):
        findings.extend(find_operational_fields(text))
        for value in forbidden_values:
            if not isinstance(value, str) or not value:
                raise ValueError('Forbidden values must be nonempty strings')
            if value.casefold() in text.casefold():
                findings.append({'kind': 'forbidden_private_value',
                                 'value_sha256': hashlib.sha256(value.encode('utf8')).hexdigest()})
    text_surface(name)
    lower = name.lower()
    if lower.endswith('.gz'):
        # gzip.decompress checks CRC/footer and all concatenated members.
        return findings + inspect_text_container(gzip.decompress(raw), name[:-3],
            forbidden_values=forbidden_values, depth=depth+1)
    if lower.endswith(('.zip', '.npz')):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            text_surface(archive.comment.decode('utf8', errors='replace'))
            names = archive.namelist()
            if len(names) != len(set(n.casefold() for n in names)):
                raise ValueError('Duplicate container members')
            for entry in archive.infolist():
                _safe_member(entry.filename)
                text_surface(entry.filename)
                text_surface(entry.comment.decode('utf8', errors='replace'))
                text_surface(entry.extra.decode('utf8', errors='replace'))
                if entry.is_dir():
                    continue
                if (entry.external_attr >> 16) & 0o170000 not in (0, 0o100000):
                    raise ValueError('Non-regular container member')
                findings.extend(inspect_text_container(archive.read(entry), entry.filename,
                    forbidden_values=forbidden_values, depth=depth+1))
        return findings
    if lower.endswith(('.npy', '.pkl', '.pt', '.pth', '.pdf', '.png', '.jpg', '.jpeg')):
        raise ValueError('Binary metadata requires the typed release scanner')
    encoding = 'utf16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    text_surface(raw.decode(encoding, errors='strict'))
    return findings
