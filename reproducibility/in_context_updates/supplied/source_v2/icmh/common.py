"""Shared primitives for ICMH1-RIPPLE-v2: hashing, atomic writes, hash-chained journal.

Standard library only. No model, provider, or network access in this module.
"""
import hashlib
import json
import os
from pathlib import Path

STUDY_ID = 'icmh1-ripple-v2'
SYSTEM_INSTRUCTION = 'Answer with only the entity name.'
MAX_NEW_TOKENS = 16

# Frozen context headers, copied verbatim from the v7 source renderer
# (evcpu/schedule.py render_readout_prompt), minus the bracketed record metadata.
HEADER_CHRONOLOGICAL = ('The following are authoritative record assignments, in chronological order. '
                        'Later scalar assignments replace earlier values at the same address and time; '
                        'positive-membership records accumulate.')
HEADER_CANONICAL = 'The following are the current authoritative records.'
HEADER_GIVEN_FACTS = 'The following task-relevant records are supplied.'

CONTEXTS = ('none', 'given_facts', 'canonical', 'chronological_B', 'chronological_D')
PHASES = ('development', 'qualification', 'final', 'repeat_670', 'repeat_618')


class IntegrityError(Exception):
    pass


def need(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest_ = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest_.update(block)
    return digest_.hexdigest()


def git_blob_sha1(path):
    """Git blob object id, for verifying small non-LFS repository files."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return sha_bytes(canon(value).encode('utf-8'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha_file(path)


def write_new_json(path, value):
    """Refuse to overwrite an existing receipt."""
    need(not Path(path).exists(), 'Receipt already exists; reconcile instead of overwriting: %s' % path)
    return write_json(path, value)


class Journal:
    """Append-only hash-chained JSONL journal; every append is fsynced."""

    def __init__(self, path):
        self.path = Path(path)
        self._head = None
        self._head_loaded = False

    def head(self):
        """Chain head, verified once from disk, then maintained in memory by this writer."""
        if not self._head_loaded:
            _, self._head, incomplete = self.load()
            need(not incomplete, 'Journal has an incomplete tail; reconcile before appending: %s' % self.path)
            self._head_loaded = True
        return self._head

    def load(self):
        if not self.path.exists():
            return [], None, False
        records, previous, incomplete = [], None, False
        with self.path.open('r', encoding='utf-8') as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    incomplete = True
                    break
                expected = sha_bytes((canon(row['payload']) + (row['previous'] or '')).encode('utf-8'))
                need(row['sha256'] == expected, 'Journal hash mismatch at record %d' % (len(records) + 1))
                need(row['previous'] == previous, 'Journal chain break at record %d' % (len(records) + 1))
                previous = row['sha256']
                records.append(row['payload'])
        return records, previous, incomplete

    def append(self, payload):
        previous = self.head()
        row = {'payload': payload, 'previous': previous,
               'sha256': sha_bytes((canon(payload) + (previous or '')).encode('utf-8'))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(canon(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._head = row['sha256']
        return row['sha256']

    def completed_units(self):
        records, _, _ = self.load()
        return {r['unit_id'] for r in records if r.get('event') == 'generation'}
