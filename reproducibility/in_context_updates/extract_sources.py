"""Import the scientific subset of the completed Qwen delivery, without inference.

All archive members are checked before any output is written. Raw scientific
bytes are unchanged; gzip is lossless and deterministic. No cloud code is run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def extract(archive, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh scientific-subset directory")
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError("Duplicate archive member")
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or ':' in n or '\\' in n:
                raise ValueError("Unsafe archive member")
        manifest = json.loads(z.read('EXPORT_MANIFEST.json'))
        if set(names) != set(manifest['files']) | {'EXPORT_MANIFEST.json'}:
            raise ValueError("Undeclared/missing archive members")
        for name, sha in manifest['files'].items():
            if digest(z.read(name)) != sha:
                raise ValueError('Archive integrity failure: ' + name)
        for name, sha in json.loads(z.read('lp1/MANIFEST.json'))['files'].items():
            if digest(z.read('lp1/' + name)) != sha:
                raise ValueError('Frozen package integrity failure: ' + name)
        selected = {}
        for name in names:
            if name.startswith(('lp1/inputs/built/', 'lp1/source/icmh/', 'lp1/protocol/')):
                selected[name] = name.removeprefix('lp1/')
            if name.startswith('lp1/provenance/source_v2/icmh/'):
                selected[name] = name.removeprefix('lp1/provenance/')
            if name.startswith('state/analysis/'):
                selected[name] = name.replace('state/analysis/', 'expected/') + '.gz'
            if name.startswith('state/runs/') and name.endswith('/journal.jsonl'):
                selected[name] = name.replace('state/runs/', 'journals/') + '.gz'
        for name in ('PARSER_AMENDMENT.md', 'checks/test_label_precedence.py',
                     'checks/score_sensitivity.py', 'review/parser_change.diff'):
            selected['lp1/' + name] = name
        for name in ('PARSER_FREEZE.json', 'QUALIFICATION_FREEZE.json', 'FINAL_FREEZE.json',
                     'MODEL_RECEIPT.json', 'RUNTIME_FREEZE.json', 'PHASE_GATE.json'):
            selected['state/' + name] = 'freezes/' + name
        rows = []
        for member, relative in sorted(selected.items()):
            raw = z.read(member)
            data = gzip.compress(raw, mtime=0) if relative.endswith('.gz') else raw
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            rows.append(dict(path=relative, member=member, bytes=len(data), sha256=digest(data),
                             source_sha256=digest(raw), transformation='gzip-lossless' if data != raw else 'none'))
        provenance = dict(schema_version=1, archive=Path(archive).name,
            archive_sha256=digest(Path(archive).read_bytes()),
            archive_bytes=Path(archive).stat().st_size,
            verified_archive_members=len(manifest['files']), terminal_status=manifest['status'],
            files=rows, omitted='Cloud orchestration, allocation/billing/host logs, synthetic oracle runs, duplicated pre-amendment inputs, broad upstream preparation pool.',
            boundary='Local delivered-byte verification, not runtime attestation. Frozen metadata paths are historical, never replay dependencies.')
        (output / 'MANIFEST.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')
    return dict(imported=len(rows), verified=provenance['verified_archive_members'], archive_sha256=provenance['archive_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(extract(a.archive, a.output), indent=2))
