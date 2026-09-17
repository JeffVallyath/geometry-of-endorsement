"""Fail-closed portable package and pre-evaluation source-collision verification."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root=ROOT):
    root=Path(root);manifest=root/'ARTIFACT_MANIFEST.json'
    if not manifest.is_file():raise FileNotFoundError('Missing required ARTIFACT_MANIFEST.json')
    data=json.loads(manifest.read_text())
    for name,record in data['files'].items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            raise ValueError('Nonportable artifact member')
        path=root/name
        if not path.is_file():raise FileNotFoundError('Missing required artifact: '+name)
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Artifact hash/size mismatch: '+name)
    audit=verify_fixed_populations(root)
    return dict(status='PASS',files=len(data['files']),remaining_collisions=0,
                fresh_unique_sources=audit['fresh_unique_sources'],model_evaluations=0)

def verify_fixed_populations(root=ROOT):
    from generate_cases import build, stable
    root=Path(root)
    pops,audit=build(root/'cases/PRIOR_SOURCE_HASHES.json')
    if audit['remaining_collisions']!=0:raise ValueError('Source-hash collision')
    for name,expected in pops.items():
        actual=[json.loads(line) for line in (root/f'cases/fixed/{name}.jsonl').read_text().splitlines()]
        if stable(expected)!=stable(actual):raise ValueError('Fixed population differs from declared generator: '+name)
    return audit

if __name__=='__main__':print(json.dumps(verify()))
