"""Verify the explicit public artifact inventory without network/model access."""
from __future__ import annotations

import re

from .common import ROOT, load_json, safe_path, sha256


def verify(root=ROOT) -> dict:
    manifest = load_json(root / "reproducibility" / "manifest.json")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported public manifest")
    if not manifest["files"] or not manifest["experiments"]:
        raise ValueError("The public artifact inventory is not yet populated")
    seen = set()
    inventory = {}
    for row in manifest["files"]:
        name = row["path"]
        if name.casefold() in seen:
            raise ValueError("Duplicate artifact binding")
        seen.add(name.casefold())
        if not isinstance(row['bytes'], int) or row['bytes'] < 0 or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid artifact size/hash binding')
        path = safe_path(name, root)
        if not path.is_file():
            raise ValueError(f"Required public artifact missing: {name}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Public artifact changed: {name}")
        inventory[name] = row
    identities = set()
    for study in manifest['experiments']:
        if study['id'] in identities or not study['artifact_paths']:
            raise ValueError('Duplicate or empty experiment binding')
        identities.add(study['id'])
        for name in study['artifact_paths'] + ([study['claim_contract']] if study.get('claim_contract') else []):
            if name not in inventory:
                raise ValueError('Experiment references an unbound artifact: ' + name)
        for model in study['models']:
            if not model.get('id') or not re.fullmatch(r'[0-9a-f]{40}', model.get('revision', '')):
                raise ValueError('Immutable model revision missing')
    for name in manifest.get('source_manifests', []):
        if name not in inventory:
            raise ValueError('Unbound source manifest')
        for row in load_json(safe_path(name, root))['files']:
            if row['path'] not in inventory or row['sha256'] != inventory[row['path']]['sha256']:
                raise ValueError('Source adaptation identity does not match public inventory: ' + row['path'])
    for item in manifest.get('external_artifacts', []):
        if item.get('public_url') is None and item['status'] not in ('unavailable_to_external_reproducer', 'not_retained'):
            raise ValueError('Unavailable external artifact must not be advertised as public')
        for row in item.get('members', []):
            # Validate names syntactically without pretending the files are local.
            safe_path(row['path'], root)
            if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or row['bytes'] < 0:
                raise ValueError('Invalid external member identity')
    return {"status": "artifact_hashes_verified", "files": len(seen),
            "experiments": len(identities),
            "scope": "committed public evidence; not a new model reproduction"}


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2))
