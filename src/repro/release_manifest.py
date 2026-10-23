"""Verify published bytes independently of the private source archives."""
import argparse
import json
from pathlib import PurePosixPath

from .common import ROOT, load_json, safe_path, sha256


def local_runtime_file(relative):
    """Editable installation and reviewer outputs are never release payloads."""
    parts = PurePosixPath(relative).parts
    return (parts[0] == 'reproduced' or parts[0] in {'.git', '.revision-venv', '.venv', '.pytest_cache'}
            or '__pycache__' in parts or any(part == '.git' for part in parts)
            or parts[:2] == ('src', 'anonymous_review.egg-info'))


def verify(root=ROOT, *, strict=False):
    root = root.resolve()
    manifest = load_json(root / 'release_manifest.json')
    if manifest['schema'] != 'ANONYMOUS_REVIEW_RELEASE_V1':
        raise ValueError('Unknown release manifest schema')
    seen = set()
    for row in manifest['files']:
        relative = row['path']
        if relative.casefold() in seen or relative == 'release_manifest.json':
            raise ValueError('Duplicate or self-referential manifest entry')
        seen.add(relative.casefold())
        path = safe_path(relative, root)
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Release integrity mismatch: ' + relative)
    expected = {row['path'] for row in manifest['files']} | {'release_manifest.json'}
    actual = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError('Release contains a symbolic link: ' + relative)
        if path.is_file() and (strict or not local_runtime_file(relative)):
            actual.add(relative)
    if actual != expected:
        raise ValueError('Release inventory differs: ' + ', '.join(sorted(actual ^ expected)[:8]))
    return {'status': 'PASS', 'verified_files': len(seen), 'strict_inventory': strict,
            'scope': 'Delivered byte integrity, not proof of scientific validity or anonymity'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='Reject local installation and replay products too')
    print(json.dumps(verify(strict=parser.parse_args().strict), indent=2))


if __name__ == '__main__':
    main()
