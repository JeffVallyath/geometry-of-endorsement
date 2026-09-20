"""Extension manifest over the original verified closed-inventory reader."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

def sha(path):
    # This reader also runs in the provider's Python 3.10 bootstrap.
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def verify_directory(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'RUN_CAPSULE.json').read_text())
    if manifest['schema'] != 'QBR_REPAIR_VERIFICATION_CAPSULE_V1':
        raise ValueError('Wrong extension capsule')
    for name, expected in manifest['files'].items():
        target = root.joinpath(*PurePosixPath(name).parts)
        if target.is_symlink() or not target.resolve().is_relative_to(root) or sha(target) != expected:
            raise ValueError('Frozen extension member differs: ' + name)
    contract = json.loads((root / 'EXECUTION_CONTRACT.json').read_text())
    fields = ['max_active_instances', 'max_elapsed_hours_from_first_allocation', 'max_all_in_usd',
              'scored_hard_cap_including_recovery', 'compilation_hard_cap_including_recovery']
    if [contract[k] for k in fields] != [1, 2, 8, 16000, 500]:
        raise ValueError('New authorization envelope differs')
    original = json.loads((root / 'original/RUN_CAPSULE.json').read_text())
    if any(manifest['files']['original/' + n] != v for n, v in original['files'].items()):
        raise ValueError('Original scientific capsule members changed')
    return manifest

def extract(archive, out, expected):
    archive, out = Path(archive), Path(out)
    if sha(archive) != expected:
        raise ValueError('Outer capsule digest mismatch')
    with zipfile.ZipFile(archive) as z:
        name = 'original/fhv/borrowed/capsule_reader.py'
        manifest = json.loads(z.read('RUN_CAPSULE.json'))
        raw = z.read(name)
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
            raise ValueError('Original capsule reader differs')
        if any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in z.infolist()):
            raise ValueError('Symlink capsule member')
    scope = {'__name__': '_qbrv_borrowed_reader', '__file__': str(out / name)}
    exec(compile(raw, name, 'exec'), scope)
    scope['verify_directory'] = verify_directory
    return scope['extract'](archive, out, expected)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['extract', 'verify'])
    p.add_argument('--archive', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--sha256')
    p.add_argument('--root', type=Path)
    a = p.parse_args()
    result = extract(a.archive, a.out, a.sha256) if a.action == 'extract' else verify_directory(a.root)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
