"""Strict integrity checks for ordinary, visible reviewer input files."""
import hashlib
import json
from pathlib import Path, PurePosixPath


class EvidenceError(ValueError):
    pass


class MissingEvidence(FileNotFoundError):
    pass


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def inside(root, name):
    root=Path(root).resolve();rel=PurePosixPath(name)
    if (not name or not rel.parts or rel.is_absolute() or '\\' in name or ':' in name
            or '..' in rel.parts or '\x00' in name or rel.as_posix()!=name
            or any(p.endswith((' ','.')) for p in rel.parts)):
        raise EvidenceError('Paths must be normalized, relative and portable')
    path=root.joinpath(*rel.parts);current=path
    while current!=root:
        if current.is_symlink() or getattr(current,'is_junction',lambda:False)():
            raise EvidenceError('Linked paths are not permitted')
        current=current.parent
    if not path.resolve().is_relative_to(root):raise EvidenceError('Path escapes repository')
    return path


def checked_file(root, name, record):
    path=inside(root,name)
    if not path.is_file():raise MissingEvidence('Missing included file: '+name)
    if path.stat().st_size!=record['bytes'] or digest_file(path)!=record['sha256']:
        raise EvidenceError('Included file differs: '+name)
    return path


def inventory(root):
    result=set()
    for path in Path(root).rglob('*'):
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
            raise EvidenceError('Linked path in repository')
        if path.is_file() and '__pycache__' not in path.parts:
            result.add(path.relative_to(root).as_posix())
    return result


class InputStore:
    def __init__(self,root):
        self.root=Path(root).resolve()
        binding_path=self.root/'provenance/INPUTS_BINDING.json'
        if not binding_path.is_file():raise MissingEvidence('Missing visible-input binding')
        binding=json.loads(binding_path.read_bytes())
        if binding['file']!='provenance/INPUTS.json':raise EvidenceError('Unexpected input manifest location')
        manifest=checked_file(self.root,binding['file'],binding)
        self.index=json.loads(manifest.read_bytes())
        if self.index.get('schema')!='REVIEWER_VISIBLE_INPUTS_V1':raise EvidenceError('Unknown input schema')

    def verify(self,identifier):
        if identifier not in self.index['studies']:raise MissingEvidence('Study is outside this reviewer bundle: '+identifier)
        row=self.index['studies'][identifier];root=inside(self.root,row['root']);files=row['files']
        if not files:raise MissingEvidence('Empty study input inventory')
        if len({n.casefold() for n in files})!=len(files):raise EvidenceError('Case-colliding input inventory')
        for name,record in files.items():checked_file(root,name,record)
        if inventory(root)!=set(files):raise EvidenceError('Unexpected study input inventory: '+identifier)
        return root,{'status':'PASS','study':identifier,'verified_files':len(files),
                     'verified_bytes':sum(r['bytes'] for r in files.values()),
                     'input_mode':'Visible included files; no packs, downloads or extraction step'}


def verify_release(root):
    root=Path(root).resolve();path=root/'provenance/RELEASE_MANIFEST.json'
    if not path.is_file():raise MissingEvidence('Missing reviewer release manifest')
    manifest=json.loads(path.read_bytes())
    if manifest.get('schema')!='REVIEWER_RELEASE_MANIFEST_V1':raise EvidenceError('Unknown release schema')
    files=manifest['files']
    for name,record in files.items():checked_file(root,name,record)
    actual={n for n in inventory(root) if not n.startswith('reproduced/')}
    if actual!=set(files)|{'provenance/RELEASE_MANIFEST.json'}:
        raise EvidenceError('Missing or extra release files')
    inputs=InputStore(root)
    return {'status':'PASS','verified_files':len(files),'included_studies':list(inputs.index['studies']),
            'separate_evidence_packs_required':False,
            'scope':'Exact delivered bytes and file inventory; scientific reconstruction has its own RUN_RECEIPT.json'}
