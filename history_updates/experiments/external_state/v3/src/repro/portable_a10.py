"""Verified, unpacked inputs for the supplied-plan recomposition.

The release manifest binds this mapping and every referenced source file. The
mapping records exact member bytes extracted from the verified original ZIP.
This is an explicit packaging derivative, not a claim of original ZIP identity.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

SOURCE_HASH='b7775b696632ea15f3d173089dab8f70ae9cf0a06612ced46aae2b77ed07766f'


class VerifiedDirectoryArchive:
    def __init__(self, root):
        self.root=Path(root).resolve()
        binding=json.loads((self.root/'A10_BINDING.json').read_bytes())
        if binding.get('schema')!='A10_EXTRACTED_DEPENDENCIES_V1' or binding.get('source_archive_sha256')!=SOURCE_HASH:
            raise ValueError('Unexpected A10 source binding')
        self.files=binding['files']
        prefix='reproducibility/question_blind_repair/'
        expected={prefix+'scores/designs.jsonl.gz',prefix+'supplied/verification/tools/replay.py'}
        expected |= {prefix+f'scores/primary-seed{s}-part{p:02d}.jsonl.gz' for s in (0,1) for p in range(1,9)}
        if set(self.files)!=expected:
            raise ValueError('Missing or extra A10 replay member')
        self.paths={}
        for name,row in self.files.items():
            relative=row['file'];parts=PurePosixPath(relative)
            if parts.is_absolute() or '..' in parts.parts or '\\' in relative or ':' in relative:
                raise ValueError('Unsafe A10 member mapping')
            path=(self.root/relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Missing A10 source member')
            with path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=row['bytes'] or digest!=row['sha256']:
                raise ValueError('Changed A10 member: '+name)
            self.paths[name]=path

    def __enter__(self):return self
    def __exit__(self,*args):return False
    def namelist(self):return list(self.files)
    def read(self,name):return self.paths[name].read_bytes()
