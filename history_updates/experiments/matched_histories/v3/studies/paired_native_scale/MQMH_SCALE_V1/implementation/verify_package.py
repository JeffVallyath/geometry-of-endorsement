"""Verify immutable package payloads. No internet, writes, or model calls."""
from pathlib import Path, PurePosixPath
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
def verify():
    manifest=ROOT/'MANIFEST.sha256'
    if not manifest.is_file():raise ValueError('Missing package manifest; do not derive it')
    seen=set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest,relative=line.split('  ',1);p=PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or relative in seen:raise ValueError('Unsafe/duplicate manifest entry')
        seen.add(relative);f=ROOT/relative
        if f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Changed or missing: '+relative)
    required={'SCALE_GOAL.txt','SCALE_PROTOCOL.md','SCALE_CONFIG.json','SCALE_ACCEPTANCE_TESTS.md','SCALE_LAUNCHER.txt'}
    if not required<=seen:raise ValueError('Missing governing payload')
    n=len((ROOT/'SCALE_LAUNCHER.txt').read_text(encoding='utf-8'))
    if n>=4000:raise ValueError('Launcher must be under4000 characters')
    c=json.loads((ROOT/'SCALE_CONFIG.json').read_text());assert c['evaluation']['reads_two_models']==10812
    print(json.dumps({'verified_payloads':len(seen),'launcher_characters':n,'status':'PASS'}))
if __name__=='__main__':verify()
