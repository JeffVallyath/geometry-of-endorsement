"""Verify the complete delivered payload inventory, without importing GPU code."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def verify():
    names=set()
    for l in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        h,name=l.split(None,1);name=name.strip();p=ROOT/name
        if name in names:raise ValueError('DUPLICATE_MANIFEST_ENTRY')
        if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('PATH_ESCAPE')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('HASH_MISMATCH '+name)
        names.add(name)
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts}
    if actual!=names:raise ValueError('PAYLOAD_INVENTORY_CHANGED')
    return {'manifest':'PASS','verified_payloads':len(names),'neural_inference':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
