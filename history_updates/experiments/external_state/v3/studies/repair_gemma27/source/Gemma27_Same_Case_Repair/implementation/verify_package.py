from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def check(root):
    manifest=root/'MANIFEST.sha256';names=[]
    for line in manifest.read_text().splitlines():
        h,name=line.split(None,1);name=name.strip().lstrip('*');p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('HASH_MISMATCH: '+str(p))
        names.append(name)
    payloads={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts}
    if payloads!=set(names):raise ValueError('UNMANIFESTED_OR_MISSING_PAYLOAD')
    return len(names)
if __name__=='__main__':
    print('main_payloads',check(ROOT));print('inherited_payloads',check(ROOT/'inherited/PPR32_Repair_128'))
