"""Saved-score functions separated from the original neural/private worker imports.

panel_index, current_rows and compare preserve the frozen function bodies.
load_panel verifies the score member against its public completion record.
Unavailable raw batches and context seals are omitted from that record.
The public entry point verifies every declared file before replay.
"""
import gzip
import json
import math
from pathlib import Path
from bindings import load, sha
from analyze import CONDITIONS


def panel_index(panel):
    from analyze import decode
    physical={(r['context'],r['query_id']):r for r in panel['physical']}
    if len(physical)!=len(panel['physical']): raise ValueError('Duplicate physical rows')
    qs={q for c,q in physical};expected={'NATIVE_FINAL'}|{f'o{o}/{c}' for o in range(8) for c in CONDITIONS}
    if set(panel['contexts'])!=expected or len(qs)!=34: raise ValueError('Missing fixed rows')
    expected_physical={(panel['aliases'].get(c,c),q) for c in expected for q in qs}
    if set(physical)!=expected_physical:raise ValueError('Missing/extra physical response')
    if any(r['batch_size']!=1 or not r['master_unchanged'] for r in physical.values()):
        raise ValueError('Single-query immutable-cache contract failed')
    decoded={}
    for context in expected:
        target=panel['aliases'].get(context,context)
        if target in panel['aliases']: raise ValueError('Cyclic alias')
        if panel['contexts'][context]['cache_identity']!=panel['contexts'][target]['cache_identity']:
            raise ValueError('Alias lacks exact prefix/cache/metadata identity')
        for qid in qs: decoded[context,qid]=decode(physical[target,qid])
    return decoded


def load_panel(folder):
    folder = Path(folder)
    done = load(folder / 'DONE.json')
    if sha(folder / 'SCORES.json.gz') != done['files']['SCORES.json.gz']:
        raise ValueError('Completed score panel hash changed')
    with gzip.open(folder / 'SCORES.json.gz', 'rt', encoding='utf8') as stream:
        return json.load(stream)


def current_rows(panel):
    physical={(r['context'],r['query_id']):r for r in panel['physical']}
    return {(context,qid):physical[panel['aliases'].get(context,context),qid]
            for context in panel['contexts'] for qid in sorted({r['query_id'] for r in panel['physical']})}


def compare(old,new,old_format=False):
    if old_format:
        raw={(r['context'],r['query_id']):dict(logps=r['logps'],**r['recorded_checks']) for r in old['physical']}
        original={(c,q):raw[old['aliases'].get(c,c),q] for c,q in current_rows(new)}
    else: original=current_rows(old)
    fresh=current_rows(new);differences=[];failures=[];maximum=0.
    for key,row in fresh.items():
        before=original[key]
        delta=max(abs(float(a)-float(b)) for a,b in zip(before['logps'],row['logps']))
        finite_delta=math.isfinite(delta)
        changed=[f for f in ('prediction','valid','tokens_sha256','finite','tie','argmax_in_labels') if before[f]!=row[f]]
        maximum=max(maximum,delta) if finite_delta else maximum
        if delta!=0 or changed:
            differences.append(dict(context=key[0],query_id=key[1],max_logp_delta=delta if finite_delta else None,changed_fields=changed,
                original_logps=before['logps'],fresh_logps=row['logps']))
        if not finite_delta or delta>1e-6 or changed:failures.append(differences[-1])
    return dict(status='PASS' if not failures else 'NUMERICAL_REPLAY_FAILURE',rows=len(fresh),
        tolerance=1e-6,max_logp_delta=maximum,all_discrepancies=differences,failed_rows=failures)
