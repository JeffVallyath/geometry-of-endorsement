"""Scoring-side fixed CHUNK=12 serving; never constructs an updated context.

Only public question text/answer symbols enter. Truth labels remain outside.
Exact-input reuse is scoped to this one compiled context and one helper call.
"""
import math
import time
from inputs import UpdatedContext, digest, _sync
from canonical.tasks import suffix_text

def serve_questions(context, questions, *, backbone):
    if type(context) is not UpdatedContext or context._artifact is None:
        raise ValueError('Already-compiled context required')
    if any(set(q)!={'text','labels'} for q in questions):
        raise ValueError('Only question text and public answer symbols accepted')
    art=context._artifact
    if backbone.cache_hash(art['cache'])!=context.artifact_sha256:
        raise ValueError('Immutable master changed')
    encoded=[];pending={}
    for q in questions:
        enc=backbone.encode(context.text,suffix_text(q['text']))
        if enc['prefix_ids']!=art['prefix_ids']:
            raise ValueError('Question crossed prefix token boundary')
        label_ids=backbone.label_ids(enc['full_text'],q['labels'])
        key=(q['text'],tuple(enc['suffix_ids']),tuple(label_ids))
        item=(q,enc,label_ids,key);encoded.append(item);pending.setdefault(key,item)
    results={};work=list(pending.values())
    for start in range(0,len(work),12):
        chunk=work[start:start+12]
        _sync(backbone);before=time.perf_counter()
        scored=backbone.ask_batch(art,[x[1]['suffix_ids'] for x in chunk],[x[2] for x in chunk],verify_master=True)
        _sync(backbone);seconds=time.perf_counter()-before
        if len(scored)!=len(chunk):raise ValueError('Scored batch length mismatch')
        for slot,(item,result) in enumerate(zip(chunk,scored,strict=True)):
            if not result['master_unchanged']:raise ValueError('Mutable master')
            results[item[3]]=(result,start//12,slot,seconds,len(chunk))
    out=[]
    for q,enc,ids,key in encoded:
        r,batch,slot,seconds,n=results[key]
        out.append(dict(schema='SCORED_ANSWER_V1',question=q['text'],question_sha256=digest(q['text']),
            prediction=r['prediction'],answer=q['labels'][r['prediction']],logps=r['logps'],
            answer_mass=r['answer_mass'],argmax_in_labels=r['argmax_in_labels'],tie=r['tie'],
            valid=bool(r.get('finite',False) and not r['tie'] and all(math.isfinite(x) for x in r['logps'])),
            master_unchanged=True,artifact_sha256=context.artifact_sha256,suffix_ids=enc['suffix_ids'],
            label_token_ids=ids,physical_batch=batch,physical_slot=slot,batch_size=n,batch_seconds=seconds))
    return out
