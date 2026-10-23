"""Validate uncached repeats and historical-return requests, separately from accuracy."""
import argparse,json
from pathlib import Path
from prepare_data import ROOT,load,gzrows,dump
from ranker_contract import choose,key

def read(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(l) for l in f]
def validate(main,repeat,returns,loader_expected=None,loader_actual=None):
    refs={r['request_id']:r for r in read(main)}
    logical={r['logical_id']:r for r in gzrows(ROOT/'prepared/LOGICAL_ROWS.jsonl.gz')}
    controls=load(ROOT/'prepared/CONTROLS.json')
    totals={}
    for name,file,idsfile,logicalids in [
        ('repeat',repeat,'REPEAT_REQUEST_IDS.json',controls['repeat_rows']),
        ('return',returns,'RETURN_REQUEST_IDS.json',[lid for t in controls['return_programs'] for lid in t['sequence']])]:
        actual=read(file);ids=load(ROOT/'prepared'/idsfile)
        if len(actual)!=len(ids) or any(r['request_id']!=i for r,i in zip(actual,ids)):raise ValueError('INCOMPLETE_UNCACHED_CONTROL_SCHEDULE')
        mismatches=[];offset=0;different_full_logits=0
        for lid in logicalids:
            row=logical[lid];n=len(row['score_request_ids']);batch=actual[offset:offset+n];offset+=n
            original=[refs[i] for i in row['score_request_ids']]
            a=choose(row['candidates'],[x['margin'] for x in batch]);b=choose(row['candidates'],[x['margin'] for x in original])
            if (key(a) if a else None)!=(key(b) if b else None):mismatches.append(lid)
            different_full_logits+=sum(x.get('full_logits_sha256')!=y.get('full_logits_sha256') for x,y in zip(batch,original))
            if any(x.get('input_token_ids')!=y.get('input_token_ids') for x,y in zip(batch,original)):raise ValueError('SAME_CURRENT_REQUEST_DIFFERENT_TOKENS')
        totals[name]={'actual_uncached_scores':len(actual),'logical_reads':len(logicalids),'selection_mismatches':mismatches,
            'different_full_logits_hashes':different_full_logits,'complete':True}
    if loader_expected or loader_actual:
        if not loader_expected or not loader_actual:raise ValueError('BOTH_LOADER_FILES_REQUIRED')
        ex=load(loader_expected);ac=read(loader_actual)
        if len(ex)!=len(ac) or any(x['request_id']!=y['request_id'] for x,y in zip(ex,ac)):raise ValueError('LOADER_SCHEDULE')
        maximum=max(abs(x[k]-y[k]) for x,y in zip(ex,ac) for k in ['yes_logit','no_logit','margin'])
        totals['loader']={'max_absolute_difference':maximum,'pass':maximum<=1e-4}
    totals['selection_invariance_pass']=all(not totals[k]['selection_mismatches'] for k in ('repeat','return'))
    return totals
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('main','repeat','returns','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--loader-expected');p.add_argument('--loader-actual');a=p.parse_args()
    result=validate(a.main,a.repeat,a.returns,a.loader_expected,a.loader_actual);dump(a.out,result);print(json.dumps(result,indent=2))
