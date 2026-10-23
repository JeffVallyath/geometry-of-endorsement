"""Original standard-library score-reader and comparator definitions.

The public wrapper reconstructs statistics from recovered score projections.
Original native-cache receipts and the raw journal remain outside this release.

No scientific implementation is imported. No neural model, training, bootstrap,
threshold change or additional independent observation is created here.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

TIMING={'batch_seconds','qualification_seconds','preprocessing_seconds','compile_seconds'}
load=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def table(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def stripped(value):
    if isinstance(value,dict):return {k:stripped(v) for k,v in value.items() if k not in TIMING}
    if isinstance(value,list):return [stripped(v) for v in value]
    return value
def truth(values,spec):
    x=values[spec['a']][spec['p']];y=values[spec['b']][spec['p']]
    if spec['kind']=='direct':return x
    if spec['kind']=='opposes':return 1-x
    if spec['kind']=='same':return int(x==y)
    if spec['kind']=='both':return int(bool(x) and bool(y))
    if spec['kind']=='either':return int(bool(x) or bool(y))
    raise ValueError('Unknown frozen question kind')

def decode(row):
    lp=row['logps']
    if len(lp)!=2:raise ValueError('Not the frozen binary scorer')
    finite=all(math.isfinite(x) for x in lp)
    if finite!=row['finite']:raise ValueError('Finite binding')
    valid=finite and not row['tie'] and row['argmax_in_labels']
    if bool(valid)!=row['valid']:raise ValueError('Validity binding')
    if finite:
        if row['prediction']!=int(lp[1]>lp[0]) or row['tie']!=(lp[0]==lp[1]):raise ValueError('Hard-answer binding')
        mass=sum(math.exp(x) for x in lp)
        if abs(mass-row['answer_mass'])>1e-12 or not 0<mass<=1.00001:raise ValueError('Answer-mass binding')
        if max(math.exp(x) for x in lp)>max(0,1-mass)+1e-6 and not row['argmax_in_labels']:
            raise ValueError('Full-vocabulary argmax certificate contradiction')
    return dict(prediction=row['prediction'],valid=bool(valid))

def read_case(folder,reverse):
    done=load(folder/'DONE.json')
    for name,digest in done['files'].items():
        if Path(name).name!=name or sha(folder/name)!=digest:raise ValueError('Case member changed: '+name)
    with gzip.open(folder/'SCORES.json.gz','rt',encoding='utf8') as f:panel=json.load(f)
    allowed=lambda key:reverse or not key.startswith(('RB/','RC/'))
    panel=dict(panel,physical=[r for r in panel['physical'] if allowed(r['context'])],
         contexts={k:v for k,v in panel['contexts'].items() if allowed(k)},
         tokens={k:v for k,v in panel['tokens'].items() if allowed(k)},
         aliases={k:v for k,v in panel.get('aliases',{}).items() if allowed(k)})
    index={(r['context'],r['query_id']):r for r in panel['physical']}
    if len(index)!=len(panel['physical']):raise ValueError('Duplicate score row')
    journal={};counts=Counter();raw_bindings={}
    with (folder/'RAW_BATCHES.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if not allowed(r['context']):continue
            if len(r['results'])!=1 or len(r['tokens'])!=1:raise ValueError('Single-question readout changed')
            context=r['context']
            if r['seed']!=panel['seed'] or r['artifact_sha256']!=panel['contexts'][context]['artifact_sha256']:
                raise ValueError('Raw context/seed binding')
            for qid,value in r['results'].items():
                key=(context,qid)
                if key in journal:raise ValueError('Duplicate physical journal row')
                if dict(context=context,query_id=qid,**value)!=index[key]:raise ValueError('Compact/raw score disagreement')
                if r['tokens'][qid]!=panel['tokens'][context][qid]:raise ValueError('Compact/raw token disagreement')
                if not value['master_unchanged'] or value['batch_size']!=1 or value['physical_slot']!=0:
                    raise ValueError('Master copy or single-query isolation failed')
                if r['batch']!=value['physical_batch'] or r['batch']!=counts[context]:
                    raise ValueError('Physical journal batch binding')
                raw_bindings.setdefault(context,{})[qid]={k:r[k] for k in ('seed','batch','artifact_sha256','qualifications')}
                decode(value);journal[key]=value;counts[context]+=1
    if set(journal)!=set(index) or any(v!=34 for v in counts.values()):raise ValueError('Raw inventory incomplete')
    seal=load(folder/'CONTEXT_SEAL.json')
    for field in ('contexts','refresh_receipts'):seal[field]={k:v for k,v in seal[field].items() if allowed(k)}
    if seal['question_files_opened'] or seal['contexts']!=panel['contexts']:raise ValueError('Seal/panel boundary differs')
    for key,r in seal['refresh_receipts'].items():
        if not all(r[x] for x in ('metadata_unchanged','all_keys_unchanged','outside_values_unchanged')):
            raise ValueError('Copy/outside-mask control failed')
        if len(r['layers_audit'])!=42 or any(not t['keys_unchanged'] or not t['outside_values_unchanged'] for t in r['layers_audit']):
            raise ValueError('Native cache layer audit incomplete')
    panel['raw_bindings']=raw_bindings
    return panel,seal,dict(raw_rows=len(index),contexts=len(counts),all_declared_case_hashes_verified=True,
                          raw_score_token_binding_pass=True,cache_and_blind_boundary_receipts_pass=True)

def compare(old,new,path=''):
    differences=[];failures=[];maximum=0.
    def walk(a,b,where,numerical=False):
        nonlocal maximum
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):failures.append(dict(path=where,kind='keys',old=sorted(a),new=sorted(b)))
            for k in sorted(set(a)&set(b)):walk(a[k],b[k],where+'/'+k,numerical or k in ('logps','answer_mass'))
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):failures.append(dict(path=where,kind='length',old=len(a),new=len(b)))
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,where+'/'+str(i),numerical)
        elif a!=b:
            delta=abs(a-b) if numerical and isinstance(a,(int,float)) and isinstance(b,(int,float)) else None
            ok=delta is not None and math.isfinite(delta) and delta<=1e-6
            if delta is not None and math.isfinite(delta):maximum=max(maximum,delta)
            row=dict(path=where,old=a,new=b,delta=delta,within_original_tolerance=ok)
            differences.append(row)
            if not ok:failures.append(row)
    walk(stripped(old),stripped(new),path)
    return dict(score_binding_exact_equal=not differences and not failures,tolerance_pass=not failures,
         tolerance=1e-6,max_delta=maximum,discrepancies=differences,failures=failures,
         excluded_wall_clock_fields=sorted(TIMING),raw_record_exact_equal=old==new)

def metrics(root,questions,plan,panel,reverse):
    seed=panel['seed'];rid=root['root_id']
    records={(r['context'],r['query_id']):decode(r) for r in panel['physical']}
    qs={q['query_id']:q for q in questions};gold={k:truth(root['terminal']['values'],q['spec']) for k,q in qs.items()}
    def context(arm,o):
        return f'o{o}/INV_PAIR_NLL_s{seed}' if arm=='A' else f'o0/INV_PAIR_NLL_s{seed}' if arm=='D' else f'{arm}/o{o}'
    def response(arm,o,q):return records[context(arm,o),q]
    def good(arm,o,q):
        r=response(arm,o,q);return r['valid'] and r['prediction']==gold[q]
    def all_good(arm,ids):return all(good(arm,o,q) for o in range(8) for q in ids)
    targets=[];families=[];reverses=[];donors=[]
    for q,mapping in plan['gates']['targets'].items():
        controls=plan['gates']['direct']+mapping['same_pair']+mapping['cross_pair']
        required=[q]+controls
        def witness(arm):
            native=records['NATIVE_FINAL',q]
            pairs=[p for p in plan['structural_pairs'] if all(response(arm,o,q)['valid'] for o in p)
                   and response(arm,p[0],q)['prediction']!=response(arm,p[1],q)['prediction']]
            return bool(native['valid'] and native['prediction']==gold[q] and good(arm,0,q) and all_good(arm,controls) and pairs)
        baseline=witness('A')
        row=dict(root_id=rid,seed=seed,query_id=q,draw=mapping['draw'],gold=gold[q],W=baseline)
        for arm in ('A','B','C','D'):
            row[arm+'_target_all_correct']=all_good(arm,[q])
            row[arm+'_controls_preserved']=all_good(arm,required)
            row[arm+'_repair']=baseline and row[arm+'_controls_preserved']
            row[arm+'_qualifying_witness']=witness(arm)
        targets.append(row)
        if reverse:
            for arm in ('RB','RC'):
                pairs=[p for p in plan['structural_pairs'] if all(response(arm,o,q)['valid'] for o in p)
                       and response(arm,p[0],q)['prediction']!=response(arm,p[1],q)['prediction']]
                clean=good('D',0,q);preserved=all_good(arm,controls)
                reverses.append(dict(root_id=rid,seed=seed,query_id=q,draw=mapping['draw'],arm=arm,
                    clean_reference_correct=clean,controls_preserved=preserved,matched_pairs=pairs,
                    eligible_history_dependent_error=bool(clean and preserved and pairs),
                    wrong_histories=[o for o in range(8) if not good(arm,o,q)],
                    invalid_histories=[o for o in range(8) if not response(arm,o,q)['valid']]))
    for draw in (0,1):
        for family,kinds in [('direct',{'direct'}),('joint',{'same','both','either'}),('opposes',{'opposes'}),('all34',None)]:
            ids=[q for q,v in qs.items() if int(q.split('-d')[1][0])==draw and (kinds is None or v['spec']['kind'] in kinds)]
            for arm in ('A','B','C','D'):
                slots=[(o,q) for o in range(8) for q in ids]
                families.append(dict(root_id=rid,seed=seed,draw=draw,family=family,arm=arm,
                    scheduled=len(slots),correct=sum(good(arm,o,q) for o,q in slots),
                    baseline_correct=sum(good('A',o,q) for o,q in slots),
                    harm=sum(good('A',o,q) and not good(arm,o,q) for o,q in slots),
                    benefit=sum(not good('A',o,q) and good(arm,o,q) for o,q in slots),
                    invalid=sum(not response(arm,o,q)['valid'] for o,q in slots),
                    all_histories_all_questions=all_good(arm,ids)))
    if reverse:
        for arm in ('RB','RC'):
            for q in qs:
                for o in range(8):
                    a,r=response('A',o,q),response(arm,o,q)
                    donors.append(dict(root_id=rid,seed=seed,arm=arm,query_id=q,origin=o,gold=gold[q],
                        donor_prediction=a['prediction'],reverse_prediction=r['prediction'],
                        donor_valid=a['valid'],reverse_valid=r['valid'],donor_correct=good('A',o,q),
                        reverse_correct=good(arm,o,q),clean_reference_correct=good('D',0,q),
                        donor_response_agreement=a['valid'] and r['valid'] and a['prediction']==r['prediction']))
    return dict(targets=targets,families=families,reverse_targets=reverses,donor_agreement=donors)

def audit_events(path,expected_roots):
    sealed=set();opened=Counter();begun=set()
    for line in path.read_text().splitlines():
        r=json.loads(line);rid=r.get('root_id')
        if r['event']=='ROOT_COMPILE_BEGIN':begun.add(rid)
        elif r['event']=='ROOT_CONTEXTS_SEALED':
            if rid not in begun:raise ValueError('Seal before compilation')
            sealed.add(rid)
        elif r['event']=='EVALUATION_FILE_OPEN':
            if rid not in sealed or not r['sealed']:raise ValueError('Pre-question boundary failed')
            opened[rid]+=1
    if sealed!=set(expected_roots) or set(opened)!=set(expected_roots):raise ValueError('Missing case boundary events')
    return dict(path=str(path),sha256=sha(path),roots=sorted(sealed),all_evaluation_opens_after_seal=True)
