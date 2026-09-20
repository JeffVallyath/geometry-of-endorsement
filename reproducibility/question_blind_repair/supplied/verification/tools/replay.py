"""Independent, standard-library CPU reconstruction from retained raw records.

No scientific implementation is imported. No neural model, training, bootstrap,
threshold change or additional independent observation is created here.
"""
import argparse
from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

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

def main(args):
    output=args.output
    if output.exists():raise FileExistsError('Use a fresh replay output directory')
    output.mkdir(parents=True)
    capsule=args.capsule;original=capsule/'original';contract=load(capsule/'EXECUTION_CONTRACT.json')
    manifest=load(capsule/'RUN_CAPSULE.json')
    for name,expected in manifest['files'].items():
        p=capsule/name
        if not p.resolve().is_relative_to(capsule.resolve()) or p.is_symlink() or sha(p)!=expected:
            raise ValueError('Frozen capsule member differs: '+name)
    old_roots={0:args.seed0,1:args.seed1}
    comparisons=[];audits=[];all_rows={name:[] for name in ('targets','families','reverse_targets','donor_agreement')};events=[]
    worked=[];retained_new={}
    for phase,roots in [('anchor',[contract['anchor']]),('repeat',contract['roots'])]:
        for seed in (0,1):
            events.append(audit_events(args.fresh/f'{phase}-seed{seed}'/'EVENTS.jsonl',roots))
            for rid in roots:
                reverse=phase=='repeat' and rid in contract['reverse_roots']
                old,old_seal,oa=read_case(old_roots[seed]/'final'/rid,reverse)
                new,new_seal,na=read_case(args.fresh/f'{phase}-seed{seed}'/'cases'/rid,reverse)
                old['physical'].sort(key=lambda r:(r['context'],r['query_id']))
                new['physical'].sort(key=lambda r:(r['context'],r['query_id']))
                c=compare(old,new);sc=compare(old_seal,new_seal,'/CONTEXT_SEAL')
                c.update(phase=phase,root_id=rid,seed=seed,rows=len(new['physical']),seal=sc)
                c['all_score_and_binding_fields_pass']=c['tolerance_pass'] and sc['tolerance_pass']
                comparisons.append(c);audits.append(dict(phase=phase,root_id=rid,seed=seed,old=oa,fresh=na))
                design=original/'generated'/rid
                root,questions,plan=load(design/'root.json'),load(design/'questions.json'),load(design/'PLAN.json')
                if new_seal['mask']!=load(design/'MASK.json'):raise ValueError('Frozen mask mismatch')
                if phase=='repeat':
                    for label,panel in [('archived',old),('fresh',new)]:
                        m=metrics(root,questions,plan,panel,reverse)
                        for kind,rows in m.items():all_rows[kind].extend(dict(dataset=label,**r) for r in rows)
                    retained_new[rid,seed]=(new,questions,root)
    for kind,rows in all_rows.items():table(output/(kind.upper()+'.csv'),rows)
    # The named worked example is fixed by the handoff, never chosen from new outcomes.
    panel,questions,root=retained_new['QBR1-FINAL-0127',1]
    index={(r['context'],r['query_id']):r for r in panel['physical']}
    slots=[('A_history1','o1/INV_PAIR_NLL_s1'),('A_history2','o2/INV_PAIR_NLL_s1'),
           ('B16_23_history2','B/o2'),('C24_31_history2','C/o2'),('D_reference','o0/INV_PAIR_NLL_s1'),
           ('RB16_23_history2','RB/o2'),('RC24_31_history2','RC/o2')]
    for q in questions:
        s=q['spec']
        if '-d0-' not in q['query_id'] or s['p']!=0:continue
        desired=(s['kind']=='direct' and s['a'] in (1,2)) or (s['kind'] in ('same','both','either') and {s['a'],s['b']}=={1,2})
        if not desired:continue
        row=dict(query_id=q['query_id'],question=q['text'],gold_label=q['labels'][truth(root['terminal']['values'],s)])
        for label,ctx in slots:
            r=index[ctx,q['query_id']];row[label]=q['labels'][r['prediction']] if r['valid'] else 'INVALID'
        worked.append(row)
    if len(worked)!=5:raise ValueError('Frozen worked-example question mapping differs')
    table(output/'WORKED_EXAMPLE.csv',worked)
    summary={}
    for dataset in ('archived','fresh'):
        target=[r for r in all_rows['targets'] if r['dataset']==dataset]
        families=[r for r in all_rows['families'] if r['dataset']==dataset]
        reverse=[r for r in all_rows['reverse_targets'] if r['dataset']==dataset]
        summary[dataset]=dict(selected_cases=6,case_seed_blocks=12,target_rows=len(target),
            baseline_W=sum(r['W'] for r in target),B_repairs=sum(r['B_repair'] for r in target),
            C_repairs=sum(r['C_repair'] for r in target),B_remaining_qualifying_witnesses=sum(r['B_qualifying_witness'] for r in target),
            eligible_RB_targets=sum(r['eligible_history_dependent_error'] for r in reverse if r['arm']=='RB'),
            eligible_RC_targets=sum(r['eligible_history_dependent_error'] for r in reverse if r['arm']=='RC'),
            response_counts=[dict(family=fam,arm=arm,**{key:sum(r[key] for r in families if r['family']==fam and r['arm']==arm)
                for key in ('scheduled','correct','baseline_correct','harm','benefit','invalid')}) for fam in ('direct','joint','opposes','all34') for arm in ('A','B','C','D')])
    comparison_summary=dict(status='ALL_RETAINED_SCORE_BINDINGS_REPRODUCED' if all(r['all_score_and_binding_fields_pass'] for r in comparisons) else 'DISCREPANCIES_RETAINED',
        post_hoc_selection=True,primary_bootstrap_unchanged=True,new_independent_observations=0,
        blocks=len(comparisons),physical_rows=sum(r['rows'] for r in comparisons),
        score_binding_exact_blocks=sum(r['score_binding_exact_equal'] and r['seal']['score_binding_exact_equal'] for r in comparisons),
        max_score_delta=max(r['max_delta'] for r in comparisons),comparisons=comparisons)
    save(output/'REPEAT_COMPARISON.json',comparison_summary)
    save(output/'RAW_AND_BOUNDARY_AUDIT.json',dict(cases=audits,events=events))
    save(output/'SELECTED_RECOUNT.json',summary)
    save(output/'CPU_REPLAY.json',dict(status='CPU_REPLAY_COMPLETE',at=dt.datetime.now(dt.timezone.utc).isoformat(),
        python=sys.version,independent_implementation_no_scientific_imports=True,
        original_capsule_sha256=contract['original_capsule_sha256'],extension_manifest_sha256=sha(capsule/'RUN_CAPSULE.json'),
        comparison_status=comparison_summary['status'],physical_rows=comparison_summary['physical_rows'],
        data_products={p.name:sha(p) for p in output.iterdir() if p.is_file()},new_population_inference=False))
    print(json.dumps(dict(comparison={k:v for k,v in comparison_summary.items() if k!='comparisons'},
          selected={k:{x:y for x,y in v.items() if x!='response_counts'} for k,v in summary.items()}),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capsule',type=Path,required=True)
    p.add_argument('--seed0',type=Path,required=True)
    p.add_argument('--seed1',type=Path,required=True)
    p.add_argument('--fresh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
