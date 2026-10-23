"""Single owned-GPU attempt: TRAIN/DEV preflight, four fits, freeze, complete schedule."""
from pathlib import Path
import argparse
import collections
import datetime as dt
import json
import sys
import time
import numpy as np
import torch
from baseline_core import ENDPOINTS,canon,digest,direct_request,parse_program,gmello,mello,drive_coroutines
from model_runtime import ROOT,CFG,CONTRACT,read,publish,sha_file,check_time,Engine,Journal,Retriever,train
sys.path[:0]=[str(ROOT/'ppr/implementation'),str(ROOT/'ppr/vendor')]
from ppr_requests import record_view,score_requests
from ranker_contract import choose
from execution import build_store,execute
from codec import decode,norm
from pst1_core import Store
from state_grounding import current_payload

def aliases(case):
    multi=collections.defaultdict(set)
    for entity,label in case['labels'].items():multi[norm(label)].add(entity)
    return {key:next(iter(values)) for key,values in multi.items() if len(values)==1}
def snapshot(case,history,phase,question):
    store=build_store(case,history)
    view=store.view(0 if phase=='pre' else None)
    edges,payload=record_view(case,history,phase)
    assert sorted([s,p,o] for (s,p),o in view.items())==edges
    return {'question':question,'relation_registry':payload['relation_registry'],'current_records':payload['current_records']},edges

def endpoints_for_requests(engine,journal,items,retriever,nonce=None):
    """Items already own independently constructed current states; dedup complete method input only."""
    endpoint=engine.endpoint;demo=read(ROOT/'demonstrations.json');demo_text=(ROOT/'mello_demonstrations.txt').read_text(encoding='utf-8')
    keys=[digest({'endpoint':endpoint,'fingerprint':engine.fingerprint,'payload':p,'demonstrations':demo if endpoint=='GMELLO_LOCAL' else demo_text if endpoint=='MELLO_CURRENT' else None,'nonce':None if nonce is None else [nonce,i]}) for i,p in enumerate(items)]
    todo={k:p for k,p in zip(keys,items) if k not in journal.index['method']}
    results={}
    if endpoint.startswith(('QA_','PROGRAM_')):
        family=endpoint.split('_')[0];ks=list(todo);requests=[direct_request(todo[k],family) for k in ks]
        generated=engine.calls(requests,nonce=nonce)
        for key,g in zip(ks,generated):
            if family=='QA':out={'answer_text':g['text'],'status':'CAP_HIT' if g['cap_hit'] else 'GENERATED','trace':g}
            else:
                program,status=parse_program(g['text'],todo[key]);out={'answer_text':'','status':status,'program':program,'trace':g}
                if g['cap_hit']:out['status']='CAP_HIT'
                elif program:
                    edges=[[e['subject']['id'],e['relation']['id'],e['object']['id']] for e in todo[key]['current_records']]
                    try:
                        result=Store(edges).query(program['start_id'],program['path'])
                        out.update(entity=result['endpoint'],execution=result,status='VALID')
                    except ValueError as exc:out.update(status='INVALID_EXECUTION',error=str(exc))
            results[key]=out
    else:
        coroutines={k:(gmello(p,demo,retriever) if endpoint=='GMELLO_LOCAL' else mello(p,demo_text,retriever)) for k,p in todo.items()}
        results=drive_coroutines(coroutines,lambda reqs:engine.calls(reqs,nonce=nonce))
    for key,out in results.items():journal.add('method',{'method_id':key,'endpoint':endpoint,'payload':todo[key],'result':out,'nonce':nonce},'method_id')
    return [dict(journal.index['method'][k]['result'],method_id=k) for k in keys]

def invalid_control_output(value):
    if isinstance(value,dict):return bool(value.get('invalid_control_tokens',False)) or any(invalid_control_output(v) for v in value.values() if isinstance(v,(dict,list)))
    if isinstance(value,list):return any(invalid_control_output(v) for v in value)
    return False

def evaluate_case(engine,journal,case,rows,retriever,nonce=None):
    start=time.perf_counter();before_calls=len(journal.index['calls']);before_retrieval=retriever.seconds if retriever else 0.
    queries={q['query_id']:q['question'] for q in case['queries']};prepared=[]
    for row in rows:
        payload,edges=snapshot(case,row['history'],row['phase'],queries[row['query_id']])
        prepared.append((row,payload,edges))
    prep_seconds=time.perf_counter()-start
    if engine.endpoint.startswith('REPAIR_'):
        groups=[];requests=[]
        for row,payload,edges in prepared:
            cs,rs=score_requests(case,row['history'],row['phase'],payload['question']);groups.append(cs);requests.extend(rs)
        values=engine.calls(requests,kind='score',nonce=nonce);offset=0;outputs=[]
        for (row,payload,edges),cs in zip(prepared,groups):
            local=values[offset:offset+len(cs)];offset+=len(cs)
            selected=choose(cs,[v['margin'] for v in local]);outcome=execute(case,row['history'],row['phase'],selected)
            outputs.append({'outcome':outcome,'program':None if selected is None else {'start_id':selected['start_id'],'path':selected['path']},
                'selected':selected,'candidates':cs,'scores':local,'answer_text':outcome.get('answer',''),'call_ids':[v['call_id'] for v in local]})
    else:
        outputs=[]
        for (_,payload,_),out in zip(prepared,endpoints_for_requests(engine,journal,[p for _,p,_ in prepared],retriever,nonce)):
            if invalid_control_output(out):outcome={'entity':None,'status':'INVALID_OUTPUT'}
            elif out.get('entity') is not None:outcome={'entity':out['entity'],'status':'VALID'}
            else:outcome=decode(out.get('answer_text',''),aliases(case),set(case['labels']),truncated=out['status']=='CAP_HIT')
            if out['status'].startswith('INVALID'):outcome={'entity':None,'status':out['status']}
            outputs.append(dict(out,outcome=outcome))
    results=[dict(row,**out,question=payload['question'],state_sha256=digest(edges),request_payload_sha256=digest(payload)) for (row,payload,edges),out in zip(prepared,outputs)]
    timing={'wall_seconds':time.perf_counter()-start,'state_preparation_seconds':prep_seconds,'new_model_calls':len(journal.index['calls'])-before_calls,
        'retrieval_seconds':(retriever.seconds-before_retrieval) if retriever else 0.,'logical_reads':len(rows),'includes_state_retrieval_enumeration_execution_rendering':True}
    return results,timing

def dev_case(item):
    root=item['root'];initial=root['initial'];edits=root['edits'];post=Store(initial);post.apply(edits)
    labels={k:v[0] for k,v in root['labels'].items()}
    case={'case_id':item['key'],'labels':labels,'histories':{'H0':{'initial':initial,'edits':edits}},
          'queries':[{'query_id':f'{q["kind"]}:{q["index"]}','question':q['question']} for q in item['annotations']['questions']]}
    # DEV retains all source aliases in source structures; fixed common renderer uses source first labels.
    return case

def development(engine,journal,retriever,limit=None):
    items=read(ROOT/'development.json')['retained'];items=items if limit is None else items[:limit]
    counts=collections.Counter();rows=[];begin=time.perf_counter()
    for item in items:
        case=dev_case(item);jobs=[{'case_id':case['case_id'],'history':'H0','phase':'post','query_id':q['query_id'],'arm':engine.endpoint} for q in case['queries']]
        measured,_=evaluate_case(engine,journal,case,jobs,retriever)
        ann={f'{q["kind"]}:{q["index"]}':q for q in item['annotations']['questions']}
        for r in measured:
            q=ann[r['query_id']];state=Store(item['root']['initial']);state.apply(item['root']['edits']);truth=state.query(q['subject_id'],q['path'])['endpoint']
            counts['questions']+=1;counts['valid_outputs']+=r['outcome']['status']=='VALID';counts['correct']+=r['outcome']['entity']==truth
            if r.get('program') is not None:counts['parsed_programs']+=1
            rows.append(dict(r,correct=r['outcome']['entity']==truth))
    result={'endpoint':engine.endpoint,'counts':dict(counts),'seconds':time.perf_counter()-begin,'rows':rows,'protected_outcomes':False}
    publish(ROOT/'development'/f'{engine.endpoint}_{len(items)}.json',result)
    return result

def loader_check(engine):
    seed=engine.endpoint[-1];expected=read(ROOT/f'original_adapters/seed{seed}/LOADER_CANARY.json')
    actual=engine.calls(expected['requests'],kind='score',nonce='original_loader_check')
    error=max(abs(a[k]-b[k]) for a,b in zip(actual,expected['scores']) for k in ['yes_logit','no_logit','margin'])
    assert error<=expected['tolerance'],('ORIGINAL_LOADER_DISAGREEMENT',error)
    publish(ROOT/'evidence'/f'{engine.endpoint}_LOADER.json',{'status':'PASS','error':error,'tolerance':expected['tolerance']})

def preflight():
    for name,h in read(ROOT/'CODE_MANIFEST.json').items():assert sha_file(ROOT/name)==h,('FROZEN_CODE_CHANGED',name)
    journal=Journal();journal.status('DISPOSABLE_TRAIN_VALIDATION')
    canaries=[]
    for family in ('QA','PROGRAM'):
        path=ROOT/f'training_records/{family}_0_canary/COMPLETE.json'
        canaries.append(read(path) if path.exists() else train(family,0,True))
        journal.export()
    # Canary adapters are not main adapters and must never initialize a main fit.
    retriever=Retriever();engine=Engine('BASE',journal)
    dev_measure={}
    try:
        for endpoint in ('MELLO_CURRENT','GMELLO_LOCAL'):
            engine.endpoint=endpoint;result=development(engine,journal,retriever,limit=4)
            assert result['counts'].get('valid_outputs',0)>0,('BASELINE_NONFUNCTIONAL_ON_DEV',endpoint,result['counts'])
            dev_measure[endpoint]={k:v for k,v in result.items() if k!='rows'}
        engine.endpoint='QA_BASE_TECHNICAL'
        result=development(engine,journal,retriever,limit=4);dev_measure['QA']= {k:v for k,v in result.items() if k!='rows'}
        engine.endpoint='PROGRAM_BASE_TECHNICAL'
        result=development(engine,journal,retriever,limit=4);dev_measure['PROGRAM']={k:v for k,v in result.items() if k!='rows'}
    finally:engine.close()
    # Original adapter loader checks are TRAIN requests, not protected endpoint outcomes.
    for endpoint in ('REPAIR_0','REPAIR_1'):
        engine=Engine(endpoint,journal)
        try:loader_check(engine)
        finally:engine.close()
    census=read(ROOT/'CPU_CENSUS.json')
    train_seconds=sum(c['training_step_seconds']/16*768*2 for c in canaries)
    inference=0.
    for endpoint in ('QA','PROGRAM','MELLO_CURRENT','GMELLO_LOCAL'):
        m=dev_measure[endpoint];mult=2 if endpoint in ('QA','PROGRAM') else 1
        inference+=m['seconds']/m['counts']['questions']*census['unique_neural_inputs_with_controls_upper_bound']*mult
    score_calls=[r['result'] for r in journal.index['calls'].values() if r['kind']=='score']
    scoring=sum(r['amortized_seconds'] for r in score_calls)/len(score_calls)*census['candidate_calls_with_controls_upper_bound']*2
    science_forecast=(train_seconds+inference+scoring+600)*1.25
    remaining_science=dt.datetime.fromisoformat(CONTRACT['clock']['science_cutoff_utc'].replace('Z','+00:00')).timestamp()-time.time()
    remaining_wall=dt.datetime.fromisoformat(CONTRACT['clock']['hard_cutoff_utc'].replace('Z','+00:00')).timestamp()-time.time()
    allocation=read(ROOT/'RENTAL.json');elapsed=time.time()-allocation['allocated_epoch']
    remaining_gpu=(7-allocation.get('prior_billed_gpu_hours',0))*3600-elapsed
    projected_cost=allocation.get('prior_cloud_usd_upper_bound',0)+(elapsed+science_forecast+2700)/3600*allocation['all_in_hourly_upper_bound']
    passed=science_forecast<=remaining_science and science_forecast+2700<=min(remaining_wall,remaining_gpu) and projected_cost<=40
    report={'status':'PASS' if passed else 'FORECAST_DOES_NOT_FIT','canaries':[{k:v for k,v in c.items() if k!='history'} for c in canaries],
        'development_measurements':dev_measure,'training_seconds':train_seconds,'inference_seconds':inference,'scoring_seconds':scoring,
        'science_forecast_with_25pct':science_forecast,'closeout_seconds':2700,'remaining_science_seconds':remaining_science,
        'remaining_wall_seconds':remaining_wall,'remaining_gpu_seconds':remaining_gpu,'projected_cloud_usd_upper_bound':projected_cost,
        'protected_outcomes_observed':False,'cohort_seed_component_trimming':False}
    publish(ROOT/'GPU_PREFLIGHT.json',report);journal.export();print(json.dumps(report),flush=True)
    assert passed,'COMPLETE_WORK_FORECAST_FAILED_NO_TRIMMING'

def controls(engine,journal,cases,retriever):
    selection=read(ROOT/'CONTROL_SELECTION.json');byid={c['case_id']:c for c in cases}
    # All post questions in all six conditions for each original fixed repeat case.
    for cid in selection['fixed_uncached_case_ids']:
        key=f'{engine.endpoint}/repeat/{cid}'
        if key in journal.index['controls']:continue
        case=byid[cid];jobs=[{'case_id':cid,'history':h,'phase':'post','query_id':q['query_id'],'arm':engine.endpoint} for h in ['H0','H1','S0','S1','C','C+'] for q in case['queries']]
        rows,timing=evaluate_case(engine,journal,case,jobs,retriever,nonce=key)
        equal=[]
        for row in rows:
            main=journal.index['logical']['/'.join(str(row[k]) for k in ['case_id','history','phase','query_id','arm'])]
            equal.append(row['outcome']==main['outcome'] and row.get('program')==main.get('program'))
        journal.add('controls',{'control_id':key,'kind':'fixed_uncached_case','rows':rows,'all_equal':all(equal),'timing':timing},'control_id')
        journal.status('UNCACHED_CASE',endpoint=engine.endpoint,case_id=cid)
    for cid,qid in selection['uncached_pairs']:
        key=f'{engine.endpoint}/snapshot/{cid}/{qid}'
        if key in journal.index['controls']:continue
        jobs=[{'case_id':cid,'history':h,'phase':phase,'query_id':qid,'arm':engine.endpoint,'slot':slot} for slot,h,phase in [('current_before','H0','post'),('original','H0','pre'),('current_after','H0','post'),('equivalent_current','S0','post')]]
        rows,timing=evaluate_case(engine,journal,byid[cid],jobs,retriever,nonce=key)
        journal.add('controls',{'control_id':key,'kind':'snapshot_return','rows':rows,'timing':timing,
            'state_return_equal':rows[0]['state_sha256']==rows[2]['state_sha256']==rows[3]['state_sha256'],
            'outcome_return_equal':rows[0]['outcome']==rows[2]['outcome']==rows[3]['outcome'],
            'program_return_equal':rows[0].get('program')==rows[2].get('program')==rows[3].get('program')},'control_id')
        journal.status('SNAPSHOT_RETURN',endpoint=engine.endpoint,case_id=cid)
    journal.export()

def run():
    for name,h in read(ROOT/'CODE_MANIFEST.json').items():assert sha_file(ROOT/name)==h,('FROZEN_CODE_CHANGED',name)
    assert read(ROOT/'GPU_PREFLIGHT.json')['status']=='PASS';journal=Journal();retriever=Retriever()
    if not (ROOT/'SCIENTIFIC_FREEZE.json').exists():
        for family in ('QA','PROGRAM'):
            for seed in (0,1):
                fit=f'{family}_{seed}';completed=ROOT/f'training_records/{fit}/COMPLETE.json'
                if not completed.exists():train(family,seed)
                engine=Engine(fit,journal)
                try:
                    result=development(engine,journal,retriever)
                    assert result['counts'].get('valid_outputs',0)>0,('TRAINED_BASELINE_NONFUNCTIONAL',fit,result['counts'])
                finally:engine.close()
                journal.export()
        engine=Engine('BASE',journal)
        try:
            for endpoint in ('MELLO_CURRENT','GMELLO_LOCAL'):
                engine.endpoint=endpoint;development(engine,journal,retriever)
        finally:engine.close()
        freeze={'protocol_sha256':read(ROOT/'CPU_READINESS.json')['protocol_sha256'],'config':CFG,'source_capsule':read(ROOT/'RUN_CAPSULE.json'),
            'code_manifest':read(ROOT/'CODE_MANIFEST.json'),'data':{n:sha_file(ROOT/n) for n in ['RUNTIME_CASES.json','SCHEDULE.json','CONTROL_SELECTION.json','training_join.jsonl.gz','sample_orders.json','development.json','demonstrations.json','mello_demonstrations.txt']},
            'adapters':{fit:sha_file(ROOT/f'adapters/{fit}/adapter_model.safetensors') for fit in ['QA_0','QA_1','PROGRAM_0','PROGRAM_1']},
            'created_epoch':time.time(),'protected_inference_before_freeze':False,'no_test_driven_revisions':True}
        freeze['freeze_sha256']=digest(freeze);publish(ROOT/'SCIENTIFIC_FREEZE.json',freeze);journal.export()
    frozen=read(ROOT/'SCIENTIFIC_FREEZE.json')
    assert frozen['code_manifest']==read(ROOT/'CODE_MANIFEST.json') and frozen['config']==CFG
    for n,h in frozen['data'].items():assert sha_file(ROOT/n)==h
    for fit,h in frozen['adapters'].items():assert sha_file(ROOT/f'adapters/{fit}/adapter_model.safetensors')==h
    cases=read(ROOT/'RUNTIME_CASES.json');schedule=read(ROOT/'SCHEDULE.json');timed=set(read(ROOT/'CONTROL_SELECTION.json')['timed_subjects'])
    for endpoint in ENDPOINTS:
        engine=Engine(endpoint,journal)
        try:
            if endpoint.startswith('REPAIR_'):loader_check(engine)
            for case in cases:
                check_time();jobs=[r for r in schedule if r['case_id']==case['case_id'] and r['arm']==endpoint]
                todo=[r for r in jobs if '/'.join(str(r[k]) for k in ['case_id','history','phase','query_id','arm']) not in journal.index['logical']]
                if not todo:continue
                rows,timing=evaluate_case(engine,journal,case,todo,retriever)
                for row in rows:
                    row['logical_id']='/'.join(str(row[k]) for k in ['case_id','history','phase','query_id','arm'])
                    row['recorded_epoch']=time.time();row['freeze_sha256']=frozen['freeze_sha256'];journal.add('logical',row,'logical_id')
                if case['case_id'] in timed:journal.add('timings',{'timing_id':endpoint+'/'+str(case['case_id'])+'/'+str(len(journal.index['timings'])),'endpoint':endpoint,'case_id':case['case_id'],'completed_before_resume':len(jobs)-len(todo),**timing},'timing_id')
                journal.status('PROTECTED_EVALUATION',endpoint=endpoint,case_id=case['case_id'])
                if len(journal.index['logical'])%1000<len(rows):journal.export()
            controls(engine,journal,cases,retriever)
        finally:engine.close()
    assert len(journal.index['logical'])==55536 and len(journal.index['controls'])==8*(12+16)
    publish(ROOT/'COMPLETE.json',{'technical_status':'COMPLETE','logical_reads':55536,'controls':len(journal.index['controls']),'science_results':'UNASSESSED_EVALUATOR_NOT_ON_WORKER','freeze_sha256':frozen['freeze_sha256'],'epoch':time.time()})
    journal.export();journal.status('COMPLETE')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','run'],required=True);args=ap.parse_args()
    preflight() if args.stage=='preflight' else run()
