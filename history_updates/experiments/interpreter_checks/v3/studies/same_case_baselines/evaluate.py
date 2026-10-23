"""CPU-only evaluator and replay, strictly separated from model runtime."""
from pathlib import Path
import argparse
import ast
import collections
import gzip
import hashlib
from itertools import combinations
import json
import sys
import numpy as np
from baseline_core import ENDPOINTS,digest,parse_program,sql_chain

ROOT=Path(__file__).resolve().parent
H=['H0','H1','S0','S1'];ALL=H+['C','C+']
def read(p):return json.loads(Path(p).read_bytes())
def load_lines(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(l) for l in f]
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def lines(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def bootstrap(complete,ids):
    vectors={a:np.asarray([complete[a][cid] for cid in ids],dtype=float) for a in ENDPOINTS}
    means={family:(vectors[family+'_0']+vectors[family+'_1'])/2 for family in ['REPAIR','QA','PROGRAM']}
    means.update({a:vectors[a] for a in ['MELLO_CURRENT','GMELLO_LOCAL']})
    ix=np.random.default_rng(2026101101).integers(0,len(ids),size=(10000,len(ids)))
    result={}
    for baseline in ['QA','PROGRAM','MELLO_CURRENT','GMELLO_LOCAL']:
        d=means['REPAIR']-means[baseline];lo,hi=np.quantile(d[ix].mean(axis=1),[.00625,.99375])
        seed_rows={}
        for seed in (0,1):
            right=vectors[f'{baseline}_{seed}'] if baseline in ['QA','PROGRAM'] else means[baseline]
            sd=vectors[f'REPAIR_{seed}']-right
            seed_rows[str(seed)]={'difference':float(sd.mean()),'descriptive_interval':np.quantile(sd[ix].mean(axis=1),[.00625,.99375]).tolist()}
        result[baseline]={'difference':float(d.mean()),'interval':[float(lo),float(hi)],'confidence':.9875,'draws':10000,'seed':2026101101,
            'unit':'subject; all questions, histories and seeds linked','substantively_positive':bool(d.mean()>=.05 and lo>0),
            'equivalence_claim':False,'separate_seeds':seed_rows,'subject_differences':d.tolist()}
    return result,{k:v.tolist() for k,v in means.items()}

def source_endpoints(path):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='endpoints')
    namespace={'ARMS':tuple(H),'ALL_ARMS':tuple(ALL),'combinations':combinations}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['endpoints']

def evaluate(raw,data,out):
    raw=Path(raw);data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gold_rows=read(data/'EVALUATOR_ONLY.json');gold={(r['case_id'],r['query_id']):r for r in gold_rows}
    cases=read(data/'RUNTIME_CASES.json');ids=[c['case_id'] for c in cases];case_by={c['case_id']:c for c in cases}
    source=json.loads(gzip.decompress((data/'cohort.json.gz').read_bytes()));source_by={c['case_id']:c for c in source['cases']}
    sys.path[:0]=[str(data/'ppr/implementation'),str(data/'ppr/vendor')]
    from ppr_requests import score_requests,record_view
    from execution import execute
    from ranker_contract import choose
    from codec import decode,norm
    from pst1_core import Store
    endpoint_fn=source_endpoints(data/'mqmh.py')
    freeze=read(raw/'SCIENTIFIC_FREEZE.json');logical=load_lines(raw/'results/logical.jsonl')
    schedule=read(data/'SCHEDULE.json');key=lambda r:tuple(r[k] for k in ['case_id','history','phase','query_id','arm'])
    lookup={key(r):r for r in logical}
    assert len(lookup)==len(logical)==55536 and set(lookup)=={key(r) for r in schedule},'INCOMPLETE_OR_DUPLICATE_SCHEDULE'
    calls={r['call_id']:r for r in load_lines(raw/'results/calls.jsonl')}
    methods={r['method_id']:r for r in load_lines(raw/'results/method.jsonl')}
    scored=[];statuses=collections.defaultdict(collections.Counter)
    for r in logical:
        assert r['freeze_sha256']==freeze['freeze_sha256'] and r['recorded_after_scientific_freeze'] is True
        g=gold[r['case_id'],r['query_id']];expected=g['post'] if r['phase']=='post' else g['pre'][r['history']]
        case=case_by[r['case_id']];question=next(q['question'] for q in case['queries'] if q['query_id']==r['query_id'])
        assert r['question']==question
        edges,payload=record_view(case,r['history'],r['phase']);payload={'question':question,'relation_registry':payload['relation_registry'],'current_records':payload['current_records']}
        assert r['state_sha256']==digest(edges) and r['request_payload_sha256']==digest(payload)
        multi=collections.defaultdict(set)
        for entity,label in case['labels'].items():multi[norm(label)].add(entity)
        aliases={name:next(iter(v)) for name,v in multi.items() if len(v)==1}
        if r['arm'].startswith('REPAIR_'):
            candidates,requests=score_requests(case,r['history'],r['phase'],question)
            assert candidates==r['candidates'] and len(requests)==len(r['call_ids'])
            actual=[]
            for req,call_id in zip(requests,r['call_ids']):
                call=calls[call_id];assert call['request']==req and call['kind']=='score'
                actual.append(call['result']['margin'])
            selected=choose(candidates,actual);assert selected==r['selected']
            assert execute(case,r['history'],r['phase'],selected)==r['outcome']
        else:
            method=methods[r['method_id']];assert method['payload']==payload
            result=method['result']
            for field in ['answer_text','status','trace']:assert result.get(field)==r.get(field)
            if r['arm'].startswith('PROGRAM_'):
                parsed,status=parse_program(result['trace']['text'],payload)
                assert parsed==result['program']
                if r['outcome']['status']=='VALID':
                    replay=Store(edges).query(parsed['start_id'],parsed['path']);assert replay['endpoint']==r['outcome']['entity']
            elif result.get('route')=='graph':
                replay=sql_chain(payload,result['program']);assert replay['entity']==r['outcome']['entity']
            elif r['outcome']['status']=='VALID':
                assert decode(result['answer_text'],aliases,set(case['labels']))==r['outcome']
        correct=r['outcome']['status']=='VALID' and r['outcome']['entity']==expected['entity']
        program=r.get('program');faithful=None if program is None else program['start_id']==expected['start'] and program['path']==expected['path']
        answer_text=r.get('answer_text','');first_line=answer_text.strip().splitlines()
        secondary=decode(first_line[0] if first_line else '',aliases,set(case['labels']))
        executed_program=r['arm'].startswith(('REPAIR_','PROGRAM_')) or r.get('route')=='graph'
        row=dict(r,correct=correct,expected_entity=expected['entity'],program_faithful=faithful,
                 wrong_program_right_endpoint=bool(correct and faithful is False and executed_program),branch=g['branch'],
                 first_line_secondary_correct=None if executed_program else secondary['status']=='VALID' and secondary['entity']==expected['entity'])
        scored.append(row);lookup[key(r)]=row;statuses[r['arm']][r['outcome']['status']]+=1
    def get(cid,h,q,a,phase='post'):return lookup[cid,h,phase,q,a]
    complete={};summary={};guards=[];robust={}
    for a in ENDPOINTS:
        complete[a]={};robust[a]={};post=[r for r in scored if r['arm']==a and r['phase']=='post'];pre=[r for r in scored if r['arm']==a and r['phase']=='pre']
        for c in cases:
            cid=c['case_id'];complete[a][cid]=all(get(cid,h,q['query_id'],a)['correct'] for h in H for q in c['queries'])
            for q in c['queries']:robust[a][cid,q['query_id']]=all(get(cid,h,q['query_id'],a)['correct'] for h in H)
            values={(cid,h,q['query_id']):{'correct':get(cid,h,q['query_id'],a)['correct'],'entity_id':get(cid,h,q['query_id'],a)['outcome']['entity']} for h in ALL for q in c['queries']}
            guards.extend(dict(x,endpoint=a) for x in endpoint_fn(source_by[cid],values)['targets'])
        bybranch={b:{'questions':sum(g['branch']==b for g in gold_rows),'robust_correct':sum(robust[a][g['case_id'],g['query_id']] for g in gold_rows if g['branch']==b)} for b in ['J','F','Fdecoy','K']}
        ag=[g for g in guards if g['endpoint']==a]
        agreement=[]
        for g in gold_rows:
            rr=[get(g['case_id'],h,g['query_id'],a) for h in H]
            agreement.append({'same_entity':len({r['outcome']['entity'] for r in rr})==1,'all_wrong':all(not r['correct'] for r in rr),'all_refusal':all(r['outcome']['status']=='REFUSAL' for r in rr)})
        summary[a]={'subjects':128,'complete_subjects':sum(complete[a].values()),'complete_rate':sum(complete[a].values())/128,
            'targets':384,'robust_targets':bybranch['J']['robust_correct'],'controls':517,'robust_controls':sum(bybranch[b]['robust_correct'] for b in ['F','Fdecoy','K']),
            'by_branch':bybranch,'post_correct':sum(r['correct'] for r in post),'post_reads':len(post),'pre_correct':sum(r['correct'] for r in pre),'pre_reads':len(pre),
            'status_counts':dict(statuses[a]),'program_outputs':sum(r.get('program') is not None for r in post),
            'first_line_secondary_post_correct':sum(r['first_line_secondary_correct'] is True for r in post),
            'first_line_secondary_applicable_outputs':sum(r['first_line_secondary_correct'] is not None for r in post),
            'exact_program_faithful_post':sum(r['program_faithful'] is True for r in post),'wrong_program_right_endpoint':sum(r['wrong_program_right_endpoint'] for r in post),
            'guard_coverage':sum(g['common_guard'] for g in ag),'fact_coverage':sum(g['current_facts'] for g in ag),'reference_coverage':sum(g['references'] for g in ag),
            'guarded_witnesses':sum(g['qualified'] for g in ag),'guard_denominator':384,
            'same_entity_four_histories':sum(x['same_entity'] for x in agreement),'uniformly_wrong':sum(x['same_entity'] and x['all_wrong'] for x in agreement),
            'uniform_refusal':sum(x['all_refusal'] for x in agreement),'accuracy_gap_is_not_history_leakage':True}
    contrasts,means=bootstrap(complete,ids)
    old=read(data/'HISTORICAL_CROSSWALK.json');old94={(r['case_id'],r['query_id']) for r in old if r['historical_qwen_flag']};new32={(r['case_id'],r['query_id']) for r in old if r['new_HISTORY']['qualified']}
    assert (len(old94),len(new32),len(old94&new32))==(94,32,15)
    original=json.loads(gzip.decompress((data/'qwen_responses.json.gz').read_bytes()))['primary']
    old_lookup={(r['case_id'],r['arm'],r['query_id']):r for r in original}
    native=[r for r in load_lines(data/'HISTORICAL_SCORED.jsonl') if r['arm']=='HISTORY' and r['phase']=='post']
    native_lookup={(r['case_id'],r['history'],r['query_id']):r for r in native}
    cohorts={};witness_rows=[]
    for title,members in [('original_94',old94),('later_32',new32),('overlap_15',old94&new32),('later_not_original_17',new32-old94)]:
        cohorts[title]={'denominator':len(members),'endpoints':{}}
        for a in ENDPOINTS:
            repaired=0
            for cid,qid in sorted(members):
                control_keys=[(h,q['query_id']) for h in ALL for q in case_by[cid]['queries'] if not q['query_id'].startswith('J')]+[('C',qid),('C+',qid)]
                losses=[]
                for h,q in control_keys:
                    before=(old_lookup[cid,h,q]['valid'] and old_lookup[cid,h,q]['entity_id']==gold[cid,q]['post']['entity']) if title=='original_94' else native_lookup[cid,h,q]['correct']
                    if before and not get(cid,h,q,a)['correct']:losses.append([h,q])
                good=robust[a][cid,qid] and not losses;repaired+=good
                witness_rows.append({'cohort':title,'endpoint':a,'case_id':cid,'query_id':qid,'target_correct_all_histories':robust[a][cid,qid],'lost_guards_or_controls':losses,'preserved_repair':good})
            cohorts[title]['endpoints'][a]={'repaired':repaired,'rate':repaired/len(members)}
    paired=[]
    for baseline in ['QA','PROGRAM','MELLO_CURRENT','GMELLO_LOCAL']:
        for seed in (0,1):
            left=f'REPAIR_{seed}';right=f'{baseline}_{seed}' if baseline in ['QA','PROGRAM'] else baseline
            for g in gold_rows:
                cid,qid=g['case_id'],g['query_id'];l=robust[left][cid,qid];r=robust[right][cid,qid]
                paired.append({'repair':left,'baseline':right,'case_id':cid,'query_id':qid,'branch':g['branch'],'repair_robust_correct':l,'baseline_robust_correct':r,'fixed':l and not r,'harmed':r and not l})
    controls=load_lines(raw/'results/controls.jsonl');assert len(controls)==224
    control_failures=[r['control_id'] for r in controls if not (r['all_equal'] if r['kind']=='fixed_uncached_case' else r['state_return_equal'] and r['outcome_return_equal'] and r['program_return_equal'])]
    validity={'status':'PASS' if not control_failures else 'FAILED_UNCACHED_OR_SNAPSHOT_GATE','schedule_complete':True,'logical_reads':55536,'full_subjects':128,'all_eight_endpoints':True,'four_primary_fits':all(read(raw/f'training_records/{a}/COMPLETE.json')['steps']==768 for a in ['QA_0','QA_1','PROGRAM_0','PROGRAM_1']),
        'freeze_before_recorded_protected_reads':True,'control_failures':control_failures,'no_technical_failure_scored_as_poor_baseline':True}
    results={'status':'MEASURED' if validity['status']=='PASS' else 'MEASURED_WITH_FAILED_VALIDITY_GATE','endpoints':summary,'primary_contrasts':contrasts,'family_subject_means':means,'historical_witness_cohorts':cohorts,
             'comparison_validity':validity['status'],'legacy_gates_unchanged':True,'nonsignificance_is_not_equivalence':True}
    write(out/'COMPARISON_VALIDITY.json',validity);write(out/'SCIENTIFIC_RESULTS.json',results)
    write(out/'SUBJECT_VECTORS.json',{'case_ids':ids,'complete':complete,'family_means':means})
    lines(out/'SCORED_LOGICAL.jsonl',scored);lines(out/'GUARD_LEDGER.jsonl',guards);lines(out/'WITNESS_RECOVERY.jsonl',witness_rows);lines(out/'PAIRED_FIXES_HARMS.jsonl',paired)
    lines(out/'ERROR_LEDGER.jsonl',[r for r in scored if not r['correct'] or r['program_faithful'] is False])
    table=['| Endpoint | Fully correct subjects /128 | Robust targets /384 | Robust controls /517 |','|---|---:|---:|---:|']
    table += [f'| {a} | {summary[a]["complete_subjects"]} | {summary[a]["robust_targets"]} | {summary[a]["robust_controls"]} |' for a in ENDPOINTS]
    (out/'comparison_table.md').write_text('\n'.join(table)+'\n',encoding='utf8')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=evaluate(args.raw,args.data,args.out);print(json.dumps({'status':result['status'],'comparison_validity':result['comparison_validity']}))
