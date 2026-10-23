"""Whole-workload native witnesses, first-line sensitivity, harms, and error traces.
These classifications are not available to the GPU worker.
"""
import itertools
from gemma_prepare import ALL,HISTORIES,dump,write_rows,load,ROOT
from codec import decode,norm

def witness_rule(answers,fact_ok,reference_ok):
    """Answers are VALID source entities, not raw strings or two wrong labels."""
    return bool(fact_ok and reference_ok and any(
        a['status']=='VALID' and b['status']=='VALID' and a['entity']!=b['entity'] and bool(a['correct'])!=bool(b['correct'])
        for a,b in itertools.combinations(answers,2)))

def additional(rows,cases,ev,out):
    idx={(r['arm'],r['case_id'],r['history'],r['phase'],r['query_id']):r for r in rows}
    native=[];repairs=[];q9={(w['case_id'],w['query_id']) for w in load(ROOT/'prepared/QWEN9_CHALLENGE.json')}
    for cid,c in cases.items():
        fs=[q['query_id'] for q in c['queries'] if ev[cid,q['query_id']]['branch']=='F']
        controls=[q['query_id'] for q in c['queries'] if ev[cid,q['query_id']]['branch']!='J']
        ks=[q['query_id'] for q in c['queries'] if ev[cid,q['query_id']]['branch']=='K']
        for q in c['queries']:
            qid=q['query_id']
            if ev[cid,qid]['branch']!='J':continue
            rr=[idx['NATIVE_GEMMA',cid,h,'post',qid] for h in HISTORIES]
            facts=all(idx['NATIVE_GEMMA',cid,h,'post',f]['correct'] for h in HISTORIES for f in fs)
            refs=all(idx['NATIVE_GEMMA',cid,h,'post',qid]['correct'] for h in ('C','C+'))
            qualified=witness_rule(rr,facts,refs)
            decoys=[x for x in controls if ev[cid,x]['branch']=='Fdecoy']
            decoypass=all(idx['NATIVE_GEMMA',cid,h,'post',d]['correct'] for h in HISTORIES for d in decoys)
            kpass=bool(ks) and all(idx['NATIVE_GEMMA',cid,h,'post',k]['correct'] for h in HISTORIES for k in ks)
            excess=((1-rr[0]['correct'])+(1-rr[1]['correct'])-(1-rr[2]['correct'])-(1-rr[3]['correct']))/2 if facts and refs else 0
            native.append({'case_id':cid,'query_id':qid,'fact_guard':facts,'reference_guard':refs,'eligible':facts and refs,'witness':qualified,'K_controlled_witness':qualified and kpass,'Fdecoy_controlled_witness':qualified and decoypass,'relevance_difference':excess,'is_old_Qwen9':(cid,qid) in q9})
            if qualified:
                for arm in ('STATE_BASE','STATE_0','STATE_1','CURRENT_TEXT','CURRENT_JSON'):
                    target=all(idx[arm,cid,h,'post',qid]['correct'] for h in HISTORIES)
                    fidelity=all(idx[arm,cid,h,'post',qid]['faithful'] is True for h in HISTORIES) if arm.startswith('STATE') else None
                    # Require every retained control and both references. In-scope abstention is failure.
                    guard=all(idx[arm,cid,h,'post',x]['correct'] for h in HISTORIES for x in controls)
                    rpass=all(idx[arm,cid,h,'post',qid]['correct'] for h in ('C','C+'))
                    repairs.append({'arm':arm,'case_id':cid,'query_id':qid,'target_correct':target,'faithful':fidelity,'all_controls':guard,'references':rpass,'repaired':bool(target and guard and rpass and (fidelity is not False))})
    errors=[];paired=[];sensitivity=[]
    for r in rows:
        if not r['correct'] or r.get('faithful') is False:
            errors.append(dict(r,question=ev[r['case_id'],r['query_id']]['question'],reference=ev[r['case_id'],r['query_id']]['post'] if r['phase']=='post' else ev[r['case_id'],r['query_id']]['pre'][r['history']]))
        if 'raw_text' in r:
            case=cases[r['case_id']];text=r['raw_text'].strip().splitlines();first=text[0] if text else ''
            parsed=decode(first,{norm(v):k for k,v in case['labels'].items()},set(case['labels']),truncated=r['truncated'],invalid_control_tokens=r['invalid_control_tokens'])
            e=ev[r['case_id'],r['query_id']];g=e['post'] if r['phase']=='post' else e['pre'][r['history']]
            sensitivity.append({k:r[k] for k in ('arm','case_id','query_id','history','phase')}|{'whole_correct':r['correct'],'first_line_correct':parsed['entity']==g['entity'],'raw_text':r['raw_text']})
        if r['arm'] in ('STATE_0','STATE_1'):
            for b in ('NATIVE_GEMMA','CURRENT_TEXT','CURRENT_JSON','STATE_BASE'):
                k=(b,r['case_id'],r['history'],r['phase'],r['query_id'])
                if k not in idx:continue
                old=idx[k];paired.append({'candidate':r['arm'],'baseline':b,'case_id':r['case_id'],'query_id':r['query_id'],'history':r['history'],'phase':r['phase'],'fixed':not old['correct'] and r['correct'],'harmed':old['correct'] and not r['correct'],'candidate_status':r['status'],'baseline_status':old['status']})
    write_rows(out/'NATIVE_GEMMA_ALL_384_TARGETS.jsonl.gz',native)
    dump(out/'NATIVE_GEMMA_WITNESS_REPAIRS.json',repairs)
    write_rows(out/'EVERY_ERROR_OR_REFUSAL.jsonl.gz',errors)
    write_rows(out/'EVERY_READ_COMPARISON.jsonl.gz',paired)
    write_rows(out/'FIRST_LINE_SENSITIVITY.jsonl.gz',sensitivity)
    W=sum(r['witness'] for r in native)
    stats={'own_native_witnesses':W,'eligible_targets':sum(r['eligible'] for r in native),'target_denominator':384,'own_native_witness_subjects':len({r['case_id'] for r in native if r['witness']}),'K_witnesses':sum(r['K_controlled_witness'] for r in native),'K_scheduled_denominator':195,'Fdecoy_controlled_witnesses':sum(r['Fdecoy_controlled_witness'] for r in native),'own_witness_repairs':{arm:{'numerator':sum(r['repaired'] for r in repairs if r['arm']==arm),'denominator':W,'status':'MEASURED' if W else 'NO_ELIGIBLE_NATIVE_FAILURES_NOT_A_REPAIR_PASS'} for arm in ('STATE_BASE','STATE_0','STATE_1','CURRENT_TEXT','CURRENT_JSON')},'Qwen9_intersection':sum(r['witness'] and r['is_old_Qwen9'] for r in native),'guarded_relevance_contrast':sum(r['relevance_difference'] for r in native)/384,'read_harms_total':sum(r['harmed'] for r in paired),'harm_unit':'Repeated condition/baseline comparisons, not independent failures','relevance_status':'Descriptive secondary quantity; no new confirmatory relevance claim'}
    dump(out/'DIAGNOSTIC_SUMMARY.json',stats)
    return stats
