"""CPU replay of actual 32B outputs. Never invents outputs for missing requests."""
from pathlib import Path
import argparse,collections,csv,json,math,sys
from prepare_data import ROOT,LEGACY,HISTORIES,ALL,load,gzrows,dump,write_rows,canonical
from execution import execute
from codec import decode,norm
from ranker_contract import choose,key as pkey
from metrics import paired_interval

def responses(path,required):
    with Path(path).open(encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    by={r['request_id']:r for r in rows}
    if len(rows)!=len(by) or set(by)!=set(required):raise ValueError('INCOMPLETE_OR_DUPLICATE_MODEL_OUTPUT')
    return by

def evaluate(base,seed0,seed1,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bank=gzrows(ROOT/'prepared/REQUESTS.jsonl.gz');scores={r['request_id'] for r in bank if r['kind']=='score'}
    bp=responses(base,{r['request_id'] for r in bank})
    sp=[responses(p,scores) for p in (seed0,seed1)]
    logical=gzrows(ROOT/'prepared/LOGICAL_ROWS.jsonl.gz')
    cases={r['case_id']:r for r in load(LEGACY/'prepared/RUNTIME_CASES.json')}
    ev={(r['case_id'],r['query_id']):r for r in load(ROOT/'prepared/EVALUATOR_ONLY.json')}
    allrows=[]
    for r in logical:
        c=cases[r['case_id']];e=ev[r['case_id'],r['query_id']]
        gold=e['post'] if r['phase']=='post' else e['pre'][r['history']]
        for arm in ('CURRENT_TEXT','CURRENT_JSON','STATE_BASE','STATE_0','STATE_1'):
            d={k:r[k] for k in ('case_id','history','phase','query_id','logical_id')};d['arm']=arm;d['branch']=e['branch']
            selected=None
            if arm.startswith('STATE'):
                records=bp if arm=='STATE_BASE' else sp[int(arm[-1])]
                margins=[records[i]['margin'] for i in r['score_request_ids']]
                for i,m in zip(r['score_request_ids'],margins):
                    if not math.isfinite(m) or abs(m-(records[i]['yes_logit']-records[i]['no_logit']))>1e-6:raise ValueError('MARGIN_MISMATCH')
                selected=choose(r['candidates'],margins)
                result=execute(c,r['history'],r['phase'],selected)
                d.update(entity=result['entity'],status=result['status'],selected=selected,margins=margins,
                    faithful=selected is not None and selected['start_id']==gold['start'] and selected['path']==gold['path'])
                if selected is not None:
                    # Independent dictionary walk rather than trusting endpoint in the option.
                    h=c['histories'][r['history']];state={(s,p):o for s,p,o in h['initial']}
                    if r['phase']=='post':state.update({(s,p):o for s,p,o in h['edits']})
                    x=selected['start_id']
                    for rel in selected['path']:x=state[x,rel]
                    if x!=d['entity']:raise ValueError('EXECUTION_DISAGREEMENT')
                if selected is None:
                    best=max(range(len(margins)),key=lambda i:(margins[i],-i)) if margins else None
                    top=r['candidates'][best] if best is not None else None
                    d['correct_top_refusal']=top is not None and top['start_id']==gold['start'] and top['path']==gold['path']
            else:
                raw=bp[r[arm]];parsed=decode(raw['text'],{norm(v):k for k,v in c['labels'].items()},set(c['labels']),
                    truncated=raw['truncated'],invalid_control_tokens=raw['invalid_control_tokens'])
                d.update(entity=parsed['entity'],status=parsed['status'],faithful=None,raw_text=raw['text'],truncated=raw['truncated'],invalid_control_tokens=raw['invalid_control_tokens'])
            d['correct']=d['entity']==gold['entity'];allrows.append(d)
    for r in gzrows(ROOT/'prepared/NATIVE32_COMMON_CODEC.jsonl.gz'):
        allrows.append(dict(r,arm='NATIVE32_SAVED',branch=ev[r['case_id'],r['query_id']]['branch'],faithful=None))
    arms=['NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON','STATE_BASE','STATE_0','STATE_1']
    idx={(r['arm'],r['case_id'],r['history'],r['phase'],r['query_id']):r for r in allrows}
    summaries={};complete_vectors={};robust={};fidelity={};invariance={};harms=[];known=[]
    for arm in arms:
        complete=[];faithcomplete=[];ok={};fok={};viol=[]
        for cid,c in cases.items():
            cq=[];cf=[]
            for q in c['queries']:
                qid=q['query_id'];rr=[idx[arm,cid,h,'post',qid] for h in HISTORIES]
                good=all(x['correct'] for x in rr);faith=all(x.get('faithful') is True and x['correct'] for x in rr) if arm.startswith('STATE') else None
                ok[cid,qid]=good;fok[cid,qid]=faith;cq.append(good);cf.append(faith)
                # All failures count, including refusal and unanimous wrong output.
                six=[idx[arm,cid,h,'post',qid] for h in ALL]
                signatures={canonical([x['status'],x['entity'],x.get('selected',{}).get('start_id') if x.get('selected') else None,x.get('selected',{}).get('path') if x.get('selected') else None]) for x in six}
                if len(signatures)>1:viol.append({'case_id':cid,'query_id':qid})
            complete.append(all(cq));faithcomplete.append(all(x is True for x in cf) if arm.startswith('STATE') else None)
        robust[arm]=ok;fidelity[arm]=fok;complete_vectors[arm]=[int(x) for x in complete];invariance[arm]=viol
        rolecounts={role:sum(g for k,g in ok.items() if ev[k]['branch']==role) for role in ('J','F','Fdecoy','K')}
        post=[r for r in allrows if r['arm']==arm and r['phase']=='post' and r['history']=='H0']
        before=[r for r in allrows if r['arm']==arm and r['phase']=='pre']
        summaries[arm]={'complete_subjects':sum(complete),'complete_faithful_subjects':sum(x is True for x in faithcomplete) if arm.startswith('STATE') else None,
            'robust_questions':sum(ok.values()),'robust_by_family':rolecounts,'post_refusals_H0':sum(r['status']=='REFUSAL' for r in post),
            'post_wrong_returns_H0':sum(not r['correct'] and r['status']=='VALID' for r in post),
            'post_wrong_program_right_answer_H0':sum(r['correct'] and r['faithful'] is False for r in post),
            'pre_correct':sum(r['correct'] for r in before) if before else None,'pre_total':len(before) if before else None,
            'current_output_differences_from_logical_replay':len(viol),'uncached_validation':'NOT_READ_BY_THIS_EVALUATOR; required separate real-control audit'}
        if arm.startswith('STATE'):
            summaries[arm]['sampled_capability_target_met']=(sum(x is True for x in faithcomplete)>=122 and rolecounts['J']>=381 and sum(rolecounts[k] for k in ('F','Fdecoy','K'))==517)
    for arm in ('STATE_0','STATE_1'):
        for comp in ('NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON','STATE_BASE'):
            for key,old in robust[comp].items():
                new=robust[arm][key]
                harms.append(dict(candidate=arm,baseline=comp,case_id=key[0],query_id=key[1],baseline_correct=old,candidate_correct=new,
                    fixed=not old and new,harmed=old and not new))
        for w in load(ROOT/'prepared/KNOWN_32B_WITNESSES.json'):
            cid,qid=w['case_id'],w['query_id'];ctrl=[q['query_id'] for q in cases[cid]['queries'] if not q['query_id'].startswith('J')]
            target_ok=robust[arm][cid,qid] and fidelity[arm][cid,qid]
            control_ok=all(idx[arm,cid,h,'post',q]['correct'] for h in HISTORIES for q in ctrl)
            refs_ok=all(idx[arm,cid,h,'post',qid]['correct'] for h in ('C','C+'))
            known.append(dict(arm=arm,**w,correct_across_histories=bool(target_ok),controls_preserved=control_ok,
                references_preserved=refs_ok,repaired=bool(target_ok and control_ok and refs_ok)))
    mean=[(a+b)/2 for a,b in zip(complete_vectors['STATE_0'],complete_vectors['STATE_1'])]
    contrasts={b:paired_interval(mean,complete_vectors[b],seed=2026100708,draws=10000,level=.975) for b in ('NATIVE32_SAVED','CURRENT_TEXT')}
    result={'endpoints':summaries,'primary_paired_contrasts':contrasts,'known_9_repairs':{a:sum(r['repaired'] for r in known if r['arm']==a) for a in ('STATE_0','STATE_1')},
        'denominators':{'subjects':128,'targets':384,'controls':517,'all_questions':901,'known_witnesses':9},
        'isolation_claim':'Requires source/input checks PLUS uncached control audit; zero replay disagreement alone is not that audit',
        'comparative_gain':all(r['difference']>=.05 and r['interval'][0]>0 for r in contrasts.values()),
        'native_comparator':'Saved SCALE32 outputs recoded once with common original PPR whole-output codec; frozen SCALE counts unchanged'}
    write_rows(out/'ALL_OUTCOMES.jsonl.gz',allrows);write_rows(out/'FIXES_HARMS.jsonl.gz',harms);dump(out/'KNOWN_9_REPAIR_LEDGER.json',known)
    dump(out/'SUMMARY.json',result);dump(out/'INVARIANCE_REPLAY.json',invariance)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','seed0','seed1','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.base,a.seed0,a.seed1,a.out),indent=2))
