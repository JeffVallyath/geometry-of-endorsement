"""Required accounting beyond the headline. Replays outputs, never calls a model."""
from pathlib import Path
import argparse,collections,gzip,json
from prepare_data import ROOT,LEGACY,load,gzrows,dump,write_rows,HISTORIES,ALL
from scale_codec import parse_response

def supplement(analysis):
    analysis=Path(analysis)
    rows=gzrows(analysis/'ALL_OUTCOMES.jsonl.gz')
    cases={r['case_id']:r for r in load(LEGACY/'prepared/RUNTIME_CASES.json')}
    sources={r['case_id']:r for r in json.loads(gzip.decompress((LEGACY/'source/cohort.json.gz').read_bytes()))['cases']}
    ev={(r['case_id'],r['query_id']):r for r in load(ROOT/'prepared/EVALUATOR_ONLY.json')}
    raw={(r['case_id'],r['arm'],r['query_id']):r for r in gzrows(ROOT/'baseline/native32_raw.jsonl.gz')}
    idx={(r['arm'],r['case_id'],r['history'],r['phase'],r['query_id']):r for r in rows}
    arms=['NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON','STATE_BASE','STATE_0','STATE_1']
    transitions=[];first=[];subjects=[];rescues=[]
    for r in rows:
        if r['phase']=='post' and r['history'] in HISTORIES and r['query_id'].startswith('J') and r['arm']!='NATIVE32_SAVED':
            old=idx[r['arm'],r['case_id'],r['history'],'pre',r['query_id']]
            transitions.append({'arm':r['arm'],'case_id':r['case_id'],'history':r['history'],'query_id':r['query_id'],
                'before_correct':old['correct'],'after_correct':r['correct'],'before_entity':old['entity'],'after_entity':r['entity'],
                'before_status':old['status'],'after_status':r['status'],'before_faithful':old['faithful'],'after_faithful':r['faithful'],
                'after_uses_previous_reference_answer':not r['correct'] and r['entity'] is not None and r['entity']==ev[r['case_id'],r['query_id']]['pre'][r['history']]['entity']})
        if r['arm'] in ('NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON'):
            original=raw[r['case_id'],r['history'],r['query_id']] if r['arm']=='NATIVE32_SAVED' else {
                'text':r['raw_text'],'truncated':r['truncated'],'control_tokens_valid':not r['invalid_control_tokens']}
            entity,strict=parse_response(original,sources[r['case_id']])
            e=ev[r['case_id'],r['query_id']];g=e['post'] if r['phase']=='post' else e['pre'][r['history']]
            first.append({'arm':r['arm'],'case_id':r['case_id'],'history':r['history'],'phase':r['phase'],'query_id':r['query_id'],
                'entity':entity,'strict_entity':strict,'correct':entity==g['entity'],'strict_correct':strict==g['entity'],
                'primary_correct':r['correct'],'primary_entity':r['entity'],'raw_text':original['text']})
    fi={(r['arm'],r['case_id'],r['history'],r['phase'],r['query_id']):r for r in first}
    sensitivity={}
    for arm in ('NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON'):
        sensitivity[arm]={}
        for field in ('correct','strict_correct'):
            count=0;robust=0
            for cid,c in cases.items():
                oks=[all(fi[arm,cid,h,'post',q['query_id']][field] for h in HISTORIES) for q in c['queries']]
                count+=all(oks);robust+=sum(oks)
            sensitivity[arm][field]={'complete_subjects':count,'robust_questions':robust}
    for arm in arms:
        for cid,c in cases.items():
            subjects.append({'arm':arm,'case_id':cid,'complete':all(idx[arm,cid,h,'post',q['query_id']]['correct'] for h in HISTORIES for q in c['queries'])})
    sc={(r['arm'],r['case_id']):r['complete'] for r in subjects};subject_harms=[]
    for cand in ('STATE_0','STATE_1'):
        for baseline in ('NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON','STATE_BASE'):
            for cid in cases:
                a,b=sc[baseline,cid],sc[cand,cid]
                subject_harms.append({'candidate':cand,'baseline':baseline,'case_id':cid,'baseline_complete':a,'candidate_complete':b,'fixed':not a and b,'harmed':a and not b})
            for r in rows:
                if r['arm']!=cand or r['history'] not in HISTORIES:continue
                k=(baseline,r['case_id'],r['history'],r['phase'],r['query_id'])
                if k not in idx:continue  # Native has no pre measurements; do not invent any.
                old=idx[k]
                if old['status']=='REFUSAL':
                    rescues.append({'candidate':cand,'baseline':baseline,'case_id':r['case_id'],'history':r['history'],
                        'phase':r['phase'],'query_id':r['query_id'],'candidate_status':r['status'],
                        'rescued_correct':r['correct'],'new_wrong_return':r['status']=='VALID' and not r['correct']})
    write_rows(analysis/'PRE_POST_TRANSITIONS.jsonl.gz',transitions)
    write_rows(analysis/'FIRST_LINE_SENSITIVITY.jsonl.gz',first)
    write_rows(analysis/'SUBJECTS.jsonl.gz',subjects)
    write_rows(analysis/'SUBJECT_FIXES_HARMS.jsonl.gz',subject_harms)
    write_rows(analysis/'REFUSAL_TRANSITIONS.jsonl.gz',rescues)
    result={'pre_post_target_transitions':len(transitions),'native_pre':'NOT_MEASURED',
        'first_line_sensitivity':sensitivity,'subject_fix_harm_rows':len(subject_harms),
        'refusal_transition_rows':len(rescues),'interpretation':'Supplementary historical first-line parsers, not replacements for common whole-response primary codec',
        'neural_inference_performed':False}
    dump(analysis/'SUPPLEMENTARY_SUMMARY.json',result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True,type=Path);a=p.parse_args()
    print(json.dumps(supplement(a.analysis),indent=2))
