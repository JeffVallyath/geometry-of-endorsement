"""CPU-only reporting completeness; consumes actual outputs, never calls a model.

Preserves all original headline metrics. Executes each uncached control selection
against source state, and each current/original/current triplet against one retained
snapshot store with its result cache explicitly cleared for every query.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'PPR32_Repair_128'
sys.path.insert(0,str(PACKAGE/'implementation'))
from prepare_data import load,gzrows,dump,write_rows,canonical
from execution import build_store
from ranker_contract import choose,key,request
from state_grounding import current_payload
from prepare import candidates as make_candidates
from codec import norm

def records(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f]

def exposure_ledger():
    questions={norm(json.loads(r['request']['messages'][1]['content'])['question'])
               for r in gzrows(PACKAGE/'training/training_pairs.jsonl.gz')}
    ledger=[{'case_id':e['case_id'],'query_id':e['query_id'],'question':e['question'],
             'branch':e['branch'],'exact_training_question_overlap':norm(e['question']) in questions}
            for e in load(PACKAGE/'prepared/EVALUATOR_ONLY.json')]
    assert len(ledger)==901
    flagged={(r['case_id'],r['query_id']) for r in ledger if r['exact_training_question_overlap']}
    original=load(PACKAGE/'legacy_ppr/audit/SUPERVISED_EXPOSURE.json')
    assert flagged=={(r['case_id'],r['query_id']) for r in original['overlaps']}
    counts=collections.Counter(r['branch'] for r in ledger if r['exact_training_question_overlap'])
    assert dict(counts)=={'F':21,'Fdecoy':12} and len(flagged)==33
    return ledger

def state_requests(case,store,row):
    state=store.view(0 if row['phase']=='pre' else None)
    triples=sorted([s,p,o] for (s,p),o in state.items())
    labels={k:[v] for k,v in case['labels'].items()}
    registry=load(PACKAGE/'legacy_ppr/mapping/RELATION_REGISTRY.json')
    payload=current_payload(triples,[],labels,registry)
    # Reuse the original all-literal expansion and canonical program ordering.
    paths=make_candidates(triples,labels,row['question'])
    if paths!=row['candidates']:raise ValueError('ACTUAL_STORE_CANDIDATES_CHANGED')
    reqs=[{'kind':'score',**request(row['question'],payload,p)} for p in paths]
    ids=[hashlib.sha256(canonical(r).encode()).hexdigest() for r in reqs]
    if ids!=row['score_request_ids']:raise ValueError('ACTUAL_STORE_REQUESTS_CHANGED')
    return hashlib.sha256(canonical(payload).encode()).hexdigest()

def execute_event(case,store,row,actual,gold):
    if len(actual)!=len(row['score_request_ids']) or [r['request_id'] for r in actual]!=row['score_request_ids']:
        raise ValueError('UNCACHED_EVENT_SCHEDULE_MISMATCH')
    for r in actual:
        if not math.isfinite(r['margin']) or abs(r['margin']-(r['yes_logit']-r['no_logit']))>1e-6:
            raise ValueError('UNCACHED_MARGIN_MISMATCH')
    payload_hash=state_requests(case,store,row)
    selected=choose(row['candidates'],[r['margin'] for r in actual])
    store.cache.clear()  # This audit explicitly requests uncached deterministic execution.
    if selected is None:
        result={'entity':None,'answer':None,'status':'REFUSAL','trace':None}
    else:
        trace=store.query(selected['start_id'],selected['path'],0 if row['phase']=='pre' else None)
        if trace['cache_hit']:raise ValueError('UNCACHED_EXECUTOR_CACHE_HIT')
        if trace['endpoint']!=selected['endpoint'] or trace['dependencies']!=selected['edges']:
            raise ValueError('UNCACHED_EXECUTION_MISMATCH')
        # Independent traversal of the explicitly selected snapshot.
        mapping=store.view(0 if row['phase']=='pre' else None);node=selected['start_id']
        for rel in selected['path']:node=mapping[node,rel]
        if node!=trace['endpoint']:raise ValueError('UNCACHED_DICTIONARY_DISAGREEMENT')
        result={'entity':node,'answer':case['labels'][node],'status':'VALID','trace':trace}
    return {**{k:row[k] for k in ('logical_id','case_id','history','phase','query_id','question')},
        'current_or_selected_snapshot_payload_sha256':payload_hash,'selected':selected,
        'margins':[r['margin'] for r in actual],'actual_call_seconds':sum(r['seconds'] for r in actual),
        'request_ids':row['score_request_ids'],**result,
        'correct':result['entity']==gold['entity'],
        'faithful':selected is not None and selected['start_id']==gold['start'] and selected['path']==gold['path'],
        'gold_used_only_after_selection':True,'executor_cache_hit':False}

def build_reports(result_root=None,out=None):
    result_root=Path(result_root or ROOT/'results');out=Path(out or ROOT/'analysis')
    exposure=exposure_ledger();dump(out/'TRAINING_EXPOSURE_LEDGER.json',exposure)
    exposed={(r['case_id'],r['query_id']):r['exact_training_question_overlap'] for r in exposure}
    cases={c['case_id']:c for c in load(PACKAGE/'legacy_ppr/prepared/RUNTIME_CASES.json')}
    ev={(r['case_id'],r['query_id']):r for r in load(PACKAGE/'prepared/EVALUATOR_ONLY.json')}
    logical={r['logical_id']:r for r in gzrows(PACKAGE/'prepared/LOGICAL_ROWS.jsonl.gz')}
    controls=load(PACKAGE/'prepared/CONTROLS.json')
    events=[];return_summaries=[]
    for arm,stage in [('STATE_BASE','base'),('STATE_0','seed0'),('STATE_1','seed1')]:
        for kind in ('repeat','return'):
            rows=records(result_root/f'{stage}_{kind}/responses.jsonl')
            expected=load(PACKAGE/f'prepared/{"REPEAT" if kind=="repeat" else "RETURN"}_REQUEST_IDS.json')
            if len(rows)!=len(expected) or [r['request_id'] for r in rows]!=expected:
                raise ValueError('INCOMPLETE_UNCACHED_CONTROL_SCHEDULE')
            groups=[[lid] for lid in controls['repeat_rows']] if kind=='repeat' else [p['sequence'] for p in controls['return_programs']]
            offset=0
            for group_index,group in enumerate(groups):
                first=logical[group[0]];case=cases[first['case_id']]
                store=build_store(case,first['history'])
                before=dict(store.view());original=dict(store.view(0));triplet=[]
                for position,lid in enumerate(group):
                    row=logical[lid];n=len(row['score_request_ids']);actual=rows[offset:offset+n];offset+=n
                    e=ev[row['case_id'],row['query_id']];gold=e['post'] if row['phase']=='post' else e['pre'][row['history']]
                    event=execute_event(case,store,row,actual,gold)
                    event.update(arm=arm,control=kind,group_index=group_index,position=position,
                                 exact_training_question_overlap=exposed[row['case_id'],row['query_id']])
                    events.append(event);triplet.append(event)
                if store.view()!=before or store.view(0)!=original:raise ValueError('READ_MUTATED_SNAPSHOTS')
                if kind=='return':
                    a,h,b=triplet
                    return_summaries.append({'arm':arm,'program_index':group_index,'case_id':first['case_id'],
                        'query_id':first['query_id'],'same_retained_store':True,'executor_result_cache_used':False,
                        'snapshots_unchanged':True,'current_payload_preserved':a['current_or_selected_snapshot_payload_sha256']==b['current_or_selected_snapshot_payload_sha256'],
                        'current_selection_preserved':(key(a['selected']) if a['selected'] else None)==(key(b['selected']) if b['selected'] else None),
                        'current_answer_preserved':(a['status'],a['entity'])==(b['status'],b['entity']),
                        'current_before_correct':a['correct'],'historical_correct':h['correct'],
                        'current_after_correct':b['correct'],'historical_faithful':h['faithful']})
            assert offset==len(rows)
    assert len(events)==3*(492+48) and len(return_summaries)==3*16
    write_rows(out/'UNCACHED_EXECUTION_TRACES.jsonl.gz',events)
    write_rows(out/'UNCACHED_ERRORS_AND_REFUSALS.jsonl.gz',[r for r in events if not r['correct'] or r['status']!='VALID'])
    dump(out/'CURRENT_ORIGINAL_CURRENT_LEDGER.json',return_summaries)
    outcomes=gzrows(out/'ALL_OUTCOMES.jsonl.gz')
    readable=[]
    for row in outcomes:
        case=cases[row['case_id']];e=ev[row['case_id'],row['query_id']]
        gold=e['post'] if row['phase']=='post' else e['pre'][row['history']]
        readable.append({**row,'question':e['question'],
            'expected_entity':gold['entity'],'expected_answer':case['labels'][gold['entity']],
            'returned_answer':case['labels'].get(row['entity']) if row['entity'] else None,
            'exact_training_question_overlap':exposed[row['case_id'],row['query_id']]})
    assert len(readable)==40116
    write_rows(out/'ALL_ANSWERS_WITH_SOURCE_LABELS_AND_EXPOSURE.jsonl.gz',readable)
    summary={'actual_uncached_reads':len(events),'actual_uncached_repeat_reads_per_scorer':492,
        'current_original_current_programs_per_scorer':16,'executor_cache_hits':0,
        'training_overlap_occurrences':33,'target_training_question_overlap':0,'K_training_question_overlap':0,
        'return_programs':{arm:{'historical_correct':sum(x['historical_correct'] for x in return_summaries if x['arm']==arm),
            'historical_total':16,'current_preservation_failures':sum(not(x['current_payload_preserved'] and x['current_selection_preserved'] and x['current_answer_preserved']) for x in return_summaries if x['arm']==arm)}
            for arm in ('STATE_BASE','STATE_0','STATE_1')},
        'primary_headline_metrics_modified':False,'neural_inference_performed_by_this_script':False}
    dump(out/'REPORTING_COMPLETENESS.json',summary)
    return summary

if __name__=='__main__':print(json.dumps(build_reports(),indent=2))
