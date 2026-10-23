"""Controller-only PPR1 evaluation; never part of the inference payload."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'input/PPR1_Same_Case_Public_Repair'
sys.path[:0]=[str(INPUT/'implementation'),str(INPUT/'vendor')]
from metrics import paired_interval
from ppr_requests import record_view
from execution import execute
from ranker_contract import choose
ARMS=['HISTORY','REFLECT','CURRENT_TEXT','CURRENT_JSON','STATE_BASE','STATE_0','STATE_1']
HISTORIES=['H0','H1','S0','S1']

def event_order(row,kind):
    key=kind+'_event_order'
    return row[key] if key in row else row[kind+'_utc']


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def load_lines(path):
    with path.open(encoding='utf-8') as handle:return [json.loads(line) for line in handle]
def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:handle.write(canonical(row)+'\n')
def key(r):return (r['case_id'],r['history'],r['phase'],r['query_id'],r['arm'])
def lid(r):return '/'.join(str(r[k]) for k in ('case_id','history','phase','query_id','arm'))
def nid(r):return '/'.join(str(r[k]) for k in ('case_id','history','phase'))

def historical():
    """Execute original analysis functions unchanged; only absent archive IO imports are omitted."""
    path=INPUT/'reference/mqmh.py'; tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not n.level]
    module=types.ModuleType('frozen_mqmh_replay');module.__dict__['ROOT']=INPUT
    exec(compile(tree,str(path),'exec'),module.__dict__)
    cohort=json.loads(gzip.decompress((INPUT/'source/cohort.json.gz').read_bytes()))
    responses={name:json.loads(gzip.decompress((INPUT/f'source/{name}_responses.json.gz').read_bytes()))
               for name in ['qwen','gemma']}
    summaries,targets,example=module.analyze(cohort,responses)
    module.assert_equivalent(summaries,read(INPUT/'source/RESULTS.json'))
    assert summaries['qwen']['primary']['qualified']==94
    assert summaries['gemma']['primary']['qualified']==101
    return module,cohort,responses,summaries,targets,example

def evaluate(raw,out):
    assert not out.exists(),'preserve completed evaluator output'
    cases=read(INPUT/'prepared/RUNTIME_CASES.json');by_id={c['case_id']:c for c in cases}
    gold={(r['case_id'],r['query_id']):r for r in read(INPUT/'prepared/EVALUATOR_ONLY.json')}
    schedule=read(INPUT/'prepared/SCHEDULE.json')
    exposure=read(INPUT/'audit/SUPERVISED_EXPOSURE.json')
    exposed={(r['case_id'],r['query_id']) for r in exposure['overlaps']}
    assert len(exposed)==33 and all(not q.startswith('J') and q!='K' for _,q in exposed)
    calls={r['call_id']:r for r in load_lines(raw/'calls.jsonl')}
    notes={r['note_id']:r for r in load_lines(raw/'notes.jsonl')}
    rows=load_lines(raw/'logical.jsonl');lookup={key(r):r for r in rows}
    assert len(rows)==len(lookup)==48594 and set(lookup)=={key(r) for r in schedule['logical_answer_reads']}
    assert len(notes)==1280 and set(notes)=={nid(r) for r in schedule['reflection_preparations']}
    assert all(r['call_id'] in calls for r in notes.values())
    scored=[]
    for row in rows:
        assert row['logical_id']==lid(row)
        assert all(cid in calls for cid in row['call_ids'])
        ev=gold[row['case_id'],row['query_id']]
        expected=ev['post'] if row['phase']=='post' else ev['pre'][row['history']]
        entity=row['outcome']['entity'];correct=entity is not None and entity==expected['entity']
        program=None
        if row['arm'].startswith('STATE'):
            program=row['selected'] is not None and row['selected']['start_id']==expected['start'] and row['selected']['path']==expected['path']
            assert choose(row['candidates'],row['margins'])==row['selected']
            assert execute(by_id[row['case_id']],row['history'],row['phase'],row['selected'])==row['outcome']
        edges,payload=record_view(by_id[row['case_id']],row['history'],row['phase'])
        assert sha(edges)==row['state_sha256'] and sha(payload)==row['request_payload_sha256']
        if row['arm']=='REFLECT':
            note=notes[row['note_id']];note_call=calls[note['call_id']]
            assert event_order(note_call,'completed')<=event_order(calls[row['call_ids'][0]],'started')
            assert calls[row['call_ids'][0]]['request']['messages'][1]['content']==note_call['request']['messages'][1]['content']
            assert calls[row['call_ids'][0]]['request']['messages'][2]['content']==note['text']
        scored.append({**row,'branch':ev['branch'],'expected_entity':expected['entity'],
                       'researcher_exposed':True,'qsa1_supervised_question_overlap':(row['case_id'],row['query_id']) in exposed,
                       'expected_program':expected,'correct':correct,'program_faithful':program,
                       'wrong_program_right_answer':program is False and correct})
    lookup={key(r):r for r in scored};ids=[c['case_id'] for c in cases]
    def get(cid,h,q,a,phase='post'):return lookup[cid,h,phase,q,a]
    complete={};family={};arm_summary={}
    for arm in ARMS:
        complete[arm]=[all(get(cid,h,q['query_id'],arm)['correct'] for h in HISTORIES for q in by_id[cid]['queries']) for cid in ids]
        selected=[r for r in scored if r['arm']==arm and r['phase']=='post']
        family[arm]={}
        for role in ['J','F','Fdecoy','K']:
            queries=[(cid,q['query_id']) for cid in ids for q in by_id[cid]['queries'] if gold[cid,q['query_id']]['branch']==role]
            flags=[all(get(cid,h,qid,arm)['correct'] for h in HISTORIES) for cid,qid in queries]
            occurrences=[r['correct'] for r in selected if r['branch']==role and r['history'] in HISTORIES]
            family[arm][role]={'queries':len(flags),'correct_all_four_histories':sum(flags),
                'robust_query_accuracy':sum(flags)/len(flags),'history_occurrences':len(occurrences),
                'correct_history_occurrences':sum(occurrences),'occurrence_accuracy':sum(occurrences)/len(occurrences)}
        all_targets=[all(get(cid,h,qid,arm)['correct'] for h in HISTORIES) for cid in ids for qid in ['J0','J1','J2']]
        program_rows=[r for r in selected if r['history'] in HISTORIES and r['program_faithful'] is not None]
        absolute=sum(complete[arm])/128>=.95 and all(v['robust_query_accuracy']>=.99 for v in family[arm].values())
        agreements=[]
        for cid in ids:
            for q in by_id[cid]['queries']:
                views=[get(cid,h,q['query_id'],arm) for h in [*HISTORIES,'C','C+']]
                agreements.append({'state_equal':len({r['state_sha256'] for r in views})==1,
                    'entity_equal':len({r['outcome']['entity'] for r in views})==1,
                    'raw_output_equal':len({r.get('raw_text',r['outcome'].get('answer')) for r in views})==1,
                    'uniformly_wrong':all(not r['correct'] for r in views),
                    'uniform_refusal':all(r['outcome']['status']=='REFUSAL' for r in views)})
        arm_summary[arm]={'complete_subjects':sum(complete[arm]),'subjects':128,'complete_rate':sum(complete[arm])/128,
            'complete_rate_interval':paired_interval(complete[arm],[0]*128,level=.95),
            'agreement_separate_from_correctness':{'queries':901,**{name:sum(r[name] for r in agreements) for name in agreements[0]},
                'same_entity_but_all_wrong':sum(r['entity_equal'] and r['uniformly_wrong'] for r in agreements)},
            'complete_original_target_phrasings':sum(all_targets),'original_target_phrasings':384,
            'all_six_post_occurrences':len(selected),'all_six_correct':sum(r['correct'] for r in selected),
            'status_counts':dict(Counter(r['outcome']['status'] for r in selected)),
            'wrong_program_right_answer':sum(r['wrong_program_right_answer'] for r in selected),
            'program_faithful_occurrences':sum(r['program_faithful'] for r in program_rows),
            'program_scored_occurrences':len(program_rows),'endpoint_absolute_floors':absolute,
            'exact_program_obligation':all(r['program_faithful'] for r in program_rows) if program_rows else None,
            'family':family[arm]}
    seedmean=(np.asarray(complete['STATE_0'],float)+np.asarray(complete['STATE_1'],float))/2
    contrasts={}
    for base in ['CURRENT_TEXT','REFLECT','CURRENT_JSON','STATE_BASE','HISTORY']:
        c=paired_interval(seedmean,complete[base]);c['useful_gain']=c['difference']>=.05 and c['interval'][0]>0
        c['primary']=base in ['CURRENT_TEXT','REFLECT'];contrasts[base]=c
    paired=[];signed={}
    for base in ['HISTORY','REFLECT','CURRENT_TEXT','CURRENT_JSON']:
        for candidate in ARMS:
            if base==candidate:continue
            root_harms=[float(complete[base][i])-float(complete[candidate][i]) for i in range(128)]
            signed[f'{candidate}-vs-{base}']={'signed_root_harm_95pct':paired_interval(root_harms,[0]*128,level=.95),'family':{}}
            for role in ['J','F','Fdecoy','K']:
                means=[]
                for cid in ids:
                    qs=[q['query_id'] for q in by_id[cid]['queries'] if gold[cid,q['query_id']]['branch']==role]
                    # Zero weight for roots without this family; preserve all 128 clusters and disclose it.
                    differences=[float(get(cid,h,q,base)['correct'])-float(get(cid,h,q,candidate)['correct']) for h in HISTORIES for q in qs]
                    means.append(sum(differences)/len(differences) if differences else 0.)
                signed[f'{candidate}-vs-{base}']['family'][role]=paired_interval(means,[0]*128,level=.95)
            for cid in ids:
                for q in by_id[cid]['queries']:
                    qid=q['query_id'];b=all(get(cid,h,qid,base)['correct'] for h in HISTORIES)
                    a=all(get(cid,h,qid,candidate)['correct'] for h in HISTORIES)
                    paired.append({'case_id':cid,'query_id':qid,'branch':gold[cid,qid]['branch'],
                        'baseline':base,'candidate':candidate,'baseline_robust_correct':b,'candidate_robust_correct':a,
                        'fixed':not b and a,'introduced_harm':b and not a,
                        'candidate_refusal':any(get(cid,h,qid,candidate)['outcome']['status']=='REFUSAL' for h in HISTORIES)})
    module,cohort,responses,oldsummary,oldtargets,example=historical()
    sourcecases={c['case_id']:c for c in cohort['cases']}
    oldflags={(r['model'],r['case_id'],r['query_id']):r['qualified'] for r in oldtargets}
    oldlookup={(r['case_id'],r['arm'],r['query_id']):r for r in responses['qwen']['primary']}
    new_native={}
    for cid in ids:
        native={(cid,h,q['query_id']):{'correct':get(cid,h,q['query_id'],'HISTORY')['correct'],
            'entity_id':get(cid,h,q['query_id'],'HISTORY')['outcome']['entity']}
            for h in [*HISTORIES,'C','C+'] for q in by_id[cid]['queries']}
        for row in module.endpoints(sourcecases[cid],native)['targets']:new_native[cid,row['query_id']]=row
    crosswalk=[]
    for cid in ids:
        for qid in ['J0','J1','J2']:
            controlkeys=[(h,q['query_id']) for h in [*HISTORIES,'C','C+'] for q in by_id[cid]['queries'] if not q['query_id'].startswith('J')]
            controlkeys += [('C',qid),('C+',qid)]
            old_correct=lambda h,q: bool(oldlookup[cid,h,q]['valid'] and oldlookup[cid,h,q]['entity_id']==gold[cid,q]['post']['entity'])
            record={'case_id':cid,'query_id':qid,'historical_qwen_flag':oldflags['qwen',cid,qid],
                    'historical_gemma_flag':oldflags['gemma',cid,qid],'new_HISTORY':new_native[cid,qid],
                    'historical_witness_reproduced':oldflags['qwen',cid,qid] and new_native[cid,qid]['qualified'],
                    'arms':{}}
            for arm in ARMS:
                target=all(get(cid,h,qid,arm)['correct'] for h in HISTORIES)
                oldharms=[{'history':h,'query_id':q} for h,q in controlkeys if old_correct(h,q) and not get(cid,h,q,arm)['correct']]
                newharms=[{'history':h,'query_id':q} for h,q in controlkeys if get(cid,h,q,'HISTORY')['correct'] and not get(cid,h,q,arm)['correct']]
                record['arms'][arm]={'all_history_target_correct':target,'lost_original_controls':oldharms,
                    'lost_new_HISTORY_controls':newharms,'historical_preserved_repair':target and not oldharms,
                    'new_HISTORY_preserved_repair':target and not newharms}
            crosswalk.append(record)
    repairs={}
    for title,predicate,field in [('all_original_94',lambda r:r['historical_qwen_flag'],'historical_preserved_repair'),
        ('reproduced_original_subset',lambda r:r['historical_witness_reproduced'],'historical_preserved_repair'),
        ('all_new_HISTORY_witnesses',lambda r:r['new_HISTORY']['qualified'],'new_HISTORY_preserved_repair'),
        ('newly_qualified_not_historical',lambda r:r['new_HISTORY']['qualified'] and not r['historical_qwen_flag'],'new_HISTORY_preserved_repair')]:
        subset=[r for r in crosswalk if predicate(r)]
        repairs[title]={'denominator':len(subset),'arms':{arm:{'repaired':sum(r['arms'][arm][field] for r in subset),
            'rate':sum(r['arms'][arm][field] for r in subset)/len(subset) if subset else None} for arm in ARMS}}
    assert repairs['all_original_94']['denominator']==94
    transitions=[]
    for cid in ids:
        for h in HISTORIES:
            for qid in ['J0','J1','J2']:
                for arm in ARMS:
                    before=get(cid,h,qid,arm,'pre');after=get(cid,h,qid,arm)
                    transitions.append({'case_id':cid,'history':h,'query_id':qid,'arm':arm,
                        'pre_correct':before['correct'],'post_correct':after['correct'],
                        'both_correct':before['correct'] and after['correct'],
                        'old_answer_error_after_update':after['outcome']['entity']==before['expected_entity'] and before['expected_entity']!=after['expected_entity']})
    controls=load_lines(raw/'controls.jsonl');assert len(controls)==48
    timings=load_lines(raw/'timings.jsonl');selected=read(INPUT/'prepared/CONTROL_SELECTION.json')
    assert {(r['case_id'],r['arm']) for r in timings}=={(cid,arm) for cid in selected['timed_subjects'] for arm in ARMS}
    traces=[]
    for cid in selected['timed_subjects']:
        traces.append({'case_id':cid,'query_id':'J0','history':'H0','phase':'post',
            'source':by_id[cid]['histories']['H0'],
            'computed_original':record_view(by_id[cid],'H0','pre')[0],
            'computed_current':record_view(by_id[cid],'H0','post')[0],
            'arms':{arm:get(cid,'H0','J0',arm) for arm in ARMS}})
    labels=[]
    if all(contrasts[b]['useful_gain'] for b in ['CURRENT_TEXT','REFLECT']):labels.append('PAIRED_PUBLIC_GAIN')
    if arm_summary['CURRENT_TEXT']['endpoint_absolute_floors']:labels.append('CURRENT_REPLAY_SUFFICIENT')
    if not any(contrasts[b]['useful_gain'] for b in ['CURRENT_TEXT','REFLECT']):labels.append('NO_COMPARATIVE_GAIN')
    if sum(complete['REFLECT'])>sum(complete['HISTORY']):labels.append('REFLECTION_GAIN')
    if repairs['reproduced_original_subset']['denominator']!=94:labels.append('BASELINE_DRIFT')
    labels.append('INTERPRETATION_LIMIT')
    actual=Counter((c['runtime']['checkpoint'],c['kind']) for c in calls.values())
    summary={'study':'PPR1_SAME_CASE_PUBLIC_REPAIR','status':'MEASURED','labels':labels,
        'subjects':128,'original_targets':384,'retained_controls':517,'logical_reads':48594,'reflection_preparations':1280,
        'actual_calls':[{'checkpoint':a,'kind':b,'calls':n} for (a,b),n in sorted(actual.items())],
        'arms':arm_summary,'primary_and_descriptive_contrasts':contrasts,'historical_repairs':repairs,
        'historical_original_qualified':{'qwen':94,'gemma':101},'history_wrapper_is_new':True,
        'both_seed_absolute_capability':all(arm_summary[a]['endpoint_absolute_floors'] and arm_summary[a]['exact_program_obligation'] for a in ['STATE_0','STATE_1']),
        'uncached_control_pairs':48,'timed_subjects':8,'real_traces':8,
        'limitations':['All 128 subjects are researcher-exposed.','33 retained direct/decoy occurrences overlap supervised QSA1 questions; targets and remaining-chain queries do not.',
            'Exact inversion of 32 source templates is a transport privilege, not arbitrary language parsing.',
            'Historical C/C+ guard-selected correctness is not new evidence.','No equal-compute, fresh-prevalence, general prompting-impossibility, production-safety or novel-algorithm claim.',
            'Signed family harm intervals retain all 128 clusters; absent-family subjects contribute zero.'],
        'agent_token_charges':'unavailable'}
    dump(out/'SUMMARY.json',summary);dump(out/'ROOT_VECTORS.json',{'case_ids':ids,'complete':complete,'paired_seed_mean':seedmean.tolist()})
    lines(out/'SCORED_LOGICAL.jsonl',scored);lines(out/'PAIRED_FIXES_HARMS.jsonl',paired)
    dump(out/'HISTORICAL_CROSSWALK.json',crosswalk);dump(out/'HISTORICAL_REPLAY.json',oldsummary)
    dump(out/'HARM_INTERVALS.json',signed);lines(out/'BEFORE_AFTER.jsonl',transitions)
    dump(out/'EXPOSURE_TAGS.json',exposure)
    for i,trace in enumerate(traces):dump(out/f'traces/{i+1:02d}_{trace["case_id"]}.json',trace)
    dump(out/'STATE_SNAPSHOTS.json',[{'case_id':cid,'history':h,'initial':record_view(by_id[cid],h,'pre')[0],
        'current':record_view(by_id[cid],h,'post')[0]} for cid in ids for h in [*HISTORIES,'C','C+']])
    dump(out/'ACCOUNTING.json',{'logical_reads':len(rows),'unique_calls':len(calls),'reflection_logical_notes':len(notes),
        'actual_note_calls':len({n['call_id'] for n in notes.values()}),'logical_call_references':sum(len(r['call_ids']) for r in rows),
        'call_batches':len({(event_order(c,'started'),c['runtime']['checkpoint']) for c in calls.values()}),
        'cache_reuse_is_not_independent_neural_success':True})
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=evaluate(args.raw,args.out)
    print(json.dumps({'status':result['status'],'labels':result['labels'],'complete_subjects':{k:v['complete_subjects'] for k,v in result['arms'].items()}},indent=2))
