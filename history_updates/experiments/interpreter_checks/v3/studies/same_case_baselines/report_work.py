"""Descriptive workload accounting from saved calls; never modifies frozen scoring."""
from pathlib import Path
import collections
import json
from evaluate import read,load_lines,write
from baseline_core import ENDPOINTS

def call_ids(value):
    found=set()
    if isinstance(value,dict):
        if isinstance(value.get('call_id'),str):found.add(value['call_id'])
        if isinstance(value.get('call_ids'),list):found.update(value['call_ids'])
        for child in value.values():
            if isinstance(child,(dict,list)):found.update(call_ids(child))
    elif isinstance(value,list):
        for child in value:found.update(call_ids(child))
    return found

def account_calls(ids,calls,freeze_epoch):
    ids=set(ids);assert ids<=calls.keys(),'MISSING_REFERENCED_NEURAL_CALL'
    selected=[calls[key] for key in sorted(ids)]
    return {'unique_neural_calls':len(selected),
            'input_tokens':sum(c['result']['input_tokens'] for c in selected),
            'generated_tokens':sum(c['result'].get('output_tokens',0) for c in selected),
            'candidate_score_calls':sum(c['kind']=='score' for c in selected),
            'generation_calls':sum(c['kind']=='generate' for c in selected),
            'amortized_neural_seconds':sum(c['result']['amortized_seconds'] for c in selected),
            'prefreeze_exact_request_reuse_calls':sum(not c['started_after_scientific_freeze'] for c in selected),
            'generation_cap_hits':sum(bool(c['result'].get('cap_hit')) for c in selected)}

def report_work(raw,evaluator_data,out):
    raw=Path(raw);data=Path(evaluator_data);out=Path(out)
    call_rows=load_lines(raw/'results/calls.jsonl');calls={c['call_id']:c for c in call_rows};assert len(calls)==len(call_rows)
    logical=load_lines(raw/'results/logical.jsonl');controls=load_lines(raw/'results/controls.jsonl');timings=load_lines(raw/'results/timings.jsonl')
    freeze_epoch=None  # Original before/after relations are included per call.
    selection=read(data/'CONTROL_SELECTION.json');timed=set(selection['timed_subjects']);assert len(timed)==8
    expected=collections.Counter((r['arm'],r['case_id']) for r in read(data/'SCHEDULE.json') if r['case_id'] in timed)
    summary={};timing_failures=[]
    for endpoint in ENDPOINTS:
        main=[r for r in logical if r['arm']==endpoint]
        main_ids=call_ids(main)
        diagnostics=[r for r in controls if r['control_id'].startswith(endpoint+'/')]
        diagnostic_ids=call_ids(diagnostics)
        periods=[r for r in timings if r['endpoint']==endpoint]
        bycase={cid:[r for r in periods if r['case_id']==cid] for cid in timed}
        timed_rows=[]
        for cid,parts in sorted(bycase.items()):
            n=sum(r['logical_reads'] for r in parts)
            if n!=expected[endpoint,cid]:timing_failures.append({'endpoint':endpoint,'case_id':cid,'observed':n,'expected':expected[endpoint,cid]})
            timed_rows.append({'case_id':cid,'logical_reads':n,'expected_logical_reads':expected[endpoint,cid],
                'wall_seconds':sum(r['wall_seconds'] for r in parts),'state_preparation_seconds':sum(r['state_preparation_seconds'] for r in parts),
                'retrieval_seconds':sum(r['retrieval_seconds'] for r in parts),'new_model_calls':sum(r['new_model_calls'] for r in parts),'fragments':len(parts)})
        unique_methods={r['method_id']:r for r in main if 'method_id' in r}
        rounds=collections.Counter();routes=collections.Counter()
        for r in unique_methods.values():
            routes[r.get('route','direct')]+=1
            if endpoint=='MELLO_CURRENT':
                for step in r['trace']:rounds[str(step['round'])]+=1
        summary[endpoint]={'logical_main_reads':len(main),'main':account_calls(main_ids,calls,freeze_epoch),
            'diagnostics':account_calls(diagnostic_ids,calls,freeze_epoch),
            'combined':account_calls(main_ids|diagnostic_ids,calls,freeze_epoch),
            'unique_method_requests':len(unique_methods),'unique_method_routes':dict(routes),'mello_round_counts_unique_requests':dict(rounds),
            'timed_subjects':timed_rows,'timed_workload_wall_seconds':sum(r['wall_seconds'] for r in timed_rows),
            'timing_scope':'End-to-end active complete-case processing, including independent state preparation, retrieval, candidate enumeration, model calls, execution and rendering; model initialization and training are separate.',
            'cache_policy':'Exact complete requests only; cached logical reads are not independent neural calls or samples.',
            'breakdown_limit':'Candidate enumeration/execution/rendering are included in wall time but not separately instrumented; do not invent those sub-timings.'}
    report={'status':'PASS' if not timing_failures else 'INCOMPLETE_TIMING_COVERAGE','timing_failures':timing_failures,'endpoints':summary,
        'all_saved_calls':len(calls),'counts_include_actual_tokens_not_token_caps':True,'training_separate':True,
        'no_equal_compute_or_speed_claim':True}
    write(out/'COST_TIMING_WORKLOAD.json',report)
    table=['# Timing and workload appendix','',
        'All token counts below count saved neural executions once per complete request; repeated logical reads are not additional samples. Diagnostic calls are uncached. Training is reported separately. No equal-compute or speed advantage is inferred from matched optimizer steps.','',
        '| Endpoint | Main unique calls | Main prompt tokens | Main generated tokens | Diagnostic calls | Eight-subject end-to-end seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for a,x in summary.items():
        m=x['main'];table.append(f'| {a} | {m["unique_neural_calls"]} | {m["input_tokens"]} | {m["generated_tokens"]} | {x["diagnostics"]["unique_neural_calls"]} | {x["timed_workload_wall_seconds"]:.3f} |')
    table+=['','Timing includes state preparation, retrieval, candidate enumeration, neural calls, execution and rendering. Only state preparation and retrieval are separately instrumented; other component times are not imputed. See COST_TIMING_WORKLOAD.json for each fixed subject, candidate/round counts, cap hits, cache accounting and completeness.','',
            '']
    (out/'timing_appendix.md').write_text('\n'.join(table)+'\n',encoding='utf8')
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=report_work(args.raw,args.data,args.out)
    print(json.dumps({'status':result['status'],'timing_failures':result['timing_failures']}))
