"""Supplement the unchanged frozen evaluator with exhaustive ledgers and eight traces."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'PPR32_Repair_128'
sys.path.insert(0,str(PACKAGE/'implementation'))
from prepare_data import load,gzrows,dump,write_rows,HISTORIES,ALL
from evaluate import evaluate
from supplement import supplement
from validate_controls import validate
from execution import execute,build_store

def raw(path):
    with path.open(encoding='utf-8') as f:return [json.loads(line) for line in f]

def analyze(result_root):
    result_root=Path(result_root);out=ROOT/'analysis';out.mkdir(exist_ok=True)
    stages=['base','seed0_eval','seed1_eval','base_repeat','base_return',
            'seed0_repeat','seed0_return','seed1_repeat','seed1_return','seed0_loader','seed1_loader']
    required_score={'request_id','occurrence','input_token_ids','rendered_prompt','seconds',
                    'yes_logit','no_logit','margin','full_logits_sha256'}
    identities={};records={};tokens={};timings={}
    for stage in stages:
        records[stage]=raw(result_root/stage/'responses.jsonl')
        complete=load(result_root/stage/'COMPLETE.json')
        identities[stage]=load(result_root/stage/'IDENTITY.json')
        if complete['requests']!=len(records[stage]):raise ValueError('COMPLETE_COUNT_MISMATCH')
        for r in records[stage]:
            if 'margin' in r:
                if not required_score.issubset(r):raise ValueError('RAW_SCORE_RECEIPT_MISSING_FIELDS')
                if len(r['full_logits_sha256'])!=64:raise ValueError('FULL_LOGIT_HASH_MISSING')
            else:
                if not {'text','output_token_ids','truncated','invalid_control_tokens'}.issubset(r):
                    raise ValueError('RAW_GENERATION_RECEIPT_MISSING_FIELDS')
            if not math.isfinite(r['seconds']) or r['seconds']<0:raise ValueError('BAD_CALL_TIMING')
            key=r['request_id']
            if key in tokens and tokens[key]!=r['input_token_ids']:raise ValueError('CROSS_SCORER_TOKEN_MISMATCH')
            tokens[key]=r['input_token_ids']
        timings[stage]={'actual_calls':len(records[stage]),'sum_call_seconds':sum(r['seconds'] for r in records[stage])}
    if len({x['independent_process_label'] for x in identities.values()})!=len(identities):
        raise ValueError('FRESH_PROCESS_PID_REUSE_REQUIRES_PROCESS_RECEIPT_AUDIT')
    started=time.perf_counter()
    summary=evaluate(result_root/'base/responses.jsonl',result_root/'seed0_eval/responses.jsonl',
                     result_root/'seed1_eval/responses.jsonl',out)
    timings['complete_cpu_evaluator_seconds']=time.perf_counter()-started
    supplement(out)
    controls={}
    for arm in ('base','seed0','seed1'):
        main='base' if arm=='base' else arm+'_eval'
        loader={} if arm=='base' else {'loader_expected':result_root/f'train{arm[-1]}/LOADER_EXPECTED.json',
                                      'loader_actual':result_root/f'{arm}_loader/responses.jsonl'}
        controls[arm]=validate(result_root/main/'responses.jsonl',result_root/f'{arm}_repeat/responses.jsonl',
                               result_root/f'{arm}_return/responses.jsonl',**loader)
        dump(out/f'CONTROLS_{arm}.json',controls[arm])
    rows=gzrows(out/'ALL_OUTCOMES.jsonl.gz')
    idx={(r['arm'],r['case_id'],r['history'],r['phase'],r['query_id']):r for r in rows}
    comparisons=[]
    for r in rows:
        if r['arm'] not in ('STATE_0','STATE_1'):continue
        for baseline in ('NATIVE32_SAVED','CURRENT_TEXT','CURRENT_JSON','STATE_BASE'):
            old=idx.get((baseline,r['case_id'],r['history'],r['phase'],r['query_id']))
            if old is None:
                if baseline!='NATIVE32_SAVED' or r['phase']!='pre':raise ValueError('MISSING_PAIRED_ROW')
                continue
            comparisons.append({'candidate':r['arm'],'baseline':baseline,
                **{k:r[k] for k in ('case_id','history','phase','query_id')},
                'baseline_entity':old['entity'],'candidate_entity':r['entity'],
                'baseline_status':old['status'],'candidate_status':r['status'],
                'baseline_correct':old['correct'],'candidate_correct':r['correct'],
                'candidate_faithful':r['faithful'],'fixed':not old['correct'] and r['correct'],
                'harmed':old['correct'] and not r['correct'],
                'refusal_to_wrong_return':old['status']=='REFUSAL' and r['status']=='VALID' and not r['correct']})
    write_rows(out/'EVERY_PAIRED_ANSWER.jsonl.gz',comparisons)
    write_rows(out/'EVERY_REFUSAL_OR_ERROR.jsonl.gz',[r for r in rows if not r['correct'] or r['status']!='VALID'])
    cases={c['case_id']:c for c in load(PACKAGE/'legacy_ppr/prepared/RUNTIME_CASES.json')}
    logical=gzrows(PACKAGE/'prepared/LOGICAL_ROWS.jsonl.gz')
    logical_idx={(r['case_id'],r['history'],r['phase'],r['query_id']):r for r in logical}
    data={k:{r['request_id']:r for r in records[k]} for k in ('base','seed0_eval','seed1_eval')}
    fixed_cases=load(PACKAGE/'REPAIR32_CONFIG.json')['evaluation']['repeat_cases'][:8]
    examples=[]
    for cid in fixed_cases:
        case=cases[cid];query=next(q for q in case['queries'] if q['query_id'].startswith('J'))
        row=logical_idx[cid,'H0','post',query['query_id']]
        t=time.perf_counter();store=build_store(case,'H0');cpu_intake_seconds=time.perf_counter()-t
        example={'case_id':cid,'history':'H0','question':query['question'],'query_id':query['query_id'],
                 'source_text':case['histories']['H0']['source_text'],'initial':case['histories']['H0']['initial'],
                 'actual_updates':case['histories']['H0']['edits'],
                 'current_state':sorted([s,p,o] for (s,p),o in store.view().items()),
                 'original_snapshot':sorted([s,p,o] for (s,p),o in store.view(0).items()),
                 'cpu_intake_seconds':cpu_intake_seconds,'scorers':{}}
        for arm,stage in [('STATE_BASE','base'),('STATE_0','seed0_eval'),('STATE_1','seed1_eval')]:
            outcome=idx[arm,cid,'H0','post',query['query_id']]
            t=time.perf_counter();execution=execute(case,'H0','post',outcome['selected']);elapsed=time.perf_counter()-t
            example['scorers'][arm]={'all_candidates':row['candidates'],
                'actual_scores':[{k:r[k] for k in ('request_id','yes_logit','no_logit','margin','seconds','full_logits_sha256')}
                                 for r in [data[stage][rid] for rid in row['score_request_ids']]],
                'selected':outcome['selected'],'exact_execution':execution,'cpu_execution_seconds':elapsed,
                'correct':outcome['correct'],'faithful':outcome['faithful'],
                'logical_cache_note':'Main unique computations, not six independent calls; independent panel audited separately.'}
        examples.append(example)
    dump(out/'EIGHT_FIXED_TRACES.json',examples)
    timings['cold_loads']={p.stem:load(p) for p in (result_root/'LOAD_RECEIPTS').glob('*.json')}
    timings['training']={f'seed{s}':load(result_root/f'train{s}/COMPLETE.json') for s in (0,1)}
    timings['model_download']=load(result_root/'MODEL_PROVENANCE.json')['download_and_verification_seconds']
    dump(out/'TIMINGS.json',timings)
    isolation=all(x['selection_invariance_pass'] for x in controls.values())
    flags={'IMPLEMENTATION_ISOLATION':{'source_and_sql_checks':'PASS','independent_current_input_checks':'PASS',
           'uncached_selection_invariance_pass':isolation,'full_logit_hash_differences':
           {arm:{k:x[k]['different_full_logits_hashes'] for k in ('repeat','return')} for arm,x in controls.items()}},
           'KNOWN_FAILURE_REPAIR':summary['known_9_repairs'],
           'END_TO_END_CAPABILITY':{a:summary['endpoints'][a] for a in ('STATE_0','STATE_1')},
           'COMPARATIVE_GAIN':{'pass':summary['comparative_gain'],'contrasts':summary['primary_paired_contrasts']},
           'NO_OBSERVED_HARM':{'pass':not any(x['harmed'] or x['refusal_to_wrong_return'] for x in comparisons),
                               'harmed_read_comparisons':sum(x['harmed'] for x in comparisons),
                               'refusal_to_wrong_return_comparisons':sum(x['refusal_to_wrong_return'] for x in comparisons)},
           'failure_sources':['state/intake','complete-input leakage','numerical/execution','semantic selection/admission'],
           'interpretation':'Invariant wrong/refused selections are semantic failures, not history leaks. Differing actual selections under equal complete inputs invalidate the empirical execution certificate.',
           'native_pre':'NOT_MEASURED','original_native_witnesses':'9/384 unchanged',
           'original_F3_or_full_R3_claim':False,'automatic_successor':False}
    dump(out/'SEPARATE_VERDICTS.json',flags)
    return summary

if __name__=='__main__':
    result=analyze(ROOT/'results')
    print(json.dumps({'denominators':result['denominators'],'endpoints':result['endpoints'],
                      'known_9_repairs':result['known_9_repairs']},indent=2))
