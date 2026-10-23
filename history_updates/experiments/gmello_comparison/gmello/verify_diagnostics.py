"""Supplemental CPU verification; frozen methods, metrics and traces are unchanged."""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import math
import sys
from evaluate import read, write
from replay import lines
from gmello import digest

ROOT=Path(__file__).resolve().parent

def extracted_function(path,name,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def verify_gates(raw):
    terminal=read(raw/'COMPLETE.json')
    amended=terminal.get('main_logical_reads')==34710 and read(raw/'SCIENTIFIC_FREEZE.json').get('schema')=='GMELLO_R0_MAIN_V3_SCIENTIFIC_FREEZE'
    gate_raw=ROOT/'parent_evidence/raw' if amended else raw
    source=ROOT/'runtime_source/coordinator.py'
    if not source.exists():source=ROOT/'payload/coordinator.py'
    captured={}
    def mapped_read(path):
        relative=Path(path).relative_to(gate_raw)
        if relative.parts[:2]==('inference','data'):return read(ROOT/relative)
        if relative.parts[0]=='development_evaluator':return read(ROOT/'evaluator_data'/relative.name)
        return read(path)
    function=extracted_function(source,'development_gate',{'ROOT':gate_raw,'read':mapped_read,'collections':collections,'publish':lambda p,v:captured.update({Path(p).name:v})})
    gates=[]
    for path in sorted((gate_raw/'evidence').glob('DEV_GATE_R*.json')):
        saved=read(path); observed=function(saved['revision']); assert observed==saved,('DEV_GATE_RECOMPUTATION',path.name)
        gates.append(saved)
    terminal=read(raw/'COMPLETE.json')
    if amended:
        from authorization import validate_authorization
        frozen=read(raw/'SCIENTIFIC_FREEZE.json')
        authorization=validate_authorization(frozen,read(ROOT/'inference/INFERENCE_MANIFEST.json'))
        assert [g['revision'] for g in gates]==[0,1,2] and all(g['status']=='UNQUALIFIED' for g in gates)
        assert frozen['graph_gate']=={k:v for k,v in gates[0].items() if k!='rows'}
        assert not list((raw/'inference/out').glob('dev_r*')),'UNAUTHORIZED_NEW_DEV_SWEEP'
        expected=read(ROOT/'inference/data/TECHNICAL_CANARY_REQUESTS.json')
        observed=read(raw/'inference/out/TECHNICAL_CANARY.json')
        assert observed['technical_status']=='COMPLETE' and observed['requests']==6 and observed['quality_threshold_used'] is False
        calls={row['call_id']:row for row in lines(raw/'inference/out/calls.jsonl')}
        parent_calls={row['call_id']:row for row in lines(gate_raw/'inference/out/calls.jsonl')}
        assert len(expected)==len(observed['measurements'])==6
        for index,(request,measurement) in enumerate(zip(expected,observed['measurements'])):
            parent=parent_calls[request['parent_call_id']]
            assert request['request']==parent['request'] and request['input_ids']==parent['input_ids']
            assert measurement['parent_call_id']==request['parent_call_id']
            call=calls[measurement['generation']['call_id']]
            assert call['request']==request['request'] and call['input_ids']==request['input_ids']
            assert call['nonce']=='GMELLO_R0_MAIN_V3_FIXED_TECHNICAL_CANARY' and call['request_index']==index
            assert measurement['generation']==dict(call['result'],call_id=call['call_id'])
        return {'status':'PASS','recomputed_revisions':3,'parent_graph_gate_status':'UNQUALIFIED','selected_revision':0,
                'new_full_dev_rounds':0,'fixed_technical_canaries':6,'main_authorization_sha256':authorization['authorization_sha256'],
                'selection_after_development_inspection':True,'fallback_counted_toward_gate':False}
    if terminal.get('main_logical_reads')==34710:
        frozen=read(raw/'SCIENTIFIC_FREEZE.json'); passed=[g['revision'] for g in gates if g['status']=='PASS']
        assert passed and frozen['selected_revision']==min(passed)
        assert [g['revision'] for g in gates]==list(range(frozen['selected_revision']+1))
    if terminal.get('scientific_status')=='UNQUALIFIED_GRAPH_INTERFACE':
        assert [g['revision'] for g in gates]==[0,1,2] and all(g['status']=='UNQUALIFIED' for g in gates)
    return {'status':'PASS','recomputed_revisions':len(gates),'fallback_counted_toward_gate':False}

def verify_repair_controls(folder,evidence,prior,tokenizer,freeze_epoch=None,require_controls=True):
    folder,evidence,prior=map(Path,(folder,evidence,prior))
    sys.path[:0]=[str(prior/'evaluator_data/ppr/implementation'),str(prior/'evaluator_data/ppr/vendor')]
    from ppr_requests import record_view,score_requests
    from ranker_contract import choose
    from execution import execute
    calls={r['call_id']:r for r in lines(folder/'calls.jsonl') if r['endpoint'].startswith('REPAIR_') and r['nonce'] is not None}
    assert calls
    tokens={}; checked=set(); loader_reports={}
    adapters={'REPAIR_0':'abdfc30d4ad24cedad04d99a06797b854115bccdc0853dd240e133f8dd3119c3','REPAIR_1':'b9b84813e04f39d47e55a6bb8be25f34b8921970e11cd3e707dbd357b8cc6ce1'}
    for key,call in calls.items():
        assert call['kind']=='score' and call['fingerprint']['adapter']==adapters[call['endpoint']]
        assert call['fingerprint']['base']=='a09a35458c702b33eeacc393d103063234e8bc28'
        request_hash=digest(call['request'])
        if request_hash not in tokens:tokens[request_hash]=tokenizer.apply_chat_template(call['request']['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
        assert tokens[request_hash]==call['input_ids'] and len(call['input_ids'])==call['result']['input_tokens']
        values=call['result'];assert all(math.isfinite(values[k]) for k in ('yes_logit','no_logit','margin'))
        assert values['margin']==values['yes_logit']-values['no_logit']
    def check_call(key,request,nonce,index):
        call=calls[key];assert call['request']==request and call['nonce']==nonce
        assert key==digest({'fingerprint':call['fingerprint'],'request':request,'input_ids':call['input_ids'],'kind':'score','nonce':[nonce,index]})
        checked.add(key);return dict(call['result'],call_id=key)
    for endpoint in adapters:
        canary=read(prior/f'original_adapters/seed{endpoint[-1]}/LOADER_CANARY.json')
        found=[]
        for key,call in calls.items():
            nonce=call['nonce'];name=nonce[0] if isinstance(nonce,list) else nonce
            if call['endpoint']==endpoint and name=='original_loader_check':found.append((key,call))
        assert len(found)==len(canary['requests'])
        errors=[]
        for index,(request,expected) in enumerate(zip(canary['requests'],canary['scores'])):
            matches=[(key,call) for key,call in found if call['request']==request and key==digest({'fingerprint':call['fingerprint'],'request':request,'input_ids':call['input_ids'],'kind':'score','nonce':[call['nonce'],index]})]
            assert len(matches)==1
            key,call=matches[0];actual=check_call(key,request,call['nonce'],index)
            errors.extend(abs(actual[k]-expected[k]) for k in ('yes_logit','no_logit','margin'))
        saved=read(evidence/f'{endpoint}_LOADER.json');assert saved['error']==max(errors) and saved['tolerance']==canary['tolerance'] and max(errors)<=canary['tolerance']
        loader_reports[endpoint]={'status':'PASS','error':max(errors),'tolerance':canary['tolerance']}
    panels=[r for r in lines(folder/'controls.jsonl') if r['control_id'].startswith(('REPAIR_0/','REPAIR_1/'))] if (folder/'controls.jsonl').exists() else []
    if not require_controls:
        assert not panels and checked==set(calls)
        return {'status':'PASS','loaders':loader_reports,'fresh_score_calls':len(calls),'control_panels':0,'controls_not_run_after_readiness_stop':True}
    cases={c['case_id']:c for c in read(prior/'evaluator_data/RUNTIME_CASES.json')}
    original={tuple(r[k] for k in ('case_id','history','phase','query_id','arm')):r for r in lines(prior/'raw/results/logical.jsonl') if r['arm'].startswith('REPAIR_')}
    selection=read(prior/'evaluator_data/CONTROL_SELECTION.json')
    expected_ids={f'{a}/repeat/{cid}' for a in adapters for cid in selection['fixed_uncached_case_ids']}|{f'{a}/snapshot/{cid}/{qid}' for a in adapters for cid,qid in selection['uncached_pairs']}
    assert len(panels)==56 and {p['control_id'] for p in panels}==expected_ids
    failures=[];logical=0
    for panel in panels:
        offset=0
        for row in panel['rows']:
            case=cases[row['case_id']];question=next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
            assert row['question']==question
            edges,payload=record_view(case,row['history'],row['phase'])
            payload={'question':question,'relation_registry':payload['relation_registry'],'current_records':payload['current_records']}
            assert row['state_sha256']==digest(edges) and row['request_payload_sha256']==digest(payload)
            candidates,requests=score_requests(case,row['history'],row['phase'],question)
            assert row['candidates']==candidates and len(requests)==len(row['call_ids'])
            values=[]
            for request,key in zip(requests,row['call_ids']):
                call=calls[key]
                if freeze_epoch is not None:assert call['started_after_scientific_freeze'] is True
                values.append(check_call(key,request,panel['control_id'],offset));offset+=1
            assert row['scores']==values
            chosen=choose(candidates,[v['margin'] for v in values])
            assert row['selected']==chosen and row['outcome']==execute(case,row['history'],row['phase'],chosen)
            assert row['program']==(None if chosen is None else {'start_id':chosen['start_id'],'path':chosen['path']})
            logical+=1
        if panel['kind']=='fixed_uncached_case':
            equal=all(row['outcome']==original[tuple(row[k] for k in ('case_id','history','phase','query_id','arm'))]['outcome'] and row['program']==original[tuple(row[k] for k in ('case_id','history','phase','query_id','arm'))]['program'] for row in panel['rows'])
            assert equal==panel['all_equal']
        else:
            rows=panel['rows'];assert len(rows)==4
            flags={name:rows[0].get(field)==rows[2].get(field)==rows[3].get(field) for name,field in [('state_return_equal','state_sha256'),('outcome_return_equal','outcome'),('program_return_equal','program')]}
            assert all(panel[k]==v for k,v in flags.items());equal=all(flags.values())
        if not equal:failures.append(panel['control_id'])
    assert checked==set(calls),'UNEXPLAINED_FRESH_REPAIR_CALL'
    return {'status':'PASS','loaders':loader_reports,'fresh_score_calls':len(calls),'control_panels':len(panels),'logical_control_reads':logical,'parity_failures':failures,'neural_logits_regenerated':False}

def verify_gmello_controls(raw):
    signature=extracted_function(ROOT/'inference/run.py','selected_signature',{})
    lookup={tuple(r[k] for k in ('case_id','history','phase','query_id','arm')):r for p in (raw/'inference/out/main/cases').glob('*.json') for r in read(p)['rows']}
    panels=[read(p) for p in sorted((raw/'inference/out/main/controls').glob('*.json'))]
    assert len(panels)==28
    failures=[]
    for panel in panels:
        checks=[]
        if panel['kind']=='fixed_uncached_case':
            for row in panel['rows']:
                old=lookup[tuple(row[k] for k in ('case_id','history','phase','query_id','arm'))]
                checks.append({'arm':row['arm'],'history':row['history'],'query_id':row['query_id'],'fields':{k:signature(row)[k]==signature(old)[k] for k in signature(row)}})
        else:
            for endpoint in ('G_CORE','G_FULL','G_TEXT'):
                current=[r for r in panel['rows'] if r['arm']==endpoint and r['slot']!='original']
                checks.append({'arm':endpoint,'fields':{k:all(signature(r)[k]==signature(current[0])[k] for r in current[1:]) for k in signature(current[0])}})
        equal=all(all(c['fields'].values()) for c in checks)
        assert panel['comparisons']==checks and panel['all_equal']==equal
        if not equal:failures.append(panel['control_id'])
    saved=read(raw/'inference/out/main/DIAGNOSTICS.json')
    assert saved['panels']==28 and saved['endpoint_panels']==84 and set(saved['failures'])==set(failures)
    return {'status':'PASS','panels':28,'endpoint_panels':84,'differences':failures}

def verify(raw,prior,out):
    raw,prior=Path(raw),Path(prior)
    full=read(raw/'COMPLETE.json').get('main_logical_reads')==34710
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'inference/tokenizer',local_files_only=True)
    report={'status':'PASS','frozen_metrics_changed':False,'development':verify_gates(raw)}
    report['repair']=verify_repair_controls(raw/'repair/original/results',raw/'repair/original/evidence',prior,tokenizer,
                      True if full else None,full)
    if full:
        parity=read(raw/'evidence/REPAIR_PARITY.json')
        assert set(parity['failures'])==set(report['repair']['parity_failures'])
        assert parity['status']==('PASS' if not parity['failures'] else 'FAILED_PARITY')
        report['gmello']=verify_gmello_controls(raw)
    write(out,report);print(json.dumps(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.raw,a.prior,a.out)
