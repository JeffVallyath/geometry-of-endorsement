"""Fresh-process portable verification of requests, tokens, states, codec and metrics."""
import argparse,hashlib,json,sys
from pathlib import Path
from evaluate import INPUT,ARMS,read,load_lines,sha,key,nid,evaluate,dump
sys.path[:0]=[str(INPUT/'implementation'),str(INPUT/'vendor')]
from ppr_requests import answer_request,reflection_request,score_requests
from codec import decode,norm
from execution import execute
from ranker_contract import choose

def replay(result,recomputed):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(result/'model_tokenizer',local_files_only=True)
    cases={c['case_id']:c for c in read(INPUT/'prepared/RUNTIME_CASES.json')}
    rows=load_lines(result/'raw/logical.jsonl');calls={r['call_id']:r for r in load_lines(result/'raw/calls.jsonl')}
    notes={r['note_id']:r for r in load_lines(result/'raw/notes.jsonl')}
    controls=load_lines(result/'raw/controls.jsonl');checked=set();references=0
    def check(callid,request,kind,nonce=None,index=0):
        nonlocal references
        call=calls[callid];references+=1
        assert call['request']==request and call['kind']==kind
        tokens=tok.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True)
        assert tokens==call['prompt_token_ids'] and len(tokens)<=6144
        expected=sha({'runtime':call['runtime'],'request':request,'prompt_token_ids':tokens,
                      'kind':kind,'uncached_nonce':None if nonce is None else [nonce,index]})
        assert expected==callid
        assert call['runtime']['dtype']=='bfloat16' and call['runtime']['transformers']=='4.56.2'
        if kind=='score':
            r=call['result'];assert abs((r['yes_logit']-r['no_logit'])-r['margin'])<1e-7
            assert r['token_ids']==[tok.encode('Yes',add_special_tokens=False)[0],tok.encode('No',add_special_tokens=False)[0]]
        else:
            ids=call['result']['output_token_ids'];textids=ids[:-1] if call['result']['terminated_eos'] else ids
            assert tok.decode(textids,skip_special_tokens=False)==call['result']['text']
            assert len(ids)<=request['max_new_tokens']
        checked.add(callid)
    for note in notes.values():
        req=reflection_request(cases[note['case_id']],note['history'],note['phase'])
        check(note['call_id'],req,'note');assert note['text']==calls[note['call_id']]['result']['text']
    def state(row,nonce=None):
        case=cases[row['case_id']];q=next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
        candidates,requests=score_requests(case,row['history'],row['phase'],q)
        assert candidates==row['candidates'] and len(requests)==len(row['call_ids'])
        for i,(cid,req) in enumerate(zip(row['call_ids'],requests)):check(cid,req,'score',nonce,i)
        margins=[calls[cid]['result']['margin'] for cid in row['call_ids']]
        assert margins==row['margins'] and choose(candidates,margins)==row['selected']
        assert execute(case,row['history'],row['phase'],row['selected'])==row['outcome']
    for row in rows:
        if row['arm'].startswith('STATE'):state(row);continue
        case=cases[row['case_id']];q=next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
        note=notes[nid(row)]['text'] if row['arm']=='REFLECT' else None
        request=answer_request(case,row['history'],row['phase'],q,row['arm'],note)
        check(row['call_ids'][0],request,'answer')
        multimap={}
        for identity,label in case['labels'].items():multimap.setdefault(norm(label),set()).add(identity)
        aliases={label:next(iter(ids)) for label,ids in multimap.items() if len(ids)==1}
        actual=calls[row['call_ids'][0]]['result']
        assert decode(actual['text'],aliases,set(case['labels']),truncated=actual['truncated'],
                      invalid_control_tokens=actual['invalid_control_tokens'])==row['outcome']
    selection=read(INPUT/'prepared/CONTROL_SELECTION.json')
    assert {(r['arm'],r['case_id'],r['query_id']) for r in controls}=={(a,c,q) for a in ['STATE_BASE','STATE_0','STATE_1'] for c,q in selection['uncached_pairs']}
    for control in controls:
        assert [s['slot'] for s in control['steps']]==['current_before','original','current_after','equivalent_current']
        for row in control['steps']:state(row,control['control_id']+'/'+row['slot'])
        all_ids=[cid for row in control['steps'] for cid in row['call_ids']]
        assert len(all_ids)==len(set(all_ids)),'Uncached control reuse'
    assert set(calls)==checked,'Unreferenced actual computation must be accounted explicitly'
    actual_summary=evaluate(result/'raw',recomputed)
    assert actual_summary==read(result/'evaluation/SUMMARY.json')
    assert read(recomputed/'ROOT_VECTORS.json')==read(result/'evaluation/ROOT_VECTORS.json')
    receipt={'status':'PASS','logical_reads':len(rows),'reflection_preparations':len(notes),
        'unique_calls_replayed':len(checked),'request_references_replayed':references,
        'independent_metric_recalculation':True,'token_journal_replay':True,'state_and_codec_replay':True,
        'protected_neural_inference':False,'original_historical_counts':{'qwen':94,'gemma':101}}
    dump(result/'REPLAY_VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--recomputed',type=Path,required=True)
    a=p.parse_args();replay(a.result,a.recomputed)
