"""PPR1 frozen inference integration. No evaluator, optimizer, or training path."""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'implementation'), str(ROOT/'vendor')]
from ppr_requests import answer_request, reflection_request, score_requests, record_view
from execution import execute, build_store
from codec import decode, norm
from ranker_contract import choose

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)

def utc():
    return datetime.now(timezone.utc).isoformat()

def deadline():
    end = datetime.fromisoformat(json.loads((ROOT/'CLOCK.json').read_bytes())['science_deadline_utc']).timestamp()
    if time.time() >= end:
        raise TimeoutError('FROZEN_SCIENCE_WALL_DEADLINE')
    return end

def verified_payload():
    manifest = json.loads((ROOT/'PAYLOAD_MANIFEST.json').read_bytes())
    for name, expected in manifest['files'].items():
        assert file_sha(ROOT/name) == expected, ('PAYLOAD_CHANGED', name)
    return sha(manifest)

class Journal:
    def __init__(self):
        self.out = ROOT/'out'
        self.out.mkdir(exist_ok=True)
        self.calls = self.load('calls', 'call_id')
        self.notes = self.load('notes', 'note_id')
        self.rows = self.load('logical', 'logical_id')
        self.controls = self.load('controls', 'control_id')
        self.timings = self.load('timings', 'timing_id')
        self.last_export = 0

    def load(self, name, key):
        path = self.out/f'{name}.jsonl'
        result = {}
        if path.exists():
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)  # partial/corrupt receipt is not silently discarded
                    assert row[key] not in result, ('DUPLICATE_JOURNAL_ID', name, row[key])
                    result[row[key]] = row
        return result

    def add(self, name, row, key):
        target = {'calls':self.calls, 'notes':self.notes, 'logical':self.rows,
                  'controls':self.controls, 'timings':self.timings}[name]
        assert row[key] not in target
        with (self.out/f'{name}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(canonical(row)+'\n'); handle.flush(); os.fsync(handle.fileno())
        target[row[key]] = row

    def status(self, stage, **extra):
        result = {'stage':stage, 'time_utc':utc(), 'pid':os.getpid(), 'logical_reads':len(self.rows),
                  'reflection_preparations':len(self.notes), 'actual_calls':len(self.calls),
                  'controls':len(self.controls), **extra}
        publish(self.out/'STATUS.json', result)
        print(canonical(result), flush=True)
        self.export()

    def export(self, force=False):
        if not force and time.monotonic()-self.last_export < 90:
            return
        path = ROOT/'export.pending.zip'
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for source in sorted(self.out.iterdir()):
                if source.is_file() and source.suffix != '.tmp':
                    archive.write(source, 'out/'+source.name)
            archive.writestr('EXPORT_RECEIPT.json', canonical({'time_utc':utc(),
                'files':{p.name:file_sha(p) for p in self.out.iterdir() if p.is_file() and p.suffix!='.tmp'}}))
        path.replace(ROOT/'export.zip')
        self.last_export = time.monotonic()

class Engine:
    def __init__(self, checkpoint, model_path, payload_hash):
        import numpy as np
        import torch
        import transformers
        import peft
        import flash_attn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert torch.__version__ == '2.8.0+cu128'
        assert transformers.__version__ == '4.56.2'
        assert peft.__version__ == '0.17.1'
        assert torch.cuda.device_count() == 1
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tok.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', device_map={'':'cuda:0'}, local_files_only=True)
        if checkpoint != 'BASE':
            self.model = peft.PeftModel.from_pretrained(self.model, ROOT/'adapters'/checkpoint, is_trainable=False)
        self.model.eval(); self.model.requires_grad_(False); self.model.config.use_cache = False
        assert self.model.config._attn_implementation == 'flash_attention_2'
        self.checkpoint = checkpoint
        self.yes = self.tok.encode('Yes', add_special_tokens=False)
        self.no = self.tok.encode('No', add_special_tokens=False)
        assert len(self.yes) == len(self.no) == 1
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.identity = {'checkpoint':checkpoint, 'payload_sha256':payload_hash,
            'base_revision':'a09a35458c702b33eeacc393d103063234e8bc28',
            'adapter_sha256':None if checkpoint=='BASE' else file_sha(ROOT/'adapters'/checkpoint/'adapter_model.safetensors'),
            'tokenizer_sha256':file_sha(Path(model_path)/'tokenizer.json'),
            'tokenizer_config_sha256':file_sha(Path(model_path)/'tokenizer_config.json'),
            'torch':torch.__version__, 'transformers':transformers.__version__, 'peft':peft.__version__,
            'flash_attn':flash_attn.__version__, 'gpu':torch.cuda.get_device_name(),
            'cuda':torch.version.cuda, 'dtype':'bfloat16', 'scoring':'Yes-No raw float32-converted logits',
            'batch_policy':'8, stable submitted order, left padding; actual batch dimensions in each receipt',
            'tf32':torch.backends.cuda.matmul.allow_tf32, 'seed':0, 'sampling':False,
            'input_cap':6144, 'truncation':False, 'score_use_cache':False, 'generation_use_cache':True}

    def tokens(self, request):
        ids = self.tok.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True)
        assert 0 < len(ids) <= 6144, ('INPUT_CAP_NO_TRUNCATION', len(ids))
        return ids

    def padded(self, sequences):
        n = max(map(len, sequences)); torch = self.torch
        ids = [[self.tok.pad_token_id]*(n-len(s))+s for s in sequences]
        mask = [[0]*(n-len(s))+[1]*len(s) for s in sequences]
        ids = torch.tensor(ids, device='cuda'); mask = torch.tensor(mask, device='cuda')
        pos = mask.cumsum(-1)-1; pos.masked_fill_(mask==0, 0)
        return ids, mask, pos

    def calculate(self, requests, kind, *, force_cap=False, token_override=None):
        deadline()
        sequences = token_override if token_override is not None else [self.tokens(r) for r in requests]
        assert all(0 < len(s) <= 6144 for s in sequences)
        ids, mask, positions = self.padded(sequences)
        torch = self.torch; torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode():
            if kind == 'score':
                logits = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                    use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                pair = logits[:, [self.yes[0], self.no[0]]]
                margins = pair[:, 0]-pair[:, 1]
                binary = torch.log_softmax(pair, dim=-1)
                assert torch.allclose(margins, binary[:,0]-binary[:,1], atol=1e-5, rtol=1e-5)
                assert torch.isfinite(pair).all()
                values = [{'margin':m, 'yes_logit':p[0], 'no_logit':p[1],
                           'token_ids':[self.yes[0], self.no[0]]}
                          for p,m in zip(pair.cpu().tolist(), margins.cpu().tolist())]
            else:
                cap = requests[0]['max_new_tokens']
                assert all(r['max_new_tokens']==cap for r in requests)
                options = dict(max_new_tokens=cap, do_sample=False, use_cache=True,
                               pad_token_id=self.tok.pad_token_id, temperature=None, top_p=None, top_k=None)
                if force_cap: options['min_new_tokens'] = cap
                generated = self.model.generate(input_ids=ids, attention_mask=mask, **options)
                values = []
                for sequence in generated[:, ids.shape[1]:].cpu().tolist():
                    stop = next((i for i,t in enumerate(sequence) if t in self.eos), len(sequence))
                    terminated = stop < len(sequence)
                    raw = sequence[:stop+(1 if terminated else 0)]
                    content = sequence[:stop]
                    invalid = any(t in self.tok.all_special_ids for t in content)
                    values.append({'text':self.tok.decode(content, skip_special_tokens=False),
                        'raw_text_with_special':self.tok.decode(raw, skip_special_tokens=False),
                        'output_token_ids':raw, 'n_output':len(raw), 'terminated_eos':terminated,
                        'truncated':not terminated and len(raw)>=cap, 'invalid_control_tokens':invalid})
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        for value, sequence in zip(values, sequences):
            value.update(n_prompt=len(sequence), batch_seconds=elapsed, batch_size=len(requests),
                         padded_prompt_tokens=ids.shape[1], amortized_seconds=elapsed/len(requests))
        return values, sequences

    def close(self):
        del self.model; gc.collect(); self.torch.cuda.empty_cache()

def cached_calls(engine, journal, requests, kind, nonce=None):
    keys = []; pending = {}
    for i, request in enumerate(requests):
        token_ids = engine.tokens(request)
        key = sha({'runtime':engine.identity, 'request':request, 'prompt_token_ids':token_ids,
                   'kind':kind, 'uncached_nonce':None if nonce is None else [nonce, i]})
        keys.append(key)
        if key not in journal.calls:
            pending.setdefault(key, request)
    todo = list(pending.items())
    for start in range(0, len(todo), 8):
        batch = todo[start:start+8]; began = utc()
        values, tokens = engine.calculate([req for _,req in batch], kind)
        for (key, request), value, token_ids in zip(batch, values, tokens):
            journal.add('calls', {'call_id':key, 'kind':kind, 'runtime':engine.identity,
                'request':request, 'prompt_token_ids':token_ids, 'result':value,
                'started_utc':began, 'completed_utc':utc(), 'uncached_nonce':nonce}, 'call_id')
    return [journal.calls[k] for k in keys]

def note_id(row):
    return '/'.join(str(row[k]) for k in ('case_id','history','phase'))

def logical_id(row):
    return note_id(row)+'/'+row['query_id']+'/'+row['arm']

def aliases(case):
    multimap = {}
    for entity, label in case['labels'].items():
        multimap.setdefault(norm(label), set()).add(entity)
    return {label:next(iter(ids)) for label,ids in multimap.items() if len(ids)==1}

def score_row(engine, journal, case, row, nonce=None):
    question = next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
    candidates, requests = score_requests(case, row['history'], row['phase'], question)
    calls = cached_calls(engine, journal, requests, 'score', nonce=nonce)
    margins = [c['result']['margin'] for c in calls]
    selected = choose(candidates, margins)
    outcome = execute(case, row['history'], row['phase'], selected)
    edges, payload = record_view(case, row['history'], row['phase'])
    return {**row, 'question':question, 'candidates':candidates, 'margins':margins,
            'selected':selected, 'outcome':outcome, 'call_ids':[c['call_id'] for c in calls],
            'state_sha256':sha(edges), 'request_payload_sha256':sha(payload)}

def prepare_notes(engine, journal, cases, schedule):
    for case in cases:
        cid = case['case_id']
        todo = [row for row in schedule['reflection_preparations']
                if row['case_id']==cid and note_id(row) not in journal.notes]
        if not todo: continue
        requests = [reflection_request(case, r['history'], r['phase']) for r in todo]
        calls = cached_calls(engine, journal, requests, 'note')
        for row, call in zip(todo, calls):
            journal.add('notes', {**row, 'note_id':note_id(row), 'call_id':call['call_id'],
                'text':call['result']['text'], 'truncated':call['result']['truncated'],
                'invalid_control_tokens':call['result']['invalid_control_tokens'],
                'before_any_question':True}, 'note_id')
        journal.status('REFLECTION_PREPARATIONS', case_id=cid)
    assert len(journal.notes)==1280

def run_arm(engine, journal, cases, schedule, arm, controls):
    timed = set(controls['timed_subjects'])
    for case in cases:
        cid = case['case_id']; started = time.perf_counter(); began = utc()
        rows = [r for r in schedule['logical_answer_reads'] if r['case_id']==cid and r['arm']==arm]
        todo = [r for r in rows if logical_id(r) not in journal.rows]
        if not todo: continue
        completed_before = len(rows)-len(todo); calls_before = len(journal.calls)
        if arm.startswith('STATE'):
            # Batch every unique candidate request in this complete case. Selection remains pointwise.
            all_requests = []
            for row in todo:
                question = next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
                all_requests.extend(score_requests(case,row['history'],row['phase'],question)[1])
            cached_calls(engine,journal,all_requests,'score')
            for row in todo:
                result = score_row(engine, journal, case, row)
                journal.add('logical', {**result, 'logical_id':logical_id(row)}, 'logical_id')
        else:
            requests = []
            for row in todo:
                question = next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
                note = journal.notes[note_id(row)]['text'] if arm=='REFLECT' else None
                requests.append(answer_request(case, row['history'], row['phase'], question, arm, note))
            calls = cached_calls(engine, journal, requests, 'answer')
            for row, call in zip(todo, calls):
                output = call['result']
                outcome = decode(output['text'], aliases(case), set(case['labels']),
                    truncated=output['truncated'], invalid_control_tokens=output['invalid_control_tokens'])
                edges, payload = record_view(case, row['history'], row['phase'])
                result = {**row, 'logical_id':logical_id(row), 'call_ids':[call['call_id']],
                    'outcome':outcome, 'raw_text':output['text'], 'state_sha256':sha(edges),
                    'request_payload_sha256':sha(payload),
                    'note_id':note_id(row) if arm=='REFLECT' else None}
                journal.add('logical', result, 'logical_id')
        if cid in timed:
            # Actual complete logical workload; cache savings and note amortization are explicit.
            note_calls = {v['call_id'] for v in journal.notes.values() if v['case_id']==cid} if arm=='REFLECT' else set()
            journal.add('timings', {'timing_id':f'{cid}/{arm}/{len(journal.timings)}', 'case_id':cid,
                'arm':arm, 'logical_reads':len(todo), 'completed_before_resume':completed_before,
                'whole_case_complete':len(todo)+completed_before==len(rows), 'started_utc':began,
                'wall_seconds':time.perf_counter()-started, 'new_model_calls':len(journal.calls)-calls_before,
                'reflection_preparation_amortized_seconds':sum(journal.calls[k]['result']['amortized_seconds'] for k in note_calls),
                'cache_allowed':True, 'claim':'Actual scheduled full-case workload, not uncached or equal-compute timing'}, 'timing_id')
        journal.status(arm, case_id=cid)

def snapshot_controls(engine, journal, cases, arm, selection):
    by_id = {c['case_id']:c for c in cases}
    for cid, qid in selection['uncached_pairs']:
        control_id = f'{arm}/{cid}/{qid}'
        if control_id in journal.controls: continue
        case = by_id[cid]; results = []
        for slot, history, phase in [('current_before','H0','post'), ('original','H0','pre'),
                                      ('current_after','H0','post'), ('equivalent_current','S0','post')]:
            row = {'case_id':cid, 'history':history, 'phase':phase, 'query_id':qid, 'arm':arm}
            results.append({'slot':slot, **score_row(engine, journal, case, row, nonce=f'{control_id}/{slot}')})
        journal.add('controls', {'control_id':control_id, 'arm':arm, 'case_id':cid, 'query_id':qid,
            'steps':results, 'fresh_stores':True, 'uncached':True,
            'return_selection_equal':results[0]['selected']==results[2]['selected'],
            'equivalent_selection_equal':results[0]['selected']==results[3]['selected'],
            'return_state_equal':results[0]['state_sha256']==results[2]['state_sha256'],
            'equivalent_state_equal':results[0]['state_sha256']==results[3]['state_sha256'],
            'agreement_is_not_correctness':True}, 'control_id')
        journal.status('UNCACHED_CONTROLS', arm=arm, case_id=cid, query_id=qid)

def census(engine, cases, schedule):
    unique = {'note':{}, 'answer':{}, 'score':{}}; reflect = 0; maxlen = {'note':0,'answer':0,'score':0}
    by_id = {c['case_id']:c for c in cases}
    def add(kind, req):
        tokens = engine.tokens(req); unique[kind][sha([req,tokens])] = len(tokens)
        maxlen[kind] = max(maxlen[kind], len(tokens))
    for row in schedule['reflection_preparations']:
        add('note', reflection_request(by_id[row['case_id']], row['history'], row['phase']))
    for row in schedule['logical_answer_reads']:
        case = by_id[row['case_id']]
        question = next(q['question'] for q in case['queries'] if q['query_id']==row['query_id'])
        if row['arm']=='REFLECT':
            # Unknown generated note: conservative no-reuse count and cap allowance, no protected generation.
            req = answer_request(case,row['history'],row['phase'],question,'REFLECT','')
            n = len(engine.tokens(req))+256
            assert n<=6144, ('REFLECTION_MAX_BOUND_OVER_CAP',n)
            maxlen['answer'] = max(maxlen['answer'],n); reflect += 1
        elif row['arm'] in ('HISTORY','CURRENT_TEXT','CURRENT_JSON'):
            add('answer',answer_request(case,row['history'],row['phase'],question,row['arm']))
        elif row['arm']=='STATE_BASE':
            for req in score_requests(case,row['history'],row['phase'],question)[1]: add('score',req)
    return {'unique_notes':len(unique['note']), 'unique_answer_requests_without_reflect':len(unique['answer']),
            'reflect_answers_no_cache_upper_bound':reflect,
            'unique_candidate_scores_per_checkpoint':len(unique['score']),
            'max_prompt_lengths':maxlen, 'logical_reads':48594, 'notes':1280,
            'state_control_score_upper_bound':3*16*4*64}

def preflight(model_path, cases, schedule, payload_hash, journal):
    verify = json.loads((ROOT/'MODEL_FILES.json').read_bytes())
    actual = {name:file_sha(Path(model_path)/name) for name in verify}
    assert actual==verify, 'BASE_MODEL_BYTES'
    publish(journal.out/'MODEL_VERIFICATION.json',{'status':'PASS','files':actual,'time_utc':utc()})
    for checkpoint in ('seed0','seed1'):
        engine = Engine(checkpoint, model_path, payload_hash)
        expected = json.loads((ROOT/'canaries'/f'{checkpoint}.json').read_bytes())
        values, tokens = engine.calculate(expected['requests'],'score')
        error = max(abs(a[k]-b[k]) for a,b in zip(values,expected['scores']) for k in ('yes_logit','no_logit','margin'))
        result = {'checkpoint':checkpoint,'status':'PASS' if error<=expected['tolerance'] else 'FAIL',
                  'max_absolute_error':error,'tolerance':expected['tolerance'],'actual':values,
                  'requests':expected['requests'],'prompt_token_ids':tokens,'runtime':engine.identity}
        publish(journal.out/f'{checkpoint}_LOADER_CANARY.json',result); journal.export(force=True)
        assert error<=expected['tolerance'], ('ORIGINAL_LOADER_CANARY_FAILED',checkpoint,error)
        engine.close()
    engine = Engine('BASE',model_path,payload_hash)
    counts = census(engine,cases,schedule); timings = {}
    base_req = {'messages':[{'role':'system','content':'Use only these synthetic records.'},
                           {'role':'user','content':'Initial record: Alice is employed by Example Company.'}],
                'max_new_tokens':64,'do_sample':False}
    base_tokens = engine.tokens(base_req); filler = engine.tok.encode(' record',add_special_tokens=False)[0]
    for kind,cap in [('score',None),('answer',64),('note',256)]:
        req = dict(base_req, max_new_tokens=cap or 64)
        n = counts['max_prompt_lengths'][kind]
        shaped = base_tokens[:-3]+[filler]*max(0,n-len(base_tokens))+base_tokens[-3:]
        samples=[]; benchmark_receipts=[]
        for repeat in range(2):
            values,_ = engine.calculate([req]*8,kind,force_cap=kind!='score',token_override=[shaped]*8)
            samples.append(values[0]['batch_seconds'])
            benchmark_receipts.append(values)
        timings[kind]={'batch_size':8,'input_tokens':len(shaped),'output_cap':cap,
                       'sample_seconds':samples,'seconds_per_request':max(samples)/8,
                       'synthetic_tokens_sha256':sha(shaped),'synthetic_prompt_tokens':shaped,
                       'request_shape':req,'force_cap_synthetic_only':kind!='score',
                       'actual_receipts':benchmark_receipts,'protected_data':False}
    # Conservative adapter scoring overhead, all possible uncached controls, setup/export reserve.
    seconds = ((counts['unique_answer_requests_without_reflect']+counts['reflect_answers_no_cache_upper_bound'])*timings['answer']['seconds_per_request']
        + counts['unique_notes']*timings['note']['seconds_per_request']
        + (3*counts['unique_candidate_scores_per_checkpoint']+counts['state_control_score_upper_bound'])*timings['score']['seconds_per_request']*1.5
        + 1800)
    reserved = seconds*1.25
    rental = json.loads((ROOT/'RENTAL.json').read_bytes())
    spent_seconds = time.time()-rental['allocated_epoch']
    prior_gpu_seconds = rental.get('prior_billed_gpu_hours',0)*3600
    forecast_cost = rental.get('prior_cloud_usd_upper_bound',0)+(spent_seconds+reserved)/3600*rental['all_in_hourly_upper_bound']
    remaining = deadline()-time.time()
    fits = reserved+900<=remaining and prior_gpu_seconds+spent_seconds+reserved+900<=8*3600 and forecast_cost+max(1,rental['all_in_hourly_upper_bound']/4)<=40
    result = {'status':'PASS' if fits else 'FEASIBILITY_STOP','counts':counts,'synthetic_benchmarks':timings,
              'science_setup_export_seconds_upper_bound':seconds,'with_25pct_reserve_seconds':reserved,
              'wall_remaining_seconds':remaining,'forecast_cloud_usd':forecast_cost,
              'prior_billed_gpu_hours':rental.get('prior_billed_gpu_hours',0),'prior_cloud_usd_upper_bound':rental.get('prior_cloud_usd_upper_bound',0),
              'shutdown_reserve_seconds':900,'time_utc':utc(),'runtime':engine.identity,
              'protected_answers_opened':False,'sampling_not_shrunk':True}
    publish(journal.out/'FORECAST.json',result); engine.close(); journal.export(force=True)
    assert fits, 'FULL_SCOPE_FORECAST_DOES_NOT_FIT'
    journal.status('PREFLIGHT_COMPLETE'); journal.export(force=True)

def run(model_path,cases,schedule,selection,payload_hash,journal):
    forecast = json.loads((journal.out/'FORECAST.json').read_bytes())
    assert forecast['status']=='PASS'
    assert forecast['with_25pct_reserve_seconds']+900<=deadline()-time.time(), 'STALE_FORECAST'
    engine = Engine('BASE',model_path,payload_hash)
    prepare_notes(engine,journal,cases,schedule)
    for arm in ('HISTORY','REFLECT','CURRENT_TEXT','CURRENT_JSON','STATE_BASE'):
        run_arm(engine,journal,cases,schedule,arm,selection)
    snapshot_controls(engine,journal,cases,'STATE_BASE',selection); engine.close()
    for checkpoint,arm in [('seed0','STATE_0'),('seed1','STATE_1')]:
        engine = Engine(checkpoint,model_path,payload_hash)
        run_arm(engine,journal,cases,schedule,arm,selection)
        snapshot_controls(engine,journal,cases,arm,selection); engine.close()
    assert len(journal.rows)==48594 and len(journal.notes)==1280 and len(journal.controls)==48
    assert len(journal.timings)>=56
    publish(journal.out/'SCIENCE_COMPLETE.json',{'status':'COMPLETE','logical_reads':len(journal.rows),
        'reflection_preparations':len(journal.notes),'actual_calls':len(journal.calls),
        'uncached_control_pairs':len(journal.controls),'time_utc':utc(),'training_steps':0})
    journal.status('SCIENCE_COMPLETE'); journal.export(force=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['preflight','run'],required=True)
    parser.add_argument('--model',default='/workspace/models/qwen'); args=parser.parse_args()
    journal=Journal()
    try:
        payload_hash=verified_payload(); deadline()
        cases=json.loads((ROOT/'prepared/RUNTIME_CASES.json').read_bytes())
        schedule=json.loads((ROOT/'prepared/SCHEDULE.json').read_bytes())
        selection=json.loads((ROOT/'prepared/CONTROL_SELECTION.json').read_bytes())
        journal.status('STARTING_'+args.stage.upper())
        if args.stage=='preflight': preflight(args.model,cases,schedule,payload_hash,journal)
        else: run(args.model,cases,schedule,selection,payload_hash,journal)
    except BaseException as exc:
        import traceback
        publish(journal.out/f'TECHNICAL_FAILURE_{int(time.time())}.json',
                {'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                 'time_utc':utc(),'stage':args.stage,'completed_logical_reads':len(journal.rows),
                 'completed_notes':len(journal.notes),'never_rerun_completed_outputs':True})
        journal.status('TECHNICAL_STOP'); journal.export(force=True)
        raise

if __name__=='__main__': main()
