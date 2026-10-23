"""Pinned BF16 neural backend, response-only LoRA fits, complete-request journaling."""
from pathlib import Path
import datetime as dt
import gc
import gzip
import hashlib
import json
import os
import random
import time
import zipfile
import numpy as np
import torch
import transformers
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer,get_cosine_schedule_with_warmup
from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from baseline_core import canon,digest,direct_request

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'CONFIG.json').read_bytes())
CONTRACT=json.loads((ROOT/'contract.json').read_bytes())
MODEL='/workspace/models/qwen'
def read(p):return json.loads(Path(p).read_bytes())
def publish(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha_file(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check_time():
    end=dt.datetime.fromisoformat(CONTRACT['clock']['science_cutoff_utc'].replace('Z','+00:00')).timestamp()
    if time.time()>=end:raise TimeoutError('SCIENCE_CUTOFF')
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def tokenizer():
    assert transformers.__version__==CFG['transformers']
    return AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def base_model():
    assert torch.__version__==CFG['torch'] and torch.cuda.device_count()==1
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
              attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True)
    assert model.config._attn_implementation=='flash_attention_2'
    return model
def base_digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'lora_' in n:continue
        h.update(n.replace('base_model.model.','').replace('.base_layer.','.').encode())
        h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class Journal:
    def __init__(self):
        self.folder=ROOT/'results';self.folder.mkdir(exist_ok=True)
        self.index={}
        for kind,key in [('calls','call_id'),('logical','logical_id'),('controls','control_id'),('timings','timing_id'),('method','method_id')]:
            path=self.folder/(kind+'.jsonl');rows={}
            if path.exists():
                with path.open(encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line);assert row[key] not in rows;rows[row[key]]=row
            self.index[kind]=rows
    def add(self,kind,row,key):
        identifier=row[key]
        if identifier in self.index[kind]:
            assert self.index[kind][identifier]==row;return
        # fsync per logical/call item; partial writes are technical, never silently discarded.
        with (self.folder/(kind+'.jsonl')).open('a',encoding='utf-8') as f:
            f.write(canon(row)+'\n');f.flush();os.fsync(f.fileno())
        self.index[kind][identifier]=row
    def status(self,stage,**fields):
        report={'stage':stage,'epoch':time.time(),'counts':{k:len(v) for k,v in self.index.items()},**fields}
        publish(ROOT/'WORKER_STATUS.json',report);print(json.dumps(report),flush=True)
    def export(self):
        selected=[]
        for folder in ['results','adapters','training_records','development','evidence']:
            selected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and not p.name.endswith('.pending'))
        # Incomplete fits can resume from the exact adapter/optimizer/RNG checkpoint on transport recovery.
        selected.extend(p for p in (ROOT/'recovery').glob('*.pt') if not (ROOT/'training_records'/p.stem/'COMPLETE.json').exists())
        for name in ['CONFIG.json','WORKER_STATUS.json','SETUP_COMPLETE.json','GPU_PREFLIGHT.json','SCIENTIFIC_FREEZE.json','COMPLETE.json']:
            if (ROOT/name).exists():selected.append(ROOT/name)
        tmp=ROOT/'export.pending.zip'
        manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha_file(p) for p in selected}
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in selected:z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
            z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
        os.replace(tmp,ROOT/'export.zip')

class Engine:
    def __init__(self,endpoint,journal):
        check_time();seed_all(0);self.endpoint=endpoint;self.journal=journal;self.tok=tokenizer();self.model=base_model()
        adapter=None
        if endpoint.startswith('REPAIR_'):adapter=ROOT/'original_adapters'/('seed'+endpoint[-1])
        elif endpoint.startswith(('QA_','PROGRAM_')):adapter=ROOT/'adapters'/endpoint
        if adapter:
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False)
        self.model.eval();self.model.config.use_cache=True
        self.fingerprint={'base':CFG['revision'],'adapter':sha_file(adapter/'adapter_model.safetensors') if adapter else None,
            'tokenizer':sha_file(ROOT/'tokenizer/tokenizer.json'),'config':digest(CFG),'code':read(ROOT/'CODE_MANIFEST.json'),
            'numerics':'BF16-FA2-transformers-4.56.2','retriever':[CFG['retriever_id'],CFG['retriever_revision']]}
        self.yes=self.tok.encode('Yes',add_special_tokens=False);self.no=self.tok.encode('No',add_special_tokens=False)
        assert len(self.yes)==len(self.no)==1
    def tokens(self,r):
        seq=self.tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
        assert len(seq)+r.get('max_new_tokens',0)<=CFG['max_sequence_length'],'NO_SEMANTIC_TRUNCATION'
        return seq
    def pad(self,seqs):
        n=max(map(len,seqs));ids=[[self.tok.pad_token_id]*(n-len(s))+s for s in seqs]
        mask=[[0]*(n-len(s))+[1]*len(s) for s in seqs]
        ids=torch.tensor(ids,device='cuda');mask=torch.tensor(mask,device='cuda')
        pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
        return ids,mask,pos
    def calls(self,requests,kind='generate',nonce=None):
        sequences=[self.tokens(r) for r in requests]
        keys=[digest({'fingerprint':self.fingerprint,'request':r,'input_ids':ids,'kind':kind,'nonce':None if nonce is None else [nonce,i]}) for i,(r,ids) in enumerate(zip(requests,sequences))]
        unique={}
        for key,req,seq in zip(keys,requests,sequences):
            if key not in self.journal.index['calls']:unique[key]=(req,seq)
        groups={}
        for key,(req,seq) in unique.items():groups.setdefault((req.get('max_new_tokens',0),tuple(req.get('stop_strings',[]))),[]).append(key)
        for group in groups.values():
            batch=CFG['scoring_batch'] if kind=='score' else CFG['generation_batch']
            for begin in range(0,len(group),batch):
                check_time();chunk=group[begin:begin+batch];items=[unique[k] for k in chunk]
                ids,mask,pos=self.pad([x[1] for x in items]);began=time.perf_counter();began_epoch=time.time()
                with torch.inference_mode():
                    if kind=='score':
                        logits=self.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                        pair=logits[:,[self.yes[0],self.no[0]]]
                        assert torch.isfinite(pair).all()
                        values=pair.cpu().tolist()
                        outputs=[{'yes_logit':v[0],'no_logit':v[1],'margin':v[0]-v[1]} for v in values]
                    else:
                        req=items[0][0];cap=req['max_new_tokens'];stops=req['stop_strings']
                        kw={'max_new_tokens':cap,'do_sample':False,'pad_token_id':self.tok.pad_token_id,'eos_token_id':self.tok.eos_token_id,'use_cache':True}
                        if stops:kw.update(stop_strings=stops,tokenizer=self.tok)
                        generated=self.model.generate(input_ids=ids,attention_mask=mask,**kw)[:,ids.shape[1]:].cpu().tolist()
                        outputs=[]
                        for raw in generated:
                            if self.tok.eos_token_id in raw:raw=raw[:raw.index(self.tok.eos_token_id)+1]
                            else:
                                while raw and raw[-1]==self.tok.pad_token_id:raw.pop()
                            decoded=self.tok.decode(raw,skip_special_tokens=True)
                            found=[(decoded.index(s),s) for s in stops if s in decoded]
                            text=decoded[:min(found)[0]] if found else decoded
                            outputs.append({'text':text,'raw_decoded':decoded,'output_ids':raw,'output_tokens':len(raw),
                                'invalid_control_tokens':any(t in self.tok.all_special_ids for t in (raw[:-1] if raw and raw[-1]==self.tok.eos_token_id else raw)),
                                'cap_hit':len(raw)>=cap and self.tok.eos_token_id not in raw and not found,
                                'stopped_on':min(found)[1] if found else ('eos' if self.tok.eos_token_id in raw else None)})
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                for key,(req,seq),out in zip(chunk,items,outputs):
                    out.update(input_tokens=len(seq),batch_size=len(chunk),batch_seconds=elapsed,amortized_seconds=elapsed/len(chunk))
                    self.journal.add('calls',{'call_id':key,'kind':kind,'endpoint':self.endpoint,'request':req,'input_ids':seq,
                        'fingerprint':self.fingerprint,'result':out,'nonce':nonce,'started_epoch':began_epoch,'completed_epoch':time.time()},'call_id')
        return [dict(self.journal.index['calls'][k]['result'],call_id=k) for k in keys]
    def close(self):
        del self.model;gc.collect();torch.cuda.empty_cache()

class Retriever:
    def __init__(self):
        path='/workspace/models/contriever';self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).eval().to('cpu');self.cache={};self.seconds=0.;self.input_tokens=0
        torch.set_num_threads(4)
    def embed(self,texts):
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        for i in range(0,len(missing),32):
            chunk=missing[i:i+32];inputs=self.tok(chunk,padding=True,truncation=False,return_tensors='pt')
            assert inputs['input_ids'].shape[1]<=512,'RETRIEVER_TRUNCATION_FORBIDDEN'
            began=time.perf_counter()
            with torch.inference_mode():
                out=self.model(**inputs)[0];mask=inputs['attention_mask']
                out=out.masked_fill(~mask[...,None].bool(),0.)
                emb=out.sum(dim=1)/mask.sum(dim=1)[...,None]
            self.seconds+=time.perf_counter()-began;self.input_tokens+=int(mask.sum())
            self.cache.update(zip(chunk,emb))
        return torch.stack([self.cache[t] for t in texts])
    def __call__(self,query,facts,k):
        if not facts:return []
        q=self.embed([query])[0];mem=self.embed(facts);scores=(q@mem.T).tolist()
        indices=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k]
        return [{'index':i,'score':scores[i],'fact':facts[i]} for i in indices]

def encode_train(tok,row,family):
    request=direct_request(row['current_payload'],family)
    target=row['answer_label'] if family=='QA' else json.dumps(row['correct_program'],ensure_ascii=False,separators=(',',':'))
    prompt=tok.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
    full=tok.apply_chat_template(request['messages']+[{'role':'assistant','content':target}],tokenize=True,return_dict=False)
    assert full[:len(prompt)]==prompt
    tail=full[len(prompt):];assert tok.eos_token_id in tail;tail=tail[:tail.index(tok.eos_token_id)+1]
    assert tail[-1]==tok.eos_token_id and tok.decode(tail[:-1])==target
    assert len(prompt)+len(tail)<=CFG['training_max_length'],'NO_TRAINING_TRUNCATION'
    return {'input_ids':prompt+tail,'labels':[-100]*len(prompt)+tail,'prompt_tokens':len(prompt),'response_tokens':len(tail)}

def train(family,seed,canary=False):
    check_time();seed_all(seed);tok=tokenizer();label=f'{family}_{seed}'+('_canary' if canary else '')
    folder=ROOT/'training_records'/label;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.json').exists(),'NEVER_REPEAT_COMPLETED_FIT'
    raw=[json.loads(l) for l in gzip.decompress((ROOT/'training_join.jsonl.gz').read_bytes()).splitlines()]
    raw_by_id={r['sample_id']:r for r in raw};order=read(ROOT/'sample_orders.json')[str(seed)]['sample_order']
    encoded=[encode_train(tok,raw_by_id[k],family) for k in order]
    model=base_model();model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=CFG['targets'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    assert all(('lora_' in n)==p.requires_grad for n,p in model.named_parameters())
    before_hash=base_digest(model);initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG['lr'],betas=tuple(CFG['betas']),eps=CFG['eps'],weight_decay=CFG['weight_decay'])
    scheduler=get_cosine_schedule_with_warmup(optimizer,39,768)
    target_steps=16 if canary else 768;start_step=0;history=[]
    checkpoint=ROOT/'recovery'/f'{label}.pt';checkpoint.parent.mkdir(exist_ok=True)
    identity=digest({'family':family,'seed':seed,'order':order,'data':sha_file(ROOT/'training_join.jsonl.gz'),'config':CFG})
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['identity']==identity and ck['base_hash']==before_hash
        set_peft_model_state_dict(model,ck['adapter']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        start_step=ck['step'];history=ck['history']
    publish(folder/'ORDER.json',{'family':family,'seed':seed,'sample_order':order,'identity':identity,'canary':canary})
    began=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
    for step in range(start_step,target_steps):
        check_time();optimizer.zero_grad(set_to_none=True);step_began=time.perf_counter();loss_sum=0.
        effective=encoded[step*16:(step+1)*16];total_response=sum(r['response_tokens'] for r in effective)
        for m in range(0,16,CFG['microbatch']):
            batch=effective[m:m+CFG['microbatch']];n=max(len(r['input_ids']) for r in batch)
            ids=[r['input_ids']+[tok.pad_token_id]*(n-len(r['input_ids'])) for r in batch]
            mask=[[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in batch]
            labs=[r['labels']+[-100]*(n-len(r['labels'])) for r in batch]
            tensors={k:torch.tensor(v,device='cuda') for k,v in [('input_ids',ids),('attention_mask',mask),('labels',labs)]}
            supervised=sum(r['response_tokens'] for r in batch)
            assert int((tensors['labels']!=-100).sum())==supervised
            assert torch.all(tensors['labels'][tensors['attention_mask']==0]==-100)
            out=model(**tensors,use_cache=False);assert torch.isfinite(out.loss)
            weight=supervised/total_response;loss_sum+=float(out.loss.detach())*weight;(out.loss*weight).backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm)
        optimizer.step();scheduler.step()
        history.append({'step':step+1,'loss':loss_sum,'grad_norm':float(norm),'lr':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-step_began})
        if (step+1)%16==0 or step+1==target_steps:
            ck={'identity':identity,'base_hash':before_hash,'step':step+1,'history':history,'adapter':get_peft_model_state_dict(model),
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            tmp=checkpoint.with_suffix('.pending');torch.save(ck,tmp);os.replace(tmp,checkpoint)
            publish(folder/'steps.json',history);print(json.dumps({'stage':'TRAIN','fit':label,**history[-1]}),flush=True)
            if (step+1)%64==0 or step+1==target_steps:
                Journal().export()
    after_hash=base_digest(model);assert after_hash==before_hash
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(initial[n],p.detach().cpu())]
    assert changed
    dest=ROOT/'adapters'/label;model.save_pretrained(dest,safe_serialization=True)
    # Independent save/reload logits on the same TRAIN example, no outcome-driven retry.
    model.eval();probe=torch.tensor([encoded[0]['input_ids'][:encoded[0]['prompt_tokens']]],device='cuda')
    with torch.inference_mode():expected=model(input_ids=probe,use_cache=False,logits_to_keep=1).logits[:,-1,:].float().cpu()
    peak=torch.cuda.max_memory_allocated();nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model,optimizer,scheduler,initial;gc.collect();torch.cuda.empty_cache()
    reloaded=PeftModel.from_pretrained(base_model(),dest,is_trainable=False).eval()
    with torch.inference_mode():actual=reloaded(input_ids=probe,use_cache=False,logits_to_keep=1).logits[:,-1,:].float().cpu()
    error=float((expected-actual).abs().max());assert error<=1e-4,('SAVE_RELOAD',error)
    del reloaded;gc.collect();torch.cuda.empty_cache()
    report={'status':'PASS','fit':label,'canary':canary,'steps':target_steps,'identity':identity,'base_hash_before':before_hash,'base_hash_after':after_hash,
            'base_frozen':True,'changed_lora_tensors':len(changed),'trainable_parameters':nparams,'save_reload_max_logit_error':error,
            'input_tokens':sum(r['prompt_tokens'] for r in encoded[:target_steps*16]),'supervised_tokens':sum(r['response_tokens'] for r in encoded[:target_steps*16]),
            'presentations':target_steps*16,'unique_questions':len({raw_by_id[k]['question_id'] for k in order[:target_steps*16]}),
            'unique_source_cases':len({raw_by_id[k]['source_root'] for k in order[:target_steps*16]}),'peak_gpu_bytes':peak,
            'wall_seconds':time.perf_counter()-began,'training_step_seconds':sum(r['seconds'] for r in history),
            'adapter_sha256':sha_file(dest/'adapter_model.safetensors'),'history':history,'final_checkpoint_only':True,
            'reset_contract':'Each subsequent fit independently reseeds all RNGs and loads fresh pinned base and fresh optimizer; canary adapter is never used as initializer.'}
    publish(folder/'COMPLETE.json',report);seed_all(seed)
    return report
