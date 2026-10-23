"""Protocol-fixed BF16/FA2 pointwise model, response-only training and score backend."""
import datetime as dt, gc, gzip, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qsa_common import ROOT, CFG, check_time, publish, run_ranker

MODEL='/workspace/models/qwen'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def tokenizer():
    assert transformers.__version__=='4.56.2'
    return AutoTokenizer.from_pretrained(MODEL,local_files_only=True)

def base_model():
    assert torch.__version__=='2.8.0+cu128'
    assert torch.cuda.device_count()==1
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
             attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True)
    model.config.use_cache=False
    assert model.config._attn_implementation=='flash_attention_2'
    return model

def base_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if 'lora_' in name: continue
        h.update(name.replace('base_model.model.','').replace('.base_layer.','.').encode())
        h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def encode_train(tok,row):
    prompt=tok.apply_chat_template(row['request']['messages'],tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template(row['request']['messages']+[{'role':'assistant','content':row['target']}],tokenize=True)
    assert full[:len(prompt)]==prompt
    tail=full[len(prompt):]; tail=tail[:tail.index(tok.eos_token_id)+1]
    assert len(tail)==2 and tok.decode(tail[:1])==row['target'] and tail[-1]==tok.eos_token_id
    ids=prompt+tail
    assert len(ids)<=3072
    return {'input_ids':ids,'labels':[-100]*len(prompt)+tail,'pair_id':row['pair_id']}

def collate(rows,tok):
    n=max(len(r['input_ids']) for r in rows)
    ids=[]; mask=[]; labels=[]
    for r in rows:
        k=n-len(r['input_ids'])
        ids.append(r['input_ids']+[tok.pad_token_id]*k)
        mask.append([1]*len(r['input_ids'])+[0]*k)
        labels.append(r['labels']+[-100]*k)
    result={k:torch.tensor(v,device='cuda') for k,v in [('input_ids',ids),('attention_mask',mask),('labels',labels)]}
    assert int((result['labels']!=-100).sum())==2*len(rows)
    assert torch.all(result['labels'][result['attention_mask']==0]==-100)
    return result

class Scorer:
    def __init__(self,model,tok,batch_size=8):
        self.model,self.tok,self.batch_size=model,tok,batch_size
        self.yes=tok.encode('Yes',add_special_tokens=False); self.no=tok.encode('No',add_special_tokens=False)
        assert len(self.yes)==len(self.no)==1
    def run(self,requests,namespace='canary'):
        self.model.eval(); out=[]
        for begin in range(0,len(requests),self.batch_size):
            check_time(); reqs=requests[begin:begin+self.batch_size]
            ids=[self.tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True) for r in reqs]
            assert all(len(x)<=3072 for x in ids)
            n=max(map(len,ids)); padded=[[self.tok.pad_token_id]*(n-len(x))+x for x in ids]
            mask=[[0]*(n-len(x))+[1]*len(x) for x in ids]
            input_ids=torch.tensor(padded,device='cuda'); attention=torch.tensor(mask,device='cuda')
            positions=attention.cumsum(-1)-1; positions.masked_fill_(attention==0,0)
            t=time.perf_counter()
            with torch.inference_mode():
                logits=self.model(input_ids=input_ids,attention_mask=attention,position_ids=positions,
                                  use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                pair=logits[:,[self.yes[0],self.no[0]]]
                margins=pair[:,0]-pair[:,1]
                binary=torch.log_softmax(pair,dim=-1)
                assert torch.allclose(margins,binary[:,0]-binary[:,1],atol=1e-5,rtol=1e-5)
                assert torch.isfinite(pair).all()
                values=pair.cpu().tolist(); ms=margins.cpu().tolist()
            elapsed=time.perf_counter()-t
            for j,(req,seq,v,m) in enumerate(zip(reqs,ids,values,ms)):
                out.append({'slot':f'{namespace}/{begin+j}','margin':m,'yes_logit':v[0],'no_logit':v[1],
                            'token_ids':[self.yes[0],self.no[0]],'n_prompt':len(seq),'batch_seconds':elapsed,
                            'batch_size':len(reqs),'backend':'transformers-4.56.2-BF16-FA2','cache_reuse':False})
        return out

def training(seed,canary=False):
    check_time(); seed_all(seed); tok=tokenizer()
    with gzip.open(ROOT/'training/training_pairs.jsonl.gz','rt',encoding='utf-8') as f: raw=[json.loads(l) for l in f]
    encoded=[encode_train(tok,r) for r in raw]
    assert len(encoded)==12288
    model=base_model()
    config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                      bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    assert all(('lora_' in n)==p.requires_grad for n,p in model.named_parameters())
    frozen_hash=base_digest(model)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=math.ceil(.05*768),num_training_steps=768)
    permutation=torch.randperm(12288,generator=torch.Generator().manual_seed(seed)).tolist()
    sample_order=[raw[i]['sample_id'] for i in permutation]
    steps=16 if canary else 768
    label='canary' if canary else f'seed{seed}'
    folder=ROOT/'training_records'/label; folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.json').exists(), 'NEVER_REPEAT_COMPLETED_TRAINING'
    publish(folder/'ORDER.json',{'seed':seed,'sample_order':sample_order,'initialization':'fresh pinned base',
             'steps':steps,'corpus_draws':12288,'unique_pairs':len({r['pair_id'] for r in raw}),
             'response_tokens_per_example':2,'warmup_steps':math.ceil(.05*768),'base_sha256_before':frozen_hash})
    micro=2; accum=8
    recovery=ROOT/'recovery'/f'{label}.pt'; recovery.parent.mkdir(exist_ok=True)
    history=[];start_step=0
    if recovery.exists():
        ck=torch.load(recovery,map_location='cpu',weights_only=False)
        assert ck['seed']==seed and ck['base_hash']==frozen_hash and ck['sample_order']==sample_order
        set_peft_model_state_dict(model,ck['adapter'])
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        start_step=ck['step'];history=ck['history']
    t0=time.perf_counter(); model.train()
    for step in range(start_step,steps):
        check_time()
        global_ledger=ROOT/'TRAINING_CLOCK.json'
        if global_ledger.exists():
            clock=json.loads(global_ledger.read_bytes())
            if time.time()-clock['started_epoch']>180*60: raise TimeoutError('COMBINED_TRAINING_CAP')
        optimizer.zero_grad(set_to_none=True); loss_sum=0.; step_start=time.perf_counter()
        for m in range(accum):
            idx=permutation[step*16+m*micro:step*16+(m+1)*micro]
            batch=collate([encoded[i] for i in idx],tok)
            out=model(**batch,use_cache=False)
            assert torch.isfinite(out.loss)
            loss_sum+=float(out.loss.detach())/accum
            (out.loss/accum).backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm)
        optimizer.step(); scheduler.step()
        rec={'step':step+1,'loss':loss_sum,'grad_norm':float(norm),'lr':scheduler.get_last_lr()[0],
             'seconds':time.perf_counter()-step_start,'elapsed_seconds':time.perf_counter()-t0}
        history.append(rec)
        ck={'seed':seed,'base_hash':frozen_hash,'sample_order':sample_order,'step':step+1,'history':history,
            'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        checkpoint_start=time.perf_counter()
        tmp=recovery.with_suffix('.pending');torch.save(ck,tmp);os.replace(tmp,recovery)
        rec['checkpoint_seconds']=time.perf_counter()-checkpoint_start
        (folder/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
        if (step+1)%16==0: print(json.dumps(dict(rec,seed=seed,canary=canary)),flush=True)
        if not canary and step+1==384:
            run_ranker(f'POINTWISE_TRAINED_{seed}',Scorer(model,tok),'dev',tag='midpoint')
            model.train()
    assert base_digest(model)==frozen_hash, 'BASE_WEIGHTS_CHANGED'
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(initial[n],p.detach().cpu())]
    assert changed and all('lora_' in n for n in changed)
    reqs=[r['request'] for r in raw[:8]]
    before=Scorer(model,tok).run(reqs)
    output=ROOT/'adapters'/label
    model.save_pretrained(output,safe_serialization=True)
    model.eval()
    publish(folder/'LOADER_CANARY.json',{'requests':reqs,'scores':before,'tolerance':1e-4})
    if not canary:
        run_ranker(f'POINTWISE_TRAINED_{seed}',Scorer(model,tok),'dev',tag='final')
    else:
        run_ranker('CANARY',Scorer(model,tok),'dev',tag='throughput')
    publish(folder/'COMPLETE.json',{'status':'COMPLETE','seed':seed,'canary':canary,'steps':steps,
            'base_sha256_before':frozen_hash,'base_sha256_after':frozen_hash,'changed_lora_parameters':changed,
            'duration_seconds':time.perf_counter()-t0,'last_checkpoint_only':True,'microbatch':micro,'accumulation':accum,
            'loss_mask':'exactly Yes/No plus EOS; all prompt and padding -100'})

def score_arm(arm):
    check_time(); seed_all(0); tok=tokenizer(); model=base_model()
    if arm!='POINTWISE_BASE':
        seed=int(arm[-1]); adapter=ROOT/'adapters'/f'seed{seed}'
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        expected=json.loads((ROOT/'training_records'/f'seed{seed}'/'LOADER_CANARY.json').read_bytes())
        actual=Scorer(model,tok).run(expected['requests'])
        err=max(abs(a[k]-b[k]) for a,b in zip(actual,expected['scores']) for k in ['yes_logit','no_logit','margin'])
        assert err<=expected['tolerance'], ('ADAPTER_LOADER_DISAGREEMENT',err)
        publish(ROOT/'raw'/f'{arm}_LOADER_AGREEMENT.json',{'status':'PASS','max_absolute_error':err,'tolerance':expected['tolerance']})
    scorer=Scorer(model,tok)
    run_ranker(arm,scorer)
