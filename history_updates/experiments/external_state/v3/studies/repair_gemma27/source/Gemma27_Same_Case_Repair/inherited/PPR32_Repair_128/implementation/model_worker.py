"""32B adapter training and isolated request scoring. GPU validation is NOT performed here.
This worker never imports the evaluator, source cohort, native responses, or PPR labels.
Use an external spending watchdog; --deadline-epoch is an additional local time limit.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, math, os, random, time

def read_json(p):return json.loads(Path(p).read_bytes())
def read_rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.pending')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def check_time(a):
    if time.time()>=a.deadline_epoch:raise TimeoutError('SCIENCE_DEADLINE')
def append(p,x):
    with Path(p).open('a',encoding='utf-8') as f:
        f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n');f.flush()
def seed_all(seed,torch,np):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def digest_base(model,torch):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'lora_' in n:continue
        name=n.replace('base_model.model.','').replace('.base_layer.','.')
        h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def load_model(a,torch):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    config=read_json(Path(a.model_dir)/'config.json')
    if config.get('hidden_size')!=5120 or config.get('num_hidden_layers')!=64:
        raise ValueError('NOT_THE_DECLARED_32B_ARCHITECTURE')
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model.config.use_cache=False
    return model,tok

def encode_train(tok,row):
    prefix=tok.apply_chat_template(row['request']['messages'],tokenize=True,add_generation_prompt=True)
    ids=tok.apply_chat_template(row['request']['messages']+[{'role':'assistant','content':row['target']}],tokenize=True)
    if ids[:len(prefix)]!=prefix:raise ValueError('TRAINING_TEMPLATE_BOUNDARY')
    suffix=ids[len(prefix):];suffix=suffix[:suffix.index(tok.eos_token_id)+1]
    if len(suffix)!=2 or tok.decode(suffix[:1])!=row['target']:raise ValueError('YES_NO_EOS_NOT_TWO_TOKENS')
    ids=prefix+suffix
    if len(ids)>3072:raise ValueError('TRAINING_INPUT_TOO_LONG_NO_TRUNCATION')
    return {'input_ids':ids,'labels':[-100]*len(prefix)+suffix}

def train(a,torch,np):
    from peft import LoraConfig,get_peft_model
    from transformers import get_cosine_schedule_with_warmup
    seed_all(a.seed,torch,np);check_time(a)
    rows=read_rows(a.training);order=read_json(a.order)['sample_order']
    byid={r['sample_id']:r for r in rows}
    if len(rows)!=12288 or set(order)!=set(byid) or len(order)!=12288:raise ValueError('FROZEN_TRAINING_INVENTORY')
    model,tok=load_model(a,torch)
    encoded=[encode_train(tok,byid[i]) for i in order]
    if a.microbatch not in (1,2):raise ValueError('MICROBATCH')
    accum=16//a.microbatch
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    if not all(('lora_' in n)==p.requires_grad for n,p in model.named_parameters()):raise ValueError('BASE_NOT_FROZEN')
    before=digest_base(model,torch)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,
        betas=(.9,.999),eps=1e-8,weight_decay=.01)
    sch=get_cosine_schedule_with_warmup(opt,num_warmup_steps=39,num_training_steps=768)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('TRAIN_OUTPUT_MUST_BE_EMPTY; no silent retraining or overwrite')
    steps=16 if a.action=='canary' else 768
    write_json(out/'PLAN.json',{'seed':a.seed,'steps':steps,'schedule_total_steps':768,'warmup_steps':39,
        'microbatch':a.microbatch,'gradient_accumulation':accum,'effective_batch':16,
        'base_sha256_before':before,'sample_order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'loss':'two response tokens Yes/No and EOS; prompt/padding ignored','attention':'sdpa'})
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        check_time(a);t=time.perf_counter();opt.zero_grad(set_to_none=True);loss_sum=0.
        for i in range(accum):
            batch=encoded[step*16+i*a.microbatch:step*16+(i+1)*a.microbatch]
            n=max(len(r['input_ids']) for r in batch)
            ids=[r['input_ids']+[tok.pad_token_id]*(n-len(r['input_ids'])) for r in batch]
            mask=[[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in batch]
            labs=[r['labels']+[-100]*(n-len(r['labels'])) for r in batch]
            feed={k:torch.tensor(v,device='cuda') for k,v in [('input_ids',ids),('attention_mask',mask),('labels',labs)]}
            if int((feed['labels']!=-100).sum())!=2*len(batch):raise ValueError('LOSS_MASK')
            loss=model(**feed,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
            loss_sum+=float(loss.detach());(loss/accum).backward()
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0))
        if not math.isfinite(norm) or norm<=0:raise ValueError('NONFINITE_OR_ZERO_ADAPTER_GRADIENT')
        if any(p.grad is not None for n,p in model.named_parameters() if 'lora_' not in n):raise ValueError('BASE_GRADIENT')
        opt.step();sch.step();torch.cuda.synchronize()
        append(out/'steps.jsonl',{'step':step+1,'mean_loss':loss_sum/accum,'gradient_norm':norm,
            'seconds':time.perf_counter()-t,'max_allocated_bytes':torch.cuda.max_memory_allocated()})
        if (step+1)%16==0:print(json.dumps({'seed':a.seed,'step':step+1,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    after=digest_base(model,torch)
    if after!=before:raise ValueError('BASE_WEIGHTS_CHANGED')
    check_time(a);model.eval();model.requires_grad_(False)
    loader_requests=[];expected=[]
    for row in rows[:8]:
        request={'kind':'score',**row['request']}
        request['request_id']=hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        loader_requests.append(request)
        ids=tok.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True)
        with torch.inference_mode():
            logits=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1).logits[0,-1].float()
        y=float(logits[tok.encode('Yes',add_special_tokens=False)[0]]);n=float(logits[tok.encode('No',add_special_tokens=False)[0]])
        expected.append({'request_id':request['request_id'],'yes_logit':y,'no_logit':n,'margin':y-n})
    with gzip.open(out/'LOADER_REQUESTS.jsonl.gz','wt',encoding='utf-8') as f:
        for r in loader_requests:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(out/'LOADER_EXPECTED.json',expected)
    model.save_pretrained(out/'adapter',safe_serialization=True)
    write_json(out/'COMPLETE.json',{'seed':a.seed,'steps':steps,'base_sha256_before':before,'base_sha256_after':after,
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'canary_only':a.action=='canary','final_checkpoint_only':True})
    # The canary adapter MUST be discarded. Real seeds restart in new processes.

def infer(a,torch,np):
    from peft import PeftModel
    seed_all(0,torch,np);check_time(a)
    model,tok=load_model(a,torch)
    if a.adapter:model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
    model.eval();model.requires_grad_(False)
    requests=read_rows(a.requests)
    if a.kind:requests=[r for r in requests if r['kind']==a.kind]
    if a.ids:
        requested=read_json(a.ids);lookup={r['request_id']:r for r in requests}
        # A repeated request ID means actual uncached recomputation, not a skipped row.
        requests=[lookup[i] for i in requested]
    yes=tok.encode('Yes',add_special_tokens=False);no=tok.encode('No',add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1:raise ValueError('BINARY_LABEL_TOKENIZATION')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('INFERENCE_OUTPUT_MUST_BE_EMPTY')
    adapter_hash=None
    if a.adapter:adapter_hash=hashlib.sha256((Path(a.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()
    write_json(out/'IDENTITY.json',{'repo':'Qwen/Qwen2.5-32B-Instruct','revision':'5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
        'adapter_sha256':adapter_hash,'attention':'sdpa','dtype':'bfloat16','batch':1,'seed':0,
        'tokenizer_sha256':hashlib.sha256((Path(a.model_dir)/'tokenizer.json').read_bytes()).hexdigest(),
        'request_inventory_sha256':hashlib.sha256(Path(a.requests).read_bytes()).hexdigest(),
        'no_inter_query_KV_cache':True,'selected_request_ids':a.ids,'pid':os.getpid()})
    for i,r in enumerate(requests):
        check_time(a);ids=tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
        if not 0<len(ids)<=6144:raise ValueError('INFERENCE_LENGTH_NO_TRUNCATION')
        inp=torch.tensor([ids],device='cuda');t=time.perf_counter()
        with torch.inference_mode():
            if r['kind']=='score':
                logits=model(input_ids=inp,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                y,n=float(logits[yes[0]]),float(logits[no[0]])
                if not math.isfinite(y+n):raise ValueError('NONFINITE_SCORE')
                v={'yes_logit':y,'no_logit':n,'margin':y-n,'yes_token':yes[0],'no_token':no[0],
                   'full_logits_sha256':hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()}
            elif r['kind']=='generate':
                eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
                gen=model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),max_new_tokens=64,
                    do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.pad_token_id,
                    temperature=None,top_p=None,top_k=None)[0,len(ids):].cpu().tolist()
                stop=next((j for j,x in enumerate(gen) if x in eos),len(gen));content=gen[:stop]
                v={'text':tok.decode(content,skip_special_tokens=False),'output_token_ids':gen,
                   'truncated':stop==len(gen) and len(gen)>=64,
                   'invalid_control_tokens':any(x in tok.all_special_ids for x in content)}
            else:raise ValueError('REQUEST_KIND')
        torch.cuda.synchronize()
        v.update(request_id=r['request_id'],occurrence=i,input_token_ids=ids,
            rendered_prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True),
            seconds=time.perf_counter()-t)
        append(out/'responses.jsonl',v)
        if (i+1)%128==0:print(json.dumps({'done':i+1,'total':len(requests)}),flush=True)
    write_json(out/'COMPLETE.json',{'requests':len(requests),'independent_full_prefix':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['canary','train','infer'])
    for name in ('model-dir','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--training');p.add_argument('--order');p.add_argument('--seed',type=int,choices=[0,1],default=0)
    p.add_argument('--microbatch',type=int,choices=[1,2],default=1)
    p.add_argument('--requests');p.add_argument('--kind',choices=['score','generate']);p.add_argument('--adapter');p.add_argument('--ids')
    a=p.parse_args()
    import torch,numpy as np,transformers,peft
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('ONE_CUDA_GPU_REQUIRED')
    if str(torch.__version__)!='2.8.0+cu128' or transformers.__version__!='4.56.2' or peft.__version__!='0.17.1':raise RuntimeError('PINNED_STACK_REQUIRED')
    if a.action in ('canary','train'):
        if not a.training or not a.order:p.error('--training and --order required')
        train(a,torch,np)
    else:
        if not a.requests:p.error('--requests required')
        infer(a,torch,np)
if __name__=='__main__':main()
