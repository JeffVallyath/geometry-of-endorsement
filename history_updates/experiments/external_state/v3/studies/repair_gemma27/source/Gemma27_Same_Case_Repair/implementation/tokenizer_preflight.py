"""Metadata-only preflight before GPU rental. Uses no test answers or model weights."""
import argparse,json,hashlib
from pathlib import Path
from gemma_prepare import ROOT,INHERITED,load,gzrows,dump
from gemma_interface import encode_train,encode_prompt,label_ids,validate_config
from resolve_model import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True,type=Path);p.add_argument('--model-lock',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    from transformers import AutoTokenizer
    lock=load(a.model_lock)
    for name,h in lock['metadata_sha256'].items():
        if sha(a.model_dir/name)!=h:raise ValueError('LOCKED_METADATA_CHANGED')
    validate_config(load(a.model_dir/'config.json'))
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    y,n,end=label_ids(tok)
    train=gzrows(INHERITED/'training/training_pairs.jsonl.gz');lens=[]
    for r in train:lens.append(len(encode_train(tok,r)['input_ids']))
    bank=gzrows(ROOT/'prepared/REQUESTS.jsonl.gz');lengths={}
    for r in bank:
        ids=encode_prompt(tok,r['messages']);text=encode_prompt(tok,r['messages'],tokenize=False)
        if tok.encode(text,add_special_tokens=False)!=ids:raise ValueError('RENDER_TOKEN_DISAGREEMENT')
        if not 0<len(ids)<=6144 or len(ids)+64>8192:raise ValueError('NO_INFERENCE_TRUNCATION')
        if ids.count(tok.bos_token_id)!=1:raise ValueError('BOS_DUPLICATION')
        lengths[r['request_id']]=len(ids)
    d={'repo':lock['repo'],'revision':lock['revision'],'yes_token':y,'no_token':n,'turn_end_token':end,'eos_token':tok.eos_token_id,'training_draws':len(train),'max_train_tokens':max(lens),'max_inference_tokens':max(lengths.values()),'requests':len(bank),'loss_tokens_per_row':2,'source_question_bytes_unchanged':True,'weights_loaded':False,'neural_outputs_examined':False}
    dump(a.out/'TOKENIZER_PREFLIGHT.json',d);dump(a.out/'REQUEST_LENGTHS.json',lengths)
    selected=sorted(range(len(train)),key=lambda i:(-lens[i],train[i]['sample_id']))[:8]
    dump(a.out/'LONGEST_TRAINING_SAMPLE_IDS.json',[train[i]['sample_id'] for i in selected])
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
