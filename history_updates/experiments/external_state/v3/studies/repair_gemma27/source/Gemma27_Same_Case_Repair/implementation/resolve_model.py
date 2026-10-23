"""Resolve official Gemma access and an immutable SHA; never accepts license terms.
Run metadata before renting. Weights are fetched only with --weights.
The resulting MODEL_LOCK.json is an execution artifact, not a fabricated prefilled pin.
"""
from pathlib import Path
import argparse,hashlib,json,re
from gemma_interface import MODEL_ID,validate_config

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_lock(lock_path,model_dir):
    lock=json.loads(Path(lock_path).read_text());d=Path(model_dir)
    if lock['repo']!=MODEL_ID or not re.fullmatch('[0-9a-f]{40}',lock['revision']):raise ValueError('MODEL_NOT_PINNED')
    for name,h in lock['metadata_sha256'].items():
        if sha(d/name)!=h:raise ValueError('MODEL_METADATA_CHANGED:'+name)
    validate_config(json.loads((d/'config.json').read_text()))
    index=json.loads((d/'model.safetensors.index.json').read_text())
    shards=set(index['weight_map'].values())
    if any(not (d/n).is_file() for n in shards):raise ValueError('MISSING_WEIGHT_SHARDS')
    if set(lock.get('weight_files_sha256',{}))!=shards:raise ValueError('MISSING_LOCKED_WEIGHT_HASHES')
    for name in sorted(shards):
        if sha(d/name)!=lock['weight_files_sha256'][name]:raise ValueError('WEIGHT_SHARD_CHANGED:'+name)
    return lock

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--cache-dir',type=Path);p.add_argument('--weights',action='store_true');a=p.parse_args()
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    if a.out.exists():
        lock=json.loads(a.out.read_text());assert lock['repo']==MODEL_ID;revision=lock['revision']
    else:
        info=HfApi().model_info(MODEL_ID,revision='main');revision=info.sha
        if not re.fullmatch('[0-9a-f]{40}',revision or ''):raise ValueError('NO_IMMUTABLE_MODEL_REVISION')
        lock={'repo':MODEL_ID,'revision':revision,'resolution':'one official main resolution before model outcomes','metadata_sha256':{}}
    names=['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','model.safetensors.index.json']
    downloaded={}
    for name in names:
        # Gating errors propagate before rental. No mirrors, bypasses, or auto-accept.
        path=Path(hf_hub_download(MODEL_ID,name,revision=revision,cache_dir=a.cache_dir))
        downloaded[name]=path
    validate_config(json.loads(downloaded['config.json'].read_text()))
    actual={name:sha(path) for name,path in downloaded.items()}
    if lock['metadata_sha256'] and actual!=lock['metadata_sha256']:raise ValueError('PINNED_METADATA_HASH_MISMATCH')
    lock['metadata_sha256']=actual
    d=downloaded['config.json'].parent
    lock['metadata_snapshot_path']=str(d)
    if a.weights:
        d=Path(snapshot_download(MODEL_ID,revision=revision,cache_dir=a.cache_dir,allow_patterns=names+['model-*.safetensors']))
        index=json.loads((d/'model.safetensors.index.json').read_text())
        lock['weight_files_sha256']={name:sha(d/name) for name in sorted(set(index['weight_map'].values()))}
        lock['weight_bytes']=sum((d/name).stat().st_size for name in lock['weight_files_sha256'])
        lock['weight_snapshot_path']=str(d)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'repo':MODEL_ID,'revision':revision,'snapshot':str(d),'weights_fetched':a.weights}))
if __name__=='__main__':main()
