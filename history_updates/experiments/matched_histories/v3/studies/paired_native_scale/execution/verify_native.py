"""Token-level CPU verification against the frozen runtime and official tokenizer."""
import argparse
import json
from pathlib import Path

from native_worker import digest,json_hash,load_requests,request_key,token_boundaries,save


def verify(directory,requests,metadata,freeze=None):
    from transformers import AutoTokenizer,GenerationConfig
    canary='canary' in directory.name
    expected=load_requests(requests,canary)
    expected={request_key(r):r for r in expected}
    tokenizer=AutoTokenizer.from_pretrained(metadata,local_files_only=True)
    official=GenerationConfig.from_pretrained(metadata,local_files_only=True)
    eos=official.eos_token_id
    eos=list(eos) if isinstance(eos,list) else [eos]
    vocab=tokenizer.get_vocab()
    if '<|im_end|>' in vocab:eos.append(vocab['<|im_end|>'])
    eos=sorted(set(eos))
    controls=set(tokenizer.all_special_ids)|{v for k,v in vocab.items() if k.startswith('<|') and k.endswith('|>')}
    raw_path=directory/'raw.jsonl'
    completions=list(directory.glob('COMPLETE-*.json'))
    if len(completions)!=1:raise ValueError('One complete runtime receipt required')
    completion=json.loads(completions[0].read_text(encoding='utf-8'))
    if completion['raw_sha256']!=digest(raw_path) or completion['rows']!=len(expected):
        raise ValueError('Complete runtime receipt does not match downloaded raw bytes')
    environment_rows=[json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('environment-*.json')]
    environments={row['source_environment_sha256']:row for row in environment_rows}
    if len(environments)!=len(environment_rows):raise ValueError('Duplicate source environment identity')
    seen=set();input_count=output_count=0
    frozen=json.loads(freeze.read_text(encoding='utf-8')) if freeze else None
    for row in map(json.loads,raw_path.read_text(encoding='utf-8').splitlines()):
        key=request_key(row)
        if key in seen or key not in expected:raise ValueError('Extra/duplicate output')
        request=expected[key]
        if row['request_sha256']!=json_hash(request):raise ValueError('Wrong request hash')
        ids=tokenizer.apply_chat_template(request['messages'],tokenize=True,add_generation_prompt=True)
        rendered=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True)
        if ids!=row['input_token_ids'] or rendered!=row['rendered_prompt']:
            raise ValueError('Model input differs from the exact official template')
        if tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=row['decoded_model_visible_prompt']:
            raise ValueError('Decoded prompt mismatch')
        output=row['output_token_ids']
        boundary=token_boundaries(output,set(eos),controls)
        answer_ids=boundary.pop('answer_token_ids')
        if any(row[k]!=v for k,v in boundary.items()) or row['eos_token_ids']!=eos:
            raise ValueError('Stopping/control-token audit mismatch')
        if tokenizer.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=row['raw_decoded_output']:
            raise ValueError('Raw output decode mismatch')
        if tokenizer.decode(answer_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=row['text']:
            raise ValueError('Answer decode is not terminal-token-only removal')
        env=environments[row['environment_sha256']]
        if row['template_sha256']!=env['template_sha256'] or row['run_binding']!=completion['run_binding']:
            raise ValueError('Environment/template/run binding mismatch')
        if frozen and not row['started_after_scientific_freeze']:
            raise ValueError('Protected output predates scientific freeze')
        seen.add(key);input_count+=len(ids);output_count+=len(output)
    if seen!=set(expected):raise ValueError('Missing request outputs')
    return dict(status='PASS',rows=len(seen),input_tokens=input_count,output_tokens=output_count,
                exact_template=True,exact_raw_and_answer_decode=True,exact_eos_and_truncation=True,
                raw_sha256=digest(raw_path),complete_runtime_receipt_verified=True,
                freeze_precedes_outputs=bool(frozen) if not canary else 'independent pre-freeze canaries')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('directory','requests','metadata','out'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--freeze',type=Path)
    args=parser.parse_args()
    result=verify(args.directory,args.requests,args.metadata,args.freeze)
    save(args.out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
