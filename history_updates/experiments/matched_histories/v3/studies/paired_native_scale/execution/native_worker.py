"""Frozen native worker: only runtime messages/configuration; no evaluator imports.

Canaries are generated once; semantic checks are performed in a separate process.
All output journals are append-only, fsynced, and resume only exact missing IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_requests(path, canary=False):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    rows = json.loads(text) if canary else [json.loads(s) for s in text.splitlines()]
    fields = {'canary_id', 'messages'} if canary else {'case_id', 'arm', 'query_id', 'messages'}
    keys = set()
    for row in rows:
        if set(row) != fields:
            raise ValueError('Runtime schema forbids extra evaluator fields')
        if len(row['messages']) != 2 or [m['role'] for m in row['messages']] != ['system', 'user']:
            raise ValueError('Exactly the fixed system/user pair is required')
        if any(set(m) != {'role', 'content'} or not isinstance(m['content'], str) for m in row['messages']):
            raise ValueError('Invalid model-visible messages')
        key = request_key(row)
        if key in keys:
            raise ValueError('Duplicate request')
        keys.add(key)
    return rows


def request_key(row):
    if 'canary_id' in row:
        return ('canary', row['canary_id'])
    return (row['case_id'], row['arm'], row['query_id'])


def token_boundaries(ids, eos, controls, maximum=48):
    """Only one terminal EOS may be removed; never silently discard trailing text."""
    ids = list(ids)
    boundaries = [i for i, token in enumerate(ids) if token in eos]
    valid = len(ids) <= maximum
    if boundaries:
        valid = valid and boundaries == [len(ids)-1]
        answer_ids = ids[:-1] if boundaries == [len(ids)-1] else ids
    else:
        answer_ids = ids
    valid = valid and not any(token in controls for token in answer_ids)
    return dict(answer_token_ids=answer_ids,
                terminated=bool(boundaries),
                truncated=not boundaries and len(ids) >= maximum,
                control_tokens_valid=bool(valid),
                termination_positions=boundaries)


def resumed_ids(path, rows, binding):
    expected = {request_key(r): json_hash(r) for r in rows}
    done = set()
    if not Path(path).exists():
        return done
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = request_key(row)
            if key in done or key not in expected or row['request_sha256'] != expected[key]:
                raise ValueError('Duplicate, changed, or unknown resumed output')
            if row['run_binding'] != binding:
                raise ValueError('Cannot resume across scientific bindings')
            done.add(key)
    return done


def generation_policy(eos, pad, bos):
    return dict(do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=48, min_new_tokens=0, repetition_penalty=1.0,
                temperature=1.0, top_p=1.0, top_k=0, use_cache=True,
                eos_token_id=eos, pad_token_id=pad, bos_token_id=bos)


def environment(cfg):
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'accelerate', 'numpy', 'safetensors',
                 'tokenizers', 'huggingface-hub')}
    if platform.python_version_tuple()[:2] != ('3', '12'):
        raise RuntimeError('Pinned Python 3.12 required')
    for name, expected in cfg['environment'].items():
        if name in versions and versions[name].split('+')[0] != expected:
            raise RuntimeError(f'Pinned version mismatch: {name}')
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  machine=platform.machine(), packages=versions,
                  pid=os.getpid(), started_at=time.time())
    if platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('x86-64 required')
    result['nvidia_smi'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'
    ], text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()
    result['installed_wheels'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--model', choices=('small', 'large'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('canary', 'primary', 'repeat'), required=True)
    parser.add_argument('--length-requests', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--science-deadline', type=float, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    rows = load_requests(args.requests, args.mode == 'canary')
    if args.mode == 'primary' and len(rows) != 5406:
        raise ValueError('The primary requires all 5406 independent requests')
    if args.mode != 'canary' and not args.freeze:
        raise ValueError('Protected generation requires the common scientific freeze')
    freeze = json.loads(args.freeze.read_text(encoding='utf-8')) if args.freeze else None
    if freeze:
        if freeze['worker_sha256'] != digest(__file__) or freeze['config_sha256'] != digest(args.config):
            raise ValueError('Changed frozen science')
        if args.mode == 'primary' and freeze['runtime_sha256'] != digest(args.requests):
            raise ValueError('Changed frozen requests')
        if not freeze['both_canaries_passed'] or not freeze['forecast_accepted']:
            raise ValueError('Freeze is missing technical gates')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = environment(cfg)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.set_grad_enabled(False)
    torch.manual_seed(cfg['inference']['seed'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA GPU is required')
    memory = torch.cuda.get_device_properties(0).total_memory
    if memory / 2**30 < cfg['limits']['minimum_gpu_memory_gib']:
        raise RuntimeError('GPU memory below authorized limit')
    pin = cfg['models'][args.model]
    tokenizer = AutoTokenizer.from_pretrained(pin['repo'], revision=pin['revision'], trust_remote_code=False)
    official = GenerationConfig.from_pretrained(pin['repo'], revision=pin['revision'])
    vocab = tokenizer.get_vocab()
    eos = official.eos_token_id
    eos = list(eos) if isinstance(eos, list) else [eos]
    if '<|im_end|>' in vocab:
        eos.append(vocab['<|im_end|>'])
    eos = sorted(set(eos))
    if any(type(t) is not int or t not in set(vocab.values()) for t in eos):
        raise ValueError('EOS must be actual tokenizer IDs')
    controls = set(tokenizer.all_special_ids)
    controls.update(token for label, token in vocab.items() if label.startswith('<|') and label.endswith('|>'))
    template = tokenizer.get_chat_template()
    metadata_dir = args.out/'metadata'
    if not metadata_dir.exists():
        metadata_dir.mkdir()
        tokenizer.save_pretrained(metadata_dir)
        official.save_pretrained(metadata_dir/'official-generation')
    policy = generation_policy(eos, official.pad_token_id or tokenizer.pad_token_id or eos[0], official.bos_token_id)
    generation = GenerationConfig(**policy)
    prompts = {}
    length_rows = load_requests(args.length_requests) if args.length_requests else rows
    lengths = []
    token_journal = args.out/'prompt_tokens.jsonl'
    if token_journal.exists():
        previous_tokens = {request_key(r): r for r in map(json.loads, token_journal.read_text(encoding='utf-8').splitlines())}
    else:
        previous_tokens = {}
    for row in length_rows + (rows if args.length_requests else []):
        key = request_key(row)
        if key in prompts:
            continue
        rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        if ids != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError('Chat template token boundaries disagree')
        visible = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        record = dict(**{k:v for k,v in row.items() if k != 'messages'},
                      request_sha256=json_hash(row), input_token_ids=ids,
                      rendered_prompt=rendered, decoded_model_visible_prompt=visible)
        prompts[key] = record
        lengths.append(len(ids))
        if key in previous_tokens:
            if previous_tokens[key] != record:
                raise ValueError('Changed tokenization on resume')
        else:
            append(token_journal, record)
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(pin['repo'], revision=pin['revision'],
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0}, trust_remote_code=False)
    model.eval()
    torch.cuda.synchronize()
    loaded = time.time()
    if model.config._attn_implementation != 'sdpa' or model.dtype != torch.bfloat16:
        raise RuntimeError('Unexpected dtype/backend')
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('Offload prohibited')
    context_limit = model.config.max_position_embeddings
    if max(lengths)+48 >= context_limit:
        raise RuntimeError('Prompt does not safely fit the configured context')
    if not (metadata_dir/'config.json').exists():
        model.config.save_pretrained(metadata_dir)
    env.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               total_gpu_bytes=memory, allocated_bytes=torch.cuda.memory_allocated(),
               model_load_seconds=loaded-load_started, context_limit=context_limit,
               max_prompt_tokens=max(lengths), prompt_count=len(length_rows),
               dtype=str(model.dtype), attention=model.config._attn_implementation,
               no_offload=True, model_revision=pin, generation_policy=policy,
               template=template, template_sha256=hashlib.sha256(template.encode()).hexdigest())
    if args.mode == 'canary':
        # Shape-only throughput measurement: no cohort prompt or protected output.
        filler_id = tokenizer.encode(' record', add_special_tokens=False)[0]
        shape_ids = torch.full((1, max(lengths)), filler_id, dtype=torch.long, device='cuda')
        shape_mask = torch.ones_like(shape_ids)
        shape_times = []
        for _ in range(3):
            torch.cuda.synchronize()
            shape_start = time.perf_counter()
            with torch.inference_mode():
                shape_output = model(input_ids=shape_ids, attention_mask=shape_mask,
                                     use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            shape_times.append(time.perf_counter()-shape_start)
            del shape_output
        del shape_ids, shape_mask
        env['synthetic_max_length_prefill'] = dict(token_id=filler_id,
            length=max(lengths), seconds=shape_times,
            purpose='Shape-only performance; not a cohort read or semantic canary')
    env['metadata_hashes'] = {p.relative_to(metadata_dir).as_posix():digest(p) for p in metadata_dir.rglob('*') if p.is_file()}
    attempt_path = args.out/f'environment-{os.getpid()}-{time.time_ns()}.json'
    save(attempt_path, env)
    binding = json_hash(dict(worker=digest(__file__), config=digest(args.config),
                             requests=digest(args.requests), model=pin,
                             template=env['template_sha256'], policy=policy))
    raw_path = args.out/'raw.jsonl'
    done = resumed_ids(raw_path, rows, binding)
    for index, row in enumerate(rows):
        key = request_key(row)
        if key in done:
            continue
        if time.time() >= args.science_deadline:
            raise TimeoutError('No new science after seven-hour deadline')
        prompt = prompts[key]
        input_ids = torch.tensor([prompt['input_token_ids']], dtype=torch.long, device='cuda')
        mask = torch.ones_like(input_ids)
        torch.cuda.synchronize()
        request_started = time.time()
        with torch.inference_mode():
            if args.mode == 'canary':
                direct = model(input_ids=input_ids, attention_mask=mask, use_cache=False,
                               logits_to_keep=1).logits[:, -1, :].float()
                cached = model(input_ids=input_ids, attention_mask=mask, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].float()
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            output = model.generate(input_ids=input_ids, attention_mask=mask,
                generation_config=generation, return_dict_in_generate=True,
                output_logits=args.mode == 'canary')
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter()-generation_started
        ids = output.sequences[0, input_ids.shape[1]:].tolist()
        boundary = token_boundaries(ids, eos, controls)
        answer_ids = boundary.pop('answer_token_ids')
        audit = None
        if args.mode == 'canary':
            generated = output.logits[0].float()
            audit = dict(direct_top=int(direct.argmax()), cached_top=int(cached.argmax()),
                         generation_top=int(generated.argmax()),
                         forward_generation_max_abs=float((direct-generated).abs().max()),
                         uncached_cached_max_abs=float((direct-cached).abs().max()),
                         full_vocab=int(direct.numel()))
            audit['pass'] = (audit['direct_top'] == audit['cached_top'] == audit['generation_top']
                             and audit['forward_generation_max_abs'] <= .01
                             and audit['uncached_cached_max_abs'] <= .01)
        result = dict(**{k:v for k,v in row.items() if k != 'messages'},
            text=tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            raw_decoded_output=tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            output_token_ids=ids, input_token_ids=prompt['input_token_ids'],
            rendered_prompt=prompt['rendered_prompt'], decoded_model_visible_prompt=prompt['decoded_model_visible_prompt'],
            request_sha256=json_hash(row), run_binding=binding, eos_token_ids=eos,
            template_sha256=env['template_sha256'], environment_sha256=digest(attempt_path),
            started_at=request_started, finished_at=time.time(), elapsed_seconds=time.time()-request_started,
            generation_seconds=generation_seconds,
            process_id=os.getpid(), mode=args.mode, **boundary)
        if audit is not None:
            result['forward_check'] = audit
        append(raw_path, result)
        del output, input_ids, mask
        if audit is not None:
            del direct, cached, generated
        if (index+1) % 100 == 0 or index+1 == len(rows):
            print(json.dumps(dict(completed=index+1, total=len(rows), mode=args.mode,
                                  model=args.model, elapsed=time.time()-started)), flush=True)
    receipt = dict(status='COMPLETE', model=args.model, mode=args.mode,
                   rows=len(rows), run_binding=binding, raw_sha256=digest(raw_path),
                   elapsed_seconds=time.time()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   completed_at=time.time(), process_id=os.getpid())
    save(args.out/f'COMPLETE-{os.getpid()}-{time.time_ns()}.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
