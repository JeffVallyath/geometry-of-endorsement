"""ICMH1-RIPPLE-v2 inference worker (instance side).

One plan, one journal, batch size one, a fresh KV cache per independent generation.
The worker reads only the rendered plan: no gold answers, aliases, entity ids, B/D
labels or eligibility flags are available to it. No parameter edits, no training,
no quantization, no model comparison, no judge.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

from .common import MAX_NEW_TOKENS, SYSTEM_INSTRUCTION, Journal, need, read_json, sha_file, write_json


def utc_now():
    return datetime.now(timezone.utc)


def run(plan_path, journal_path, model_dir, run_label, deadline_utc, receipt_path):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    plan = read_json(plan_path)
    plan_sha = sha_file(plan_path)
    deadline = datetime.fromisoformat(deadline_utc)
    journal = Journal(journal_path)
    done = journal.completed_units()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                device_map='cuda', attn_implementation='sdpa')
    model.eval()
    load_seconds = time.monotonic() - load_started

    eos = model.generation_config.eos_token_id
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                                  repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                  eos_token_id=eos_ids,
                                  pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0])

    environment = {
        'event': 'worker_start', 'unit_id': 'worker_start', 'run_label': run_label, 'phase': plan['phase'],
        'at': utc_now().isoformat(), 'plan_sha256': plan_sha, 'plan_units': plan['unit_count'],
        'already_journaled': len(done), 'model_dir': str(model_dir),
        'model_receipt_sha256': sha_file(Path(model_dir).parent / 'MODEL_RECEIPT.json'),
        'system_instruction': SYSTEM_INSTRUCTION, 'load_seconds': load_seconds,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'transformers': transformers.__version__,
                    'gpu': torch.cuda.get_device_name(0), 'dtype': str(next(model.parameters()).dtype),
                    'arch_list': torch.cuda.get_arch_list()[:6], 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32},
        'generation': {k: v for k, v in generation.to_dict().items()
                       if k in ('do_sample', 'num_beams', 'max_new_tokens', 'repetition_penalty',
                                'temperature', 'top_p', 'top_k', 'eos_token_id', 'pad_token_id', 'use_cache')},
        'model_config': {'name_or_path': model.config._name_or_path, 'architectures': model.config.architectures,
                         'hidden_size': model.config.hidden_size, 'num_hidden_layers': model.config.num_hidden_layers,
                         'vocab_size': model.config.vocab_size, 'torch_dtype': str(model.config.torch_dtype)},
        'tokenizer': {'class': type(tokenizer).__name__, 'vocab_size': len(tokenizer),
                      'chat_template_sha256': __import__('hashlib').sha256(
                          (tokenizer.chat_template or '').encode('utf-8')).hexdigest()},
        'offline': {'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
                    'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE')},
    }
    journal.append(environment)

    stopped = None
    processed = 0
    for unit in plan['units']:
        if unit['unit_id'] in done:
            continue
        if utc_now() >= deadline:
            stopped = 'DEADLINE'
            break
        messages = [{'role': 'system', 'content': SYSTEM_INSTRUCTION},
                    {'role': 'user', 'content': unit['user_text']}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat, return_tensors='pt', add_special_tokens=False)
        input_ids = encoded['input_ids'].to('cuda')
        attention = encoded['attention_mask'].to('cuda')
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=attention,
                                    generation_config=generation, use_cache=True)
        elapsed = time.monotonic() - started
        new_tokens = output[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        ended_eos = any(token in eos_ids for token in new_tokens)
        journal.append({
            'event': 'generation', 'unit_id': unit['unit_id'], 'run_label': run_label, 'phase': plan['phase'],
            'root_id': unit['root_id'], 'kind': unit['kind'], 'context': unit['context'],
            'reference': unit['reference'], 'at': utc_now().isoformat(),
            'chat_sha256': __import__('hashlib').sha256(chat.encode('utf-8')).hexdigest(),
            'input_tokens': int(input_ids.shape[1]), 'output_token_ids': new_tokens,
            'output_tokens': len(new_tokens), 'text': text, 'raw_text': raw,
            'ended_eos': bool(ended_eos), 'truncated': bool(len(new_tokens) >= MAX_NEW_TOKENS and not ended_eos),
            'elapsed_seconds': elapsed,
        })
        processed += 1

    summary = {'event': 'worker_complete', 'unit_id': 'worker_complete', 'run_label': run_label,
               'phase': plan['phase'], 'at': utc_now().isoformat(), 'generated_now': processed,
               'stopped': stopped, 'load_seconds': load_seconds,
               'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
               'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
               'device_total_bytes': int(torch.cuda.get_device_properties(0).total_memory)}
    journal.append(summary)
    result = {'phase': plan['phase'], 'run_label': run_label, 'planned': plan['unit_count'],
              'journaled': len(journal.completed_units()), 'generated_now': processed, 'stopped': stopped,
              'status': 'COMPLETE' if len(journal.completed_units()) == plan['unit_count'] else 'INCOMPLETE',
              'load_seconds': load_seconds, 'journal_sha256': sha_file(journal_path),
              'peak_allocated_bytes': summary['peak_allocated_bytes']}
    write_json(receipt_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--run-label', required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expect-plan-sha256', required=True)
    arguments = parser.parse_args(argv)
    need(sha_file(arguments.plan) == arguments.expect_plan_sha256, 'Plan differs from the bound hash')
    return run(arguments.plan, arguments.journal, arguments.model_dir, arguments.run_label,
               arguments.deadline_utc, arguments.receipt)


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
