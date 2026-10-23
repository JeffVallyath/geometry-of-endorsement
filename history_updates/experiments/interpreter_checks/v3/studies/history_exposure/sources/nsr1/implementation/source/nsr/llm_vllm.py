"""vLLM backend: greedy decoding, pinned local checkpoint, explicit model-specific stop tokens, raw token logging.

Prompts are tokenized here with the checkpoint's own chat template; outputs are decoded here from the raw
generated token ids (stop token removed, no special-token skipping), so the path raw token -> decoded payload ->
parsed value is explicit and auditable.
"""
from __future__ import annotations

import json
import time

from .pipeline import Gen

STOP = {'qwen': [151645, 151643], 'gemma': [107, 1]}  # <|im_end|>, <|endoftext|> ; <end_of_turn>, <eos>


class VLLMGen(Gen):
    def __init__(self, model_path, model_id, max_model_len=16384, gpu_mem=0.90):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.model_id, self.model_path = model_id, model_path
        self.stop = STOP[model_id]
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(model=model_path, dtype='bfloat16', max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=True, seed=0, generation_config='vllm')
        self.max_model_len = max_model_len

    def encode(self, messages):
        if self.model_id == 'gemma':  # Gemma-2 has no system role: the system text opens the first user turn
            msgs, sysm = [], ''
            for m in messages:
                if m['role'] == 'system':
                    sysm = m['content'] + '\n\n'
                elif m['role'] == 'user' and sysm:
                    msgs.append(dict(role='user', content=sysm + m['content']))
                    sysm = ''
                else:
                    msgs.append(m)
            messages = msgs
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def run(self, reqs):
        from vllm import SamplingParams
        out = [None] * len(reqs)
        by_cap = {}
        for i, r in enumerate(reqs):
            by_cap.setdefault(r['max_new'], []).append(i)
        for cap, idx in by_cap.items():
            sp = SamplingParams(temperature=0.0, max_tokens=cap, stop_token_ids=self.stop, skip_special_tokens=False,
                                detokenize=False)
            ids = [self.encode(reqs[i]['messages']) for i in idx]
            too_long = [i for i, x in zip(idx, ids) if len(x) + cap > self.max_model_len]
            if too_long:
                raise RuntimeError(f'required input exceeds context (no truncation allowed): {len(too_long)} prompts')
            t0 = time.perf_counter()
            res = self.llm.generate([dict(prompt_token_ids=x) for x in ids], sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            for i, x, o in zip(idx, ids, res):
                c = o.outputs[0]
                toks = list(c.token_ids)
                stopped = c.finish_reason == 'stop'
                body = toks[:-1] if toks and toks[-1] in self.stop else toks
                out[i] = dict(text=self.tok.decode(body, skip_special_tokens=False), token_ids=toks,
                              finish='stop' if stopped else str(c.finish_reason), stop_reason=c.stop_reason,
                              n_prompt=len(x), n_out=len(toks), batch_seconds=round(dt, 4), batch_size=len(idx))
        return out


def canary(model_path, model_id, out_path, backend='vllm'):
    """Real-model stop-token and format canaries (no efficacy; development inputs only)."""
    from . import pipeline as P, prompts
    if backend == 'hf':
        from .llm_hf import HFGen
        g = HFGen(model_path, model_id)
    else:
        g = VLLMGen(model_path, model_id, max_model_len=8192)
    q = prompts.EXAMPLES[2][0]
    cases = [
        ('valid_json', P.msgs(prompts.compiler_system(), prompts.compiler_user(q)), 192),
        ('code_fence', P.msgs('Output the JSON object {"op":"UNSUPPORTED","reason":"x"} inside a ```json code fence and nothing else.', 'Go.'), 64),
        ('terminal_punctuation', P.msgs('Answer with the single word Yes followed by a period.', 'Is water wet?'), 16),
        ('extra_prose', P.msgs('Explain in two sentences why the sky is blue, then output {"op":"UNSUPPORTED","reason":"x"}.', 'Go.'), 96),
        ('truncation', P.msgs('Count from 1 to 200, one number per line.', 'Go.'), 24),
        ('invalid_ast', P.msgs('Output exactly this JSON object and nothing else: {"op":"EXEC","code":"1+1"}', 'Go.'), 48),
        ('native_label', P.msgs(prompts.native_system(), prompts.native_user(prompts.EX_RECORDS, prompts.EX_UPDATES, q)), 32),
        ('record', P.msgs(prompts.record_system(), prompts.record_user(prompts.EX_RECORDS[3])), 512),
    ]
    outs = g.run([dict(messages=m, max_new=cap) for _, m, cap in cases])
    rows = []
    for (name, m, cap), o in zip(cases, outs):
        row = dict(case=name, cap=cap, finish=o['finish'], stop_reason=o['stop_reason'], n_out=o['n_out'],
                   last_token=o['token_ids'][-1] if o['token_ids'] else None, text=o['text'],
                   last_is_declared_stop=bool(o['token_ids']) and o['token_ids'][-1] in g.stop,
                   stop_reason_is_declared=o['stop_reason'] in g.stop or o['stop_reason'] is None)
        try:
            obj = P.payload(o['text'])
            row['payload'] = 'JSON_OBJECT'
            if name in ('valid_json', 'invalid_ast', 'code_fence', 'extra_prose'):
                try:
                    row['validated'] = P.validate_query(obj, q)[0]
                except P.NSRError as exc:
                    row['validated'] = 'REJECTED_' + exc.code
        except P.NSRError as exc:
            row['payload'] = 'REJECTED_' + exc.code
        if name in ('terminal_punctuation', 'native_label', 'truncation'):
            row['native_rule'] = P.native_answer(o, ('Yes', 'No'))
        rows.append(row)
    rec = dict(model_id=model_id, model_path=model_path, backend=backend, declared_stop_token_ids=g.stop,
               stop_token_strings=[g.tok.convert_ids_to_tokens(t) for t in g.stop],
               generation_config_eos=json.load(open(model_path + '/generation_config.json')).get('eos_token_id'),
               chat_template_tail=g.tok.decode(g.encode([dict(role='user', content='x')])[-6:]), cases=rows)
    json.dump(rec, open(out_path, 'w'), indent=1)
    for r in rows:
        print(r['case'], r['finish'], r['stop_reason'], r['last_token'], r.get('payload'), r.get('validated'), repr(r['text'][:70]))
    return g, rec


if __name__ == '__main__':
    import sys
    canary(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else 'vllm')
