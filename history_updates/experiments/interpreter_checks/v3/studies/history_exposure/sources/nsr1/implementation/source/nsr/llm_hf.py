"""Plain transformers backend (technical fallback for checkpoints the vLLM build cannot serve, e.g. Gemma-2's
attention soft-capping). Greedy decoding, eager attention, explicit model-specific stop tokens, raw token logging.
Batches are separate model rows (left-padded); no content of one prompt enters another prompt's context.
"""
from __future__ import annotations

import time

from .pipeline import Gen

STOP = {'qwen': [151645, 151643], 'gemma': [107, 1]}


class HFGen(Gen):
    def __init__(self, model_path, model_id, max_model_len=8192, token_budget=36000, max_batch=48):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id, self.model_path, self.stop = model_id, model_path, STOP[model_id]
        self.tok = AutoTokenizer.from_pretrained(model_path)
        kw = dict(attn_implementation='eager') if model_id == 'gemma' else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kw).cuda().eval()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop[-1]
        self.max_model_len, self.token_budget, self.max_batch = max_model_len, token_budget, max_batch

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
        torch = self.torch
        ids = [self.encode(r['messages']) for r in reqs]
        if any(len(x) + r['max_new'] > self.max_model_len for x, r in zip(ids, reqs)):
            raise RuntimeError('required input exceeds context (no truncation allowed)')
        order = sorted(range(len(reqs)), key=lambda i: (reqs[i]['max_new'], -len(ids[i])))
        out = [None] * len(reqs)
        i = 0
        while i < len(order):
            cap = reqs[order[i]]['max_new']
            longest = len(ids[order[i]])
            n = max(1, min(self.max_batch, self.token_budget // (longest + cap)))
            batch = [j for j in order[i:i + n] if reqs[j]['max_new'] == cap]
            i += len(batch)
            L = max(len(ids[j]) for j in batch)
            x = torch.full((len(batch), L), self.pad, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for row, j in enumerate(batch):
                x[row, L - len(ids[j]):] = torch.tensor(ids[j])
                mask[row, L - len(ids[j]):] = 1
            t0 = time.perf_counter()
            with torch.no_grad():
                y = self.model.generate(x.cuda(), attention_mask=mask.cuda(), max_new_tokens=cap, do_sample=False,
                                        eos_token_id=self.stop, pad_token_id=self.pad)
            dt = time.perf_counter() - t0
            for row, j in enumerate(batch):
                gen = y[row, L:].tolist()
                toks, stopped = [], False
                for t in gen:
                    toks.append(t)
                    if t in self.stop:
                        stopped = True
                        break
                body = toks[:-1] if stopped else toks
                out[j] = dict(text=self.tok.decode(body, skip_special_tokens=False), token_ids=toks,
                              finish='stop' if stopped else 'length', stop_reason=toks[-1] if stopped else None,
                              n_prompt=len(ids[j]), n_out=len(toks), batch_seconds=round(dt, 4), batch_size=len(batch))
        return out
