from __future__ import annotations
import time, math
import numpy as np
from . import config as Q
from . import base_adapter as V1
from .base_adapter import BoundaryError

class Backbone(V1.Backbone):
    def __init__(self, cache=None, cfg=Q.GEMMA, model=None, tokenizer=None, require_cuda=True, sites=None):
        cfg2 = dict(cfg); cfg2['late_site'] = cfg['late_site'] if cfg.get('late_site') is not None else cfg['site']
        self.template_kwargs = dict(cfg.get('template_kwargs') or {})
        super().__init__(cache=cache, cfg=cfg2, model=model, tokenizer=tokenizer, require_cuda=require_cuda, sites=(sites if sites is not None else cfg2['sites']))
        self.site = int(cfg['site']); self.next_input = {}; self.captured_full = {}

    def _reset(self):
        super()._reset(); self.next_input = {}; self.captured_full = {}; self.row_energy = None

    def render(self, prefix_text, suffix_text):
        full = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prefix_text+suffix_text}], tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        off = full.find(prefix_text)
        if off < 0 or full.find(prefix_text, off+1) >= 0: raise BoundaryError('prefix text not unique in the rendered prompt')
        return full, off

    def generation_prompt(self):
        """The exact generation-prompt text the actor's template appends after the user turn (Qwen: includes the empty think block)."""
        full, off = self.render('x', 'y'); return full[off+2:]

    def _witness(self, site):
        def cb(module, args, kwargs):
            if self.plan is None: return
            h = args[0] if args else kwargs['hidden_states']
            if site in self.plan.get('capture_next', ()) and site not in self.next_input: self.next_input[site] = h.detach()
            if site not in self.after or site in self.witnessed: return
            rows, pos, new = self.after[site]
            if int(pos.max()) >= h.shape[1]: return
            if not self.t.equal(h[rows, pos].detach(), new): raise RuntimeError(f'post-hook state not propagated to next block at site {site}')
            self.witnessed.add(site)
        return cb

    def ask_batch(self, art, suffix_batch, label_ids_batch, late=None, verify_master=False):
        """Score B suffixes against ONE clone of the compiled artifact (the clone is batch-repeated; the master is never touched).
        Returns one result dict per suffix (same fields as `ask`).  Validated against single-suffix asks in the E0 panel."""
        t = self.t; start = time.monotonic(); P = art['prefix_len']; B = len(suffix_batch)
        before = self.cache_hash(art['cache']) if verify_master else None
        cache = self.clone_cache(art['cache'])
        if art.get('batch', 1) == 1 and B > 1: cache.batch_repeat_interleave(B)
        elif art.get('batch', 1) not in (1, B): raise ValueError('artifact batch does not match the suffix batch')
        S = max(len(s) for s in suffix_batch); ids = t.full((B, S), self.pad_id, dtype=t.long, device=self.device); mask = t.ones(B, P+S, dtype=t.long, device=self.device); last = []
        for r, s in enumerate(suffix_batch): ids[r, :len(s)] = t.tensor(s, device=self.device); mask[r, P+len(s):] = 0; last.append(len(s)-1)
        edit = None
        if late is not None:
            maps = [m if m.get('z') is not None or m.get('mode') in ('replacement', 'delta') else dict(m, z=self.z_from_artifact(art, late['site'], m['clause_positions'])) for m in late['maps']]
            edit = dict(site=int(late['site']), index=(t.arange(B, device=self.device), t.tensor(last, device=self.device)), maps=maps)
        self.plan = dict(edit=edit, capture=set()); self._reset()
        try:
            with t.no_grad():
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
                self.forward_count += 1
                if edit is not None and (edit['site'] not in self.applied or edit['site'] not in self.witnessed): raise RuntimeError('late edit not applied/witnessed')
                tail = out.last_hidden_state[t.arange(B, device=self.device), t.tensor(last, device=self.device)]
                lp = t.log_softmax(self.logits32(tail), dim=-1); am = lp.argmax(-1).tolist()
                results = []
                for r in range(B):
                    sel = lp[r, t.as_tensor(label_ids_batch[r], device=lp.device)]; logps = [float(v) for v in sel]
                    results.append(dict(logps=logps, prediction=int(np.argmax(logps)), answer_mass=float(sum(math.exp(v) for v in logps)), argmax_in_labels=am[r] in label_ids_batch[r],
                                        tie=bool(len(set(logps)) < len(logps)), prompt_tokens=P+len(suffix_batch[r]), suffix_tokens=len(suffix_batch[r]), finite=all(math.isfinite(v) for v in logps),
                                        realized=(dict(self.realized, realized_norms=[[x[r]] for x in self.realized['realized_norms']], energy=self.realized['row_energy'][r]) if self.realized else None)))
            if verify_master:
                after = self.cache_hash(art['cache'])
                for res in results: res['master_hash_before'] = before; res['master_hash_after'] = after; res['master_unchanged'] = before == after == art['hash']
                if not results[0]['master_unchanged']: raise RuntimeError('master artifact changed during ask_batch')
            if self.cache_length(art['cache']) != P: raise RuntimeError('master artifact length changed')
            for res in results: res['seconds'] = (time.monotonic()-start)/B
            return results
        finally:
            self.plan = None; self._reset()

    def describe(self):
        d = super().describe(); d.update(site=self.site, template_kwargs=self.template_kwargs, generation_prompt=self.generation_prompt(), serving='KV_CACHE_CLONE + batched clone asks (validated in E0)'); return d


