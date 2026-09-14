"""SRS2 model-side adapter: the validated QBRC196 Backbone (frozen BF16 backbone, FP32 model-correct readout, validated token boundaries,
post-block residual hooks with next-block propagation witness, question-blind compile -> immutable hashed key/value artifact, per-question
clones) extended with (a) an independent actor (Qwen3-8B, non-thinking chat template, own normalization/head), (b) batched prefix
compilation with per-row edit inputs for the query-aware arms (one recompiled prefix per question), (c) batched clone-served asks,
(d) fixed-delta maps for the equal-energy and random-direction diagnostics, and (e) physical witness capture (post-block residual,
next-block input, key/value tensors).

`compile` has no question argument.  Query tokens enter only `ask*`/`full_forward` on a clone or a discarded per-step cache; the
query feature q of the aware arms is produced by `ask_capture_batch` on an UNEDITED source artifact and passed to the map as data.
"""
from __future__ import annotations
import contextlib
import hashlib
import math
import time
import numpy as np
import srs2_common as Q
import qbrc_adapter as V1
from qbrc_adapter import validate_cache, parameter_checksum, tensor_bytes, BoundaryError  # noqa: F401

class Backbone(V1.Backbone):
    def __init__(self, cache=None, cfg=Q.GEMMA, model=None, tokenizer=None, require_cuda=True, sites=None):
        cfg2 = dict(cfg); cfg2['late_site'] = cfg['late_site'] if cfg.get('late_site') is not None else cfg['site']
        self.template_kwargs = dict(cfg.get('template_kwargs') or {})
        super().__init__(cache=cache, cfg=cfg2, model=model, tokenizer=tokenizer, require_cuda=require_cuda, sites=(sites if sites is not None else cfg2['sites']))
        self.site = int(cfg['site']); self.next_input = {}; self.captured_full = {}

    def _reset(self):
        super()._reset(); self.next_input = {}; self.captured_full = {}; self.row_energy = None

    # ---------------------------------------------------------------- tokens (actor-specific chat template)
    def render(self, prefix_text, suffix_text):
        full = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prefix_text+suffix_text}], tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        off = full.find(prefix_text)
        if off < 0 or full.find(prefix_text, off+1) >= 0: raise BoundaryError('prefix text not unique in the rendered prompt')
        return full, off

    def generation_prompt(self):
        """The exact generation-prompt text the actor's template appends after the user turn (Qwen: includes the empty think block)."""
        full, off = self.render('x', 'y'); return full[off+2:]

    # ---------------------------------------------------------------- hooks (per-row z, fixed-delta maps, witness capture)
    def _hook(self, site):
        t = self.t
        def cb(module, args, out):
            plan = self.plan
            if plan is None: return out
            h = out[0] if isinstance(out, tuple) else out
            if site in plan.get('capture', ()) and site not in self.captured: self.captured[site] = h.detach(); self.captured_full[site] = h.detach()
            edit = plan.get('edit')
            if edit is None or edit['site'] != site or site in self.applied: return out
            rows, pos = edit['index']
            if int(pos.max()) >= h.shape[1] or int(rows.max()) >= h.shape[0]: raise ValueError('edit index outside the current forward')
            h2 = h.clone(); realized = []; intended = []; zs = []; row_energy = t.zeros(h.shape[0], device=h.device)
            for m in edit['maps']:
                rows, pos = m.get('index', edit['index'])
                sel = h2[rows, pos]
                if m.get('mode') == 'replacement':
                    new32 = t.as_tensor(m['target'], device=h.device, dtype=t.float32); delta = new32-sel.float()
                elif m.get('mode') == 'delta':
                    delta = t.as_tensor(m['delta'], device=h.device, dtype=t.float32)
                    if delta.shape != sel.shape: raise ValueError('fixed delta shape')
                    new32 = sel.float()+delta
                else:
                    z = m.get('z')
                    if z is None:
                        cp = t.as_tensor(m['clause_positions'], device=h.device); ur = t.unique(rows)
                        zr = {int(r): h2[int(r), cp].float().mean(0) for r in ur}          # PRE-edit (for this map) clause mean of THIS row's current residual
                        z = t.stack([zr[int(r)] for r in rows]) if len(ur) > 1 else zr[int(ur[0])]
                    zs.append(z.detach() if z.ndim == 1 else z.detach()[0]); delta = m['fn'](sel.float(), z)
                    if delta.shape != sel.shape: raise ValueError('delta shape')
                    new32 = sel.float()+delta
                if not t.isfinite(new32).all(): raise ValueError('NONFINITE_EDITED_STATE')
                new = new32.to(h.dtype); h2[rows, pos] = new
                rn = (new.detach().float()-sel.detach().float()).norm(dim=-1); realized.append(rn); intended.append(delta.detach().float().norm(dim=-1))
                row_energy.index_add_(0, rows, rn**2)
            self.realized = dict(site=site, positions=int(len(pos)), intended_norms=[x.tolist() for x in intended], realized_norms=[x.tolist() for x in realized],
                                 energy=float(sum((x**2).sum() for x in realized)), maps=len(edit['maps']), z_norms=[float(z.norm()) for z in zs], row_energy=row_energy.tolist(),
                                 frobenius=float(math.sqrt(sum(float((x**2).sum()) for x in realized))))
            self.z_seen[site] = zs; rows, pos = edit['index']; self.after[site] = (rows, pos, h2[rows, pos].detach()); self.applied.add(site)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2
        return cb

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

    # ---------------------------------------------------------------- compile (question-blind; optional batch of per-row edit inputs)
    def compile(self, prefix_ids, maps=None, index=None, site=None, capture=(), grad=False, replay_positions=None, batch=1, rows=None, capture_next=()):
        """Prefix forward once with the (optional) edit plan; returns the immutable key/value artifact (hashed when batch == 1).
        maps: list of dict(fn, clause_positions|z[, index]) | dict(mode='replacement', target) | dict(mode='delta', delta) applied in order
        at `index` positions of `site`.  With batch > 1 the same prefix is forwarded `batch` times and `rows` gives the row of every index
        entry (aware arms: one recompiled prefix per question, q supplied to the map as data).  No question argument exists."""
        t = self.t; P = len(prefix_ids); edit = None
        if maps:
            pos = t.as_tensor(list(index), dtype=t.long, device=self.device)
            rws = t.zeros(len(index), dtype=t.long, device=self.device) if rows is None else t.as_tensor(list(rows), dtype=t.long, device=self.device)
            edit = dict(site=int(site), index=(rws, pos), maps=maps)
        self.plan = dict(edit=edit, capture=set(capture), capture_next=set(capture_next)); self._reset()
        cache = self.new_cache(); ids = t.tensor([list(prefix_ids)]*batch, device=self.device); mask = t.ones_like(ids)
        ctx = contextlib.nullcontext() if grad else t.no_grad()
        try:
            with ctx:
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            if edit is not None and edit['site'] not in self.applied: raise RuntimeError('prefix edit was not applied')
            if edit is not None and edit['site'] not in self.witnessed: raise RuntimeError('prefix edit propagation not witnessed')
            cache = out.past_key_values if out.past_key_values is not None else cache
            if self.cache_length(cache) != P: raise RuntimeError('cache length differs from prefix length')
            art = dict(cache=cache, prefix_len=P, prefix_ids=list(prefix_ids), realized=self.realized, captured={s: v[0] for s, v in self.captured.items()}, captured_full=dict(self.captured_full),
                       next_input={s: v[0] for s, v in self.next_input.items()}, grad=grad, batch=batch, prefix_sha256=Q.digest(np.asarray(prefix_ids, dtype='<i8').tobytes()))
            if not grad and batch == 1: art['hash'] = self.cache_hash(cache); art['bytes'] = self.cache_bytes(cache)
            if edit is not None and not grad: art['_after_states'] = self.after[edit['site']][2].detach().float().cpu().numpy().copy()
            if replay_positions is not None and site is not None:
                rp = t.as_tensor(list(replay_positions), device=self.device)
                st = (self.captured[site][0, rp] if edit is None else self.after[site][2]).detach().float().cpu().numpy().copy()
                art['replay'] = dict(site=int(site), positions=list(map(int, replay_positions)), states=st, sha256=Q.digest(st.astype('<f4').tobytes()))
            return art
        finally:
            self.plan = None; self._reset()

    def layer_kv(self, cache, layer):
        k, v = self.cache_pairs(cache)[layer]; return k.detach().float().cpu().numpy(), v.detach().float().cpu().numpy()

    # ---------------------------------------------------------------- batched asks on a clone (master untouched)
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

    def ask_capture_batch(self, art, suffix_batch, site):
        """Natural (unedited) suffix forward of B questions on a clone of an UNEDITED artifact; returns the final-real-token post-block state
        at `site` per question (float32 numpy [B, d]).  This is the only origin of the query feature q of the aware arms."""
        t = self.t; P = art['prefix_len']; B = len(suffix_batch); cache = self.clone_cache(art['cache'])
        if B > 1: cache.batch_repeat_interleave(B)
        S = max(len(s) for s in suffix_batch); ids = t.full((B, S), self.pad_id, dtype=t.long, device=self.device); mask = t.ones(B, P+S, dtype=t.long, device=self.device); last = []
        for r, s in enumerate(suffix_batch): ids[r, :len(s)] = t.tensor(s, device=self.device); mask[r, P+len(s):] = 0; last.append(len(s)-1)
        self.plan = dict(edit=None, capture={int(site)}); self._reset()
        try:
            with t.no_grad(): self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            h = self.captured_full[int(site)]; return h[t.arange(B, device=h.device), t.tensor(last, device=h.device)].float().cpu().numpy().copy()
        finally:
            self.plan = None; self._reset()

    # ---------------------------------------------------------------- full-forward path (training; aware arms with per-row prefix edits)
    def full_forward(self, prefix_ids, suffix_batch, label_ids_batch, prefix_maps=None, prefix_index=None, prefix_site=None, late=None, grad=True, replay=None, prefix_rows=None, prefix_batch=1):
        """Prefix forward (hooked at prefix positions only; optionally `prefix_batch` rows with per-row maps) + batched suffix forward on a
        fresh per-step cache.  Returns per-row label log-probs [B, L], full-vocab log-probs at the last real token, realized-norm records."""
        t = self.t; B = len(suffix_batch); P = len(prefix_ids); maps = prefix_maps
        if replay is not None: maps = [dict(mode='replacement', target=replay['states'])]; prefix_index = replay['positions']; prefix_site = replay['site']
        if prefix_batch not in (1, B): raise ValueError('prefix batch must be 1 or the suffix batch size')
        art = self.compile(prefix_ids, maps=maps, index=prefix_index, site=prefix_site, grad=grad and maps is not None and replay is None, capture={late['site']} if late else (), batch=prefix_batch, rows=prefix_rows)
        cache = art['cache']
        if late is not None:
            late = dict(late, maps=[m if m.get('z') is not None or m.get('mode') in ('replacement', 'delta') else dict(m, z=self.z_from_artifact(art, late['site'], m['clause_positions'])) for m in late['maps']])
        if B > 1 and prefix_batch == 1: cache.batch_repeat_interleave(B)
        S = max(len(s) for s in suffix_batch); ids = t.full((B, S), self.pad_id, dtype=t.long, device=self.device); mask = t.ones(B, P+S, dtype=t.long, device=self.device); last = []
        for r, s in enumerate(suffix_batch): ids[r, :len(s)] = t.tensor(s, device=self.device); mask[r, P+len(s):] = 0; last.append(len(s)-1)
        edit = None
        if late is not None: edit = dict(site=int(late['site']), index=(t.arange(B, device=self.device), t.tensor(last, device=self.device)), maps=late['maps'])
        self.plan = dict(edit=edit, capture=set(late.get('capture', ())) if late else set()); self._reset()
        ctx = contextlib.nullcontext() if grad else t.no_grad()
        try:
            with ctx:
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            if edit is not None and (edit['site'] not in self.applied or edit['site'] not in self.witnessed): raise RuntimeError('late edit not applied/witnessed')
            tail = out.last_hidden_state[t.arange(B, device=self.device), t.tensor(last, device=self.device)]
            lp = t.log_softmax(self.logits32(tail), dim=-1)
            sel = [lp[r, t.as_tensor(label_ids_batch[r], device=lp.device)] for r in range(B)]
            return dict(label_logps=sel, full_logps=lp, prefix_realized=art['realized'], late_realized=self.realized, captured=self.captured, prefix_len=P)
        finally:
            self.plan = None; self._reset()

    def describe(self):
        d = super().describe(); d.update(site=self.site, template_kwargs=self.template_kwargs, generation_prompt=self.generation_prompt(), serving='KV_CACHE_CLONE + batched clone asks (validated in E0)'); return d
