from __future__ import annotations
import contextlib, math
import numpy as np
from . import config as Q
from . import batch_adapter as V2A

class Backbone(V2A.Backbone):
    def _reset(self):
        super()._reset(); self.workspace_after = None; self.map_records = None

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
            if h.shape[0] != 1: raise ValueError('RSO3 compiles batch-1 prefixes only')
            mode = edit.get('workspace', 'fp32'); records = []
            if mode == 'legacy':
                h2 = h.clone()
                for m in edit['maps']:
                    rws, ps = m.get('index', edit['index']); sel = h2[rws, ps]
                    if m.get('mode') == 'replacement': new32 = t.as_tensor(m['target'], device=h.device, dtype=t.float32)
                    elif m.get('mode') == 'delta':
                        delta = t.as_tensor(m['delta'], device=h.device, dtype=t.float32)
                        if delta.shape != sel.shape: raise ValueError('fixed delta shape')
                        new32 = sel.float()+delta
                    elif m.get('kind') == 'setter': raise ValueError('setters run in the FP32 workspace only')
                    else:
                        cp = t.as_tensor(m['clause_positions'], device=h.device); z = h2[0, cp].float().mean(0)      # PRE-edit (for this map) BF16 clause mean, as in V2
                        delta = m['fn'](sel.float(), z)
                        if delta.shape != sel.shape: raise ValueError('delta shape')
                        new32 = sel.float()+delta
                    if not t.isfinite(new32).all(): raise ValueError('NONFINITE_EDITED_STATE')
                    new = new32.to(h.dtype); h2[rws, ps] = new
                    records.append(dict(positions=int(len(ps)), intended=(new32.detach()-sel.detach().float()).norm(dim=-1).tolist(), realized=(new.detach().float()-sel.detach().float()).norm(dim=-1).tolist()))
                    m['_delta_sq'] = ((new32-sel.float())**2).sum(-1)
                after_ws = h2[rows, pos].detach().float()
            else:
                w = h.float()                       # FP32 working residual of the whole prefix (no gradient from the frozen backbone)
                for m in edit['maps']:
                    rws, ps = m.get('index', edit['index']); sel = w[rws, ps]
                    if m.get('mode') == 'replacement': new = t.as_tensor(m['target'], device=h.device, dtype=t.float32)
                    elif m.get('mode') == 'delta':
                        delta = t.as_tensor(m['delta'], device=h.device, dtype=t.float32)
                        if delta.shape != sel.shape: raise ValueError('fixed delta shape')
                        new = sel+delta
                    elif m.get('kind') == 'setter': new = m['fn'](sel)                       # full replacement of the addressed-clause matrix
                    else:
                        cp = t.as_tensor(m['clause_positions'], device=h.device); z = w[0, cp].mean(0)              # current FP32 clause mean (pre this map)
                        delta = m['fn'](sel, z)
                        if delta.shape != sel.shape: raise ValueError('delta shape')
                        new = sel+delta
                    if new.shape != sel.shape: raise ValueError('map output shape')
                    if not t.isfinite(new).all(): raise ValueError('NONFINITE_EDITED_STATE')
                    m['_delta_sq'] = ((new-sel)**2).sum(-1)
                    records.append(dict(positions=int(len(ps)), intended=(new.detach()-sel.detach()).norm(dim=-1).tolist(), realized=None))
                    w = w.index_put((rws, ps), new)
                after_ws = w[rows, pos].detach()
                h2 = h.index_put((rows, pos), w[rows, pos].to(h.dtype))                      # BF16 cast only here, when returning to the backbone
            new_bf = h2[rows, pos]; realized = (new_bf.detach().float()-h[rows, pos].detach().float()).norm(dim=-1)
            self.realized = dict(site=site, positions=int(len(pos)), workspace=mode, maps=len(edit['maps']), map_records=records, realized_norms=[realized.tolist()],
                                 intended_norms=[r['intended'] for r in records], energy=float((realized**2).sum()), frobenius=float(math.sqrt(float((realized**2).sum()))),
                                 cast_error=float((after_ws-new_bf.detach().float()).norm(dim=-1).max()) if len(pos) else 0.0)
            self.after[site] = (rows, pos, new_bf.detach()); self.workspace_after = after_ws; self.applied.add(site)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2
        return cb

    def compile(self, prefix_ids, maps=None, index=None, site=None, capture=(), grad=False, replay_positions=None, batch=1, rows=None, capture_next=(), workspace='fp32'):
        """Prefix forward once with the (optional) edit program; returns the immutable key/value artifact (hashed when not grad).
        maps: ordered list of dict(kind='setter', fn, index) | dict(kind='affine', fn, clause_positions, index) | dict(mode='delta'|'replacement', ...).
        index: union of edited positions.  No question argument exists."""
        t = self.t; P = len(prefix_ids); edit = None
        if batch != 1 or rows is not None: raise ValueError('RSO3 compiles batch-1 prefixes only')
        if maps:
            pos = t.as_tensor(list(index), dtype=t.long, device=self.device); rws = t.zeros(len(index), dtype=t.long, device=self.device)
            edit = dict(site=int(site), index=(rws, pos), maps=maps, workspace=workspace)
        self.plan = dict(edit=edit, capture=set(capture), capture_next=set(capture_next)); self._reset()
        cache = self.new_cache(); ids = t.tensor([list(prefix_ids)], device=self.device); mask = t.ones_like(ids)
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
                       next_input={s: v[0] for s, v in self.next_input.items()}, grad=grad, batch=1, prefix_sha256=Q.digest(np.asarray(prefix_ids, dtype='<i8').tobytes()), workspace=(workspace if edit else None))
            if not grad: art['hash'] = self.cache_hash(cache); art['bytes'] = self.cache_bytes(cache)
            if edit is not None and not grad:
                art['_after_states'] = self.after[edit['site']][2].detach().float().cpu().numpy().copy(); art['_workspace_states'] = self.workspace_after.detach().float().cpu().numpy().copy()
                art['_index'] = list(map(int, index))
            if replay_positions is not None and site is not None:
                rp = t.as_tensor(list(replay_positions), device=self.device)
                st = (self.captured[site][0, rp] if edit is None else self.after[site][2]).detach().float().cpu().numpy().copy()
                art['replay'] = dict(site=int(site), positions=list(map(int, replay_positions)), states=st, sha256=Q.digest(st.astype('<f4').tobytes()))
            return art
        finally:
            self.plan = None; self._reset()

    def describe(self):
        d = super().describe(); d.update(program_workspace='FP32 whole-prefix working residual inside the site hook; BF16 cast only when written back to the block output; legacy per-command BF16 path retained for the historical V2 references'); return d


