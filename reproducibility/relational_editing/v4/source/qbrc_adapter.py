"""QBRC196 model-side adapter: frozen BF16 backbone, FP32 readout, validated prefix/suffix token boundaries, multi-position
post-block residual hooks, question-blind context compilation (prefix forward -> immutable key/value artifact, hashed), per-question
cache clones, batched gradient training path, and the predeclared residual-replay fallback.

Nothing here reads a question inside `compile`: its inputs are prefix token ids and an edit plan built from the source prefix and
the explicit addressed edit request only.  Query tokens only ever enter `ask`/`train_forward`, on a clone (inference) or on a
fresh per-step cache (training) that is discarded afterwards.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import qbrc_common as Q

class BoundaryError(ValueError): pass

def validate_cache(cache, cfg):
    snapshot = Path(cache)/('models--'+cfg['id'].replace('/', '--'))/'snapshots'/cfg['revision']
    config = Q.load(snapshot/'config.json'); index = Q.load(snapshot/'model.safetensors.index.json')
    expected = ['tokenizer.json', 'tokenizer_config.json', *set(index['weight_map'].values())]
    missing = [n for n in expected if not (snapshot/n).is_file()]
    if missing: raise FileNotFoundError('Pinned snapshot missing '+','.join(missing))
    if config['num_hidden_layers'] != cfg['layers']: raise ValueError('wrong model depth')
    if config.get('final_logit_softcapping') != cfg['cap']: raise ValueError('wrong model-specific output transform')
    return snapshot

def parameter_checksum(model, torch, sites):
    out = {}
    with torch.no_grad():
        W = model.lm_head.weight.detach()
        out['head_dtype'] = str(W.dtype); out['head_sample_sum'] = float(W[::97, ::13].to(torch.float64).sum().item())
        out['head_sample_sha256'] = hashlib.sha256(W[::997, ::7].to(torch.float32).cpu().numpy().tobytes()).hexdigest()
        norm = getattr(model.model, 'norm', None)
        if norm is not None and hasattr(norm, 'weight'): out['final_norm_sum'] = float(norm.weight.detach().to(torch.float64).sum().item())
        layers = model.model.layers
        for i in sorted({len(layers)//2, *[s for s in sites if s < len(layers)], *[s+1 for s in sites if s+1 < len(layers)]}):
            w = next(layers[i].parameters()).detach(); out[f'layer{i}_sample_sum'] = float(w.flatten()[::1009].to(torch.float64).sum().item())
        out['all_parameter_sum'] = float(sum(p.detach().to(torch.float64).sum().item() for p in model.parameters()))
    return out

def tensor_bytes(t):
    import torch
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16: return t.view(torch.int16).cpu().numpy().tobytes()
    return t.cpu().numpy().tobytes()

class Backbone:
    def __init__(self, cache=None, cfg=Q.GEMMA, model=None, tokenizer=None, require_cuda=True, sites=None):
        import torch
        self.t = torch; self.cfg = dict(cfg)
        self.sites = sorted(set((sites if sites is not None else cfg['sites']))|{cfg['late_site']})
        if model is None:
            validate_cache(cache, cfg)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            options = dict(revision=cfg['revision'], cache_dir=str(cache), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(cfg['id'], **options)
            model = AutoModelForCausalLM.from_pretrained(cfg['id'], torch_dtype=torch.bfloat16, device_map='auto', **options)
        self.tokenizer = tokenizer; self.model = model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        if require_cuda and not all(p.dtype == torch.bfloat16 and p.device.type == 'cuda' for p in self.model.parameters() if p.is_floating_point()):
            raise RuntimeError('model must be unmodified BF16 on GPU, no offload')
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()): raise RuntimeError('preexisting hooks')
        if torch.is_inference_mode_enabled(): raise RuntimeError('adapter must be built outside inference_mode')
        head = self.model.lm_head
        if getattr(head, 'bias', None) is not None: raise RuntimeError('output head carries a bias')
        self.layers = self.model.model.layers
        if len(self.layers) != cfg['layers']: raise RuntimeError(f'actual module depth {len(self.layers)} != pinned {cfg["layers"]}')
        if max(self.sites)+1 >= len(self.layers): raise RuntimeError('site witness needs a following block')
        self.previous_precision = dict(tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32), matmul=str(torch.get_float32_matmul_precision()))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        self.W32 = head.weight.detach().clone().to(torch.float32); self.W32.requires_grad_(False)
        cap = getattr(self.model.config, 'final_logit_softcapping', None); self.cap = float(cap) if cap is not None else None
        if self.cap != cfg['cap']: raise RuntimeError('incorrect model readout softcapping')
        self.hidden = int(self.W32.shape[1]); self.vocab = int(self.W32.shape[0]); self.device = next(self.model.parameters()).device
        if self.hidden != cfg['hidden']: raise RuntimeError(f'hidden size {self.hidden} != pinned {cfg["hidden"]}')
        self.checksum_at_init = parameter_checksum(self.model, torch, self.sites)
        self.forward_count = 0; self.backward_count = 0; self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.plan = None; self._reset()
        self.handles = [self.layers[s].register_forward_hook(self._hook(s)) for s in self.sites]
        self.handles += [self.layers[s+1].register_forward_pre_hook(self._witness(s), with_kwargs=True) for s in self.sites]
        self._prefix_cache = {}

    def _reset(self):
        self.captured = {}; self.after = {}; self.applied = set(); self.witnessed = set(); self.realized = None; self.z_seen = {}

    # ---------------------------------------------------------------- hooks
    def _hook(self, site):
        t = self.t
        def cb(module, args, out):
            plan = self.plan
            if plan is None: return out
            h = out[0] if isinstance(out, tuple) else out
            if site in plan.get('capture', ()) and site not in self.captured: self.captured[site] = h.detach()
            edit = plan.get('edit')
            if edit is None or edit['site'] != site or site in self.applied: return out
            rows, pos = edit['index']
            if int(pos.max()) >= h.shape[1] or int(rows.max()) >= h.shape[0]: raise ValueError('edit index outside the current forward')
            h2 = h.clone(); realized = []; intended = []; zs = []
            for m in edit['maps']:
                rows, pos = m.get('index', edit['index'])
                sel = h2[rows, pos]
                if m.get('mode') == 'replacement':
                    new32 = t.as_tensor(m['target'], device=h.device, dtype=t.float32); delta = new32-sel.float()
                else:
                    z = m.get('z')
                    if z is None:
                        cp = t.as_tensor(m['clause_positions'], device=h.device)
                        z = h2[0, cp].float().mean(0)            # PRE-edit (for this map) residual mean over addressed clause tokens
                    zs.append(z.detach()); delta = m['fn'](sel.float(), z)
                    if delta.shape != sel.shape: raise ValueError('delta shape')
                    new32 = sel.float()+delta
                if not t.isfinite(new32).all(): raise ValueError('NONFINITE_EDITED_STATE')
                new = new32.to(h.dtype)
                h2[rows, pos] = new
                realized.append((new.detach().float()-sel.detach().float()).norm(dim=-1)); intended.append(delta.detach().float().norm(dim=-1))
            self.realized = dict(site=site, positions=int(len(pos)), intended_norms=[x.tolist() for x in intended], realized_norms=[x.tolist() for x in realized],
                                 energy=float(sum((x**2).sum() for x in realized)), maps=len(edit['maps']), z_norms=[float(z.norm()) for z in zs])
            self.z_seen[site] = zs
            rows, pos = edit['index']; self.after[site] = (rows, pos, h2[rows, pos].detach()); self.applied.add(site)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2
        return cb

    def _witness(self, site):
        def cb(module, args, kwargs):
            if self.plan is None or site not in self.after or site in self.witnessed: return
            h = args[0] if args else kwargs['hidden_states']; rows, pos, new = self.after[site]
            if int(pos.max()) >= h.shape[1]: return
            if not self.t.equal(h[rows, pos].detach(), new): raise RuntimeError(f'post-hook state not propagated to next block at site {site}')
            self.witnessed.add(site)
        return cb

    # ---------------------------------------------------------------- tokens
    def render(self, prefix_text, suffix_text):
        full = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prefix_text+suffix_text}], tokenize=False, add_generation_prompt=True)
        off = full.find(prefix_text)
        if off < 0 or full.find(prefix_text, off+1) >= 0: raise BoundaryError('prefix text not unique in the rendered prompt')
        return full, off

    def prefix_ids(self, prefix_text):
        key = hashlib.sha256(prefix_text.encode()).hexdigest()
        if key not in self._prefix_cache:
            full, off = self.render(prefix_text, 'x')
            self._prefix_cache[key] = self.tokenizer.encode(full[:off+len(prefix_text)], add_special_tokens=False)
        return list(self._prefix_cache[key])

    def encode(self, prefix_text, suffix_text):
        """Token ids of the actual full prompt with a validated prefix boundary: prefix ids are a token prefix of the full prompt."""
        full, off = self.render(prefix_text, suffix_text); pre = self.prefix_ids(prefix_text)
        ids = self.tokenizer.encode(full, add_special_tokens=False)
        if ids[:len(pre)] != pre: raise BoundaryError(f'prefix ids ({len(pre)}) are not a token prefix of the full prompt ({len(ids)})')
        if len(ids) == len(pre): raise BoundaryError('empty suffix')
        return dict(full_text=full, prefix_ids=pre, suffix_ids=ids[len(pre):], full_ids=ids, header_offset=off)

    def clause_positions(self, prefix_text, span):
        """Token positions (in the prefix id sequence) whose character offsets overlap the addressed clause span; validated."""
        full, off = self.render(prefix_text, 'x'); pre_text = full[:off+len(prefix_text)]
        enc = self.tokenizer(pre_text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc['input_ids']) != self.prefix_ids(prefix_text): raise BoundaryError('offset encoding differs from prefix ids')
        a, b = off+span[0], off+span[1]; pos = [i for i, (s, e) in enumerate(enc['offset_mapping']) if e > a and s < b]
        if not pos or pos != list(range(pos[0], pos[-1]+1)): raise BoundaryError('clause tokens not contiguous')
        covered = pre_text[enc['offset_mapping'][pos[0]][0]:enc['offset_mapping'][pos[-1]][1]]
        if prefix_text[span[0]:span[1]].strip() not in covered: raise BoundaryError('clause span not covered by its tokens')
        return pos

    def label_ids(self, full_text, labels):
        base = self.tokenizer.encode(full_text, add_special_tokens=False); ids = []
        for c in labels:
            seq = self.tokenizer.encode(full_text+c, add_special_tokens=False)
            if seq[:len(base)] != base or len(seq) != len(base)+1: raise BoundaryError(f'label {c!r} is not a single token after the generation prompt')
            ids.append(int(seq[len(base)]))
        if len(set(ids)) != len(ids): raise BoundaryError('label token ids collide')
        return ids

    # ---------------------------------------------------------------- readout
    def logits32(self, tail):
        pre = self.t.matmul(tail.to(self.t.float32), self.W32.T)
        return self.t.tanh(pre/self.cap)*self.cap if self.cap is not None else pre

    def readout(self, tail, label_ids):
        """Full-vocabulary FP32 log-softmax; label log-probs, answer mass, full-vocab argmax."""
        lp = self.t.log_softmax(self.logits32(tail), dim=-1)       # [B, V]
        ids = self.t.as_tensor(label_ids, device=lp.device)
        sel = lp[:, ids] if lp.ndim == 2 else lp[ids]
        return lp, sel

    # ---------------------------------------------------------------- cache primitives
    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.model.config)

    def cache_pairs(self, cache):
        pairs = []
        for i, layer in enumerate(cache.layers):
            k = getattr(layer, 'keys', None); v = getattr(layer, 'values', None)
            if k is None or v is None: raise RuntimeError(f'layer {i} cache uninitialized')
            pairs.append((k, v))
        return pairs

    def cache_hash(self, cache):
        h = hashlib.sha256()
        for k, v in self.cache_pairs(cache):
            h.update(str(tuple(k.shape)).encode()); h.update(tensor_bytes(k)); h.update(str(tuple(v.shape)).encode()); h.update(tensor_bytes(v))
        return h.hexdigest()

    def cache_bytes(self, cache): return int(sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k, v in self.cache_pairs(cache)))

    def clone_cache(self, cache): return copy.deepcopy(cache)

    def cache_length(self, cache): return int(cache.get_seq_length())

    # ---------------------------------------------------------------- compile (question-blind)
    def compile(self, prefix_ids, maps=None, index=None, site=None, capture=(), grad=False, replay_positions=None):
        """Prefix forward once with the (optional) edit plan; returns the immutable key/value artifact.
        maps: list of dict(fn, clause_positions|z) or dict(mode='replacement', target) applied in order at `index` positions of `site`.
        No question argument exists.  With grad=True the artifact carries the editor graph (training only)."""
        t = self.t; P = len(prefix_ids)
        edit = None
        if maps:
            rows = t.zeros(len(index), dtype=t.long, device=self.device); pos = t.as_tensor(list(index), dtype=t.long, device=self.device)
            edit = dict(site=int(site), index=(rows, pos), maps=maps)
        self.plan = dict(edit=edit, capture=set(capture)); self._reset()
        cache = self.new_cache(); ids = t.tensor([prefix_ids], device=self.device); mask = t.ones_like(ids)
        ctx = contextlib.nullcontext() if grad else t.no_grad()
        try:
            with ctx:
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            if edit is not None and edit['site'] not in self.applied: raise RuntimeError('prefix edit was not applied')
            if edit is not None and edit['site'] not in self.witnessed: raise RuntimeError('prefix edit propagation not witnessed')
            cache = out.past_key_values if out.past_key_values is not None else cache
            if self.cache_length(cache) != P: raise RuntimeError('cache length differs from prefix length')
            art = dict(cache=cache, prefix_len=P, prefix_ids=list(prefix_ids), realized=self.realized, captured={s: v[0] for s, v in self.captured.items()}, grad=grad,
                       prefix_sha256=Q.digest(np.asarray(prefix_ids, dtype='<i8').tobytes()))
            if not grad: art['hash'] = self.cache_hash(cache); art['bytes'] = self.cache_bytes(cache)
            if replay_positions is not None and site is not None:
                rp = t.as_tensor(list(replay_positions), device=self.device)
                st = (self.captured[site][0, rp] if edit is None else self.after[site][2]).detach().float().cpu().numpy().copy()
                art['replay'] = dict(site=int(site), positions=list(map(int, replay_positions)), states=st, sha256=Q.digest(st.astype('<f4').tobytes()))
            return art
        finally:
            self.plan = None; self._reset()

    def z_from_artifact(self, art, site, clause_positions):
        return art['captured'][site][clause_positions].float().mean(0)

    # ---------------------------------------------------------------- ask (query-side; master artifact untouched)
    def ask(self, art, suffix_ids, label_ids, late=None, generate=False, max_new_tokens=Q.MAX_NEW_TOKENS, verify_master=False):
        """Score (and optionally greedily generate) one suffix against a CLONE of the compiled artifact.
        late: dict(site, maps) applied at the final real prompt token of the suffix (query-time editor)."""
        t = self.t; start = time.monotonic(); P = art['prefix_len']; S = len(suffix_ids)
        before = self.cache_hash(art['cache']) if verify_master else None
        cache = self.clone_cache(art['cache'])
        ids = t.tensor([suffix_ids], device=self.device); mask = t.ones(1, P+S, dtype=t.long, device=self.device)
        edit = None
        if late is not None:
            maps = [m if m.get('z') is not None or m.get('mode') == 'replacement' else dict(m, z=self.z_from_artifact(art, late['site'], m['clause_positions'])) for m in late['maps']]
            edit = dict(site=int(late['site']), index=(t.zeros(1, dtype=t.long, device=self.device), t.tensor([S-1], device=self.device)), maps=maps)
        self.plan = dict(edit=edit, capture=set()); self._reset()
        try:
            with t.no_grad():
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
                self.forward_count += 1
                if edit is not None and (edit['site'] not in self.applied or edit['site'] not in self.witnessed): raise RuntimeError('late edit not applied/witnessed')
                tail = out.last_hidden_state[0, -1]; lp, sel = self.readout(tail.unsqueeze(0), label_ids); sel = sel[0]
                logps = [float(v) for v in sel]; argmax_full = int(lp[0].argmax())
                res = dict(logps=logps, prediction=int(np.argmax(logps)), answer_mass=float(sum(math.exp(v) for v in logps)), argmax_in_labels=argmax_full in label_ids,
                           tie=bool(len(set(logps)) < len(logps)), prompt_tokens=P+S, suffix_tokens=S, realized=self.realized, finite=all(math.isfinite(v) for v in logps))
            if generate:
                self.plan = dict(edit=edit, capture=set()); self._reset(); gcache = self.clone_cache(art['cache'])
                full = t.tensor([self._full_ids(art, suffix_ids)], device=self.device)
                with t.no_grad():
                    g = self.model.generate(input_ids=full, attention_mask=t.ones_like(full), past_key_values=gcache, use_cache=True, max_new_tokens=max_new_tokens,
                                            do_sample=False, pad_token_id=self.pad_id)
                toks = g[0, full.shape[1]:].tolist(); self.forward_count += len(toks)
                text = self.tokenizer.decode(toks, skip_special_tokens=True)
                res['generation'] = dict(text=text, tokens=toks, truncated=len(toks) == max_new_tokens, first_token_is_argmax=(toks[:1] == [argmax_full]))
            res['seconds'] = time.monotonic()-start
            if verify_master:
                after = self.cache_hash(art['cache']); res['master_hash_before'] = before; res['master_hash_after'] = after; res['master_unchanged'] = before == after == art['hash']
                if not res['master_unchanged']: raise RuntimeError('master artifact changed during ask')
            if self.cache_length(art['cache']) != P: raise RuntimeError('master artifact length changed')
            return res
        finally:
            self.plan = None; self._reset()

    def ask_capture(self, art, suffix_ids, label_ids, site):
        """Natural suffix forward on a clone; returns the final-real-token post-block state at `site` (float32 numpy).  No edit."""
        t = self.t; P = art['prefix_len']; S = len(suffix_ids); cache = self.clone_cache(art['cache'])
        ids = t.tensor([suffix_ids], device=self.device); mask = t.ones(1, P+S, dtype=t.long, device=self.device)
        self.plan = dict(edit=None, capture={int(site)}); self._reset()
        try:
            with t.no_grad(): self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            return self.captured[int(site)][0, S-1].float().cpu().numpy().copy()
        finally:
            self.plan = None; self._reset()

    def _full_ids(self, art, suffix_ids):
        if 'prefix_ids' not in art: raise ValueError('artifact lacks prefix ids for generation')
        return list(art['prefix_ids'])+list(suffix_ids)

    # ---------------------------------------------------------------- full-forward paths (training, replay fallback, reference)
    def full_forward(self, prefix_ids, suffix_batch, label_ids_batch, prefix_maps=None, prefix_index=None, prefix_site=None, late=None, grad=True, replay=None):
        """Prefix forward (hooked at prefix positions only) + batched suffix forward on a fresh per-step cache.  Returns per-row label
        log-probs [B, L], full-vocab log-probs at the last real token, and realized-norm records.  Causal attention makes the prefix
        states query-independent; this is the training path and the reference semantics for the compiled cached path."""
        t = self.t; B = len(suffix_batch); P = len(prefix_ids)
        maps = prefix_maps
        if replay is not None:
            maps = [dict(mode='replacement', target=replay['states'])]; prefix_index = replay['positions']; prefix_site = replay['site']
        art = self.compile(prefix_ids, maps=maps, index=prefix_index, site=prefix_site, grad=grad and maps is not None and replay is None,
                           capture={late['site']} if late else ())
        cache = art['cache']
        if late is not None:
            late = dict(late, maps=[m if m.get('z') is not None or m.get('mode') == 'replacement' else dict(m, z=self.z_from_artifact(art, late['site'], m['clause_positions'])) for m in late['maps']])
        if B > 1: cache.batch_repeat_interleave(B)
        S = max(len(s) for s in suffix_batch); ids = t.full((B, S), self.pad_id, dtype=t.long, device=self.device); mask = t.ones(B, P+S, dtype=t.long, device=self.device)
        last = []
        for r, s in enumerate(suffix_batch):
            ids[r, :len(s)] = t.tensor(s, device=self.device); mask[r, P+len(s):] = 0; last.append(len(s)-1)
        edit = None
        if late is not None:
            edit = dict(site=int(late['site']), index=(t.arange(B, device=self.device), t.tensor(last, device=self.device)), maps=late['maps'])
        self.plan = dict(edit=edit, capture=set(late.get('capture', ())) if late else set()); self._reset()
        ctx = contextlib.nullcontext() if grad else t.no_grad()
        try:
            with ctx:
                out = self.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
            self.forward_count += 1
            if edit is not None and (edit['site'] not in self.applied or edit['site'] not in self.witnessed): raise RuntimeError('late edit not applied/witnessed')
            tail = out.last_hidden_state[t.arange(B, device=self.device), t.tensor(last, device=self.device)]
            lp = self.t.log_softmax(self.logits32(tail), dim=-1)
            sel = [lp[r, t.as_tensor(label_ids_batch[r], device=lp.device)] for r in range(B)]
            return dict(label_logps=sel, full_logps=lp, prefix_realized=art['realized'], late_realized=self.realized, captured=self.captured, prefix_len=P)
        finally:
            self.plan = None; self._reset()

    def backward(self, loss):
        loss.backward(); self.backward_count += 1

    # ---------------------------------------------------------------- bookkeeping
    def describe(self):
        return dict(model=self.cfg, sites=self.sites, hidden=self.hidden, vocab=self.vocab, final_logit_softcapping=self.cap, actual_layers=len(self.layers),
                    fp32_readout='detached clone outside inference_mode, requires_grad False', backbone_requires_grad=any(p.requires_grad for p in self.model.parameters()),
                    tf32_disabled=not self.t.backends.cuda.matmul.allow_tf32, matmul_precision=self.t.get_float32_matmul_precision(),
                    attn_implementation=getattr(self.model.config, '_attn_implementation', None), layer_types=list(getattr(self.model.config, 'layer_types', []) or [])[:4])

    def verify_unchanged(self):
        now = parameter_checksum(self.model, self.t, self.sites)
        return dict(unchanged=now == self.checksum_at_init, at_init=self.checksum_at_init, now=now)

    def finish(self):
        for h in self.handles: h.remove()
        result = self.verify_unchanged()
        self.t.backends.cuda.matmul.allow_tf32 = self.previous_precision['tf32']; self.t.backends.cudnn.allow_tf32 = self.previous_precision['cudnn']
        self.t.set_float32_matmul_precision(self.previous_precision['matmul'])
        if not result['unchanged']: raise RuntimeError('model parameter checksum changed')
        return result
