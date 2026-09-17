from __future__ import annotations
import copy, hashlib, math
from pathlib import Path
import numpy as np
from . import config as Q

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

    def logits32(self, tail):
        pre = self.t.matmul(tail.to(self.t.float32), self.W32.T)
        return self.t.tanh(pre/self.cap)*self.cap if self.cap is not None else pre

    def readout(self, tail, label_ids):
        """Full-vocabulary FP32 log-softmax; label log-probs, answer mass, full-vocab argmax."""
        lp = self.t.log_softmax(self.logits32(tail), dim=-1)       # [B, V]
        ids = self.t.as_tensor(label_ids, device=lp.device)
        sel = lp[:, ids] if lp.ndim == 2 else lp[ids]
        return lp, sel

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

    def z_from_artifact(self, art, site, clause_positions):
        return art['captured'][site][clause_positions].float().mean(0)

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


