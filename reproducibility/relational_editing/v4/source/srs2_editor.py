"""SRS2 trainable maps.  One rank-16 per-token affine class for all four arms:

    delta_t = clip_cap( ([h_t - mu ; z - mu ; q - mu_q] C + b) V ) * strength

C is 3d x 16, V is 16 x d.  SHARED arms pass q = 0 (no query forward exists); QUERY arms pass the detached final source-question state of an
UNEDITED source+question forward.  Gemma warm start: the exact V1 PREFIX_DISTRIBUTED epoch-4 factors per seed with ZERO-padded query rows, so
QUERY_TAIL at initialization emulates SHARED_TAIL exactly for any q.  Qwen: zero-functional random init (nonzero random input factors, zero
output factors), its own FIT centers and cap.  Legacy 2d maps reproduce the frozen V1 references bit-for-bit in semantics.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
import srs2_common as Q

SCHEMA = 'SRS2_RANK16_AUGMENTED_V1'

class AugmentedEditor(nn.Module):
    def __init__(self, width, rank=Q.RANK, seed=0):
        super().__init__(); g = torch.Generator().manual_seed(int(seed))
        self.input_factor = nn.Parameter(torch.randn(3*width, rank, generator=g)*(.01/(3*width)**.5))
        self.output_factor = nn.Parameter(torch.zeros(rank, width)); self.bias = nn.Parameter(torch.zeros(rank))
        self.register_buffer('center', torch.zeros(width)); self.register_buffer('center_q', torch.zeros(width))
    def raw(self, h, z, q):
        zz = z.expand_as(h) if z.ndim == 1 else z
        qq = torch.zeros_like(h) if q is None else (q.expand_as(h) if q.ndim == 1 else q)
        d = h.shape[-1]
        # the [h-mu ; z-mu] product uses exactly the legacy shapes so a zero query block adds an exact 0.0 (bit-identical warm-start emulation)
        pre = torch.cat((h-self.center, zz-self.center), dim=-1)@self.input_factor[:2*d]+(qq-self.center_q)@self.input_factor[2*d:]
        return (pre+self.bias)@self.output_factor

class Bank:
    """Two augmented maps (desired value 0/1) sharing one site, one cap, one frozen center pair; shared across people/projects/positions/sizes."""
    def __init__(self, bb, arm, site, cap, seed, actor, maps=None):
        self.bb = bb; self.arm = arm; self.site = int(site); self.cap = float(cap); self.seed = int(seed); self.actor = actor
        self.maps = maps if maps is not None else {v: AugmentedEditor(bb.hidden, Q.RANK, seed=seed*97+v).to(bb.device) for v in (0, 1)}
        self.stats = dict(updates=0, saturated=0, raw_norm_sum=0.0)
    @property
    def aware(self): return Q.AWARE[self.arm]
    def parameters(self): return [p for e in self.maps.values() for p in e.parameters()]
    def set_centers(self, center, center_q):
        with torch.no_grad():
            for e in self.maps.values(): e.center.copy_(torch.as_tensor(center, dtype=torch.float32).to(self.bb.device)); e.center_q.copy_(torch.as_tensor(center_q, dtype=torch.float32).to(self.bb.device))
    @classmethod
    def from_v1(cls, bb, arm, site, cap, seed, actor, v1_dir, center_q):
        """Warm start from the exact V1 factors: input rows [h-mu ; z-mu] copied, the q block zero, bias/output copied, center = stored mu."""
        bank = cls(bb, arm, site, cap, seed, actor); meta = Q.load(Path(v1_dir)/'META.json'); files = {}
        for v, e in bank.maps.items():
            p = Path(v1_dir)/f'relation_{v}.npz'; files[f'relation_{v}'] = Q.sha(p)
            with np.load(p, allow_pickle=False) as z:
                C = np.asarray(z['input_factor'], dtype=np.float32); d = C.shape[0]//2
                if C.shape != (2*bb.hidden, Q.RANK): raise ValueError('unexpected V1 factor shape')
                with torch.no_grad():
                    e.input_factor.copy_(torch.as_tensor(np.concatenate((C, np.zeros((d, C.shape[1]), np.float32)), axis=0)))
                    e.output_factor.copy_(torch.as_tensor(np.asarray(z['output_factor'], np.float32))); e.bias.copy_(torch.as_tensor(np.asarray(z['bias'], np.float32)))
                    e.center.copy_(torch.as_tensor(np.asarray(z['center'], np.float32)).to(bb.device)); e.center_q.copy_(torch.as_tensor(center_q, dtype=torch.float32).to(bb.device))
                if float(z['cap']) != cap: raise ValueError('V1 cap differs from the inherited cap')
        bank.init = dict(kind='v1_warm_start', v1_dir=str(v1_dir), v1_files=files, v1_meta_sha256=Q.sha(Path(v1_dir)/'META.json'), v1_seed=meta['seed'], v1_epoch=meta['epoch'], v1_family=meta['family'])
        return bank
    @classmethod
    def fresh(cls, bb, arm, site, cap, seed, actor, center, center_q):
        bank = cls(bb, arm, site, cap, seed, actor); bank.set_centers(center, center_q); bank.init = dict(kind='zero_functional_random_input', seed=seed); return bank
    def fn(self, new_value, strength=1.0, q=None, record=True):
        e = self.maps[int(new_value)]; cap = self.cap; qq = None if q is None else torch.as_tensor(q, dtype=torch.float32, device=self.bb.device)
        if (not self.aware) and qq is not None: raise ValueError('a SHARED arm received a query feature')
        def f(h, z):
            raw = e.raw(h, z, qq); norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-12); d = raw*torch.clamp(cap/norm, max=1.)
            if record: self.stats['updates'] += int(h.shape[0]); self.stats['saturated'] += int((norm.detach() > cap).sum()); self.stats['raw_norm_sum'] += float(norm.detach().sum())
            return d*strength
        return f
    def save(self, path, **meta):
        path = Path(path); path.mkdir(parents=True, exist_ok=False); files = {}
        for v, e in sorted(self.maps.items()):
            p = path/f'relation_{v}.npz'
            np.savez(p, input_factor=e.input_factor.detach().cpu().numpy(), output_factor=e.output_factor.detach().cpu().numpy(), bias=e.bias.detach().cpu().numpy(),
                     center=e.center.detach().cpu().numpy(), center_q=e.center_q.detach().cpu().numpy(), cap=np.float64(self.cap), schema=np.asarray(SCHEMA))
            files[f'relation_{v}'] = Q.sha(p)
        Q.dump(path/'META.json', dict(at=Q.now(), schema=SCHEMA, actor=self.actor, arm=self.arm, site=self.site, seed=self.seed, cap=self.cap, aware=self.aware, footprint=Q.FOOTPRINT[self.arm], files=files, init=getattr(self, 'init', None), **meta))
        return files
    @classmethod
    def load(cls, bb, path):
        meta = Q.load(Path(path)/'META.json'); bank = cls(bb, meta['arm'], meta['site'], meta['cap'], meta['seed'], meta['actor'])
        for v, e in bank.maps.items():
            with np.load(Path(path)/f'relation_{v}.npz', allow_pickle=False) as z:
                if str(z['schema']) != SCHEMA: raise ValueError('checkpoint schema')
                with torch.no_grad():
                    e.input_factor.copy_(torch.as_tensor(z['input_factor'])); e.output_factor.copy_(torch.as_tensor(z['output_factor'])); e.bias.copy_(torch.as_tensor(z['bias']))
                    e.center.copy_(torch.as_tensor(z['center'])); e.center_q.copy_(torch.as_tensor(z['center_q']))
        bank.init = meta.get('init'); bank.meta = meta; return bank
    def frobenius(self):
        return {v: dict(input=float(e.input_factor.norm()), output=float(e.output_factor.norm()), bias=float(e.bias.norm()), q_block=float(e.input_factor[2*self.bb.hidden:].norm())) for v, e in self.maps.items()}

class LegacyBank:
    """Frozen V1 two-block maps ([h-mu ; z-mu]) for the historical SHARED_TAIL (PREFIX_DISTRIBUTED@15) and LATE@27 references; never trained here."""
    def __init__(self, bb, v1_dir, name):
        self.bb = bb; self.name = name; meta = Q.load(Path(v1_dir)/'META.json'); self.site = int(meta['site']); self.family = meta['family']; self.seed = int(meta['seed'])
        self.cap = float(meta['caps']['relation'])*float(meta.get('cap_scale', 1.0)); self.maps = {}; self.files = {}
        for v in (0, 1):
            p = Path(v1_dir)/f'relation_{v}.npz'; self.files[f'relation_{v}'] = Q.sha(p)
            with np.load(p, allow_pickle=False) as z:
                self.maps[v] = {k: torch.as_tensor(np.asarray(z[k], np.float32), device=bb.device) for k in ('input_factor', 'output_factor', 'bias', 'center')}
                if abs(float(z['cap'])-self.cap) > 1e-6: raise ValueError('legacy cap mismatch')
        self.meta_sha256 = Q.sha(Path(v1_dir)/'META.json'); self.dir = str(v1_dir); self.aware = False
    def fn(self, new_value, strength=1.0, q=None, record=False):
        m = self.maps[int(new_value)]; cap = self.cap
        def f(h, z):
            zz = z.expand_as(h) if z.ndim == 1 else z
            x = torch.cat((h-m['center'], zz-m['center']), dim=-1); raw = (x@m['input_factor']+m['bias'])@m['output_factor']
            norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-12); return raw*torch.clamp(cap/norm, max=1.)*strength
        return f
