from pathlib import Path
import numpy as np
import torch
from . import config as Q
from .setters import CoordinateSetter, Command, normalize_commands, orthonormalize
SCHEMA = "RSO3_SETTER_V1"

class SetterBank:
    def __init__(self, bb, arm, site, seed, actor, cap_ref, setter=None):
        self.bb = bb; self.arm = arm; self.kind = Q.KIND[arm]; self.site = int(site); self.seed = int(seed); self.actor = actor; self.cap_ref = float(cap_ref)
        self.model = setter if setter is not None else CoordinateSetter(bb.hidden, Q.RANK, self.kind, seed=seed*101+(0 if self.kind == 'invariant' else 1)).to(bb.device)
        self.frozen_u = None; self.init = None; self.stats = dict(updates=0, delta_sq_sum=0.0)

    aware = False

    def parameters(self): return list(self.model.parameters())

    def basis(self): return self.frozen_u if self.frozen_u is not None else self.model.basis()

    def fn(self, new_value, strength=1.0, record=True):
        """Map callable for the FP32 workspace: clause matrix [tokens, d] (FP32) -> replaced clause matrix (FP32).  strength must be 1."""
        if float(strength) != 1.0: raise ValueError('setters have strength exactly 1; no multiplier exists')
        v = int(new_value); u = self.basis()
        def f(clause32):
            out = self.model(clause32, v, frozen_basis=u)
            if record: self.stats['updates'] += int(clause32.shape[0]); self.stats['delta_sq_sum'] += float(((out-clause32).detach()**2).sum())
            return out
        return f

    @classmethod
    def load(cls, bb, path):
        meta = Q.load(Path(path)/'META.json'); bank = cls(bb, meta['arm'], meta['site'], meta['seed'], meta['actor'], meta['cap_ref'])
        with np.load(Path(path)/'setter.npz', allow_pickle=False) as z:
            if str(z['schema']) != SCHEMA or str(z['kind']) != bank.kind: raise ValueError('checkpoint schema/kind')
            with torch.no_grad():
                bank.model.basis_raw.copy_(torch.as_tensor(z['basis_raw'])); bank.model.head.copy_(torch.as_tensor(z['head'])); bank.model.bias.copy_(torch.as_tensor(z['bias'])); bank.model.center.copy_(torch.as_tensor(z['center']))
            realized = torch.as_tensor(z['basis_realized']).to(bb.device); bank.frozen_u = realized.clone()
            with torch.no_grad(): bank.load_basis_error = float((bank.model.basis()-realized).abs().max())
        bank.init = meta.get('init'); bank.meta = meta; return bank



def canonical_program(addresses, values):
    """normalize_commands over Command(address, value): last write per exact address, sorted addresses; returns (addresses, values)."""
    cmds = normalize_commands([Command(tuple(int(i) for i in cp), int(v)) for cp, v in zip(addresses, values)])
    return [c.address for c in cmds], [c.value for c in cmds]

