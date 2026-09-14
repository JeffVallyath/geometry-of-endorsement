"""RSO3 editors.

SetterBank: the supplied CoordinateSetter (reference/setters.py; kind 'invariant' or 'free', shared orthonormal U = QR(basis_raw) with
canonical column signs, two heads indexed by the requested value only, no strength/clip/rounding inside).  Applied as a full replacement of
the addressed-clause matrix inside the FP32 workspace; at inference the realized U is frozen once per program.

AffineBank: the exact V2 rank-16 per-token affine map class (srs2_editor.Bank; cap-then-strength, q = 0) for CONTINUED_AFFINE, the
frozen V2 clause/tail references, the CANONICAL_V2 replay comparator and Qwen's independently trained affine comparator.

Warm starts (Gemma): U from the leading 16 right singular vectors of the concatenated two V2 clause output factors of the same seed;
heads by ridge regression (1e-3 after fixed feature scaling) on FIT source-clause states toward the incumbent V2 clause edit's projected
coordinates.  Qwen: U from FIT native paired-development clause-summary differences; FREE = exact identity (zero-functional), INVARIANT =
zero heads (its initial projection is not an identity).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
import rso3_common as Q
from reference.setters import CoordinateSetter, Command, normalize_commands, invariant_numpy, orthonormalize   # noqa: F401
from srs2_editor import Bank as V2Bank, AugmentedEditor   # noqa: F401

SCHEMA = 'RSO3_SETTER_V1'

class SetterBank:
    def __init__(self, bb, arm, site, seed, actor, cap_ref, setter=None):
        self.bb = bb; self.arm = arm; self.kind = Q.KIND[arm]; self.site = int(site); self.seed = int(seed); self.actor = actor; self.cap_ref = float(cap_ref)
        self.model = setter if setter is not None else CoordinateSetter(bb.hidden, Q.RANK, self.kind, seed=seed*101+(0 if self.kind == 'invariant' else 1)).to(bb.device)
        self.frozen_u = None; self.init = None; self.stats = dict(updates=0, delta_sq_sum=0.0)
    aware = False
    def parameters(self): return list(self.model.parameters())
    def basis(self): return self.frozen_u if self.frozen_u is not None else self.model.basis()
    def basis64(self):
        """Float64 orthonormalization of the same raw basis (same sign convention) for independent NumPy cross-checks."""
        with torch.no_grad(): return orthonormalize(self.model.basis_raw.detach().double()).cpu().numpy()
    def freeze_basis(self):
        with torch.no_grad(): self.frozen_u = self.model.basis().detach().clone()
        return self.frozen_u
    @torch.no_grad()
    def set_center(self, mu): self.model.center.copy_(torch.as_tensor(np.asarray(mu, np.float32)).to(self.bb.device))
    @torch.no_grad()
    def set_basis(self, u):
        u = torch.as_tensor(np.asarray(u, np.float32)).to(self.bb.device)
        if u.shape != (self.bb.hidden, Q.RANK): raise ValueError('basis shape')
        self.model.basis_raw.copy_(u); q = self.model.basis()
        err = float((q-u).abs().max())
        if err > 1e-4: raise ValueError(f'orthonormalization does not reproduce the initial basis ({err})')
        return err
    @torch.no_grad()
    def set_heads(self, head, bias):
        self.model.head.copy_(torch.as_tensor(np.asarray(head, np.float32)).to(self.bb.device)); self.model.bias.copy_(torch.as_tensor(np.asarray(bias, np.float32)).to(self.bb.device))
    def fn(self, new_value, strength=1.0, record=True):
        """Map callable for the FP32 workspace: clause matrix [tokens, d] (FP32) -> replaced clause matrix (FP32).  strength must be 1."""
        if float(strength) != 1.0: raise ValueError('setters have strength exactly 1; no multiplier exists')
        v = int(new_value); u = self.basis()
        def f(clause32):
            out = self.model(clause32, v, frozen_basis=u)
            if record: self.stats['updates'] += int(clause32.shape[0]); self.stats['delta_sq_sum'] += float(((out-clause32).detach()**2).sum())
            return out
        return f
    def save(self, path, **meta):
        path = Path(path); path.mkdir(parents=True, exist_ok=False); m = self.model
        with torch.no_grad(): u = m.basis().detach().cpu().numpy()
        p = path/'setter.npz'
        np.savez(p, basis_raw=m.basis_raw.detach().cpu().numpy(), basis_realized=u, head=m.head.detach().cpu().numpy(), bias=m.bias.detach().cpu().numpy(), center=m.center.detach().cpu().numpy(), cap_ref=np.float64(self.cap_ref), kind=np.asarray(self.kind), schema=np.asarray(SCHEMA))
        files = {'setter': Q.sha(p)}
        Q.dump(path/'META.json', dict(at=Q.now(), schema=SCHEMA, actor=self.actor, arm=self.arm, kind=self.kind, site=self.site, seed=self.seed, rank=Q.RANK, cap_ref=self.cap_ref, files=files, init=self.init, norms=self.norms(), **meta))
        return files
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
    def norms(self):
        m = self.model
        with torch.no_grad(): return dict(head0=float(m.head[0].norm()), head1=float(m.head[1].norm()), bias0=float(m.bias[0].norm()), bias1=float(m.bias[1].norm()), basis_raw=float(m.basis_raw.norm()))
    @torch.no_grad()
    def nested_free_copy(self):
        """Independent test helper: the free setter whose function equals this invariant setter (real-arithmetic nesting)."""
        other = CoordinateSetter(self.bb.hidden, Q.RANK, 'free', seed=self.seed).to(self.bb.device); self.model.copy_invariant_function_into_free(other); return other

class AffineBank(V2Bank):
    """The exact V2 map class with q = 0 (shared arms).  `fn` returns a delta (cap-then-strength) for the workspace."""
    @classmethod
    def from_v2(cls, bb, arm, seed):
        bank = V2Bank.load(bb, Q.v2_factor_dir(arm, seed)); bank.__class__ = cls; bank.init = dict(kind='v2_selected_factors', dir=str(Q.v2_factor_dir(arm, seed)), files=dict(bank.meta['files']), meta_sha256=Q.sha(Q.v2_factor_dir(arm, seed)/'META.json'), v2_arm=arm, v2_seed=seed)
        return bank
    @classmethod
    def fresh_clause(cls, bb, site, cap, seed, actor, center, center_q):
        bank = V2Bank.fresh(bb, 'SHARED_CLAUSE', site, cap, seed, actor, center, center_q); bank.__class__ = cls; return bank

# ---------------------------------------------------------------- initializations
def svd_basis(rows, rank=Q.RANK):
    """Leading `rank` right singular vectors (d x rank) of a row matrix, canonical signs (largest-magnitude entry positive)."""
    m = np.asarray(rows, np.float64)
    if m.ndim != 2 or m.shape[0] < rank: raise ValueError('need at least rank rows')
    _, s, vt = np.linalg.svd(m, full_matrices=False); u = vt[:rank].T
    for j in range(rank):
        k = int(np.argmax(np.abs(u[:, j])))
        if u[k, j] < 0: u[:, j] *= -1
    return u.astype(np.float32), s[:rank].astype(np.float64)

def gemma_basis_init(seed):
    v = np.concatenate([np.load(Q.v2_factor_dir('SHARED_CLAUSE', seed)/f'relation_{b}.npz', allow_pickle=False)['output_factor'] for b in (0, 1)], axis=0)
    u, s = svd_basis(v); return u, s, dict(kind='svd_of_concatenated_v2_clause_output_factors', v2_seed=seed, rows=int(v.shape[0]), singular_values=s.tolist())

def ridge_heads(features, targets, lam=Q.RIDGE):
    """Ridge regression with intercept (dual form; features scaled by one fixed scalar) -> A (F x r), c (r), scale, relative residual."""
    X = np.asarray(features, np.float64); Y = np.asarray(targets, np.float64); n, F = X.shape
    scale = float(np.sqrt((X**2).mean())) or 1.0; Xs = X/scale; xm = Xs.mean(0); ym = Y.mean(0); Xc = Xs-xm; Yc = Y-ym
    K = Xc@Xc.T; alpha = np.linalg.solve(K+lam*np.eye(n), Yc); A = Xc.T@alpha; c = ym-xm@A
    pred = Xs@A+c; res = float(np.sqrt(((pred-Y)**2).sum()/max(1e-12, (Y**2).sum())))
    return (A/scale).astype(np.float32), c.astype(np.float32), scale, res

def warm_start_heads(kind, clause_states, edited_states, u, mu):
    """clause_states/edited_states: list of [tokens, d] FP32 arrays per FIT group (source clause states and the incumbent's edited states);
    returns head (2d x r), bias (r), scale and the relative approximation error on the same states."""
    feats = []; tg = []
    for h, e in zip(clause_states, edited_states):
        hc = np.asarray(h, np.float64)-mu; ec = np.asarray(e, np.float64)-mu; r = hc-(hc@u)@u.T
        x = r if kind == 'invariant' else hc; z = np.broadcast_to(x.mean(0), x.shape)
        feats.append(np.concatenate((x, z), axis=1)); tg.append(ec@u)
    A, c, scale, res = ridge_heads(np.concatenate(feats), np.concatenate(tg)); return A, c, scale, res

def free_identity_heads(u, width):
    """FREE zero-functional init: F(H, Hbar) = (H - mu) U  ->  output = H exactly."""
    A = np.zeros((2*width, Q.RANK), np.float32); A[:width] = u; return A, np.zeros(Q.RANK, np.float32)

# ---------------------------------------------------------------- canonical comparator (systems baseline)
def canonical_program(addresses, values):
    """normalize_commands over Command(address, value): last write per exact address, sorted addresses; returns (addresses, values)."""
    cmds = normalize_commands([Command(tuple(int(i) for i in cp), int(v)) for cp, v in zip(addresses, values)])
    return [c.address for c in cmds], [c.value for c in cmds]
