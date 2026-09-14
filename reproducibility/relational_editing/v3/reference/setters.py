"""Model-free reference operators, not a completed production backbone adapter.

Heads receive one addressed clause at a time. No evaluator metadata or question
may be passed. U is shared across values. Exact-arithmetic identities require a
fixed orthonormal U and no post-update clip/scale. Tests exercise the algebra;
correct LM answers are a separate empirical requirement.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import nn


def orthonormalize(x: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(x, mode='reduced')
    signs = torch.where(torch.diagonal(r) < 0, -torch.ones_like(torch.diagonal(r)), torch.ones_like(torch.diagonal(r)))
    return q * signs.unsqueeze(0)


class CoordinateSetter(nn.Module):
    """Fixed-shape desired-bit conditioned local-coordinate overwrite.

    kind='invariant' uses only the complement; kind='free' also reads mutable
    coordinates. Identical parameter shapes; no inference-time task labels.
    """
    def __init__(self, width: int, rank: int = 16, kind: str = 'invariant', seed: int = 0):
        super().__init__()
        if not (0 < rank < width) or kind not in ('invariant', 'free'):
            raise ValueError('rank must be in (0,width); kind invariant or free')
        self.width, self.rank, self.kind = width, rank, kind
        g = torch.Generator().manual_seed(seed)
        self.basis_raw = nn.Parameter(torch.randn(width, rank, generator=g) / width**0.5)
        self.head = nn.Parameter(torch.randn(2, 2*width, rank, generator=g) * .001)
        self.bias = nn.Parameter(torch.zeros(2, rank))
        self.register_buffer('center', torch.zeros(width))

    def basis(self):
        return orthonormalize(self.basis_raw)

    def forward(self, clause: torch.Tensor, desired: int, *, frozen_basis: torch.Tensor | None = None):
        if clause.ndim != 2 or clause.shape[0] == 0 or clause.shape[1] != self.width:
            raise ValueError('Expected nonempty addressed-clause matrix [tokens,width]')
        if desired not in (0, 1):
            raise ValueError('SET value must be 0 or 1')
        if clause.dtype not in (torch.float32, torch.float64):
            raise TypeError('Use FP32/FP64 workspace; cast to backbone dtype only on read')
        if not torch.isfinite(clause).all():
            raise ValueError('Nonfinite source state')
        u = self.basis() if frozen_basis is None else frozen_basis
        if u.shape != (self.width, self.rank):
            raise ValueError('Invalid frozen basis shape')
        h = clause - self.center
        remainder = h - (h @ u) @ u.T
        feature = remainder if self.kind == 'invariant' else h
        z = feature.mean(dim=0, keepdim=True).expand_as(feature)
        coords = torch.cat((feature, z), dim=1) @ self.head[desired] + self.bias[desired]
        return self.center + remainder + coords @ u.T

    @torch.no_grad()
    def copy_invariant_function_into_free(self, other: 'CoordinateSetter'):
        """Exact function nesting in real arithmetic, for independent tests only."""
        if self.kind != 'invariant' or other.kind != 'free' or self.width != other.width or self.rank != other.rank:
            raise ValueError('Requires equal-shape invariant -> free')
        other.basis_raw.copy_(self.basis_raw); other.center.copy_(self.center); other.bias.copy_(self.bias)
        u = self.basis()
        for v in (0, 1):
            for block in (0, 1):
                a = self.head[v, block*self.width:(block+1)*self.width]
                other.head[v, block*self.width:(block+1)*self.width].copy_(a - u @ (u.T @ a))


@dataclass(frozen=True)
class Command:
    address: tuple[int, ...]
    value: int
    def __post_init__(self):
        if not self.address or self.value not in (0, 1): raise ValueError('Invalid command')
        if min(self.address) < 0 or tuple(sorted(set(self.address))) != self.address:
            raise ValueError('Address must be sorted unique nonnegative indices')


def normalize_commands(commands: Sequence[Command]) -> tuple[Command, ...]:
    """Systems baseline: last write to each exact address, deterministic order.
    No gold, source bits, or question; not an in-place learned method.
    """
    last = {}
    for cmd in commands:
        if not isinstance(cmd, Command): raise TypeError('Commands only')
        last[cmd.address] = cmd
    keys = sorted(last)
    for i, key in enumerate(keys):
        for key2 in keys[i+1:]:
            if set(key).intersection(key2): raise ValueError('Partial overlapping addresses not supported')
    return tuple(last[k] for k in keys)


def apply_program(source: torch.Tensor, commands: Sequence[Command], setter: CoordinateSetter):
    """Pure local edit program. Caller keeps original source unchanged.
    No hidden normalization, deduplication, rollback, or query-dependent input.
    """
    if source.ndim != 2: raise ValueError('Expected [prefix_positions,width]')
    state = source.clone()
    u = setter.basis()  # fixed once per program; in inference store this U
    for cmd in commands:
        if max(cmd.address) >= state.shape[0]: raise ValueError('Address outside prefix')
        idx = torch.as_tensor(cmd.address, device=state.device, dtype=torch.long)
        update = setter(state.index_select(0, idx), cmd.value, frozen_basis=u)
        state = state.index_copy(0, idx, update)
    return state


def invariant_numpy(h: np.ndarray, u: np.ndarray, a: np.ndarray, b: np.ndarray, center: np.ndarray):
    """Independent NumPy implementation for numerical cross-checks."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or not len(h): raise ValueError('Expected clause')
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-8): raise ValueError('U must be orthonormal')
    z = h - center
    r = z - z @ u @ u.T
    f = np.concatenate((r, np.broadcast_to(r.mean(0), r.shape)), axis=1)
    return center + r + (f @ a + b) @ u.T
