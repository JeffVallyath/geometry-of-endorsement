"""Narrow supplied prepare interface with verbatim saved-basis loading.

The immutable export's load method recomputes QR solely for a diagnostic.
V9 forbids that computation. This dependency adapter retains the exact writer,
prepare, FP32 hook and BF16-return implementation, substituting only the loader.
Model/hook/loading operations are serialized, never called from reader workers.
"""
from contextlib import contextmanager
from pathlib import Path
import inputs as I

def load_saved_editor(backbone, actor, recipe, seed, root=I.ROOT):
    import numpy as np
    import torch
    from canonical.bank import SetterBank, SCHEMA
    from canonical.config import MODELS
    if actor not in ('gemma','qwen') or seed not in (0,1):
        raise ValueError('Fixed actor/seed required')
    data=I.inventory(root)
    if recipe=='CANONICAL_REPLAY':
        recipe=data['canonical_replay'][actor][str(seed)]
    elif recipe not in I.RECIPES:
        raise ValueError('Undeclared recipe')
    records=[r for r in data['selected'] if (r['actor'],r['recipe'],r['seed'])==(actor,recipe,seed)]
    if len(records)!=1: raise ValueError('Unique selected factor required')
    rec=records[0]; path=Path(root)/rec['path']
    for name,key in [('setter.npz','setter_sha256'),('META.json','meta_sha256')]:
        if I.file_hash(path/name)!=rec[key]: raise ValueError('Changed selected factor')
    meta=I.json.loads((path/'META.json').read_text())
    if (meta['actor'],meta['seed'],meta['site'],meta['rank'])!=(actor,seed,MODELS[actor]['site'],16):
        raise ValueError('Factor identity/site/rank mismatch')
    bank=SetterBank(backbone,meta['arm'],meta['site'],seed,actor,meta['cap_ref'])
    with np.load(path/'setter.npz',allow_pickle=False) as z, torch.no_grad():
        if str(z['schema'])!=SCHEMA or str(z['kind'])!=bank.kind: raise ValueError('Wrong factor schema')
        for field in ('basis_raw','head','bias','center'):
            value=torch.as_tensor(z[field],device=backbone.device)
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite saved factors')
            getattr(bank.model,field).copy_(value)
        bank.frozen_u=torch.as_tensor(z['basis_realized'],device=backbone.device).clone()
        u=bank.frozen_u
        if u.shape!=(backbone.hidden,16) or not torch.isfinite(u).all(): raise ValueError('Saved basis invalid')
        if not torch.allclose(u.double().T@u.double(),torch.eye(16,dtype=torch.float64,device=u.device),atol=2e-6,rtol=0):
            raise ValueError('Saved basis Gram witness invalid')
    bank.meta=meta; bank.init=meta.get('init'); bank.model.eval()
    for p in bank.parameters(): p.requires_grad_(False)
    return bank,rec

@contextmanager
def saved_loader():
    previous=I.load_editor
    if previous is not _ORIGINAL_LOADER:
        raise RuntimeError('Concurrent or unexpected loader mutation')
    I.load_editor=load_saved_editor
    try: yield
    finally: I.load_editor=previous

_ORIGINAL_LOADER=I.load_editor

def prepare(source_text, commands, *, method, backbone, actor, seed=None):
    with saved_loader():
        return I.prepare(source_text,commands,method=method,backbone=backbone,actor=actor,seed=seed)

def move_context(context, device):
    """Lossless master-cache transport; no recompilation or metadata reset."""
    art=context._artifact
    for layer in art['cache'].layers:
        layer.keys=layer.keys.to(device)
        layer.values=layer.values.to(device)
    return context
