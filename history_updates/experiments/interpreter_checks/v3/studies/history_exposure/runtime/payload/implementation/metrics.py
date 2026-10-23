"""Small tested statistical primitives; no missing-result defaults."""
from __future__ import annotations
import numpy as np

def paired_interval(left,right,seed=2026100607,draws=10000,level=.975):
    a=np.asarray(left,dtype=float);b=np.asarray(right,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)==0:raise ValueError('PAIRED_SUBJECT_VECTORS')
    d=a-b;rng=np.random.default_rng(seed)
    v=d[rng.integers(0,len(d),size=(draws,len(d)))].mean(axis=1)
    alpha=(1-level)/2
    return {'difference':float(d.mean()),'interval':[float(x) for x in np.quantile(v,[alpha,1-alpha])],'subjects':len(d),'draws':draws,'confidence':level}

def preserved_repair(baseline,candidate,targets,controls):
    """Inputs map fully qualified row keys to correctness booleans.
    A disappearing disagreement or a refusal is not a repaired witness.
    """
    required=set(targets)|set(controls)
    if not required<=set(baseline) or not required<=set(candidate):raise ValueError('MISSING_ROWS')
    return all(candidate[t] for t in targets) and all(not baseline[c] or candidate[c] for c in controls)
