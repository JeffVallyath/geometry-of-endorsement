"""Plain endpoint metrics. Units are scenes, not repeated question views."""
from collections import defaultdict
import numpy as np

def summarize(rows):
    if not rows: raise ValueError('empty records')
    groups=defaultdict(list)
    for r in rows:
        required={'scene_id','gold','source_gold','prediction','source_prediction'}
        if not required<=r.keys():raise ValueError('missing keys')
        groups[r['scene_id']].append(r)
    stats=[]
    for sid,rs in groups.items():
        changed=[r for r in rs if r['gold']!=r['source_gold']]
        invariant=[r for r in rs if r['gold']==r['source_gold']]
        stats.append({'scene_id':sid,'changed':np.mean([r['prediction']==r['gold'] for r in changed]) if changed else None,
         'invariant_harm':np.mean([r['source_prediction']==r['gold'] and r['prediction']!=r['gold'] for r in invariant]) if invariant else None,
         'all_correct':float(all(r['prediction']==r['gold'] for r in rs)),
         'accuracy':np.mean([r['prediction']==r['gold'] for r in rs]),'queries':len(rs)})
    def avg(k):
        xs=[r[k] for r in stats if r[k] is not None];return float(np.mean(xs)) if xs else None
    return {'n_scenes':len(stats),'n_queries':len(rows),**{k:avg(k) for k in ('changed','invariant_harm','all_correct','accuracy')},'per_scene':stats}

def paired_interval(a,b,key='changed',seed=260910,n=10000,level=.95):
    a={r['scene_id']:r for r in a['per_scene']};b={r['scene_id']:r for r in b['per_scene']}
    if a.keys()!=b.keys():raise ValueError('unmatched scenes')
    d=np.array([a[s][key]-b[s][key] for s in sorted(a) if a[s][key] is not None and b[s][key] is not None])
    if not len(d):raise ValueError('no valid scene pairs')
    rng=np.random.default_rng(seed);out=np.empty(n)
    for i in range(n):out[i]=d[rng.integers(len(d),size=len(d))].mean()
    q=(1-level)/2
    return {'effect':float(d.mean()),'lower':float(np.quantile(out,q)),'upper':float(np.quantile(out,1-q)),'scenes':len(d),'level':level}
