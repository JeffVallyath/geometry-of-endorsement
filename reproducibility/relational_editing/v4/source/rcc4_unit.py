# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""RCC4 scientific unit inside the durable scientific-unit host (markers make every stage re-enterable).

Per actor (Gemma, then Qwen):
Q  load -> native qualification on the fresh CAL under the V3-selected procedure only (same symmetric primitive policy; an actor that fails
   stops at the natural panel) -> E0 interface panel (rso3_unit) -> warm-start gradient check on the loaded V3 checkpoints (basis/head/bias
   gradients nonzero, no backbone gradient, scores move) -> disposable timing benchmark.  A technical defect stops that actor before D.
D  rcc4_train.phase_d (four matched coverage conditions x two seeds, <=2 epochs, complete-single CAL selection) -> FREEZE.json.
F  rcc4_final.phase_f (sealed FULL evaluation) under the actor's effective deadline (Gemma leaves the Qwen reserve untouched).
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time
import traceback
from pathlib import Path
import numpy as np
import rcc4_common as C
import rso3_common as Q
import rso3_adapter as A
import rso3_unit as U
from rso3_unit import event, done

BASE = 'external-artifacts'

def warm_gradient_check(bb, fit, out, deadline, model, proc, store):
    if done(out/'GRADIENT_CHECK.json'): return Q.load(out/'GRADIENT_CHECK.json')
    import torch; import rcc4_train as R; import rso3_train as T
    scenes = sorted(fit, key=lambda s: C.hash_rank('grad', s.scene_id))[:4]; res = {}; bl = [R.scene_bundle(bb, s, 'FIT', proc, coverage='complete') for s in scenes]
    for arm in C.ARMS:
        Q.budget(deadline, 120); t0 = time.monotonic(); bank = R.load_v3_bank(bb, model['key'], arm, 0, store, model['site'])
        opt = torch.optim.AdamW(bank.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0); before = []; after = []; g = None; loss_v = None; backbone_grad = None
        for step in range(2):
            losses = []
            for b in bl:
                sub = b['queries'][:4]; pl = T.plan_for(bank, b, 1.0, record=False); r = bb.full_forward(b['prefix'], [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, workspace=pl['workspace'])
                lp = torch.stack([r['label_logps'][i][sub[i]['rec']['gold']] for i in range(4)]); (before if step == 0 else after).extend(lp.detach().float().cpu().tolist()); l_ = -lp.mean()/len(bl)+C.BETA*torch.cat([m['_delta_sq'] for m in pl['maps']]).mean()/(bank.cap_ref**2)/len(bl)
                bb.backward(l_); losses.append(float(l_.detach())); del r, lp, l_
            if step == 0:
                loss_v = float(sum(losses)); g = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in bank.model.named_parameters()}
                backbone_grad = any(p.grad is not None for p in bb.model.parameters()); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad()
            else: opt.zero_grad()
        nonzero = all(g[k] is not None and g[k] > 0 for k in ('basis_raw', 'head', 'bias'))
        res[arm] = dict(loss=loss_v, grad_norms=g, backbone_grad=backbone_grad, finite=all(v is not None and np.isfinite(v) for v in g.values()), editor_grad_nonzero=nonzero, max_score_move=float(np.max(np.abs(np.asarray(after)-np.asarray(before)))), warm_start=bank.init, seconds=time.monotonic()-t0)
    rep = dict(at=Q.now(), actor=model['key'], procedure=proc, scenes=[s.scene_id for s in scenes], queries=16, coverage='complete', results=res, backbone_unchanged=bb.verify_unchanged()['unchanged'],
               pass_=all(v['finite'] and v['editor_grad_nonzero'] and not v['backbone_grad'] and v['max_score_move'] > 0 for v in res.values()))
    Q.dump(out/'GRADIENT_CHECK.json', rep); return rep

def phase_q(bb, out, deadline, model, fit, cal, store):
    qual = U.qualify(bb, cal, out, deadline, model); event(out, 'QUALIFICATION', selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'], equality=qual['equality_summary'])
    proc = qual['selected_procedure'] or qual['natural_panel_procedure']
    ic = U.interface_checks(bb, cal, out, deadline, model, proc); event(out, 'INTERFACE_CHECK', pass_=ic['pass_'], cached_vs_full=ic['cached_vs_full'], batched=ic['batched_vs_single'], identities={k: v for k, v in ic['setter_identities'].items() if k != 'rows'})
    gc = warm_gradient_check(bb, fit, out, deadline, model, proc, store); event(out, 'GRADIENT_CHECK', pass_=gc['pass_'], results={k: (v['editor_grad_nonzero'], v['backbone_grad'], v['max_score_move']) for k, v in gc['results'].items()})
    bench = U.benchmark(bb, fit, out, model, proc); event(out, 'BENCHMARK', seconds=bench['seconds'])
    return dict(qualification=qual['selected_procedure'], interface=ic['pass_'], gradient=gc['pass_'], technical_pass=bool(ic['pass_'] and gc['pass_']))

def main():
    raise RuntimeError("Original provider launcher is not distributed. Use the CPU reproduction entry point; optional model reruns require an explicit local runtime configuration.")

if __name__ == '__main__': main()
