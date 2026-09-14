# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""QBRC196 scientific unit inside the durable scientific-unit host.  Phases (markers make every stage re-enterable):

Q  qualification/interface (per model): load -> adapter/boundary checks -> C0..C2 native qualification on 48 fixed CAL development
   requests (<=32 new tokens) -> E0 checks on a 16-request CAL panel (cached path vs full forward, zero/self edit identity, clone/
   master/order checks, natural counterfactual reference) -> gradient check on 16 FIT queries -> timing benchmark.
D  development (per model): source captures/means/caps -> train the three construction families (sites x lrs, seed 0) with complete
   FIT/CAL curves -> declared selection -> seed-1 reruns -> FREEZE.json (before any FINAL artifact exists).
F  fresh evaluation (per model): seal all compiled artifacts (hashes) BEFORE any FINAL question is generated -> evaluate every
   condition on both draws -> matched random/wrong-address controls -> composition -> receiver/aggregator (relation).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import qbrc_common as Q
import qbrc_adapter as A

BASE = 'external-artifacts'

def event(output, name, **fields):
    row = dict(at=Q.now(), event=name, **fields)
    with (Path(output)/'stages.jsonl').open('a', encoding='utf8', newline='\n') as f: f.write(Q.canonical(row)+'\n')
    print(Q.canonical(row), flush=True)

def done(path): return Path(path).is_file()

# ---------------------------------------------------------------- qualification requests (fixed by hash before any outcome)
def qualification_requests(cal):
    """48 requests: per task 24 = 12 changed-type and 12 invariant-type CAL queries (draws 0/1), hash-ordered by query id."""
    out = []
    for task in Q.TASKS:
        pool = []
        for s in Q.by_task(cal, task):
            for draw in (0, 1):
                for q in Q.questions(s, 'CAL', draw): pool.append((s, q, draw))
        changed = sorted([x for x in pool if x[1].affected], key=lambda x: Q.hash_rank('qual', x[1].query_id))[:12]
        invariant = sorted([x for x in pool if not x[1].affected], key=lambda x: Q.hash_rank('qual', x[1].query_id))[:12]
        if len(changed) != 12 or len(invariant) != 12: raise ValueError('qualification pool')
        out.extend(changed+invariant)
    return out

def check_requests(split_scenes, split, n, salt):
    rows = []
    for s in split_scenes:
        for draw in (0, 1):
            for q in Q.questions(s, split, draw): rows.append((s, q, draw))
    return sorted(rows, key=lambda x: Q.hash_rank(salt, x[1].query_id))[:n]

def native_row(bb, s, q, draw, procedure, generate=True, target=None):
    text, span = Q.prefix_text(s, procedure, target=target); enc = bb.encode(text, Q.suffix_text(q, procedure)); lids = bb.label_ids(enc['full_text'], q.labels)
    art = bb.compile(enc['prefix_ids']); res = bb.ask(art, enc['suffix_ids'], lids, generate=generate)
    row = Q.query_record(s, q, s.split, draw); gold = q.source_gold_index if target is None else q.gold_index
    row.update(procedure=procedure, prefix_tokens=len(enc['prefix_ids']), suffix_tokens=len(enc['suffix_ids']), logps=res['logps'], score_prediction=res['prediction'], answer_mass=res['answer_mass'],
               argmax_in_labels=res['argmax_in_labels'], target_index=int(gold), score_correct=res['prediction'] == gold, prefix_hash=art['hash'], world='source' if target is None else 'counterfactual')
    if generate:
        sym, status = Q.parse_symbol(res['generation']['text'], q.labels); pred = list(q.labels).index(sym) if sym is not None else None
        row.update(generation=res['generation']['text'], generation_tokens=len(res['generation']['tokens']), truncated=res['generation']['truncated'], parse_status=status,
                   generation_prediction=pred, generation_correct=pred == gold, first_token_is_argmax=res['generation']['first_token_is_argmax'])
    return row

def qualify(bb, cal, out, deadline, model):
    if done(out/'QUALIFICATION.json'): return Q.load(out/'QUALIFICATION.json')
    reqs = qualification_requests(cal); attempts = []; selected = {}; generations = 0
    for procedure in Q.PROCEDURES:
        remaining_tasks = [t for t in Q.TASKS if t not in selected]
        if not remaining_tasks: break
        rows = []
        for s, q, draw in reqs:
            if s.task not in remaining_tasks: continue
            Q.budget(deadline, 120); rows.append(native_row(bb, s, q, draw, procedure)); generations += 1
        Q.write_rows(out/f'qualification_{procedure}.jsonl', rows)
        summary = {}
        for task in remaining_tasks:
            rs = [r for r in rows if r['task'] == task]; ch = [r for r in rs if r['affected']]; inv = [r for r in rs if not r['affected']]
            parse = float(np.mean([r['parse_status'].startswith('OK') for r in rs])); acc_ch = float(np.mean([r['generation_correct'] for r in ch])); acc_inv = float(np.mean([r['generation_correct'] for r in inv]))
            summary[task] = dict(n=len(rs), parse_rate=parse, changed_type_accuracy=acc_ch, invariant_type_accuracy=acc_inv, score_accuracy=float(np.mean([r['score_correct'] for r in rs])),
                                 generation_score_agreement=float(np.mean([r['generation_prediction'] == r['score_prediction'] for r in rs])), truncated=int(sum(r['truncated'] for r in rs)),
                                 qualified=parse >= Q.QUAL_PARSE_MIN and acc_ch >= Q.QUAL_ACC_MIN and acc_inv >= Q.QUAL_ACC_MIN, mean_new_tokens=float(np.mean([r['generation_tokens'] for r in rs])))
            if summary[task]['qualified']: selected[task] = procedure
        attempts.append(dict(procedure=procedure, tasks=remaining_tasks, summary=summary, generations=len(rows)))
        event(out, 'QUALIFICATION_ATTEMPT', procedure=procedure, summary={t: (v['parse_rate'], v['changed_type_accuracy'], v['invariant_type_accuracy'], v['qualified']) for t, v in summary.items()})
    # natural counterfactual competence on the same requests under the selected procedure (scoring only; informational)
    ref = []
    for s, q, draw in reqs:
        proc = selected.get(s.task)
        if proc is None: continue
        ref.append(native_row(bb, s, q, draw, proc, generate=False, target=Q.apply_edit(s)))
    Q.write_rows(out/'qualification_counterfactual_scores.jsonl', ref)
    rep = dict(at=Q.now(), model=model['key'], selected_procedures=selected, attempts=attempts, total_generations=generations, extra_generations=max(0, generations-Q.QUAL_REQUESTS), natural_panel_procedures={t: max(((a['procedure'], a['summary'][t]) for a in attempts if t in a['summary']), key=lambda x: (x[1]['changed_type_accuracy']+x[1]['invariant_type_accuracy'], x[1]['parse_rate']))[0] for t in Q.TASKS if t not in selected},
               rule='first procedure with >=95% parseable generations (<=32 new tokens) and >=90% native source-gold accuracy separately on changed-type and invariant-type CAL requests, per task',
               counterfactual_score_accuracy={t: float(np.mean([r['score_correct'] for r in ref if r['task'] == t])) for t in selected},
               qualified_tasks=sorted(selected), unqualified_tasks=[t for t in Q.TASKS if t not in selected])
    Q.dump(out/'QUALIFICATION.json', rep); return rep

# ---------------------------------------------------------------- E0 interface checks on the real model
def interface_checks(bb, fit, cal, out, deadline, model, procedures):
    if done(out/'INTERFACE_CHECK.json'): return Q.load(out/'INTERFACE_CHECK.json')
    import torch; from editor import AddressedLowRankEditor
    task = sorted(procedures)[0] if procedures else 'relation'; proc = procedures.get(task, 'C0')
    reqs = [x for x in check_requests(Q.by_task(cal, task), 'CAL', 64, 'e0') ][:16]
    late_site = model['late_site']; sites = list(model['sites'])
    rows = []; zero_ok = []; clone_ok = []; hashes = {}
    ed0 = AddressedLowRankEditor(bb.hidden, Q.RANK, seed=0).to(bb.device)
    for s, q, draw in reqs:
        Q.budget(deadline, 120); text, span = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q, proc)); lids = bb.label_ids(enc['full_text'], q.labels); cp = bb.clause_positions(text, span)
        pre = enc['prefix_ids']; art = bb.compile(pre, capture=set(sites)|{late_site}); hashes[s.scene_id] = art['hash']
        cached = bb.ask(art, enc['suffix_ids'], lids, verify_master=True)
        full = bb.full_forward(pre, [enc['suffix_ids']], [lids], grad=False)['label_logps'][0].detach().float().cpu().numpy().tolist()
        re_art = bb.compile(pre); recompile_equal = re_art['hash'] == art['hash']
        z_ok = True
        for site, idx in ((late_site, [len(pre)-1]), (sites[0], list(range(cp[0], len(pre))))):
            za = bb.compile(pre, maps=[dict(fn=lambda h, z, e=ed0: e(h, z, torch.ones(h.shape[0], device=h.device))-h, clause_positions=cp)], index=idx, site=site)
            z_ok = z_ok and za['hash'] == art['hash']
        zero_ok.append(z_ok); clone_ok.append(bb.cache_hash(bb.clone_cache(art['cache'])) == art['hash'])
        tgt = Q.apply_edit(s); ttext, _ = Q.prefix_text(s, proc, target=tgt); tenc = bb.encode(ttext, Q.suffix_text(q, proc)); tart = bb.compile(tenc['prefix_ids'])
        tc = bb.ask(tart, tenc['suffix_ids'], lids); tf = bb.full_forward(tenc['prefix_ids'], [tenc['suffix_ids']], [lids], grad=False)['label_logps'][0].detach().float().cpu().numpy().tolist()
        rows.append(dict(query_id=q.query_id, scene_id=s.scene_id, cached_logps=cached['logps'], full_logps=full, max_abs_diff=float(np.max(np.abs(np.asarray(cached['logps'])-np.asarray(full)))),
                         same_choice=int(np.argmax(cached['logps'])) == int(np.argmax(full)), master_unchanged=cached['master_unchanged'], recompile_hash_equal=recompile_equal,
                         counterfactual_cached=tc['logps'], counterfactual_full=tf, counterfactual_max_abs=float(np.max(np.abs(np.asarray(tc['logps'])-np.asarray(tf)))),
                         counterfactual_same_choice=int(np.argmax(tc['logps'])) == int(np.argmax(tf)), source_gold=q.source_gold_index, gold=q.gold_index, prefix_tokens=len(pre), clause_tokens=len(cp),
                         cache_bytes=art['bytes'], prefix_hash=art['hash']))
    # order independence: run the 16 requests forward then reversed on shared master artifacts
    arts = {}; fwd = []; rev = []
    for s, q, draw in reqs:
        text, span = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q, proc)); lids = bb.label_ids(enc['full_text'], q.labels)
        if s.scene_id not in arts: arts[s.scene_id] = bb.compile(enc['prefix_ids'])
        fwd.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], lids)['logps'])
    for s, q, draw in reversed(reqs):
        text, span = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q, proc)); lids = bb.label_ids(enc['full_text'], q.labels)
        rev.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], lids)['logps'])
    order_identical = fwd == rev[::-1]; masters_unchanged = all(bb.cache_hash(a['cache']) == a['hash'] for a in arts.values())
    rep = dict(at=Q.now(), model=model['key'], procedure=proc, task=task, requests=len(rows), rows=rows, zero_self_edit_hash_identical=all(zero_ok), clone_hash_identical=all(clone_ok),
               cached_vs_full=dict(max_abs_logp=max(r['max_abs_diff'] for r in rows), same_choice=all(r['same_choice'] for r in rows), mean_abs=float(np.mean([r['max_abs_diff'] for r in rows]))),
               counterfactual_reference=dict(max_abs_logp=max(r['counterfactual_max_abs'] for r in rows), same_choice=all(r['counterfactual_same_choice'] for r in rows)),
               order_independent=order_identical, masters_unchanged_after_reordered_queries=masters_unchanged, recompile_deterministic=all(r['recompile_hash_equal'] for r in rows),
               master_unchanged_every_ask=all(r['master_unchanged'] for r in rows), serving_path='KV_CACHE_CLONE', notes='same-path zero/self edit must match exactly; cached vs full forward differences are BF16 shape-dependent kernel effects, recorded not zero-tolerance')
    rep['pass_'] = rep['zero_self_edit_hash_identical'] and rep['clone_hash_identical'] and rep['order_independent'] and rep['master_unchanged_every_ask'] and rep['masters_unchanged_after_reordered_queries'] and rep['cached_vs_full']['same_choice']
    Q.dump(out/'INTERFACE_CHECK.json', rep); return rep

def gradient_check(bb, fit, out, deadline, model, procedures):
    if done(out/'GRADIENT_CHECK.json'): return Q.load(out/'GRADIENT_CHECK.json')
    import torch; from editor import AddressedLowRankEditor
    task = sorted(procedures)[0] if procedures else 'relation'; proc = procedures.get(task, 'C0')
    scenes = sorted(Q.by_task(fit, task), key=lambda s: Q.hash_rank('grad', s.scene_id))[:4]; res = {}
    for fam in Q.FAMILIES:
        for site in ([model['late_site']] if fam == 'LATE' else list(model['sites'])):
            Q.budget(deadline, 120)
            ed = AddressedLowRankEditor(bb.hidden, Q.RANK, seed=3).to(bb.device); opt = torch.optim.AdamW(ed.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
            maps = [dict(fn=lambda h, z, e=ed: e(h, z, torch.ones(h.shape[0], device=h.device))-h, clause_positions=None)]
            before = []; after = []; gnorm = None; loss_v = None; t0 = time.monotonic()
            for step in range(2):
                losses = []
                for s in scenes:
                    text, span = Q.prefix_text(s, proc); cp = bb.clause_positions(text, span); qs = Q.questions(s, 'FIT', 0)[:4]
                    encs = [bb.encode(text, Q.suffix_text(q, proc)) for q in qs]; lids = [bb.label_ids(e['full_text'], q.labels) for e, q in zip(encs, qs)]
                    m = [dict(maps[0], clause_positions=cp)]; pre = encs[0]['prefix_ids']
                    if fam == 'LATE': r = bb.full_forward(pre, [e['suffix_ids'] for e in encs], lids, late=dict(site=site, maps=m), grad=step == 0)
                    else:
                        idx = [len(pre)-1] if fam == 'PREFIX_LAST' else list(range(cp[0], len(pre)))
                        r = bb.full_forward(pre, [e['suffix_ids'] for e in encs], lids, prefix_maps=m, prefix_index=idx, prefix_site=site, grad=step == 0)
                    lp = torch.stack([r['label_logps'][i][qs[i].gold_index] for i in range(len(qs))])
                    (before if step == 0 else after).extend(lp.detach().float().cpu().tolist())
                    if step == 0: losses.append(-lp.mean())
                if step == 0:
                    loss = torch.stack(losses).mean(); bb.backward(loss); loss_v = float(loss)
                    gnorm = {n: float(p.grad.norm()) for n, p in ed.named_parameters()}
                    backbone_grad = any(p.grad is not None for p in bb.model.parameters())
                    torch.nn.utils.clip_grad_norm_(ed.parameters(), 1.0); opt.step(); opt.zero_grad()
            res[f'{fam}@{site}'] = dict(loss=loss_v, grad_norms=gnorm, backbone_grad=backbone_grad, finite=all(np.isfinite(v) for v in gnorm.values()), output_factor_grad_nonzero=gnorm['output_factor'] > 0,
                                       max_score_move=float(np.max(np.abs(np.asarray(after)-np.asarray(before)))), seconds=time.monotonic()-t0)
    rep = dict(at=Q.now(), model=model['key'], task=task, procedure=proc, scenes=[s.scene_id for s in scenes], queries=16, results=res, backbone_unchanged=bb.verify_unchanged()['unchanged'],
               pass_=all(v['finite'] and v['output_factor_grad_nonzero'] and not v['backbone_grad'] and v['max_score_move'] > 0 for v in res.values()))
    Q.dump(out/'GRADIENT_CHECK.json', rep); return rep

def benchmark(bb, fit, out, model, procedures):
    if done(out/'BENCHMARK.json'): return Q.load(out/'BENCHMARK.json')
    import torch; from editor import AddressedLowRankEditor
    task = sorted(procedures)[0] if procedures else 'relation'; proc = procedures.get(task, 'C0')
    s = sorted(Q.by_task(fit, task), key=lambda s: Q.hash_rank('bench', s.scene_id))[0]; text, span = Q.prefix_text(s, proc); cp = bb.clause_positions(text, span)
    qs = Q.questions(s, 'FIT', 0)[:4]; encs = [bb.encode(text, Q.suffix_text(q, proc)) for q in qs]; lids = [bb.label_ids(e['full_text'], q.labels) for e, q in zip(encs, qs)]; pre = encs[0]['prefix_ids']
    ed = AddressedLowRankEditor(bb.hidden, Q.RANK, seed=4).to(bb.device); maps = [dict(fn=lambda h, z, e=ed: e(h, z, torch.ones(h.shape[0], device=h.device))-h, clause_positions=cp)]
    def timed(fn, n=3):
        fn(); torch.cuda.synchronize(); t0 = time.monotonic()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.monotonic()-t0)/n
    art = bb.compile(pre, capture={model['late_site']})
    t = dict(compile=timed(lambda: bb.compile(pre)), compile_capture=timed(lambda: bb.compile(pre, capture={model['late_site']})), compile_distributed=timed(lambda: bb.compile(pre, maps=maps, index=list(range(cp[0], len(pre))), site=model['sites'][-1])),
             ask=timed(lambda: bb.ask(art, encs[0]['suffix_ids'], lids[0])), ask_late=timed(lambda: bb.ask(art, encs[0]['suffix_ids'], lids[0], late=dict(site=model['late_site'], maps=maps))),
             ask_generate32=timed(lambda: bb.ask(art, encs[0]['suffix_ids'], lids[0], generate=True), n=1), clone=timed(lambda: bb.clone_cache(art['cache'])), hash=timed(lambda: bb.cache_hash(art['cache'])))
    def step(fam, site):
        idx = [len(pre)-1] if fam == 'PREFIX_LAST' else list(range(cp[0], len(pre)))
        if fam == 'LATE': r = bb.full_forward(pre, [e['suffix_ids'] for e in encs], lids, late=dict(site=site, maps=maps), grad=True)
        else: r = bb.full_forward(pre, [e['suffix_ids'] for e in encs], lids, prefix_maps=maps, prefix_index=idx, prefix_site=site, grad=True)
        loss = -torch.stack([r['label_logps'][i][qs[i].gold_index] for i in range(4)]).mean(); bb.backward(loss); ed.zero_grad()
    for fam in Q.FAMILIES:
        for site in ([model['late_site']] if fam == 'LATE' else list(model['sites'])): t[f'train_step4_{fam}@{site}'] = timed(lambda: step(fam, site), n=2)
    rep = dict(at=Q.now(), model=model['key'], prefix_tokens=len(pre), suffix_tokens=[len(e['suffix_ids']) for e in encs], cache_bytes=art['bytes'], seconds=t, note='disposable timing panel; not a measurement')
    Q.dump(out/'BENCHMARK.json', rep); return rep

def phase_q(bb, out, deadline, model, fit, cal):
    qual = qualify(bb, cal, out, deadline, model); event(out, 'QUALIFICATION', selected=qual['selected_procedures'], unqualified=qual['unqualified_tasks'])
    procs = qual['selected_procedures']
    ic = interface_checks(bb, fit, cal, out, deadline, model, procs); event(out, 'INTERFACE_CHECK', pass_=ic['pass_'], cached_vs_full=ic['cached_vs_full'], zero=ic['zero_self_edit_hash_identical'])
    gc = gradient_check(bb, fit, out, deadline, model, procs); event(out, 'GRADIENT_CHECK', pass_=gc['pass_'], results={k: (v['output_factor_grad_nonzero'], v['max_score_move']) for k, v in gc['results'].items()})
    bench = benchmark(bb, fit, out, model, procs); event(out, 'BENCHMARK', seconds=bench['seconds'])
    return dict(qualification=qual, interface=ic['pass_'], gradient=gc['pass_'])

def main():
    raise RuntimeError("Original provider launcher is not distributed. Use the CPU reproduction entry point; optional model reruns require an explicit local runtime configuration.")

if __name__ == '__main__': main()
