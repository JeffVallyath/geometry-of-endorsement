"""SRS2 phase D: fair development of the four matched arms on FIT/CAL only (both order seeds), then FREEZE before any FINAL access.

D0 captures  per FIT scene: source (and natural counterfactual) prefixes compiled question-blind at the actor's site -> frozen FIT prefix mean
             (Qwen center; Gemma keeps the stored V1 center), clause-summary change statistics (Qwen cap = 2 x p95; Gemma keeps the inherited
             103.18592071533203), and the query feature q of EVERY FIT/CAL question from an UNEDITED source+question forward (frozen store;
             mu_q = FIT mean).  q is data for the aware arms only; shared arms never see it.
D1 training  per arm: seed 0 at both LRs for LR_EPOCHS complete epochs, the better CAL LR extended to at most E_max epochs (stop after two
             epochs without CAL frontier improvement), seed 1 at the chosen LR with the same opportunity.  AdamW (.9,.999) eps 1e-8 wd 0, grad
             clip 1, groups of four questions from one scene, changed/invariant balanced within a scene, no-op scenes invariant-only.
             Complete FIT (strength 1) and CAL (strengths .5/.75/1/1.25) evaluated at every epoch boundary through the cached serving path
             (shared) or the per-question recompiled path (aware).
D2 selection per arm by the declared two-seed rule -> FREEZE.json.  Nothing here reads a FINAL or WORKFLOW scene.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import srs2_common as Q
from srs2_editor import Bank, LegacyBank
from srs2_unit import event, done

# ---------------------------------------------------------------- encodings
def scene_bundle(bb, s, split, proc, program='single', order='original', variant=0, world=None, correction=False):
    """Compiler-side prefix (ids, per-edit clause positions, desired values) + evaluator-side queries (suffix ids, labels, records) for one
    scene under one edit program / prefix order / wording.  `world` overrides the rendered records (natural counterfactual prefixes)."""
    text, spans = Q.prefix_text(s, proc, order, variant, world=world, correction=correction); pre = bb.prefix_ids(text)
    edits = P_programs(s)[program]; addresses = []; values = []
    for e in edits:
        addresses.append(tuple(bb.clause_positions(text, spans[(e.actor, e.project)]))); values.append(int(e.value))
    final = Q.edited_world(s, program); qs = []
    for draw in (0, 1):
        for q in Q.questions(s, split, draw, world=final):
            enc = bb.encode(text, Q.suffix_text(q['text'])); qs.append(dict(q=q, suffix=enc['suffix_ids'], labels=bb.label_ids(enc['full_text'], q['labels']), rec=Q.query_record(s, q, split, draw, order, variant, program)))
    req = Q.P.CompileRequest(tuple(pre), tuple(addresses), tuple(values), 'development')
    return dict(scene=s, split=split, program=program, order=order, variant=variant, proc=proc, text=text, prefix=pre, clause=addresses[0], addresses=addresses, values=values, queries=qs,
                world=final, request_key=req.key(), n_prefix=len(pre), touched=[(e.actor, e.project) for e in edits], correction=correction)

def P_programs(s): return Q.P.programs(s)

def bundles(bb, scene_list, split, proc, noop=True):
    out = [scene_bundle(bb, s, split, proc) for s in scene_list]
    if noop: out += [scene_bundle(bb, s, split, proc, program='noop') for s in Q.noop_scenes(scene_list)]
    return out

def footprint_index(cp, n_prefix, arm): return list(Q.P.token_mask(cp, n_prefix, Q.FOOTPRINT[arm]))

# ---------------------------------------------------------------- edit plans
def plan_for(bank, b, arm, strength=1.0, q_batch=None, record=True):
    """Ordered maps (+ per-map index) for one bundle's edit program.  q_batch: None (shared) or [B, d] array -> batched per-question prefix."""
    import torch
    B = 1 if q_batch is None else int(len(q_batch)); maps = []; rows_all = []; pos_all = []
    for cp, v in zip(b['addresses'], b['values']):
        idx = footprint_index(cp, b['n_prefix'], arm)
        rows = [r for r in range(B) for _ in idx]; pos = idx*B
        q_sel = None if q_batch is None else np.repeat(np.asarray(q_batch, np.float32), len(idx), axis=0)
        fn = bank.fn(v, strength, q=q_sel, record=record) if isinstance(bank, Bank) else bank.fn(v, strength)
        maps.append(dict(fn=fn, clause_positions=list(cp), index=(torch.as_tensor(rows, dtype=torch.long, device=bank.bb.device), torch.as_tensor(pos, dtype=torch.long, device=bank.bb.device))))
        rows_all += rows; pos_all += pos
    union = sorted(set(zip(rows_all, pos_all)))
    return dict(maps=maps, index=[p for _, p in union], rows=[r for r, _ in union], site=bank.site, batch=B)

def compile_plan(bb, b, pl):
    return bb.compile(b['prefix'], maps=pl['maps'], index=pl['index'], site=pl['site'], batch=pl['batch'], rows=pl['rows'] if pl['batch'] > 1 else None)

# ---------------------------------------------------------------- query features
class QStore:
    """Frozen query features: query_id -> float32[d] final-token state at the site from an UNEDITED source+question forward."""
    def __init__(self, path=None):
        self.q = {}; self.path = path
        if path and Path(path).is_file():
            with np.load(path, allow_pickle=False) as z: self.q = {k: z[k] for k in z.files}
    def compute(self, bb, bundle_list, site, deadline=None):
        for b in bundle_list:
            missing = [qq for qq in b['queries'] if qq['rec']['query_id'] not in self.q]
            if not missing: continue
            if deadline: Q.budget(deadline, 120)
            art = bb.compile(b['prefix'])
            for k in range(0, len(missing), 12):
                chunk = missing[k:k+12]; feats = bb.ask_capture_batch(art, [qq['suffix'] for qq in chunk], site)
                for qq, f in zip(chunk, feats): self.q[qq['rec']['query_id']] = f.astype('<f4')
    def batch(self, bundle_queries): return np.stack([self.q[qq['rec']['query_id']] for qq in bundle_queries])
    def save(self, path): np.savez(path, **self.q); self.path = path; return Q.sha(path)

# ---------------------------------------------------------------- evaluation
def evaluate(bb, bundle_list, arm, bank, strength, source_preds, qstore=None, seed=0, tag='', deadline=None, chunk=12, site=None):
    """Every query of every bundle.  Shared arms: one compiled artifact per bundle (hashed), batched clone asks.  Aware arms: one recompiled
    prefix per question (batched rows), q from the frozen store or computed on the fly from the unedited source artifact."""
    rng = random.Random(f'{seed}|{tag}'); order = list(range(len(bundle_list))); rng.shuffle(order); rows = []; norms = []; energies = []; frob = []; edited = []; bytes_ = []; losses = []; t0 = time.monotonic()
    aware = isinstance(bank, Bank) and bank.aware
    for i in order:
        b = bundle_list[i]
        if deadline: Q.budget(deadline, 120)
        qs = b['queries']
        if not aware:
            pl = plan_for(bank, b, arm, strength, record=False); art = compile_plan(bb, b, pl); r = art['realized']
            norms.extend(x for xs in r['realized_norms'] for x in xs); energies.append(r['energy']); frob.append(r['frobenius']); edited.append(len(pl['index'])); bytes_.append(art['bytes'])
            res = []
            for k in range(0, len(qs), chunk): res += bb.ask_batch(art, [qq['suffix'] for qq in qs[k:k+chunk]], [qq['labels'] for qq in qs[k:k+chunk]])
            hashes = [art['hash']]*len(qs); per_energy = [r['energy']]*len(qs)
        else:
            if qstore is not None and all(qq['rec']['query_id'] in qstore.q for qq in qs): qb = qstore.batch(qs)
            else: src = bb.compile(b['prefix']); qb = np.concatenate([bb.ask_capture_batch(src, [qq['suffix'] for qq in qs[k:k+12]], site or bank.site) for k in range(0, len(qs), 12)])
            res = []; hashes = []; per_energy = []
            for k in range(0, len(qs), 12):
                sub = qs[k:k+12]; pl = plan_for(bank, b, arm, strength, q_batch=qb[k:k+12], record=False); art = compile_plan(bb, b, pl); r = art['realized']
                res += bb.ask_batch(art, [qq['suffix'] for qq in sub], [qq['labels'] for qq in sub]); hashes += [None]*len(sub); per_energy += r['row_energy'][:len(sub)]
                norms.extend(x for xs in r['realized_norms'] for x in xs); energies.extend(r['row_energy'][:len(sub)]); frob.append(r['frobenius']/math.sqrt(len(sub))); edited.append(len(pl['index'])//len(sub))
        for qq, rr, hsh, en in zip(qs, res, hashes, per_energy):
            rec = dict(qq['rec']); sp = source_preds[rec['query_id']]
            rec.update(prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], source_prediction=sp['prediction'], source_logps=sp['logps'], gold_logp=rr['logps'][rec['gold']],
                       argmax_in_labels=rr['argmax_in_labels'], tie=rr['tie'], method=arm, seed=getattr(bank, 'seed', None), strength=strength, artifact_hash=hsh, aware=aware, energy=en, request_key=b['request_key'])
            losses.append(-rec['gold_logp']); rows.append(rec)
    summ = Q.summarize(rows); summ.update(loss=float(np.mean(losses)), mean_realized_norm=float(np.mean(norms)) if norms else 0.0, mean_energy=float(np.mean(energies)) if energies else 0.0,
                                         mean_frobenius=float(np.mean(frob)) if frob else 0.0, edited_tokens_per_artifact=float(np.mean(edited)) if edited else 0.0,
                                         cache_bytes=float(np.mean(bytes_)) if bytes_ else None, seconds=time.monotonic()-t0, n_rows=len(rows), strength=strength, arm=arm)
    return rows, summ

def natural_eval(bb, bundle_list, out, tag, world_key=None, source_preds=None, chunk=12, deadline=None):
    """No-edit predictions on the bundle's own prefix (source) or on a natural prefix rendering `world` (counterfactual / text correction)."""
    path = out/f'{tag}.jsonl'
    if done(path): return {r['query_id']: r for r in Q.read_rows(path)}
    rows = []
    for b in bundle_list:
        if deadline: Q.budget(deadline, 120)
        if world_key is None: text = b['text']; art = bb.compile(b['prefix'])
        else:
            text, _ = Q.prefix_text(b['scene'], b['proc'], b['order'], b['variant'], world=b['world'] if world_key == 'world' else None, correction=(world_key == 'correction')); art = bb.compile(bb.prefix_ids(text))
        qs = b['queries']; suff = [bb.encode(text, Q.suffix_text(qq['q']['text']))['suffix_ids'] for qq in qs] if world_key else [qq['suffix'] for qq in qs]
        res = []
        for k in range(0, len(qs), chunk): res += bb.ask_batch(art, suff[k:k+chunk], [qq['labels'] for qq in qs[k:k+chunk]])
        for qq, rr in zip(qs, res):
            rec = dict(qq['rec']); rec.update(prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], argmax_in_labels=rr['argmax_in_labels'], artifact_hash=art['hash'], method=tag.upper(),
                       source_prediction=(source_preds[rec['query_id']]['prediction'] if source_preds else rr['prediction']), gold_logp=rr['logps'][rec['gold']], prefix_tokens=art['prefix_len'])
            rows.append(rec)
    Q.write_rows(path, rows); return {r['query_id']: r for r in rows}

# ---------------------------------------------------------------- D0 captures
def captures(bb, fit_b, cal_b, out, model, deadline):
    if done(out/'CAPTURES.json'): return Q.load(out/'CAPTURES.json')
    site = model['site']; z_src = []; z_dst = []; mu_sum = None; mu_n = 0
    for b in [x for x in fit_b if x['program'] == 'single']:
        Q.budget(deadline, 120); a = bb.compile(b['prefix'], capture={site}); h = a['captured'][site][1:].float()
        mu_sum = h.sum(0).cpu().numpy() if mu_sum is None else mu_sum+h.sum(0).cpu().numpy(); mu_n += b['n_prefix']-1
        z_src.append(bb.z_from_artifact(a, site, list(b['clause'])).cpu().numpy())
        ttext, tsp = Q.prefix_text(b['scene'], b['proc'], world=b['world']); tpre = bb.prefix_ids(ttext); tcp = bb.clause_positions(ttext, tsp[b['touched'][0]]); c = bb.compile(tpre, capture={site})
        z_dst.append(bb.z_from_artifact(c, site, tcp).cpu().numpy())
    d = np.array([np.linalg.norm(x-y) for x, y in zip(z_dst, z_src)]); mu = (mu_sum/mu_n).astype('<f4')
    qs = QStore(); qs.compute(bb, fit_b, site, deadline); fit_ids = {qq['rec']['query_id'] for b in fit_b for qq in b['queries']}
    mu_q = np.mean([qs.q[k] for k in fit_ids], axis=0).astype('<f4'); qs.compute(bb, cal_b, site, deadline); qsha = qs.save(out/'qstore.npz')
    cap_new = float(2*np.quantile(d, .95)); cap = Q.GEMMA_CAP if model['key'] == 'gemma' else cap_new
    np.savez(out/'captures.npz', mu=mu, mu_q=mu_q, z_src=np.stack(z_src).astype('<f4'), z_dst=np.stack(z_dst).astype('<f4'))
    rep = dict(at=Q.now(), actor=model['key'], site=site, cap=cap, cap_rule=model['cap_rule'], cap_new_fit_rule=cap_new, delta_stats=dict(p95=float(np.quantile(d, .95)), median=float(np.median(d)), mean=float(d.mean()), n=int(len(d))),
               mu_rule='frozen FIT source mean over all prefix token states except position 0 (Qwen center); Gemma arms keep the stored V1 center for exact warm-start function preservation',
               mu_q_rule='frozen FIT mean of the final source-question state from unedited source+question forwards (query feature block only)', mu_norm=float(np.linalg.norm(mu)), mu_q_norm=float(np.linalg.norm(mu_q)),
               native_delta_scale=dict(z_src_norm=float(np.mean([np.linalg.norm(x) for x in z_src])), z_dst_minus_src=float(d.mean())), q_store=dict(queries=len(qs.q), sha256=qsha), captures_sha256=Q.sha(out/'captures.npz'), scenes=len(z_src))
    Q.dump(out/'CAPTURES.json', rep); return rep

def load_captures(out):
    with np.load(out/'captures.npz', allow_pickle=False) as z: return Q.load(out/'CAPTURES.json'), z['mu'].copy(), z['mu_q'].copy()

def make_bank(bb, arm, seed, model, cap, mu, mu_q, v1_dirs):
    if model['key'] == 'gemma': return Bank.from_v1(bb, arm, model['site'], cap, seed, 'gemma', v1_dirs[seed], mu_q)
    return Bank.fresh(bb, arm, model['site'], cap, seed, model['key'], mu, mu_q)

# ---------------------------------------------------------------- D1 training
def groups_for(bundle_list, seed):
    out = []
    for i, b in enumerate(bundle_list):
        qs = b['queries']; ch = [k for k, qq in enumerate(qs) if qq['rec']['changed']]; inv = [k for k, qq in enumerate(qs) if not qq['rec']['changed']]
        w = np.zeros(len(qs))
        if ch and inv: w[ch] = .5/len(ch); w[inv] = .5/len(inv)
        else: w[:] = 1.0/len(qs)                       # no-op bundles: invariant loss only, all questions invariant
        idx = list(range(len(qs))); random.Random(f'{seed}|{b["scene"].scene_id}|{b["program"]}').shuffle(idx)
        for k in range(0, len(idx), Q.GROUP): out.append(dict(bundle=i, queries=idx[k:k+Q.GROUP], weights=[float(w[j]) for j in idx[k:k+Q.GROUP]]))
    return out

def train_epochs(bb, bank, arm, fit_b, cal_b, lr, seed, epochs, cdir, src_fit, src_cal, qstore, deadline, log, out, start_epoch=1, opt=None, curve=None):
    import torch
    opt = opt or torch.optim.AdamW(bank.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0); curve = curve if curve is not None else []
    name = f'{arm}_lr{lr}_s{seed}'; aware = bank.aware; visits = 0; steps = 0
    for epoch in range(start_epoch, epochs+1):
        groups = groups_for(fit_b, seed*1000+epoch); random.Random(f'order|{seed}|{epoch}').shuffle(groups); t0 = time.monotonic(); losses = []; sat0 = dict(bank.stats)
        for g in groups:
            Q.budget(deadline, 120); b = fit_b[g['bundle']]; qs = [b['queries'][k] for k in g['queries']]
            if aware: qb = qstore.batch(qs); pl = plan_for(bank, b, arm, 1.0, q_batch=qb, record=True); r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, prefix_rows=pl['rows'], prefix_batch=len(qs))
            else: pl = plan_for(bank, b, arm, 1.0, record=True); r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True)
            lp = torch.stack([r['label_logps'][k][qs[k]['rec']['gold']] for k in range(len(qs))])
            loss = -(torch.as_tensor(g['weights'], device=lp.device, dtype=lp.dtype)*lp).sum()
            bb.backward(loss); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss)); visits += len(qs); steps += 1
        train_s = time.monotonic()-t0; sat = (bank.stats['saturated']-sat0['saturated'])/max(1, bank.stats['updates']-sat0['updates'])
        ck = cdir/f'epoch{epoch}'; bank.save(ck, epoch=epoch, lr=lr, steps=steps, visits=visits, frobenius=bank.frobenius())
        t1 = time.monotonic(); fit_rows, fit_sum = evaluate(bb, fit_b, arm, bank, 1.0, src_fit, qstore, seed=seed, tag=f'{name}|fit{epoch}', deadline=deadline)
        Q.write_rows(ck/'fit_rows.jsonl', fit_rows); fit_sum.pop('per_scene', None)
        for strength in Q.STRENGTHS:
            cal_rows, cal_sum = evaluate(bb, cal_b, arm, bank, strength, src_cal, qstore, seed=seed, tag=f'{name}|cal{epoch}|{strength}', deadline=deadline)
            Q.write_rows(ck/f'cal_rows_m{strength}.jsonl', cal_rows); ps = cal_sum.pop('per_scene')
            curve.append(dict(arm=arm, lr=lr, seed=seed, epoch=epoch, strength=strength, checkpoint=str(ck), cal=cal_sum, fit=fit_sum, feasible=Q.feasible(cal_sum), frontier=Q.frontier_score(cal_sum),
                              norm=cal_sum['mean_realized_norm'], frobenius=cal_sum['mean_frobenius'], saturation=sat, loss_train=float(np.mean(losses)), visits=visits, steps=steps, train_seconds=train_s, eval_seconds=time.monotonic()-t1))
        best = max((c['frontier'] for c in curve if c['epoch'] == epoch), default=None)
        log(out, 'EPOCH', candidate=name, epoch=epoch, train_loss=float(np.mean(losses)), fit_changed=fit_sum['changed'], fit_harm=fit_sum['harm'], cal_best_frontier=best, saturation=sat, train_seconds=train_s, eval_seconds=time.monotonic()-t1)
        Q.dump(cdir/f'CURVE_epoch{epoch}.json', dict(at=Q.now(), curve=[c for c in curve if c['epoch'] == epoch]))
    return opt, curve

def epoch_frontier(curve, epoch): return max(c['frontier'] for c in curve if c['epoch'] == epoch)

def unchanged_from_init(ckpt, v1_dir):
    """True when the selected factors are numerically identical to the warm start (a retained capability, not a learned improvement)."""
    for v in (0, 1):
        with np.load(Path(ckpt)/f'relation_{v}.npz', allow_pickle=False) as a, np.load(Path(v1_dir)/f'relation_{v}.npz', allow_pickle=False) as b:
            C = np.asarray(b['input_factor'], np.float32); Cp = np.concatenate((C, np.zeros((C.shape[0]//2, C.shape[1]), np.float32)), axis=0)
            if not (np.array_equal(np.asarray(a['input_factor'], np.float32), Cp) and np.array_equal(np.asarray(a['output_factor'], np.float32), np.asarray(b['output_factor'], np.float32)) and np.array_equal(np.asarray(a['bias'], np.float32), np.asarray(b['bias'], np.float32))): return False
    return True

def develop_arm(bb, arm, fit_b, cal_b, model, cap, mu, mu_q, v1_dirs, e_max, out, src_fit, src_cal, qstore, deadline, log):
    """Seed 0: both LRs for LR_EPOCHS; better CAL LR extended to <= e_max with patience; seed 1 at that LR with the same epochs.  Then the
    declared two-seed selection.  Re-enterable per arm via DONE marker (an interrupted arm restarts from initialization; noted)."""
    adir = out/'arms'/arm
    if done(adir/'ARM_DONE.json'): return Q.load(adir/'ARM_DONE.json')
    if adir.exists() and any(adir.iterdir()):
        import shutil; aside = adir.parent/f'{arm}_partial_{int(time.time())}'; shutil.move(str(adir), str(aside)); log(out, 'ARM_RESTART', arm=arm, moved_partial_to=str(aside), note='an interrupted arm restarts from its initialization; the partial run is retained, not merged')
    adir.mkdir(parents=True, exist_ok=True); runs = {}
    for lr in Q.LRS:
        bank = make_bank(bb, arm, 0, model, cap, mu, mu_q, v1_dirs); cdir = adir/f'lr{lr}_s0'
        if done(cdir/'DONE.json'): runs[lr] = Q.load(cdir/'DONE.json'); continue
        opt, curve = train_epochs(bb, bank, arm, fit_b, cal_b, lr, 0, Q.LR_EPOCHS, cdir, src_fit, src_cal, qstore, deadline, log, out)
        runs[lr] = dict(lr=lr, seed=0, curve=curve, epochs_run=Q.LR_EPOCHS, bank=bank, opt=opt, init=bank.init)
    choice = max(Q.LRS, key=lambda lr: (max(c['frontier'] for c in runs[lr]['curve']), -lr)); log(out, 'LR_CHOICE', arm=arm, lr=choice, frontiers={str(lr): max(c['frontier'] for c in runs[lr]['curve']) for lr in Q.LRS})
    run = runs[choice]; cdir = adir/f'lr{choice}_s0'
    if not done(cdir/'DONE.json'):
        bank, opt, curve = run['bank'], run['opt'], run['curve']; best = max(epoch_frontier(curve, e) for e in range(1, Q.LR_EPOCHS+1)); best_epoch = max(range(1, Q.LR_EPOCHS+1), key=lambda e: epoch_frontier(curve, e)); no_improve = Q.LR_EPOCHS-best_epoch
        epochs_run = Q.LR_EPOCHS
        for epoch in range(Q.LR_EPOCHS+1, e_max+1):
            if no_improve >= Q.PATIENCE: log(out, 'EXTENSION_STOP', arm=arm, epoch=epoch-1, rule='two complete epochs without CAL frontier improvement'); break
            if Q.budget(deadline, 0) < 1200: log(out, 'DEV_DEADLINE', arm=arm, note='extension cut by the development allocation; later epochs NOT_MEASURED'); break
            opt, curve = train_epochs(bb, bank, arm, fit_b, cal_b, choice, 0, epoch, cdir, src_fit, src_cal, qstore, deadline, log, out, start_epoch=epoch, opt=opt, curve=curve); epochs_run = epoch
            f = epoch_frontier(curve, epoch)
            if f > best: best = f; no_improve = 0
            else: no_improve += 1
        Q.dump(cdir/'DONE.json', dict(at=Q.now(), arm=arm, lr=choice, seed=0, epochs_run=epochs_run, curve=curve, init=run['init'], stats=bank.stats)); runs[choice] = Q.load(cdir/'DONE.json')
        for lr in Q.LRS:
            if lr != choice and not done(adir/f'lr{lr}_s0'/'DONE.json'): Q.dump(adir/f'lr{lr}_s0'/'DONE.json', dict(at=Q.now(), arm=arm, lr=lr, seed=0, epochs_run=Q.LR_EPOCHS, curve=runs[lr]['curve'], init=runs[lr]['init'], not_extended=True))
    s0 = Q.load(cdir/'DONE.json'); epochs0 = s0['epochs_run']; cdir1 = adir/f'lr{choice}_s1'
    if not done(cdir1/'DONE.json'):
        bank1 = make_bank(bb, arm, 1, model, cap, mu, mu_q, v1_dirs); _, curve1 = train_epochs(bb, bank1, arm, fit_b, cal_b, choice, 1, epochs0, cdir1, src_fit, src_cal, qstore, deadline, log, out)
        Q.dump(cdir1/'DONE.json', dict(at=Q.now(), arm=arm, lr=choice, seed=1, epochs_run=epochs0, curve=curve1, init=bank1.init, stats=bank1.stats))
    s1 = Q.load(cdir1/'DONE.json')
    cands = []
    for epoch in range(1, min(epochs0, s1['epochs_run'])+1):
        for strength in Q.STRENGTHS:
            c0 = next(c for c in s0['curve'] if c['epoch'] == epoch and c['strength'] == strength); c1 = next(c for c in s1['curve'] if c['epoch'] == epoch and c['strength'] == strength)
            cands.append(dict(lr=choice, epoch=epoch, strength=strength, seeds={0: c0['cal'], 1: c1['cal']}, norm=float(np.mean([c0['norm'], c1['norm']])), frobenius=float(np.mean([c0['frobenius'], c1['frobenius']])),
                              checkpoints={0: c0['checkpoint'], 1: c1['checkpoint']}, fit={0: c0['fit'], 1: c1['fit']}))
    sel = Q.select_recipe(cands); sel['cal'] = {s: next(c for c in cands if c['epoch'] == sel['epoch'] and c['strength'] == sel['strength'])['seeds'][s] for s in (0, 1)}
    # retained-capability label: an unchanged warm start (no positive step moved the weights) is not a newly learned improvement
    sel['warm_start_unchanged'] = {s: unchanged_from_init(sel['checkpoints'][s], v1_dirs[s]) for s in (0, 1)} if v1_dirs else None
    rep = dict(at=Q.now(), arm=arm, lr_choice=choice, epochs_seed0=epochs0, epochs_seed1=s1['epochs_run'], selection=sel, candidates=[{k: v for k, v in c.items() if k != 'fit'} for c in cands],
               opportunity=dict(visits_seed0=s0['curve'][-1]['visits'], steps_seed0=s0['curve'][-1]['steps'], visits_seed1=s1['curve'][-1]['visits']), init=s0['init'])
    Q.dump(adir/'ARM_DONE.json', rep); log(out, 'ARM_SELECTED', arm=arm, lr=choice, epoch=sel['epoch'], strength=sel['strength'], feasible=sel['feasible'], label=sel['label'],
                                       worse_changed=min(sel['cal'][s]['changed'] for s in (0, 1)), worse_harm=max(sel['cal'][s]['harm'] for s in (0, 1)))
    return rep

# ---------------------------------------------------------------- phase D driver
def epoch_plan(bench, arms, fit_b, cal_b, dev_seconds):
    n_groups = sum(math.ceil(len(b['queries'])/Q.GROUP) for b in fit_b); n_fit = sum(len(b['queries']) for b in fit_b); n_cal = sum(len(b['queries']) for b in cal_b)
    per = {}
    for arm in arms:
        aware = Q.AWARE[arm]; step = bench[f'train_step4_{"aware" if aware else "shared"}']
        ev = ((len(fit_b)+4*len(cal_b))*bench['compile']+(n_fit+4*n_cal)/24*bench['ask_batch24']) if not aware else ((n_fit+4*n_cal)/12*bench['compile_aware12']+(n_fit+4*n_cal)/12*bench['ask_batch12'])
        per[arm] = dict(train=n_groups*step*1.1, eval=ev*1.15)
    def total(e_max): return sum((Q.LR_EPOCHS+2*e_max)*(v['train']+v['eval']) for v in per.values())
    e_max = max([e for e in range(Q.LR_EPOCHS, Q.MAX_EPOCHS+1) if total(e) <= dev_seconds] or [Q.LR_EPOCHS])
    return dict(e_max=e_max, per_arm_epoch_seconds=per, predicted_total_seconds=total(e_max), groups_per_epoch=n_groups, dev_seconds=dev_seconds,
                rule='E_max = largest e in [LR_EPOCHS, MAX_EPOCHS] with sum over arms of (LR_EPOCHS + 2e) epoch-times <= development allocation, fixed from the disposable benchmark before any training outcome')

AMENDMENT = Q.REPORT/'NATIVE_QUALIFICATION_AMENDMENT_V1.md'

def amended_qualification(qual, qdir, forced_proc):
    """NATIVE_QUALIFICATION_AMENDMENT_V1 (2026-09-10): the original qualification stays FAILED; Gemma continues under the forced procedure only if
    parse >= 95% and direct/negation >= 90% hold on BOTH worlds in the corrected battery and the interface and gradient checks pass."""
    att = next((a for a in qual['attempts'] if a['procedure'] == forced_proc), None); ic = Q.load(qdir/'INTERFACE_CHECK.json') if (qdir/'INTERFACE_CHECK.json').is_file() else None; gc = Q.load(qdir/'GRADIENT_CHECK.json') if (qdir/'GRADIENT_CHECK.json').is_file() else None
    checks = {}
    for w in ('source', 'counterfactual'):
        sm = (att or {}).get('summary', {}).get(w, {}); checks[f'{w}_parse_ge_95'] = bool(sm and sm['parse_rate'] >= Q.QUAL_PARSE); checks[f'{w}_direct_negation_ge_90'] = bool(sm and sm['direct_negation_accuracy'] >= Q.QUAL_DIRECT)
    checks['interface_pass'] = bool(ic and ic['pass_']); checks['gradient_pass'] = bool(gc and gc['pass_'])
    return dict(status=('AMENDED_'+forced_proc+'_ORIGINAL_FAILED') if all(checks.values()) else 'AMENDMENT_CONDITIONS_NOT_MET', original_qualification='FAILED', forced_procedure=forced_proc, checks=checks,
                equality={w: (att or {}).get('summary', {}).get(w, {}).get('equality_accuracy') for w in ('source', 'counterfactual')}, amendment_file=str(AMENDMENT), amendment_sha256=Q.sha(AMENDMENT) if AMENDMENT.is_file() else None,
                note='chosen after inspecting native CAL performance (r1/r2); no new-study editor outcome existed before it; the 85% equality bar is unchanged and the original result stays FAILED')

def phase_d(bb, out, deadline, model, fit, cal, qdir, arms, dev_hours, v1_dirs, forced_proc=None):
    qual = Q.load(qdir/'QUALIFICATION.json'); proc = qual['selected_procedure']; bench = Q.load(qdir/'BENCHMARK.json')['seconds']; qstatus = dict(status='ORIGINAL_PASS', procedure=proc)
    if not proc and forced_proc:
        qstatus = amended_qualification(qual, qdir, forced_proc)
        if not done(out/'QUALIFICATION_STATUS.json'): Q.dump(out/'QUALIFICATION_STATUS.json', dict(at=Q.now(), actor=model['key'], **qstatus))
        if qstatus['status'].startswith('AMENDED'): proc = forced_proc; event(out, 'AMENDED_QUALIFICATION', actor=model['key'], procedure=proc, checks=qstatus['checks'], equality=qstatus['equality'])
        else: Q.dump(out/'DEV_STOP.json', dict(at=Q.now(), status='AMENDMENT_CONDITIONS_NOT_MET', detail=qstatus)); return dict(status='MODEL_UNQUALIFIED', detail=qstatus)
    if not proc: Q.dump(out/'DEV_STOP.json', dict(at=Q.now(), status='MODEL_UNQUALIFIED', qualification=qual)); return dict(status='MODEL_UNQUALIFIED')
    log = event; dev_s = min(Q.budget(deadline, 0), dev_hours*3600); dev_deadline = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=dev_s)).isoformat()
    log(out, 'DEV_START', procedure=proc, arms=arms, dev_deadline=dev_deadline, dev_hours=dev_hours)
    fit_b = bundles(bb, fit, 'FIT', proc); cal_b = bundles(bb, cal, 'CAL', proc)
    for b in fit_b+cal_b: b['proc'] = proc
    log(out, 'BUNDLES', fit=len(fit_b), cal=len(cal_b), fit_noop=sum(b['program'] == 'noop' for b in fit_b), cal_noop=sum(b['program'] == 'noop' for b in cal_b), fit_queries=sum(len(b['queries']) for b in fit_b))
    src_fit = natural_eval(bb, fit_b, out, 'source_fit', deadline=deadline); src_cal = natural_eval(bb, cal_b, out, 'source_cal', deadline=deadline)
    cf_cal = natural_eval(bb, [b for b in cal_b if b['program'] == 'single'], out, 'counterfactual_cal', world_key='world', source_preds=src_cal, deadline=deadline)
    base = Q.summarize(list(src_fit.values())); base.pop('per_scene'); cal_base = Q.summarize(list(src_cal.values())); cal_base.pop('per_scene'); cf_base = Q.summarize(list(cf_cal.values())); cf_base.pop('per_scene')
    if not done(out/'BASELINES.json'): Q.dump(out/'BASELINES.json', dict(at=Q.now(), no_edit_fit=base, no_edit_cal=cal_base, counterfactual_cal=cf_base, note='no-edit = the zero-functional behaviour (Qwen init); counterfactual = natural destination-prefix reference'))
    cap = captures(bb, fit_b, cal_b, out, model, deadline); capd, mu, mu_q = load_captures(out); qstore = QStore(out/'qstore.npz'); log(out, 'CAPTURES', cap=cap['cap'], cap_new_rule=cap['cap_new_fit_rule'], mu_q_norm=cap['mu_q_norm'], q_store=cap['q_store']['queries'])
    plan = epoch_plan(bench, arms, fit_b, cal_b, Q.budget(dev_deadline, 0)); plan['at'] = Q.now(); plan['arms'] = list(arms)
    if not done(out/'DEV_PLAN.json'): Q.dump(out/'DEV_PLAN.json', plan)
    plan = Q.load(out/'DEV_PLAN.json'); log(out, 'DEV_PLAN', e_max=plan['e_max'], predicted=plan['predicted_total_seconds'], dev_seconds=plan['dev_seconds'])
    results = {}
    for arm in arms:
        if Q.budget(deadline, 0) < 2400: log(out, 'DEV_DEADLINE', arm=arm, note='NOT_MEASURED: development deadline reached before this arm'); continue
        results[arm] = develop_arm(bb, arm, fit_b, cal_b, model, cap['cap'], mu, mu_q, v1_dirs, plan['e_max'], out, src_fit, src_cal, qstore, dev_deadline if Q.budget(dev_deadline, 0) > 0 else deadline, log)
    freeze = dict(at=Q.now(), actor=model['key'], procedure=proc, qualification_status=qstatus, site=model['site'], cap=cap['cap'], arms=list(arms), e_max=plan['e_max'],
                  selection={a: dict(lr=r['selection']['lr'], epoch=r['selection']['epoch'], strength=r['selection']['strength'], feasible=r['selection']['feasible'], label=r['selection']['label'], checkpoints=r['selection']['checkpoints'],
                                     cal=r['selection']['cal'], norm=r['selection']['norm'], frobenius=r['selection']['frobenius']) for a, r in results.items()},
                  not_measured=[a for a in arms if a not in results], captures_sha256=cap['captures_sha256'], q_store_sha256=cap['q_store']['sha256'], v1_dirs={str(k): str(v) for k, v in (v1_dirs or {}).items()},
                  final_access='none: no FINAL or WORKFLOW scene was read in phase D', rule='per arm: worse-seed changed accuracy among recipes feasible in each seed (<=5% harm, >=80% each direction), ties higher all-24, lower harm, smaller norm, earlier epoch; else diagnostic by worse-seed changed - 2*harm')
    if not done(out/'FREEZE.json'): Q.dump(out/'FREEZE.json', freeze)
    log(out, 'FREEZE', selection={a: (v['lr'], v['epoch'], v['strength'], v['feasible']) for a, v in freeze['selection'].items()})
    return dict(status='DEVELOPMENT_COMPLETE', selection=freeze['selection'], not_measured=freeze['not_measured'])
