"""RSO3 phase D: single-operation development of the matched arms on FIT/CAL only, then FREEZE before any FINAL/WORKFLOW access.

D0 captures  per actor: FIT source prefixes compiled question-blind at the site -> source-clause states, clause-summary displacements to the
             natural counterfactual clause (Qwen cap_ref = 2 x p95; Gemma keeps the V2 per-token reference 103.19), the common fixed center
             mu (Gemma: the V2 stored center; Qwen: FIT source mean), U initializations (Gemma: SVD of the concatenated V2 clause output
             factors per seed; Qwen: SVD of FIT paired-development clause-summary differences), head warm starts (Gemma: ridge toward the
             incumbent V2 clause edit's projected coordinates on FIT source-clause states; Qwen: FREE identity / INVARIANT zero) and the
             reference nesting test.  Nothing here reads a question, label or gold.
D1 training  per arm: seed 0 at both LRs for LR_EPOCHS complete epochs, the better CAL LR extended to at most E_max (<= 4) epochs with
             patience 2, seed 1 at the chosen LR for the same epochs.  AdamW (.9,.999) eps 1e-8 wd 0, grad clip 1, groups of four questions
             from one single-operation bundle (flip or declared no-op), changed/invariant balanced within a group, full-vocabulary
             target-label NLL, plus beta * mean(||delta||^2 / cap_ref^2) for both setters.  Complete FIT (strength 1) and CAL (setters at
             strength 1; affine at the V2 grid) evaluated at every epoch boundary through the cached serving path.
D2 selection per arm by the declared two-seed rule on CAL singles/no-ops -> FREEZE.json.  No multi-command result exists before FREEZE.
"""
from __future__ import annotations
import datetime as dt
import math
import random
import time
from pathlib import Path
import numpy as np
import rso3_common as Q
from rso3_editor import SetterBank, AffineBank, gemma_basis_init, svd_basis, warm_start_heads, free_identity_heads, canonical_program
from rso3_unit import event, done

# ---------------------------------------------------------------- encodings
def scene_bundle(bb, s, split, proc, program='single', order='original', variant=0):
    """Compiler-side prefix (ids, per-command clause positions, requested values) + evaluator-side queries for one scene/program/order."""
    text, spans = Q.prefix_text(s, proc, order, variant); pre = bb.prefix_ids(text); edits = Q.programs(s)[program]
    addresses = [tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in edits]; values = [int(e.value) for e in edits]
    world = Q.final_world(s, program); qs = []
    for draw in (0, 1):
        for q in Q.questions(s, split, draw, program):
            enc = bb.encode(text, Q.suffix_text(q['text'])); qs.append(dict(q=q, suffix=enc['suffix_ids'], labels=bb.label_ids(enc['full_text'], q['labels']), rec=Q.query_record(s, q, split, draw, order, variant, program)))
    req = Q.OLD.CompileRequest(tuple(pre), tuple(addresses), tuple(values), 'development')
    return dict(scene=s, split=split, program=program, order=order, variant=variant, proc=proc, text=text, prefix=pre, clause=addresses[0], addresses=addresses, values=values, queries=qs, edits=edits,
                world=world, request_key=req.key(), n_prefix=len(pre), touched=[(e.actor, e.project) for e in edits])

def bundles(bb, scene_list, split, proc, noop=True):
    out = [scene_bundle(bb, s, split, proc) for s in scene_list]
    if noop: out += [scene_bundle(bb, s, split, proc, program='noop') for s in Q.noop_scenes(scene_list)]
    return out

# ---------------------------------------------------------------- edit plans
def plan_for(bank, b, strength=1.0, record=True, workspace=None, footprint=None, canonical=False):
    """Ordered maps for one bundle's declared command program.  Every declared command becomes one map (no deduplication); only the
    separately labelled CANONICAL comparator normalizes (last write per address, sorted addresses) before building its maps."""
    import torch
    addresses, values = (canonical_program(b['addresses'], b['values']) if canonical else (b['addresses'], b['values']))
    setter = isinstance(bank, SetterBank); ws = workspace or ('fp32' if setter else 'legacy'); fp = footprint or ('clause' if setter else Q.S.FOOTPRINT[bank.arm]); maps = []; pos_all = []
    for cp, v in zip(addresses, values):
        idx = Q.token_mask(cp, b['n_prefix'], fp); it = (torch.zeros(len(idx), dtype=torch.long, device=bank.bb.device), torch.as_tensor(idx, dtype=torch.long, device=bank.bb.device))
        if setter: maps.append(dict(kind='setter', fn=bank.fn(v, 1.0, record=record), index=it, clause_positions=list(cp)))
        else: maps.append(dict(kind='affine', fn=bank.fn(v, strength, record=record), index=it, clause_positions=list(cp)))
        pos_all += idx
    union = sorted(set(pos_all))
    return dict(maps=maps, index=union, site=bank.site, workspace=ws, commands=len(maps), canonical=canonical, footprint=fp, strength=(1.0 if setter else strength))

def compile_plan(bb, b, pl, capture=(), capture_next=()):
    return bb.compile(b['prefix'], maps=pl['maps'], index=pl['index'], site=pl['site'], workspace=pl['workspace'], capture=capture, capture_next=capture_next)

# ---------------------------------------------------------------- evaluation (shared question-blind artifacts; batched clone asks)
def evaluate(bb, bundle_list, arm, bank, strength, source_preds, seed=0, tag='', deadline=None, chunk=Q.CHUNK, workspace=None):
    rng = random.Random(f'{seed}|{tag}'); order = list(range(len(bundle_list))); rng.shuffle(order); rows = []; norms = []; energies = []; frob = []; edited = []; bytes_ = []; losses = []; casts = []; t0 = time.monotonic()
    for i in order:
        b = bundle_list[i]
        if deadline: Q.budget(deadline, 120)
        qs = b['queries']; pl = plan_for(bank, b, strength, record=False, workspace=workspace); art = compile_plan(bb, b, pl); r = art['realized']
        norms.extend(r['realized_norms'][0]); energies.append(r['energy']); frob.append(r['frobenius']); edited.append(len(pl['index'])); bytes_.append(art['bytes']); casts.append(r['cast_error'])
        res = []
        for k in range(0, len(qs), chunk): res += bb.ask_batch(art, [qq['suffix'] for qq in qs[k:k+chunk]], [qq['labels'] for qq in qs[k:k+chunk]])
        for qq, rr in zip(qs, res):
            rec = dict(qq['rec']); sp = source_preds[rec['query_id']]
            rec.update(prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], source_prediction=sp['prediction'], source_logps=sp['logps'], gold_logp=rr['logps'][rec['gold']],
                       argmax_in_labels=rr['argmax_in_labels'], tie=rr['tie'], method=arm, seed=getattr(bank, 'seed', None), strength=pl['strength'], artifact_hash=art['hash'], energy=r['energy'], request_key=b['request_key'], workspace=pl['workspace'])
            losses.append(-rec['gold_logp']); rows.append(rec)
    summ = Q.summarize(rows); summ.update(loss=float(np.mean(losses)), mean_realized_norm=float(np.mean(norms)) if norms else 0.0, mean_energy=float(np.mean(energies)) if energies else 0.0, mean_frobenius=float(np.mean(frob)) if frob else 0.0,
                                         edited_tokens_per_artifact=float(np.mean(edited)) if edited else 0.0, cache_bytes=float(np.mean(bytes_)) if bytes_ else None, max_cast_error=float(max(casts)) if casts else 0.0,
                                         seconds=time.monotonic()-t0, n_rows=len(rows), strength=pl['strength'] if bundle_list else strength, arm=arm)
    return rows, summ

def natural_eval(bb, bundle_list, out, tag, world_key=None, source_preds=None, chunk=Q.CHUNK, deadline=None):
    """No-edit predictions on the bundle's own prefix (source) or on a natural prefix rendering the final world / corrections."""
    path = out/f'{tag}.jsonl'
    if done(path): return {r['query_id']: r for r in Q.read_rows(path)}
    rows = []
    for b in bundle_list:
        if deadline: Q.budget(deadline, 120)
        if world_key is None: text = b['text']; art = bb.compile(b['prefix'])
        else:
            text, _ = Q.prefix_text(b['scene'], b['proc'], b['order'], b['variant'], world=(b['world'] if world_key == 'world' else None), corrections=(b['edits'] if world_key == 'correction' else None)); art = bb.compile(bb.prefix_ids(text))
        qs = b['queries']; suff = [bb.encode(text, Q.suffix_text(qq['q']['text']))['suffix_ids'] for qq in qs] if world_key else [qq['suffix'] for qq in qs]
        res = []
        for k in range(0, len(qs), chunk): res += bb.ask_batch(art, suff[k:k+chunk], [qq['labels'] for qq in qs[k:k+chunk]])
        for qq, rr in zip(qs, res):
            rec = dict(qq['rec']); rec.update(prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], argmax_in_labels=rr['argmax_in_labels'], artifact_hash=art['hash'], method=tag.upper(),
                                              source_prediction=(source_preds[rec['query_id']]['prediction'] if source_preds else rr['prediction']), gold_logp=rr['logps'][rec['gold']], prefix_tokens=art['prefix_len'])
            rows.append(rec)
    Q.write_rows(path, rows); return {r['query_id']: r for r in rows}

# ---------------------------------------------------------------- D0 captures and initializations
def captures(bb, fit_b, out, model, deadline):
    if done(out/'CAPTURES.json'): return Q.load(out/'CAPTURES.json')
    site = model['site']; z_src = []; z_dst = []; mu_sum = None; mu_n = 0; clause_states = []; groups = []
    for b in fit_b:
        Q.budget(deadline, 120); a = bb.compile(b['prefix'], capture={site}); h = a['captured'][site].float()
        if b['program'] == 'single':
            mu_sum = h[1:].sum(0).cpu().numpy() if mu_sum is None else mu_sum+h[1:].sum(0).cpu().numpy(); mu_n += b['n_prefix']-1
            z_src.append(h[list(b['clause'])].mean(0).cpu().numpy())
            ttext, tsp = Q.prefix_text(b['scene'], b['proc'], world=b['world']); tpre = bb.prefix_ids(ttext); tcp = bb.clause_positions(ttext, tsp[b['touched'][0]]); c = bb.compile(tpre, capture={site})
            z_dst.append(c['captured'][site][tcp].float().mean(0).cpu().numpy())
        clause_states.append(h[list(b['clause'])].cpu().numpy().astype(np.float32)); groups.append(dict(scene_id=b['scene'].scene_id, program=b['program'], value=b['values'][0], tokens=len(b['clause'])))
    d = np.array([np.linalg.norm(x-y) for x, y in zip(z_dst, z_src)]); mu_fit = (mu_sum/mu_n).astype('<f4')
    if model['key'] == 'gemma' and bb.hidden == 3584:
        with np.load(Q.v2_factor_dir('SHARED_CLAUSE', 0)/'relation_0.npz', allow_pickle=False) as z: mu = np.asarray(z['center'], np.float32)
        mu_rule = 'the V2 stored center (identical across V2 arms/seeds/values): one common fixed source center for every Gemma method'; cap_ref = Q.GEMMA_CAP; cap_rule = 'V2 per-token reference 103.18592071533203'
    else:
        mu = mu_fit; mu_rule = 'frozen FIT source mean over all prefix token states except position 0'; cap_ref = float(2*np.quantile(d, .95)); cap_rule = '2 x p95 of native FIT clause-summary displacement ||z_dst - z_src||'
    np.savez(out/'captures.npz', mu=mu, mu_fit=mu_fit, z_src=np.stack(z_src).astype('<f4'), z_dst=np.stack(z_dst).astype('<f4'), **{f'clause_{i}': c for i, c in enumerate(clause_states)})
    Q.dump(out/'CAPTURE_GROUPS.json', groups)
    rep = dict(at=Q.now(), actor=model['key'], site=site, cap_ref=cap_ref, cap_rule=cap_rule, cap_new_fit_rule=float(2*np.quantile(d, .95)), delta_stats=dict(p95=float(np.quantile(d, .95)), median=float(np.median(d)), mean=float(d.mean()), n=int(len(d))),
               mu_rule=mu_rule, mu_norm=float(np.linalg.norm(mu)), mu_fit_norm=float(np.linalg.norm(mu_fit)), groups=len(groups), captures_sha256=Q.sha(out/'captures.npz'), inputs='FIT source prefixes and natural FIT counterfactual prefixes only; no question or label')
    Q.dump(out/'CAPTURES.json', rep); return rep

def load_captures(out):
    with np.load(out/'captures.npz', allow_pickle=False) as z: return Q.load(out/'CAPTURES.json'), z['mu'].copy(), [z[f'clause_{i}'].copy() for i in range(len(Q.load(out/'CAPTURE_GROUPS.json')))], z['z_src'].copy(), z['z_dst'].copy()

def initializations(bb, fit_b, out, model, cap_ref, mu, clause_states, z_src, z_dst, deadline):
    """Per seed: U (+ singular values) and warm-start heads for both setters, the approximation errors and the nesting test."""
    if done(out/'INIT.json'): return Q.load(out/'INIT.json')
    import torch
    site = model['site']; rep = dict(at=Q.now(), actor=model['key'], seeds={}); groups = Q.load(out/'CAPTURE_GROUPS.json')
    for seed in Q.SEEDS:
        if model['key'] == 'gemma' and bb.hidden == 3584:
            u, sv, urule = gemma_basis_init(seed); inc = AffineBank.from_v2(bb, 'SHARED_CLAUSE', seed); strength = Q.v2_strength('SHARED_CLAUSE'); edited = []
            for b, h in zip(fit_b, clause_states):        # the incumbent's edited clause states in the FP32 workspace (function of source states only)
                Q.budget(deadline, 120); pl = plan_for(inc, b, strength, record=False, workspace='fp32'); art = compile_plan(bb, b, pl); edited.append(art['_workspace_states'].astype(np.float32))
            heads = {}
            for arm in ('INVARIANT_SET', 'FREE_OVERWRITE'):
                kind = Q.KIND[arm]; A = np.zeros((2, 2*bb.hidden, Q.RANK), np.float32); c = np.zeros((2, Q.RANK), np.float32); err = {}; scale = {}
                for v in (0, 1):
                    sel = [i for i, g in enumerate(groups) if g['value'] == v]
                    A[v], c[v], scale[v], err[v] = warm_start_heads(kind, [clause_states[i] for i in sel], [edited[i] for i in sel], u.astype(np.float64), mu.astype(np.float64))
                heads[arm] = dict(head=A, bias=c, error=err, scale=scale, groups={v: int(sum(1 for g in groups if g['value'] == v)) for v in (0, 1)}, rule='ridge (1e-3 after one fixed feature scaling) toward the incumbent V2 clause edit projected coordinates (H_edit - mu) U on FIT source-clause states')
            hrule = 'warm start toward the incumbent (V2 SHARED_CLAUSE, FP32 workspace, strength 1.25); not additional final supervision'
        else:
            u, sv = svd_basis(np.asarray(z_dst, np.float64)-np.asarray(z_src, np.float64)); urule = dict(kind='svd_of_fit_paired_development_clause_summary_differences', rows=int(len(z_src)), singular_values=sv.tolist())
            A_free, c_free = free_identity_heads(u, bb.hidden)
            heads = {'FREE_OVERWRITE': dict(head=np.stack([A_free, A_free]), bias=np.stack([c_free, c_free]), error=None, scale=None, rule='zero-functional identity: F(H,Hbar) = (H-mu)U so the output equals H exactly'),
                     'INVARIANT_SET': dict(head=np.zeros((2, 2*bb.hidden, Q.RANK), np.float32), bias=np.zeros((2, Q.RANK), np.float32), error=None, scale=None, rule='zero heads: the initial map is the projection R + mu (not an identity)')}
            hrule = 'independent zero-functional initialization; no cross-model factor reuse'
        np.savez(out/f'init_seed{seed}.npz', u=u, singular_values=sv, **{f'{arm}_head': h['head'] for arm, h in heads.items()}, **{f'{arm}_bias': h['bias'] for arm, h in heads.items()})
        # numerical checks on real FIT clause states: orthonormality, nesting (free copy of the invariant function), identity/zero-functional behaviour
        inv = SetterBank(bb, 'INVARIANT_SET', site, seed, model['key'], cap_ref); inv.set_center(mu); inv.set_basis(u); inv.set_heads(heads['INVARIANT_SET']['head'], heads['INVARIANT_SET']['bias'])
        fre = SetterBank(bb, 'FREE_OVERWRITE', site, seed, model['key'], cap_ref); fre.set_center(mu); fre.set_basis(u); fre.set_heads(heads['FREE_OVERWRITE']['head'], heads['FREE_OVERWRITE']['bias'])
        nested = inv.nested_free_copy(); checks = dict(nest=0.0, idem=0.0, over=0.0, free_identity=0.0, inv_vs_free_init=0.0, orth=float((inv.basis().detach().T@inv.basis().detach()-torch.eye(Q.RANK, device=bb.device)).abs().max()))
        with torch.no_grad():
            for h in clause_states[:24]:
                x = torch.as_tensor(h, device=bb.device)
                for v in (0, 1):
                    a = inv.model(x, v); b_ = nested(x, v); checks['nest'] = max(checks['nest'], float((a-b_).abs().max()))
                    checks['idem'] = max(checks['idem'], float((inv.model(a, v)-a).abs().max())); checks['over'] = max(checks['over'], float((inv.model(a, 1-v)-inv.model(x, 1-v)).abs().max()))
                    checks['inv_vs_free_init'] = max(checks['inv_vs_free_init'], float((a-fre.model(x, v)).abs().max()))
                    if not (model['key'] == 'gemma' and bb.hidden == 3584): checks['free_identity'] = max(checks['free_identity'], float((fre.model(x, v)-x).abs().max()))
        rep['seeds'][str(seed)] = dict(u_rule=urule, head_rule=hrule, heads={arm: {k: v for k, v in h.items() if k not in ('head', 'bias')} for arm, h in heads.items()}, checks=checks, file=str(out/f'init_seed{seed}.npz'), sha256=Q.sha(out/f'init_seed{seed}.npz'))
    rep['cap_ref'] = cap_ref; rep['mu_norm'] = float(np.linalg.norm(mu)); Q.dump(out/'INIT.json', rep); return rep

def make_bank(bb, arm, seed, model, cap_ref, mu, out):
    site = model['site']
    if arm in Q.KIND:
        bank = SetterBank(bb, arm, site, seed, model['key'], cap_ref); bank.set_center(mu)
        with np.load(out/f'init_seed{seed}.npz', allow_pickle=False) as z: bank.set_basis(z['u']); bank.set_heads(z[f'{arm}_head'], z[f'{arm}_bias'])
        bank.init = dict(kind=('gemma_v2_svd_ridge_warm_start' if (model['key'] == 'gemma' and bb.hidden == 3584) else 'fit_svd_zero_functional'), file=str(out/f'init_seed{seed}.npz'), sha256=Q.sha(out/f'init_seed{seed}.npz')); return bank
    if arm == 'CONTINUED_AFFINE': return AffineBank.from_v2(bb, 'SHARED_CLAUSE', seed)
    if arm == 'AFFINE_CLAUSE': return AffineBank.fresh_clause(bb, site, cap_ref, seed, model['key'], mu, np.zeros(bb.hidden, np.float32))
    raise ValueError(arm)

def strengths_for(bank): return Q.STRENGTHS_SETTER if isinstance(bank, SetterBank) else Q.STRENGTHS_AFFINE

# ---------------------------------------------------------------- D1 training
def groups_for(bundle_list, seed):
    out = []
    for i, b in enumerate(bundle_list):
        qs = b['queries']; ch = [k for k, qq in enumerate(qs) if qq['rec']['changed']]; inv = [k for k, qq in enumerate(qs) if not qq['rec']['changed']]; w = np.zeros(len(qs))
        if ch and inv: w[ch] = .5/len(ch); w[inv] = .5/len(inv)
        else: w[:] = 1.0/len(qs)
        idx = list(range(len(qs))); random.Random(f'{seed}|{b["scene"].scene_id}|{b["program"]}').shuffle(idx)
        for k in range(0, len(idx), Q.GROUP): out.append(dict(bundle=i, queries=idx[k:k+Q.GROUP], weights=[float(w[j]) for j in idx[k:k+Q.GROUP]]))
    return out

def save_bank(bank, path, **meta): return bank.save(path, **meta)

def load_bank(bb, path):
    meta = Q.load(Path(path)/'META.json')
    if meta.get('schema') == 'RSO3_SETTER_V1': return SetterBank.load(bb, path)
    return AffineBank.load(bb, path)

def train_epochs(bb, bank, arm, fit_b, cal_b, lr, seed, epochs, cdir, src_fit, src_cal, deadline, log, out, start_epoch=1, opt=None, curve=None):
    import torch
    opt = opt or torch.optim.AdamW(bank.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0); curve = curve if curve is not None else []
    name = f'{arm}_lr{lr}_s{seed}'; setter = isinstance(bank, SetterBank); visits = 0; steps = 0
    for epoch in range(start_epoch, epochs+1):
        groups = groups_for(fit_b, seed*1000+epoch); random.Random(f'order|{seed}|{epoch}').shuffle(groups); t0 = time.monotonic(); losses = []; regs = []
        for g in groups:
            Q.budget(deadline, 120); b = fit_b[g['bundle']]; qs = [b['queries'][k] for k in g['queries']]
            pl = plan_for(bank, b, 1.0, record=True); r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, workspace=pl['workspace'])
            lp = torch.stack([r['label_logps'][k][qs[k]['rec']['gold']] for k in range(len(qs))]); nll = -(torch.as_tensor(g['weights'], device=lp.device, dtype=lp.dtype)*lp).sum(); loss = nll
            if setter:
                reg = Q.BETA*torch.cat([m['_delta_sq'] for m in pl['maps']]).mean()/(bank.cap_ref**2); loss = nll+reg; regs.append(float(reg))
            bb.backward(loss); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(nll)); visits += len(qs); steps += 1
        train_s = time.monotonic()-t0; ck = cdir/f'epoch{epoch}'; save_bank(bank, ck, epoch=epoch, lr=lr, steps=steps, visits=visits)
        if setter: bank.freeze_basis()
        t1 = time.monotonic(); fit_rows, fit_sum = evaluate(bb, fit_b, arm, bank, 1.0, src_fit, seed=seed, tag=f'{name}|fit{epoch}', deadline=deadline)
        Q.write_rows(ck/'fit_rows.jsonl', fit_rows); fit_sum.pop('per_scene', None)
        for strength in strengths_for(bank):
            cal_rows, cal_sum = evaluate(bb, cal_b, arm, bank, strength, src_cal, seed=seed, tag=f'{name}|cal{epoch}|{strength}', deadline=deadline)
            Q.write_rows(ck/f'cal_rows_m{strength}.jsonl', cal_rows); cal_sum.pop('per_scene')
            curve.append(dict(arm=arm, lr=lr, seed=seed, epoch=epoch, strength=strength, checkpoint=str(ck), cal=cal_sum, fit=fit_sum, feasible=Q.feasible(cal_sum), frontier=Q.frontier_score(cal_sum), norm=cal_sum['mean_realized_norm'], frobenius=cal_sum['mean_frobenius'],
                              loss_train=float(np.mean(losses)), reg_train=(float(np.mean(regs)) if regs else None), visits=visits, steps=steps, train_seconds=train_s, eval_seconds=time.monotonic()-t1))
        if setter: bank.frozen_u = None
        best = max((c['frontier'] for c in curve if c['epoch'] == epoch), default=None)
        log(out, 'EPOCH', candidate=name, epoch=epoch, train_loss=float(np.mean(losses)), reg=(float(np.mean(regs)) if regs else None), fit_changed=fit_sum['changed'], fit_harm=fit_sum['harm'], cal_best_frontier=best, train_seconds=train_s, eval_seconds=time.monotonic()-t1)
        Q.dump(cdir/f'CURVE_epoch{epoch}.json', dict(at=Q.now(), curve=[c for c in curve if c['epoch'] == epoch]))
    return opt, curve

def epoch_frontier(curve, epoch): return max(c['frontier'] for c in curve if c['epoch'] == epoch)

def develop_arm(bb, arm, fit_b, cal_b, model, cap_ref, mu, e_max, out, src_fit, src_cal, deadline, log):
    adir = out/'arms'/arm
    if done(adir/'ARM_DONE.json'): return Q.load(adir/'ARM_DONE.json')
    if adir.exists() and any(adir.iterdir()):
        import shutil; aside = adir.parent/f'{arm}_partial_{int(time.time())}'; shutil.move(str(adir), str(aside)); log(out, 'ARM_RESTART', arm=arm, moved_partial_to=str(aside))
    adir.mkdir(parents=True, exist_ok=True); runs = {}
    for lr in Q.LRS:
        bank = make_bank(bb, arm, 0, model, cap_ref, mu, out); cdir = adir/f'lr{lr}_s0'
        if done(cdir/'DONE.json'): runs[lr] = Q.load(cdir/'DONE.json'); continue
        opt, curve = train_epochs(bb, bank, arm, fit_b, cal_b, lr, 0, Q.LR_EPOCHS, cdir, src_fit, src_cal, deadline, log, out)
        runs[lr] = dict(lr=lr, seed=0, curve=curve, epochs_run=Q.LR_EPOCHS, bank=bank, opt=opt, init=bank.init)
    choice = max(Q.LRS, key=lambda lr: (max(c['frontier'] for c in runs[lr]['curve']), -lr)); log(out, 'LR_CHOICE', arm=arm, lr=choice, frontiers={str(lr): max(c['frontier'] for c in runs[lr]['curve']) for lr in Q.LRS})
    run = runs[choice]; cdir = adir/f'lr{choice}_s0'
    if not done(cdir/'DONE.json'):
        bank, opt, curve = run['bank'], run['opt'], run['curve']; best = max(epoch_frontier(curve, e) for e in range(1, Q.LR_EPOCHS+1)); best_epoch = max(range(1, Q.LR_EPOCHS+1), key=lambda e: epoch_frontier(curve, e)); no_improve = Q.LR_EPOCHS-best_epoch; epochs_run = Q.LR_EPOCHS
        for epoch in range(Q.LR_EPOCHS+1, e_max+1):
            if no_improve >= Q.PATIENCE: log(out, 'EXTENSION_STOP', arm=arm, epoch=epoch-1); break
            if Q.budget(deadline, 0) < 1200: log(out, 'DEV_DEADLINE', arm=arm, note='extension cut by the development allocation; later epochs NOT_MEASURED'); break
            opt, curve = train_epochs(bb, bank, arm, fit_b, cal_b, choice, 0, epoch, cdir, src_fit, src_cal, deadline, log, out, start_epoch=epoch, opt=opt, curve=curve); epochs_run = epoch
            f = epoch_frontier(curve, epoch)
            if f > best: best = f; no_improve = 0
            else: no_improve += 1
        Q.dump(cdir/'DONE.json', dict(at=Q.now(), arm=arm, lr=choice, seed=0, epochs_run=epochs_run, curve=curve, init=run['init'], stats=bank.stats)); runs[choice] = Q.load(cdir/'DONE.json')
        for lr in Q.LRS:
            if lr != choice and not done(adir/f'lr{lr}_s0'/'DONE.json'): Q.dump(adir/f'lr{lr}_s0'/'DONE.json', dict(at=Q.now(), arm=arm, lr=lr, seed=0, epochs_run=Q.LR_EPOCHS, curve=runs[lr]['curve'], init=runs[lr]['init'], not_extended=True))
    s0 = Q.load(cdir/'DONE.json'); epochs0 = s0['epochs_run']; cdir1 = adir/f'lr{choice}_s1'
    if not done(cdir1/'DONE.json'):
        bank1 = make_bank(bb, arm, 1, model, cap_ref, mu, out); _, curve1 = train_epochs(bb, bank1, arm, fit_b, cal_b, choice, 1, epochs0, cdir1, src_fit, src_cal, deadline, log, out)
        Q.dump(cdir1/'DONE.json', dict(at=Q.now(), arm=arm, lr=choice, seed=1, epochs_run=epochs0, curve=curve1, init=bank1.init, stats=bank1.stats))
    s1 = Q.load(cdir1/'DONE.json'); cands = []
    for epoch in range(1, min(epochs0, s1['epochs_run'])+1):
        for strength in sorted({c['strength'] for c in s0['curve']}):
            c0 = next(c for c in s0['curve'] if c['epoch'] == epoch and c['strength'] == strength); c1 = next(c for c in s1['curve'] if c['epoch'] == epoch and c['strength'] == strength)
            cands.append(dict(lr=choice, epoch=epoch, strength=strength, seeds={0: c0['cal'], 1: c1['cal']}, norm=float(np.mean([c0['norm'], c1['norm']])), frobenius=float(np.mean([c0['frobenius'], c1['frobenius']])), checkpoints={0: c0['checkpoint'], 1: c1['checkpoint']}, fit={0: c0['fit'], 1: c1['fit']}))
    sel = Q.select_recipe(cands); sel['cal'] = {s: next(c for c in cands if c['epoch'] == sel['epoch'] and c['strength'] == sel['strength'])['seeds'][s] for s in (0, 1)}
    rep = dict(at=Q.now(), arm=arm, lr_choice=choice, epochs_seed0=epochs0, epochs_seed1=s1['epochs_run'], selection=sel, candidates=[{k: v for k, v in c.items() if k != 'fit'} for c in cands],
               opportunity=dict(visits_seed0=s0['curve'][-1]['visits'], steps_seed0=s0['curve'][-1]['steps'], visits_seed1=s1['curve'][-1]['visits']), init=s0['init'])
    Q.dump(adir/'ARM_DONE.json', rep); log(out, 'ARM_SELECTED', arm=arm, lr=choice, epoch=sel['epoch'], strength=sel['strength'], feasible=sel['feasible'], label=sel['label'], worse_changed=min(sel['cal'][s]['changed'] for s in (0, 1)), worse_harm=max(sel['cal'][s]['harm'] for s in (0, 1)))
    return rep

def epoch_plan(bench, arms, fit_b, cal_b, dev_seconds):
    n_groups = sum(math.ceil(len(b['queries'])/Q.GROUP) for b in fit_b); per = {}
    for arm in arms:
        n_str = len(Q.STRENGTHS_SETTER if arm in Q.KIND else Q.STRENGTHS_AFFINE); ev = (len(fit_b)+n_str*len(cal_b))*(bench['compile_shared']+bench['ask_batch12'])
        per[arm] = dict(train=n_groups*bench['train_step4_setter' if arm in Q.KIND else 'train_step4_affine']*1.1, eval=ev*1.15)
    def total(e): return sum((Q.LR_EPOCHS+2*e)*(v['train']+v['eval']) for v in per.values())
    e_max = max([e for e in range(Q.LR_EPOCHS, Q.MAX_EPOCHS+1) if total(e) <= dev_seconds] or [Q.LR_EPOCHS])
    return dict(e_max=e_max, per_arm_epoch_seconds=per, predicted_total_seconds=total(e_max), groups_per_epoch=n_groups, dev_seconds=dev_seconds, rule='E_max = largest e in [2, 4] with sum over arms of (2 + 2e) epoch-times <= development allocation, from the disposable benchmark before any training outcome')

def phase_d(bb, out, deadline, model, fit, cal, qdir, arms, dev_hours):
    qual = Q.load(qdir/'QUALIFICATION.json'); proc = qual['selected_procedure']; bench = Q.load(qdir/'BENCHMARK.json')['seconds']
    if not proc: Q.dump(out/'DEV_STOP.json', dict(at=Q.now(), status='MODEL_UNQUALIFIED', qualification=qual)); return dict(status='MODEL_UNQUALIFIED')
    log = event; dev_s = min(Q.budget(deadline, 0), dev_hours*3600); dev_deadline = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=dev_s)).isoformat()
    log(out, 'DEV_START', procedure=proc, arms=arms, dev_deadline=dev_deadline, dev_hours=dev_hours)
    fit_b = bundles(bb, fit, 'FIT', proc); cal_b = bundles(bb, cal, 'CAL', proc)
    log(out, 'BUNDLES', fit=len(fit_b), cal=len(cal_b), fit_noop=sum(b['program'] == 'noop' for b in fit_b), cal_noop=sum(b['program'] == 'noop' for b in cal_b), fit_queries=sum(len(b['queries']) for b in fit_b))
    src_fit = natural_eval(bb, fit_b, out, 'source_fit', deadline=deadline); src_cal = natural_eval(bb, cal_b, out, 'source_cal', deadline=deadline)
    cf_cal = natural_eval(bb, [b for b in cal_b if b['program'] == 'single'], out, 'counterfactual_cal', world_key='world', source_preds=src_cal, deadline=deadline)
    if not done(out/'BASELINES.json'):
        base = {k: {kk: vv for kk, vv in Q.summarize(list(v.values())).items() if kk != 'per_scene'} for k, v in (('no_edit_fit', src_fit), ('no_edit_cal', src_cal), ('counterfactual_cal', cf_cal))}
        Q.dump(out/'BASELINES.json', dict(at=Q.now(), **base))
    cap = captures(bb, fit_b, out, model, deadline); capd, mu, clause_states, z_src, z_dst = load_captures(out); log(out, 'CAPTURES', cap_ref=cap['cap_ref'], cap_new_rule=cap['cap_new_fit_rule'], mu_norm=cap['mu_norm'])
    init = initializations(bb, fit_b, out, model, cap['cap_ref'], mu, clause_states, z_src, z_dst, deadline); log(out, 'INIT', seeds={s: dict(checks=v['checks'], heads={a: h.get('error') for a, h in v['heads'].items()}) for s, v in init['seeds'].items()})
    plan = epoch_plan(bench, arms, fit_b, cal_b, Q.budget(dev_deadline, 0)); plan['at'] = Q.now(); plan['arms'] = list(arms)
    if not done(out/'DEV_PLAN.json'): Q.dump(out/'DEV_PLAN.json', plan)
    plan = Q.load(out/'DEV_PLAN.json'); log(out, 'DEV_PLAN', e_max=plan['e_max'], predicted=plan['predicted_total_seconds'], dev_seconds=plan['dev_seconds'])
    results = {}
    for arm in arms:
        if Q.budget(deadline, 0) < 2400: log(out, 'DEV_DEADLINE', arm=arm, note='NOT_MEASURED: development deadline reached before this arm'); continue
        results[arm] = develop_arm(bb, arm, fit_b, cal_b, model, cap['cap_ref'], mu, plan['e_max'], out, src_fit, src_cal, dev_deadline if Q.budget(dev_deadline, 0) > 0 else deadline, log)
    freeze = dict(at=Q.now(), actor=model['key'], procedure=proc, qualification=dict(policy=qual.get('policy'), selected=proc, equality=qual.get('equality_summary')), site=model['site'], cap_ref=cap['cap_ref'], arms=list(arms), e_max=plan['e_max'],
                  selection={a: dict(lr=r['selection']['lr'], epoch=r['selection']['epoch'], strength=r['selection']['strength'], feasible=r['selection']['feasible'], label=r['selection']['label'], checkpoints=r['selection']['checkpoints'], cal=r['selection']['cal'], norm=r['selection']['norm'], frobenius=r['selection']['frobenius']) for a, r in results.items()},
                  not_measured=[a for a in arms if a not in results], captures_sha256=cap['captures_sha256'], init={s: v['sha256'] for s, v in init['seeds'].items()}, frozen_references={k: dict(v, dir={s: str(Q.v2_factor_dir(v['arm'], s)) for s in Q.SEEDS}) for k, v in Q.FROZEN.items() if model['key'] == 'gemma'},
                  final_access='none: no FINAL or WORKFLOW scene was read in phase D', training='single operations only (flip + declared no-op); no multi-command, repetition or restoration result entered fitting, checkpoint choice or strength choice',
                  rule='per arm: worse-seed changed accuracy among recipes feasible in each seed (<=5% harm, >=80% each direction), ties higher all-24, lower harm, smaller norm, earlier epoch; else diagnostic by worse-seed changed - 2*harm; setters strength exactly 1; affine CAL grid {.5,.75,1,1.25}')
    if not done(out/'FREEZE.json'): Q.dump(out/'FREEZE.json', freeze)
    log(out, 'FREEZE', selection={a: (v['lr'], v['epoch'], v['strength'], v['feasible']) for a, v in freeze['selection'].items()})
    return dict(status='DEVELOPMENT_COMPLETE', selection=freeze['selection'], not_measured=freeze['not_measured'])
