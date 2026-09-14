"""QBRC196 phase D: development of the three rank-16 construction families on FIT/CAL only.

D0 source captures: per FIT scene the source and natural-counterfactual prefixes are compiled (question-blind) and the addressed-clause
   summaries z at every site are pooled -> per-site/task caps (2 x p95 of ||z_dst - z_src||), frozen FIT source means (mu), and the
   FIT mean-difference late direction (h_last of counterfactual prefix+query minus source prefix+query, per task/direction).
D1 training (seed 0): predeclared candidate order over families x sites x lrs; AdamW, clip 1.0, groups of four queries from one scene
   with equal changed/invariant weighting, scenes weighted equally; complete FIT and CAL evaluated at every epoch boundary through the
   cached serving path; stop after two successive CAL evaluations without improvement; epochs per candidate fixed from measured time.
D2 selection by the declared rule, seed-1 reruns of each family's best setting, mean comparator calibration -> FREEZE.json.
Nothing here touches FINAL scenes.
"""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import qbrc_common as Q
from qbrc_unit import event, done

# ---------------------------------------------------------------- encodings
def scene_bundle(bb, s, split, proc):
    text, span = Q.prefix_text(s, proc); pre = bb.prefix_ids(text); cp = bb.clause_positions(text, span); qs = []
    for draw in (0, 1):
        for q in Q.questions(s, split, draw):
            enc = bb.encode(text, Q.suffix_text(q, proc)); qs.append(dict(q=q, suffix=enc['suffix_ids'], labels=bb.label_ids(enc['full_text'], q.labels), rec=Q.query_record(s, q, split, draw)))
    tgt = Q.apply_edit(s); ttext, tspan = Q.prefix_text(s, proc, target=tgt); tpre = bb.prefix_ids(ttext); tcp = bb.clause_positions(ttext, tspan)
    tq = []
    for draw in (0, 1):
        for q in Q.questions(s, split, draw):
            enc = bb.encode(ttext, Q.suffix_text(q, proc)); tq.append(dict(suffix=enc['suffix_ids']))
    req = Q.compile_request(s, proc, bb.cfg['revision'], 'development')
    return dict(scene=s, text=text, prefix=pre, clause=cp, queries=qs, target_text=ttext, target_prefix=tpre, target_clause=tcp, target_queries=tq, request_key=req.key(),
                new_value=req.new_value, task=s.task, n_prefix=len(pre))

def bundles(bb, scenes, split, procedures):
    return [scene_bundle(bb, s, split, procedures[s.task]) for s in scenes if s.task in procedures]

# ---------------------------------------------------------------- editors
class EditorBank:
    """Four addressed rank-16 maps (task x desired value) sharing one site, one frozen center and per-task caps."""
    def __init__(self, bb, site, family, caps, center, seed, cap_scale=1.0):
        import torch; from editor import AddressedLowRankEditor
        self.bb = bb; self.site = int(site); self.family = family; self.caps = {k: float(v)*cap_scale for k, v in caps.items()}; self.seed = int(seed); self.cap_scale = cap_scale
        self.maps = {}
        for i, (task, v) in enumerate([(t, v) for t in Q.TASKS for v in (0, 1)]):
            if task not in self.caps: continue          # only qualified tasks carry maps (the seed index stays task-stable)
            e = AddressedLowRankEditor(bb.hidden, Q.RANK, seed=seed*97+i).to(bb.device)
            with torch.no_grad(): e.center.copy_(center.to(bb.device))
            self.maps[(task, v)] = e
        self.stats = dict(updates=0, saturated=0, raw_norm_sum=0.0, realized_norm_sum=0.0)

    def parameters(self): return [p for e in self.maps.values() for p in e.parameters()]

    def fn(self, task, new_value, strength=1.0, record=True):
        e = self.maps[(task, int(new_value))]; cap = self.caps[task]; t = self.bb.t
        def f(h, z):
            zz = z.expand_as(h) if z.ndim == 1 else z
            x = t.cat((h-e.center, zz-e.center), dim=-1); raw = (x@e.input_factor+e.bias)@e.output_factor
            norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-12); d = raw*t.clamp(cap/norm, max=1.)
            if record:
                self.stats['updates'] += int(h.shape[0]); self.stats['saturated'] += int((norm.detach() > cap).sum()); self.stats['raw_norm_sum'] += float(norm.detach().sum())
            return d*strength
        return f

    def raw_norms(self, task, new_value, h, z):
        e = self.maps[(task, int(new_value))]; t = self.bb.t; zz = z.expand_as(h) if z.ndim == 1 else z
        x = t.cat((h-e.center, zz-e.center), dim=-1); return ((x@e.input_factor+e.bias)@e.output_factor).norm(dim=-1)

    def save(self, path, **meta):
        path = Path(path); path.mkdir(parents=True, exist_ok=False); files = {}
        for (task, v), e in sorted(self.maps.items()):
            p = path/f'{task}_{v}.npz'
            np.savez(p, input_factor=e.input_factor.detach().cpu().numpy(), output_factor=e.output_factor.detach().cpu().numpy(), bias=e.bias.detach().cpu().numpy(),
                     center=e.center.detach().cpu().numpy(), cap=np.float64(self.caps[task]), schema=np.asarray('QBRC_RANK16_CHECKPOINT_V1'))
            files[f'{task}_{v}'] = Q.sha(p)
        Q.dump(path/'META.json', dict(at=Q.now(), site=self.site, family=self.family, seed=self.seed, caps=self.caps, cap_scale=self.cap_scale, files=files, **meta)); return files

    @classmethod
    def load(cls, bb, path):
        import torch
        meta = Q.load(Path(path)/'META.json'); z = np.load(sorted(Path(path).glob('*.npz'))[0]); center = torch.as_tensor(z['center'])
        bank = cls(bb, meta['site'], meta['family'], {k: v/meta.get('cap_scale', 1.0) for k, v in meta['caps'].items()}, center, meta['seed'], meta.get('cap_scale', 1.0))
        for (task, v), e in bank.maps.items():
            with np.load(Path(path)/f'{task}_{v}.npz') as zz:
                with torch.no_grad():
                    e.input_factor.copy_(torch.as_tensor(zz['input_factor'])); e.output_factor.copy_(torch.as_tensor(zz['output_factor'])); e.bias.copy_(torch.as_tensor(zz['bias'])); e.center.copy_(torch.as_tensor(zz['center']))
        return bank

class MeanBank:
    """FIT mean-difference late direction per task/direction, capped, scaled by a calibrated strength."""
    def __init__(self, bb, site, vectors, caps):
        self.bb = bb; self.site = int(site); self.vectors = {k: bb.t.as_tensor(v, dtype=bb.t.float32, device=bb.device) for k, v in vectors.items()}; self.caps = caps
    def fn(self, task, new_value, strength=1.0):
        v = self.vectors[(task, int(new_value))]; cap = self.caps[task]; n = float(v.norm()); d = v*min(1., cap/max(n, 1e-12))*strength
        return lambda h, z: d.expand_as(h)

# ---------------------------------------------------------------- edit plans
def plan_for(bank, b, family, strength=1.0, record=True, extra=None):
    """Maps + index for the compile (prefix families) or the late spec (LATE) for one scene bundle."""
    fn = bank.fn(b['task'], b['new_value'], strength, record) if not isinstance(bank, MeanBank) else bank.fn(b['task'], b['new_value'], strength)
    m = dict(fn=fn, clause_positions=b['clause'])
    if family == 'LATE': return dict(late=dict(site=bank.site, maps=[m]))
    idx = [b['n_prefix']-1] if family == 'PREFIX_LAST' else list(range(b['clause'][0], b['n_prefix']))
    return dict(maps=[m], index=idx, site=bank.site)

# ---------------------------------------------------------------- evaluation through the cached serving path
def evaluate(bb, bundles_, family, bank, strength, source_rows, seed=0, tag='', deadline=None, generate=False):
    """Every query of every bundle, scenes and questions in a seeded random order; returns rows with prediction/source_prediction."""
    rng = random.Random(f'{seed}|{tag}'); order = list(range(len(bundles_))); rng.shuffle(order); rows = []; norms = []; energies = []; edited_tokens = []; bytes_ = []
    t0 = time.monotonic(); losses = []
    for i in order:
        b = bundles_[i]
        if deadline: Q.budget(deadline, 120)
        if family == 'LATE' or family == 'LATE_MEAN':
            art = bb.compile(b['prefix'], capture={bank.site}); late = plan_for(bank, b, 'LATE', strength, record=False)['late']
        else:
            pl = plan_for(bank, b, family, strength, record=False); art = bb.compile(b['prefix'], maps=pl['maps'], index=pl['index'], site=pl['site']); late = None
            r = art['realized']; norms.extend(r['realized_norms'][0]); energies.append(r['energy']); edited_tokens.append(r['positions'])
        bytes_.append(art['bytes']); qi = list(range(len(b['queries']))); rng.shuffle(qi)
        for j in qi:
            qq = b['queries'][j]; res = bb.ask(art, qq['suffix'], qq['labels'], late=late, generate=generate)
            if late is not None: norms.extend(res['realized']['realized_norms'][0]); energies.append(res['realized']['energy']); edited_tokens.append(1)
            rec = dict(qq['rec']); src = source_rows[rec['query_id']]
            rec.update(prediction=res['prediction'], logps=res['logps'], answer_mass=res['answer_mass'], source_prediction=src['prediction'], source_logps=src['logps'], gold_logp=res['logps'][rec['gold']],
                       argmax_in_labels=res['argmax_in_labels'], tie=res['tie'], family=family, strength=strength, artifact_hash=art['hash'], request_key=b['request_key'])
            if generate: rec['generation'] = res['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(res['generation']['text'], rec['labels'])
            losses.append(-rec['gold_logp']); rows.append(rec)
    summary = Q.frontier_rows(rows); summary.update(loss=float(np.mean(losses)), mean_realized_norm=float(np.mean(norms)) if norms else 0.0, mean_energy=float(np.mean(energies)) if energies else 0.0,
                                                    edited_tokens_per_scene=float(np.mean(edited_tokens)) if edited_tokens else 0.0, cache_bytes=float(np.mean(bytes_)), seconds=time.monotonic()-t0, n_rows=len(rows))
    return rows, summary

def source_eval(bb, bundles_, out, tag, counterfactual=False):
    """No-edit (source) or natural-counterfactual predictions for every query; cached path; the fixed baselines."""
    path = out/f'{tag}.jsonl'
    if done(path): return {r['query_id']: r for r in Q.read_rows(path)}
    rows = []
    for b in bundles_:
        art = bb.compile(b['target_prefix'] if counterfactual else b['prefix'])
        for k, qq in enumerate(b['queries']):
            suffix = b['target_queries'][k]['suffix'] if counterfactual else qq['suffix']
            res = bb.ask(art, suffix, qq['labels']); rec = dict(qq['rec']); rec.update(prediction=res['prediction'], logps=res['logps'], answer_mass=res['answer_mass'], argmax_in_labels=res['argmax_in_labels'], artifact_hash=art['hash'], world='counterfactual' if counterfactual else 'source'); rows.append(rec)
    Q.write_rows(path, rows); return {r['query_id']: r for r in rows}

# ---------------------------------------------------------------- D0 captures
def captures(bb, fit_b, out, model, procedures, deadline):
    if done(out/'CAPTURES.json'): return Q.load(out/'CAPTURES.json')
    import torch; sites = sorted(set(model['sites'])|{model['late_site']}); late = model['late_site']
    z_src = {s: [] for s in sites}; z_dst = {s: [] for s in sites}; mu_sum = {s: None for s in sites}; mu_n = 0; tasks = []; keys = []
    late_src = []; late_dst = []; late_keys = []
    for b in fit_b:
        Q.budget(deadline, 120)
        a = bb.compile(b['prefix'], capture=set(sites)); c = bb.compile(b['target_prefix'], capture=set(sites))
        for s in sites:
            z_src[s].append(bb.z_from_artifact(a, s, b['clause']).cpu().numpy()); z_dst[s].append(bb.z_from_artifact(c, s, b['target_clause']).cpu().numpy())
            h = a['captured'][s][1:].float(); mu_sum[s] = h.sum(0).cpu().numpy() if mu_sum[s] is None else mu_sum[s]+h.sum(0).cpu().numpy()
        mu_n += b['n_prefix']-1; tasks.append(b['task']); keys.append((b['task'], b['new_value']))
        # late mean direction: last-token states for every FIT query under source and counterfactual prefixes
        for k, qq in enumerate(b['queries']):
            rs = bb.ask_capture(a, qq['suffix'], qq['labels'], late); rd = bb.ask_capture(c, b['target_queries'][k]['suffix'], qq['labels'], late)
            late_src.append(rs); late_dst.append(rd); late_keys.append((b['task'], b['new_value']))
    caps = {}; stats = {}
    for s in sites:
        for task in procedures:
            d = np.array([np.linalg.norm(z_dst[s][i]-z_src[s][i]) for i in range(len(tasks)) if tasks[i] == task])
            caps[f'{s}|{task}'] = float(2*np.quantile(d, .95)); stats[f'{s}|{task}'] = dict(p95=float(np.quantile(d, .95)), median=float(np.median(d)), mean=float(d.mean()), n=int(len(d)), z_src_norm=float(np.mean([np.linalg.norm(z_src[s][i]) for i in range(len(tasks)) if tasks[i] == task])))
    mus = {s: (mu_sum[s]/mu_n).astype('<f4') for s in sites}; late_src = np.stack(late_src); late_dst = np.stack(late_dst)
    mu_late = late_src.mean(0).astype('<f4'); vectors = {}
    for task in procedures:
        for v in (0, 1):
            sel = [i for i, k in enumerate(late_keys) if k == (task, v)]
            vectors[f'{task}_{v}'] = (late_dst[sel]-late_src[sel]).mean(0).astype('<f4') if sel else np.zeros(bb.hidden, '<f4')
    np.savez(out/'captures.npz', **{f'mu_{s}': mus[s] for s in sites}, mu_late=mu_late, **{f'late_mean_{k}': v for k, v in vectors.items()},
             **{f'z_src_{s}': np.stack(z_src[s]).astype('<f4') for s in sites}, **{f'z_dst_{s}': np.stack(z_dst[s]).astype('<f4') for s in sites})
    rep = dict(at=Q.now(), sites=sites, late_site=late, caps=caps, cap_rule='2 x 95th percentile of FIT ||z_dst - z_src|| per site/task (pooled clause summaries; source and destination clauses may differ in token count)',
               delta_stats=stats, mu_rule='frozen FIT source mean over all prefix token states excluding position 0 (prefix families); frozen FIT source mean of final-query-token states (LATE)',
               late_mean_norms={k: float(np.linalg.norm(v)) for k, v in vectors.items()}, late_queries=int(len(late_keys)), scenes=len(fit_b), captures_sha256=Q.sha(out/'captures.npz'))
    Q.dump(out/'CAPTURES.json', rep); return rep

def load_captures(bb, out, model):
    import torch; z = np.load(out/'captures.npz'); cap = Q.load(out/'CAPTURES.json')
    sites = cap['sites']; mus = {s: torch.as_tensor(z[f'mu_{s}']) for s in sites}; mu_late = torch.as_tensor(z['mu_late'])
    vectors = {(k.split('_')[0], int(k.split('_')[1])): z['late_mean_'+k] for k in cap['late_mean_norms']}
    return cap, mus, mu_late, vectors

# ---------------------------------------------------------------- D1 training
def groups_for(fit_b, seed):
    out = []
    for i, b in enumerate(fit_b):
        idx = list(range(len(b['queries']))); random.Random(f'{seed}|{b["scene"].scene_id}').shuffle(idx)
        chunks = [idx[k:k+Q.GROUP] for k in range(0, len(idx), Q.GROUP)]
        for c in chunks: out.append(dict(bundle=i, queries=c, scene_weight=1.0/len(chunks)))
    return out

def train_candidate(bb, fit_b, cal_b, family, site, lr, seed, caps, center, epochs, out, src_fit, src_cal, deadline, first_at_site, cap_state, log):
    """Train one candidate for up to `epochs` complete epochs with the stopping rule; evaluate complete FIT (m1) and CAL (m.5/1/2) at each boundary."""
    import torch
    name = f'{family}@{site}_lr{lr}_s{seed}'; cdir = out/'candidates'/name
    if done(cdir/'DONE.json'): return Q.load(cdir/'DONE.json')
    cdir.mkdir(parents=True, exist_ok=True); scale = cap_state.get(str(site), 1.0)
    bank = EditorBank(bb, site, family, {t: caps[f'{site}|{t}'] for t in Q.TASKS if f'{site}|{t}' in caps}, center, seed, cap_scale=scale)
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    curve = []; best = -1e9; no_improve = 0; visits = 0; steps = 0; t_train = 0.0; t_eval = 0.0; fit_hist = []; restarted = False
    for epoch in range(1, epochs+1):
        groups = groups_for(fit_b, seed*1000+epoch); random.Random(f'order|{seed}|{epoch}').shuffle(groups); t0 = time.monotonic(); losses = []; sat0 = dict(bank.stats)
        for g in groups:
            Q.budget(deadline, 120); b = fit_b[g['bundle']]; qs = [b['queries'][k] for k in g['queries']]
            ch = [k for k, qq in enumerate(qs) if qq['rec']['affected']]; inv = [k for k, qq in enumerate(qs) if not qq['rec']['affected']]
            w = np.zeros(len(qs))
            if ch and inv: w[ch] = .5/len(ch); w[inv] = .5/len(inv)
            else: w[:] = 1.0/len(qs)
            pl = plan_for(bank, b, family, 1.0, record=True)
            if family == 'LATE': r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], late=pl['late'], grad=True)
            else: r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True)
            lp = torch.stack([r['label_logps'][k][qs[k]['q'].gold_index] for k in range(len(qs))])
            loss = -(torch.as_tensor(w, device=lp.device, dtype=lp.dtype)*lp).sum()*g['scene_weight']
            bb.backward(loss); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss)/g['scene_weight']); visits += len(qs); steps += 1
        t_train += time.monotonic()-t0; sat = (bank.stats['saturated']-sat0['saturated'])/max(1, bank.stats['updates']-sat0['updates'])
        ck = cdir/f'epoch{epoch}'; files = bank.save(ck, epoch=epoch, lr=lr, steps=steps, visits=visits)
        t1 = time.monotonic(); fit_rows, fit_sum = evaluate(bb, fit_b, family, bank, 1.0, src_fit, seed=seed, tag=f'{name}|fit{epoch}', deadline=deadline)
        Q.write_rows(ck/'fit_rows.jsonl', fit_rows); fit_hist.append(fit_sum['mean_score'])
        cal_summaries = {}
        for strength in Q.STRENGTHS:
            cal_rows, cal_sum = evaluate(bb, cal_b, family, bank, strength, src_cal, seed=seed, tag=f'{name}|cal{epoch}|{strength}', deadline=deadline)
            Q.write_rows(ck/f'cal_rows_m{strength}.jsonl', cal_rows); cal_summaries[str(strength)] = cal_sum
            curve.append(dict(candidate=name, family=family, site=site, lr=lr, seed=seed, epoch=epoch, strength=strength, checkpoint=str(ck), score=cal_sum['mean_score'],
                              harm=float(np.mean([cal_sum[t]['invariant_harm'] for t in Q.TASKS if cal_sum.get(t)])), changed=float(np.mean([cal_sum[t]['changed'] for t in Q.TASKS if cal_sum.get(t)])),
                              norm=cal_sum['mean_realized_norm'], per_task={t: cal_sum[t] for t in Q.TASKS if cal_sum.get(t)}, fit=fit_sum, saturation=sat, loss_train=float(np.mean(losses)), visits=visits, steps=steps))
        t_eval += time.monotonic()-t1
        ep_best = max(c['score'] for c in curve if c['epoch'] == epoch)
        log(out, 'EPOCH', candidate=name, epoch=epoch, train_loss=float(np.mean(losses)), fit_score=fit_sum['mean_score'], fit_loss=fit_sum['loss'], cal_best=ep_best, saturation=sat, train_seconds=t_train, eval_seconds=t_eval, visits=visits)
        # predeclared cap expansion: first candidate at a site, saturated > 25% of FIT updates and FIT not improved after epoch 1 -> x4 for that site, restart from init
        if epoch == 1 and first_at_site and scale == 1.0 and sat > .25 and fit_sum['mean_score'] <= src_fit['__score__']:
            cap_state[str(site)] = 4.0; log(out, 'CAP_EXPANSION', site=site, candidate=name, saturation=sat, fit_score=fit_sum['mean_score'], baseline=src_fit['__score__'])
            Q.dump(cdir/'CAP_EXPANSION_RESTART.json', dict(at=Q.now(), original_cap_scale=1.0, new_cap_scale=4.0, saturation=sat, epoch1=curve[-3:]))
            (cdir/'DONE.json').unlink(missing_ok=True); restarted = True; break
        if ep_best > best+1e-12: best = ep_best; no_improve = 0
        else: no_improve += 1
        if no_improve >= 2: log(out, 'EARLY_STOP', candidate=name, epoch=epoch); break
    if restarted:
        import shutil; shutil.move(str(cdir), str(cdir.parent/(name+'_capscale1_restarted')))
        return train_candidate(bb, fit_b, cal_b, family, site, lr, seed, caps, center, epochs, out, src_fit, src_cal, deadline, False, cap_state, log)
    rep = dict(at=Q.now(), candidate=name, family=family, site=site, lr=lr, seed=seed, epochs_run=max(c['epoch'] for c in curve), curve=curve, visits=visits, steps=steps, train_seconds=t_train, eval_seconds=t_eval,
               cap_scale=scale, caps=bank.caps, saturation_final=bank.stats['saturated']/max(1, bank.stats['updates']), stats=bank.stats)
    Q.dump(cdir/'DONE.json', rep); return rep

# ---------------------------------------------------------------- phase D driver
def phase_d(bb, out, deadline, model, fit, cal, qdir, dev_hours=4.0):
    import torch
    qual = Q.load(qdir/'QUALIFICATION.json'); procedures = qual['selected_procedures']; bench = Q.load(qdir/'BENCHMARK.json')['seconds']
    if not procedures:
        Q.dump(out/'DEV_STOP.json', dict(at=Q.now(), status='MODEL_UNQUALIFIED', qualification=qual)); return dict(status='MODEL_UNQUALIFIED')
    log = event; dev_deadline_s = min(Q.budget(deadline, 0), dev_hours*3600); dev_deadline = (__import__('datetime').datetime.now(__import__('datetime').timezone.utc)+__import__('datetime').timedelta(seconds=dev_deadline_s)).isoformat()
    log(out, 'DEV_START', procedures=procedures, dev_deadline=dev_deadline, dev_hours=dev_hours)
    fit_b = bundles(bb, fit, 'FIT', procedures); cal_b = bundles(bb, cal, 'CAL', procedures)
    log(out, 'BUNDLES', fit=len(fit_b), cal=len(cal_b), fit_queries=sum(len(b['queries']) for b in fit_b), cal_queries=sum(len(b['queries']) for b in cal_b))
    src_fit = source_eval(bb, fit_b, out, 'source_fit'); src_cal = source_eval(bb, cal_b, out, 'source_cal'); cf_cal = source_eval(bb, cal_b, out, 'counterfactual_cal', counterfactual=True)
    src_rows = list(src_fit.values()); base = Q.frontier_rows([dict(r, source_prediction=r['prediction']) for r in src_rows]); src_fit['__score__'] = base['mean_score']
    cal_base = Q.frontier_rows([dict(r, source_prediction=src_cal[r['query_id']]['prediction']) for r in src_cal.values()]); cf_base = Q.frontier_rows([dict(r, source_prediction=src_cal[r['query_id']]['prediction']) for r in cf_cal.values()])
    if not done(out/'BASELINES.json'): Q.dump(out/'BASELINES.json', dict(at=Q.now(), no_edit_fit=base, no_edit_cal=cal_base, counterfactual_cal=cf_base, note='no-edit = epoch-0/initialization behaviour of every learned candidate (zero functional map); counterfactual = natural destination-prefix reference'))
    log(out, 'BASELINES', no_edit_cal=cal_base['mean_score'], counterfactual_cal={t: (cf_base[t]['changed'], cf_base[t]['invariant_harm']) for t in Q.TASKS if cf_base.get(t)})
    cap = captures(bb, fit_b, out, model, procedures, deadline); log(out, 'CAPTURES', caps=cap['caps'], late_mean_norms=cap['late_mean_norms'])
    cap, mus, mu_late, vectors = load_captures(bb, out, model); caps = cap['caps']
    # epoch budget from the measured benchmark (decided before any training outcome)
    n_groups = sum(math.ceil(len(b['queries'])/Q.GROUP) for b in fit_b); n_fit = sum(len(b['queries']) for b in fit_b); n_cal = sum(len(b['queries']) for b in cal_b)
    candidates = [(fam, site, lr) for lr in Q.LRS for fam, site in [('LATE', model['late_site']), ('PREFIX_LAST', model['sites'][1]), ('PREFIX_DISTRIBUTED', model['sites'][1]), ('PREFIX_LAST', model['sites'][0]), ('PREFIX_DISTRIBUTED', model['sites'][0])]]
    step_t = {c: bench.get(f'train_step4_{c[0]}@{c[1]}', 1.0) for c in candidates}
    eval_t = (len(fit_b)+3*len(cal_b))*bench['compile']+(n_fit+3*n_cal)*bench['ask']
    epoch_t = {c: n_groups*step_t[c]*1.1+eval_t*1.15 for c in candidates}
    total_one_epoch = sum(epoch_t.values())+sum(epoch_t[c] for c in candidates[:3])  # + seed-1 reruns (one per family, assume the slowest of each family's settings ~ first three)
    remaining = Q.budget(dev_deadline, 0)-cap.get('_elapsed', 0)
    epochs = int(max(1, min(Q.MAX_EPOCHS, math.floor(remaining/total_one_epoch))))
    plan = dict(at=Q.now(), candidates=[f'{c[0]}@{c[1]}_lr{c[2]}' for c in candidates], epochs_per_candidate=epochs, epoch_seconds=epoch_t, eval_seconds=eval_t, groups_per_epoch=n_groups, remaining_dev_seconds=remaining,
                rule='fixed from measured benchmark timings before any training outcome: epochs = min(4, floor(remaining / (sum over 10 candidates + 3 seed-1 reruns of one-epoch time)))', dev_deadline=dev_deadline)
    if not done(out/'DEV_PLAN.json'): Q.dump(out/'DEV_PLAN.json', plan)
    log(out, 'DEV_PLAN', epochs=epochs, one_epoch_total=total_one_epoch, remaining=remaining)
    if remaining < total_one_epoch: log(out, 'DEV_WARNING', note='fewer than one epoch per candidate fits the development allocation; running one epoch each in the predeclared order until the development deadline')
    cap_state = Q.load(out/'CAP_STATE.json') if done(out/'CAP_STATE.json') else {}
    results = {}; seen_sites = set()
    for fam, site, lr in candidates:
        if Q.budget(deadline, 0) < 1800: log(out, 'DEV_DEADLINE', candidate=f'{fam}@{site}_lr{lr}', note='NOT_MEASURED: development deadline reached before this predeclared candidate'); continue
        try: Q.budget(dev_deadline, 0)
        except TimeoutError: log(out, 'DEV_DEADLINE', candidate=f'{fam}@{site}_lr{lr}', note='NOT_MEASURED: development deadline reached before this predeclared candidate'); continue
        center = mu_late if fam == 'LATE' else mus[site]; first = site not in seen_sites; seen_sites.add(site)
        results[f'{fam}@{site}_lr{lr}_s0'] = train_candidate(bb, fit_b, cal_b, fam, site, lr, 0, caps, center, epochs, out, src_fit, src_cal, deadline, first, cap_state, log)
        (out/'CAP_STATE.json').unlink(missing_ok=True); Q.dump(out/'CAP_STATE.json', cap_state)
    # selection per family (declared rule), then seed-1 reruns at the selected setting
    selection = {}
    for fam in Q.FAMILIES:
        curve = [c for r in results.values() if r['family'] == fam for c in r['curve']]
        if not curve: selection[fam] = dict(status='NOT_MEASURED'); continue
        best = Q.choose_setting([dict(c, norm=c['norm']) for c in curve]); selection[fam] = dict(best, status='SELECTED')
    if not done(out/'SELECTION_SEED0.json'): Q.dump(out/'SELECTION_SEED0.json', dict(at=Q.now(), selection=selection, rule='feasible (<=5% invariant harm, changed >=70%) by score, else best score labelled diagnostic-nonfeasible; ties lower harm, lower norm, earlier epoch'))
    log(out, 'SELECTION_SEED0', selection={f: (v.get('site'), v.get('lr'), v.get('epoch'), v.get('strength'), v.get('score'), v.get('feasible')) for f, v in selection.items()})
    for fam, sel in selection.items():
        if sel['status'] != 'SELECTED': continue
        if Q.budget(deadline, 0) < 1800: log(out, 'DEV_DEADLINE', candidate=f'{fam} seed1', note='NOT_MEASURED'); continue
        center = mu_late if fam == 'LATE' else mus[sel['site']]
        results[f'{fam}@{sel["site"]}_lr{sel["lr"]}_s1'] = train_candidate(bb, fit_b, cal_b, fam, sel['site'], sel['lr'], 1, caps, center, sel['epoch'], out, src_fit, src_cal, deadline, False, cap_state, log)
    # late mean comparator calibration on CAL
    mean_bank = MeanBank(bb, model['late_site'], vectors, {t: caps[f'{model["late_site"]}|{t}'] for t in procedures}); mean_curve = []
    for strength in Q.MEAN_STRENGTHS:
        rows, summ = evaluate(bb, cal_b, 'LATE_MEAN', mean_bank, strength, src_cal, tag=f'mean|{strength}', deadline=deadline)
        Q.write_rows(out/f'late_mean_cal_m{strength}.jsonl', rows)
        mean_curve.append(dict(strength=strength, score=summ['mean_score'], harm=float(np.mean([summ[t]['invariant_harm'] for t in Q.TASKS if summ.get(t)])), changed=float(np.mean([summ[t]['changed'] for t in Q.TASKS if summ.get(t)])), norm=summ['mean_realized_norm'], epoch=0, per_task={t: summ[t] for t in Q.TASKS if summ.get(t)}))
    mean_sel = Q.choose_setting(mean_curve)
    freeze = dict(at=Q.now(), model=model['key'], procedures=procedures, caps=caps, cap_state=cap_state, selection=selection, late_mean=dict(curve=mean_curve, selected=mean_sel, site=model['late_site']),
                  seeds_reported='both seeds evaluated on FINAL at the seed-0-selected epoch/strength; seed 1 was never used for selection', candidates={k: dict(epochs_run=v['epochs_run'], visits=v['visits'], steps=v['steps'], cap_scale=v['cap_scale']) for k, v in results.items()},
                  checkpoints={fam: {f's{seed}': str(out/'candidates'/f'{fam}@{sel["site"]}_lr{sel["lr"]}_s{seed}'/f'epoch{sel["epoch"]}') for seed in Q.SEEDS} for fam, sel in selection.items() if sel['status'] == 'SELECTED'},
                  late_opportunity={fam: results.get(f'{fam}@{sel.get("site")}_lr{sel.get("lr")}_s0', {}).get('visits') for fam, sel in selection.items()}, captures_sha256=cap['captures_sha256'], final_access='none: no FINAL scene was read in phase D')
    if not done(out/'FREEZE.json'): Q.dump(out/'FREEZE.json', freeze)
    log(out, 'FREEZE', selection={f: (v.get('site'), v.get('lr'), v.get('epoch'), v.get('strength'), v.get('score')) for f, v in selection.items()}, late_mean=(mean_sel['strength'], mean_sel['score']))
    return dict(status='DEVELOPMENT_COMPLETE', selection=selection, late_mean=mean_sel)
