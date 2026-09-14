"""RCC4 phase D: matched single-edit coverage development of the four conditions per actor, warm-started from the frozen V3 selected
checkpoints, then FREEZE before any FINAL/WORKFLOW access.

Conditions   INV_SPARSE_CONTINUE / INV_COMPLETE_SINGLE (from the V3 INVARIANT_SET checkpoint), FREE_SPARSE_CONTINUE / FREE_COMPLETE_SINGLE
             (from the V3 FREE_OVERWRITE checkpoint); per actor and seed the SAME V3 checkpoint starts both coverage conditions of an arm.
Supervision  single flips + declared no-ops only (every fourth FIT scene), FIT 96 / CAL 48 fresh RCC4 scenes.  COMPLETE: the 24 questions
             of two label-code draws (twelve specifications each, including both/either and the second comparison pair).  SPARSE: its
             twelve original questions each used twice, weights halved, for the same 24 question-forward slots.  Per scene the changed and
             invariant questions share the weight equally (no-op scenes: invariant only), total weight 1 per scene in both conditions.
Opportunity  identical scene/no-op groups, six optimizer steps of four slots per scene per epoch, identical group order across the four
             conditions (seeded per epoch), inherited V3-selected learning rate per arm/actor, AdamW (.9,.999) eps 1e-8 wd 0, grad clip 1,
             full-vocabulary gold NLL + beta 0.01 * mean(||delta||^2 / cap_ref^2), at most TWO complete epochs, no learning-rate grid.
Selection    checkpoints 0 (the V3 checkpoint = retention baseline), 1, 2 evaluated on the COMPLETE single-edit CAL bundles (24 questions
             per artifact); the declared feasibility-first rule per condition and seed; CANONICAL_CURRENT editor per seed and the workflow
             candidate per actor chosen by the same rule -> FREEZE.json.  No repetition, restoration or multi-address program exists here.
"""
from __future__ import annotations
import datetime as dt
import random
import shutil
import time
from pathlib import Path
import numpy as np
import rcc4_common as C
import rso3_common as Q
import rso3_train as T
from rso3_editor import SetterBank
from rso3_unit import event, done

# ---------------------------------------------------------------- encodings (coverage-aware FIT/CAL bundles)
def scene_bundle(bb, s, split, proc, program='single', order='original', variant=0, coverage='sparse'):
    text, spans = C.prefix_text(s, proc, order, variant); pre = bb.prefix_ids(text); edits = C.programs(s)[program]
    addresses = [tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in edits]; values = [int(e.value) for e in edits]
    world = C.final_world(s, program); qs = []
    for draw in (0, 1):
        for q in C.questions(s, split, draw, program, coverage):
            enc = bb.encode(text, C.suffix_text(q['text'])); rec = C.query_record(s, q, split, draw, order, variant, program); rec['coverage'] = coverage
            qs.append(dict(q=q, suffix=enc['suffix_ids'], labels=bb.label_ids(enc['full_text'], q['labels']), rec=rec))
    req = Q.OLD.CompileRequest(tuple(pre), tuple(addresses), tuple(values), 'development')
    return dict(scene=s, split=split, program=program, order=order, variant=variant, proc=proc, text=text, prefix=pre, clause=addresses[0], addresses=addresses, values=values, queries=qs, edits=edits,
                world=world, request_key=req.key(), n_prefix=len(pre), touched=[(e.actor, e.project) for e in edits], coverage=coverage)

def bundles(bb, scene_list, split, proc, coverage):
    out = [scene_bundle(bb, s, split, proc, coverage=coverage) for s in scene_list]
    out += [scene_bundle(bb, s, split, proc, program='noop', coverage=coverage) for s in C.noop_scenes(scene_list)]
    return out

# ---------------------------------------------------------------- equal weighted slots
def slots_for(b, seed):
    """24 (question index, weight) slots for one scene bundle: COMPLETE uses its 24 distinct questions once; SPARSE uses its 12 questions
    twice with halved weights.  Changed/invariant balance within the scene; total weight exactly 1 in both cases."""
    qs = b['queries']; n = len(qs); reps = C.SLOTS//n
    if n*reps != C.SLOTS: raise ValueError(f'bundle has {n} questions; cannot fill {C.SLOTS} slots')
    ch = [k for k, qq in enumerate(qs) if qq['rec']['changed']]; inv = [k for k, qq in enumerate(qs) if not qq['rec']['changed']]; w = np.zeros(n)
    if ch and inv: w[ch] = .5/len(ch); w[inv] = .5/len(inv)
    else: w[:] = 1.0/n
    slots = [(k, float(w[k])/reps) for k in range(n) for _ in range(reps)]
    random.Random(f'{seed}|{b["scene"].scene_id}|{b["program"]}').shuffle(slots); return slots

def groups_for(bundle_list, seed):
    out = []
    for i, b in enumerate(bundle_list):
        slots = slots_for(b, seed)
        for k in range(0, len(slots), C.GROUP): out.append(dict(bundle=i, queries=[q for q, _ in slots[k:k+C.GROUP]], weights=[w for _, w in slots[k:k+C.GROUP]]))
    return out

def epoch_order(groups, seed, epoch):
    order = list(range(len(groups))); random.Random(f'order|{seed}|{epoch}').shuffle(order); return order

# ---------------------------------------------------------------- warm start
def load_v3_bank(bb, actor, arm, seed, store, site=None):
    v = C.verify_v3_checkpoint(actor, arm, seed, store); bank = SetterBank.load(bb, v['dir']); bank.frozen_u = None; site = C.MODELS[actor]['site'] if site is None else int(site)
    if bank.arm != arm or bank.seed != seed or bank.site != site: raise ValueError('V3 checkpoint identity mismatch')
    bank.init = dict(kind='v3_selected_checkpoint_warm_start', dir=v['dir'], sha256=v['sha256'], v3_lr=v['v3_lr'], v3_epoch=v['v3_epoch'], load_basis_error=bank.load_basis_error, optimizer_state='fresh AdamW moments (V3 checkpoints hold no optimizer state)')
    return bank

def eval_checkpoint(bb, bank, cond, seed, cal_c, src_cal, ck, deadline):
    bank.freeze_basis(); rows, summ = T.evaluate(bb, cal_c, cond, bank, 1.0, src_cal, seed=seed, tag=f'{cond}|s{seed}|{ck.name}', deadline=deadline); bank.frozen_u = None
    Q.write_rows(ck/'cal_rows_complete.jsonl', rows); summ.pop('per_scene', None); sparse = Q.summarize([r for r in rows if int(r['query_id'].rsplit('-', 1)[1]) < C.SPARSE_SPECS]); sparse.pop('per_scene', None)
    return summ, dict(changed=sparse['changed'], harm=sparse['harm'], all24=sparse['all24'], changed_by_direction=sparse['changed_by_direction'], note='same rows restricted to the six original (sparse) specifications; diagnostic continuity with V3 CAL')

def train_condition(bb, cond, seed, fit_b, cal_c, src_cal, model, out, deadline, log, store, forecast=None):
    spec = C.CONDITIONS[cond]; arm = spec['arm']; lr = C.V3_SELECTED[model['key']][arm]['lr']; cdir = out/'arms'/cond/f's{seed}'
    if done(cdir/'DONE.json'): return Q.load(cdir/'DONE.json')
    if cdir.exists() and any(cdir.iterdir()):
        aside = cdir.parent/f's{seed}_partial_{int(time.time())}'; shutil.move(str(cdir), str(aside)); log(out, 'CONDITION_RESTART', condition=cond, seed=seed, moved_partial_to=str(aside))
    cdir.mkdir(parents=True, exist_ok=True); import torch
    bank = load_v3_bank(bb, model['key'], arm, seed, store, model['site']); curve = []
    ck0 = cdir/'epoch0'; bank.save(ck0, epoch=0, lr=lr, steps=0, slots=0, condition=cond, coverage=spec['coverage'], warm_start=bank.init)
    t1 = time.monotonic(); summ, sparse = eval_checkpoint(bb, bank, cond, seed, cal_c, src_cal, ck0, deadline)
    curve.append(dict(condition=cond, arm=arm, coverage=spec['coverage'], lr=lr, seed=seed, checkpoint=0, epoch=0, strength=1.0, path=str(ck0), cal=summ, cal_sparse_subset=sparse, feasible=Q.feasible(summ), rank=list(C.rank_key(summ, 0)), loss_train=None, reg_train=None, steps=0, slots=0, train_seconds=0.0, eval_seconds=time.monotonic()-t1, norm=summ['mean_realized_norm'], frobenius=summ['mean_frobenius']))
    log(out, 'CHECKPOINT', condition=cond, seed=seed, checkpoint=0, cal_changed=summ['changed'], cal_harm=summ['harm'], cal_all24=summ['all24'], feasible=Q.feasible(summ))
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0); steps = 0; slots = 0
    for epoch in range(1, C.EPOCHS+1):
        groups = groups_for(fit_b, seed*1000+epoch); order = epoch_order(groups, seed, epoch); t0 = time.monotonic(); losses = []; regs = []; wsum = []
        for gi in order:
            Q.budget(deadline, 120); g = groups[gi]; b = fit_b[g['bundle']]; qs = [b['queries'][k] for k in g['queries']]
            pl = T.plan_for(bank, b, 1.0, record=True); r = bb.full_forward(b['prefix'], [qq['suffix'] for qq in qs], [qq['labels'] for qq in qs], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, workspace=pl['workspace'])
            lp = torch.stack([r['label_logps'][k][qs[k]['rec']['gold']] for k in range(len(qs))]); nll = -(torch.as_tensor(g['weights'], device=lp.device, dtype=lp.dtype)*lp).sum()
            reg = C.BETA*torch.cat([m['_delta_sq'] for m in pl['maps']]).mean()/(bank.cap_ref**2); loss = nll+reg
            bb.backward(loss); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(nll)); regs.append(float(reg)); wsum.append(float(sum(g['weights']))); steps += 1; slots += len(qs)
        train_s = time.monotonic()-t0; ck = cdir/f'epoch{epoch}'; bank.save(ck, epoch=epoch, lr=lr, steps=steps, slots=slots, condition=cond, coverage=spec['coverage'], warm_start=bank.init)
        t1 = time.monotonic(); summ, sparse = eval_checkpoint(bb, bank, cond, seed, cal_c, src_cal, ck, deadline)
        curve.append(dict(condition=cond, arm=arm, coverage=spec['coverage'], lr=lr, seed=seed, checkpoint=epoch, epoch=epoch, strength=1.0, path=str(ck), cal=summ, cal_sparse_subset=sparse, feasible=Q.feasible(summ), rank=list(C.rank_key(summ, epoch)), loss_train=float(np.mean(losses)), reg_train=float(np.mean(regs)),
                          mean_group_weight=float(np.mean(wsum)), steps=steps, slots=slots, groups_this_epoch=len(groups), train_seconds=train_s, eval_seconds=time.monotonic()-t1, norm=summ['mean_realized_norm'], frobenius=summ['mean_frobenius']))
        log(out, 'CHECKPOINT', condition=cond, seed=seed, checkpoint=epoch, train_loss=float(np.mean(losses)), reg=float(np.mean(regs)), steps=steps, slots=slots, cal_changed=summ['changed'], cal_harm=summ['harm'], cal_all24=summ['all24'], feasible=Q.feasible(summ), train_seconds=train_s)
        if forecast is not None and not done(out/'FORECAST.json'): forecast(train_s, curve[-1]['eval_seconds'])
    sel = C.select_checkpoint([dict(checkpoint=c['checkpoint'], cal=c['cal']) for c in curve])
    rep = dict(at=Q.now(), condition=cond, arm=arm, coverage=spec['coverage'], seed=seed, lr=lr, epochs=C.EPOCHS, curve=curve, selection=dict(sel, cal=next(c['cal'] for c in curve if c['checkpoint'] == sel['checkpoint']), path=next(c['path'] for c in curve if c['checkpoint'] == sel['checkpoint'])),
               warm_start=bank.init, stats=bank.stats, opportunity=dict(steps=steps, slots=slots, groups_per_epoch=len(groups_for(fit_b, 1)), epochs=C.EPOCHS))
    Q.dump(cdir/'DONE.json', rep); log(out, 'CONDITION_DONE', condition=cond, seed=seed, selected=sel['checkpoint'], feasible=sel['feasible'], steps=steps, slots=slots)
    return rep

# ---------------------------------------------------------------- forecast (after the first matched epoch; never reads FINAL)
def forecast_fn(out, model, bench, n_conditions, deadline, effective_deadline, fit_b, cal_c):
    def fn(train_s, eval_s):
        per_cs = 2*train_s+3*eval_s; remaining_dev = per_cs*(n_conditions*2-1)+eval_s*0
        sizes = C.SIZES[model['key']]['FULL']; n_single = 4*2+2*2; n_prog = n_single+2; n_wf = 6
        per_single = bench['compile_shared']+2*bench['ask_batch12']; per_prog = bench['compile_repeat8']*0.6+bench['compile_shared']*0.4+3*bench['ask_batch12']; per_wf = bench['compile_shared']+4*bench['ask_generate32']+bench['hash']
        pred = dict(seal=sizes['final']*2*n_single*bench['compile_shared']*1.1, final=sizes['final']*2*(n_single+3)*per_single*1.25, programs=sizes['program']*len(C.PROGRAMS)*(n_prog+3)*per_prog*1.2*0.8, workflow=sizes['workflow']*len(C.WORKFLOW_PROGRAMS)*(n_wf+3)*per_wf*1.2*0.7, panels=600)
        pred['total'] = sum(pred.values()); rem = Q.budget(effective_deadline, 0)
        rep = dict(at=Q.now(), actor=model['key'], first_epoch_train_seconds=train_s, eval_seconds=eval_s, predicted_remaining_development_seconds=remaining_dev, predicted_evaluation_seconds=pred, remaining_to_effective_deadline_seconds=rem, remaining_to_measurement_deadline_seconds=Q.budget(deadline, 0),
                   fits=remaining_dev+pred['total'] <= rem, rule='forecast from the first matched epoch and the disposable benchmark; contracted FULL sizes are never shrunk; if the forecast does not fit, development stops early at complete matched conditions and evaluation stages record NOT_MEASURED when their time runs out', final_read=False)
        Q.dump(out/'FORECAST.json', rep); event(out, 'FORECAST', fits=rep['fits'], remaining_dev=remaining_dev, predicted_eval=pred['total'], remaining=rem)
    return fn

# ---------------------------------------------------------------- driver
def phase_d(bb, out, deadline, model, fit, cal, qdir, store, effective_deadline=None):
    qual = Q.load(qdir/'QUALIFICATION.json'); proc = qual['selected_procedure']; bench = Q.load(qdir/'BENCHMARK.json')['seconds']; effective_deadline = effective_deadline or deadline
    if not proc: Q.dump(out/'DEV_STOP.json', dict(at=Q.now(), status='MODEL_UNQUALIFIED', qualification=qual)); return dict(status='MODEL_UNQUALIFIED')
    log = event; log(out, 'DEV_START', procedure=proc, conditions=list(C.CONDITIONS), epochs=C.EPOCHS, seed_binding=C.PR.check_seed())
    fit_s = bundles(bb, fit, 'FIT', proc, 'sparse'); fit_c = bundles(bb, fit, 'FIT', proc, 'complete'); cal_c = bundles(bb, cal, 'CAL', proc, 'complete')
    leak = dict(fit_programs=sorted({b['program'] for b in fit_s+fit_c}), cal_programs=sorted({b['program'] for b in cal_c}), fit_scenes=len(fit), cal_scenes=len(cal), fit_noop=sum(b['program'] == 'noop' for b in fit_s), cal_noop=sum(b['program'] == 'noop' for b in cal_c),
                sparse_questions_per_scene=sorted({len(b['queries']) for b in fit_s}), complete_questions_per_scene=sorted({len(b['queries']) for b in fit_c}), cal_questions_per_scene=sorted({len(b['queries']) for b in cal_c}),
                slots_per_scene=dict(sparse=sorted({len(slots_for(b, 0)) for b in fit_s}), complete=sorted({len(slots_for(b, 0)) for b in fit_c})), groups_per_epoch=dict(sparse=len(groups_for(fit_s, 1)), complete=len(groups_for(fit_c, 1))),
                weight_per_scene=dict(sparse=sorted({round(sum(w for _, w in slots_for(b, 0)), 9) for b in fit_s}), complete=sorted({round(sum(w for _, w in slots_for(b, 0)), 9) for b in fit_c})),
                scene_id_overlap=sorted({s.scene_id for s in fit} & {s.scene_id for s in cal}), final_or_workflow_read=False, dedup=Q.load(C.DATA/'DEDUP.json')['collisions'])
    ok = (leak['fit_programs'] == ['noop', 'single'] and leak['cal_programs'] == ['noop', 'single'] and leak['sparse_questions_per_scene'] == [12] and leak['complete_questions_per_scene'] == [24] and leak['cal_questions_per_scene'] == [24] and leak['slots_per_scene'] == dict(sparse=[24], complete=[24])
          and leak['groups_per_epoch']['sparse'] == leak['groups_per_epoch']['complete'] and leak['weight_per_scene'] == dict(sparse=[1.0], complete=[1.0]) and not leak['scene_id_overlap'] and leak['dedup'] == 0)
    if not done(out/'LEAKAGE_CHECK.json'): Q.dump(out/'LEAKAGE_CHECK.json', dict(at=Q.now(), pass_=ok, **leak))
    if not ok: raise RuntimeError('leakage/opportunity check failed: '+Q.canonical(leak))
    log(out, 'BUNDLES', **{k: v for k, v in leak.items() if k in ('fit_noop', 'cal_noop', 'groups_per_epoch', 'slots_per_scene')})
    src_cal = T.natural_eval(bb, cal_c, out, 'source_cal_complete', deadline=deadline)
    cf_cal = T.natural_eval(bb, [b for b in cal_c if b['program'] == 'single'], out, 'counterfactual_cal_complete', world_key='world', source_preds=src_cal, deadline=deadline)
    if not done(out/'BASELINES.json'):
        base = {k: {kk: vv for kk, vv in Q.summarize(list(v.values())).items() if kk != 'per_scene'} for k, v in (('no_edit_cal_complete', src_cal), ('counterfactual_cal_complete', cf_cal))}
        Q.dump(out/'BASELINES.json', dict(at=Q.now(), **base))
    fc = forecast_fn(out, model, bench, len(C.CONDITIONS), deadline, effective_deadline, fit_c, cal_c); results = {}
    def enough_time():
        """Development stops early (never shrinking the contracted evaluation) when the next condition-seed plus the predicted primary
        evaluation (seal, singles, programs, panels) would exceed the actor's effective deadline; the forecast comes from the first matched epoch."""
        if not done(out/'FORECAST.json'): return True, None
        f = Q.load(out/'FORECAST.json'); need = (C.EPOCHS*f['first_epoch_train_seconds']+(C.EPOCHS+1)*f['eval_seconds'])*1.15; p = f['predicted_evaluation_seconds']
        primary = p['seal']+p['final']+p['programs']+p['panels']; rem = Q.budget(effective_deadline, 0); return rem >= need+primary, dict(need=need, primary_eval=primary, remaining=rem)
    for cond in C.CONDITION_ORDER:
        fit_b = fit_c if C.CONDITIONS[cond]['coverage'] == 'complete' else fit_s
        for seed in C.SEEDS:
            ok_time, why = enough_time()
            if not ok_time: log(out, 'DEV_STOP_EARLY', condition=cond, seed=seed, note='NOT_MEASURED: remaining time is reserved for the contracted primary evaluation; development stops at complete matched conditions', **why); continue
            results.setdefault(cond, {})[seed] = train_condition(bb, cond, seed, fit_b, cal_c, src_cal, model, out, deadline, log, store, forecast=fc)
    complete = {c: r for c, r in results.items() if len(r) == 2}; incomplete = [c for c in C.CONDITION_ORDER if c not in complete]
    selection = {c: {s: dict(checkpoint=r[s]['selection']['checkpoint'], feasible=r[s]['selection']['feasible'], label=r[s]['selection']['label'], rule=r[s]['selection']['rule'], path=r[s]['selection']['path'], cal=r[s]['selection']['cal'], lr=r[s]['lr']) for s in C.SEEDS} for c, r in complete.items()}
    canonical = {s: C.select_canonical({c: selection[c][s] for c in selection}) for s in C.SEEDS} if selection else {}
    workflow = C.select_workflow_candidate({s: {c: selection[c][s] for c in selection} for s in C.SEEDS}) if selection else None
    freeze = dict(at=Q.now(), actor=model['key'], procedure=proc, site=model['site'], seed_binding=C.SEED, conditions=list(C.CONDITIONS), epochs=C.EPOCHS, checkpoints=list(C.CHECKPOINTS), selection=selection, canonical_current=canonical, workflow_candidate=workflow,
                  not_measured=incomplete, v3_warm_starts={arm: {str(s): C.verify_v3_checkpoint(model['key'], arm, s, store) for s in C.SEEDS} for arm in C.ARMS}, frozen_v3=dict(C.FROZEN_V3), anchor_affine=(C.ANCHOR_AFFINE if model['key'] == 'gemma' else None),
                  final_access='none: no FINAL or WORKFLOW scene was read in phase D', training='single flips + declared no-ops only; 24 question-forward slots per scene per epoch in every condition; no repetition, restoration or multi-address program entered fitting or checkpoint choice',
                  rule='per condition and seed over checkpoints 0/1/2 on complete single-edit CAL bundles: feasible (<=5% harm, >=80% each direction) first, then highest all-24 whole-bundle correctness, higher changed accuracy, lower harm, earlier checkpoint; infeasible -> V3 frontier (changed - 2*harm) reported as not capable; CANONICAL_CURRENT per seed and the workflow candidate per actor by the same rule')
    if not done(out/'FREEZE.json'): Q.dump(out/'FREEZE.json', freeze)
    log(out, 'FREEZE', selection={c: {s: (v[s]['checkpoint'], v[s]['feasible']) for s in v} for c, v in selection.items()}, canonical={s: v['condition'] for s, v in canonical.items()}, workflow=(workflow or {}).get('condition'), not_measured=incomplete)
    return dict(status='DEVELOPMENT_COMPLETE' if complete else 'DEVELOPMENT_INCOMPLETE', selection=selection, not_measured=incomplete)
