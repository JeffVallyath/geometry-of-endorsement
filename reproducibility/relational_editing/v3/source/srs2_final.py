"""SRS2 phase F: fresh evaluation against sealed, question-blind compiled contexts.

F0 SCOPE    FULL/SMALL chosen from measured timings before any FINAL outcome (fixed scene IDs, every size/direction cell, both orders, all methods, both seeds).
F1 SEAL     every shared artifact (source, natural counterfactual, text correction, the four selected arms x seeds that are shared, the frozen V1
            SHARED_TAIL) compiled from prefix + explicit edit program only; hashes recorded BEFORE the first FINAL question exists.
F2 EVAL     both prefix orders, both answer-mapping draws, all 24 questions; aware arms recompile one prefix per question (never labelled blind);
            V1 LATE applied at the final query token of the source artifact; sampled master-hash verification; generation panel.
F3 CTRL     matched-mask random update and wrong-address update (scored against the intended and the actually requested world).
F4 ENERGY   SHARED_CLAUSE vs SHARED_TAIL after scaling BOTH full-prefix updates down to the smaller Frobenius norm (before any question).
F5 PROGRAMS single / no-op / repeat / restore / AB / BA / ABC / CBA inside ONE compilation on the algebra subset; gold from the true final world.
F6 WORKFLOW three receiver reports from the same compiled artifact -> separate unedited majority aggregator, two report phrasings, actual generations.
F7 WITNESS  post-block residual, next-block input and key/value tensors for four fixed scenes (source, selected SHARED_TAIL both seeds, counterfactual).
"""
from __future__ import annotations
import math
import random
import time
from pathlib import Path
import numpy as np
import srs2_common as Q
import srs2_train as T
from srs2_editor import Bank, LegacyBank
from srs2_unit import event, done, v1_dirs

RECEIVER_Q = 'Does {who} favor the {project} proposal?\nAnswer with exactly one of No, Yes.'
PHRASINGS = {0: 'Report on {who}: {who} {verb} the {project} proposal.', 1: "{who}'s recorded position: {stance} the {project} proposal."}

# ---------------------------------------------------------------- banks
def load_banks(bb, freeze, ddir, model):
    banks = {}
    for arm, sel in freeze['selection'].items():
        for seed in Q.SEEDS:
            ck = Path(sel['checkpoints'][str(seed)])
            if not ck.is_dir(): ck = ddir/'arms'/arm/ck.parent.name/ck.name
            banks[(arm, seed)] = dict(bank=Bank.load(bb, ck), strength=sel['strength'], site=model['site'], checkpoint=str(ck), meta_sha256=Q.sha(ck/'META.json'), selection=sel, aware=Q.AWARE[arm], legacy=False)
    if model['key'] == 'gemma':
        for seed in Q.SEEDS:
            banks[('V1_SHARED_TAIL', seed)] = dict(bank=LegacyBank(bb, v1_dirs('PREFIX_DISTRIBUTED')[seed], 'V1_SHARED_TAIL'), strength=1.0, site=15, checkpoint=str(v1_dirs('PREFIX_DISTRIBUTED')[seed]), aware=False, legacy=True, footprint='tail')
            banks[('V1_LATE', seed)] = dict(bank=LegacyBank(bb, v1_dirs('LATE')[seed], 'V1_LATE'), strength=1.0, site=27, checkpoint=str(v1_dirs('LATE')[seed]), aware=False, legacy=True, footprint='late')
    return banks

def cname(key): return f'{key[0]}_s{key[1]}'
def shared_keys(banks): return [k for k, v in banks.items() if not v['aware'] and k[0] != 'V1_LATE']
def aware_keys(banks): return [k for k, v in banks.items() if v['aware']]
def late_keys(banks): return [k for k in banks if k[0] == 'V1_LATE']

def plan(entry, b, strength=None, q_batch=None):
    arm = b.get('_arm') or entry.get('arm'); s = entry['strength'] if strength is None else strength
    if entry['legacy']: return T.plan_for(entry['bank'], b, 'SHARED_TAIL', s, record=False)
    return T.plan_for(entry['bank'], b, entry['bank'].arm, s, q_batch=q_batch, record=False)

def compile_shared(bb, b, entry, strength=None): return T.compile_plan(bb, b, plan(entry, b, strength))

def late_plan(entry, b): return dict(site=entry['site'], maps=[dict(fn=entry['bank'].fn(v, entry['strength']), clause_positions=list(cp)) for cp, v in zip(b['addresses'], b['values'])])

# ---------------------------------------------------------------- scoring helpers
CHUNK = 12   # one uniform batched clone-serving shape for every condition (validated against single asks in E0)

def score_shared(bb, b, art, src_preds, name, rows, extra=None, late=None, sample_every=8, generate=False, suffixes=None):
    qs = b['queries']; suff = suffixes or [qq['suffix'] for qq in qs]; res = []
    for k in range(0, len(qs), CHUNK): res += bb.ask_batch(art, suff[k:k+CHUNK], [qq['labels'] for qq in qs[k:k+CHUNK]], late=late, verify_master=(k == 0 and art.get('hash') is not None))
    for j, (qq, rr) in enumerate(zip(qs, res)):
        rec = dict(qq['rec']); sp = src_preds.get(rec['query_id']) if src_preds else None
        rec.update(condition=name, prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], gold_logp=rr['logps'][rec['gold']], argmax_in_labels=rr['argmax_in_labels'], tie=rr['tie'],
                   source_prediction=sp['prediction'] if sp else rr['prediction'], artifact_hash=art.get('hash'), master_verified=rr.get('master_unchanged'), realized_norm=(float(np.mean(rr['realized']['realized_norms'][0])) if rr.get('realized') else None))
        if generate and j < 12:
            g = bb.ask(art, suff[j], qq['labels'], late=late, generate=True); rec['generation'] = g['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(g['generation']['text'], rec['labels']); rec['first_token_is_argmax'] = g['generation']['first_token_is_argmax']
        if extra: rec.update(extra)
        rows.append(rec)

def score_aware(bb, b, entry, src_art, src_preds, name, rows, extra=None, generate=False, single_prefix=False):
    qs = b['queries']; site = entry['site']; qb = np.concatenate([bb.ask_capture_batch(src_art, [qq['suffix'] for qq in qs[k:k+CHUNK]], site) for k in range(0, len(qs), CHUNK)])
    for k in range(0, len(qs), CHUNK):
        sub = qs[k:k+CHUNK]
        if single_prefix:   # serving-mode control: one batch-1 prefix compile per question; suffixes still served in the uniform batch shape
            arts = [T.compile_plan(bb, b, plan(entry, b, q_batch=qb[k+j:k+j+1])) for j in range(len(sub))]; pl = plan(entry, b, q_batch=qb[k:k+1]); art = arts[0]; r = dict(art['realized'], row_energy=[a_['realized']['energy'] for a_ in arts])
            res = [bb.ask_batch(a_, [qq['suffix'] for qq in sub], [qq['labels'] for qq in sub])[j] for j, a_ in enumerate(arts)]
        else:
            pl = plan(entry, b, q_batch=qb[k:k+CHUNK]); art = T.compile_plan(bb, b, pl); r = art['realized']
            res = bb.ask_batch(art, [qq['suffix'] for qq in sub], [qq['labels'] for qq in sub])
        for j, (qq, rr) in enumerate(zip(sub, res)):
            rec = dict(qq['rec']); sp = src_preds[rec['query_id']]
            rec.update(condition=name, prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], gold_logp=rr['logps'][rec['gold']], argmax_in_labels=rr['argmax_in_labels'], tie=rr['tie'], source_prediction=sp['prediction'],
                       artifact_hash=None, aware=True, energy=r['row_energy'][j], edited_tokens=len(pl['index'])//len(sub), realized_norm=float(np.mean(rr['realized']['realized_norms'][0])) if rr.get('realized') else None, q_norm=float(np.linalg.norm(qb[k+j])))
            if generate and k == 0:
                a1 = T.compile_plan(bb, b, plan(entry, b, q_batch=qb[k+j:k+j+1])); g = bb.ask(a1, qq['suffix'], qq['labels'], generate=True); rec['generation'] = g['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(g['generation']['text'], rec['labels'])
            if extra: rec.update(extra)
            rows.append(rec)

def natural_rows(bb, b, text, src_preds, name, rows, generate=False, extra=None):
    pre = bb.prefix_ids(text); art = bb.compile(pre); suff = [bb.encode(text, Q.suffix_text(qq['q']['text']))['suffix_ids'] for qq in b['queries']]
    score_shared(bb, b, art, src_preds, name, rows, extra=dict(prefix_tokens=len(pre), **(extra or {})), generate=generate, suffixes=suff); return art

# ---------------------------------------------------------------- F0 scope
def choose_scope(bb, out, deadline, model, size, reserve_hours, banks, bench, n_final_full):
    if done(out/'SCOPE.json'): return Q.load(out/'SCOPE.json')
    n_sh = len(shared_keys(banks))+3; n_aw = len(aware_keys(banks)); n_late = len(late_keys(banks))
    per_art = n_sh*(bench['compile']+2*bench['ask_batch24'])+n_aw*(2*bench['ask_capture12']+2*bench['compile_aware12']+2*bench['ask_batch12'])+n_late*2*bench['ask_batch24']
    def predict(sz):
        s = Q.SIZES[model['key']][sz]; final = s['final']*2*per_art*1.25+Q.PANELS['variant1']*2*per_art*.5+8*24*bench['ask_generate32']*4
        programs = s['algebra']*8*(5*bench['compile']+5*2*bench['ask_batch24']+2*(2*bench['ask_capture12']+2*bench['compile_aware12']+2*bench['ask_batch12']))*1.2
        workflow = s['workflow']*(6+n_late)*(bench['compile']+3*bench['ask_generate32']+2*bench['ask_generate32'])*1.2
        panels = (Q.PANELS['control']*3+Q.PANELS['energy']*2*4)*(bench['compile']+2*bench['ask_batch24'])*1.2+Q.PANELS['witness']*4*bench['compile']*3+600
        return dict(final=final, programs=programs, workflow=workflow, panels=panels, total=final+programs+workflow+panels)
    remaining = Q.budget(deadline, 0)-Q.BUDGET['closure_reserve_minutes']*60-reserve_hours*3600
    pred = {sz: predict(sz) for sz in ('FULL', 'SMALL')}
    chosen = size if size != 'AUTO' else ('FULL' if pred['FULL']['total'] <= remaining else 'SMALL')
    rep = dict(at=Q.now(), scope=chosen, sizes=Q.SIZES[model['key']][chosen], predicted_seconds=pred, remaining_seconds=remaining, reserve_hours_after=reserve_hours, requested=size,
               rule='decided from the disposable benchmark before any FINAL outcome; SMALL keeps every size/direction cell, both orders, all methods, both seeds (fixed hashed IDs)', fits=pred[chosen]['total'] <= remaining)
    Q.dump(out/'SCOPE.json', rep); return rep

# ---------------------------------------------------------------- F1 seal
def prefix_bundle(bb, s, proc, program='single', order='early', variant=0, world=None, correction=False):
    text, spans = Q.prefix_text(s, proc, order, variant, world=world, correction=correction); pre = bb.prefix_ids(text); edits = Q.P.programs(s)[program]
    addresses = [tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in edits]; values = [int(e.value) for e in edits]
    return dict(scene=s, proc=proc, program=program, order=order, variant=variant, text=text, prefix=pre, clause=addresses[0], addresses=addresses, values=values, n_prefix=len(pre), touched=[(e.actor, e.project) for e in edits],
                request_key=Q.P.CompileRequest(tuple(pre), tuple(addresses), tuple(values), 'seal').key())

def seal(bb, scenes, variant1, proc, banks, out, deadline, model):
    if done(out/'SEAL.json'): return Q.load(out/'SEAL.json')
    rows = []; t0 = time.monotonic(); keys = shared_keys(banks)
    for s in scenes:
        for order in Q.ORDERS:
            for variant in ((0, 1) if s.scene_id in variant1 else (0,)):
                Q.budget(deadline, 120); b = prefix_bundle(bb, s, proc, 'single', order, variant); world = Q.edited_world(s, 'single')
                src = bb.compile(b['prefix']); ctext, _ = Q.prefix_text(s, proc, order, variant, world=world); cf = bb.compile(bb.prefix_ids(ctext)); ttext, _ = Q.prefix_text(s, proc, order, variant, correction=True); tc = bb.compile(bb.prefix_ids(ttext))
                row = dict(scene_id=s.scene_id, order=order, variant=variant, program='single', prefix_tokens=len(b['prefix']), clause_tokens=len(b['clause']), clause_start=b['clause'][0], tail_tokens=len(b['prefix'])-b['clause'][0], request_key=b['request_key'],
                           artifacts=dict(SOURCE=dict(hash=src['hash'], bytes=src['bytes']), COUNTERFACTUAL=dict(hash=cf['hash'], bytes=cf['bytes'], prefix_tokens=cf['prefix_len']), TEXT_CORRECTION=dict(hash=tc['hash'], bytes=tc['bytes'], prefix_tokens=tc['prefix_len'])))
                for key in keys:
                    entry = banks[key]; art = compile_shared(bb, b, entry); r = art['realized']
                    row['artifacts'][cname(key)] = dict(hash=art['hash'], bytes=art['bytes'], edited_tokens=r['positions'], energy=r['energy'], frobenius=r['frobenius'], mean_realized_norm=float(np.mean(r['realized_norms'][0])), method_digest=entry.get('meta_sha256'), strength=entry['strength'])
                rows.append(row)
    Q.write_rows(out/'SEAL.jsonl', rows)
    rep = dict(at=Q.now(), artifacts=len(rows), scenes=len(scenes), conditions=['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+[cname(k) for k in keys], seal_sha256=Q.sha(out/'SEAL.jsonl'), seconds=time.monotonic()-t0,
               compiler_inputs='prefix token ids, explicit clause spans and desired values of the edit program, method weights; no question, label mapping, gold, affected flag or target world existed before this file', questions_generated_before_seal=False)
    Q.dump(out/'SEAL.json', rep); return rep

# ---------------------------------------------------------------- F2 fresh evaluation
def evaluate_final(bb, scenes, variant1, proc, banks, out, deadline, model, gen_scenes):
    if done(out/'EVAL_DONE.json'): return Q.load(out/'EVAL_DONE.json')
    seal_by = {(r['scene_id'], r['order'], r['variant']): r for r in Q.read_rows(out/'SEAL.jsonl')}; sk = shared_keys(banks); ak = aware_keys(banks); lk = late_keys(banks)
    conds = ['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+[cname(k) for k in sk+ak+lk]; writers = {c: [] for c in conds}; mismatches = []; t0 = time.monotonic(); first_q = None; n_done = 0
    order_ = list(scenes); random.Random('final-order').shuffle(order_)
    for s in order_:
        for order in Q.ORDERS:
            for variant in ((0, 1) if s.scene_id in variant1 else (0,)):
                Q.budget(deadline, 300)
                if first_q is None: first_q = Q.now()
                b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order, variant); sr = seal_by[(s.scene_id, order, variant)]; gen = s.scene_id in gen_scenes and order == 'early' and variant == 0
                src = bb.compile(b['prefix'], capture={model['site']}|({27} if lk else set()))
                if src['hash'] != sr['artifacts']['SOURCE']['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, variant=variant, condition='SOURCE'))
                src_rows = []; score_shared(bb, b, src, None, 'SOURCE', src_rows, generate=gen); src_preds = {r['query_id']: r for r in src_rows}; writers['SOURCE'].extend(src_rows)
                ctext, _ = Q.prefix_text(s, proc, order, variant, world=b['world']); cf = natural_rows(bb, b, ctext, src_preds, 'COUNTERFACTUAL', writers['COUNTERFACTUAL'], generate=gen)
                ttext, _ = Q.prefix_text(s, proc, order, variant, correction=True); tc = natural_rows(bb, b, ttext, src_preds, 'TEXT_CORRECTION', writers['TEXT_CORRECTION'], generate=gen)
                for nm, art in (('COUNTERFACTUAL', cf), ('TEXT_CORRECTION', tc)):
                    if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, variant=variant, condition=nm))
                for key in sk:
                    entry = banks[key]; art = compile_shared(bb, b, entry); nm = cname(key)
                    if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, variant=variant, condition=nm))
                    score_shared(bb, b, art, src_preds, nm, writers[nm], extra=dict(edited_tokens=art['realized']['positions'], energy=art['realized']['energy'], frobenius=art['realized']['frobenius'], cache_bytes=art['bytes'], seed=key[1], method=key[0]), generate=gen and key[1] == 0)
                for key in ak: score_aware(bb, b, banks[key], src, src_preds, cname(key), writers[cname(key)], extra=dict(seed=key[1], method=key[0]), generate=gen and key[1] == 0)
                for key in lk: score_shared(bb, b, src, src_preds, cname(key), writers[cname(key)], late=late_plan(banks[key], b), extra=dict(edited_tokens=1, seed=key[1], method=key[0]), generate=gen and key[1] == 0)
        n_done += 1
        if n_done % 8 == 0: event(out, 'FINAL_PROGRESS', scenes=n_done, seconds=time.monotonic()-t0, mismatches=len(mismatches))
    for c, rows in writers.items(): Q.write_rows(out/'final'/f'{c}.jsonl', rows)
    rep = dict(at=Q.now(), first_question_generated_at=first_q, seal_at=Q.load(out/'SEAL.json')['at'], scenes=len(scenes), conditions=conds, rows={c: len(r) for c, r in writers.items()}, seal_mismatches=mismatches, recompile_deterministic=not mismatches,
               seconds=time.monotonic()-t0, generation_scenes=sorted(gen_scenes), master_verified_samples={c: int(sum(1 for r in rows if r.get('master_verified'))) for c, rows in writers.items()}, master_failures={c: int(sum(1 for r in rows if r.get('master_verified') is False)) for c, rows in writers.items()})
    Q.dump(out/'EVAL_DONE.json', rep); return rep

# ---------------------------------------------------------------- F3 controls
def controls(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'CONTROLS.json'): return Q.load(out/'CONTROLS.json')
    key = ('SHARED_TAIL', 0)
    if key not in banks: Q.dump(out/'CONTROLS.json', dict(at=Q.now(), status='NOT_MEASURED', reason='SHARED_TAIL seed 0 absent')); return Q.load(out/'CONTROLS.json')
    entry = banks[key]; rows = {'LEARNED': [], 'RANDOM_MATCHED': [], 'WRONG_ADDRESS': []}; t0 = time.monotonic(); import torch; g = torch.Generator(device='cpu').manual_seed(2609)
    for s in scenes:
        Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', 'early', 0); src = bb.compile(b['prefix']); src_rows = []; score_shared(bb, b, src, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}
        pl = plan(entry, b); art = T.compile_plan(bb, b, pl); score_shared(bb, b, art, src_preds, 'LEARNED', rows['LEARNED'], extra=dict(energy=art['realized']['energy']))
        norms = torch.as_tensor(art['realized']['realized_norms'][0], dtype=torch.float32); d = torch.randn(len(pl['index']), bb.hidden, generator=g); d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-12)*norms.unsqueeze(-1)
        rart = bb.compile(b['prefix'], maps=[dict(mode='delta', delta=d.numpy())], index=pl['index'], site=pl['site']); score_shared(bb, b, rart, src_preds, 'RANDOM_MATCHED', rows['RANDOM_MATCHED'], extra=dict(matched_norm=float(norms.mean()), energy=rart['realized']['energy']))
        # wrong address: the same bank applied to another holder's record on the same project; gold under the intended world and under the wrong-address world
        wa = (s.target_actor+1) % len(s.actors); wv = 1-s.values[wa][s.target_project]; wtext, wspans = Q.prefix_text(s, proc, 'early', 0)
        wb = dict(b, addresses=[tuple(bb.clause_positions(wtext, wspans[(wa, s.target_project)]))], values=[wv]); wart = T.compile_plan(bb, wb, plan(entry, wb))
        wworld = Q.P.set_value(s, Q.P.Edit(wa, s.target_project, wv)); wq = {q['query_id']: q for draw in (0, 1) for q in Q.P.questions(s, edited=wworld, challenge=True, draw=draw)}
        wrows = []; score_shared(bb, b, wart, src_preds, 'WRONG_ADDRESS', wrows, extra=dict(wrong_actor=s.actors[wa], wrong_new_value=wv, energy=wart['realized']['energy']))
        for r in wrows: r['wrong_world_gold'] = int(wq[r['query_id']]['gold']); r['wrong_world_changed'] = bool(wq[r['query_id']]['changed'])
        rows['WRONG_ADDRESS'].extend(wrows)
        if ('QUERY_TAIL', 0) in banks:   # serving-mode control: the aware arm scored with batch-1 prefix compiles vs its batched-prefix path
            score_aware(bb, b, banks[('QUERY_TAIL', 0)], src, src_preds, 'QUERY_TAIL_s0_BATCHED_PREFIX', rows.setdefault('QUERY_TAIL_s0_BATCHED_PREFIX', []))
            score_aware(bb, b, banks[('QUERY_TAIL', 0)], src, src_preds, 'QUERY_TAIL_s0_SINGLE_PREFIX', rows.setdefault('QUERY_TAIL_s0_SINGLE_PREFIX', []), single_prefix=True)
    for c, rs in rows.items(): Q.write_rows(out/'controls'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED'); Q.dump(out/'CONTROLS.json', rep); return rep

# ---------------------------------------------------------------- F4 equal-energy diagnostic
def energy_panel(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'ENERGY.json'): return Q.load(out/'ENERGY.json')
    keys = [(a, s) for s in Q.SEEDS for a in ('SHARED_CLAUSE', 'SHARED_TAIL') if (a, s) in banks]
    if len(keys) < 2: Q.dump(out/'ENERGY.json', dict(at=Q.now(), status='NOT_MEASURED', reason='both shared arms required')); return Q.load(out/'ENERGY.json')
    rows = {}; meta = []; t0 = time.monotonic(); site = model['site']
    for s in scenes:
        for order in Q.ORDERS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order, 0); src = bb.compile(b['prefix'], capture={site}); src_rows = []; score_shared(bb, b, src, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}
            for seed in Q.SEEDS:
                deltas = {}
                for arm in ('SHARED_CLAUSE', 'SHARED_TAIL'):
                    entry = banks[(arm, seed)]; pl = plan(entry, b); art = bb.compile(b['prefix'], maps=pl['maps'], index=pl['index'], site=pl['site'], capture={site})
                    delta = _delta_from(bb, b, pl, art, site); deltas[arm] = dict(index=pl['index'], delta=delta, frob=float(np.linalg.norm(delta)))
                    score_shared(bb, b, art, src_preds, f'{arm}_s{seed}_unmodified', rows.setdefault(f'{arm}_s{seed}_unmodified', []), extra=dict(frobenius=deltas[arm]['frob'], energy=art['realized']['energy']))
                na, nb = deltas['SHARED_CLAUSE']['frob'], deltas['SHARED_TAIL']['frob']; common = Q.P.common_frobenius(deltas['SHARED_CLAUSE']['delta'], deltas['SHARED_TAIL']['delta'])
                meta.append(dict(scene_id=s.scene_id, order=order, seed=seed, clause_frobenius=na, tail_frobenius=nb, common=min(na, nb) if common else None, zero_norm=common is None, clause_tokens=len(deltas['SHARED_CLAUSE']['index']), tail_tokens=len(deltas['SHARED_TAIL']['index'])))
                if common is None: continue
                for arm, dl in zip(('SHARED_CLAUSE', 'SHARED_TAIL'), common):
                    art = bb.compile(b['prefix'], maps=[dict(mode='delta', delta=dl.astype(np.float32))], index=deltas[arm]['index'], site=site)
                    score_shared(bb, b, art, src_preds, f'{arm}_s{seed}_matched', rows.setdefault(f'{arm}_s{seed}_matched', []), extra=dict(frobenius=float(np.linalg.norm(dl)), energy=art['realized']['energy']))
    for c, rs in rows.items(): Q.write_rows(out/'energy'/f'{c}.jsonl', rs)
    Q.write_rows(out/'energy'/'META.jsonl', meta)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], rows={c: len(r) for c, r in rows.items()}, zero_norm_cases=int(sum(m['zero_norm'] for m in meta)), seconds=time.monotonic()-t0, status='MEASURED',
               rule='both full-prefix update tensors scaled DOWN to the smaller nonzero Frobenius norm, computed before any question; direction unchanged; unmodified and matched panels are distinct'); Q.dump(out/'ENERGY.json', rep); return rep

def _delta_from(bb, b, pl, art, site):
    """Realized per-position update tensor of a compiled artifact: post-edit states minus the pre-edit states at the edited positions."""
    pre = art['captured_full'][site][0, pl['index']].float().cpu().numpy(); post = art.get('_after_states')
    if post is None or post.shape != pre.shape: raise RuntimeError('after-states not captured')
    return (post-pre).astype(np.float32)

# ---------------------------------------------------------------- F5 edit programs
def programs(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'PROGRAMS.json'): return Q.load(out/'PROGRAMS.json')
    keys = [k for k in banks if k[0] in ('SHARED_TAIL', 'SHARED_CLAUSE', 'QUERY_TAIL')]; rows = {}; t0 = time.monotonic(); n = 0
    for s in scenes:
        for program in Q.PROGRAMS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, program, 'early', 0); src = bb.compile(b['prefix'], capture={model['site']}); src_rows = []
            score_shared(bb, b, src, None, 'SOURCE', src_rows, extra=dict(program=program)); src_preds = {r['query_id']: r for r in src_rows}; rows.setdefault('SOURCE', []).extend(src_rows)
            ntext, _ = Q.prefix_text(s, proc, 'early', 0, world=b['world']); natural_rows(bb, b, ntext, src_preds, 'NATURAL_FINAL_WORLD', rows.setdefault('NATURAL_FINAL_WORLD', []), extra=dict(program=program))
            for key in keys:
                entry = banks[key]; nm = cname(key)
                if entry['aware']: score_aware(bb, b, entry, src, src_preds, nm, rows.setdefault(nm, []), extra=dict(program=program))
                else:
                    art = compile_shared(bb, b, entry); score_shared(bb, b, art, src_preds, nm, rows.setdefault(nm, []), extra=dict(program=program, edited_tokens=art['realized']['positions'], energy=art['realized']['energy'], maps=art['realized']['maps']))
        n += 1
        if n % 8 == 0: event(out, 'PROGRAMS_PROGRESS', scenes=n, seconds=time.monotonic()-t0)
    for c, rs in rows.items(): Q.write_rows(out/'programs'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], programs=list(Q.PROGRAMS), rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED',
               note='every program executes inside ONE prefix compilation; z recomputed from the current edited residual before each map; gold from the true final world; parameters never trained on multi-edit programs'); Q.dump(out/'PROGRAMS.json', rep); return rep

# ---------------------------------------------------------------- F6 receiver / majority aggregator workflow
def workflow(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'WORKFLOW.json'): return Q.load(out/'WORKFLOW.json')
    keys = [k for k in banks if k[0] in ('SHARED_TAIL', 'QUERY_TAIL', 'V1_LATE')]; rows = []; t0 = time.monotonic(); n = 0; gp = bb.generation_prompt()
    for s in scenes:
        Q.budget(deadline, 300); b = prefix_bundle(bb, s, proc, 'single', 'original', 0); a = s.target_actor; p = s.target_project; idx = [a, (a+1) % len(s.actors), (a+2) % len(s.actors)]; who = [s.actors[i] for i in idx]; P_ = s.projects[p]
        world = Q.edited_world(s, 'single'); gold_src = int(sum(s.values[i][p] for i in idx) >= 2); gold_tgt = int(sum(world.values[i][p] for i in idx) >= 2)
        texts = {'SOURCE': b['text'], 'COUNTERFACTUAL': Q.prefix_text(s, proc, 'original', 0, world=world)[0], 'TEXT_CORRECTION': Q.prefix_text(s, proc, 'original', 0, correction=True)[0]}
        src = bb.compile(b['prefix'], capture={model['site']}|({27} if any(k[0] == 'V1_LATE' for k in keys) else set())); arts = {'SOURCE': src, 'COUNTERFACTUAL': bb.compile(bb.prefix_ids(texts['COUNTERFACTUAL'])), 'TEXT_CORRECTION': bb.compile(bb.prefix_ids(texts['TEXT_CORRECTION']))}
        conds = [('SOURCE', None), ('COUNTERFACTUAL', None), ('TEXT_CORRECTION', None)]+[(cname(k), k) for k in keys]
        for nm, key in conds:
            entry = banks[key] if key else None; text = texts.get(nm, b['text']); art = arts.get(nm)
            if entry is not None and not entry['aware'] and key[0] != 'V1_LATE': art = compile_shared(bb, b, entry)
            reports = {}; ok = True; qfeat = None
            encs = [bb.encode(text, Q.suffix_text(RECEIVER_Q.format(who=w, project=P_))) for w in who]; lids = [bb.label_ids(e['full_text'], ('No', 'Yes')) for e in encs]
            if entry is not None and entry['aware']: qfeat = bb.ask_capture_batch(src, [e['suffix_ids'] for e in encs], entry['site'])
            for i, w in enumerate(who):
                if entry is not None and entry['aware']: a1 = T.compile_plan(bb, b, plan(entry, b, q_batch=qfeat[i:i+1])); res = bb.ask(a1, encs[i]['suffix_ids'], lids[i], generate=True)
                elif key is not None and key[0] == 'V1_LATE': res = bb.ask(src, encs[i]['suffix_ids'], lids[i], late=late_plan(entry, b), generate=True)
                else: res = bb.ask(art, encs[i]['suffix_ids'], lids[i], generate=True)
                sym, status = Q.parse_symbol(res['generation']['text'], ('No', 'Yes')); truth = (world if nm != 'SOURCE' else s).values[idx[i]][p]
                reports[w] = dict(text=res['generation']['text'], symbol=sym, status=status, score_prediction=res['prediction'], gold=int(truth), correct=(sym == ('Yes' if truth else 'No')), truncated=res['generation']['truncated']); ok = ok and sym is not None
            for phrasing in (0, 1):
                if ok:
                    lines = [PHRASINGS[phrasing].format(who=w, project=P_, verb=('supports' if reports[w]['symbol'] == 'Yes' else 'opposes'), stance=('in favor of' if reports[w]['symbol'] == 'Yes' else 'against')) for w in who]
                    agg = (f'Three committee members were asked separately about the {P_} proposal.\n'+'\n'.join(lines)+f'\nIs the {P_} proposal supported by a majority of {who[0]}, {who[1]} and {who[2]}?\nAnswer with exactly one of No, Yes.'+Q.INSTRUCTION)
                    full = bb.tokenizer.apply_chat_template([{'role': 'user', 'content': agg}], tokenize=False, add_generation_prompt=True, **bb.template_kwargs); ids = bb.tokenizer.encode(full, add_special_tokens=False); al = bb.label_ids(full, ('No', 'Yes'))
                    gres = bb.ask(bb.compile(ids[:1]), ids[1:], al, generate=True); asym, astatus = Q.parse_symbol(gres['generation']['text'], ('No', 'Yes')); final = (1 if asym == 'Yes' else 0) if asym is not None else None; atext = gres['generation']['text']
                else: agg = None; asym = None; astatus = 'RECEIVER_FAILED'; final = None; atext = None
                gold = gold_tgt if nm != 'SOURCE' else gold_src
                rows.append(dict(scene_id=s.scene_id, condition=nm, method=(key[0] if key else nm), seed=(key[1] if key else None), phrasing=phrasing, holders=who, project=P_, receiver=reports, receivers_all_correct=all(r['correct'] for r in reports.values()),
                                 receiver_accuracy=float(np.mean([r['correct'] for r in reports.values()])), aggregator_prompt=agg, aggregator_text=atext, aggregator_answer=asym, aggregator_status=astatus, chain_output=final, gold_source=gold_src, gold_target=gold_tgt,
                                 majority_changes=gold_src != gold_tgt, correct=(final == gold), correct_vs_target=(final == gold_tgt), cell=list(Q.cell(s)), source_prefix_tokens=len(b['prefix'])))
        n += 1
        if n % 8 == 0: event(out, 'WORKFLOW_PROGRESS', scenes=n, seconds=time.monotonic()-t0)
    Q.write_rows(out/'workflow'/'rows.jsonl', rows)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], conditions=[c for c, _ in conds], rows=len(rows), seconds=time.monotonic()-t0, status='MEASURED', generation_prompt=gp, phrasings=PHRASINGS,
               note='three receiver calls per episode generate actual reports from the same compiled artifact; their parsed symbols go to a separate unedited aggregator; invalid outputs are failures (no gold substitution); independent WORKFLOW scenes'); Q.dump(out/'WORKFLOW.json', rep); return rep

# ---------------------------------------------------------------- F7 physical witnesses
def witnesses(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'WITNESS_INDEX.json'): return Q.load(out/'WITNESS_INDEX.json')
    site = model['site']; wdir = out/'witness'; wdir.mkdir(parents=True, exist_ok=True); index = []
    for s in scenes:
        Q.budget(deadline, 300); b = prefix_bundle(bb, s, proc, 'single', 'early', 0); world = Q.edited_world(s, 'single'); ctext, _ = Q.prefix_text(s, proc, 'early', 0, world=world)
        conds = {'SOURCE': (b['prefix'], None), 'COUNTERFACTUAL': (bb.prefix_ids(ctext), None)}
        for seed in Q.SEEDS:
            if ('SHARED_TAIL', seed) in banks: conds[f'SHARED_TAIL_s{seed}'] = (b['prefix'], banks[('SHARED_TAIL', seed)])
        for nm, (pre, entry) in conds.items():
            if entry is None: art = bb.compile(pre, capture={site}, capture_next={site})
            else: pl = plan(entry, b); art = bb.compile(pre, maps=pl['maps'], index=pl['index'], site=pl['site'], capture={site}, capture_next={site})
            k0, v0 = bb.layer_kv(art['cache'], site); k1, v1 = bb.layer_kv(art['cache'], site+1); post = art['captured'][site].float().cpu().numpy(); nxt = art['next_input'][site].float().cpu().numpy()
            edited = art.get('_after_states'); path = wdir/f'{s.scene_id}_{nm}.npz'
            np.savez(path, prefix_ids=np.asarray(pre, '<i8'), post_block_residual=post, next_block_input=nxt, keys_site=k0, values_site=v0, keys_next=k1, values_next=v1, edited_positions=np.asarray(pl['index'] if entry else [], '<i8'),
                     edited_states=(edited if edited is not None else np.zeros((0, bb.hidden), np.float32)))
            index.append(dict(scene_id=s.scene_id, condition=nm, path=str(path), sha256=Q.sha(path), bytes=path.stat().st_size, artifact_hash=art['hash'], prefix_tokens=len(pre), site=site, next_site=site+1, edited_tokens=len(pl['index']) if entry else 0,
                              energy=art['realized']['energy'] if entry else 0.0, next_input_equals_post_block=bool(np.array_equal(post, nxt)), checkpoint=entry['checkpoint'] if entry else None))
    rep = dict(at=Q.now(), site=site, scenes=[s.scene_id for s in scenes], files=index, note='post-block residual at the site (all prefix positions), the actual input to the next block, and key/value tensors of the site and next layer; float32 upcasts of the BF16 states (exact)')
    Q.dump(out/'WITNESS_INDEX.json', rep); return rep

# ---------------------------------------------------------------- natural panel (unqualified actor)
def natural_panel(bb, fin, qual, out, deadline, model, size):
    if done(out/'NATURAL_PANEL.json'): return Q.load(out/'NATURAL_PANEL.json')
    global CHUNK; old_chunk = CHUNK; CHUNK = 1      # unqualified actor: single-ask serving (its batched-ask check flipped a choice); restored on exit
    try: return _natural_panel(bb, fin, qual, out, deadline, model, size)
    finally: CHUNK = old_chunk

def _natural_panel(bb, fin, qual, out, deadline, model, size):
    proc = qual['natural_panel_procedure']; n = Q.SIZES[model['key']]['SMALL' if size == 'SMALL' else 'FULL']['final']; scenes = Q.subset(fin, n, 'SRS2-FINAL'); rows = {}; t0 = time.monotonic()
    for s in scenes:
        for order in Q.ORDERS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order, 0); src = bb.compile(b['prefix']); src_rows = []; score_shared(bb, b, src, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}; rows.setdefault('SOURCE', []).extend(src_rows)
            ctext, _ = Q.prefix_text(s, proc, order, 0, world=b['world']); natural_rows(bb, b, ctext, src_preds, 'COUNTERFACTUAL', rows.setdefault('COUNTERFACTUAL', []))
    for c, rs in rows.items(): Q.write_rows(out/'natural'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), procedure=proc, scenes=len(scenes), rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED', serving='single asks (batch 1)', note='native competence only; no learned intervention is evaluated or claimed'); Q.dump(out/'NATURAL_PANEL.json', rep); return rep

# ---------------------------------------------------------------- driver
def phase_f(bb, out, deadline, model, fit, cal, fin, wf, qdir, ddir, size, reserve_hours):
    freeze = Q.load(ddir/'FREEZE.json'); proc = freeze['procedure']; out.mkdir(parents=True, exist_ok=True)
    if not done(out/'FREEZE_VERIFIED.json'): Q.dump(out/'FREEZE_VERIFIED.json', dict(at=Q.now(), freeze_sha256=Q.sha(ddir/'FREEZE.json'), selection=freeze['selection'], procedure=proc))
    banks = load_banks(bb, freeze, ddir, model); event(out, 'BANKS', conditions=[cname(k) for k in banks], strengths={cname(k): v['strength'] for k, v in banks.items()})
    bench = Q.load(qdir/'BENCHMARK.json')['seconds']; scope = choose_scope(bb, out, deadline, model, size, reserve_hours, banks, bench, len(fin)); sizes = scope['sizes']; event(out, 'SCOPE', scope=scope['scope'], sizes=sizes, predicted=scope['predicted_seconds'][scope['scope']]['total'], remaining=scope['remaining_seconds'])
    final_scenes = Q.subset(fin, sizes['final'], 'SRS2-FINAL'); algebra = Q.subset(final_scenes, sizes['algebra'], 'SRS2-ALGEBRA'); wf_scenes = Q.subset(wf, sizes['workflow'], 'SRS2-WORKFLOW')
    energy = Q.subset(final_scenes, min(Q.PANELS['energy'], len(final_scenes)), 'SRS2-ENERGY'); control = Q.subset(final_scenes, min(Q.PANELS['control'], len(final_scenes)), 'SRS2-CONTROL')
    variant1 = {s.scene_id for s in Q.subset(final_scenes, min(Q.PANELS['variant1'], len(final_scenes)), 'SRS2-VARIANT1')}; witness = sorted(final_scenes, key=lambda s: Q.hash_rank('SRS2-WITNESS', s.scene_id))[:Q.PANELS['witness']]
    gen_scenes = {s.scene_id for s in Q.subset(final_scenes, min(Q.PANELS['generation'], len(final_scenes)), 'SRS2-GENERATION')}
    if not done(out/'SUBSETS.json'): Q.dump(out/'SUBSETS.json', dict(at=Q.now(), final=[s.scene_id for s in final_scenes], algebra=[s.scene_id for s in algebra], workflow=[s.scene_id for s in wf_scenes], energy=[s.scene_id for s in energy], control=[s.scene_id for s in control], variant1=sorted(variant1), witness=[s.scene_id for s in witness], generation=sorted(gen_scenes)))
    sl = seal(bb, final_scenes, variant1, proc, banks, out, deadline, model); event(out, 'SEALED', artifacts=sl['artifacts'], sha256=sl['seal_sha256'])
    ev = evaluate_final(bb, final_scenes, variant1, proc, banks, out, deadline, model, gen_scenes); event(out, 'FINAL_EVAL', rows=ev['rows'], mismatches=len(ev['seal_mismatches']))
    for stage, fn, arg in (('WITNESS', witnesses, witness), ('CONTROLS', controls, control), ('ENERGY', energy_panel, energy), ('PROGRAMS', programs, algebra), ('WORKFLOW', workflow, wf_scenes)):
        marker = {'WITNESS': 'WITNESS_INDEX.json'}.get(stage, f'{stage}.json')
        try:
            if Q.budget(deadline, 0) < 900:
                if not done(out/marker): Q.dump(out/marker, dict(at=Q.now(), status='NOT_MEASURED', reason='insufficient remaining time'))
                event(out, stage, status='NOT_MEASURED'); continue
            r = fn(bb, arg, proc, banks, out, deadline, model); event(out, stage, status=(r or {}).get('status', 'MEASURED'))
        except TimeoutError:
            if not done(out/marker): Q.dump(out/marker, dict(at=Q.now(), status='NOT_MEASURED', reason='resource stop'))
            event(out, stage, status='NOT_MEASURED'); raise
    return dict(status='FINAL_COMPLETE', scope=scope['scope'])
