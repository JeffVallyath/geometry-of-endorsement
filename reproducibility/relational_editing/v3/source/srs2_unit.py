# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""SRS2 scientific unit inside the durable scientific-unit host.  Phases (markers make every stage re-enterable):

Q  interface/anchors (per actor): load -> native qualification on the fixed 96-request CAL battery (procedures C1 -> C1S -> T4, source and
   natural-counterfactual inputs, <=32 greedy tokens) -> [Gemma] anchor replay of 8 fixed old FINAL scenes per seed with the exact V1 prompts,
   factors and cap against the recorded V1 rows -> E0 interface panel (cached vs full forward, batched-ask vs single-ask, zero/self edit,
   clone, recompile, query-order, master, next-block propagation, QUERY_TAIL-at-init == SHARED_TAIL for nonzero q) -> gradient check for
   all arms -> disposable timing benchmark.
D  development (srs2_train.phase_d) -> FREEZE.json before any FINAL/WORKFLOW scene is read.
F  fresh evaluation (srs2_final.phase_f): seal -> FINAL -> controls -> energy-matched -> edit programs -> workflow -> witnesses.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # allocator setting only (peak 37.4 GB of 40 GB in the r2 benchmark); no numerical effect
import random
import time
import traceback
from pathlib import Path
import numpy as np
import srs2_common as Q
import srs2_adapter as A

BASE = 'external-artifacts'

def event(output, name, **fields):
    row = dict(at=Q.now(), event=name, **fields)
    with (Path(output)/'stages.jsonl').open('a', encoding='utf8', newline='\n') as f: f.write(Q.canonical(row)+'\n')
    print(Q.canonical(row), flush=True)

def done(path): return Path(path).is_file()

def v1_dirs(family='PREFIX_DISTRIBUTED'):
    tag = {'PREFIX_DISTRIBUTED': 'PREFIX_DISTRIBUTED@15_lr0.001', 'LATE': 'LATE@27_lr0.0003'}[family]
    return {s: Q.HIST/'frozen_factors'/f'{tag}_s{s}' for s in Q.SEEDS}

# ---------------------------------------------------------------- native rows
def native_row(bb, s, q, draw, proc, world=None, generate=True, order='original'):
    text, _ = Q.prefix_text(s, proc, order, world=world); enc = bb.encode(text, Q.suffix_text(q['text'])); lids = bb.label_ids(enc['full_text'], q['labels'])
    art = bb.compile(enc['prefix_ids']); res = bb.ask(art, enc['suffix_ids'], lids, generate=generate)
    row = Q.query_record(s, q, s.split, draw, order); gold = q['source_gold'] if world is None else int(Q.P.evaluate(world, q['spec']))   # gold of the rendered world
    row.update(procedure=proc, world='source' if world is None else 'counterfactual', prefix_tokens=len(enc['prefix_ids']), suffix_tokens=len(enc['suffix_ids']), logps=res['logps'], score_prediction=res['prediction'],
               answer_mass=res['answer_mass'], argmax_in_labels=res['argmax_in_labels'], target_index=int(gold), score_correct=res['prediction'] == gold, prefix_hash=art['hash'])
    if generate:
        sym, status = Q.parse_symbol(res['generation']['text'], q['labels']); pred = list(q['labels']).index(sym) if sym is not None else None
        row.update(generation=res['generation']['text'], generation_tokens=len(res['generation']['tokens']), truncated=res['generation']['truncated'], parse_status=status, generation_prediction=pred,
                   generation_correct=pred == gold, first_token_is_argmax=res['generation']['first_token_is_argmax'])
    return row

# ---------------------------------------------------------------- qualification battery (fixed by hash before any outcome)
def qualification_requests(cal):
    """96 CAL requests: direct/negation 8 cells (stance x target relevance x truth) x 8, equality 4 cells (stance x truth) x 8; hash-ordered."""
    pool = []
    for s in cal:
        for draw in (0, 1):
            for q in Q.questions(s, 'CAL', draw, world=s): pool.append((s, q, draw))
    out = []
    for fam_group, fams, rel_split, per, total in (('dn', ('direct', 'opposes'), True, 8, 64), ('eq', ('same',), False, 8, 32)):
        chosen = []; leftovers = []
        for stance in (0, 1):
            for truth in (0, 1):
                for rel in ((True, False) if rel_split else (None,)):
                    c = [x for x in pool if x[1]['family'] in fams and x[0].values[x[0].target_actor][x[0].target_project] == stance and x[1]['source_gold'] == truth
                         and (rel is None or ((x[1]['spec']['a'] == x[0].target_actor or x[1]['spec'].get('b') == x[0].target_actor) and x[1]['spec']['p'] == x[0].target_project) == rel)]
                    c = sorted(c, key=lambda x: Q.hash_rank('qual', x[1]['query_id'])); chosen.extend(c[:per]); leftovers.extend(c[per:])
        # a cell the fixed CAL pool cannot fill (equality has one question per draw) is topped up from the other cells in the same hash order
        chosen.extend(sorted(leftovers, key=lambda x: Q.hash_rank('qual', x[1]['query_id']))[:total-len(chosen)])
        if len(chosen) != total: raise ValueError(f'qualification group short: {fam_group} -> {len(chosen)}')
        out.extend(chosen)
    if len(out) != Q.QUAL_N or len({x[1]['query_id'] for x in out}) != Q.QUAL_N: raise ValueError('qualification battery size')
    return out

def qualify(bb, cal, out, deadline, model):
    if done(out/'QUALIFICATION.json'): return Q.load(out/'QUALIFICATION.json')
    reqs = qualification_requests(cal); attempts = []; selected = None; generations = 0
    for proc in Q.PROCEDURES:
        rows = []
        for s, q, draw in reqs:
            Q.budget(deadline, 120); rows.append(native_row(bb, s, q, draw, proc)); rows.append(native_row(bb, s, q, draw, proc, world=Q.edited_world(s, 'single'))); generations += 2
        Q.write_rows(out/f'qualification_{proc}.jsonl', rows); summary = {}
        for world in ('source', 'counterfactual', 'pooled'):
            rs = [r for r in rows if world == 'pooled' or r['world'] == world]; dn = [r for r in rs if r['family'] in ('direct', 'opposes')]; eq = [r for r in rs if r['family'] == 'same']
            summary[world] = dict(n=len(rs), parse_rate=float(np.mean([r['parse_status'].startswith('OK') for r in rs])), direct_negation_accuracy=float(np.mean([r['generation_correct'] for r in dn])),
                                  equality_accuracy=float(np.mean([r['generation_correct'] for r in eq])), score_accuracy=float(np.mean([r['score_correct'] for r in rs])), truncated=int(sum(r['truncated'] for r in rs)),
                                  generation_score_agreement=float(np.mean([r['generation_prediction'] == r['score_prediction'] for r in rs])), mean_new_tokens=float(np.mean([r['generation_tokens'] for r in rs])),
                                  by_family={f: float(np.mean([r['generation_correct'] for r in rs if r['family'] == f])) for f in ('direct', 'opposes', 'same')})
            summary[world]['qualified'] = summary[world]['parse_rate'] >= Q.QUAL_PARSE and summary[world]['direct_negation_accuracy'] >= Q.QUAL_DIRECT and summary[world]['equality_accuracy'] >= Q.QUAL_EQUALITY
        ok = summary['source']['qualified'] and summary['counterfactual']['qualified']
        attempts.append(dict(procedure=proc, summary=summary, generations=len(rows), qualified=ok)); event(out, 'QUALIFICATION_ATTEMPT', procedure=proc, qualified=ok, source=(summary['source']['parse_rate'], summary['source']['direct_negation_accuracy'], summary['source']['equality_accuracy']), counterfactual=(summary['counterfactual']['parse_rate'], summary['counterfactual']['direct_negation_accuracy'], summary['counterfactual']['equality_accuracy']))
        if ok: selected = proc; break
    best_native = max(attempts, key=lambda a: (a['summary']['pooled']['direct_negation_accuracy']+a['summary']['pooled']['equality_accuracy'], a['summary']['pooled']['parse_rate']))['procedure']
    rep = dict(at=Q.now(), actor=model['key'], selected_procedure=selected, natural_panel_procedure=best_native, attempts=attempts, total_generations=generations, requests=Q.QUAL_N,
               rule='first procedure (C1 -> C1S -> T4) with >=95% parseable generations (<=32 new tokens), >=90% direct/negation accuracy and >=85% equality accuracy on the fixed 96-request CAL battery, required on BOTH the source and the natural-counterfactual inputs; at most three procedures; no outcome-dependent exclusion',
               battery=dict(direct_negation=64, equality=32, balanced_by='stance x target relevance x truth (direct/negation), stance x truth (equality)'), template=bb.generation_prompt())
    Q.dump(out/'QUALIFICATION.json', rep); return rep

# ---------------------------------------------------------------- anchors (Gemma only): exact V1 replay of 8 old FINAL scenes per seed
def anchors(bb, out, deadline, model):
    if done(out/'ANCHORS.json'): return Q.load(out/'ANCHORS.json')
    if model['key'] != 'gemma' or bb.cfg.get('id') == 'tiny': Q.dump(out/'ANCHORS.json', dict(at=Q.now(), status='NOT_APPLICABLE', actor=model['key'], tiny=bb.cfg.get('id') == 'tiny')); return Q.load(out/'ANCHORS.json')
    from srs2_editor import LegacyBank
    spec = Q.load(Q.ANCHORS/'ANCHORS.json'); old = {s.scene_id: s for s in Q.old_scenes('FINAL')}; QB = Q.QB; report = dict(at=Q.now(), status=None, seeds={}); rows_out = []; t0 = time.monotonic()
    tol = dict(same_choice_required=True, max_abs_logp=0.05, note='inherited V1 verified tolerance: same choice on every row; |dlogp| differences are BF16 shape/kernel effects recorded not zero-tolerance; artifact hash equality additionally proves bit-identical compiled contexts')
    for seed in Q.SEEDS:
        bank = LegacyBank(bb, v1_dirs('PREFIX_DISTRIBUTED')[seed], f'V1_SHARED_TAIL_s{seed}'); ref = {r['query_id']: r for r in Q.read_rows(Q.ANCHORS/f'PREFIX_DISTRIBUTED_s{seed}.jsonl')}
        src_ref = {r['query_id']: r for r in Q.read_rows(Q.ANCHORS/'SOURCE.jsonl')}; seal = {r['scene_id']: r for r in Q.read_rows(Q.ANCHORS/'V1_r4_gemma_final_phase_SEAL.jsonl')}
        stats = dict(rows=0, same_choice=0, max_abs_logp=0.0, source_same_choice=0, source_max_abs_logp=0.0, hash_equal=0, source_hash_equal=0, scenes=0, mismatches=[])
        for sid in spec['anchors'][str(seed)]:
            Q.budget(deadline, 120); s = old[sid]; text, span = QB.prefix_text(s, 'C1'); pre = bb.prefix_ids(text); cp = bb.clause_positions(text, span); req = QB.compile_request(s, 'C1', bb.cfg['revision'], 'anchor')
            src = bb.compile(pre); fn = bank.fn(req.new_value, 1.0); art = bb.compile(pre, maps=[dict(fn=fn, clause_positions=cp)], index=list(range(cp[0], len(pre))), site=bank.site)
            stats['scenes'] += 1; stats['hash_equal'] += int(art['hash'] == seal[sid]['artifacts'][f'PREFIX_DISTRIBUTED_s{seed}']['hash']); stats['source_hash_equal'] += int(src['hash'] == seal[sid]['artifacts']['SOURCE']['hash'])
            for draw in (0, 1):
                for q in QB.questions(s, 'FINAL', draw):
                    enc = bb.encode(text, QB.suffix_text(q, 'C1')); lids = bb.label_ids(enc['full_text'], q.labels)
                    r_edit = bb.ask(art, enc['suffix_ids'], lids); r_src = bb.ask(src, enc['suffix_ids'], lids); rr = ref[q.query_id]; sr = src_ref[q.query_id]
                    d = float(np.max(np.abs(np.asarray(r_edit['logps'])-np.asarray(rr['logps'])))); ds = float(np.max(np.abs(np.asarray(r_src['logps'])-np.asarray(sr['logps']))))
                    stats['rows'] += 1; stats['same_choice'] += int(r_edit['prediction'] == rr['prediction']); stats['max_abs_logp'] = max(stats['max_abs_logp'], d)
                    stats['source_same_choice'] += int(r_src['prediction'] == sr['prediction']); stats['source_max_abs_logp'] = max(stats['source_max_abs_logp'], ds)
                    if r_edit['prediction'] != rr['prediction'] or d > tol['max_abs_logp']: stats['mismatches'].append(dict(query_id=q.query_id, replay=r_edit['logps'], recorded=rr['logps']))
                    rows_out.append(dict(seed=seed, scene_id=sid, query_id=q.query_id, replay_logps=r_edit['logps'], recorded_logps=rr['logps'], replay_prediction=r_edit['prediction'], recorded_prediction=rr['prediction'], gold=rr['gold'],
                                         source_replay_logps=r_src['logps'], source_recorded_logps=sr['logps'], artifact_hash=art['hash'], recorded_artifact_hash=rr['artifact_hash'], realized_norm=float(np.mean(art['realized']['realized_norms'][0])), recorded_realized_norm=rr.get('realized_norm')))
        stats['pass_'] = stats['same_choice'] == stats['rows'] and stats['max_abs_logp'] <= tol['max_abs_logp'] and stats['source_same_choice'] == stats['rows']
        report['seeds'][str(seed)] = dict(stats, factors=bank.files, cap=bank.cap, site=bank.site)
    Q.write_rows(out/'anchor_rows.jsonl', rows_out)
    report.update(status='TECHNICAL_PASS' if all(v['pass_'] for v in report['seeds'].values()) else 'TECHNICAL_MISMATCH', tolerance=tol, seconds=time.monotonic()-t0, anchors=spec['anchors'],
                  inputs='old V1 prompts (old FIT C1 demonstrations, old FINAL scenes, old questions/labels), V1 PREFIX_DISTRIBUTED@15 epoch-4 factors and cap 103.18592071533203 before strength 1, BF16 backbone, FP32 softcapped readout; no successor data or query feature entered this replay')
    Q.dump(out/'ANCHORS.json', report); return report

# ---------------------------------------------------------------- E0 interface checks on the real model
def check_requests(cal, n, salt):
    rows = [(s, q, draw) for s in cal for draw in (0, 1) for q in Q.questions(s, 'CAL', draw)]
    return sorted(rows, key=lambda x: Q.hash_rank(salt, x[1]['query_id']))[:n]

def interface_checks(bb, cal, out, deadline, model, proc, mu_q_dummy=None):
    if done(out/'INTERFACE_CHECK.json'): return Q.load(out/'INTERFACE_CHECK.json')
    import torch; from srs2_editor import Bank, LegacyBank
    site = model['site']; reqs = check_requests(cal, 16, 'e0'); rows = []; zero_ok = []; clone_ok = []; emul = []; batch_diffs = []; batch_same = []; prefix_batch_diffs = []; prefix_batch_same = []
    zero_bank = Bank.fresh(bb, 'SHARED_TAIL', site, 1.0, 0, model['key'], np.zeros(bb.hidden, np.float32), np.zeros(bb.hidden, np.float32))
    by_scene = {}
    for s, q, draw in reqs: by_scene.setdefault(s.scene_id, (s, []))[1].append((q, draw))
    for sid, (s, qs) in by_scene.items():
        Q.budget(deadline, 120); text, spans = Q.prefix_text(s, proc); cp = bb.clause_positions(text, spans[(s.target_actor, s.target_project)]); pre = bb.prefix_ids(text)
        art = bb.compile(pre, capture={site}, capture_next={site}); tail_idx = list(range(cp[0], len(pre)))
        for footprint, idx in (('clause', list(cp)), ('tail', tail_idx)):
            za = bb.compile(pre, maps=[dict(fn=zero_bank.fn(1, 1.0, record=False), clause_positions=list(cp))], index=idx, site=site); zero_ok.append(za['hash'] == art['hash'])
        clone_ok.append(bb.cache_hash(bb.clone_cache(art['cache'])) == art['hash'])
        # next-block input equals the post-block output at the site (propagation witness on the unedited artifact)
        prop = bool(torch.equal(art['captured'][site], art['next_input'][site]))
        # QUERY_TAIL at initialization must emulate SHARED_TAIL exactly even with a nonzero q (Gemma: exact V1 warm start; Qwen: zero-functional)
        if model['key'] == 'gemma':
            leg = LegacyBank(bb, v1_dirs('PREFIX_DISTRIBUTED')[0], 'v1'); aware = Bank.from_v1(bb, 'QUERY_TAIL', site, Q.GEMMA_CAP, 0, 'gemma', v1_dirs('PREFIX_DISTRIBUTED')[0], np.random.default_rng(1).normal(size=bb.hidden).astype(np.float32))
            qrand = np.random.default_rng(2).normal(size=(1, bb.hidden)).astype(np.float32)*50
            a1 = bb.compile(pre, maps=[dict(fn=leg.fn(1, 1.0), clause_positions=list(cp))], index=tail_idx, site=site)
            a2 = bb.compile(pre, maps=[dict(fn=aware.fn(1, 1.0, q=np.repeat(qrand, len(tail_idx), axis=0), record=False), clause_positions=list(cp))], index=tail_idx, site=site)
            emul.append(dict(hash_equal=a1['hash'] == a2['hash'], max_abs_norm_diff=float(np.max(np.abs(np.asarray(a1['realized']['realized_norms'][0])-np.asarray(a2['realized']['realized_norms'][0])))), q_norm=float(np.linalg.norm(qrand))))
        suff = []; lids = []; singles = []
        for q, draw in qs:
            enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels']); suff.append(enc['suffix_ids']); lids.append(l)
            cached = bb.ask(art, enc['suffix_ids'], l, verify_master=True); singles.append(cached)
            full = bb.full_forward(pre, [enc['suffix_ids']], [l], grad=False)['label_logps'][0].detach().float().cpu().numpy().tolist()
            re_art = bb.compile(pre); recompile_equal = re_art['hash'] == art['hash']
            tgt = Q.edited_world(s, 'single'); ttext, _ = Q.prefix_text(s, proc, world=tgt); tenc = bb.encode(ttext, Q.suffix_text(q['text'])); tart = bb.compile(tenc['prefix_ids'])
            tc = bb.ask(tart, tenc['suffix_ids'], l); tf = bb.full_forward(tenc['prefix_ids'], [tenc['suffix_ids']], [l], grad=False)['label_logps'][0].detach().float().cpu().numpy().tolist()
            rows.append(dict(query_id=q['query_id'], scene_id=sid, cached_logps=cached['logps'], full_logps=full, max_abs_diff=float(np.max(np.abs(np.asarray(cached['logps'])-np.asarray(full)))), same_choice=int(np.argmax(cached['logps'])) == int(np.argmax(full)),
                             master_unchanged=cached['master_unchanged'], recompile_hash_equal=recompile_equal, counterfactual_cached=tc['logps'], counterfactual_full=tf, counterfactual_max_abs=float(np.max(np.abs(np.asarray(tc['logps'])-np.asarray(tf)))),
                             counterfactual_same_choice=int(np.argmax(tc['logps'])) == int(np.argmax(tf)), propagation_next_block_equal=prop, prefix_tokens=len(pre), clause_tokens=len(cp), cache_bytes=art['bytes'], prefix_hash=art['hash']))
        # batched clone asks vs single asks (same master, one clone repeated); and the aware arm batched-prefix path vs batch-1 prefix compiles
        batched = bb.ask_batch(art, suff, lids, verify_master=True)
        for a, b_ in zip(singles, batched): batch_diffs.append(float(np.max(np.abs(np.asarray(a['logps'])-np.asarray(b_['logps']))))); batch_same.append(a['prediction'] == b_['prediction'])
        if model['key'] == 'gemma':
            import srs2_train as T_; bq = bb.ask_capture_batch(art, suff, site); bqs = dict(prefix=pre, addresses=[tuple(cp)], values=[1], n_prefix=len(pre))
            pa = T_.plan_for(aware, bqs, 'QUERY_TAIL', 1.0, q_batch=bq, record=False); ab = bb.ask_batch(T_.compile_plan(bb, bqs, pa), suff, lids)
            for j in range(len(suff)):
                a1 = T_.compile_plan(bb, bqs, T_.plan_for(aware, bqs, 'QUERY_TAIL', 1.0, q_batch=bq[j:j+1], record=False)); s1 = bb.ask(a1, suff[j], lids[j])
                prefix_batch_diffs.append(float(np.max(np.abs(np.asarray(s1['logps'])-np.asarray(ab[j]['logps']))))); prefix_batch_same.append(s1['prediction'] == ab[j]['prediction'])
    # order independence on shared masters
    arts = {}; fwd = []; rev = []
    for s, q, draw in reqs:
        text, spans = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels'])
        if s.scene_id not in arts: arts[s.scene_id] = bb.compile(enc['prefix_ids'])
        fwd.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], l)['logps'])
    for s, q, draw in reversed(reqs):
        text, spans = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels']); rev.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], l)['logps'])
    order_identical = fwd == rev[::-1]; masters_unchanged = all(bb.cache_hash(a['cache']) == a['hash'] for a in arts.values())
    rep = dict(at=Q.now(), actor=model['key'], procedure=proc, site=site, requests=len(rows), rows=rows, zero_self_edit_hash_identical=all(zero_ok), clone_hash_identical=all(clone_ok),
               cached_vs_full=dict(max_abs_logp=max(r['max_abs_diff'] for r in rows), same_choice=all(r['same_choice'] for r in rows), mean_abs=float(np.mean([r['max_abs_diff'] for r in rows]))),
               batched_vs_single=dict(max_abs_logp=max(batch_diffs), same_choice=all(batch_same), n=len(batch_diffs)), aware_batched_prefix_vs_single_prefix=(dict(max_abs_logp=max(prefix_batch_diffs), same_choice=all(prefix_batch_same), n=len(prefix_batch_diffs)) if prefix_batch_diffs else 'NOT_APPLICABLE'), counterfactual_reference=dict(max_abs_logp=max(r['counterfactual_max_abs'] for r in rows), same_choice=all(r['counterfactual_same_choice'] for r in rows)),
               order_independent=order_identical, masters_unchanged_after_reordered_queries=masters_unchanged, recompile_deterministic=all(r['recompile_hash_equal'] for r in rows), master_unchanged_every_ask=all(r['master_unchanged'] for r in rows),
               propagation_next_block=all(r['propagation_next_block_equal'] for r in rows), query_tail_init_emulates_shared_tail=(dict(all_hash_equal=all(e['hash_equal'] for e in emul), n=len(emul), max_abs_norm_diff=max(e['max_abs_norm_diff'] for e in emul)) if emul else 'NOT_APPLICABLE(zero-functional init)'),
               serving_path='KV_CACHE_CLONE + uniform batch-12 clone asks', batched_ask_tolerance='same choice required; |dlogp| recorded (BF16 batch-shape kernel effect: batch 1 reproduces single asks exactly, batch>1 shifts label log-probs by ~0.05-0.2 nats)', generation_prompt=bb.generation_prompt())
    rep['pass_'] = (rep['zero_self_edit_hash_identical'] and rep['clone_hash_identical'] and rep['order_independent'] and rep['master_unchanged_every_ask'] and rep['masters_unchanged_after_reordered_queries'] and rep['cached_vs_full']['same_choice']
                    and rep['batched_vs_single']['same_choice'] and rep['propagation_next_block'] and (not emul or all(e['hash_equal'] for e in emul)) and rep['recompile_deterministic'])
    Q.dump(out/'INTERFACE_CHECK.json', rep); return rep

# ---------------------------------------------------------------- gradient check for every arm
def gradient_check(bb, fit, out, deadline, model, proc):
    if done(out/'GRADIENT_CHECK.json'): return Q.load(out/'GRADIENT_CHECK.json')
    import torch; from srs2_editor import Bank; import srs2_train as T
    site = model['site']; scenes = sorted(fit, key=lambda s: Q.hash_rank('grad', s.scene_id))[:4]; res = {}
    bl = [T.scene_bundle(bb, s, 'FIT', proc) for s in scenes]; qs = T.QStore(); qs.compute(bb, bl, site)
    mu = np.zeros(bb.hidden, np.float32); mu_q = np.mean([v for v in qs.q.values()], axis=0).astype(np.float32)
    for arm in Q.ARMS:
        Q.budget(deadline, 120); t0 = time.monotonic()
        bank = (Bank.from_v1(bb, arm, site, Q.GEMMA_CAP, 0, 'gemma', v1_dirs('PREFIX_DISTRIBUTED')[0], mu_q) if model['key'] == 'gemma' else Bank.fresh(bb, arm, site, 50.0, 3, model['key'], mu, mu_q))
        opt = torch.optim.AdamW(bank.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0); before = []; after = []; g = None; loss_v = None; backbone_grad = None
        used = sorted({v for b in bl for v in b['values']}); qblock = {}     # only the desired-value maps addressed by these scenes receive gradient
        for step in range(2):
            losses = []
            for b in bl:      # one backward per scene (gradients accumulate); holding four graphs at once does not fit beside the backbone
                sub = b['queries'][:4]
                if bank.aware: pl = T.plan_for(bank, b, arm, 1.0, q_batch=qs.batch(sub), record=False); r = bb.full_forward(b['prefix'], [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, prefix_rows=pl['rows'], prefix_batch=4)
                else: pl = T.plan_for(bank, b, arm, 1.0, record=False); r = bb.full_forward(b['prefix'], [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True)
                lp = torch.stack([r['label_logps'][i][sub[i]['rec']['gold']] for i in range(4)]); (before if step == 0 else after).extend(lp.detach().float().cpu().tolist()); l_ = -lp.mean()/len(bl); bb.backward(l_); losses.append(float(l_)); del r, lp, l_
            loss = float(sum(losses))
            if step == 0:
                loss_v = loss; g = {f'{v}.{n}': float(p.grad.norm()) if p.grad is not None else None for v, e in bank.maps.items() for n, p in e.named_parameters() if v in used}
                backbone_grad = any(p.grad is not None for p in bb.model.parameters()); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad()
            else:
                # the query block receives gradient only once the output factor is nonzero: measured after the first step (zero-functional init) as well as at init
                qblock = {v: float(e.input_factor.grad[2*bb.hidden:].norm()) if e.input_factor.grad is not None else None for v, e in bank.maps.items() if v in used}; opt.zero_grad()
        res[arm] = dict(loss=loss_v, grad_norms=g, q_block_grad=qblock, used_values=used, backbone_grad=backbone_grad, finite=all(v is not None and np.isfinite(v) for v in g.values()), output_factor_grad_nonzero=all(g[f'{v}.output_factor'] > 0 for v in used),
                        q_block_grad_nonzero=(all(v is not None and v > 0 for v in qblock.values()) if bank.aware else None), max_score_move=float(np.max(np.abs(np.asarray(after)-np.asarray(before)))), seconds=time.monotonic()-t0)
    rep = dict(at=Q.now(), actor=model['key'], procedure=proc, scenes=[s.scene_id for s in scenes], queries=16, results=res, backbone_unchanged=bb.verify_unchanged()['unchanged'],
               pass_=all(v['finite'] and v['output_factor_grad_nonzero'] and not v['backbone_grad'] and v['max_score_move'] > 0 and (v['q_block_grad_nonzero'] is not False) for v in res.values()))
    Q.dump(out/'GRADIENT_CHECK.json', rep); return rep

# ---------------------------------------------------------------- disposable timing benchmark
def benchmark(bb, fit, out, model, proc):
    if done(out/'BENCHMARK.json'): return Q.load(out/'BENCHMARK.json')
    import torch; from srs2_editor import Bank; import srs2_train as T
    site = model['site']; s = sorted(fit, key=lambda s: Q.hash_rank('bench', s.scene_id))[0]; b = T.scene_bundle(bb, s, 'FIT', proc); qs = T.QStore(); qs.compute(bb, [b], site)
    mu_q = np.mean([v for v in qs.q.values()], axis=0).astype(np.float32); mu = np.zeros(bb.hidden, np.float32)
    banks = {arm: (Bank.from_v1(bb, arm, site, Q.GEMMA_CAP, 0, 'gemma', v1_dirs('PREFIX_DISTRIBUTED')[0], mu_q) if model['key'] == 'gemma' else Bank.fresh(bb, arm, site, 50.0, 4, model['key'], mu, mu_q)) for arm in ('SHARED_TAIL', 'QUERY_TAIL')}
    pre = b['prefix']; q12 = b['queries'][:12]; q24 = b['queries'][:24] if len(b['queries']) >= 24 else b['queries']*2
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    def timed(fn, n=3):
        fn(); sync(); t0 = time.monotonic()
        for _ in range(n): fn()
        sync(); return (time.monotonic()-t0)/n
    art = bb.compile(pre, capture={site}); pls = T.plan_for(banks['SHARED_TAIL'], b, 'SHARED_TAIL', 1.0, record=False); pla = T.plan_for(banks['QUERY_TAIL'], b, 'QUERY_TAIL', 1.0, q_batch=qs.batch(q12), record=False)
    t = dict(compile=timed(lambda: bb.compile(pre)), compile_capture=timed(lambda: bb.compile(pre, capture={site})), compile_shared=timed(lambda: T.compile_plan(bb, b, pls)), compile_aware12=timed(lambda: T.compile_plan(bb, b, pla)),
             ask=timed(lambda: bb.ask(art, q12[0]['suffix'], q12[0]['labels'])), ask_batch12=timed(lambda: bb.ask_batch(art, [x['suffix'] for x in q12], [x['labels'] for x in q12])),
             ask_batch24=timed(lambda: bb.ask_batch(art, [x['suffix'] for x in q24], [x['labels'] for x in q24])), ask_capture12=timed(lambda: bb.ask_capture_batch(art, [x['suffix'] for x in q12], site)),
             ask_generate32=timed(lambda: bb.ask(art, q12[0]['suffix'], q12[0]['labels'], generate=True), n=1), clone=timed(lambda: bb.clone_cache(art['cache'])), hash=timed(lambda: bb.cache_hash(art['cache'])))
    def step(arm):
        bank = banks[arm]; sub = q12[:4]
        if bank.aware: pl = T.plan_for(bank, b, arm, 1.0, q_batch=qs.batch(sub), record=False); r = bb.full_forward(pre, [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, prefix_rows=pl['rows'], prefix_batch=4)
        else: pl = T.plan_for(bank, b, arm, 1.0, record=False); r = bb.full_forward(pre, [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True)
        loss = -torch.stack([r['label_logps'][i][sub[i]['rec']['gold']] for i in range(4)]).mean(); bb.backward(loss)
        for e in bank.maps.values(): e.zero_grad()
    t['train_step4_shared'] = timed(lambda: step('SHARED_TAIL'), n=2); t['train_step4_aware'] = timed(lambda: step('QUERY_TAIL'), n=2)
    rep = dict(at=Q.now(), actor=model['key'], prefix_tokens=len(pre), suffix_tokens=[len(x['suffix']) for x in q12], cache_bytes=art['bytes'], seconds=t, gpu_memory_peak_bytes=(int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None), note='disposable timing panel; not a measurement')
    Q.dump(out/'BENCHMARK.json', rep); return rep

def phase_q(bb, out, deadline, model, fit, cal):
    qual = qualify(bb, cal, out, deadline, model); event(out, 'QUALIFICATION', selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'])
    an = anchors(bb, out, deadline, model); event(out, 'ANCHORS', status=an['status'], seeds={k: (v['rows'], v['same_choice'], v['max_abs_logp'], v['hash_equal'], v['scenes']) for k, v in an.get('seeds', {}).items()})
    proc = qual['selected_procedure'] or qual['natural_panel_procedure']
    ic = interface_checks(bb, cal, out, deadline, model, proc); event(out, 'INTERFACE_CHECK', pass_=ic['pass_'], cached_vs_full=ic['cached_vs_full'], batched=ic['batched_vs_single'], emulation=ic['query_tail_init_emulates_shared_tail'])
    gc = gradient_check(bb, fit, out, deadline, model, proc); event(out, 'GRADIENT_CHECK', pass_=gc['pass_'], results={k: (v['output_factor_grad_nonzero'], v['q_block_grad_nonzero'], v['max_score_move']) for k, v in gc['results'].items()})
    bench = benchmark(bb, fit, out, model, proc); event(out, 'BENCHMARK', seconds=bench['seconds'])
    return dict(qualification=qual['selected_procedure'], anchors=an['status'], interface=ic['pass_'], gradient=gc['pass_'])

def competitor_from(freeze):
    """Qwen's matched competitor: the strongest CAL-selected non-SHARED_TAIL Gemma arm (feasibility first, then worse-seed changed accuracy)."""
    cands = [(a, v) for a, v in freeze['selection'].items() if a != 'SHARED_TAIL']
    if not cands: return None
    return max(cands, key=lambda av: (int(av[1]['feasible']), min(av[1]['cal'][s]['changed'] for s in ('0', '1'))))[0]

def main():
    raise RuntimeError("Original provider launcher is not distributed. Use the CPU reproduction entry point; optional model reruns require an explicit local runtime configuration.")

if __name__ == '__main__': main()
