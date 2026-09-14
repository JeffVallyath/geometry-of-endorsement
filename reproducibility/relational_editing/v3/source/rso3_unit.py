# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""RSO3 scientific unit inside the durable scientific-unit host.  Phases (markers make every stage re-enterable):

Q  per actor: load -> native qualification under the NEW symmetric primitive policy (>=95% parseable within 32 tokens and >=90%
   direct/negation accuracy separately on source and counterfactual worlds; equality always measured and reported, never a veto; Gemma:
   exact V2 C1; Qwen: T4 -> C1S -> C1) -> [Gemma] anchor replay of 4 V2 single-edit and 4 V2 program scenes (RSO3-ANCHOR) through the exact
   historical per-command BF16 path against the recorded V2 rows and artifact hashes -> E0 interface panel (cached vs full, batched vs
   single asks, zero/self edit, clone, recompile, query order, master immutability, next-block propagation) -> setter numerics on real
   states (NumPy vs Torch, FP32 workspace identities, cast errors) -> gradient check for every learned arm -> disposable timing benchmark.
DF development (rso3_train.phase_d) -> FREEZE.json before any FINAL/WORKFLOW read -> fresh evaluation (rso3_final.phase_f).
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time
import traceback
from pathlib import Path
import numpy as np
import rso3_common as Q
import rso3_adapter as A

BASE = 'external-artifacts'

def event(output, name, **fields):
    row = dict(at=Q.now(), event=name, **fields)
    with (Path(output)/'stages.jsonl').open('a', encoding='utf8', newline='\n') as f: f.write(Q.canonical(row)+'\n')
    print(Q.canonical(row), flush=True)

def done(path): return Path(path).is_file()

# ---------------------------------------------------------------- native rows
def native_row(bb, s, q, draw, proc, world=None, generate=True, order='original'):
    text, _ = Q.prefix_text(s, proc, order, world=world); enc = bb.encode(text, Q.suffix_text(q['text'])); lids = bb.label_ids(enc['full_text'], q['labels'])
    art = bb.compile(enc['prefix_ids']); res = bb.ask(art, enc['suffix_ids'], lids, generate=generate)
    row = Q.query_record(s, q, s.split, draw, order); gold = q['source_gold'] if world is None else int(Q.OLD.evaluate(world, q['spec']))
    row.update(procedure=proc, world='source' if world is None else 'counterfactual', prefix_tokens=len(enc['prefix_ids']), suffix_tokens=len(enc['suffix_ids']), logps=res['logps'], score_prediction=res['prediction'],
               answer_mass=res['answer_mass'], argmax_in_labels=res['argmax_in_labels'], target_index=int(gold), score_correct=res['prediction'] == gold, prefix_hash=art['hash'])
    if generate:
        sym, status = Q.parse_symbol(res['generation']['text'], q['labels']); pred = list(q['labels']).index(sym) if sym is not None else None
        row.update(generation=res['generation']['text'], generation_tokens=len(res['generation']['tokens']), truncated=res['generation']['truncated'], parse_status=status, generation_prediction=pred, generation_correct=pred == gold, first_token_is_argmax=res['generation']['first_token_is_argmax'])
    return row

def qualification_requests(cal):
    """96 CAL requests: direct/negation 8 cells (stance x target relevance x truth) x 8, equality 4 cells (stance x truth) x 8; hash-ordered."""
    pool = [(s, q, draw) for s in cal for draw in (0, 1) for q in Q.questions(s, 'CAL', draw, 'single')]
    out = []
    for fam_group, fams, rel_split, per, total in (('dn', ('direct', 'opposes'), True, 8, 64), ('eq', ('same',), False, 8, 32)):
        chosen = []; leftovers = []
        for stance in (0, 1):
            for truth in (0, 1):
                for rel in ((True, False) if rel_split else (None,)):
                    c = [x for x in pool if x[1]['family'] in fams and x[0].values[x[0].target_actor][x[0].target_project] == stance and x[1]['source_gold'] == truth
                         and (rel is None or ((x[1]['spec']['a'] == x[0].target_actor or x[1]['spec'].get('b') == x[0].target_actor) and x[1]['spec']['p'] == x[0].target_project) == rel)]
                    c = sorted(c, key=lambda x: Q.hash_rank('qual', x[1]['query_id'])); chosen.extend(c[:per]); leftovers.extend(c[per:])
        chosen.extend(sorted(leftovers, key=lambda x: Q.hash_rank('qual', x[1]['query_id']))[:total-len(chosen)])
        if len(chosen) != total: raise ValueError(f'qualification group short: {fam_group} -> {len(chosen)}')
        out.extend(chosen)
    if len(out) != Q.QUAL_N or len({x[1]['query_id'] for x in out}) != Q.QUAL_N: raise ValueError('qualification battery size')
    return out

def qualify(bb, cal, out, deadline, model):
    if done(out/'QUALIFICATION.json'): return Q.load(out/'QUALIFICATION.json')
    reqs = qualification_requests(cal); attempts = []; selected = None; generations = 0
    for proc in Q.PROCEDURES[model['key']]:
        rows = []
        for s, q, draw in reqs:
            Q.budget(deadline, 120); rows.append(native_row(bb, s, q, draw, proc)); rows.append(native_row(bb, s, q, draw, proc, world=Q.final_world(s, 'single'))); generations += 2
        Q.write_rows(out/f'qualification_{proc}.jsonl', rows); summary = {}
        for world in ('source', 'counterfactual', 'pooled'):
            rs = [r for r in rows if world == 'pooled' or r['world'] == world]; dn = [r for r in rs if r['family'] in ('direct', 'opposes')]; eq = [r for r in rs if r['family'] == 'same']
            summary[world] = dict(n=len(rs), parse_rate=float(np.mean([r['parse_status'].startswith('OK') for r in rs])), direct_negation_accuracy=float(np.mean([r['generation_correct'] for r in dn])), equality_accuracy=float(np.mean([r['generation_correct'] for r in eq])),
                                  score_accuracy=float(np.mean([r['score_correct'] for r in rs])), truncated=int(sum(r['truncated'] for r in rs)), generation_score_agreement=float(np.mean([r['generation_prediction'] == r['score_prediction'] for r in rs])),
                                  mean_new_tokens=float(np.mean([r['generation_tokens'] for r in rs])), by_family={f: float(np.mean([r['generation_correct'] for r in rs if r['family'] == f])) for f in ('direct', 'opposes', 'same')})
            summary[world]['primitive_qualified'] = summary[world]['parse_rate'] >= Q.QUAL_PARSE and summary[world]['direct_negation_accuracy'] >= Q.QUAL_DIRECT
            summary[world]['equality_ge_85_reported_only'] = summary[world]['equality_accuracy'] >= Q.QUAL_EQUALITY_REPORTED
        ok = summary['source']['primitive_qualified'] and summary['counterfactual']['primitive_qualified']
        attempts.append(dict(procedure=proc, summary=summary, generations=len(rows), qualified=ok))
        event(out, 'QUALIFICATION_ATTEMPT', procedure=proc, qualified=ok, source=(summary['source']['parse_rate'], summary['source']['direct_negation_accuracy'], summary['source']['equality_accuracy']), counterfactual=(summary['counterfactual']['parse_rate'], summary['counterfactual']['direct_negation_accuracy'], summary['counterfactual']['equality_accuracy']))
        if ok: selected = proc; break
    best_native = max(attempts, key=lambda a: (a['summary']['pooled']['direct_negation_accuracy']+a['summary']['pooled']['equality_accuracy'], a['summary']['pooled']['parse_rate']))['procedure']
    sel = next((a for a in attempts if a['procedure'] == selected), attempts[-1])
    rep = dict(at=Q.now(), actor=model['key'], selected_procedure=selected, natural_panel_procedure=best_native, attempts=attempts, total_generations=generations, requests=Q.QUAL_N,
               policy='RSO3 symmetric primitive policy: first procedure with >=95% parseable generations (<=32 new tokens) and >=90% direct/negation accuracy on BOTH the source and the natural-counterfactual inputs; equality is always measured and stays in every primary metric but never vetoes editing; historical V2 verdicts unchanged',
               equality_summary={w: sel['summary'][w]['equality_accuracy'] for w in ('source', 'counterfactual')}, battery=dict(direct_negation=64, equality=32), template=bb.generation_prompt())
    Q.dump(out/'QUALIFICATION.json', rep); return rep

# ---------------------------------------------------------------- anchors (Gemma only): exact V2 r4 replay through the historical BF16 path
def anchors(bb, out, deadline, model):
    if done(out/'ANCHORS.json'): return Q.load(out/'ANCHORS.json')
    if model['key'] != 'gemma' or bb.cfg.get('id') == 'tiny': Q.dump(out/'ANCHORS.json', dict(at=Q.now(), status='NOT_APPLICABLE', actor=model['key'], tiny=bb.cfg.get('id') == 'tiny')); return Q.load(out/'ANCHORS.json')
    from rso3_editor import AffineBank; import rso3_train as T
    S = Q.S; spec = Q.load(Q.ANCHORS/'ANCHORS.json'); old = {s.scene_id: s for s in S.scenes('FINAL')}; seal = {(r['scene_id'], r['order']): r for r in Q.read_rows(Q.ANCHORS/'V2_r4_gemma_final_phase_SEAL.jsonl') if r['variant'] == 0}   # the V2 seal also holds wording-variant-1 rows for 32 scenes; anchors replay variant 0
    tol = dict(same_choice_required=True, max_abs_logp=0.05, note='inherited verified tolerance: same choice on every row; artifact hash equality additionally proves bit-identical compiled contexts')
    report = dict(at=Q.now(), status=None, seeds={}, single=spec['single'], programs=spec['programs']); rows_out = []; t0 = time.monotonic(); strength = Q.v2_strength('SHARED_CLAUSE')
    for seed in Q.SEEDS:
        bank = AffineBank.from_v2(bb, 'SHARED_CLAUSE', seed); ref_f = {r['query_id']: r for r in Q.read_rows(Q.ANCHORS/f'final_SHARED_CLAUSE_s{seed}.jsonl') if r['order'] == 'early' or r['order'] == 'late'}
        src_f = {(r['query_id'], r['order']): r for r in Q.read_rows(Q.ANCHORS/'final_SOURCE.jsonl')}; ref_p = {(r['query_id'], r['program']): r for r in Q.read_rows(Q.ANCHORS/f'programs_SHARED_CLAUSE_s{seed}.jsonl')}
        ref_fo = {(r['query_id'], r['order']): r for r in Q.read_rows(Q.ANCHORS/f'final_SHARED_CLAUSE_s{seed}.jsonl')}
        st = dict(rows=0, same_choice=0, max_abs_logp=0.0, source_rows=0, source_same_choice=0, source_max_abs_logp=0.0, hash_equal=0, source_hash_equal=0, artifacts=0, program_rows=0, program_same_choice=0, program_max_abs_logp=0.0, program_hash_equal=0, program_artifacts=0, mismatches=[])
        for sid in spec['single']:
            s = old[sid]
            for order in Q.ORDERS:
                Q.budget(deadline, 120); text, spans = S.prefix_text(s, 'C1', order, 0); pre = bb.prefix_ids(text); a = S.P.primary_edit(s); cp = bb.clause_positions(text, spans[(a.actor, a.project)])
                b = dict(prefix=pre, addresses=[tuple(cp)], values=[int(a.value)], n_prefix=len(pre)); src = bb.compile(pre); pl = T.plan_for(bank, b, strength, record=False, workspace='legacy', footprint='clause'); art = T.compile_plan(bb, b, pl)
                sr = seal[(sid, order)]; st['artifacts'] += 1; st['hash_equal'] += int(art['hash'] == sr['artifacts'][f'SHARED_CLAUSE_s{seed}']['hash']); st['source_hash_equal'] += int(src['hash'] == sr['artifacts']['SOURCE']['hash'])
                qs = [(q, draw) for draw in (0, 1) for q in S.questions(s, 'FINAL', draw)]; encs = [bb.encode(text, S.suffix_text(q['text'])) for q, _ in qs]; lids = [bb.label_ids(e['full_text'], q['labels']) for e, (q, _) in zip(encs, qs)]
                res = []; sres = []
                for k in range(0, len(qs), Q.CHUNK): res += bb.ask_batch(art, [e['suffix_ids'] for e in encs[k:k+Q.CHUNK]], lids[k:k+Q.CHUNK]); sres += bb.ask_batch(src, [e['suffix_ids'] for e in encs[k:k+Q.CHUNK]], lids[k:k+Q.CHUNK])
                for (q, draw), r_, sr_ in zip(qs, res, sres):
                    rr = ref_fo[(q['query_id'], order)]; so = src_f[(q['query_id'], order)]; d = float(np.max(np.abs(np.asarray(r_['logps'])-np.asarray(rr['logps'])))); ds = float(np.max(np.abs(np.asarray(sr_['logps'])-np.asarray(so['logps']))))
                    st['rows'] += 1; st['same_choice'] += int(r_['prediction'] == rr['prediction']); st['max_abs_logp'] = max(st['max_abs_logp'], d); st['source_rows'] += 1; st['source_same_choice'] += int(sr_['prediction'] == so['prediction']); st['source_max_abs_logp'] = max(st['source_max_abs_logp'], ds)
                    if r_['prediction'] != rr['prediction'] or d > tol['max_abs_logp']: st['mismatches'].append(dict(kind='single', query_id=q['query_id'], order=order, replay=r_['logps'], recorded=rr['logps']))
                    rows_out.append(dict(seed=seed, kind='single', scene_id=sid, order=order, query_id=q['query_id'], replay_logps=r_['logps'], recorded_logps=rr['logps'], replay_prediction=r_['prediction'], recorded_prediction=rr['prediction'], gold=rr['gold'], artifact_hash=art['hash'], recorded_artifact_hash=rr['artifact_hash']))
        for sid in spec['programs']:
            s = old[sid]; text, spans = S.prefix_text(s, 'C1', 'early', 0); pre = bb.prefix_ids(text)
            for program, edits in S.P.programs(s).items():
                Q.budget(deadline, 120); b = dict(prefix=pre, addresses=[tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in edits], values=[int(e.value) for e in edits], n_prefix=len(pre))
                pl = T.plan_for(bank, b, strength, record=False, workspace='legacy', footprint='clause'); art = T.compile_plan(bb, b, pl); world = S.P.apply_program(s, edits)
                qs = [(q, draw) for draw in (0, 1) for q in S.questions(s, 'FINAL', draw, world=world)]; encs = [bb.encode(text, S.suffix_text(q['text'])) for q, _ in qs]; lids = [bb.label_ids(e['full_text'], q['labels']) for e, (q, _) in zip(encs, qs)]
                res = []
                for k in range(0, len(qs), Q.CHUNK): res += bb.ask_batch(art, [e['suffix_ids'] for e in encs[k:k+Q.CHUNK]], lids[k:k+Q.CHUNK])
                hashes = {ref_p[(q['query_id'], program)]['artifact_hash'] for q, _ in qs}; st['program_artifacts'] += 1; st['program_hash_equal'] += int(hashes == {art['hash']})
                for (q, draw), r_ in zip(qs, res):
                    rr = ref_p[(q['query_id'], program)]; d = float(np.max(np.abs(np.asarray(r_['logps'])-np.asarray(rr['logps']))))
                    st['program_rows'] += 1; st['program_same_choice'] += int(r_['prediction'] == rr['prediction']); st['program_max_abs_logp'] = max(st['program_max_abs_logp'], d)
                    if r_['prediction'] != rr['prediction'] or d > tol['max_abs_logp']: st['mismatches'].append(dict(kind='program', program=program, query_id=q['query_id'], replay=r_['logps'], recorded=rr['logps']))
                    rows_out.append(dict(seed=seed, kind='program', program=program, scene_id=sid, query_id=q['query_id'], replay_logps=r_['logps'], recorded_logps=rr['logps'], replay_prediction=r_['prediction'], recorded_prediction=rr['prediction'], gold=rr['gold'], artifact_hash=art['hash'], recorded_artifact_hash=rr['artifact_hash']))
        st['pass_'] = st['same_choice'] == st['rows'] and st['program_same_choice'] == st['program_rows'] and max(st['max_abs_logp'], st['program_max_abs_logp']) <= tol['max_abs_logp'] and st['source_same_choice'] == st['source_rows']
        st['bit_identical'] = st['hash_equal'] == st['artifacts'] and st['program_hash_equal'] == st['program_artifacts'] and st['source_hash_equal'] == st['artifacts']
        report['seeds'][str(seed)] = dict(st, factors=bank.init['files'], strength=strength, site=bank.site)
    Q.write_rows(out/'anchor_rows.jsonl', rows_out)
    report.update(status='TECHNICAL_PASS' if all(v['pass_'] for v in report['seeds'].values()) else 'TECHNICAL_MISMATCH', bit_identical=all(v['bit_identical'] for v in report['seeds'].values()), tolerance=tol, seconds=time.monotonic()-t0,
                  inputs='V2 prompts (V2 FIT C1 demonstrations, V2 FINAL scenes, V2 questions/labels), the selected V2 SHARED_CLAUSE factors at strength 1.25 through the historical per-command BF16 path, BF16 backbone, FP32 readout, batch-12 clone asks; no V3 data entered this replay')
    Q.dump(out/'ANCHORS.json', report); return report

# ---------------------------------------------------------------- E0 interface + setter numerics on the real model
def check_requests(cal, n, salt):
    rows = [(s, q, draw) for s in cal for draw in (0, 1) for q in Q.questions(s, 'CAL', draw, 'single')]
    return sorted(rows, key=lambda x: Q.hash_rank(salt, x[1]['query_id']))[:n]

def interface_checks(bb, cal, out, deadline, model, proc):
    if done(out/'INTERFACE_CHECK.json'): return Q.load(out/'INTERFACE_CHECK.json')
    import torch; from rso3_editor import SetterBank, AffineBank, invariant_numpy, free_identity_heads, svd_basis; import rso3_train as T
    site = model['site']; reqs = check_requests(cal, 16, 'e0'); rows = []; zero_ok = []; clone_ok = []; batch_diffs = []; batch_same = []; ident = []; numerics = []
    zero_aff = AffineBank.fresh_clause(bb, site, 1.0, 0, model['key'], np.zeros(bb.hidden, np.float32), np.zeros(bb.hidden, np.float32))
    by_scene = {}
    for s, q, draw in reqs: by_scene.setdefault(s.scene_id, (s, []))[1].append((q, draw))
    rng = np.random.default_rng(7)
    for sid, (s, qs) in by_scene.items():
        Q.budget(deadline, 120); text, spans = Q.prefix_text(s, proc); a = Q.programs(s)['single'][0]; cp = bb.clause_positions(text, spans[(a.actor, a.project)]); pre = bb.prefix_ids(text)
        art = bb.compile(pre, capture={site}, capture_next={site}); b = dict(prefix=pre, addresses=[tuple(cp)], values=[int(a.value)], n_prefix=len(pre))
        for ws in ('legacy', 'fp32'):
            za = T.compile_plan(bb, b, T.plan_for(zero_aff, b, 1.0, record=False, workspace=ws, footprint='clause')); zero_ok.append(za['hash'] == art['hash'])
        clone_ok.append(bb.cache_hash(bb.clone_cache(art['cache'])) == art['hash']); prop = bool(torch.equal(art['captured'][site], art['next_input'][site]))
        # setter numerics on the real residual: random orthonormal U + small random heads (no training): NumPy vs Torch, FP32 identities, cast/identity errors
        h = art['captured'][site][cp].float(); u, _ = svd_basis(rng.normal(size=(64, bb.hidden))); inv = SetterBank(bb, 'INVARIANT_SET', site, 0, model['key'], 100.0); inv.set_center(np.zeros(bb.hidden, np.float32)); inv.set_basis(u)
        with torch.no_grad(): inv.model.head.normal_(0, 1e-3); inv.model.bias.normal_(0, 1.0)
        inv.freeze_basis(); mu = inv.model.center.cpu().numpy()
        with torch.no_grad():
            y = inv.model(h, 1, frozen_basis=inv.frozen_u).cpu().numpy(); yn = invariant_numpy(h.cpu().numpy().astype(np.float64), inv.basis64(), inv.model.head[1].cpu().numpy().astype(np.float64), inv.model.bias[1].cpu().numpy().astype(np.float64), mu.astype(np.float64))
        numpy_vs_torch = float(np.max(np.abs(y-yn)))
        b1 = dict(b, values=[1]); a1 = T.compile_plan(bb, b1, T.plan_for(inv, b1, record=False)); a2 = T.compile_plan(bb, dict(b1, addresses=b1['addresses']*2, values=[1, 1]), T.plan_for(inv, dict(b1, addresses=b1['addresses']*2, values=[1, 1]), record=False))
        a8 = T.compile_plan(bb, dict(b1, addresses=b1['addresses']*8, values=[1]*8), T.plan_for(inv, dict(b1, addresses=b1['addresses']*8, values=[1]*8), record=False))
        c0 = T.compile_plan(bb, dict(b1, values=[0]), T.plan_for(inv, dict(b1, values=[0]), record=False)); c10 = T.compile_plan(bb, dict(b1, addresses=b1['addresses']*2, values=[1, 0]), T.plan_for(inv, dict(b1, addresses=b1['addresses']*2, values=[1, 0]), record=False))
        ident.append(dict(scene_id=sid, ws_AA_vs_A=float(np.max(np.abs(a2['_workspace_states']-a1['_workspace_states']))), ws_A8_vs_A=float(np.max(np.abs(a8['_workspace_states']-a1['_workspace_states']))), ws_over_vs_direct=float(np.max(np.abs(c10['_workspace_states']-c0['_workspace_states']))),
                          bf16_AA_hash_equal=a2['hash'] == a1['hash'], bf16_A8_hash_equal=a8['hash'] == a1['hash'], bf16_over_hash_equal=c10['hash'] == c0['hash'], cast_error=a1['realized']['cast_error'], numpy_vs_torch=numpy_vs_torch,
                          state_scale=float(np.abs(a1['_workspace_states']).max()), update_norm=float(np.mean(a1['realized']['realized_norms'][0]))))
        # frozen V2 clause: historical BF16 path vs FP32 workspace (same factors) -- the precision comparator
        if model['key'] == 'gemma' and bb.hidden == 3584:
            fro = AffineBank.from_v2(bb, 'SHARED_CLAUSE', 0); st = Q.v2_strength('SHARED_CLAUSE'); l1 = T.compile_plan(bb, b, T.plan_for(fro, b, st, record=False, workspace='legacy')); f1 = T.compile_plan(bb, b, T.plan_for(fro, b, st, record=False, workspace='fp32'))
            numerics.append(dict(scene_id=sid, legacy_vs_fp32_state_max_abs=float(np.max(np.abs(l1['_after_states']-f1['_after_states']))), hash_equal=l1['hash'] == f1['hash'], legacy_energy=l1['realized']['energy'], fp32_energy=f1['realized']['energy']))
        suff = []; lids = []; singles = []
        for q, draw in qs:
            enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels']); suff.append(enc['suffix_ids']); lids.append(l)
            cached = bb.ask(art, enc['suffix_ids'], l, verify_master=True); singles.append(cached)
            full = bb.full_forward(pre, [enc['suffix_ids']], [l], grad=False)['label_logps'][0].detach().float().cpu().numpy().tolist(); recompile_equal = bb.compile(pre)['hash'] == art['hash']
            rows.append(dict(query_id=q['query_id'], scene_id=sid, cached_logps=cached['logps'], full_logps=full, max_abs_diff=float(np.max(np.abs(np.asarray(cached['logps'])-np.asarray(full)))), same_choice=int(np.argmax(cached['logps'])) == int(np.argmax(full)),
                             master_unchanged=cached['master_unchanged'], recompile_hash_equal=recompile_equal, propagation_next_block_equal=prop, prefix_tokens=len(pre), clause_tokens=len(cp), cache_bytes=art['bytes'], prefix_hash=art['hash']))
        batched = bb.ask_batch(art, suff, lids, verify_master=True)
        for x, y_ in zip(singles, batched): batch_diffs.append(float(np.max(np.abs(np.asarray(x['logps'])-np.asarray(y_['logps']))))); batch_same.append(x['prediction'] == y_['prediction'])
    arts = {}; fwd = []; rev = []
    for s, q, draw in reqs:
        text, _ = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels'])
        if s.scene_id not in arts: arts[s.scene_id] = bb.compile(enc['prefix_ids'])
        fwd.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], l)['logps'])
    for s, q, draw in reversed(reqs):
        text, _ = Q.prefix_text(s, proc); enc = bb.encode(text, Q.suffix_text(q['text'])); l = bb.label_ids(enc['full_text'], q['labels']); rev.append(bb.ask(arts[s.scene_id], enc['suffix_ids'], l)['logps'])
    rep = dict(at=Q.now(), actor=model['key'], procedure=proc, site=site, requests=len(rows), rows=rows, zero_self_edit_hash_identical=all(zero_ok), clone_hash_identical=all(clone_ok),
               cached_vs_full=dict(max_abs_logp=max(r['max_abs_diff'] for r in rows), same_choice=all(r['same_choice'] for r in rows), mean_abs=float(np.mean([r['max_abs_diff'] for r in rows]))),
               batched_vs_single=dict(max_abs_logp=max(batch_diffs), same_choice=all(batch_same), n=len(batch_diffs)), order_independent=fwd == rev[::-1], masters_unchanged_after_reordered_queries=all(bb.cache_hash(a['cache']) == a['hash'] for a in arts.values()),
               recompile_deterministic=all(r['recompile_hash_equal'] for r in rows), master_unchanged_every_ask=all(r['master_unchanged'] for r in rows), propagation_next_block=all(r['propagation_next_block_equal'] for r in rows),
               setter_identities=dict(rows=ident, max_ws_AA_vs_A=max(x['ws_AA_vs_A'] for x in ident), max_ws_A8_vs_A=max(x['ws_A8_vs_A'] for x in ident), max_ws_over_vs_direct=max(x['ws_over_vs_direct'] for x in ident), bf16_AA_hash_equal_fraction=float(np.mean([x['bf16_AA_hash_equal'] for x in ident])),
                                      bf16_A8_hash_equal_fraction=float(np.mean([x['bf16_A8_hash_equal'] for x in ident])), bf16_over_hash_equal_fraction=float(np.mean([x['bf16_over_hash_equal'] for x in ident])), max_numpy_vs_torch=max(x['numpy_vs_torch'] for x in ident), max_cast_error=max(x['cast_error'] for x in ident),
                                      note='untrained random-U setter on real residuals: FP32 workspace identities are designed algebra (reported as numerical error, not a scientific result); BF16 hash equality after the cast is not guaranteed'),
               frozen_v2_precision=numerics, serving_path='KV_CACHE_CLONE + uniform batch-12 clone asks (identical to V2)', generation_prompt=bb.generation_prompt())
    rep['pass_'] = (rep['zero_self_edit_hash_identical'] and rep['clone_hash_identical'] and rep['order_independent'] and rep['master_unchanged_every_ask'] and rep['masters_unchanged_after_reordered_queries'] and rep['cached_vs_full']['same_choice']
                    and rep['batched_vs_single']['same_choice'] and rep['propagation_next_block'] and rep['recompile_deterministic'] and rep['setter_identities']['max_numpy_vs_torch'] < 1e-2 and rep['setter_identities']['max_ws_AA_vs_A'] < 1e-2)
    Q.dump(out/'INTERFACE_CHECK.json', rep); return rep

# ---------------------------------------------------------------- gradient check for every learned arm
def gradient_check(bb, fit, out, deadline, model, proc):
    if done(out/'GRADIENT_CHECK.json'): return Q.load(out/'GRADIENT_CHECK.json')
    import torch; from rso3_editor import SetterBank, AffineBank, gemma_basis_init, svd_basis; import rso3_train as T
    site = model['site']; scenes = sorted(fit, key=lambda s: Q.hash_rank('grad', s.scene_id))[:4]; res = {}; bl = [T.scene_bundle(bb, s, 'FIT', proc) for s in scenes]
    arms = Q.LEARNED_ARMS if model['key'] == 'gemma' else Q.QWEN_ARMS
    for arm in arms:
        Q.budget(deadline, 120); t0 = time.monotonic()
        if arm in Q.KIND:
            bank = SetterBank(bb, arm, site, 0, model['key'], 100.0); bank.set_center(np.zeros(bb.hidden, np.float32)); bank.set_basis(gemma_basis_init(0)[0] if (model['key'] == 'gemma' and bb.hidden == 3584) else svd_basis(np.random.default_rng(3).normal(size=(64, bb.hidden)))[0])
            with torch.no_grad(): bank.model.head.normal_(0, 1e-3)
        elif arm == 'CONTINUED_AFFINE' and bb.hidden == 3584: bank = AffineBank.from_v2(bb, 'SHARED_CLAUSE', 0)
        else: bank = AffineBank.fresh_clause(bb, site, 50.0, 3, model['key'], np.zeros(bb.hidden, np.float32), np.zeros(bb.hidden, np.float32))
        opt = torch.optim.AdamW(bank.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0); before = []; after = []; g = None; loss_v = None; backbone_grad = None; used = sorted({v for b in bl for v in b['values']})
        for step in range(2):
            losses = []
            for b in bl:
                sub = b['queries'][:4]; pl = T.plan_for(bank, b, 1.0, record=False); r = bb.full_forward(b['prefix'], [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, workspace=pl['workspace'])
                lp = torch.stack([r['label_logps'][i][sub[i]['rec']['gold']] for i in range(4)]); (before if step == 0 else after).extend(lp.detach().float().cpu().tolist()); l_ = -lp.mean()/len(bl)
                if isinstance(bank, SetterBank): l_ = l_+Q.BETA*torch.cat([m['_delta_sq'] for m in pl['maps']]).mean()/(bank.cap_ref**2)/len(bl)
                bb.backward(l_); losses.append(float(l_.detach())); del r, lp, l_
            if step == 0:
                loss_v = float(sum(losses))
                if isinstance(bank, SetterBank): g = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in bank.model.named_parameters()}
                else: g = {f'{v}.{n}': float(p.grad.norm()) if p.grad is not None else None for v, e in bank.maps.items() for n, p in e.named_parameters() if v in used}
                backbone_grad = any(p.grad is not None for p in bb.model.parameters()); torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0); opt.step(); opt.zero_grad()
            else: opt.zero_grad()
        nonzero = (all(g[k] is not None and g[k] > 0 for k in ('basis_raw', 'head', 'bias')) if isinstance(bank, SetterBank) else all(g[f'{v}.output_factor'] > 0 for v in used))
        res[arm] = dict(loss=loss_v, grad_norms=g, used_values=used, backbone_grad=backbone_grad, finite=all(v is not None and np.isfinite(v) for v in g.values()), editor_grad_nonzero=nonzero, max_score_move=float(np.max(np.abs(np.asarray(after)-np.asarray(before)))), seconds=time.monotonic()-t0)
    rep = dict(at=Q.now(), actor=model['key'], procedure=proc, scenes=[s.scene_id for s in scenes], queries=16, results=res, backbone_unchanged=bb.verify_unchanged()['unchanged'],
               pass_=all(v['finite'] and v['editor_grad_nonzero'] and not v['backbone_grad'] and v['max_score_move'] > 0 for v in res.values()))
    Q.dump(out/'GRADIENT_CHECK.json', rep); return rep

# ---------------------------------------------------------------- disposable timing benchmark
def benchmark(bb, fit, out, model, proc):
    if done(out/'BENCHMARK.json'): return Q.load(out/'BENCHMARK.json')
    import torch; from rso3_editor import SetterBank, AffineBank, svd_basis; import rso3_train as T
    site = model['site']; s = sorted(fit, key=lambda s: Q.hash_rank('bench', s.scene_id))[0]; b = T.scene_bundle(bb, s, 'FIT', proc)
    sb = SetterBank(bb, 'INVARIANT_SET', site, 0, model['key'], 100.0); sb.set_center(np.zeros(bb.hidden, np.float32)); sb.set_basis(svd_basis(np.random.default_rng(4).normal(size=(64, bb.hidden)))[0])
    ab = AffineBank.fresh_clause(bb, site, 50.0, 4, model['key'], np.zeros(bb.hidden, np.float32), np.zeros(bb.hidden, np.float32))
    pre = b['prefix']; q12 = b['queries'][:12]; q24 = b['queries'][:24] if len(b['queries']) >= 24 else b['queries']*2
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    def timed(fn, n=3):
        fn(); sync(); t0 = time.monotonic()
        for _ in range(n): fn()
        sync(); return (time.monotonic()-t0)/n
    art = bb.compile(pre, capture={site}); pls = T.plan_for(sb, b, record=False); pla = T.plan_for(ab, b, 1.0, record=False); b8 = dict(b, addresses=b['addresses']*8, values=b['values']*8); pl8 = T.plan_for(sb, b8, record=False)
    t = dict(compile=timed(lambda: bb.compile(pre)), compile_capture=timed(lambda: bb.compile(pre, capture={site})), compile_shared=timed(lambda: T.compile_plan(bb, b, pls)), compile_affine_legacy=timed(lambda: T.compile_plan(bb, b, pla)), compile_repeat8=timed(lambda: T.compile_plan(bb, b8, pl8)),
             ask=timed(lambda: bb.ask(art, q12[0]['suffix'], q12[0]['labels'])), ask_batch12=timed(lambda: bb.ask_batch(art, [x['suffix'] for x in q12], [x['labels'] for x in q12])), ask_batch24=timed(lambda: bb.ask_batch(art, [x['suffix'] for x in q24], [x['labels'] for x in q24])),
             ask_generate32=timed(lambda: bb.ask(art, q12[0]['suffix'], q12[0]['labels'], generate=True), n=2), clone=timed(lambda: bb.clone_cache(art['cache'])), hash=timed(lambda: bb.cache_hash(art['cache'])))
    def step(bank):
        sub = q12[:4]; pl = T.plan_for(bank, b, 1.0, record=False); r = bb.full_forward(pre, [x['suffix'] for x in sub], [x['labels'] for x in sub], prefix_maps=pl['maps'], prefix_index=pl['index'], prefix_site=pl['site'], grad=True, workspace=pl['workspace'])
        loss = -torch.stack([r['label_logps'][i][sub[i]['rec']['gold']] for i in range(4)]).mean(); bb.backward(loss)
        for p in bank.parameters(): p.grad = None
    t['train_step4_setter'] = timed(lambda: step(sb), n=2); t['train_step4_affine'] = timed(lambda: step(ab), n=2)
    rep = dict(at=Q.now(), actor=model['key'], prefix_tokens=len(pre), suffix_tokens=[len(x['suffix']) for x in q12], cache_bytes=art['bytes'], seconds=t, gpu_memory_peak_bytes=(int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None), note='disposable timing panel; not a measurement')
    Q.dump(out/'BENCHMARK.json', rep); return rep

def phase_q(bb, out, deadline, model, fit, cal):
    qual = qualify(bb, cal, out, deadline, model); event(out, 'QUALIFICATION', selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'], equality=qual['equality_summary'])
    an = anchors(bb, out, deadline, model); event(out, 'ANCHORS', status=an['status'], bit_identical=an.get('bit_identical'), seeds={k: (v['rows'], v['same_choice'], v['max_abs_logp'], v['hash_equal'], v['program_rows'], v['program_same_choice'], v['program_hash_equal']) for k, v in an.get('seeds', {}).items()})
    proc = qual['selected_procedure'] or qual['natural_panel_procedure']
    ic = interface_checks(bb, cal, out, deadline, model, proc); event(out, 'INTERFACE_CHECK', pass_=ic['pass_'], cached_vs_full=ic['cached_vs_full'], batched=ic['batched_vs_single'], identities={k: v for k, v in ic['setter_identities'].items() if k != 'rows'})
    gc = gradient_check(bb, fit, out, deadline, model, proc); event(out, 'GRADIENT_CHECK', pass_=gc['pass_'], results={k: (v['editor_grad_nonzero'], v['backbone_grad'], v['max_score_move']) for k, v in gc['results'].items()})
    bench = benchmark(bb, fit, out, model, proc); event(out, 'BENCHMARK', seconds=bench['seconds'])
    return dict(qualification=qual['selected_procedure'], anchors=an['status'], interface=ic['pass_'], gradient=gc['pass_'])

def main():
    raise RuntimeError("Original provider launcher is not distributed. Use the CPU reproduction entry point; optional model reruns require an explicit local runtime configuration.")

if __name__ == '__main__': main()
