"""QBRC196 phase F: fresh FINAL evaluation against sealed, question-blind compiled contexts.

F1 SEAL   every compiled artifact (source, natural counterfactual, learned PREFIX methods x seeds) is compiled from prefix + explicit edit
          request only and its hash recorded BEFORE the first FINAL question is generated (SEAL.json timestamps the boundary).
F2 EVAL   questions (both draws, challenge families) are generated, artifacts recompiled and verified against the seal, every condition
          scored on a clone per question; sampled questions verify the master hash before/after.
F3 CTRL   matched random-direction and wrong-address controls on the fixed 16-scene/task subset.
F4 COMP   two independent addressed edits composed in one compilation (A then B, B then A) on the fixed 16-scene/task subset.
F5 DOWN   receiver/aggregator chain on the fixed 32 relation scenes.
"""
from __future__ import annotations
import math
import random
import time
from pathlib import Path
import numpy as np
import qbrc_common as Q
from qbrc_unit import event, done
import qbrc_train as T

def load_banks(bb, freeze, ddir, model):
    banks = {}
    for fam, sel in freeze['selection'].items():
        if sel.get('status') != 'SELECTED': continue
        for seed in Q.SEEDS:
            ck = Path(freeze['checkpoints'][fam][f's{seed}'])
            if not ck.is_dir(): ck = ddir/'candidates'/ck.parent.name/ck.name
            if not ck.is_dir(): continue
            banks[(fam, seed)] = dict(bank=T.EditorBank.load(bb, ck), strength=sel['strength'], site=sel['site'], checkpoint=str(ck), meta_sha256=Q.sha(ck/'META.json'), selection=sel)
    cap, mus, mu_late, vectors = T.load_captures(bb, ddir, model)
    mean_sel = freeze['late_mean']['selected']; caps = freeze['caps']
    banks[('LATE_MEAN', 0)] = dict(bank=T.MeanBank(bb, freeze['late_mean']['site'], vectors, {t: caps[f'{freeze["late_mean"]["site"]}|{t}'] for t in freeze['procedures']}), strength=mean_sel['strength'], site=freeze['late_mean']['site'], checkpoint=str(ddir/'captures.npz'), meta_sha256=cap['captures_sha256'], selection=mean_sel)
    return banks

def prefix_families(banks): return [k for k in banks if k[0] in ('PREFIX_LAST', 'PREFIX_DISTRIBUTED')]

def best_prefix(freeze):
    cands = [(f, s) for f, s in freeze['selection'].items() if f in ('PREFIX_LAST', 'PREFIX_DISTRIBUTED') and s.get('status') == 'SELECTED']
    if not cands: return None
    return min(cands, key=lambda x: (-round(x[1]['score'], 12), round(x[1]['harm'], 12)))[0]

def condition_name(key): return f'{key[0]}_s{key[1]}' if key[0] != 'LATE_MEAN' else 'LATE_MEAN'

# ---------------------------------------------------------------- F1 seal
def compile_learned(bb, b, entry, fam, record=False, actor=None, project=None):
    bb_ = b if actor is None else b['alt'][(actor, project)]
    pl = T.plan_for(entry['bank'], bb_, fam, entry['strength'], record=record)
    return bb.compile(b['prefix'], maps=pl['maps'], index=pl['index'], site=pl['site'])

def seal(bb, scenes, procedures, banks, out, deadline, model):
    if done(out/'SEAL.json'): return Q.load(out/'SEAL.json')
    rows = []; t0 = time.monotonic(); pref = prefix_families(banks)
    for s in scenes:
        Q.budget(deadline, 120); proc = procedures[s.task]; text, span = Q.prefix_text(s, proc); pre = bb.prefix_ids(text); cp = bb.clause_positions(text, span)
        req = Q.compile_request(s, proc, bb.cfg['revision'], 'source'); b = dict(prefix=pre, clause=cp, task=s.task, new_value=req.new_value, n_prefix=len(pre))
        src = bb.compile(pre, capture={model['late_site']}); tgt = Q.apply_edit(s); ttext, _ = Q.prefix_text(s, proc, target=tgt); tpre = bb.prefix_ids(ttext); cf = bb.compile(tpre)
        row = dict(scene_id=s.scene_id, task=s.task, procedure=proc, prefix_tokens=len(pre), clause_tokens=len(cp), request_key=req.key(), artifacts=dict(SOURCE=dict(hash=src['hash'], bytes=src['bytes']), COUNTERFACTUAL=dict(hash=cf['hash'], bytes=cf['bytes'], prefix_tokens=len(tpre))))
        for key in pref:
            entry = banks[key]; art = compile_learned(bb, b, entry, key[0]); r = art['realized']
            row['artifacts'][condition_name(key)] = dict(hash=art['hash'], bytes=art['bytes'], edited_tokens=r['positions'], energy=r['energy'], mean_realized_norm=float(np.mean(r['realized_norms'][0])), method_digest=entry['meta_sha256'],
                                                          request_key=Q.compile_request(s, proc, bb.cfg['revision'], entry['meta_sha256']).key())
        rows.append(row)
    Q.write_rows(out/'SEAL.jsonl', rows)
    rep = dict(at=Q.now(), scenes=len(rows), conditions=['SOURCE', 'COUNTERFACTUAL']+[condition_name(k) for k in pref], seal_sha256=Q.sha(out/'SEAL.jsonl'), seconds=time.monotonic()-t0,
               compiler_inputs='prefix token ids from prefix_text(scene) and the edit plan from CompileRequest (addressed clause span, task, desired value) only; no question, label mapping, option table, flag or gold was generated before this file existed',
               questions_generated_before_seal=False)
    Q.dump(out/'SEAL.json', rep); return rep

# ---------------------------------------------------------------- F2 evaluation
def final_bundle(bb, s, proc, target=None):
    b = T.scene_bundle(bb, s, 'FINAL', proc)
    if target is not None:
        qs = []
        for draw in (0, 1):
            for q in Q.questions(s, 'FINAL', draw, target=target):
                enc = bb.encode(b['text'], Q.suffix_text(q, proc)); qs.append(dict(q=q, suffix=enc['suffix_ids'], labels=bb.label_ids(enc['full_text'], q.labels), rec=Q.query_record(s, q, 'FINAL', draw)))
        b['queries'] = qs
    return b

def eval_condition(bb, b, art, late, src_preds, name, rows, sample_every=8, generate=False, extra=None):
    for j, qq in enumerate(b['queries']):
        verify = (j % sample_every == 0); res = bb.ask(art, qq['suffix'], qq['labels'], late=late, verify_master=verify, generate=generate)
        rec = dict(qq['rec']); sp = src_preds.get(rec['query_id']) if src_preds else None
        rec.update(condition=name, prediction=res['prediction'], logps=res['logps'], answer_mass=res['answer_mass'], gold_logp=res['logps'][rec['gold']], argmax_in_labels=res['argmax_in_labels'], tie=res['tie'],
                   source_prediction=sp['prediction'] if sp else res['prediction'], artifact_hash=art['hash'], master_verified=res.get('master_unchanged') if verify else None,
                   realized_norm=(float(np.mean(res['realized']['realized_norms'][0])) if res.get('realized') else None))
        if generate: rec['generation'] = res['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(res['generation']['text'], rec['labels']); rec['first_token_is_argmax'] = res['generation']['first_token_is_argmax']
        if extra: rec.update(extra)
        rows.append(rec)

def evaluate_final(bb, scenes, procedures, banks, out, deadline, model, seal_rows, gen_scenes):
    if done(out/'EVAL_DONE.json'): return Q.load(out/'EVAL_DONE.json')
    seal_by = {r['scene_id']: r for r in seal_rows}; pref = prefix_families(banks); late_keys = [k for k in banks if k[0] in ('LATE', 'LATE_MEAN')]
    conds = ['SOURCE', 'COUNTERFACTUAL']+[condition_name(k) for k in late_keys]+[condition_name(k) for k in pref]
    writers = {c: [] for c in conds}; mismatches = []; t0 = time.monotonic(); first_question_at = None; n_done = 0
    rng = random.Random('final-order'); order = list(scenes); rng.shuffle(order)
    for s in order:
        Q.budget(deadline, 300); proc = procedures[s.task]
        if first_question_at is None: first_question_at = Q.now()
        b = final_bundle(bb, s, proc); sr = seal_by[s.scene_id]; gen = s.scene_id in gen_scenes
        src = bb.compile(b['prefix'], capture={model['late_site']}); cf = bb.compile(b['target_prefix'])
        for nm, art in (('SOURCE', src), ('COUNTERFACTUAL', cf)):
            if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, condition=nm, sealed=sr['artifacts'][nm]['hash'], recompiled=art['hash']))
        # source rows first (define source predictions for harm)
        src_rows = []; eval_condition(bb, b, src, None, None, 'SOURCE', src_rows, generate=gen); src_preds = {r['query_id']: r for r in src_rows}; writers['SOURCE'].extend(src_rows)
        cf_rows = []
        for j, qq in enumerate(b['queries']):
            res = bb.ask(cf, b['target_queries'][j]['suffix'], qq['labels'], verify_master=(j % 8 == 0), generate=gen); rec = dict(qq['rec'])
            rec.update(condition='COUNTERFACTUAL', prediction=res['prediction'], logps=res['logps'], answer_mass=res['answer_mass'], gold_logp=res['logps'][rec['gold']], argmax_in_labels=res['argmax_in_labels'], tie=res['tie'], source_prediction=src_preds[rec['query_id']]['prediction'], artifact_hash=cf['hash'], master_verified=res.get('master_unchanged'))
            if gen: rec['generation'] = res['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(res['generation']['text'], rec['labels'])
            cf_rows.append(rec)
        writers['COUNTERFACTUAL'].extend(cf_rows)
        for key in late_keys:
            entry = banks[key]; late = T.plan_for(entry['bank'], b, 'LATE', entry['strength'], record=False)['late']
            eval_condition(bb, b, src, late, src_preds, condition_name(key), writers[condition_name(key)], generate=gen and key[1] == 0)
        for key in pref:
            entry = banks[key]; art = compile_learned(bb, b, entry, key[0]); nm = condition_name(key)
            if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, condition=nm, sealed=sr['artifacts'][nm]['hash'], recompiled=art['hash']))
            eval_condition(bb, b, art, None, src_preds, nm, writers[nm], generate=gen and key[1] == 0, extra=dict(edited_tokens=art['realized']['positions'], energy=art['realized']['energy'], cache_bytes=art['bytes']))
        n_done += 1
        if n_done % 16 == 0: event(out, 'FINAL_PROGRESS', scenes=n_done, seconds=time.monotonic()-t0, mismatches=len(mismatches))
    for c, rows in writers.items(): Q.write_rows(out/'final'/f'{c}.jsonl', rows)
    rep = dict(at=Q.now(), first_question_generated_at=first_question_at, seal_at=Q.load(out/'SEAL.json')['at'], scenes=len(scenes), conditions=conds, rows={c: len(r) for c, r in writers.items()},
               seal_mismatches=mismatches, recompile_deterministic=not mismatches, seconds=time.monotonic()-t0, generation_scenes=sorted(gen_scenes),
               master_verified_samples={c: int(sum(1 for r in rows if r.get('master_verified'))) for c, rows in writers.items()}, master_failures={c: int(sum(1 for r in rows if r.get('master_verified') is False)) for c, rows in writers.items()})
    Q.dump(out/'EVAL_DONE.json', rep); return rep

# ---------------------------------------------------------------- F3 controls
def controls(bb, scenes, procedures, banks, freeze, out, deadline, model):
    if done(out/'CONTROLS.json'): return Q.load(out/'CONTROLS.json')
    fam = best_prefix(freeze)
    if fam is None: Q.dump(out/'CONTROLS.json', dict(at=Q.now(), status='NOT_MEASURED', reason='no prefix method selected')); return
    entry = banks[(fam, 0)]; sub = Q.subset_by_id([s for s in scenes if s.task in procedures], Q.CONTROL_SCENES_PER_TASK, 'controls'); rows = {'LEARNED': [], 'RANDOM_MATCHED': [], 'WRONG_ADDRESS': []}; t0 = time.monotonic()
    import torch; g = torch.Generator(device='cpu').manual_seed(2609)
    for s in sub:
        Q.budget(deadline, 300); proc = procedures[s.task]; b = final_bundle(bb, s, proc)
        src = bb.compile(b['prefix']); src_rows = []; eval_condition(bb, b, src, None, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}
        art = compile_learned(bb, b, entry, fam); eval_condition(bb, b, art, None, src_preds, 'LEARNED', rows['LEARNED'], extra=dict(family=fam))
        norms = torch.as_tensor(art['realized']['realized_norms'][0], dtype=torch.float32)
        def rand_fn(h, z, norms=norms):
            d = torch.randn(h.shape, generator=g).to(h.device); d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-12)*norms.to(h.device)[:h.shape[0]].unsqueeze(-1); return d
        pl = T.plan_for(entry['bank'], b, fam, entry['strength'], record=False)
        rart = bb.compile(b['prefix'], maps=[dict(fn=rand_fn, clause_positions=b['clause'])], index=pl['index'], site=pl['site'])
        eval_condition(bb, b, rart, None, src_preds, 'RANDOM_MATCHED', rows['RANDOM_MATCHED'], extra=dict(family=fam, matched_norm=float(norms.mean())))
        # wrong address: same bank/magnitude, another member's record; gold under the intended world and under the wrong-address world both recorded
        wa = Q.wrong_address(s); wreq = Q.compile_request(s, proc, bb.cfg['revision'], entry['meta_sha256'], actor=wa['actor'], project=wa['project']); wcp = bb.clause_positions(b['text'], wreq.clause_char_span)
        wb = dict(b, clause=wcp, new_value=wreq.new_value); wpl = T.plan_for(entry['bank'], wb, fam, entry['strength'], record=False)
        wart = bb.compile(b['prefix'], maps=wpl['maps'], index=wpl['index'], site=wpl['site'])
        wworld = Q.apply_edit(s, actor=wa['actor'], project=wa['project']); wq = {q.query_id: q for draw in (0, 1) for q in Q.questions(s, 'FINAL', draw, target=wworld)}
        wrows = []; eval_condition(bb, b, wart, None, src_preds, 'WRONG_ADDRESS', wrows, extra=dict(family=fam, wrong_actor=s.actors[wa['actor']], wrong_new_value=wreq.new_value))
        for r in wrows: r['wrong_world_gold'] = int(wq[r['query_id']].gold_index); r['wrong_world_affected'] = bool(wq[r['query_id']].affected)
        rows['WRONG_ADDRESS'].extend(wrows)
    for c, rs in rows.items(): Q.write_rows(out/'controls'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), family=fam, scenes=[s.scene_id for s in sub], rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED')
    Q.dump(out/'CONTROLS.json', rep); return rep

# ---------------------------------------------------------------- F4 composition
def composition(bb, scenes, procedures, banks, freeze, out, deadline, model):
    if done(out/'COMPOSITION.json'): return Q.load(out/'COMPOSITION.json')
    fam = best_prefix(freeze); sub = Q.subset_by_id([s for s in scenes if s.task in procedures], Q.COMPOSITION_SCENES_PER_TASK, 'composition'); rows = {}; t0 = time.monotonic()
    import torch
    for s in sub:
        Q.budget(deadline, 300); proc = procedures[s.task]; b2 = Q.second_edit(s)
        joint = Q.apply_edit(Q.apply_edit(s), actor=b2['actor'], project=b2['project']); b = final_bundle(bb, s, proc, target=joint)
        reqA = Q.compile_request(s, proc, bb.cfg['revision'], 'A'); reqB = Q.compile_request(s, proc, bb.cfg['revision'], 'B', actor=b2['actor'], project=b2['project'])
        cpA = b['clause']; cpB = bb.clause_positions(b['text'], reqB.clause_char_span); n = b['n_prefix']
        src = bb.compile(b['prefix'], capture={model['late_site']}); src_rows = []; eval_condition(bb, b, src, None, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}
        rows.setdefault('SOURCE', []).extend(src_rows)
        jtext, _ = Q.prefix_text(s, proc, target=joint); jpre = bb.prefix_ids(jtext); jart = bb.compile(jpre); jrows = []
        for j, qq in enumerate(b['queries']):
            enc = bb.encode(jtext, Q.suffix_text(qq['q'], proc)); res = bb.ask(jart, enc['suffix_ids'], qq['labels']); rec = dict(qq['rec'])
            rec.update(condition='COUNTERFACTUAL_JOINT', prediction=res['prediction'], logps=res['logps'], source_prediction=src_preds[rec['query_id']]['prediction'], artifact_hash=jart['hash']); jrows.append(rec)
        rows.setdefault('COUNTERFACTUAL_JOINT', []).extend(jrows)
        def maps_for(entry, family, order):
            ms = []
            for tag in order:
                cp, nv = (cpA, reqA.new_value) if tag == 'A' else (cpB, reqB.new_value); fn = entry['bank'].fn(s.task, nv, entry['strength'], record=False) if not isinstance(entry['bank'], T.MeanBank) else entry['bank'].fn(s.task, nv, entry['strength'])
                if family == 'PREFIX_LAST': idx = [n-1]
                elif family == 'PREFIX_DISTRIBUTED': idx = list(range(cp[0], n))
                else: idx = None
                m = dict(fn=fn, clause_positions=cp)
                if idx is not None: m['index'] = (torch.zeros(len(idx), dtype=torch.long, device=bb.device), torch.tensor(idx, device=bb.device))
                ms.append(m)
            return ms
        for key in [k for k in banks if k[1] == 0 and k[0] in (fam, 'LATE', 'LATE_MEAN')]:
            entry = banks[key]; family = key[0]
            for order in ('AB', 'BA'):
                nm = f'{condition_name(key)}_{order}'
                if family in ('LATE', 'LATE_MEAN'):
                    late = dict(site=entry['site'], maps=maps_for(entry, family, order)); eval_condition(bb, b, src, late, src_preds, nm, rows.setdefault(nm, []))
                else:
                    ms = maps_for(entry, family, order); union = sorted(set(int(x) for m in ms for x in m['index'][1].tolist()))
                    art = bb.compile(b['prefix'], maps=ms, index=union, site=entry['site']); eval_condition(bb, b, art, None, src_preds, nm, rows.setdefault(nm, []), extra=dict(edited_tokens=art['realized']['positions'], energy=art['realized']['energy']))
        # output-only sanity comparators: global label flip (relation) / random label (priority), and per-scene recorded second-edit metadata
        for r in src_rows:
            flip = dict(r); flip['condition'] = 'GLOBAL_LABEL_FLIP'
            flip['prediction'] = (1-r['prediction']) if len(r['labels']) == 2 else random.Random(r['query_id']).randrange(len(r['labels'])); rows.setdefault('GLOBAL_LABEL_FLIP', []).append(flip)
        rows.setdefault('META', []).append(dict(scene_id=s.scene_id, task=s.task, edit_A=dict(actor=s.actors[s.target_actor], value=reqA.new_value), edit_B=dict(actor=s.actors[b2['actor']], value=reqB.new_value), clause_A=cpA, clause_B=cpB))
    for c, rs in rows.items(): Q.write_rows(out/'composition'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), family=fam, scenes=[s.scene_id for s in sub], rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED', note='joint gold = both edits applied; neither joint example was used in training; z recomputed from the current within-hook residual before each map')
    Q.dump(out/'COMPOSITION.json', rep); return rep

# ---------------------------------------------------------------- F5 receiver/aggregator
def downstream(bb, scenes, procedures, banks, freeze, out, deadline, model):
    if done(out/'DOWNSTREAM.json'): return Q.load(out/'DOWNSTREAM.json')
    if 'relation' not in procedures: Q.dump(out/'DOWNSTREAM.json', dict(at=Q.now(), status='NOT_MEASURED', reason='relation task not qualified')); return
    fam = best_prefix(freeze); proc = procedures['relation']; sub = sorted(Q.by_task(scenes, 'relation'), key=lambda s: Q.hash_rank('downstream', s.scene_id))[:Q.DOWNSTREAM_SCENES]; rows = []; t0 = time.monotonic()
    conds = [('SOURCE', None), ('COUNTERFACTUAL', None)]+[(condition_name(k), k) for k in banks if k[1] == 0 and k[0] in ('LATE', fam)]
    for s in sub:
        Q.budget(deadline, 300); a = s.target_actor; p = s.target_project; bidx = (a+1) % len(s.actors); A, B, P = s.actors[a], s.actors[bidx], s.targets[p]
        text, span = Q.prefix_text(s, proc); pre = bb.prefix_ids(text); cp = bb.clause_positions(text, span); req = Q.compile_request(s, proc, bb.cfg['revision'], 'down')
        b = dict(prefix=pre, clause=cp, task='relation', new_value=req.new_value, n_prefix=len(pre), text=text)
        tgt = Q.apply_edit(s); ttext, _ = Q.prefix_text(s, proc, target=tgt); tpre = bb.prefix_ids(ttext)
        gold_src = bool(s.values[a][p] and s.values[bidx][p]); gold_tgt = bool(tgt.values[a][p] and tgt.values[bidx][p])
        arts = {'SOURCE': bb.compile(pre, capture={model['late_site']}), 'COUNTERFACTUAL': bb.compile(tpre)}
        for nm, key in conds:
            if key is not None and key[0] != 'LATE': arts[nm] = compile_learned(bb, b, banks[key], key[0])
        for nm, key in conds:
            art = arts.get(nm, arts['SOURCE']); reports = {}; ok = True
            for who in (A, B):
                qtext = f'Does {who} favor the {P} proposal?\nAnswer with exactly one of No, Yes.'; ptext = ttext if nm == 'COUNTERFACTUAL' else text
                enc = bb.encode(ptext, Q.suffix_text_plain(qtext, proc)); lids = bb.label_ids(enc['full_text'], ('No', 'Yes'))
                late = T.plan_for(banks[key]['bank'], b, 'LATE', banks[key]['strength'], record=False)['late'] if (key is not None and key[0] == 'LATE') else None
                res = bb.ask(art, enc['suffix_ids'], lids, late=late, generate=True); sym, status = Q.parse_symbol(res['generation']['text'], ('No', 'Yes'))
                reports[who] = dict(text=res['generation']['text'], symbol=sym, status=status, score_prediction=res['prediction']); ok = ok and sym is not None
            if ok:
                agg = (f'Two committee members were asked separately about the {P} proposal.\nReport on {A}: {A} {"supports" if reports[A]["symbol"] == "Yes" else "opposes"} the {P} proposal.\n'
                       f'Report on {B}: {B} {"supports" if reports[B]["symbol"] == "Yes" else "opposes"} the {P} proposal.\nIs the {P} proposal supported by both {A} and {B}?\nAnswer with exactly one of No, Yes.'+Q.INSTRUCTION)
                full = bb.tokenizer.apply_chat_template([{'role': 'user', 'content': agg}], tokenize=False, add_generation_prompt=True); ids = bb.tokenizer.encode(full, add_special_tokens=False)
                lids = bb.label_ids(full, ('No', 'Yes'))
                # aggregator: plain unedited prompt through the same greedy protocol (prefix = whole prompt minus nothing; use full forward generation)
                gres = bb.ask(bb.compile(ids[:1]), ids[1:], lids, generate=True); asym, astatus = Q.parse_symbol(gres['generation']['text'], ('No', 'Yes'))
                final = (asym == 'Yes') if asym is not None else None
            else: agg = None; asym = None; astatus = 'RECEIVER_FAILED'; final = None
            rows.append(dict(scene_id=s.scene_id, condition=nm, A=A, B=B, project=P, receiver=reports, aggregator_prompt=agg, aggregator_answer=asym, aggregator_status=astatus, chain_output=final,
                             gold_source=gold_src, gold_target=gold_tgt, joint_changes=gold_src != gold_tgt, correct=(final == gold_tgt) if nm != 'SOURCE' else (final == gold_src), correct_vs_target=final == gold_tgt,
                             receiver_gold_A=bool(tgt.values[a][p]) if nm != 'SOURCE' else bool(s.values[a][p]), receiver_gold_B=bool(s.values[bidx][p])))
    Q.write_rows(out/'downstream'/'rows.jsonl', rows)
    rep = dict(at=Q.now(), family=fam, scenes=[s.scene_id for s in sub], conditions=[c for c, _ in conds], rows=len(rows), seconds=time.monotonic()-t0, status='MEASURED')
    Q.dump(out/'DOWNSTREAM.json', rep); return rep

# ---------------------------------------------------------------- natural panel for tasks that failed native qualification
def natural_panel(bb, fin, qual, out, deadline, final_scenes_per_task):
    if done(out/'NATURAL_PANEL.json'): return Q.load(out/'NATURAL_PANEL.json')
    procs = qual.get('natural_panel_procedures', {}); rows = {}; t0 = time.monotonic()
    for task, proc in procs.items():
        scenes = sorted(Q.by_task(fin, task), key=lambda s: Q.hash_rank('final-fallback', s.scene_id))[:final_scenes_per_task] if final_scenes_per_task < 64 else Q.by_task(fin, task)
        for s in scenes:
            Q.budget(deadline, 300); b = final_bundle(bb, s, proc); src = bb.compile(b['prefix']); cf = bb.compile(b['target_prefix'])
            src_rows = []; eval_condition(bb, b, src, None, None, 'SOURCE', src_rows); src_preds = {r['query_id']: r for r in src_rows}; rows.setdefault(f'{task}_SOURCE', []).extend(src_rows)
            for j, qq in enumerate(b['queries']):
                res = bb.ask(cf, b['target_queries'][j]['suffix'], qq['labels']); rec = dict(qq['rec'])
                rec.update(condition='COUNTERFACTUAL', prediction=res['prediction'], logps=res['logps'], answer_mass=res['answer_mass'], source_prediction=src_preds[rec['query_id']]['prediction'], artifact_hash=cf['hash']); rows.setdefault(f'{task}_COUNTERFACTUAL', []).append(rec)
    for c, rs in rows.items(): Q.write_rows(out/'natural'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), procedures=procs, rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED' if rows else 'NOT_APPLICABLE',
               note='native source/counterfactual competence on FINAL for tasks that failed qualification; no learned intervention is evaluated or claimed there')
    Q.dump(out/'NATURAL_PANEL.json', rep); return rep

# ---------------------------------------------------------------- driver
def phase_f(bb, out, deadline, model, fit, cal, fin, qdir, ddir, freeze_dir, final_scenes_per_task):
    freeze = Q.load(freeze_dir/'FREEZE.json'); procedures = freeze['procedures']
    if Q.sha(freeze_dir/'FREEZE.json') != Q.sha(ddir/'FREEZE.json'): raise ValueError('capsule FREEZE differs from the phase D FREEZE')
    if not done(out/'FREEZE_VERIFIED.json'): Q.dump(out/'FREEZE_VERIFIED.json', dict(at=Q.now(), freeze_sha256=Q.sha(freeze_dir/'FREEZE.json'), selection=freeze['selection'], procedures=procedures))
    banks = load_banks(bb, freeze, ddir, model); event(out, 'BANKS', conditions=[condition_name(k) for k in banks], strengths={condition_name(k): v['strength'] for k, v in banks.items()})
    scenes = [s for s in fin if s.task in procedures]
    bench = Q.load(qdir/'BENCHMARK.json')['seconds']; n_cond = 2+len(banks); per_scene = (n_cond*bench['compile']+26*n_cond*bench['ask'])*1.2
    predicted_full = per_scene*len(scenes)+bench['ask_generate32']*Q.GENERATION_SCENES_PER_TASK*2*26*4+3600; remaining = Q.budget(deadline, 0)-Q.BUDGET['closure_reserve_minutes']*60
    if final_scenes_per_task < 64 or predicted_full > remaining:
        n = min(final_scenes_per_task, Q.FINAL_FALLBACK_SCENES_PER_TASK) if predicted_full > remaining else final_scenes_per_task; scenes = Q.subset_by_id(scenes, n, 'final-fallback'); scope = f'FALLBACK_{n}'
    else: scope = 'FULL_64'
    if not done(out/'SCOPE.json'): Q.dump(out/'SCOPE.json', dict(at=Q.now(), scope=scope, scenes=len(scenes), predicted_full_seconds=predicted_full, remaining_seconds=remaining, rule='decided from measured timings before any FINAL outcome; fallback = first 32 scenes per task by hashed ID'))
    event(out, 'SCOPE', scope=scope, scenes=len(scenes))
    sl = seal(bb, scenes, procedures, banks, out, deadline, model); event(out, 'SEALED', scenes=sl['scenes'], sha256=sl['seal_sha256'])
    gen_scenes = {s.scene_id for s in Q.subset_by_id(scenes, Q.GENERATION_SCENES_PER_TASK, 'generation')}
    ev = evaluate_final(bb, scenes, procedures, banks, out, deadline, model, Q.read_rows(out/'SEAL.jsonl'), gen_scenes); event(out, 'FINAL_EVAL', rows=ev['rows'], mismatches=len(ev['seal_mismatches']))
    qual = Q.load(qdir/'QUALIFICATION.json')
    try: npn = natural_panel(bb, fin, qual, out, deadline, len(scenes)//max(1, len(procedures))); event(out, 'NATURAL_PANEL', status=npn['status'], rows=npn['rows'])
    except TimeoutError: event(out, 'NATURAL_PANEL', status='NOT_MEASURED'); raise
    for stage, fn in (('CONTROLS', controls), ('COMPOSITION', composition), ('DOWNSTREAM', downstream)):
        try:
            if Q.budget(deadline, 0) < 900: Q.dump(out/f'{stage}.json', dict(at=Q.now(), status='NOT_MEASURED', reason='insufficient remaining time')) if not done(out/f'{stage}.json') else None; event(out, stage, status='NOT_MEASURED'); continue
            r = fn(bb, scenes, procedures, banks, freeze, out, deadline, model); event(out, stage, status=(r or {}).get('status'))
        except TimeoutError:
            if not done(out/f'{stage}.json'): Q.dump(out/f'{stage}.json', dict(at=Q.now(), status='NOT_MEASURED', reason='resource stop'))
            event(out, stage, status='NOT_MEASURED'); raise
    return dict(status='FINAL_COMPLETE', scope=scope)
