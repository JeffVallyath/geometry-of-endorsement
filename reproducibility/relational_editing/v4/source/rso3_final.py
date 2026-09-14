"""RSO3 phase F: fresh evaluation against sealed, question-blind compiled contexts (FREEZE verified first).

F0 SCOPE     FULL / SMALL / QUARTER chosen from measured timings before any FINAL outcome (fixed hashed scene IDs preserving every
             size/direction cell, both orders, every method, both seeds, every program); never from outcomes.
F1 SEAL      every shared single-edit artifact (SOURCE, natural COUNTERFACTUAL, TEXT_CORRECTION, every selected/frozen method x seed)
             compiled from prefix + explicit command program only; hashes recorded BEFORE the first FINAL question exists.
F2 SINGLES   both prefix orders, both answer-mapping draws, the original 24 questions; sampled master-hash verification; generation panel.
F3 PROGRAMS  single / noop / repeat2 / repeat4 / repeat8 / restore / restore_after4 / overwrite / AB / BA / ABC / CBA / AB_cross inside ONE
             compilation each (every declared command executes; FP32 workspace for the new methods; historical BF16 path for the frozen V2
             references; CANONICAL_V2 normalizes then replays from the source), scored against the true final world with coverage questions;
             equal-terminal programs share a measurement only when their compiled tensors are byte-identical (recorded).
F4 WORKFLOW  three actual receiver reports from the compiled artifact -> separate unedited majority aggregator, for one update, duplicate
             delivery, restoration and two distinct updates; invalid outputs fail; every generated report preserved.
F5 CONTROLS  matched-mask random update and wrong-address update for the selected strict setter (small panel).
F6 WITNESS   post-block residual, next-block input, key/value tensors, FP32 workspace and BF16-cast states for four fixed scenes
             (source / edited / repeated / restored).
"""
from __future__ import annotations
import math
import random
import time
from pathlib import Path
import numpy as np
import rso3_common as Q
import rso3_train as T
from rso3_editor import SetterBank, AffineBank
from rso3_unit import event, done

RECEIVER_Q = 'Does {who} favor the {project} proposal?\nAnswer with exactly one of No, Yes.'
PHRASING = 'Report on {who}: {who} {verb} the {project} proposal.'
WORKFLOW_METHODS = ('INVARIANT_SET', 'FREE_OVERWRITE', 'CONTINUED_AFFINE', 'AFFINE_CLAUSE', 'FROZEN_V2_CLAUSE', 'CANONICAL_V2', 'CANONICAL_AFFINE')

# ---------------------------------------------------------------- banks
def load_banks(bb, freeze, ddir, model):
    """{name: entry}; entry = dict(bank, arm, seed, strength, workspace, footprint, canonical, checkpoint, ...)."""
    banks = {}
    for arm, sel in freeze['selection'].items():
        for seed in Q.SEEDS:
            ck = Path(sel['checkpoints'][str(seed)])
            if not ck.is_dir(): ck = ddir/'arms'/arm/ck.parent.name/ck.name
            bank = T.load_bank(bb, ck); setter = isinstance(bank, SetterBank)
            banks[f'{arm}_s{seed}'] = dict(bank=bank, arm=arm, seed=seed, strength=(1.0 if setter else sel['strength']), workspace=('fp32' if setter else 'legacy'), footprint='clause', canonical=False, checkpoint=str(ck), meta_sha256=Q.sha(ck/'META.json'), learned=True, setter=setter)
            if arm == 'AFFINE_CLAUSE':          # Qwen: the normalize-then-replay comparator built on its own trained affine clause map
                banks[f'CANONICAL_AFFINE_s{seed}'] = dict(bank=bank, arm='CANONICAL_AFFINE', seed=seed, strength=sel['strength'], workspace='legacy', footprint='clause', canonical=True, checkpoint=str(ck), meta_sha256=Q.sha(ck/'META.json'), learned=False, setter=False)
    if model['key'] == 'gemma' and bb.hidden == 3584:
        for name, spec in Q.FROZEN.items():
            for seed in Q.SEEDS:
                bank = AffineBank.from_v2(bb, spec['arm'], seed)
                banks[f'{name}_s{seed}'] = dict(bank=bank, arm=name, seed=seed, strength=spec['strength'], workspace=spec['workspace'], footprint=spec['footprint'], canonical=spec.get('canonical', False), checkpoint=str(Q.v2_factor_dir(spec['arm'], seed)), meta_sha256=bank.init['meta_sha256'], learned=False, setter=False)
    return banks

def plan(entry, b, record=False): return T.plan_for(entry['bank'], b, entry['strength'], record=record, workspace=entry['workspace'], footprint=entry['footprint'], canonical=entry['canonical'])
def compile_entry(bb, b, entry, capture=(), capture_next=()): return T.compile_plan(bb, b, plan(entry, b), capture=capture, capture_next=capture_next)
def single_names(banks): return [n for n, e in banks.items() if not e['canonical']]      # canonical == its frozen/affine base on a single command (recorded, not remeasured)

# ---------------------------------------------------------------- scoring helpers
def score_shared(bb, b, art, src_preds, name, rows, extra=None, sample_master=True, generate=False, suffixes=None):
    qs = b['queries']; suff = suffixes or [qq['suffix'] for qq in qs]; res = []
    for k in range(0, len(qs), Q.CHUNK): res += bb.ask_batch(art, suff[k:k+Q.CHUNK], [qq['labels'] for qq in qs[k:k+Q.CHUNK]], verify_master=(sample_master and k == 0 and art.get('hash') is not None))
    out = []
    for j, (qq, rr) in enumerate(zip(qs, res)):
        rec = dict(qq['rec']); sp = src_preds.get(rec['query_id']) if src_preds else None
        rec.update(condition=name, prediction=rr['prediction'], logps=rr['logps'], answer_mass=rr['answer_mass'], gold_logp=rr['logps'][rec['gold']], argmax_in_labels=rr['argmax_in_labels'], tie=rr['tie'], source_prediction=sp['prediction'] if sp else rr['prediction'],
                   source_logps=sp['logps'] if sp else rr['logps'], artifact_hash=art.get('hash'), master_verified=rr.get('master_unchanged'))
        if generate and j < 12:
            g = bb.ask(art, suff[j], qq['labels'], generate=True); rec['generation'] = g['generation']['text']; rec['generation_symbol'], rec['parse_status'] = Q.parse_symbol(g['generation']['text'], rec['labels']); rec['first_token_is_argmax'] = g['generation']['first_token_is_argmax']
        if extra: rec.update(extra)
        rows.append(rec); out.append(rec)
    return out

def natural_rows(bb, b, text, src_preds, name, rows, generate=False, extra=None, art=None):
    pre = bb.prefix_ids(text); art = art or bb.compile(pre); suff = [bb.encode(text, Q.suffix_text(qq['q']['text']))['suffix_ids'] for qq in b['queries']]
    score_shared(bb, b, art, src_preds, name, rows, extra=dict(prefix_tokens=len(pre), **(extra or {})), generate=generate, suffixes=suff); return art

def method_extra(entry, art, pl=None):
    r = art['realized']; return dict(method=entry['arm'], seed=entry['seed'], strength=entry['strength'], workspace=entry['workspace'], edited_tokens=r['positions'], energy=r['energy'], frobenius=r['frobenius'], commands=r['maps'], cast_error=r['cast_error'], canonical=entry['canonical'])

# ---------------------------------------------------------------- F0 scope
def choose_scope(bb, out, deadline, model, size, reserve_hours, banks, bench):
    if done(out/'SCOPE.json'): return Q.load(out/'SCOPE.json')
    n_single = len(single_names(banks))+3; n_prog = len(banks)+3; n_wf = sum(1 for e in banks.values() if e['arm'] in WORKFLOW_METHODS)+3
    per_single = bench['compile_shared']+2*bench['ask_batch12']; per_prog = bench['compile_repeat8']*0.6+bench['compile_shared']*0.4+3*bench['ask_batch12']; per_wf = bench['compile_shared']+4*bench['ask_generate32']+bench['hash']
    def predict(sz):
        s = Q.SIZES[model['key']][sz]
        final = s['final']*2*n_single*per_single*1.25+len(Q.PANELS)*0+8*24*bench['ask_generate32']*3; seal = s['final']*2*n_single*bench['compile_shared']*1.1
        programs = s['program']*len(Q.PROGRAMS)*n_prog*per_prog*1.2*0.8; workflow = s['workflow']*len(Q.WORKFLOW_PROGRAMS)*n_wf*per_wf*1.2*0.7
        panels = Q.PANELS['control']*3*per_single*1.2+Q.PANELS['witness']*8*bench['compile_capture']*3+600
        return dict(seal=seal, final=final, programs=programs, workflow=workflow, panels=panels, total=seal+final+programs+workflow+panels)
    remaining = Q.budget(deadline, 0)-Q.BUDGET['closure_reserve_minutes']*60-reserve_hours*3600; pred = {sz: predict(sz) for sz in ('FULL', 'SMALL', 'QUARTER')}
    chosen = size if size != 'AUTO' else next((sz for sz in ('FULL', 'SMALL', 'QUARTER') if pred[sz]['total'] <= remaining), 'QUARTER')
    rep = dict(at=Q.now(), scope=chosen, sizes=Q.SIZES[model['key']][chosen], predicted_seconds=pred, remaining_seconds=remaining, reserve_hours_after=reserve_hours, requested=size, fits=pred[chosen]['total'] <= remaining,
               rule='largest of FULL/SMALL/QUARTER whose predicted time from the disposable benchmark fits the remaining time minus the closure reserve and the reserve for the next actor; decided before any FINAL outcome; every size/direction cell, both orders, every method, both seeds and every program are kept')
    Q.dump(out/'SCOPE.json', rep); return rep

# ---------------------------------------------------------------- F1 seal
def prefix_bundle(bb, s, proc, program='single', order='early', variant=0):
    text, spans = Q.prefix_text(s, proc, order, variant); pre = bb.prefix_ids(text); edits = Q.programs(s)[program]
    addresses = [tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in edits]; values = [int(e.value) for e in edits]
    return dict(scene=s, proc=proc, program=program, order=order, variant=variant, text=text, prefix=pre, clause=addresses[0], addresses=addresses, values=values, n_prefix=len(pre), edits=edits, touched=[(e.actor, e.project) for e in edits],
                request_key=Q.OLD.CompileRequest(tuple(pre), tuple(addresses), tuple(values), 'seal').key())

def seal(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'SEAL.json'): return Q.load(out/'SEAL.json')
    rows = []; t0 = time.monotonic(); names = single_names(banks)
    for s in scenes:
        for order in Q.ORDERS:
            Q.budget(deadline, 120); b = prefix_bundle(bb, s, proc, 'single', order); world = Q.final_world(s, 'single')
            src = bb.compile(b['prefix']); ctext, _ = Q.prefix_text(s, proc, order, 0, world=world); cf = bb.compile(bb.prefix_ids(ctext)); ttext, _ = Q.prefix_text(s, proc, order, 0, corrections=b['edits']); tc = bb.compile(bb.prefix_ids(ttext))
            row = dict(scene_id=s.scene_id, order=order, program='single', prefix_tokens=len(b['prefix']), clause_tokens=len(b['clause']), clause_start=b['clause'][0], request_key=b['request_key'],
                       artifacts=dict(SOURCE=dict(hash=src['hash'], bytes=src['bytes']), COUNTERFACTUAL=dict(hash=cf['hash'], bytes=cf['bytes'], prefix_tokens=cf['prefix_len']), TEXT_CORRECTION=dict(hash=tc['hash'], bytes=tc['bytes'], prefix_tokens=tc['prefix_len'])))
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry); r = art['realized']
                row['artifacts'][nm] = dict(hash=art['hash'], bytes=art['bytes'], edited_tokens=r['positions'], energy=r['energy'], frobenius=r['frobenius'], cast_error=r['cast_error'], method_digest=entry['meta_sha256'], strength=entry['strength'], workspace=entry['workspace'])
            rows.append(row)
    Q.write_rows(out/'SEAL.jsonl', rows)
    rep = dict(at=Q.now(), artifacts=len(rows), scenes=len(scenes), conditions=['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+names, canonical_on_single=[n for n in banks if banks[n]['canonical']], seal_sha256=Q.sha(out/'SEAL.jsonl'), seconds=time.monotonic()-t0,
               compiler_inputs='prefix token ids, explicit clause spans and requested values of the command program, method weights; no question, label mapping, gold, affected flag or target world existed before this file', questions_generated_before_seal=False)
    Q.dump(out/'SEAL.json', rep); return rep

# ---------------------------------------------------------------- F2 singles
def evaluate_final(bb, scenes, proc, banks, out, deadline, model, gen_scenes):
    if done(out/'EVAL_DONE.json'): return Q.load(out/'EVAL_DONE.json')
    seal_by = {(r['scene_id'], r['order']): r for r in Q.read_rows(out/'SEAL.jsonl')}; names = single_names(banks); conds = ['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+names; writers = {c: [] for c in conds}; mismatches = []; t0 = time.monotonic(); first_q = None; n_done = 0
    order_ = list(scenes); random.Random('final-order').shuffle(order_)
    for s in order_:
        for order in Q.ORDERS:
            Q.budget(deadline, 300)
            if first_q is None: first_q = Q.now()
            b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order); sr = seal_by[(s.scene_id, order)]; gen = s.scene_id in gen_scenes and order == 'early'
            src = bb.compile(b['prefix'])
            if src['hash'] != sr['artifacts']['SOURCE']['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition='SOURCE'))
            src_rows = score_shared(bb, b, src, None, 'SOURCE', writers['SOURCE'], generate=gen); src_preds = {r['query_id']: r for r in src_rows}
            ctext, _ = Q.prefix_text(s, proc, order, 0, world=b['world']); cf = natural_rows(bb, b, ctext, src_preds, 'COUNTERFACTUAL', writers['COUNTERFACTUAL'], generate=gen)
            ttext, _ = Q.prefix_text(s, proc, order, 0, corrections=b['edits']); tc = natural_rows(bb, b, ttext, src_preds, 'TEXT_CORRECTION', writers['TEXT_CORRECTION'], generate=gen)
            for nm, art in (('COUNTERFACTUAL', cf), ('TEXT_CORRECTION', tc)):
                if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition=nm))
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry)
                if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition=nm))
                score_shared(bb, b, art, src_preds, nm, writers[nm], extra=dict(method_extra(entry, art), cache_bytes=art['bytes']), generate=gen and entry['seed'] == 0)
        n_done += 1
        if n_done % 8 == 0: event(out, 'FINAL_PROGRESS', scenes=n_done, seconds=time.monotonic()-t0, mismatches=len(mismatches))
    for c, rows in writers.items(): Q.write_rows(out/'final'/f'{c}.jsonl', rows)
    rep = dict(at=Q.now(), first_question_generated_at=first_q, seal_at=Q.load(out/'SEAL.json')['at'], scenes=len(scenes), conditions=conds, rows={c: len(r) for c, r in writers.items()}, seal_mismatches=mismatches, recompile_deterministic=not mismatches, seconds=time.monotonic()-t0,
               generation_scenes=sorted(gen_scenes), master_verified_samples={c: int(sum(1 for r in rows if r.get('master_verified'))) for c, rows in writers.items()}, master_failures={c: int(sum(1 for r in rows if r.get('master_verified') is False)) for c, rows in writers.items()})
    Q.dump(out/'EVAL_DONE.json', rep); return rep

# ---------------------------------------------------------------- F3 edit programs
def programs(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'PROGRAMS.json'): return Q.load(out/'PROGRAMS.json')
    rows = {}; ident = []; shared = []; t0 = time.monotonic(); n = 0; order = 'early'
    for s in scenes:
        Q.budget(deadline, 300); src = None; measured = {}   # measured[(name, hash)] -> (program, rows)  byte-identical sharing within a scene
        nat_cache = {}
        for program in Q.PROGRAMS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, program, order)
            if src is None: src = bb.compile(b['prefix'])
            src_rows = score_shared(bb, b, src, None, 'SOURCE', rows.setdefault('SOURCE', []), extra=dict(program=program), sample_master=False); src_preds = {r['query_id']: r for r in src_rows}
            ntext, _ = Q.prefix_text(s, proc, order, 0, world=b['world']); nat = nat_cache.get(ntext)
            nat_cache[ntext] = natural_rows(bb, b, ntext, src_preds, 'NATURAL_FINAL_WORLD', rows.setdefault('NATURAL_FINAL_WORLD', []), extra=dict(program=program, artifact_reused=nat is not None), art=nat)
            ttext, _ = Q.prefix_text(s, proc, order, 0, corrections=b['edits']); natural_rows(bb, b, ttext, src_preds, 'TEXT_CORRECTION_SEQ', rows.setdefault('TEXT_CORRECTION_SEQ', []), extra=dict(program=program, correction_lines=len(b['edits'])))
            for nm, entry in banks.items():
                art = compile_entry(bb, b, entry); key = (nm, art['hash']); ex = method_extra(entry, art); ex.update(program=program, index_len=len(art['_index']))
                if key in measured:
                    prev_program, prev_rows = measured[key]; prev = {r['query_id']: r for r in prev_rows}; ex['shared_measurement_from'] = prev_program; shared.append(dict(scene_id=s.scene_id, condition=nm, program=program, identical_to=prev_program, hash=art['hash']))
                    for qq in b['queries']:
                        rec = dict(qq['rec']); pr = prev.get(rec['query_id'])
                        if pr is None: continue          # coverage additions differ by program: unmatched rows are measured below
                        rec.update({k: pr[k] for k in ('condition', 'prediction', 'logps', 'answer_mass', 'argmax_in_labels', 'tie', 'source_prediction', 'source_logps', 'artifact_hash', 'master_verified')}); rec['gold_logp'] = pr['logps'][rec['gold']]; rec.update(ex); rows.setdefault(nm, []).append(rec)
                    missing = [qq for qq in b['queries'] if qq['rec']['query_id'] not in prev]
                    if missing:
                        sub = dict(b, queries=missing); score_shared(bb, sub, art, src_preds, nm, rows.setdefault(nm, []), extra=ex, sample_master=False)
                else:
                    new_rows = score_shared(bb, b, art, src_preds, nm, rows.setdefault(nm, []), extra=ex, sample_master=(program == 'single')); measured[key] = (program, new_rows)
                if program == 'single': measured.setdefault(('_single', nm), art)
                base = measured.get(('_single', nm))
                if base is not None and program != 'single' and base['_index'] == art['_index']:
                    ident.append(dict(scene_id=s.scene_id, condition=nm, program=program, workspace_max_abs_vs_single=float(np.max(np.abs(art['_workspace_states']-base['_workspace_states']))), bf16_max_abs_vs_single=float(np.max(np.abs(art['_after_states']-base['_after_states']))), hash_equal_single=art['hash'] == base['hash']))
                elif base is not None and program != 'single': ident.append(dict(scene_id=s.scene_id, condition=nm, program=program, workspace_max_abs_vs_single=None, bf16_max_abs_vs_single=None, hash_equal_single=art['hash'] == base['hash'], note='different edited index'))
        n += 1
        if n % 4 == 0: event(out, 'PROGRAMS_PROGRESS', scenes=n, seconds=time.monotonic()-t0)
    for c, rs in rows.items(): Q.write_rows(out/'programs'/f'{c}.jsonl', rs)
    Q.write_rows(out/'programs'/'IDENTITY.jsonl', ident); Q.write_rows(out/'programs'/'SHARED_MEASUREMENTS.jsonl', shared)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], order=order, programs=list(Q.PROGRAMS), rows={c: len(r) for c, r in rows.items()}, shared_measurements=len(shared), seconds=time.monotonic()-t0, status='MEASURED',
               note='every declared command executes inside ONE prefix compilation (FP32 workspace for the new methods, historical BF16 path for the frozen V2 references; CANONICAL_* normalizes then replays from the source); gold from the true final world; coverage questions flagged additional; byte-identical compiled tensors share a measurement (recorded); no parameter was trained on any program')
    Q.dump(out/'PROGRAMS.json', rep); return rep

# ---------------------------------------------------------------- F4 receiver / majority workflow
def workflow(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'WORKFLOW.json'): return Q.load(out/'WORKFLOW.json')
    names = [n for n, e in banks.items() if e['arm'] in WORKFLOW_METHODS]; rows = []; t0 = time.monotonic(); n = 0; gp = bb.generation_prompt()
    for s in scenes:
        Q.budget(deadline, 300); a = s.target_actor; p = s.target_project; idx = [a, (a+1) % len(s.actors), (a+2) % len(s.actors)]; who = [s.actors[i] for i in idx]; P_ = s.projects[p]; reports_cache = {}
        src_b = prefix_bundle(bb, s, proc, 'single', 'original'); src = bb.compile(src_b['prefix'])
        for program in Q.WORKFLOW_PROGRAMS:
            b = prefix_bundle(bb, s, proc, program, 'original'); world = Q.final_world(s, program); gold_src = int(sum(s.values[i][p] for i in idx) >= 2); gold_tgt = int(sum(world.values[i][p] for i in idx) >= 2)
            texts = {'SOURCE': b['text'], 'NATURAL_FINAL_WORLD': Q.prefix_text(s, proc, 'original', 0, world=world)[0], 'TEXT_CORRECTION_SEQ': Q.prefix_text(s, proc, 'original', 0, corrections=b['edits'])[0]}
            conds = [('SOURCE', None), ('NATURAL_FINAL_WORLD', None), ('TEXT_CORRECTION_SEQ', None)]+[(nm, banks[nm]) for nm in names]
            for nm, entry in conds:
                text = texts.get(nm, b['text'])
                if entry is None: art = src if nm == 'SOURCE' else bb.compile(bb.prefix_ids(text))
                else: art = compile_entry(bb, b, entry)
                ckey = (nm, art['hash']); reused = ckey in reports_cache
                if reused: reports = reports_cache[ckey]
                else:
                    encs = [bb.encode(text, Q.suffix_text(RECEIVER_Q.format(who=w, project=P_))) for w in who]; lids = [bb.label_ids(e['full_text'], ('No', 'Yes')) for e in encs]; reports = {}
                    for i, w in enumerate(who):
                        res = bb.ask(art, encs[i]['suffix_ids'], lids[i], generate=True); sym, status = Q.parse_symbol(res['generation']['text'], ('No', 'Yes'))
                        reports[w] = dict(text=res['generation']['text'], symbol=sym, status=status, score_prediction=res['prediction'], truncated=res['generation']['truncated'])
                    reports_cache[ckey] = reports
                ok = all(r['symbol'] is not None for r in reports.values()); truth_world = world if nm != 'SOURCE' else s
                rep_ = {w: dict(r, gold=int(truth_world.values[idx[i]][p]), correct=(r['symbol'] == ('Yes' if truth_world.values[idx[i]][p] else 'No'))) for i, (w, r) in enumerate(reports.items())}
                if ok:
                    lines = [PHRASING.format(who=w, project=P_, verb=('supports' if rep_[w]['symbol'] == 'Yes' else 'opposes')) for w in who]
                    agg = (f'Three committee members were asked separately about the {P_} proposal.\n'+'\n'.join(lines)+f'\nIs the {P_} proposal supported by a majority of {who[0]}, {who[1]} and {who[2]}?\nAnswer with exactly one of No, Yes.'+Q.INSTRUCTION)
                    full = bb.tokenizer.apply_chat_template([{'role': 'user', 'content': agg}], tokenize=False, add_generation_prompt=True, **bb.template_kwargs); ids = bb.tokenizer.encode(full, add_special_tokens=False); al = bb.label_ids(full, ('No', 'Yes'))
                    gres = bb.ask(bb.compile(ids[:1]), ids[1:], al, generate=True); asym, astatus = Q.parse_symbol(gres['generation']['text'], ('No', 'Yes')); final = (1 if asym == 'Yes' else 0) if asym is not None else None; atext = gres['generation']['text']
                else: agg = None; asym = None; astatus = 'RECEIVER_FAILED'; final = None; atext = None
                gold = gold_tgt if nm != 'SOURCE' else gold_src
                rows.append(dict(scene_id=s.scene_id, condition=nm, program=program, method=(entry['arm'] if entry else nm), seed=(entry['seed'] if entry else None), holders=who, project=P_, receiver=rep_, receivers_all_correct=all(r['correct'] for r in rep_.values()),
                                 receiver_accuracy=float(np.mean([r['correct'] for r in rep_.values()])), aggregator_prompt=agg, aggregator_text=atext, aggregator_answer=asym, aggregator_status=astatus, chain_output=final, gold_source=gold_src, gold_target=gold_tgt,
                                 majority_changes=gold_src != gold_tgt, correct=(final == gold), correct_vs_target=(final == gold_tgt), cell=list(Q.cell(s)), artifact_hash=art['hash'], reports_reused_from_identical_artifact=reused, commands=len(b['edits']),
                                 energy=(art['realized']['energy'] if art.get('realized') else 0.0)))
        n += 1
        if n % 8 == 0: event(out, 'WORKFLOW_PROGRESS', scenes=n, seconds=time.monotonic()-t0)
    Q.write_rows(out/'workflow'/'rows.jsonl', rows)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], programs=list(Q.WORKFLOW_PROGRAMS), conditions=[c for c, _ in conds], rows=len(rows), seconds=time.monotonic()-t0, status='MEASURED', generation_prompt=gp, phrasing=PHRASING,
               note='three receiver calls per episode/program/condition generate actual reports from the compiled artifact; parsed symbols go to a separate unedited aggregator; invalid outputs fail (no gold substitution); byte-identical artifacts reuse their generated reports (recorded); independent WORKFLOW scenes')
    Q.dump(out/'WORKFLOW.json', rep); return rep

# ---------------------------------------------------------------- F5 controls
def controls(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'CONTROLS.json'): return Q.load(out/'CONTROLS.json')
    nm = 'INVARIANT_SET_s0'
    if nm not in banks: Q.dump(out/'CONTROLS.json', dict(at=Q.now(), status='NOT_MEASURED', reason='INVARIANT_SET seed 0 absent')); return Q.load(out/'CONTROLS.json')
    entry = banks[nm]; rows = {'LEARNED': [], 'RANDOM_MATCHED': [], 'WRONG_ADDRESS': []}; t0 = time.monotonic(); import torch; g = torch.Generator(device='cpu').manual_seed(2609)
    for s in scenes:
        Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', 'early'); src = bb.compile(b['prefix']); src_rows = score_shared(bb, b, src, None, 'SOURCE', [], sample_master=False); src_preds = {r['query_id']: r for r in src_rows}
        art = compile_entry(bb, b, entry); score_shared(bb, b, art, src_preds, 'LEARNED', rows['LEARNED'], extra=dict(energy=art['realized']['energy']))
        norms = torch.as_tensor(art['realized']['realized_norms'][0], dtype=torch.float32); d = torch.randn(len(art['_index']), bb.hidden, generator=g); d = d/d.norm(dim=-1, keepdim=True).clamp_min(1e-12)*norms.unsqueeze(-1)
        rart = bb.compile(b['prefix'], maps=[dict(mode='delta', delta=d.numpy())], index=art['_index'], site=entry['bank'].site); score_shared(bb, b, rart, src_preds, 'RANDOM_MATCHED', rows['RANDOM_MATCHED'], extra=dict(matched_norm=float(norms.mean()), energy=rart['realized']['energy']))
        wa = (s.target_actor+1) % len(s.actors); wv = 1-s.values[wa][s.target_project]; wtext, wspans = Q.prefix_text(s, proc, 'early', 0)
        wb = dict(b, addresses=[tuple(bb.clause_positions(wtext, wspans[(wa, s.target_project)]))], values=[wv]); wart = compile_entry(bb, wb, entry)
        wworld = Q.OLD.set_value(s, Q.P3.Edit(wa, s.target_project, wv)); wq = {q['query_id']: q for draw in (0, 1) for q in Q.OLD.questions(s, edited=wworld, challenge=True, draw=draw)}
        wrows = score_shared(bb, b, wart, src_preds, 'WRONG_ADDRESS', rows['WRONG_ADDRESS'], extra=dict(wrong_actor=s.actors[wa], wrong_new_value=wv, energy=wart['realized']['energy']))
        for r in wrows: r['wrong_world_gold'] = int(wq[r['query_id']]['gold']); r['wrong_world_changed'] = bool(wq[r['query_id']]['changed'])
    for c, rs in rows.items(): Q.write_rows(out/'controls'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], condition=nm, rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED'); Q.dump(out/'CONTROLS.json', rep); return rep

# ---------------------------------------------------------------- F6 physical witnesses
def witnesses(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'WITNESS_INDEX.json'): return Q.load(out/'WITNESS_INDEX.json')
    site = model['site']; wdir = out/'witness'; wdir.mkdir(parents=True, exist_ok=True); index = []
    for s in scenes:
        Q.budget(deadline, 300); conds = {'SOURCE': ('single', None), 'COUNTERFACTUAL': ('single', 'natural')}
        for nm in ('INVARIANT_SET_s0', 'INVARIANT_SET_s1', 'FREE_OVERWRITE_s0', 'CONTINUED_AFFINE_s0', 'AFFINE_CLAUSE_s0', 'FROZEN_V2_CLAUSE_s0'):
            if nm in banks:
                for program in (('single', 'repeat4', 'restore') if nm.startswith('INVARIANT') else ('single', 'repeat4')): conds[f'{nm}|{program}'] = (program, banks[nm])
        for key, (program, entry) in conds.items():
            b = prefix_bundle(bb, s, proc, program, 'early')
            if entry is None: pre = b['prefix']; art = bb.compile(pre, capture={site}, capture_next={site}); pl = None
            elif entry == 'natural': ctext, _ = Q.prefix_text(s, proc, 'early', 0, world=Q.final_world(s, 'single')); pre = bb.prefix_ids(ctext); art = bb.compile(pre, capture={site}, capture_next={site}); pl = None
            else: pre = b['prefix']; pl = plan(entry, b); art = T.compile_plan(bb, b, pl, capture={site}, capture_next={site})
            k0, v0 = bb.layer_kv(art['cache'], site); k1, v1 = bb.layer_kv(art['cache'], site+1); post = art['captured'][site].float().cpu().numpy(); nxt = art['next_input'][site].float().cpu().numpy(); path = wdir/f'{s.scene_id}_{key.replace("|", "_")}.npz'
            np.savez(path, prefix_ids=np.asarray(pre, '<i8'), post_block_residual=post, next_block_input=nxt, keys_site=k0, values_site=v0, keys_next=k1, values_next=v1, edited_positions=np.asarray(art.get('_index', []), '<i8'),
                     edited_states_bf16=(art['_after_states'] if pl else np.zeros((0, bb.hidden), np.float32)), workspace_states_fp32=(art['_workspace_states'] if pl else np.zeros((0, bb.hidden), np.float32)))
            index.append(dict(scene_id=s.scene_id, condition=key, program=program, path=str(path), sha256=Q.sha(path), bytes=path.stat().st_size, artifact_hash=art['hash'], prefix_tokens=len(pre), site=site, next_site=site+1, edited_tokens=len(art.get('_index', [])),
                              commands=(art['realized']['maps'] if pl else 0), energy=(art['realized']['energy'] if pl else 0.0), cast_error=(art['realized']['cast_error'] if pl else 0.0), next_input_equals_post_block=bool(np.array_equal(post, nxt)), checkpoint=(entry['checkpoint'] if pl else None)))
    rep = dict(at=Q.now(), site=site, scenes=[s.scene_id for s in scenes], files=index, note='post-block residual at the site (all prefix positions), the actual input to the next block, key/value tensors of the site and next layer, FP32 workspace states and their BF16-cast states at the edited positions')
    Q.dump(out/'WITNESS_INDEX.json', rep); return rep

# ---------------------------------------------------------------- natural panel (unqualified actor)
def natural_panel(bb, fin, qual, out, deadline, model, size):
    if done(out/'NATURAL_PANEL.json'): return Q.load(out/'NATURAL_PANEL.json')
    proc = qual['natural_panel_procedure']; n = Q.SIZES[model['key']]['SMALL' if size in ('SMALL', 'QUARTER') else 'FULL']['final']; scenes = Q.subset(fin, n, 'RSO3-FINAL'); rows = {}; t0 = time.monotonic()
    for s in scenes:
        for order in Q.ORDERS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order); src = bb.compile(b['prefix']); src_rows = score_shared(bb, b, src, None, 'SOURCE', rows.setdefault('SOURCE', []), sample_master=False); src_preds = {r['query_id']: r for r in src_rows}
            ctext, _ = Q.prefix_text(s, proc, order, 0, world=b['world']); natural_rows(bb, b, ctext, src_preds, 'COUNTERFACTUAL', rows.setdefault('COUNTERFACTUAL', []))
    for c, rs in rows.items(): Q.write_rows(out/'natural'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), procedure=proc, scenes=len(scenes), rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, status='MEASURED', note='native competence only; no learned intervention is evaluated or claimed'); Q.dump(out/'NATURAL_PANEL.json', rep); return rep

# ---------------------------------------------------------------- driver
def phase_f(bb, out, deadline, model, fit, cal, fin, wf, qdir, ddir, size, reserve_hours):
    freeze = Q.load(ddir/'FREEZE.json'); proc = freeze['procedure']; out.mkdir(parents=True, exist_ok=True)
    if not done(out/'FREEZE_VERIFIED.json'): Q.dump(out/'FREEZE_VERIFIED.json', dict(at=Q.now(), freeze_sha256=Q.sha(ddir/'FREEZE.json'), selection=freeze['selection'], procedure=proc))
    banks = load_banks(bb, freeze, ddir, model); event(out, 'BANKS', conditions=list(banks), strengths={n: e['strength'] for n, e in banks.items()}, workspaces={n: e['workspace'] for n, e in banks.items()})
    bench = Q.load(qdir/'BENCHMARK.json')['seconds']; scope = choose_scope(bb, out, deadline, model, size, reserve_hours, banks, bench); sizes = scope['sizes']; event(out, 'SCOPE', scope=scope['scope'], sizes=sizes, predicted=scope['predicted_seconds'][scope['scope']], remaining=scope['remaining_seconds'])
    final_scenes = Q.subset(fin, sizes['final'], 'RSO3-FINAL'); program_scenes = Q.subset(final_scenes, sizes['program'], 'RSO3-PROGRAM'); wf_scenes = Q.subset(wf, sizes['workflow'], 'RSO3-WORKFLOW')
    control = Q.subset(final_scenes, min(Q.PANELS['control'], len(final_scenes)), 'RSO3-CONTROL'); witness = sorted(final_scenes, key=lambda s: Q.hash_rank('RSO3-WITNESS', s.scene_id))[:Q.PANELS['witness']]
    gen_scenes = {s.scene_id for s in Q.subset(final_scenes, min(Q.PANELS['generation'], len(final_scenes)), 'RSO3-GENERATION')}
    if not done(out/'SUBSETS.json'): Q.dump(out/'SUBSETS.json', dict(at=Q.now(), final=[s.scene_id for s in final_scenes], program=[s.scene_id for s in program_scenes], workflow=[s.scene_id for s in wf_scenes], control=[s.scene_id for s in control], witness=[s.scene_id for s in witness], generation=sorted(gen_scenes)))
    sl = seal(bb, final_scenes, proc, banks, out, deadline, model); event(out, 'SEALED', artifacts=sl['artifacts'], sha256=sl['seal_sha256'])
    ev = evaluate_final(bb, final_scenes, proc, banks, out, deadline, model, gen_scenes); event(out, 'FINAL_EVAL', rows=ev['rows'], mismatches=len(ev['seal_mismatches']))
    for stage, fn, arg in (('WITNESS', witnesses, witness), ('PROGRAMS', programs, program_scenes), ('WORKFLOW', workflow, wf_scenes), ('CONTROLS', controls, control)):
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
