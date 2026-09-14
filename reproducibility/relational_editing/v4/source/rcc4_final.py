"""RCC4 phase F: fresh frozen evaluation against sealed, question-blind compiled contexts (FREEZE verified first).

Conditions   the eight selected coverage editors (4 conditions x 2 seeds), the frozen V3 INVARIANT_SET and FREE_OVERWRITE (both seeds,
             re-evaluated on the same fresh population), CANONICAL_CURRENT (the CAL-selected current editor replayed once per address in a
             fixed address order from the preserved source: source backup + command history, no gold/original-value shortcut), the natural
             SOURCE / COUNTERFACTUAL (native final-world rewrite) / TEXT_CORRECTION references, and (Gemma) a 16-scene frozen V2 affine
             anchor panel for numerical continuity only.
F0 SCOPE     contracted FULL sizes (Gemma 128/64/128, Qwen 64/32/64) fixed by the contract; the predicted time is recorded, never used to shrink.
F1 SEAL      every shared single-edit artifact compiled from prefix + explicit command program only; hashes recorded BEFORE the first FINAL question.
F2 SINGLES   both prefix orders, both answer-mapping draws, the original 24 questions; generation panel.
F3 PROGRAMS  the V3 program suite (single/noop/repeat2/4/8/restore/restore_after4/overwrite/AB/BA/ABC/CBA/AB_cross) inside ONE compilation
             each on the 64/32-scene early-order subset, scored against the true final world with coverage questions (all-24 and
             all-program-questions kept distinct); byte-identical compiled tensors share a measurement across programs AND conditions
             (recorded with both provenances; repeated conditions are not extra observations).
F4 WORKFLOW  three receiver reports -> unedited majority aggregator for single / duplicate / restoration / AB on the independent WORKFLOW
             scenes for the CAL-frozen workflow candidate, the frozen V3 INVARIANT_SET, CANONICAL_CURRENT and the natural references.
F6 WITNESS   post-block residual, next-block input, key/value tensors, FP32 workspace and BF16-cast states for four fixed scenes.
"""
from __future__ import annotations
import random
import time
from pathlib import Path
import numpy as np
import rcc4_common as C
import rso3_common as Q
import rso3_train as T
import rso3_final as F
from rso3_editor import SetterBank, AffineBank
from rso3_unit import event, done
from rcc4_train import load_v3_bank

score_shared = F.score_shared; natural_rows = F.natural_rows; method_extra = F.method_extra; prefix_bundle = F.prefix_bundle

# ---------------------------------------------------------------- banks
def load_banks(bb, freeze, ddir, model, store):
    banks = {}
    for cond, sel in freeze['selection'].items():
        for seed in C.SEEDS:
            s = sel[str(seed)]; ck = Path(s['path'])
            if not ck.is_dir(): ck = ddir/'arms'/cond/f's{seed}'/ck.name
            bank = SetterBank.load(bb, ck)
            banks[f'{cond}_s{seed}'] = dict(bank=bank, arm=cond, seed=seed, strength=1.0, workspace='fp32', footprint='clause', canonical=False, checkpoint=str(ck), meta_sha256=Q.sha(ck/'META.json'), learned=True, setter=True, selected_checkpoint=s['checkpoint'], feasible=s['feasible'])
    for name, arm in C.FROZEN_V3.items():
        for seed in C.SEEDS:
            bank = load_v3_bank(bb, model['key'], arm, seed, store, model['site']); bank.freeze_basis()
            banks[f'{name}_s{seed}'] = dict(bank=bank, arm=name, seed=seed, strength=1.0, workspace='fp32', footprint='clause', canonical=False, checkpoint=bank.init['dir'], meta_sha256=bank.init['sha256']['meta'], learned=False, setter=True, frozen=True)
    for seed in C.SEEDS:
        cc = freeze['canonical_current'].get(str(seed)) or freeze['canonical_current'].get(seed)
        if cc:
            base = banks[f'{cc["condition"]}_s{seed}']
            banks[f'{C.CANONICAL}_s{seed}'] = dict(base, arm=C.CANONICAL, canonical=True, learned=False, replays=cc['condition'], privileges='original source backup + full declared command history; last requested value per address, sorted addresses; no gold, original-value shortcut or evaluator metadata')
    if model['key'] == 'gemma' and bb.hidden == 3584:
        bank = AffineBank.from_v2(bb, 'SHARED_CLAUSE', 0); spec = Q.FROZEN['FROZEN_V2_CLAUSE']
        banks[f'{C.ANCHOR_AFFINE}_s0'] = dict(bank=bank, arm=C.ANCHOR_AFFINE, seed=0, strength=spec['strength'], workspace=spec['workspace'], footprint=spec['footprint'], canonical=False, checkpoint=str(Q.v2_factor_dir('SHARED_CLAUSE', 0)), meta_sha256=bank.init['meta_sha256'], learned=False, setter=False, anchor_only=True)
    return banks

def plan(entry, b, record=False): return T.plan_for(entry['bank'], b, entry['strength'], record=record, workspace=entry['workspace'], footprint=entry['footprint'], canonical=entry['canonical'])
def compile_entry(bb, b, entry, capture=(), capture_next=()): return T.compile_plan(bb, b, plan(entry, b), capture=capture, capture_next=capture_next)
def single_names(banks): return [n for n, e in banks.items() if not e['canonical'] and not e.get('anchor_only')]
def program_names(banks): return [n for n, e in banks.items() if not e.get('anchor_only')]
def workflow_names(banks, freeze):
    cand = (freeze.get('workflow_candidate') or {}).get('condition'); arms = {cand, 'FROZEN_V3_INVARIANT_SET', C.CANONICAL}
    return [n for n, e in banks.items() if e['arm'] in arms]

# ---------------------------------------------------------------- F0 scope (contracted sizes; predicted time recorded only)
def scope(bb, out, deadline, model, banks, bench):
    if done(out/'SCOPE.json'): return Q.load(out/'SCOPE.json')
    sizes = C.SIZES[model['key']]['FULL']; n_single = len(single_names(banks))+3; n_prog = len(program_names(banks))+3; n_wf = 0
    per_single = bench['compile_shared']+2*bench['ask_batch12']; per_prog = bench['compile_repeat8']*0.6+bench['compile_shared']*0.4+3*bench['ask_batch12']; per_wf = bench['compile_shared']+4*bench['ask_generate32']+bench['hash']
    pred = dict(seal=sizes['final']*2*(n_single-3)*bench['compile_shared']*1.1, final=sizes['final']*2*n_single*per_single*1.25, programs=sizes['program']*len(C.PROGRAMS)*n_prog*per_prog*1.2*0.8, workflow=sizes['workflow']*len(C.WORKFLOW_PROGRAMS)*9*per_wf*1.2*0.7, panels=600)
    pred['total'] = sum(pred.values()); remaining = Q.budget(deadline, 0)
    rep = dict(at=Q.now(), scope='FULL', sizes=sizes, predicted_seconds=pred, remaining_seconds=remaining, fits=pred['total'] <= remaining, rule='contracted FULL sizes are fixed (never shrunk); stages run in order seal -> singles -> programs -> witness -> workflow -> anchor and record NOT_MEASURED/PARTIAL when the effective deadline is reached')
    Q.dump(out/'SCOPE.json', rep); return rep

# ---------------------------------------------------------------- F1 seal
def seal(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'SEAL.json'): return Q.load(out/'SEAL.json')
    rows = []; t0 = time.monotonic(); names = single_names(banks)
    for s in scenes:
        for order in C.ORDERS:
            Q.budget(deadline, 120); b = prefix_bundle(bb, s, proc, 'single', order); world = C.final_world(s, 'single')
            src = bb.compile(b['prefix']); ctext, _ = C.prefix_text(s, proc, order, 0, world=world); cf = bb.compile(bb.prefix_ids(ctext)); ttext, _ = C.prefix_text(s, proc, order, 0, corrections=b['edits']); tc = bb.compile(bb.prefix_ids(ttext))
            row = dict(scene_id=s.scene_id, order=order, program='single', prefix_tokens=len(b['prefix']), clause_tokens=len(b['clause']), clause_start=b['clause'][0], request_key=b['request_key'],
                       artifacts=dict(SOURCE=dict(hash=src['hash'], bytes=src['bytes']), COUNTERFACTUAL=dict(hash=cf['hash'], bytes=cf['bytes'], prefix_tokens=cf['prefix_len']), TEXT_CORRECTION=dict(hash=tc['hash'], bytes=tc['bytes'], prefix_tokens=tc['prefix_len'])))
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry); r = art['realized']
                row['artifacts'][nm] = dict(hash=art['hash'], bytes=art['bytes'], edited_tokens=r['positions'], energy=r['energy'], frobenius=r['frobenius'], cast_error=r['cast_error'], method_digest=entry['meta_sha256'], strength=entry['strength'], workspace=entry['workspace'])
            rows.append(row)
    Q.write_rows(out/'SEAL.jsonl', rows)
    rep = dict(at=Q.now(), artifacts=len(rows), scenes=len(scenes), conditions=['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+names, canonical_on_single=[n for n in banks if banks[n]['canonical']], anchor_panel_only=[n for n in banks if banks[n].get('anchor_only')], seal_sha256=Q.sha(out/'SEAL.jsonl'), seconds=time.monotonic()-t0,
               compiler_inputs='prefix token ids, explicit clause spans and requested values of the command program, method weights; no question, label mapping, gold, affected flag or target world existed before this file', questions_generated_before_seal=False)
    Q.dump(out/'SEAL.json', rep); return rep

def _copy_rows(b, prev_rows, name, rows, extra):
    prev = {r['query_id']: r for r in prev_rows}; out = []
    for qq in b['queries']:
        rec = dict(qq['rec']); pr = prev.get(rec['query_id'])
        if pr is None: continue
        rec.update({k: pr[k] for k in ('prediction', 'logps', 'answer_mass', 'argmax_in_labels', 'tie', 'source_prediction', 'source_logps', 'artifact_hash', 'master_verified') if k in pr}); rec['gold_logp'] = pr['logps'][rec['gold']]; rec['condition'] = name; rec.update(extra); rows.append(rec); out.append(rec)
    return out

# ---------------------------------------------------------------- F2 singles (byte-identical artifacts share a measurement across conditions; recorded)
def evaluate_final(bb, scenes, proc, banks, out, deadline, model, gen_scenes):
    if done(out/'EVAL_DONE.json'): return Q.load(out/'EVAL_DONE.json')
    seal_by = {(r['scene_id'], r['order']): r for r in Q.read_rows(out/'SEAL.jsonl')}; names = single_names(banks); conds = ['SOURCE', 'COUNTERFACTUAL', 'TEXT_CORRECTION']+names; writers = {c: [] for c in conds}; mismatches = []; shared = []; t0 = time.monotonic(); first_q = None; n_done = 0
    order_ = list(scenes); random.Random('rcc4-final-order').shuffle(order_)
    for s in order_:
        for order in C.ORDERS:
            Q.budget(deadline, 300)
            if first_q is None: first_q = Q.now()
            b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order); sr = seal_by[(s.scene_id, order)]; gen = s.scene_id in gen_scenes and order == 'early'
            src = bb.compile(b['prefix'])
            if src['hash'] != sr['artifacts']['SOURCE']['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition='SOURCE'))
            src_rows = score_shared(bb, b, src, None, 'SOURCE', writers['SOURCE'], generate=gen); src_preds = {r['query_id']: r for r in src_rows}
            ctext, _ = C.prefix_text(s, proc, order, 0, world=b['world']); cf = natural_rows(bb, b, ctext, src_preds, 'COUNTERFACTUAL', writers['COUNTERFACTUAL'], generate=gen)
            ttext, _ = C.prefix_text(s, proc, order, 0, corrections=b['edits']); tc = natural_rows(bb, b, ttext, src_preds, 'TEXT_CORRECTION', writers['TEXT_CORRECTION'], generate=gen)
            for nm, art in (('COUNTERFACTUAL', cf), ('TEXT_CORRECTION', tc)):
                if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition=nm))
            measured = {}
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry)
                if art['hash'] != sr['artifacts'][nm]['hash']: mismatches.append(dict(scene_id=s.scene_id, order=order, condition=nm))
                ex = dict(method_extra(entry, art), cache_bytes=art['bytes'])
                if art['hash'] in measured:
                    prev_nm, prev_rows = measured[art['hash']]; ex['shared_measurement_from_condition'] = prev_nm; shared.append(dict(scene_id=s.scene_id, order=order, condition=nm, identical_to=prev_nm, hash=art['hash']))
                    _copy_rows(b, prev_rows, nm, writers[nm], ex)
                else: measured[art['hash']] = (nm, score_shared(bb, b, art, src_preds, nm, writers[nm], extra=ex, generate=gen and entry['seed'] == 0))
        n_done += 1
        if n_done % 8 == 0: event(out, 'FINAL_PROGRESS', scenes=n_done, seconds=time.monotonic()-t0, mismatches=len(mismatches), shared=len(shared))
    for c, rows in writers.items(): Q.write_rows(out/'final'/f'{c}.jsonl', rows)
    Q.write_rows(out/'final'/'SHARED_MEASUREMENTS.jsonl', shared)
    rep = dict(at=Q.now(), first_question_generated_at=first_q, seal_at=Q.load(out/'SEAL.json')['at'], scenes=len(scenes), conditions=conds, rows={c: len(r) for c, r in writers.items()}, seal_mismatches=mismatches, recompile_deterministic=not mismatches, shared_measurements=len(shared), seconds=time.monotonic()-t0,
               generation_scenes=sorted(gen_scenes), master_verified_samples={c: int(sum(1 for r in rows if r.get('master_verified'))) for c, rows in writers.items()}, master_failures={c: int(sum(1 for r in rows if r.get('master_verified') is False)) for c, rows in writers.items()})
    Q.dump(out/'EVAL_DONE.json', rep); return rep

# ---------------------------------------------------------------- F3 edit programs
def programs(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'PROGRAMS.json'): return Q.load(out/'PROGRAMS.json')
    rows = {}; ident = []; shared = []; t0 = time.monotonic(); n = 0; order = 'early'; names = program_names(banks)
    for s in scenes:
        Q.budget(deadline, 300); src = None; measured = {}; single_art = {}; nat_cache = {}
        for program in C.PROGRAMS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, program, order)
            if src is None: src = bb.compile(b['prefix'])
            src_rows = score_shared(bb, b, src, None, 'SOURCE', rows.setdefault('SOURCE', []), extra=dict(program=program), sample_master=False); src_preds = {r['query_id']: r for r in src_rows}
            ntext, _ = C.prefix_text(s, proc, order, 0, world=b['world']); nat = nat_cache.get(ntext)
            nat_cache[ntext] = natural_rows(bb, b, ntext, src_preds, 'NATURAL_FINAL_WORLD', rows.setdefault('NATURAL_FINAL_WORLD', []), extra=dict(program=program, artifact_reused=nat is not None), art=nat)
            ttext, _ = C.prefix_text(s, proc, order, 0, corrections=b['edits']); natural_rows(bb, b, ttext, src_preds, 'TEXT_CORRECTION_SEQ', rows.setdefault('TEXT_CORRECTION_SEQ', []), extra=dict(program=program, correction_lines=len(b['edits'])))
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry); ex = method_extra(entry, art); ex.update(program=program, index_len=len(art['_index']))
                if art['hash'] in measured:
                    prev_nm, prev_program, prev_rows = measured[art['hash']]; ex['shared_measurement_from'] = prev_program; ex['shared_measurement_from_condition'] = prev_nm
                    shared.append(dict(scene_id=s.scene_id, condition=nm, program=program, identical_to=prev_program, identical_to_condition=prev_nm, hash=art['hash']))
                    copied = _copy_rows(b, prev_rows, nm, rows.setdefault(nm, []), ex); missing = [qq for qq in b['queries'] if qq['rec']['query_id'] not in {r['query_id'] for r in prev_rows}]
                    if missing: score_shared(bb, dict(b, queries=missing), art, src_preds, nm, rows.setdefault(nm, []), extra=ex, sample_master=False)
                else: measured[art['hash']] = (nm, program, score_shared(bb, b, art, src_preds, nm, rows.setdefault(nm, []), extra=ex, sample_master=(program == 'single')))
                if program == 'single': single_art[nm] = art
                base = single_art.get(nm)
                if base is not None and program != 'single' and base['_index'] == art['_index']:
                    ident.append(dict(scene_id=s.scene_id, condition=nm, program=program, workspace_max_abs_vs_single=float(np.max(np.abs(art['_workspace_states']-base['_workspace_states']))), bf16_max_abs_vs_single=float(np.max(np.abs(art['_after_states']-base['_after_states']))), hash_equal_single=art['hash'] == base['hash']))
                elif base is not None and program != 'single': ident.append(dict(scene_id=s.scene_id, condition=nm, program=program, workspace_max_abs_vs_single=None, bf16_max_abs_vs_single=None, hash_equal_single=art['hash'] == base['hash'], note='different edited index'))
        n += 1
        if n % 4 == 0: event(out, 'PROGRAMS_PROGRESS', scenes=n, seconds=time.monotonic()-t0, shared=len(shared))
    for c, rs in rows.items(): Q.write_rows(out/'programs'/f'{c}.jsonl', rs)
    Q.write_rows(out/'programs'/'IDENTITY.jsonl', ident); Q.write_rows(out/'programs'/'SHARED_MEASUREMENTS.jsonl', shared)
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], order=order, programs=list(C.PROGRAMS), conditions=names, rows={c: len(r) for c, r in rows.items()}, shared_measurements=len(shared), seconds=time.monotonic()-t0, status='MEASURED',
               note='every declared command executes inside ONE prefix compilation (FP32 workspace; CANONICAL_CURRENT normalizes then replays the selected current editor from the preserved source); gold from the true final world; coverage questions flagged additional; byte-identical compiled tensors share a measurement across programs and conditions (both provenances recorded); no parameter was trained on any program')
    Q.dump(out/'PROGRAMS.json', rep); return rep

# ---------------------------------------------------------------- F4 receiver / majority workflow (incremental rows; PARTIAL on deadline)
def workflow(bb, scenes, proc, banks, out, deadline, model, freeze):
    if done(out/'WORKFLOW.json'): return Q.load(out/'WORKFLOW.json')
    names = workflow_names(banks, freeze); path = out/'workflow'/'rows.jsonl'; path.parent.mkdir(parents=True, exist_ok=True); finished = {r['scene_id'] for r in (Q.read_rows(path) if path.is_file() else [])}; t0 = time.monotonic(); n = 0; gp = bb.generation_prompt(); status = 'MEASURED'; conds = []
    try:
        for s in scenes:
            if s.scene_id in finished: n += 1; continue
            Q.budget(deadline, 300); a = s.target_actor; p = s.target_project; idx = [a, (a+1) % len(s.actors), (a+2) % len(s.actors)]; who = [s.actors[i] for i in idx]; P_ = s.projects[p]; reports_cache = {}; scene_rows = []
            src_b = prefix_bundle(bb, s, proc, 'single', 'original'); src = bb.compile(src_b['prefix'])
            for program in C.WORKFLOW_PROGRAMS:
                b = prefix_bundle(bb, s, proc, program, 'original'); world = C.final_world(s, program); gold_src = int(sum(s.values[i][p] for i in idx) >= 2); gold_tgt = int(sum(world.values[i][p] for i in idx) >= 2)
                texts = {'SOURCE': b['text'], 'NATURAL_FINAL_WORLD': C.prefix_text(s, proc, 'original', 0, world=world)[0], 'TEXT_CORRECTION_SEQ': C.prefix_text(s, proc, 'original', 0, corrections=b['edits'])[0]}
                conds = [('SOURCE', None), ('NATURAL_FINAL_WORLD', None), ('TEXT_CORRECTION_SEQ', None)]+[(nm, banks[nm]) for nm in names]
                for nm, entry in conds:
                    text = texts.get(nm, b['text'])
                    if entry is None: art = src if nm == 'SOURCE' else bb.compile(bb.prefix_ids(text))
                    else: art = compile_entry(bb, b, entry)
                    reused = art['hash'] in reports_cache
                    if reused: reports, reused_from = reports_cache[art['hash']]
                    else:
                        encs = [bb.encode(text, C.suffix_text(F.RECEIVER_Q.format(who=w, project=P_))) for w in who]; lids = [bb.label_ids(e['full_text'], ('No', 'Yes')) for e in encs]; reports = {}
                        for i, w in enumerate(who):
                            res = bb.ask(art, encs[i]['suffix_ids'], lids[i], generate=True); sym, st = C.parse_symbol(res['generation']['text'], ('No', 'Yes'))
                            reports[w] = dict(text=res['generation']['text'], symbol=sym, status=st, score_prediction=res['prediction'], truncated=res['generation']['truncated'])
                        reports_cache[art['hash']] = (reports, (nm, program)); reused_from = None
                    ok = all(r['symbol'] is not None for r in reports.values()); truth_world = world if nm != 'SOURCE' else s
                    rep_ = {w: dict(r, gold=int(truth_world.values[idx[i]][p]), correct=(r['symbol'] == ('Yes' if truth_world.values[idx[i]][p] else 'No'))) for i, (w, r) in enumerate(reports.items())}
                    if ok:
                        lines = [F.PHRASING.format(who=w, project=P_, verb=('supports' if rep_[w]['symbol'] == 'Yes' else 'opposes')) for w in who]
                        agg = (f'Three committee members were asked separately about the {P_} proposal.\n'+'\n'.join(lines)+f'\nIs the {P_} proposal supported by a majority of {who[0]}, {who[1]} and {who[2]}?\nAnswer with exactly one of No, Yes.'+C.INSTRUCTION)
                        full = bb.tokenizer.apply_chat_template([{'role': 'user', 'content': agg}], tokenize=False, add_generation_prompt=True, **bb.template_kwargs); ids = bb.tokenizer.encode(full, add_special_tokens=False); al = bb.label_ids(full, ('No', 'Yes'))
                        gres = bb.ask(bb.compile(ids[:1]), ids[1:], al, generate=True); asym, astatus = C.parse_symbol(gres['generation']['text'], ('No', 'Yes')); final = (1 if asym == 'Yes' else 0) if asym is not None else None; atext = gres['generation']['text']
                    else: agg = None; asym = None; astatus = 'RECEIVER_FAILED'; final = None; atext = None
                    gold = gold_tgt if nm != 'SOURCE' else gold_src
                    scene_rows.append(dict(scene_id=s.scene_id, condition=nm, program=program, method=(entry['arm'] if entry else nm), seed=(entry['seed'] if entry else None), holders=who, project=P_, receiver=rep_, receivers_all_correct=all(r['correct'] for r in rep_.values()),
                                           receiver_accuracy=float(np.mean([r['correct'] for r in rep_.values()])), aggregator_prompt=agg, aggregator_text=atext, aggregator_answer=asym, aggregator_status=astatus, chain_output=final, gold_source=gold_src, gold_target=gold_tgt,
                                           majority_changes=gold_src != gold_tgt, correct=(final == gold), correct_vs_target=(final == gold_tgt), cell=list(C.cell(s)), artifact_hash=art['hash'], reports_reused_from_identical_artifact=reused, reports_reused_from=reused_from, commands=len(b['edits']),
                                           energy=(art['realized']['energy'] if art.get('realized') else 0.0)))
            with path.open('a', encoding='utf8', newline='\n') as f:
                for r in scene_rows: f.write(Q.canonical(r)+'\n')
            n += 1
            if n % 8 == 0: event(out, 'WORKFLOW_PROGRESS', scenes=n, seconds=time.monotonic()-t0)
    except TimeoutError:
        status = 'PARTIAL'; event(out, 'WORKFLOW_PARTIAL', scenes=n, note='effective deadline reached; rows of complete scenes retained; incomplete = NOT_MEASURED')
    all_rows = Q.read_rows(path) if path.is_file() else []
    rep = dict(at=Q.now(), scenes=[s.scene_id for s in scenes], scenes_measured=sorted({r['scene_id'] for r in all_rows}), programs=list(C.WORKFLOW_PROGRAMS), conditions=[c for c, _ in conds], rows=len(all_rows), seconds=time.monotonic()-t0, status=status, generation_prompt=gp, phrasing=F.PHRASING,
               note='three receiver calls per episode/program/condition generate actual reports from the compiled artifact; parsed symbols go to a separate unedited aggregator; invalid outputs fail (no gold substitution); byte-identical artifacts reuse their generated reports across conditions (recorded); independent WORKFLOW scenes; candidate frozen by CAL before any FINAL/WORKFLOW outcome')
    Q.dump(out/'WORKFLOW.json', rep)
    if status == 'PARTIAL': raise TimeoutError('workflow partial')
    return rep

# ---------------------------------------------------------------- F6 witnesses
def witnesses(bb, scenes, proc, banks, out, deadline, model, freeze):
    if done(out/'WITNESS_INDEX.json'): return Q.load(out/'WITNESS_INDEX.json')
    site = model['site']; wdir = out/'witness'; wdir.mkdir(parents=True, exist_ok=True); index = []; cand = (freeze.get('workflow_candidate') or {}).get('condition')
    for s in scenes:
        Q.budget(deadline, 300); conds = {'SOURCE': ('single', None), 'COUNTERFACTUAL': ('single', 'natural')}
        plan_ = [(f'{cand}_s0', ('single', 'repeat4', 'restore', 'AB')), (f'{cand}_s1', ('single',)), ('FROZEN_V3_INVARIANT_SET_s0', ('single', 'repeat4'))] if cand else [('FROZEN_V3_INVARIANT_SET_s0', ('single', 'repeat4'))]
        for nm, progs in plan_:
            if nm in banks:
                for program in progs: conds[f'{nm}|{program}'] = (program, banks[nm])
        for key, (program, entry) in conds.items():
            b = prefix_bundle(bb, s, proc, program, 'early')
            if entry is None: pre = b['prefix']; art = bb.compile(pre, capture={site}, capture_next={site}); pl = None
            elif entry == 'natural': ctext, _ = C.prefix_text(s, proc, 'early', 0, world=C.final_world(s, 'single')); pre = bb.prefix_ids(ctext); art = bb.compile(pre, capture={site}, capture_next={site}); pl = None
            else: pre = b['prefix']; pl = plan(entry, b); art = T.compile_plan(bb, b, pl, capture={site}, capture_next={site})
            k0, v0 = bb.layer_kv(art['cache'], site); k1, v1 = bb.layer_kv(art['cache'], site+1); post = art['captured'][site].float().cpu().numpy(); nxt = art['next_input'][site].float().cpu().numpy(); path = wdir/f'{s.scene_id}_{key.replace("|", "_")}.npz'
            np.savez(path, prefix_ids=np.asarray(pre, '<i8'), post_block_residual=post, next_block_input=nxt, keys_site=k0, values_site=v0, keys_next=k1, values_next=v1, edited_positions=np.asarray(art.get('_index', []), '<i8'),
                     edited_states_bf16=(art['_after_states'] if pl else np.zeros((0, bb.hidden), np.float32)), workspace_states_fp32=(art['_workspace_states'] if pl else np.zeros((0, bb.hidden), np.float32)))
            index.append(dict(scene_id=s.scene_id, condition=key, program=program, path=str(path), sha256=Q.sha(path), bytes=path.stat().st_size, artifact_hash=art['hash'], prefix_tokens=len(pre), site=site, next_site=site+1, edited_tokens=len(art.get('_index', [])),
                              commands=(art['realized']['maps'] if pl else 0), energy=(art['realized']['energy'] if pl else 0.0), cast_error=(art['realized']['cast_error'] if pl else 0.0), next_input_equals_post_block=bool(np.array_equal(post, nxt)), checkpoint=(entry['checkpoint'] if pl else None)))
    rep = dict(at=Q.now(), site=site, scenes=[s.scene_id for s in scenes], files=index, note='post-block residual at the site (all prefix positions), the actual input to the next block, key/value tensors of the site and next layer, FP32 workspace states and their BF16-cast states at the edited positions')
    Q.dump(out/'WITNESS_INDEX.json', rep); return rep

# ---------------------------------------------------------------- frozen-affine anchor panel (Gemma; numerical continuity only)
def anchor_panel(bb, scenes, proc, banks, out, deadline, model):
    if done(out/'ANCHOR_PANEL.json'): return Q.load(out/'ANCHOR_PANEL.json')
    names = [n for n, e in banks.items() if e.get('anchor_only')]
    if not names: Q.dump(out/'ANCHOR_PANEL.json', dict(at=Q.now(), status='NOT_APPLICABLE', actor=model['key'])); return Q.load(out/'ANCHOR_PANEL.json')
    rows = {}; t0 = time.monotonic()
    for s in scenes:
        for order in C.ORDERS:
            Q.budget(deadline, 300); b = T.scene_bundle(bb, s, 'FINAL', proc, 'single', order); src = bb.compile(b['prefix']); src_rows = score_shared(bb, b, src, None, 'SOURCE', rows.setdefault('SOURCE', []), sample_master=False); src_preds = {r['query_id']: r for r in src_rows}
            for nm in names:
                entry = banks[nm]; art = compile_entry(bb, b, entry); score_shared(bb, b, art, src_preds, nm, rows.setdefault(nm, []), extra=dict(method_extra(entry, art), cache_bytes=art['bytes']), sample_master=False)
    for c, rs in rows.items(): Q.write_rows(out/'anchor'/f'{c}.jsonl', rs)
    rep = dict(at=Q.now(), status='MEASURED', scenes=[s.scene_id for s in scenes], conditions=names, rows={c: len(r) for c, r in rows.items()}, seconds=time.monotonic()-t0, note='small frozen V2 affine anchor panel for numerical continuity with V3; not an efficacy claim'); Q.dump(out/'ANCHOR_PANEL.json', rep); return rep

# ---------------------------------------------------------------- driver
def phase_f(bb, out, deadline, model, fin, wf, qdir, ddir, store, effective_deadline=None):
    freeze = Q.load(ddir/'FREEZE.json'); proc = freeze['procedure']; out.mkdir(parents=True, exist_ok=True); effective_deadline = effective_deadline or deadline
    if not done(out/'FREEZE_VERIFIED.json'): Q.dump(out/'FREEZE_VERIFIED.json', dict(at=Q.now(), freeze_sha256=Q.sha(ddir/'FREEZE.json'), selection={c: {s: v[s]['checkpoint'] for s in v} for c, v in freeze['selection'].items()}, canonical={s: v['condition'] for s, v in freeze['canonical_current'].items()}, workflow=(freeze.get('workflow_candidate') or {}).get('condition'), procedure=proc, effective_deadline=effective_deadline))
    banks = load_banks(bb, freeze, ddir, model, store); event(out, 'BANKS', conditions=list(banks), strengths={n: e['strength'] for n, e in banks.items()}, canonical={n: e.get('replays') for n, e in banks.items() if e['canonical']}, workflow=workflow_names(banks, freeze))
    bench = Q.load(qdir/'BENCHMARK.json')['seconds']; sc = scope(bb, out, effective_deadline, model, banks, bench); sizes = sc['sizes']; event(out, 'SCOPE', scope='FULL', sizes=sizes, predicted=sc['predicted_seconds'], remaining=sc['remaining_seconds'], fits=sc['fits'])
    final_scenes = C.subset(fin, sizes['final'], 'RCC4-FINAL'); program_scenes = C.subset(final_scenes, sizes['program'], 'RCC4-PROGRAM'); wf_scenes = C.subset(wf, sizes['workflow'], 'RCC4-WORKFLOW')
    witness = sorted(final_scenes, key=lambda s: C.hash_rank('RCC4-WITNESS', s.scene_id))[:C.PANELS['witness']]; anchor = C.subset(final_scenes, min(C.PANELS['anchor'], len(final_scenes)), 'RCC4-ANCHOR')
    gen_scenes = {s.scene_id for s in C.subset(final_scenes, min(C.PANELS['generation'], len(final_scenes)), 'RCC4-GENERATION')}
    if not done(out/'SUBSETS.json'): Q.dump(out/'SUBSETS.json', dict(at=Q.now(), final=[s.scene_id for s in final_scenes], program=[s.scene_id for s in program_scenes], workflow=[s.scene_id for s in wf_scenes], witness=[s.scene_id for s in witness], anchor=[s.scene_id for s in anchor], generation=sorted(gen_scenes), rule='fixed hash-ordered subsets by scene id/metadata only, chosen before any outcome'))
    sl = seal(bb, final_scenes, proc, banks, out, effective_deadline, model); event(out, 'SEALED', artifacts=sl['artifacts'], sha256=sl['seal_sha256'])
    ev = evaluate_final(bb, final_scenes, proc, banks, out, effective_deadline, model, gen_scenes); event(out, 'FINAL_EVAL', rows=ev['rows'], mismatches=len(ev['seal_mismatches']), shared=ev['shared_measurements'])
    stages = (('PROGRAMS', lambda: programs(bb, program_scenes, proc, banks, out, effective_deadline, model), 'PROGRAMS.json'), ('WITNESS', lambda: witnesses(bb, witness, proc, banks, out, effective_deadline, model, freeze), 'WITNESS_INDEX.json'),
              ('WORKFLOW', lambda: workflow(bb, wf_scenes, proc, banks, out, effective_deadline, model, freeze), 'WORKFLOW.json'), ('ANCHOR', lambda: anchor_panel(bb, anchor, proc, banks, out, effective_deadline, model), 'ANCHOR_PANEL.json'))
    import datetime as dt
    def remaining(d): return (dt.datetime.fromisoformat(d)-dt.datetime.now(dt.timezone.utc)).total_seconds()
    for stage, fn, marker in stages:
        try:
            if remaining(effective_deadline) < 600:
                if not done(out/marker): Q.dump(out/marker, dict(at=Q.now(), status='NOT_MEASURED', reason='insufficient remaining time before the effective deadline'))
                event(out, stage, status='NOT_MEASURED'); continue
            r = fn(); event(out, stage, status=(r or {}).get('status', 'MEASURED'))
        except TimeoutError:
            if not done(out/marker): Q.dump(out/marker, dict(at=Q.now(), status='NOT_MEASURED', reason='effective deadline reached'))
            event(out, stage, status=Q.load(out/marker).get('status', 'NOT_MEASURED'))
            if remaining(deadline) < 600: raise
    return dict(status='FINAL_COMPLETE', scope='FULL', stages={m: Q.load(out/m).get('status', 'MEASURED') for _, _, m in stages if done(out/m)})
