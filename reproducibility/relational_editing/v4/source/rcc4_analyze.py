"""RCC4 CPU analysis: reads unit outputs (mirror) and writes plot-ready tables, SUMMARY.json and the V4 contract dispositions
(F0 technical, F1 per condition/seed, F2 laws + joint F2, F3 Qwen, F4 workflow, the four prespecified COMPLETE-minus-SPARSE coverage
contrasts, and the both/either operand-consistency diagnostic).  No model forwards.  Every number comes from saved per-query rows or
receipts.  All-24, all-program-questions, changed accuracy and ALL-invariant harm keep distinct denominators; disagreement counts are exact."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import rcc4_common as C
import rso3_common as Q
import rso3_analyze as A3

rows = A3.rows; load = A3.load; write_csv = A3.write_csv; _mean = A3._mean; _strip = A3._strip; metrics = A3.metrics; f1 = A3.f1; avg_scene = A3.avg_scene; program_scene_table = A3.program_scene_table; panel_scene = A3.panel_scene; _exact = A3._exact
LEARNED = tuple(C.CONDITIONS); FROZEN = tuple(C.FROZEN_V3)

def pb(a_ps, b_ps, key, level=C.CONTRAST_LEVEL):
    r = C.paired_bootstrap(a_ps, b_ps, key, level=level); return r

def contrast(per, a, b, key, level=C.CONTRAST_LEVEL):
    out = {}
    for seed in ('s0', 's1'):
        if f'{a}_{seed}' in per and f'{b}_{seed}' in per: out[seed] = pb(per[f'{a}_{seed}'], per[f'{b}_{seed}'], key, level)
    if 's0' in out and 's1' in out: out['avg'] = pb(avg_scene(per[f'{a}_s0'], per[f'{a}_s1'], key), avg_scene(per[f'{b}_s0'], per[f'{b}_s1'], key), key, level)
    return out

def spec_map(s, program):
    """query_id -> spec for one scene/program (both draws), regenerated deterministically from the bound protocol (evaluator side)."""
    return {q['query_id']: q['spec'] for draw in (0, 1) for q in C.program_queries(s, program, draw)}

def boolean_diagnostic(pr, scenes_by_id, conds):
    """For both/either questions whose operand direct queries exist in the same compiled artifact (same condition, scene, program, draw):
    operands both correct; logical consequence correct; consequence consistent with the direct predictions; new invariant error with the
    native final-world answer correct.  Raw denominators and missing-operand coverage; no filtering by native correctness anywhere else."""
    nat = {(r['scene_id'], r['program'], r['query_id']): r for r in pr.get('NATURAL_FINAL_WORLD', [])}; out = []; examples = []
    for cond in conds:
        by = defaultdict(dict)
        for r in pr.get(cond, []): by[(r['scene_id'], r['program'], r['draw'])][r['query_id']] = r
        for program in ('single', 'AB', 'BA', 'ABC', 'CBA', 'AB_cross'):
            st = dict(condition=cond, program=program, boolean_rows=0, operands_covered=0, operands_missing=0, operands_both_correct=0, consequence_correct=0, consequence_correct_given_operands_correct=0, consequence_consistent_with_direct=0, consistent_and_correct=0,
                      invariant_rows=0, new_invariant_errors=0, new_invariant_errors_native_correct=0, new_invariant_errors_operands_correct=0, changed_rows=0, changed_errors=0)
            for (sid, prog, draw), qs in by.items():
                if prog != program: continue
                sm = spec_map(scenes_by_id[sid], program)
                for qid, r in qs.items():
                    if r['family'] not in ('both', 'either'): continue
                    st['boolean_rows'] += 1; sp = sm[qid]; ops = []
                    for who in (sp['a'], sp['b']):
                        d = next((x for x2, x in qs.items() if sm.get(x2, {}).get('kind') == 'direct' and sm[x2]['a'] == who and sm[x2]['p'] == sp['p']), None); ops.append(d)
                    inv = r['gold'] == r['source_gold']; st['invariant_rows'] += int(inv); st['changed_rows'] += int(not inv)
                    err = r['prediction'] != r['gold']; new_err = inv and r['source_prediction'] == r['gold'] and err; st['new_invariant_errors'] += int(new_err); st['changed_errors'] += int((not inv) and err)
                    n_ = nat.get((sid, program, qid)); native_ok = n_ is not None and n_['prediction'] == n_['gold']
                    if new_err and native_ok: st['new_invariant_errors_native_correct'] += 1
                    if any(o is None for o in ops): st['operands_missing'] += 1; continue
                    st['operands_covered'] += 1; xs = [o['prediction'] for o in ops]; both_ok = all(o['prediction'] == o['gold'] for o in ops); st['operands_both_correct'] += int(both_ok)
                    implied = int(xs[0] and xs[1]) if r['family'] == 'both' else int(xs[0] or xs[1]); consistent = r['prediction'] == implied; correct = not err
                    st['consequence_correct'] += int(correct); st['consequence_consistent_with_direct'] += int(consistent); st['consistent_and_correct'] += int(consistent and correct)
                    if both_ok: st['consequence_correct_given_operands_correct'] += int(correct)
                    if new_err and both_ok:
                        st['new_invariant_errors_operands_correct'] += 1
                        if len(examples) < 40: examples.append(dict(condition=cond, program=program, scene_id=sid, draw=draw, query_id=qid, family=r['family'], operands=[dict(query_id=o['query_id'], prediction=o['prediction'], gold=o['gold']) for o in ops], prediction=r['prediction'], gold=r['gold'], native_correct=native_ok))
            if st['boolean_rows']: out.append(st)
    return out, examples

def analyze(run, qroot, out, store=None):
    run = Path(run); qroot = Path(qroot); out = Path(out); out.mkdir(parents=True, exist_ok=True); S = dict(at=Q.now(), seed=C.SEED, actors={}); T = defaultdict(list)
    for actor in ('gemma', 'qwen'):
        M = dict(); q = qroot/actor/'q'; qual = load(q/'QUALIFICATION.json'); ic = load(q/'INTERFACE_CHECK.json'); gc = load(q/'GRADIENT_CHECK.json'); bench = load(q/'BENCHMARK.json')
        if not qual and not (run/actor).is_dir(): continue
        min_scenes = 64 if actor == 'gemma' else 32
        M['qualification'] = qual and dict(selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'], policy=qual.get('policy'), generations=qual['total_generations'], attempts=[dict(procedure=a['procedure'], qualified=a['qualified'], **{w: {k: v for k, v in a['summary'][w].items() if k != 'n'} for w in ('source', 'counterfactual')}) for a in qual['attempts']])
        if qual:
            for a in qual['attempts']:
                for w in ('source', 'counterfactual', 'pooled'): T['T1_native_qualification'].append(dict(actor=actor, procedure=a['procedure'], world=w, **a['summary'][w]))
        M['interface'] = ic and {k: v for k, v in ic.items() if k != 'rows'}
        if M['interface']: M['interface']['setter_identities'] = {k: v for k, v in M['interface']['setter_identities'].items() if k != 'rows'}
        M['gradient'] = gc and dict(pass_=gc['pass_'], results={k: dict(editor=v['editor_grad_nonzero'], backbone=v['backbone_grad'], move=v['max_score_move']) for k, v in gc['results'].items()}); M['benchmark'] = bench and bench['seconds']
        M['F0'] = dict(status=('TECHNICAL_PASS' if (ic and ic['pass_'] and gc and gc['pass_']) else 'TECHNICAL_DEFECT'), interface=ic and ic['pass_'], gradient=gc and gc['pass_'])
        d = run/actor; fp = d/'final_phase'; fz = load(d/'FREEZE.json'); M['development'] = dict(status='NOT_MEASURED'); leak = load(d/'LEAKAGE_CHECK.json'); fc = load(d/'FORECAST.json')
        if fz:
            base = load(d/'BASELINES.json'); conds = {}
            for cond in fz['conditions']:
                conds[cond] = {}
                for seed in C.SEEDS:
                    dn = load(d/'arms'/cond/f's{seed}'/'DONE.json')
                    if not dn: conds[cond][f's{seed}'] = dict(status='NOT_MEASURED'); continue
                    for c in dn['curve']:
                        T['T2_development_curves'].append(dict(actor=actor, condition=cond, arm=c['arm'], coverage=c['coverage'], lr=c['lr'], seed=seed, checkpoint=c['checkpoint'], cal_changed=c['cal']['changed'], cal_harm=c['cal']['harm'], cal_all24=c['cal']['all24'], cal_all_program_questions=c['cal'].get('all_program_questions'),
                                                              cal_dir0=c['cal']['changed_by_direction'].get('0', c['cal']['changed_by_direction'].get(0)), cal_dir1=c['cal']['changed_by_direction'].get('1', c['cal']['changed_by_direction'].get(1)), cal_by_family=c['cal'].get('by_family'), cal_sparse_changed=c['cal_sparse_subset']['changed'], cal_sparse_harm=c['cal_sparse_subset']['harm'], cal_sparse_all12=c['cal_sparse_subset']['all24'],
                                                              feasible=c['feasible'], selected=(c['checkpoint'] == dn['selection']['checkpoint']), train_loss=c['loss_train'], reg_train=c['reg_train'], steps=c['steps'], slots=c['slots'], train_seconds=c['train_seconds'], eval_seconds=c['eval_seconds'], realized_norm=c['norm'], frobenius=c['frobenius']))
                    conds[cond][f's{seed}'] = dict(status='SELECTED', checkpoint=dn['selection']['checkpoint'], feasible=dn['selection']['feasible'], label=dn['selection']['label'], cal={k: v for k, v in dn['selection']['cal'].items() if k not in ('per_scene', 'by_cell')}, opportunity=dn['opportunity'], warm_start=dict(dir=dn['warm_start']['dir'], sha256=dn['warm_start']['sha256']), retention=(dn['selection']['checkpoint'] == 0))
            M['development'] = dict(status='COMPLETE' if not fz['not_measured'] else 'INCOMPLETE', procedure=fz['procedure'], site=fz['site'], epochs=fz['epochs'], conditions=conds, canonical_current=fz['canonical_current'], workflow_candidate=fz['workflow_candidate'], not_measured=fz['not_measured'], leakage_check=leak and dict(pass_=leak['pass_'], groups_per_epoch=leak['groups_per_epoch'], slots=leak['slots_per_scene'], weight=leak['weight_per_scene']),
                                    forecast=fc and dict(fits=fc['fits'], first_epoch_train_seconds=fc['first_epoch_train_seconds'], predicted_evaluation_seconds=fc['predicted_evaluation_seconds']['total']), baselines=base and {k: _strip(v) for k, v in base.items() if k != 'at'}, opportunities_identical=len({json.dumps(v[s]['opportunity'], sort_keys=True) for v in conds.values() for s in v if v[s].get('opportunity')}) == 1)
        F = dict(status='NOT_MEASURED'); per = {}
        if (fp/'EVAL_DONE.json').is_file():
            ev = load(fp/'EVAL_DONE.json'); seal = load(fp/'SEAL.json'); scope = load(fp/'SCOPE.json'); allrows = {c: rows(fp/'final'/f'{c}.jsonl') for c in ev['conditions']}; cf = allrows.get('COUNTERFACTUAL', [])
            for c, rs in allrows.items(): per[c] = metrics(rs, cf if c != 'COUNTERFACTUAL' else None)
            for c, rs in allrows.items():
                fam = {f: float(np.mean([r['prediction'] == r['gold'] for r in rs if r['family'] == f])) for f in ('direct', 'opposes', 'same', 'both', 'either') if any(r['family'] == f for r in rs)}
                T['T6_family_split'].append(dict(actor=actor, condition=c, **{f'acc_{f}': v for f, v in fam.items()}, boolean_changed=Q.S.summarize([r for r in rs if r['family'] in ('both', 'either')])['changed'] if any(r['family'] in ('both', 'either') for r in rs) else None, equality_changed=Q.S.summarize([r for r in rs if r['family'] == 'same'])['changed'] if any(r['family'] == 'same' for r in rs) else None))
                for o in C.ORDERS:
                    mo = Q.S.summarize([r for r in rs if r.get('order') == o]); T['T7_by_prefix_order'].append(dict(actor=actor, condition=c, order=o, changed=mo['changed'], harm=mo['harm'], all24=mo['all24'], accuracy=mo['accuracy'], n_scenes=mo['n_scenes']))
            for c, m in per.items(): T['T3_fresh_single'].append(dict(actor=actor, condition=c, **{k: v for k, v in m.items() if k not in ('per_scene', 'by_cell', 'by_family', 'by_order', 'generation')}, by_family=m['by_family'], by_cell=m['by_cell'], by_order=m['by_order'], generation=m['generation']))
            F1 = {}
            for arm in sorted({c.rsplit('_s', 1)[0] for c in per if '_s' in c}):
                seeds = {s: f1(per.get(f'{arm}_{s}'), min_scenes) for s in ('s0', 's1') if f'{arm}_{s}' in per}
                if not seeds: continue
                F1[arm] = dict(status=('F1_MET' if len(seeds) == 2 and all(v['status'] == 'PASS' for v in seeds.values()) else ('PARTIAL' if any(v['status'] == 'PASS' for v in seeds.values()) else 'FAIL')), seeds=seeds, learned=arm in LEARNED, frozen_v3=arm in FROZEN, question_blind=not ev['seal_mismatches'] and seal['questions_generated_before_seal'] is False)
            F1['natural'] = {c: f1(per.get(c), min_scenes) for c in ('COUNTERFACTUAL', 'TEXT_CORRECTION') if c in per}
            ps = {c: m['per_scene'] for c, m in per.items()}; CS = {}
            for a, b in (('INV_COMPLETE_SINGLE', 'INV_SPARSE_CONTINUE'), ('FREE_COMPLETE_SINGLE', 'FREE_SPARSE_CONTINUE'), ('INV_COMPLETE_SINGLE', 'FREE_COMPLETE_SINGLE'), ('INV_SPARSE_CONTINUE', 'FREE_SPARSE_CONTINUE'), ('INV_SPARSE_CONTINUE', 'FROZEN_V3_INVARIANT_SET'), ('INV_COMPLETE_SINGLE', 'FROZEN_V3_INVARIANT_SET'), ('FREE_SPARSE_CONTINUE', 'FROZEN_V3_FREE_OVERWRITE'), ('FREE_COMPLETE_SINGLE', 'FROZEN_V3_FREE_OVERWRITE')):
                if f'{a}_s0' in ps and f'{b}_s0' in ps: CS[f'{a}_minus_{b}'] = dict(all24=contrast(ps, a, b, 'all24'), changed=contrast(ps, a, b, 'changed', .95), harm=contrast(ps, a, b, 'harm', .95))
            for a in LEARNED+FROZEN:
                for b in ('COUNTERFACTUAL', 'TEXT_CORRECTION'):
                    if f'{a}_s0' in ps and b in ps: CS[f'{a}_minus_{b}'] = dict(all24={s: pb(ps[f'{a}_{s}'], ps[b], 'all24') for s in ('s0', 's1') if f'{a}_{s}' in ps})
            pg = load(fp/'PROGRAMS.json'); F2 = dict(status='NOT_MEASURED'); COV = dict(status='NOT_MEASURED'); BOOL = None
            if pg and pg.get('status') == 'MEASURED':
                pr = {c: rows(fp/'programs'/f'{c}.jsonl') for c in pg['rows']}; F2 = dict(status='MEASURED', conditions={}, laws={}, scenes=len(pg['scenes']), shared_measurements=pg.get('shared_measurements'))
                tables = {c: program_scene_table(pr, c) for c in pr}; single_rows = {c: {r['query_id']: r for r in pr[c] if r.get('program') == 'single'} for c in pr}
                for c in pr:
                    F2['conditions'][c] = {}
                    for program in C.PROGRAMS:
                        sub = [r for r in pr[c] if r.get('program') == program]
                        if not sub: continue
                        m = Q.summarize(sub); cmp_ = [(r, single_rows[c].get(r['query_id'])) for r in sub if not r.get('additional')]; cmp_ = [(r, s_) for r, s_ in cmp_ if s_ is not None]
                        n_dis = int(sum(r['prediction'] != s_['prediction'] for r, s_ in cmp_)); dis = (n_dis/len(cmp_)) if cmp_ else None; dlogp = float(np.mean([abs(r['gold_logp']-s_['gold_logp']) for r, s_ in cmp_])) if cmp_ else None
                        e = dict(changed=m['changed'], harm=m['harm'], conditional_harm=m['conditional_harm'], all24=m['all24'], all_program_questions=m['all_program_questions'], accuracy=m['accuracy'], accuracy_all_program_questions=m['accuracy_all_program_questions'], touched_accuracy=m['touched_accuracy'], other_address_error=m['other_address_error'],
                                 preservation=m['preservation'], nonaddressed_change=m['nonaddressed_change'], additional_accuracy=m.get('additional_accuracy'), additional_rows=m.get('additional_rows'), n_scenes=m['n_scenes'], n_rows=len(sub), disagreement_vs_single=dis, disagreement_count=n_dis, disagreement_denominator=len(cmp_), mean_abs_dlogp_vs_single=dlogp, energy=_mean(sub, 'energy'), cast_error=_mean(sub, 'cast_error'), commands=_mean(sub, 'commands'),
                                 shared_from=sorted({r['shared_measurement_from'] for r in sub if r.get('shared_measurement_from')}), shared_from_condition=sorted({r['shared_measurement_from_condition'] for r in sub if r.get('shared_measurement_from_condition')}), by_family=m['by_family'])
                        F2['conditions'][c][program] = e; T['T4_edit_programs'].append(dict(actor=actor, condition=c, program=program, **e))
                ident = rows(fp/'programs'/'IDENTITY.jsonl'); ID = defaultdict(list)
                for r in ident: ID[(r['condition'], r['program'])].append(r)
                F2['identity_errors'] = {f'{c}|{p}': dict(n=len(v), max_workspace=max([x['workspace_max_abs_vs_single'] for x in v if x['workspace_max_abs_vs_single'] is not None] or [None]), max_bf16=max([x['bf16_max_abs_vs_single'] for x in v if x['bf16_max_abs_vs_single'] is not None] or [None]), hash_equal_single_fraction=float(np.mean([x['hash_equal_single'] for x in v]))) for (c, p), v in sorted(ID.items())}
                for k, v in F2['identity_errors'].items(): T['T8_identity_errors'].append(dict(actor=actor, condition=k.split('|')[0], program=k.split('|')[1], **v))
                def laws(c):
                    e = F2['conditions'].get(c, {}); f1s = F1.get(c.rsplit('_s', 1)[0], {}).get('seeds', {}).get('s'+c.rsplit('_s', 1)[1], {}) if '_s' in c else {}
                    two = all(e.get(p, {}).get('changed') is not None and e[p]['changed'] >= .80 and e[p]['harm'] <= .05 for p in ('AB', 'BA')); three = all(e.get(p, {}).get('changed') is not None and e[p]['changed'] >= .80 and e[p]['harm'] <= .05 for p in ('ABC', 'CBA'))
                    rep = {p: bool(e.get(p, {}).get('disagreement_vs_single') is not None and e[p]['disagreement_vs_single'] <= .05) for p in ('repeat2', 'repeat4', 'repeat8')}
                    rest = {p: bool((e.get(p, {}).get('preservation') or 0) >= .95) for p in ('restore', 'restore_after4')}
                    ps_ = e.get('single', {}); single_quality = bool(ps_ and ps_['changed'] is not None and ps_['changed'] >= .85 and ps_['harm'] <= .05)
                    return dict(two_edit=two, three_edit=three, repeat=rep, repeat_all=all(rep.values()) and f1s.get('status') == 'PASS' and single_quality, restore=rest, restore_all=all(rest.values()), single_f1_quality_on_program_subset=single_quality, f1_full_panel=f1s.get('status'),
                                joint=two and three and all(rep.values()) and all(rest.values()) and f1s.get('status') == 'PASS' and single_quality, noop_preservation=e.get('noop', {}).get('preservation'), cross_project_two_edit=bool(e.get('AB_cross', {}).get('changed') is not None and e['AB_cross']['changed'] >= .80 and e['AB_cross']['harm'] <= .05),
                                joint_invariant_errors={p: (e.get(p, {}).get('harm')) for p in ('AB', 'BA', 'ABC', 'CBA')})
                F2['laws'] = {c: laws(c) for c in F2['conditions'] if c not in ('SOURCE', 'NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ')}
                for arm in sorted({c.rsplit('_s', 1)[0] for c in F2['laws']}):
                    ls = [F2['laws'][f'{arm}_s{s}'] for s in (0, 1) if f'{arm}_s{s}' in F2['laws']]
                    F2.setdefault('by_arm', {})[arm] = dict(two_edit_both_seeds=all(l['two_edit'] for l in ls), three_edit_both_seeds=all(l['three_edit'] for l in ls), repeat_both_seeds=all(l['repeat_all'] for l in ls), restore_both_seeds=all(l['restore_all'] for l in ls), joint_f2_both_seeds=all(l['joint'] for l in ls) and len(ls) == 2, seeds=len(ls))
                # panels: coverage endpoint (mean AB/ABC all-program-questions within scene), repeat/restore, two/three-edit harm, multi-program
                covP = {c: panel_scene(tables[c], C.COVERAGE_PROGRAMS, 'all_program_questions') for c in tables}; panelP = {c: panel_scene(tables[c], Q.REPEAT_RESTORE_PANEL, 'all_program_questions') for c in tables}
                harmP = {c: panel_scene(tables[c], ('AB', 'BA', 'ABC', 'CBA'), 'harm') for c in tables}; multi = [p for p in C.PROGRAMS if p != 'single']; multiP = {c: panel_scene(tables[c], multi, 'all_program_questions') for c in tables}
                def pc(P_, a, b, key): return contrast(P_, a, b, key)
                COV = dict(status='MEASURED', endpoint='mean over AB and ABC of all-program-questions-correct within scene (order-equivalent BA/CBA not double counted); seeds averaged within scene; 98.75% two-sided paired scene bootstrap, 10,000 draws', primary={}, seed_specific={}, supplementary={})
                for a, b in C.MATCHED:
                    r = pc(covP, a, b, 'all_program_questions'); COV['primary'][f'{a}_minus_{b}'] = r
                for a, b in (('INV_COMPLETE_SINGLE', 'FREE_COMPLETE_SINGLE'), ('INV_SPARSE_CONTINUE', 'FREE_SPARSE_CONTINUE'), ('INV_SPARSE_CONTINUE', 'FROZEN_V3_INVARIANT_SET'), ('INV_COMPLETE_SINGLE', 'FROZEN_V3_INVARIANT_SET'), ('FREE_SPARSE_CONTINUE', 'FROZEN_V3_FREE_OVERWRITE'), ('FREE_COMPLETE_SINGLE', 'FROZEN_V3_FREE_OVERWRITE'), ('INV_COMPLETE_SINGLE', 'CANONICAL_CURRENT'), ('FREE_COMPLETE_SINGLE', 'CANONICAL_CURRENT'), ('INV_SPARSE_CONTINUE', 'CANONICAL_CURRENT'), ('FREE_SPARSE_CONTINUE', 'CANONICAL_CURRENT')):
                    if f'{a}_s0' in covP and f'{b}_s0' in covP: COV['supplementary'][f'{a}_minus_{b}'] = dict(coverage_endpoint=pc(covP, a, b, 'all_program_questions'), repeat_restore=pc(panelP, a, b, 'all_program_questions'), two_three_edit_harm=pc(harmP, a, b, 'harm'), multi_program=pc(multiP, a, b, 'all_program_questions'))
                for a in LEARNED+FROZEN+(C.CANONICAL,):
                    for b in ('NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ'):
                        if f'{a}_s0' in covP and b in covP: COV['supplementary'][f'{a}_minus_{b}'] = dict(coverage_endpoint={s: pb(covP[f'{a}_{s}'], covP[b], 'all_program_questions') for s in ('s0', 's1') if f'{a}_{s}' in covP})
                F2['panel_means'] = {c: dict(coverage_endpoint=float(np.mean([v['all_program_questions'] for v in covP[c].values()])) if covP[c] else None, repeat_restore_all_questions=float(np.mean([v['all_program_questions'] for v in panelP[c].values()])) if panelP[c] else None, multi_program_all_questions=float(np.mean([v['all_program_questions'] for v in multiP[c].values()])) if multiP[c] else None, two_three_edit_harm=float(np.mean([v['harm'] for v in harmP[c].values()])) if harmP[c] else None, scenes=len(covP[c])) for c in tables}
                for c, v in F2['panel_means'].items(): T['T9_program_panels'].append(dict(actor=actor, condition=c, **v))
                for name, r in list(COV['primary'].items())+[(k+'|'+kk, vv) for k, v in COV['supplementary'].items() for kk, vv in v.items()]:
                    for s, x in (r or {}).items():
                        if x: T['T10_coverage_contrasts'].append(dict(actor=actor, contrast=name, seed=s, effect=x['effect'], lower=x['lower'], upper=x['upper'], scenes=x['scenes'], discordant=x['discordant'], level=x['level']))
                def coverage_verdict(a, b):
                    r = COV['primary'].get(f'{a}_minus_{b}', {}); avg = r.get('avg'); f1a = F1.get(a, {}).get('status') == 'F1_MET'; j = F2.get('by_arm', {}).get(a, {}).get('joint_f2_both_seeds', False); jb = F2.get('by_arm', {}).get(b, {}).get('joint_f2_both_seeds', False)
                    return dict(gain=avg and avg['effect'], lower=avg and avg['lower'], gain_ge_10_positive=bool(avg and avg['effect'] >= .10 and avg['lower'] > 0), complete_retains_f1=f1a, complete_joint_f2=j, sparse_joint_f2=jb, per_seed={s: (r.get(s) or {}).get('effect') for s in ('s0', 's1')},
                                coverage_explanation_supported=bool(avg and avg['effect'] >= .10 and avg['lower'] > 0 and f1a and j), absolute_capability_banked=j, ordinary_development_suffices=bool(j and jb))
                COV['verdicts'] = {f'{a}_vs_{b}': coverage_verdict(a, b) for a, b in C.MATCHED}
                BOOL, examples = boolean_diagnostic(pr, {s.scene_id: s for s in C.scenes('FINAL')}, [c for c in pr if c not in ('SOURCE', 'NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ')])
                for r in BOOL: T['T11_boolean_diagnostic'].append(dict(actor=actor, **r))
                (out/f'BOOLEAN_EXAMPLES_{actor}.json').write_text(json.dumps(examples, indent=1)+'\n', encoding='utf8')
            wfj = load(fp/'WORKFLOW.json'); F4 = dict(status='NOT_MEASURED')
            if wfj and wfj.get('status') in ('MEASURED', 'PARTIAL'):
                wr = rows(fp/'workflow'/'rows.jsonl'); by = defaultdict(list)
                for r in wr: by[(r['condition'], r['program'])].append(r)
                F4 = dict(status=wfj['status'], episodes=len(wfj.get('scenes_measured', wfj['scenes'])), contracted_episodes=len(wfj['scenes']), conditions={}, intervals={}, criteria={}, candidate=(fz or {}).get('workflow_candidate', {}).get('condition'))
                for (c, p), rs in sorted(by.items()):
                    chg = [r for r in rs if r['majority_changes']]; inv = [r for r in rs if not r['majority_changes']]
                    e = dict(n=len(rs), final_decision_accuracy=float(np.mean([r['correct'] for r in rs])), all_receivers_correct=float(np.mean([r['receivers_all_correct'] for r in rs])), complete_episode_correct=float(np.mean([r['correct'] and r['receivers_all_correct'] for r in rs])), receiver_accuracy=float(np.mean([r['receiver_accuracy'] for r in rs])),
                             receiver_failures=int(sum(1 for r in rs if r['aggregator_status'] == 'RECEIVER_FAILED')), aggregator_unparsed=int(sum(1 for r in rs if r['aggregator_answer'] is None and r['aggregator_status'] != 'RECEIVER_FAILED')), correct_when_majority_changes=float(np.mean([r['correct'] for r in chg])) if chg else None,
                             correct_when_invariant=float(np.mean([r['correct'] for r in inv])) if inv else None, reports_reused=int(sum(1 for r in rs if r.get('reports_reused_from_identical_artifact'))))
                    F4['conditions'][f'{c}|{p}'] = e; T['T5_workflow'].append(dict(actor=actor, condition=c, program=p, **e))
                def pairs(a, b, p):
                    A_ = {r['scene_id']: float(r['correct'] and r['receivers_all_correct']) for r in by.get((a, p), [])}; B_ = {r['scene_id']: float(r['correct'] and r['receivers_all_correct']) for r in by.get((b, p), [])}; ks = sorted(set(A_) & set(B_))
                    if not ks: return None
                    r_ = C.paired_bootstrap({k: dict(x=A_[k]) for k in ks}, {k: dict(x=B_[k]) for k in ks}, 'x', level=.95); r_['exact_binomial'] = _exact(A_, B_, ks); return r_
                cand = F4['candidate']
                for p in C.WORKFLOW_PROGRAMS:
                    for s in (0, 1):
                        for b in (f'FROZEN_V3_INVARIANT_SET_s{s}', f'{C.CANONICAL}_s{s}', 'NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ', 'SOURCE'):
                            if cand and (f'{cand}_s{s}', p) in by and (b, p) in by: F4['intervals'][f'{cand}_s{s}_minus_{b}|{p}'] = pairs(f'{cand}_s{s}', b, p)
                        if (f'FROZEN_V3_INVARIANT_SET_s{s}', p) in by and ('NATURAL_FINAL_WORLD', p) in by: F4['intervals'][f'FROZEN_V3_INVARIANT_SET_s{s}_minus_NATURAL_FINAL_WORLD|{p}'] = pairs(f'FROZEN_V3_INVARIANT_SET_s{s}', 'NATURAL_FINAL_WORLD', p)
                for nm in ([cand] if cand else [])+['FROZEN_V3_INVARIANT_SET', C.CANONICAL]:
                    for s in (0, 1):
                        c = F4['conditions']; k = f'{nm}_s{s}'
                        if not any(kk.startswith(k+'|') for kk in c): continue
                        crit = dict(duplicate_delivery_final_ge_90=bool(c.get(f'{k}|repeat2', {}).get('final_decision_accuracy', 0) >= .90), duplicate_delivery_receivers_ge_90=bool(c.get(f'{k}|repeat2', {}).get('all_receivers_correct', 0) >= .90),
                                    restoration_final_ge_90=bool(c.get(f'{k}|restore', {}).get('final_decision_accuracy', 0) >= .90), restoration_receivers_ge_90=bool(c.get(f'{k}|restore', {}).get('all_receivers_correct', 0) >= .90))
                        F4['criteria'][k] = dict(crit, met=all(crit.values()), note='absolute retention only; the original >10 pp comparison against the affine comparator is a V3 result and is not rerun here')
            wit = load(fp/'WITNESS_INDEX.json'); wchk = []
            if wit:
                for f in wit.get('files', []):
                    wp = Path(f['path']); wp = wp if wp.is_file() else fp/'witness'/wp.name
                    if not wp.is_file(): wchk.append(dict(condition=f['condition'], scene_id=f['scene_id'], status='ABSENT_LOCALLY')); continue
                    with np.load(wp, allow_pickle=False) as z:
                        post = z['post_block_residual']; nxt = z['next_block_input']; idx = z['edited_positions']; ed = z['edited_states_bf16']; mask = np.ones(post.shape[0], bool); mask[idx] = False
                        wchk.append(dict(condition=f['condition'], scene_id=f['scene_id'], unedited_equal=bool(np.array_equal(post[mask], nxt[mask])), edited_equal=bool(len(idx) == 0 or np.array_equal(nxt[idx], ed)), edited_tokens=int(len(idx)), max_edit_norm=(float(np.linalg.norm(ed-post[idx], axis=1).max()) if len(idx) else 0.0)))
            anc = load(fp/'ANCHOR_PANEL.json'); ANC = anc and dict(status=anc.get('status'), conditions={c: _strip({k: v for k, v in metrics(rows(fp/'anchor'/f'{c}.jsonl')).items() if k != 'per_scene'}) for c in anc.get('rows', {})} if anc.get('status') == 'MEASURED' else None)
            F = dict(status='MEASURED', scope=scope['scope'], sizes=scope['sizes'], seal=dict(artifacts=seal['artifacts'], sha256=seal['seal_sha256'], at=seal['at'], first_question_at=ev['first_question_generated_at'], mismatches=len(ev['seal_mismatches']), master_failures=ev['master_failures'], shared_measurements=ev.get('shared_measurements')),
                     conditions={c: _strip({k: v for k, v in m.items() if k != 'per_scene'}) for c, m in per.items()}, F1=F1, single_contrasts=CS, F2=F2, coverage=COV, F4=F4, anchor_panel=ANC,
                     witnesses=wit and dict(files=len(wit.get('files', [])), propagation_verified_locally=(bool(wchk) and all(w.get('unedited_equal') and w.get('edited_equal') for w in wchk if 'unedited_equal' in w)), absent_locally=sum(1 for w in wchk if w.get('status') == 'ABSENT_LOCALLY'), checks=wchk, max_cast_error=max([f['cast_error'] for f in wit.get('files', [])] or [0])))
        elif (fp/'NATURAL_PANEL.json').is_file():
            npn = load(fp/'NATURAL_PANEL.json'); F = dict(status='NATURAL_PANEL_ONLY', procedure=npn.get('procedure'), conditions={c: _strip({k: v for k, v in metrics(rows(fp/'natural'/f'{c}.jsonl')).items() if k != 'per_scene'}) for c in npn.get('rows', {})})
        M['final'] = F; S['actors'][actor] = M
    for name, table in T.items(): write_csv(out/f'{name}.csv', table)
    (out/'SUMMARY.json').write_text(json.dumps(Q.json_safe(S), indent=1)+'\n', encoding='utf8')
    return S

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    S = analyze(a.run, a.run, a.out); print(json.dumps({k: {kk: (vv.get('status') if isinstance(vv, dict) else vv) for kk, vv in v.items() if kk in ('F0', 'development', 'final')} for k, v in S['actors'].items()}, indent=1))
