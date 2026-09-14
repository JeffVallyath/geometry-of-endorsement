"""RSO3 CPU analysis: reads unit outputs (mirror) and writes plot-ready tables, SUMMARY.json and the PROGRESS_CONTRACT dispositions
(F0 technical, F1 single-edit capability, F2 in-place update operation + four predeclared contrasts, F3 independent backbone, F4 workflow).
No model forwards.  Every number comes from saved per-query rows or receipts.  Identity errors are reported as numbers, never as results."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import rso3_common as Q

def rows(path): return Q.read_rows(path) if Path(path).is_file() else []
def load(path): return Q.load(path) if Path(path).is_file() else None
def write_csv(path, table):
    if not table: Path(path).write_text(''); return
    keys = list(dict.fromkeys(k for r in table for k in r))
    with Path(path).open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in table: w.writerow({k: ('' if r.get(k) is None else (json.dumps(r[k]) if isinstance(r[k], (dict, list)) else r[k])) for k in keys})
def _mean(rs, k):
    xs = [r[k] for r in rs if r.get(k) is not None]; return float(np.mean(xs)) if xs else None
def _strip(m): return {k: v for k, v in m.items() if k not in ('per_scene', 'by_cell')} if m else m

def metrics(rs, cf_rows=None):
    if not rs: return None
    m = Q.summarize(rs); ps = m['per_scene']
    m.update(all24_interval=Q.bootstrap_mean([v['all24'] for v in ps.values()]), changed_interval=Q.bootstrap_mean([v['changed'] for v in ps.values()]), harm_interval=Q.bootstrap_mean([v['harm'] for v in ps.values()]),
             by_order={o: Q.S.summarize([r for r in rs if r.get('order') == o and not r.get('additional')])['all24'] for o in sorted({r.get('order') for r in rs})}, edited_tokens=_mean(rs, 'edited_tokens'), energy=_mean(rs, 'energy'), frobenius=_mean(rs, 'frobenius'), cast_error=_mean(rs, 'cast_error'),
             ties=int(sum(1 for r in rs if r.get('tie'))), argmax_outside_labels=int(sum(1 for r in rs if r.get('argmax_in_labels') is False)), mean_gold_logp=_mean(rs, 'gold_logp'))
    gen = [r for r in rs if 'generation' in r]
    m['generation'] = dict(rows=len(gen), parse_ok=float(np.mean([str(r.get('parse_status', '')).startswith('OK') for r in gen])), agrees_with_score=float(np.mean([r.get('generation_symbol') == r['labels'][r['prediction']] for r in gen]))) if gen else None
    if cf_rows: cf = Q.summarize(cf_rows); m['coherence_ratio'] = (m['all24']/cf['all24']) if cf['all24'] else None
    else: m['coherence_ratio'] = 1.0      # the natural counterfactual IS the reference
    return m

def f1(m, min_scenes):
    if m is None: return dict(status='NOT_MEASURED')
    checks = dict(changed_ge_85=m['changed'] is not None and m['changed'] >= .85, harm_le_5=m['harm'] is not None and m['harm'] <= .05, each_direction_ge_80=all(v is not None and v >= .80 for v in m['changed_by_direction'].values()),
                  all24_ge_70pct_of_reference=(m.get('coherence_ratio') or 0) >= .70, scenes_ge_min=m['n_scenes'] >= min_scenes)
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, changed=m['changed'], harm=m['harm'], conditional_harm=m['conditional_harm'], all24=m['all24'], coherence_ratio=m.get('coherence_ratio'), directions=m['changed_by_direction'], n_scenes=m['n_scenes'])

def avg_scene(ps0, ps1, key):
    return {k: {key: (ps0[k][key]+ps1[k][key])/2} for k in ps0 if k in ps1 and ps0[k][key] is not None and ps1[k][key] is not None}

def contrast(per, a, b, key, level=Q.CONTRAST_LEVEL):
    out = {}
    for seed in ('s0', 's1'):
        if f'{a}_{seed}' in per and f'{b}_{seed}' in per: out[seed] = Q.paired_bootstrap(per[f'{a}_{seed}']['per_scene'], per[f'{b}_{seed}']['per_scene'], key, level=level)
    if 's0' in out and 's1' in out: out['avg'] = Q.paired_bootstrap(avg_scene(per[f'{a}_s0']['per_scene'], per[f'{a}_s1']['per_scene'], key), avg_scene(per[f'{b}_s0']['per_scene'], per[f'{b}_s1']['per_scene'], key), key, level=level)
    return out

def noninferior(per, a, b, key, margin):
    out = {}
    for seed in ('s0', 's1'):
        if f'{a}_{seed}' in per and f'{b}_{seed}' in per: r = Q.one_sided_lower(per[f'{a}_{seed}']['per_scene'], per[f'{b}_{seed}']['per_scene'], key); out[seed] = dict(r, noninferior=r['lower'] >= -margin, margin=margin)
    if 's0' in out and 's1' in out: r = Q.one_sided_lower(avg_scene(per[f'{a}_s0']['per_scene'], per[f'{a}_s1']['per_scene'], key), avg_scene(per[f'{b}_s0']['per_scene'], per[f'{b}_s1']['per_scene'], key), key); out['avg'] = dict(r, noninferior=r['lower'] >= -margin, margin=margin)
    return out

def program_scene_table(pr, cond):
    """per scene x program: all_program_questions, all24 (original bundle), changed, harm, preservation, accuracy."""
    out = defaultdict(dict)
    for program in Q.PROGRAMS:
        sub = [r for r in pr.get(cond, []) if r.get('program') == program]
        if not sub: continue
        for sid, v in Q.summarize(sub)['per_scene'].items(): out[sid][program] = v
    return out

def panel_scene(table, programs, key):
    """per scene: mean over the given programs of `key` (equally weighted), only scenes with every program present."""
    return {sid: {key: float(np.mean([d[p][key] for p in programs]))} for sid, d in table.items() if all(p in d and d[p].get(key) is not None for p in programs)}

def analyze(run, qdir, out, store=None):
    run = Path(run); qdir = Path(qdir); out = Path(out); out.mkdir(parents=True, exist_ok=True); S = dict(at=Q.now(), actors={}); T = defaultdict(list)
    for actor in ('gemma', 'qwen'):
        M = dict(); q = qdir/actor; qual = load(q/'QUALIFICATION.json'); an = load(q/'ANCHORS.json'); ic = load(q/'INTERFACE_CHECK.json'); gc = load(q/'GRADIENT_CHECK.json'); bench = load(q/'BENCHMARK.json')
        if not qual and not (run/actor).is_dir(): continue
        min_scenes = 64 if actor == 'gemma' else 32
        M['qualification'] = qual and dict(selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'], policy=qual.get('policy'), generations=qual['total_generations'], attempts=[dict(procedure=a['procedure'], qualified=a['qualified'], **{w: {k: v for k, v in a['summary'][w].items() if k not in ('n',)} for w in ('source', 'counterfactual')}) for a in qual['attempts']])
        if qual:
            for a in qual['attempts']:
                for w in ('source', 'counterfactual', 'pooled'): T['T1_native_qualification'].append(dict(actor=actor, procedure=a['procedure'], world=w, **{k: v for k, v in a['summary'][w].items()}))
        M['anchors'] = an and dict(status=an.get('status'), bit_identical=an.get('bit_identical'), seeds={k: {kk: vv for kk, vv in v.items() if kk != 'mismatches'} | dict(mismatches=len(v.get('mismatches', []))) for k, v in an.get('seeds', {}).items()})
        M['interface'] = ic and {k: v for k, v in ic.items() if k not in ('rows',)} ; M['gradient'] = gc and dict(pass_=gc['pass_'], results={k: dict(editor=v['editor_grad_nonzero'], backbone=v['backbone_grad'], move=v['max_score_move']) for k, v in gc['results'].items()})
        if M['interface']: M['interface']['setter_identities'] = {k: v for k, v in M['interface']['setter_identities'].items() if k != 'rows'}
        M['benchmark'] = bench and bench['seconds']
        M['F0'] = dict(status=('TECHNICAL_PASS' if (ic and ic['pass_'] and gc and gc['pass_'] and (not an or an.get('status') in ('TECHNICAL_PASS', 'NOT_APPLICABLE'))) else 'TECHNICAL_DEFECT'), anchors=an and an.get('status'), anchors_bit_identical=an and an.get('bit_identical'), interface=ic and ic['pass_'], gradient=gc and gc['pass_'])
        d = run/actor; fp = d/'final_phase'; fz = load(d/'FREEZE.json'); M['development'] = dict(status='NOT_MEASURED'); init = load(d/'INIT.json')
        if fz:
            plan = load(d/'DEV_PLAN.json'); base = load(d/'BASELINES.json'); cap = load(d/'CAPTURES.json'); arms = {}
            for arm in fz['arms']:
                ad = load(d/'arms'/arm/'ARM_DONE.json')
                if not ad: arms[arm] = dict(status='NOT_MEASURED'); continue
                for lrdir in sorted((d/'arms'/arm).glob('lr*_s*')):
                    dn = load(lrdir/'DONE.json')
                    if not dn: continue
                    for c in dn['curve']:
                        T['T2_development_curves'].append(dict(actor=actor, arm=arm, lr=c['lr'], seed=c['seed'], epoch=c['epoch'], strength=c['strength'], cal_changed=c['cal']['changed'], cal_harm=c['cal']['harm'], cal_all24=c['cal']['all24'], cal_dir0=c['cal']['changed_by_direction'].get('0', c['cal']['changed_by_direction'].get(0)),
                                                              cal_dir1=c['cal']['changed_by_direction'].get('1', c['cal']['changed_by_direction'].get(1)), cal_feasible=c['feasible'], frontier=c['frontier'], fit_changed=c['fit']['changed'], fit_harm=c['fit']['harm'], fit_loss=c['fit']['loss'], train_loss=c['loss_train'], reg_train=c.get('reg_train'),
                                                              realized_norm=c['norm'], frobenius=c['frobenius'], visits=c['visits'], steps=c['steps'], train_seconds=c['train_seconds'], eval_seconds=c['eval_seconds'], not_extended=dn.get('not_extended', False)))
                sel = ad['selection']; arms[arm] = dict(status='SELECTED', lr=sel['lr'], epoch=sel['epoch'], strength=sel['strength'], feasible=sel['feasible'], label=sel['label'], cal={s: {k: v for k, v in sel['cal'][s].items() if k not in ('per_scene', 'by_cell', 'by_family')} for s in sel['cal']}, epochs=dict(seed0=ad['epochs_seed0'], seed1=ad['epochs_seed1']), opportunity=ad['opportunity'], init=ad['init'] and ad['init'].get('kind'))
            M['development'] = dict(status='COMPLETE', procedure=fz['procedure'], site=fz['site'], cap_ref=fz['cap_ref'], e_max=fz['e_max'], plan=plan and dict(e_max=plan['e_max'], predicted=plan['predicted_total_seconds'], dev_seconds=plan['dev_seconds']), baselines=base and {k: _strip(v) for k, v in base.items() if k != 'at'},
                                    captures=cap and dict(cap_ref=cap['cap_ref'], cap_rule=cap['cap_rule'], cap_new_rule=cap['cap_new_fit_rule'], delta_stats=cap['delta_stats'], mu_rule=cap['mu_rule']), init=init and {s: dict(checks=v['checks'], heads=v['heads'], u_rule=(v['u_rule'] if isinstance(v['u_rule'], dict) else v['u_rule'])) for s, v in init['seeds'].items()}, arms=arms, not_measured=fz.get('not_measured'))
        F = dict(status='NOT_MEASURED'); per = {}
        if (fp/'EVAL_DONE.json').is_file():
            ev = load(fp/'EVAL_DONE.json'); seal = load(fp/'SEAL.json'); scope = load(fp/'SCOPE.json'); allrows = {c: rows(fp/'final'/f'{c}.jsonl') for c in ev['conditions']}; cf = allrows.get('COUNTERFACTUAL', [])
            for c, rs in allrows.items(): per[c] = metrics(rs, cf if c != 'COUNTERFACTUAL' else None)
            for c, rs in allrows.items():
                fam = {f: float(np.mean([r['prediction'] == r['gold'] for r in rs if r['family'] == f])) for f in ('direct', 'opposes', 'same', 'both', 'either') if any(r['family'] == f for r in rs)}
                dn = [r for r in rs if r['family'] in ('direct', 'opposes')]; dm = Q.S.summarize(dn) if dn else None
                T['T6_family_split'].append(dict(actor=actor, condition=c, **{f'acc_{f}': v for f, v in fam.items()}, direct_negation_changed=dm and dm['changed'], direct_negation_harm=dm and dm['harm'], direct_negation_all_correct=dm and dm['all24'], equality_changed=Q.S.summarize([r for r in rs if r['family'] == 'same'])['changed'] if any(r['family'] == 'same' for r in rs) else None, boolean_changed=Q.S.summarize([r for r in rs if r['family'] in ('both', 'either')])['changed'] if any(r['family'] in ('both', 'either') for r in rs) else None))
                for o in Q.ORDERS:
                    mo = Q.S.summarize([r for r in rs if r.get('order') == o]); T['T7_by_prefix_order'].append(dict(actor=actor, condition=c, order=o, changed=mo['changed'], harm=mo['harm'], all24=mo['all24'], accuracy=mo['accuracy'], n_scenes=mo['n_scenes']))
            for c, m in per.items(): T['T3_fresh_single'].append(dict(actor=actor, condition=c, scope=scope['scope'], **{k: v for k, v in m.items() if k not in ('per_scene', 'by_cell', 'by_family', 'by_order', 'generation')}, by_family=m['by_family'], by_cell=m['by_cell'], by_order=m['by_order'], generation=m['generation']))
            F1 = {}
            for arm in sorted({c.rsplit('_s', 1)[0] for c in per if '_s' in c}):
                seeds = {s: f1(per.get(f'{arm}_{s}'), min_scenes) for s in ('s0', 's1') if f'{arm}_{s}' in per}
                if not seeds: continue
                both = len(seeds) == 2 and all(v['status'] == 'PASS' for v in seeds.values())
                F1[arm] = dict(status=('F1_MET' if both else ('PARTIAL' if any(v['status'] == 'PASS' for v in seeds.values()) else 'FAIL')), seeds=seeds, learned=arm in (Q.LEARNED_ARMS+Q.QWEN_ARMS), frozen=arm.startswith('FROZEN'), question_blind=not ev['seal_mismatches'] and seal['questions_generated_before_seal'] is False)
            F1['natural'] = {c: f1(per.get(c), min_scenes) for c in ('COUNTERFACTUAL', 'TEXT_CORRECTION') if c in per}
            # single-edit contrasts and all24 / changed noninferiority (EXACT named fields)
            NI = dict(all24={}, changed={}); CS = {}
            for a, b in (('INVARIANT_SET', 'FREE_OVERWRITE'), ('INVARIANT_SET', 'CONTINUED_AFFINE'), ('INVARIANT_SET', 'FROZEN_V2_CLAUSE'), ('INVARIANT_SET', 'AFFINE_CLAUSE'), ('FREE_OVERWRITE', 'CONTINUED_AFFINE'), ('CONTINUED_AFFINE', 'FROZEN_V2_CLAUSE'), ('FROZEN_V2_CLAUSE_FP32', 'FROZEN_V2_CLAUSE')):
                if f'{a}_s0' in per and f'{b}_s0' in per:
                    CS[f'{a}_minus_{b}'] = dict(all24=contrast(per, a, b, 'all24'), changed=contrast(per, a, b, 'changed', level=.95), harm=contrast(per, a, b, 'harm', level=.95))
                    NI['all24'][f'{a}_vs_{b}'] = noninferior(per, a, b, 'all24', Q.NONINFERIORITY['margin']); NI['changed'][f'{a}_vs_{b}'] = noninferior(per, a, b, 'changed', .05)
            for a in ('INVARIANT_SET', 'FREE_OVERWRITE', 'CONTINUED_AFFINE', 'AFFINE_CLAUSE'):
                for b in ('COUNTERFACTUAL', 'TEXT_CORRECTION'):
                    if f'{a}_s0' in per and b in per: CS[f'{a}_minus_{b}'] = dict(all24={s: Q.paired_bootstrap(per[f'{a}_{s}']['per_scene'], per[b]['per_scene'], 'all24') for s in ('s0', 's1') if f'{a}_{s}' in per})
            # programs
            pg = load(fp/'PROGRAMS.json'); F2 = dict(status='NOT_MEASURED')
            if pg and pg.get('status') == 'MEASURED':
                pr = {c: rows(fp/'programs'/f'{c}.jsonl') for c in pg['rows']}; F2 = dict(status='MEASURED', conditions={}, laws={}, scenes=len(pg['scenes']), shared_measurements=pg.get('shared_measurements'))
                tables = {c: program_scene_table(pr, c) for c in pr}; single_rows = {c: {r['query_id']: r for r in pr[c] if r.get('program') == 'single'} for c in pr}
                for c in pr:
                    F2['conditions'][c] = {}
                    for program in Q.PROGRAMS:
                        sub = [r for r in pr[c] if r.get('program') == program]
                        if not sub: continue
                        m = Q.summarize(sub); cmp_ = [(r, single_rows[c].get(r['query_id'])) for r in sub if not r.get('additional')]; cmp_ = [(r, s_) for r, s_ in cmp_ if s_ is not None]
                        dis = float(np.mean([r['prediction'] != s_['prediction'] for r, s_ in cmp_])) if cmp_ else None; dlogp = float(np.mean([abs(r['gold_logp']-s_['gold_logp']) for r, s_ in cmp_])) if cmp_ else None
                        e = dict(changed=m['changed'], harm=m['harm'], conditional_harm=m['conditional_harm'], all24=m['all24'], all_program_questions=m['all_program_questions'], accuracy=m['accuracy'], accuracy_all_program_questions=m['accuracy_all_program_questions'], touched_accuracy=m['touched_accuracy'], other_address_error=m['other_address_error'],
                                 preservation=m['preservation'], nonaddressed_change=m['nonaddressed_change'], additional_accuracy=m.get('additional_accuracy'), n_scenes=m['n_scenes'], disagreement_vs_single=dis, mean_abs_dlogp_vs_single=dlogp, energy=_mean(sub, 'energy'), cast_error=_mean(sub, 'cast_error'), commands=_mean(sub, 'commands'), shared_from=sorted({r['shared_measurement_from'] for r in sub if r.get('shared_measurement_from')}))
                        F2['conditions'][c][program] = e; T['T4_edit_programs'].append(dict(actor=actor, condition=c, program=program, **e))
                ident = rows(fp/'programs'/'IDENTITY.jsonl'); ID = defaultdict(list)
                for r in ident: ID[(r['condition'], r['program'])].append(r)
                F2['identity_errors'] = {f'{c}|{p}': dict(n=len(v), max_workspace=max([x['workspace_max_abs_vs_single'] for x in v if x['workspace_max_abs_vs_single'] is not None] or [None]), mean_workspace=(float(np.mean([x['workspace_max_abs_vs_single'] for x in v if x['workspace_max_abs_vs_single'] is not None])) if any(x['workspace_max_abs_vs_single'] is not None for x in v) else None),
                                                          max_bf16=max([x['bf16_max_abs_vs_single'] for x in v if x['bf16_max_abs_vs_single'] is not None] or [None]), hash_equal_single_fraction=float(np.mean([x['hash_equal_single'] for x in v]))) for (c, p), v in sorted(ID.items())}
                for k, v in F2['identity_errors'].items(): T['T8_identity_errors'].append(dict(actor=actor, condition=k.split('|')[0], program=k.split('|')[1], **v))
                def laws(c):
                    e = F2['conditions'].get(c, {}); f1s = F1.get(c.rsplit('_s', 1)[0], {}).get('seeds', {}).get('s'+c.rsplit('_s', 1)[1], {}) if '_s' in c else {}
                    two = all(e.get(p, {}).get('changed') is not None and e[p]['changed'] >= .80 and e[p]['harm'] <= .05 for p in ('AB', 'BA')); three = all(e.get(p, {}).get('changed') is not None and e[p]['changed'] >= .80 and e[p]['harm'] <= .05 for p in ('ABC', 'CBA'))
                    rep = {p: bool(e.get(p, {}).get('disagreement_vs_single') is not None and e[p]['disagreement_vs_single'] <= .05) for p in ('repeat2', 'repeat4', 'repeat8')}
                    rest = {p: bool((e.get(p, {}).get('preservation') or 0) >= .95) for p in ('restore', 'restore_after4')}
                    return dict(two_edit=two, three_edit=three, repeat=rep, repeat_all=all(rep.values()) and f1s.get('status') == 'PASS', restore=rest, restore_all=all(rest.values()), noop_preservation=e.get('noop', {}).get('preservation'), cross_project_two_edit=bool(e.get('AB_cross', {}).get('changed') is not None and e['AB_cross']['changed'] >= .80 and e['AB_cross']['harm'] <= .05))
                F2['laws'] = {c: laws(c) for c in F2['conditions'] if c not in ('SOURCE', 'NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ')}
                # four predeclared contrast families (98.75% two-sided, per seed and within-scene average) + the strong comparative advance
                panelP = {c: panel_scene(tables[c], Q.REPEAT_RESTORE_PANEL, 'all_program_questions') for c in tables}; multi = [p for p in Q.PROGRAMS if p != 'single']
                multiP = {c: panel_scene(tables[c], multi, 'all_program_questions') for c in tables}; harmP = {c: panel_scene(tables[c], ('AB', 'BA', 'ABC', 'CBA'), 'harm') for c in tables}
                def pc(P_, a, b, key):
                    o = {}
                    for s in ('s0', 's1'):
                        if f'{a}_{s}' in P_ and f'{b}_{s}' in P_: o[s] = Q.paired_bootstrap(P_[f'{a}_{s}'], P_[f'{b}_{s}'], key)
                    if 's0' in o and 's1' in o: o['avg'] = Q.paired_bootstrap(avg_scene(P_[f'{a}_s0'], P_[f'{a}_s1'], key), avg_scene(P_[f'{b}_s0'], P_[f'{b}_s1'], key), key)
                    return o
                base_free = 'FREE_OVERWRITE'; base_cont = 'CONTINUED_AFFINE' if actor == 'gemma' else 'AFFINE_CLAUSE'; canon = 'CANONICAL_V2' if actor == 'gemma' else 'CANONICAL_AFFINE'
                F2['contrasts'] = {'1_constrained_minus_free_repeat_restore_all_questions': pc(panelP, 'INVARIANT_SET', base_free, 'all_program_questions'), '2_constrained_minus_continued_repeat_restore_all_questions': pc(panelP, 'INVARIANT_SET', base_cont, 'all_program_questions'),
                                   '3_constrained_minus_free_two_three_edit_invariant_error': pc(harmP, 'INVARIANT_SET', base_free, 'harm'), '4_constrained_minus_canonical_multi_program_all_questions': pc(multiP, 'INVARIANT_SET', canon, 'all_program_questions'),
                                   'supplementary': {f'{a}_minus_{b}_repeat_restore': pc(panelP, a, b, 'all_program_questions') for a, b in ((base_free, base_cont), ('INVARIANT_SET', 'FROZEN_V2_CLAUSE'), ('INVARIANT_SET', 'FROZEN_V2_CLAUSE_FP32'), (canon, base_cont), (canon, 'FROZEN_V2_CLAUSE'), ('FROZEN_V2_CLAUSE_FP32', 'FROZEN_V2_CLAUSE')) if f'{a}_s0' in panelP and f'{b}_s0' in panelP}}
                F2['panel_means'] = {c: dict(repeat_restore_all_questions=float(np.mean([v['all_program_questions'] for v in panelP[c].values()])) if panelP[c] else None, multi_program_all_questions=float(np.mean([v['all_program_questions'] for v in multiP[c].values()])) if multiP[c] else None, two_three_edit_harm=float(np.mean([v['harm'] for v in harmP[c].values()])) if harmP[c] else None, scenes=len(panelP[c])) for c in tables}
                for c, v in F2['panel_means'].items(): T['T9_program_panels'].append(dict(actor=actor, condition=c, **v))
                def strong(s):
                    c1 = F2['contrasts']['1_constrained_minus_free_repeat_restore_all_questions'].get(s); c2 = F2['contrasts']['2_constrained_minus_continued_repeat_restore_all_questions'].get(s)
                    ni_f = NI['changed'].get(f'INVARIANT_SET_vs_{base_free}', {}).get(s); ni_c = NI['changed'].get(f'INVARIANT_SET_vs_{base_cont}', {}).get(s)
                    return dict(gain_free=c1 and c1['effect'], gain_continued=c2 and c2['effect'], met=bool(c1 and c2 and c1['effect'] >= .10 and c2['effect'] >= .10 and c1['lower'] > 0 and c2['lower'] > 0 and ni_f and ni_c and ni_f['noninferior'] and ni_c['noninferior']))
                F2['strong_comparative_advance'] = {s: strong(s) for s in ('s0', 's1', 'avg')}
                F2['dispositions'] = {c: dict(two_edit=l['two_edit'], three_edit=l['three_edit'], repeat=l['repeat_all'], restore=l['restore_all']) for c, l in F2['laws'].items()}
                for arm in sorted({c.rsplit('_s', 1)[0] for c in F2['laws']}):
                    ls = [F2['laws'][f'{arm}_s{s}'] for s in (0, 1) if f'{arm}_s{s}' in F2['laws']]
                    F2.setdefault('by_arm', {})[arm] = dict(two_edit_both_seeds=all(l['two_edit'] for l in ls), three_edit_both_seeds=all(l['three_edit'] for l in ls), repeat_both_seeds=all(l['repeat_all'] for l in ls), restore_both_seeds=all(l['restore_all'] for l in ls), seeds=len(ls))
            # workflow (F4)
            wfj = load(fp/'WORKFLOW.json'); F4 = dict(status='NOT_MEASURED')
            if wfj and wfj.get('status') == 'MEASURED':
                wr = rows(fp/'workflow'/'rows.jsonl'); by = defaultdict(list)
                for r in wr: by[(r['condition'], r['program'])].append(r)
                F4 = dict(status='MEASURED', episodes=len(wfj['scenes']), conditions={}, intervals={}, criteria={})
                for (c, p), rs in sorted(by.items()):
                    chg = [r for r in rs if r['majority_changes']]; inv = [r for r in rs if not r['majority_changes']]
                    e = dict(n=len(rs), final_decision_accuracy=float(np.mean([r['correct'] for r in rs])), all_receivers_correct=float(np.mean([r['receivers_all_correct'] for r in rs])), complete_episode_correct=float(np.mean([r['correct'] and r['receivers_all_correct'] for r in rs])), receiver_accuracy=float(np.mean([r['receiver_accuracy'] for r in rs])),
                             receiver_failures=int(sum(1 for r in rs if r['aggregator_status'] == 'RECEIVER_FAILED')), aggregator_unparsed=int(sum(1 for r in rs if r['aggregator_answer'] is None and r['aggregator_status'] != 'RECEIVER_FAILED')), correct_when_majority_changes=float(np.mean([r['correct'] for r in chg])) if chg else None,
                             correct_when_invariant=float(np.mean([r['correct'] for r in inv])) if inv else None, reports_reused=int(sum(1 for r in rs if r.get('reports_reused_from_identical_artifact'))))
                    F4['conditions'][f'{c}|{p}'] = e; T['T5_workflow'].append(dict(actor=actor, condition=c, program=p, **e))
                def pairs(a, b, p, key='complete'):
                    A = {r['scene_id']: float(r['correct'] and r['receivers_all_correct']) if key == 'complete' else float(r['correct']) for r in by.get((a, p), [])}; B = {r['scene_id']: float(r['correct'] and r['receivers_all_correct']) if key == 'complete' else float(r['correct']) for r in by.get((b, p), [])}; ks = sorted(set(A) & set(B))
                    if not ks: return None
                    r_ = Q.paired_bootstrap({k: dict(x=A[k]) for k in ks}, {k: dict(x=B[k]) for k in ks}, 'x', level=.95); r_['exact_binomial_p'] = _exact(A, B, ks); return r_
                base_cont = 'CONTINUED_AFFINE' if actor == 'gemma' else 'AFFINE_CLAUSE'; canon = 'CANONICAL_V2' if actor == 'gemma' else 'CANONICAL_AFFINE'
                for p in Q.WORKFLOW_PROGRAMS:
                    for s in (0, 1):
                        for b in (f'{base_cont}_s{s}', f'FREE_OVERWRITE_s{s}', f'FROZEN_V2_CLAUSE_s{s}', f'{canon}_s{s}', 'NATURAL_FINAL_WORLD', 'TEXT_CORRECTION_SEQ', 'SOURCE'):
                            if (f'INVARIANT_SET_s{s}', p) in by and (b, p) in by: F4['intervals'][f'INVARIANT_SET_s{s}_minus_{b}|{p}'] = pairs(f'INVARIANT_SET_s{s}', b, p)
                        for a in (f'{canon}_s{s}', f'FREE_OVERWRITE_s{s}'):
                            if (a, p) in by and (f'{base_cont}_s{s}', p) in by: F4['intervals'][f'{a}_minus_{base_cont}_s{s}|{p}'] = pairs(a, f'{base_cont}_s{s}', p)
                for s in (0, 1):
                    c = F4['conditions']; k = f'INVARIANT_SET_s{s}'
                    crit = dict(duplicate_delivery_final_ge_90=bool(c.get(f'{k}|repeat2', {}).get('final_decision_accuracy', 0) >= .90), duplicate_delivery_receivers_ge_90=bool(c.get(f'{k}|repeat2', {}).get('all_receivers_correct', 0) >= .90),
                                restoration_final_ge_90=bool(c.get(f'{k}|restore', {}).get('final_decision_accuracy', 0) >= .90), restoration_receivers_ge_90=bool(c.get(f'{k}|restore', {}).get('all_receivers_correct', 0) >= .90))
                    comp = {}
                    for p in Q.WORKFLOW_PROGRAMS:
                        g = F4['intervals'].get(f'{k}_minus_{base_cont}_s{s}|{p}'); nat = F4['intervals'].get(f'{k}_minus_NATURAL_FINAL_WORLD|{p}')
                        comp[p] = dict(gain_over_in_place_baseline=g and g['effect'], positive_interval=bool(g and g['lower'] > 0 and g['effect'] >= .10), loss_vs_native_le_5pp=bool(nat and nat['effect'] >= -.05))
                    F4['criteria'][k] = dict(crit, met=all(crit.values()), comparative=comp)
            ctl = load(fp/'CONTROLS.json'); C = dict(status='NOT_MEASURED')
            if ctl and ctl.get('status') == 'MEASURED':
                cr = {c: rows(fp/'controls'/f'{c}.jsonl') for c in ('LEARNED', 'RANDOM_MATCHED', 'WRONG_ADDRESS')}; wa = cr['WRONG_ADDRESS']
                C = dict(status='MEASURED', condition=ctl.get('condition'), learned=_strip(metrics(cr['LEARNED'])), random_matched=_strip(metrics(cr['RANDOM_MATCHED'])), wrong_address_vs_intended=_strip(metrics(wa)), wrong_address_vs_its_own_world=_strip(metrics([dict(r, gold=r['wrong_world_gold']) for r in wa])),
                         random_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in cr['RANDOM_MATCHED']])) if cr['RANDOM_MATCHED'] else None, wrong_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in wa])) if wa else None)
            wit = load(fp/'WITNESS_INDEX.json'); wchk = []
            if wit:
                for f in wit.get('files', []):
                    wp = Path(f['path']); wp = wp if wp.is_file() else fp/'witness'/wp.name
                    if not wp.is_file(): wchk.append(dict(condition=f['condition'], scene_id=f['scene_id'], status='ABSENT_LOCALLY')); continue
                    with np.load(wp, allow_pickle=False) as z:
                        post = z['post_block_residual']; nxt = z['next_block_input']; idx = z['edited_positions']; ed = z['edited_states_bf16']; mask = np.ones(post.shape[0], bool); mask[idx] = False
                        wchk.append(dict(condition=f['condition'], scene_id=f['scene_id'], unedited_equal=bool(np.array_equal(post[mask], nxt[mask])), edited_equal=bool(len(idx) == 0 or np.array_equal(nxt[idx], ed)), edited_tokens=int(len(idx)), max_edit_norm=(float(np.linalg.norm(ed-post[idx], axis=1).max()) if len(idx) else 0.0)))
            F = dict(status='MEASURED', scope=scope['scope'], sizes=scope['sizes'], seal=dict(artifacts=seal['artifacts'], sha256=seal['seal_sha256'], at=seal['at'], first_question_at=ev['first_question_generated_at'], mismatches=len(ev['seal_mismatches']), master_failures=ev['master_failures']),
                     conditions={c: _strip({k: v for k, v in m.items() if k != 'per_scene'}) for c, m in per.items()}, F1=F1, single_contrasts=CS, noninferiority=NI, F2=F2, F4=F4, controls=C,
                     witnesses=wit and dict(files=len(wit.get('files', [])), propagation_verified_locally=(bool(wchk) and all(w.get('unedited_equal') and w.get('edited_equal') for w in wchk)), checks=wchk, max_cast_error=max([f['cast_error'] for f in wit.get('files', [])] or [0])))
        elif (fp/'NATURAL_PANEL.json').is_file():
            npn = load(fp/'NATURAL_PANEL.json'); F = dict(status='NATURAL_PANEL_ONLY', procedure=npn.get('procedure'), conditions={c: _strip({k: v for k, v in metrics(rows(fp/'natural'/f'{c}.jsonl')).items() if k != 'per_scene'}) for c in npn.get('rows', {})})
        M['final'] = F; S['actors'][actor] = M
    for name, table in T.items(): write_csv(out/f'{name}.csv', table)
    (out/'SUMMARY.json').write_text(json.dumps(Q.json_safe(S), indent=1)+'\n', encoding='utf8')
    return S

def _exact(A, B, ks):
    from math import comb
    n1 = sum(1 for k in ks if A[k] > B[k]); n2 = sum(1 for k in ks if A[k] < B[k]); n = n1+n2
    if n == 0: return dict(discordant=0, p=1.0)
    k = min(n1, n2); p = min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n); return dict(discordant=n, better=n1, worse=n2, p=p)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--qdir', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    S = analyze(a.run, a.qdir, a.out); print(json.dumps({k: {kk: (vv.get('status') if isinstance(vv, dict) else vv) for kk, vv in v.items() if kk in ('F0', 'development', 'final')} for k, v in S['actors'].items()}, indent=1))
