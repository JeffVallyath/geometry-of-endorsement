"""SRS2 CPU analysis: reads unit outputs (mirror) and writes plot-ready tables, SUMMARY.json and milestone dispositions under the fixed
PROGRESS_CONTRACT rules.  No model forwards.  Every number comes from saved per-query rows or receipts."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import srs2_common as Q

def rows(path): return Q.read_rows(path) if Path(path).is_file() else []
def load(path): return Q.load(path) if Path(path).is_file() else None
def write_csv(path, table):
    if not table: Path(path).write_text(''); return
    keys = list(dict.fromkeys(k for r in table for k in r))
    with Path(path).open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in table: w.writerow({k: ('' if r.get(k) is None else (json.dumps(r[k]) if isinstance(r[k], (dict, list)) else r[k])) for k in keys})

def metrics(rs, cf_rows=None):
    if not rs: return None
    m = Q.summarize(rs); ps = m.pop('per_scene'); boot = Q.bootstrap_mean([v['all24'] for v in ps.values()])
    m.update(all24_interval=boot, changed_interval=Q.bootstrap_mean([v['changed'] for v in ps.values()]), harm_interval=Q.bootstrap_mean([v['harm'] for v in ps.values()]),
             by_order={o: Q.summarize([r for r in rs if r.get('order') == o])['all24'] for o in sorted({r.get('order') for r in rs})}, by_variant={v: Q.summarize([r for r in rs if r.get('variant') == v])['accuracy'] for v in sorted({r.get('variant') for r in rs})},
             edited_tokens=_mean(rs, 'edited_tokens'), energy=_mean(rs, 'energy'), frobenius=(_mean(rs, 'frobenius') if any(r.get('frobenius') is not None for r in rs) else (float(np.mean([r['energy']**.5 for r in rs if r.get('energy') is not None])) if any(r.get('energy') is not None for r in rs) else None)), realized_norm=_mean(rs, 'realized_norm'), cache_bytes=_mean(rs, 'cache_bytes'), prefix_tokens=_mean(rs, 'prefix_tokens'),
             ties=int(sum(1 for r in rs if r.get('tie'))), argmax_outside_labels=int(sum(1 for r in rs if r.get('argmax_in_labels') is False)), aware=bool(any(r.get('aware') for r in rs)))
    gen = [r for r in rs if 'generation' in r]
    m['generation'] = dict(rows=len(gen), parse_ok=float(np.mean([str(r.get('parse_status', '')).startswith('OK') for r in gen])), agrees_with_score=float(np.mean([r.get('generation_symbol') == r['labels'][r['prediction']] for r in gen]))) if gen else None
    if cf_rows:
        cf = Q.summarize(cf_rows); m['coherence_ratio'] = (m['all24']/cf['all24']) if cf['all24'] else None; m['per_scene'] = ps
    else: m['per_scene'] = ps
    return m

def _mean(rs, k):
    xs = [r[k] for r in rs if r.get(k) is not None]; return float(np.mean(xs)) if xs else None

def e1(m, cf, min_scenes):
    if m is None: return dict(status='NOT_MEASURED')
    checks = dict(changed_ge_85=m['changed'] is not None and m['changed'] >= .85, harm_le_5=m['harm'] is not None and m['harm'] <= .05, each_direction_ge_80=all(v is not None and v >= .80 for v in m['changed_by_direction'].values()),
                  all24_ge_70pct_of_reference=(m.get('coherence_ratio') or 0) >= .70, scenes_ge_min=m['n_scenes'] >= min_scenes)
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, changed=m['changed'], harm=m['harm'], all24=m['all24'], coherence_ratio=m.get('coherence_ratio'), directions=m['changed_by_direction'], n_scenes=m['n_scenes'])

def avg_scene(ps0, ps1, key):
    out = {}
    for k in ps0:
        if k in ps1 and ps0[k][key] is not None and ps1[k][key] is not None: out[k] = {key: (ps0[k][key]+ps1[k][key])/2}
    return out

def contrast(per, a, b, key, level=Q.CONTRAST_LEVEL):
    out = {}
    for seed in ('s0', 's1'):
        if f'{a}_{seed}' in per and f'{b}_{seed}' in per: out[seed] = Q.paired_bootstrap(per[f'{a}_{seed}']['per_scene'], per[f'{b}_{seed}']['per_scene'], key, level=level)
    if 's0' in out and 's1' in out: out['avg'] = Q.paired_bootstrap(avg_scene(per[f'{a}_s0']['per_scene'], per[f'{a}_s1']['per_scene'], key), avg_scene(per[f'{b}_s0']['per_scene'], per[f'{b}_s1']['per_scene'], key), key, level=level)
    return out

def analyze(run, qdir, out, store=None):
    run = Path(run); qdir = Path(qdir); out = Path(out); out.mkdir(parents=True, exist_ok=True); S = dict(at=Q.now(), actors={}); T = defaultdict(list)
    for actor in ('gemma', 'qwen'):
        M = dict(); q = qdir/actor; qual = load(q/'QUALIFICATION.json'); an = load(q/'ANCHORS.json'); ic = load(q/'INTERFACE_CHECK.json'); gc = load(q/'GRADIENT_CHECK.json'); bench = load(q/'BENCHMARK.json')
        if not qual and not (run/actor).is_dir(): continue
        M['qualification'] = qual and dict(selected=qual['selected_procedure'], natural=qual['natural_panel_procedure'], generations=qual['total_generations'], attempts=[dict(procedure=a['procedure'], qualified=a['qualified'], **{w: {k: v for k, v in a['summary'][w].items() if k not in ('n',)} for w in ('source', 'counterfactual')}) for a in qual['attempts']])
        if qual:
            for a in qual['attempts']:
                for w in ('source', 'counterfactual', 'pooled'): T['T1_native_qualification'].append(dict(actor=actor, procedure=a['procedure'], world=w, qualified=a['summary'][w]['qualified'], **{k: v for k, v in a['summary'][w].items() if k != 'qualified'}))
        M['anchors'] = an and dict(status=an.get('status'), seeds={k: {kk: vv for kk, vv in v.items() if kk != 'mismatches'} | dict(mismatches=len(v.get('mismatches', []))) for k, v in an.get('seeds', {}).items()}, tolerance=an.get('tolerance'))
        M['interface'] = ic and {k: v for k, v in ic.items() if k != 'rows'}; M['gradient'] = gc and dict(pass_=gc['pass_'], results={k: dict(output_factor=v['output_factor_grad_nonzero'], q_block=v['q_block_grad_nonzero'], backbone=v['backbone_grad'], move=v['max_score_move']) for k, v in gc['results'].items()})
        M['benchmark'] = bench and bench['seconds']
        M['E0'] = dict(status=('TECHNICAL_PASS' if (ic and ic['pass_'] and gc and gc['pass_'] and (not an or an.get('status') in ('TECHNICAL_PASS', 'NOT_APPLICABLE'))) else 'TECHNICAL_DEFECT'), anchors=an and an.get('status'), interface=ic and ic['pass_'], gradient=gc and gc['pass_'])
        d = run/actor; fp = d/'final_phase'; fz = load(d/'FREEZE.json'); M['development'] = dict(status='NOT_MEASURED')
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
                                                              cal_dir1=c['cal']['changed_by_direction'].get('1', c['cal']['changed_by_direction'].get(1)), cal_feasible=c['feasible'], frontier=c['frontier'], fit_changed=c['fit']['changed'], fit_harm=c['fit']['harm'], fit_loss=c['fit']['loss'], train_loss=c['loss_train'],
                                                              saturation=c['saturation'], realized_norm=c['norm'], frobenius=c['frobenius'], visits=c['visits'], steps=c['steps'], train_seconds=c['train_seconds'], eval_seconds=c['eval_seconds'], not_extended=dn.get('not_extended', False)))
                sel = ad['selection']; arms[arm] = dict(status='SELECTED', lr=sel['lr'], epoch=sel['epoch'], strength=sel['strength'], feasible=sel['feasible'], label=sel['label'], warm_start_unchanged=sel.get('warm_start_unchanged'), cal={s: {k: v for k, v in sel['cal'][s].items() if k not in ('per_scene', 'by_cell', 'by_family')} for s in sel['cal']},
                                 epochs=dict(seed0=ad['epochs_seed0'], seed1=ad['epochs_seed1']), opportunity=ad['opportunity'], init=ad['init'] and ad['init'].get('kind'))
            M['development'] = dict(status='COMPLETE', procedure=fz['procedure'], qualification_status=fz.get('qualification_status'), site=fz['site'], cap=fz['cap'], e_max=fz['e_max'], plan=plan and dict(e_max=plan['e_max'], predicted=plan['predicted_total_seconds'], dev_seconds=plan['dev_seconds']),
                                    baselines=base and dict(no_edit_cal=_strip(base['no_edit_cal']), counterfactual_cal=_strip(base['counterfactual_cal'])), captures=cap and dict(cap=cap['cap'], cap_new_rule=cap['cap_new_fit_rule'], delta_stats=cap['delta_stats'], mu_q_norm=cap['mu_q_norm']), arms=arms, not_measured=fz.get('not_measured'))
        F = dict(status='NOT_MEASURED'); per = {}
        if (fp/'EVAL_DONE.json').is_file():
            ev = load(fp/'EVAL_DONE.json'); seal = load(fp/'SEAL.json'); scope = load(fp/'SCOPE.json'); min_scenes = 64 if actor == 'gemma' else 32
            allrows = {c: rows(fp/'final'/f'{c}.jsonl') for c in ev['conditions']}; v0 = {c: [r for r in rs if r.get('variant') == 0] for c, rs in allrows.items()}; cf = v0.get('COUNTERFACTUAL', [])
            for c, rs in v0.items(): per[c] = metrics(rs, cf if c != 'COUNTERFACTUAL' else None)
            # family split (direct/negation vs equality vs held-out Boolean) and the competent-family diagnostic aggregate (additional, never a replacement endpoint)
            for c, rs in v0.items():
                fam = {f: float(np.mean([r['prediction'] == r['gold'] for r in rs if r['family'] == f])) for f in ('direct', 'opposes', 'same', 'both', 'either')}
                dn = [r for r in rs if r['family'] in ('direct', 'opposes')]; dm = Q.summarize(dn) if dn else None
                T['T6_family_split'].append(dict(actor=actor, condition=c, **{f'acc_{f}': v for f, v in fam.items()}, direct_negation_changed=dm and dm['changed'], direct_negation_harm=dm and dm['harm'], direct_negation_all_correct=dm and dm['all24'], equality_changed=Q.summarize([r for r in rs if r['family'] == 'same'])['changed'] if any(r['family'] == 'same' for r in rs) else None, boolean_changed=Q.summarize([r for r in rs if r['family'] in ('both', 'either')])['changed'] if any(r['family'] in ('both', 'either') for r in rs) else None))
            for c, rs in v0.items():
                for o in Q.ORDERS:
                    mo = Q.summarize([r for r in rs if r.get('order') == o]); T['T7_by_prefix_order'].append(dict(actor=actor, condition=c, order=o, changed=mo['changed'], harm=mo['harm'], all24=mo['all24'], accuracy=mo['accuracy'], n_scenes=mo['n_scenes']))
            for c, m in per.items(): T['T3_fresh_final'].append(dict(actor=actor, condition=c, scope=scope['scope'], **{k: v for k, v in m.items() if k not in ('per_scene', 'by_cell', 'by_family', 'by_order', 'by_variant', 'generation')}, by_family=m['by_family'], by_cell=m['by_cell'], by_order=m['by_order'], generation=m['generation']))
            E1 = {}; amended = (fz or {}).get('qualification_status', {}).get('status', 'ORIGINAL_PASS').startswith('AMENDED'); E1['qualification_status'] = (fz or {}).get('qualification_status')
            for arm in Q.ARMS+('V1_SHARED_TAIL', 'V1_LATE'):
                seeds = {s: e1(per.get(f'{arm}_{s}'), cf, min_scenes) for s in ('s0', 's1') if f'{arm}_{s}' in per}
                if not seeds: continue
                both = len(seeds) == 2 and all(v['status'] == 'PASS' for v in seeds.values())
                E1[arm] = dict(status=(('E1_NUMERICAL_CRITERIA_MET_UNDER_AMENDED_NATIVE_QUALIFICATION' if amended else 'PASS') if both else ('PARTIAL' if any(v['status'] == 'PASS' for v in seeds.values()) else 'FAIL')), seeds=seeds, question_blind=(not Q.AWARE.get(arm, False)) and arm != 'V1_LATE' and not ev['seal_mismatches'] and seal['questions_generated_before_seal'] is False)
            E1['natural'] = {c: e1(per.get(c), cf, min_scenes) for c in ('COUNTERFACTUAL', 'TEXT_CORRECTION') if c in per}
            # E2: four matched contrasts (98.75% two-sided, per seed and within-scene average) + changed-accuracy differences + noninferiority
            E2 = dict(contrasts={}, changed_differences={}, noninferiority={}, status='NOT_MEASURED')
            if all(f'{a}_s0' in per for a in Q.ARMS):
                for name, a, b, key in (('SHARED_TAIL_minus_SHARED_CLAUSE_all24', 'SHARED_TAIL', 'SHARED_CLAUSE', 'all24'), ('QUERY_TAIL_minus_QUERY_CLAUSE_all24', 'QUERY_TAIL', 'QUERY_CLAUSE', 'all24'),
                                        ('SHARED_TAIL_minus_QUERY_TAIL_all24', 'SHARED_TAIL', 'QUERY_TAIL', 'all24'), ('SHARED_TAIL_minus_QUERY_TAIL_harm', 'SHARED_TAIL', 'QUERY_TAIL', 'harm')):
                    E2['contrasts'][name] = contrast(per, a, b, key); E2['changed_differences'][name] = contrast(per, a, b, 'changed', level=.95)
                for a, b in (('SHARED_TAIL', 'SHARED_CLAUSE'), ('QUERY_TAIL', 'QUERY_CLAUSE'), ('SHARED_TAIL', 'V1_SHARED_TAIL'), ('SHARED_TAIL', 'V1_LATE'), ('SHARED_TAIL', 'COUNTERFACTUAL'), ('SHARED_TAIL', 'TEXT_CORRECTION')):
                    if f'{b}_s0' in per or b in per:
                        bb_ = b if b in per else None
                        E2['contrasts'].setdefault('supplementary', {})[f'{a}_minus_{b}_all24'] = (contrast(per, a, b, 'all24') if bb_ is None else {s: Q.paired_bootstrap(per[f'{a}_{s}']['per_scene'], per[b]['per_scene'], 'all24') for s in ('s0', 's1') if f'{a}_{s}' in per})
                # footprint benefit: >=10pp whole-scene advantage for tail over clause, positive corrected interval, <=5pp changed loss (each seed and the average)
                fb = {}
                for s in ('s0', 's1', 'avg'):
                    c = E2['contrasts']['SHARED_TAIL_minus_SHARED_CLAUSE_all24'].get(s); ch = E2['changed_differences']['SHARED_TAIL_minus_SHARED_CLAUSE_all24'].get(s)
                    fb[s] = bool(c and c['effect'] >= .10 and c['lower'] > 0 and ch and ch['effect'] >= -.05)
                # noninferiority SHARED vs QUERY: one-sided 97.5% (two-sided 95%) lower bound of SHARED - QUERY scene accuracy >= -5pp and harm difference upper bound <= +2pp
                ni = {}
                for s in ('s0', 's1', 'avg'):
                    acc = contrast(per, 'SHARED_TAIL', 'QUERY_TAIL', 'accuracy', level=.95).get(s); hm = contrast(per, 'SHARED_TAIL', 'QUERY_TAIL', 'harm', level=.95).get(s)
                    ni[s] = dict(scene_accuracy=acc, harm=hm, noninferior=bool(acc and hm and acc['lower'] >= -Q.NONINFERIORITY['scene'] and hm['upper'] <= Q.NONINFERIORITY['harm']))
                    sq = E2['contrasts']['SHARED_TAIL_minus_QUERY_TAIL_all24'].get(s); ni[s]['query_stronger'] = bool(sq and sq['upper'] < 0)
                E2.update(footprint_benefit=fb, noninferiority=ni, status='MEASURED')
            # energy-matched diagnostic
            en = load(fp/'ENERGY.json'); EN = dict(status='NOT_MEASURED')
            if en and en.get('status') == 'MEASURED':
                er = {c: rows(fp/'energy'/f'{c}.jsonl') for c in en['rows']}; em = {c: metrics(rs) for c, rs in er.items()}; meta = rows(fp/'energy'/'META.jsonl')
                EN = dict(status='MEASURED', zero_norm_cases=en['zero_norm_cases'], conditions={c: {k: m[k] for k in ('changed', 'harm', 'all24', 'accuracy', 'n_scenes', 'frobenius', 'energy')} for c, m in em.items()}, contrasts={})
                for s in Q.SEEDS:
                    for panel in ('unmodified', 'matched'):
                        a, b = f'SHARED_TAIL_s{s}_{panel}', f'SHARED_CLAUSE_s{s}_{panel}'
                        if a in em and b in em: EN['contrasts'][f'tail_minus_clause_all24_s{s}_{panel}'] = Q.paired_bootstrap(em[a]['per_scene'], em[b]['per_scene'], 'all24')
                EN['norms'] = dict(clause_mean=float(np.mean([m['clause_frobenius'] for m in meta])), tail_mean=float(np.mean([m['tail_frobenius'] for m in meta])), common_mean=float(np.mean([m['common'] for m in meta if m['common'] is not None])) if meta else None)
            # controls
            ctl = load(fp/'CONTROLS.json'); C = dict(status='NOT_MEASURED')
            if ctl and ctl.get('status') == 'MEASURED':
                cr = {c: rows(fp/'controls'/f'{c}.jsonl') for c in ('LEARNED', 'RANDOM_MATCHED', 'WRONG_ADDRESS')}; wa = cr['WRONG_ADDRESS']
                C = dict(status='MEASURED', learned=_strip(metrics(cr['LEARNED'])), random_matched=_strip(metrics(cr['RANDOM_MATCHED'])), wrong_address_vs_intended=_strip(metrics(wa)), wrong_address_vs_its_own_world=_strip(metrics([dict(r, gold=r['wrong_world_gold']) for r in wa])),
                         random_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in cr['RANDOM_MATCHED']])) if cr['RANDOM_MATCHED'] else None, wrong_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in wa])) if wa else None)
            # programs (E3)
            pg = load(fp/'PROGRAMS.json'); E3 = dict(status='NOT_MEASURED')
            if pg and pg.get('status') == 'MEASURED':
                pr = {c: rows(fp/'programs'/f'{c}.jsonl') for c in pg['rows']}; E3 = dict(status='MEASURED', conditions={}, laws={}, scenes=len(pg['scenes']))
                for c, rs in pr.items():
                    E3['conditions'][c] = {}
                    for program in Q.PROGRAMS:
                        sub = [r for r in rs if r.get('program') == program]
                        if not sub: continue
                        m = Q.summarize(sub); E3['conditions'][c][program] = dict(changed=m['changed'], harm=m['harm'], all24=m['all24'], accuracy=m['accuracy'], touched_accuracy=m['touched_accuracy'], preservation=m['preservation'], n_scenes=m['n_scenes'])
                        T['T4_edit_programs'].append(dict(actor=actor, condition=c, program=program, **E3['conditions'][c][program]))
                for c in pr:
                    if c in ('SOURCE',): continue
                    by_p = {p: {r['query_id']: r for r in pr[c] if r.get('program') == p} for p in Q.PROGRAMS}; laws = {}
                    for a, b in Q.EQUAL_TERMINAL:
                        ks = [k for k in by_p[a] if k in by_p[b]]
                        if not ks: continue
                        dis = float(np.mean([by_p[a][k]['prediction'] != by_p[b][k]['prediction'] for k in ks])); both_wrong = float(np.mean([by_p[a][k]['prediction'] != by_p[a][k]['gold'] and by_p[b][k]['prediction'] != by_p[b][k]['gold'] for k in ks]))
                        laws[f'{a}_vs_{b}'] = dict(output_disagreement=dis, both_wrong=both_wrong, n=len(ks), a_changed=E3['conditions'][c].get(a, {}).get('changed'), b_changed=E3['conditions'][c].get(b, {}).get('changed'), a_harm=E3['conditions'][c].get(a, {}).get('harm'), b_harm=E3['conditions'][c].get(b, {}).get('harm'))
                    E3['laws'][c] = laws
                def law_pass(c, seedcond):
                    e = E3['conditions'].get(c, {}); l = E3['laws'].get(c, {})
                    return dict(two_edit=bool(all(e.get(p, {}).get('changed', 0) is not None and e.get(p, {}).get('changed', 0) >= .80 and e.get(p, {}).get('harm', 1) <= .05 for p in ('AB', 'BA')) and l.get('AB_vs_BA', {}).get('output_disagreement', 1) <= .05),
                                three_edit=bool(all(e.get(p, {}).get('changed', 0) is not None and e.get(p, {}).get('changed', 0) >= .80 and e.get(p, {}).get('harm', 1) <= .05 for p in ('ABC', 'CBA')) and l.get('ABC_vs_CBA', {}).get('output_disagreement', 1) <= .05),
                                idempotence=bool(l.get('single_vs_repeat', {}).get('output_disagreement', 1) <= .05 and e.get('repeat', {}).get('changed', 0) is not None and e.get('repeat', {}).get('changed', 0) >= .80 and e.get('repeat', {}).get('harm', 1) <= .05),
                                restore=bool((e.get('restore', {}).get('preservation') or 0) >= .95), noop=bool((e.get('noop', {}).get('preservation') or 0) >= .95))
                E3['dispositions'] = {c: law_pass(c, None) for c in E3['conditions'] if c not in ('SOURCE', 'NATURAL_FINAL_WORLD')}
            # workflow (E5)
            wfj = load(fp/'WORKFLOW.json'); E5 = dict(status='NOT_MEASURED')
            if wfj and wfj.get('status') == 'MEASURED':
                wr = rows(fp/'workflow'/'rows.jsonl'); by = defaultdict(list)
                for r in wr: by[(r['condition'], r['phrasing'])].append(r)
                E5 = dict(status='MEASURED', episodes=len(wfj['scenes']), conditions={}, intervals={})
                for (c, ph), rs in sorted(by.items()):
                    chg = [r for r in rs if r['majority_changes']]; inv = [r for r in rs if not r['majority_changes']]
                    E5['conditions'][f'{c}|p{ph}'] = dict(n=len(rs), chain_correct=float(np.mean([r['correct'] for r in rs])), chain_correct_vs_target=float(np.mean([r['correct_vs_target'] for r in rs])), receiver_accuracy=float(np.mean([r['receiver_accuracy'] for r in rs])),
                                                       all_receivers_correct=float(np.mean([r['receivers_all_correct'] for r in rs])), receiver_failures=int(sum(1 for r in rs if r['aggregator_status'] == 'RECEIVER_FAILED')), aggregator_unparsed=int(sum(1 for r in rs if r['aggregator_answer'] is None and r['aggregator_status'] != 'RECEIVER_FAILED')),
                                                       correct_when_majority_changes=float(np.mean([r['correct'] for r in chg])) if chg else None, correct_when_invariant=float(np.mean([r['correct'] for r in inv])) if inv else None, needless_change=float(np.mean([r['chain_output'] != r['gold_source'] for r in inv])) if inv else None)
                    T['T5_workflow'].append(dict(actor=actor, condition=c, phrasing=ph, **E5['conditions'][f'{c}|p{ph}']))
                def pairs(a, b, ph):
                    A = {r['scene_id']: float(r['correct_vs_target']) for r in by.get((a, ph), [])}; B = {r['scene_id']: float(r['correct_vs_target']) for r in by.get((b, ph), [])}; ks = sorted(set(A) & set(B))
                    if not ks: return None
                    r_ = Q.paired_bootstrap({k: dict(x=A[k]) for k in ks}, {k: dict(x=B[k]) for k in ks}, 'x', level=.95); r_['exact_binomial_p'] = _exact(A, B, ks); return r_
                for ph in (0, 1):
                    for a, b in (('SHARED_TAIL_s0', 'V1_LATE_s0'), ('SHARED_TAIL_s1', 'V1_LATE_s1'), ('SHARED_TAIL_s0', 'COUNTERFACTUAL'), ('SHARED_TAIL_s1', 'COUNTERFACTUAL'), ('SHARED_TAIL_s0', 'QUERY_TAIL_s0'), ('SHARED_TAIL_s1', 'QUERY_TAIL_s1'), ('SHARED_TAIL_s0', 'TEXT_CORRECTION'), ('SHARED_TAIL_s0', 'SOURCE'), ('QUERY_TAIL_s0', 'V1_LATE_s0')):
                        if (a, ph) in by and (b, ph) in by: E5['intervals'][f'{a}_minus_{b}|p{ph}'] = pairs(a, b, ph)
            wit = load(fp/'WITNESS_INDEX.json')
            F = dict(status='MEASURED', scope=scope['scope'], sizes=scope['sizes'], seal=dict(artifacts=seal['artifacts'], sha256=seal['seal_sha256'], at=seal['at'], first_question_at=ev['first_question_generated_at'], mismatches=len(ev['seal_mismatches']), master_failures=ev['master_failures']),
                     conditions={c: {k: v for k, v in m.items() if k != 'per_scene'} for c, m in per.items()}, E1=E1, E2=E2, energy=EN, controls=C, E3=E3, E5=E5, witnesses=wit and dict(files=len(wit.get('files', [])), all_next_input_equal=all(f['next_input_equals_post_block'] for f in wit.get('files', []))),
                     variant1={c: metrics([r for r in rs if r.get('variant') == 1]) for c, rs in allrows.items() if any(r.get('variant') == 1 for r in rs)})
            for c in list(F['variant1']): F['variant1'][c] = _strip(F['variant1'][c])
        elif (fp/'NATURAL_PANEL.json').is_file():
            npn = load(fp/'NATURAL_PANEL.json'); F = dict(status='NATURAL_PANEL_ONLY', procedure=npn.get('procedure'), conditions={c: _strip(metrics(rows(fp/'natural'/f'{c}.jsonl'))) for c in npn.get('rows', {})})
        M['final'] = F; S['actors'][actor] = M
    for name, table in T.items(): write_csv(out/f'{name}.csv', table)
    Q.dump(out/'SUMMARY.json', S) if not (out/'SUMMARY.json').exists() else (out/'SUMMARY.json').write_text(json.dumps(Q.json_safe(S), indent=1)+'\n')
    return S

def _strip(m):
    if not m: return m
    return {k: v for k, v in m.items() if k not in ('per_scene', 'by_cell')}

def _exact(A, B, ks):
    """Exact paired sign test p-value (two-sided) on discordant scenes."""
    from math import comb
    n1 = sum(1 for k in ks if A[k] > B[k]); n2 = sum(1 for k in ks if A[k] < B[k]); n = n1+n2
    if n == 0: return dict(discordant=0, p=1.0)
    k = min(n1, n2); p = min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n); return dict(discordant=n, better=n1, worse=n2, p=p)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--qdir', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    S = analyze(a.run, a.qdir, a.out); print(json.dumps({k: {kk: (vv.get('status') if isinstance(vv, dict) else vv) for kk, vv in v.items() if kk in ('E0', 'development', 'final')} for k, v in S['actors'].items()}, indent=1))
