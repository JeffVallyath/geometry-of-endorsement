"""QBRC196 CPU analysis: reads the mirrored unit outputs and writes the five machine-readable tables, SUMMARY.json and RESULTS.md.

No model forwards.  Every number comes from saved per-query rows; milestone dispositions apply the fixed PROGRESS_CONTRACT rules.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import qbrc_common as Q

def rows(path): return Q.read_rows(path) if Path(path).is_file() else []
def load(path): return Q.load(path) if Path(path).is_file() else None
def pct(x): return 'n/a' if x is None else f'{100*x:.1f}%'
def pp(x): return 'n/a' if x is None else f'{100*x:+.1f}pp'

def write_csv(path, table):
    if not table: Path(path).write_text(''); return
    keys = list(dict.fromkeys(k for r in table for k in r))
    with Path(path).open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in table: w.writerow({k: ('' if r.get(k) is None else (json.dumps(r[k]) if isinstance(r[k], (dict, list)) else r[k])) for k in keys})

# ---------------------------------------------------------------- per-condition metrics
def metrics(rs, cf_rows=None):
    if not rs: return None
    summ = Q.score_rows(rs); f = Q.frontier_rows(rs); task = rs[0]['task']; ft = f[task]
    def acc(sub): return float(np.mean([r['prediction'] == r['gold'] for r in sub])) if sub else None
    by_family = {fam: acc([r for r in rs if r['family'] == fam]) for fam in sorted({r['family'] for r in rs})}
    by_draw = {d: acc([r for r in rs if r['draw'] == d]) for d in (0, 1)}
    by_actors = {n: acc([r for r in rs if r['n_actors'] == n]) for n in sorted({r['n_actors'] for r in rs})}
    by_options = {n: acc([r for r in rs if r.get('n_options') == n]) for n in sorted({r.get('n_options') for r in rs if r.get('n_options')})}
    changed_by_actors = {n: Q.score_rows([r for r in rs if r['n_actors'] == n])['changed'] for n in sorted({r['n_actors'] for r in rs})}
    harm_by_actors = {n: Q.score_rows([r for r in rs if r['n_actors'] == n])['invariant_harm'] for n in sorted({r['n_actors'] for r in rs})}
    changed_by_draw = {d: Q.score_rows([r for r in rs if r['draw'] == d])['changed'] for d in (0, 1)}
    harm_by_draw = {d: Q.score_rows([r for r in rs if r['draw'] == d])['invariant_harm'] for d in (0, 1)}
    gen = [r for r in rs if 'generation' in r]
    out = dict(task=task, n_scenes=summ['n_scenes'], n_queries=summ['n_queries'], accuracy=summ['accuracy'], changed=ft['changed'], invariant_harm=ft['invariant_harm'], conditional_harm=ft['conditional_harm'],
               all_correct=summ['all_correct'], false_nonaddressed_change=ft['false_nonaddressed_change'], changed_dir0=ft['changed_by_direction'][0], changed_dir1=ft['changed_by_direction'][1],
               by_family=by_family, by_draw=by_draw, changed_by_draw=changed_by_draw, harm_by_draw=harm_by_draw, by_actors=by_actors, changed_by_actors=changed_by_actors, harm_by_actors=harm_by_actors, by_options=by_options,
               answer_mass=float(np.mean([r['answer_mass'] for r in rs if r.get('answer_mass') is not None])) if any(r.get('answer_mass') is not None for r in rs) else None,
               ties=int(sum(1 for r in rs if r.get('tie'))), argmax_outside_labels=int(sum(1 for r in rs if r.get('argmax_in_labels') is False)),
               generation_rows=len(gen), generation_parse_ok=float(np.mean([str(r.get('parse_status', '')).startswith('OK') for r in gen])) if gen else None,
               generation_agrees_with_score=float(np.mean([r.get('generation_symbol') == r['labels'][r['prediction']] for r in gen])) if gen else None,
               mean_realized_norm=float(np.mean([r['realized_norm'] for r in rs if r.get('realized_norm') is not None])) if any(r.get('realized_norm') is not None for r in rs) else None,
               edited_tokens=float(np.mean([r['edited_tokens'] for r in rs if r.get('edited_tokens') is not None])) if any(r.get('edited_tokens') is not None for r in rs) else None,
               energy=float(np.mean([r['energy'] for r in rs if r.get('energy') is not None])) if any(r.get('energy') is not None for r in rs) else None,
               cache_bytes=float(np.mean([r['cache_bytes'] for r in rs if r.get('cache_bytes') is not None])) if any(r.get('cache_bytes') is not None for r in rs) else None,
               per_scene=summ['per_scene'])
    if cf_rows: out['coherence_ratio_vs_counterfactual'] = (summ['all_correct']/Q.score_rows(cf_rows)['all_correct']) if Q.score_rows(cf_rows)['all_correct'] else None
    return out

def e1(m, cf, seeds_pass=None):
    if m is None: return dict(status='NOT_MEASURED')
    checks = dict(changed_ge_85=m['changed'] is not None and m['changed'] >= .85, harm_le_5=m['invariant_harm'] is not None and m['invariant_harm'] <= .05,
                  both_directions_ge_80=all(v is not None and v >= .80 for v in (m['changed_dir0'], m['changed_dir1'])), scenes_ge_32=m['n_scenes'] >= 32,
                  coherence_ge_70pct_of_reference=(m.get('coherence_ratio_vs_counterfactual') or 0) >= .70)
    if seeds_pass is not None: checks['both_seeds'] = seeds_pass
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks)

# ---------------------------------------------------------------- main
def analyze(run, qdir, out, shutdown=None):
    run = Path(run); qdir = Path(qdir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    S = dict(at=Q.now(), models={}); T1, T2, T3, T4, T5 = [], [], [], [], []
    for model in ('gemma', 'llama'):
        M = dict(); qual = load(qdir/model/'QUALIFICATION.json'); iface = load(qdir/model/'INTERFACE_CHECK.json'); grad = load(qdir/model/'GRADIENT_CHECK.json'); bench = load(qdir/model/'BENCHMARK.json')
        M['qualification'] = qual and dict(selected=qual['selected_procedures'], natural=qual.get('natural_panel_procedures'), generations=qual.get('total_generations'), extra=qual.get('extra_generations'), attempts=[dict(procedure=a['procedure'], summary={t: {k: v for k, v in s.items() if k != 'n'} for t, s in a['summary'].items()}) for a in qual.get('attempts', [])])
        M['interface'] = iface and dict(pass_=iface['pass_'], cached_vs_full=iface['cached_vs_full'], counterfactual=iface['counterfactual_reference'], zero=iface['zero_self_edit_hash_identical'], clone=iface['clone_hash_identical'], order=iface['order_independent'], recompile=iface['recompile_deterministic'], master=iface['master_unchanged_every_ask'], prefix_tokens=iface['rows'][0]['prefix_tokens'], cache_bytes=iface['rows'][0]['cache_bytes'], serving_path=iface['serving_path'])
        M['gradient'] = grad and dict(pass_=grad['pass_'], results={k: dict(output_factor_grad=v['grad_norms']['output_factor'], backbone_grad=v['backbone_grad'], score_move=v['max_score_move']) for k, v in grad['results'].items()})
        M['benchmark'] = bench and bench['seconds']
        if qual:
            for a in qual.get('attempts', []):
                for t, s in a['summary'].items(): T1.append(dict(model=model, task=t, procedure=a['procedure'], parse=s['parse_rate'], changed_type_acc=s['changed_type_accuracy'], invariant_type_acc=s['invariant_type_accuracy'], qualified=s['qualified'], mean_new_tokens=s['mean_new_tokens']))
        d = run/model; fp = d/'final_phase'
        M['development'] = dict(status='NOT_MEASURED')
        if (d/'FREEZE.json').is_file():
            fz = load(d/'FREEZE.json'); plan = load(d/'DEV_PLAN.json'); base = load(d/'BASELINES.json'); cap = load(d/'CAPTURES.json')
            cands = {p.parent.name: load(p) for p in (d/'candidates').glob('*/DONE.json')}
            for name, c in sorted(cands.items()):
                for r in c['curve']: T2.append(dict(model=model, candidate=name, family=r['family'], site=r['site'], lr=r['lr'], seed=r['seed'], epoch=r['epoch'], strength=r['strength'], cal_score=r['score'], cal_changed=r['changed'], cal_harm=r['harm'], cal_all_correct=r['per_task'].get('relation', {}).get('all_correct') if 'relation' in r['per_task'] else None, fit_score=r['fit']['mean_score'], fit_loss=r['fit']['loss'], train_loss=r['loss_train'], saturation=r['saturation'], realized_norm=r['norm'], visits=r['visits'], steps=r['steps']))
            for r in fz['late_mean']['curve']: T2.append(dict(model=model, candidate='LATE_MEAN', family='LATE_MEAN', site=fz['late_mean']['site'], lr=None, seed=None, epoch=0, strength=r['strength'], cal_score=r['score'], cal_changed=r['changed'], cal_harm=r['harm'], realized_norm=r['norm']))
            M['development'] = dict(status='COMPLETE', procedures=fz['procedures'], caps=fz['caps'], cap_state=fz['cap_state'], plan=dict(epochs=plan['epochs_per_candidate'], candidates=plan['candidates'], remaining=plan['remaining_dev_seconds']), baselines=dict(no_edit_cal=base['no_edit_cal'].get('relation'), counterfactual_cal=base['counterfactual_cal'].get('relation')),
                                    selection={f: ({k: v for k, v in s.items() if k not in ('per_task', 'fit', 'checkpoint')} if s.get('status') == 'SELECTED' else s) for f, s in fz['selection'].items()}, late_mean=fz['late_mean']['selected'], candidates={k: dict(epochs=v['epochs_run'], visits=v['visits'], steps=v['steps'], cap_scale=v['cap_scale'], saturation=v['saturation_final']) for k, v in cands.items()}, late_opportunity=fz.get('late_opportunity'), delta_stats=cap['delta_stats'])
        # final
        F = dict(status='NOT_MEASURED'); conds = {}
        if (fp/'EVAL_DONE.json').is_file():
            ev = load(fp/'EVAL_DONE.json'); seal = load(fp/'SEAL.json'); scope = load(fp/'SCOPE.json')
            per = {c: rows(fp/'final'/f'{c}.jsonl') for c in ev['conditions']}; cf = per.get('COUNTERFACTUAL', [])
            for c, rs in per.items(): conds[c] = metrics(rs, cf if c != 'COUNTERFACTUAL' else None)
            seeds = defaultdict(dict)
            for c in conds:
                if c.endswith('_s0') or c.endswith('_s1'): seeds[c[:-3]][c[-2:]] = conds[c]
            E1 = {}
            for fam, ss in seeds.items():
                each = {s: e1(m, cf) for s, m in ss.items()}; both = all(v['status'] == 'PASS' for v in each.values()) and len(each) == 2
                E1[fam] = dict(seeds=each, both_seeds=both, question_blind=(fam.startswith('PREFIX') and not ev['seal_mismatches'] and ev['master_failures'].get(fam+'_s0', 0) == 0 and seal['questions_generated_before_seal'] is False),
                               status='PASS' if both else ('PARTIAL' if any(v['status'] == 'PASS' for v in each.values()) else 'FAIL'))
            E1['LATE_MEAN'] = dict(seeds={'s0': e1(conds.get('LATE_MEAN'), cf)}, status=e1(conds.get('LATE_MEAN'), cf)['status'], question_blind=False)
            # E2: best prefix vs LATE, paired 97.5% intervals on all_correct and changed
            fz = load(d/'FREEZE.json'); pref = [f for f in ('PREFIX_LAST', 'PREFIX_DISTRIBUTED') if f'{f}_s0' in per]
            bestp = min(pref, key=lambda f: -fz['selection'][f]['score']) if pref else None; E2 = dict(status='NOT_MEASURED')
            if bestp and 'LATE_s0' in per:
                ivs = {}
                for seed in ('s0', 's1'):
                    if f'{bestp}_{seed}' in per and f'LATE_{seed}' in per:
                        ivs[seed] = dict(all_correct=Q.paired_interval(per[f'{bestp}_{seed}'], per[f'LATE_{seed}'], 'all_correct'), changed=Q.paired_interval(per[f'{bestp}_{seed}'], per[f'LATE_{seed}'], 'changed'), invariant_harm=Q.paired_interval(per[f'{bestp}_{seed}'], per[f'LATE_{seed}'], 'invariant_harm'))
                def ok(v): return bool(v) and v['all_correct']['effect'] >= .15 and v['all_correct']['lower'] > 0 and v['changed']['effect'] >= -.05
                per_seed = {s: ok(v) for s, v in ivs.items()}; late_e1 = E1.get('LATE', {}).get('status') == 'PASS'
                status = 'PREFIX_ADVANTAGE' if per_seed and all(per_seed.values()) else ('PREFIX_ADVANTAGE_ONE_SEED' if any(per_seed.values()) else ('LATE_CREDITED' if late_e1 else 'FAIL'))
                E2 = dict(status=status, per_seed=per_seed, best_prefix=bestp, intervals=ivs, late_passes_e1=late_e1, note='one primary task comparison only (priority not qualified); the contract asked for two; the point criteria are applied per seed and both seeds are reported')
            # controls
            ctl = load(fp/'CONTROLS.json'); C = dict(status='NOT_MEASURED')
            if ctl and ctl.get('status') == 'MEASURED':
                cr = {c: rows(fp/'controls'/f'{c}.jsonl') for c in ('LEARNED', 'RANDOM_MATCHED', 'WRONG_ADDRESS')}; wa = cr['WRONG_ADDRESS']
                wrong_world = [dict(r, gold=r['wrong_world_gold']) for r in wa]
                C = dict(status='MEASURED', family=ctl['family'], learned=metrics(cr['LEARNED']), random_matched=metrics(cr['RANDOM_MATCHED']), wrong_address_vs_intended=metrics(wa), wrong_address_vs_its_own_world=metrics(wrong_world),
                         wrong_address_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in wa])) if wa else None, random_change_rate=float(np.mean([r['prediction'] != r['source_prediction'] for r in cr['RANDOM_MATCHED']])) if cr['RANDOM_MATCHED'] else None)
                for k in ('learned', 'random_matched', 'wrong_address_vs_intended', 'wrong_address_vs_its_own_world'): C[k].pop('per_scene', None)
            # composition
            comp = load(fp/'COMPOSITION.json'); CO = dict(status='NOT_MEASURED')
            if comp and comp.get('status') == 'MEASURED':
                crs = {c: rows(fp/'composition'/f'{c}.jsonl') for c in comp['rows'] if c != 'META'}; CO = dict(status='MEASURED', family=comp['family'], conditions={})
                for c, rs in crs.items():
                    if not rs: continue
                    ch = [r for r in rs if r['gold'] != r['source_gold']]; inv = [r for r in rs if r['gold'] == r['source_gold']]
                    CO['conditions'][c] = dict(endpoint_accuracy=float(np.mean([r['prediction'] == r['gold'] for r in ch])) if ch else None, jointly_invariant_damage=float(np.mean([r['source_prediction'] == r['gold'] and r['prediction'] != r['gold'] for r in inv])) if inv else None,
                                               accuracy=float(np.mean([r['prediction'] == r['gold'] for r in rs])), all_correct=Q.score_rows(rs)['all_correct'], n=len(rs), n_changed=len(ch))
                for base in {c[:-3] for c in crs if c.endswith('_AB')}:
                    a = {r['query_id']: r['prediction'] for r in crs.get(base+'_AB', [])}; b = {r['query_id']: r['prediction'] for r in crs.get(base+'_BA', [])}
                    CO['conditions'][base+'_order_disagreement'] = float(np.mean([a[k] != b[k] for k in a if k in b])) if a and b else None
            # downstream
            dn = load(fp/'DOWNSTREAM.json'); D = dict(status='NOT_MEASURED')
            if dn and dn.get('status') == 'MEASURED':
                dr = rows(fp/'downstream'/'rows.jsonl'); D = dict(status='MEASURED', family=dn['family'], conditions={}, intervals={})
                by = defaultdict(list)
                for r in dr: by[r['condition']].append(r)
                for c, rs in by.items():
                    chg = [r for r in rs if r['joint_changes']]; invr = [r for r in rs if not r['joint_changes']]
                    D['conditions'][c] = dict(n=len(rs), chain_correct_vs_target=float(np.mean([r['correct_vs_target'] for r in rs])), receiver_failures=int(sum(1 for r in rs if r['aggregator_status'] == 'RECEIVER_FAILED')), aggregator_unparsed=int(sum(1 for r in rs if r['aggregator_answer'] is None and r['aggregator_status'] != 'RECEIVER_FAILED')),
                                               correct_when_joint_changes=float(np.mean([r['correct_vs_target'] for r in chg])) if chg else None, correct_when_invariant=float(np.mean([r['correct_vs_target'] for r in invr])) if invr else None,
                                               needless_change=float(np.mean([r['chain_output'] != r['gold_source'] for r in invr])) if invr else None)
                def pairs(a, b):
                    A = {r['scene_id']: float(r['correct_vs_target']) for r in by.get(a, [])}; B = {r['scene_id']: float(r['correct_vs_target']) for r in by.get(b, [])}; ks = sorted(set(A) & set(B))
                    if not ks: return None
                    dd = np.array([A[k]-B[k] for k in ks]); rng = np.random.default_rng(Q.BOOT_SEED); o = np.array([dd[rng.integers(len(dd), size=len(dd))].mean() for _ in range(Q.BOOT_DRAWS)])
                    return dict(effect=float(dd.mean()), lower=float(np.quantile(o, .0125)), upper=float(np.quantile(o, .9875)), scenes=len(ks), level=.975)
                pc = [c for c in by if c.startswith('PREFIX')]; lc = [c for c in by if c.startswith('LATE_s')]
                if pc and lc: D['intervals']['prefix_minus_late'] = pairs(pc[0], lc[0]); D['intervals']['prefix_minus_counterfactual'] = pairs(pc[0], 'COUNTERFACTUAL'); D['intervals']['late_minus_counterfactual'] = pairs(lc[0], 'COUNTERFACTUAL')
                iv = D['intervals'].get('prefix_minus_late'); ivc = D['intervals'].get('prefix_minus_counterfactual')
                D['E5'] = dict(status='PASS' if iv and iv['effect'] >= .10 and iv['lower'] > 0 and ivc and ivc['effect'] >= -.05 else 'FAIL', prefix_minus_late=iv, prefix_minus_counterfactual=ivc) if iv else dict(status='NOT_MEASURED')
            npn = load(fp/'NATURAL_PANEL.json'); NP = {}
            if npn and npn.get('status') == 'MEASURED':
                for c in npn['rows']: NP[c] = {k: v for k, v in (metrics(rows(fp/'natural'/f'{c}.jsonl')) or {}).items() if k != 'per_scene'}
            F = dict(status='MEASURED', scope=scope['scope'], scenes=ev['scenes'], seal=dict(at=seal['at'], sha256=seal['seal_sha256'], first_question_at=ev['first_question_generated_at'], mismatches=len(ev['seal_mismatches']), recompile_deterministic=ev['recompile_deterministic'], master_failures=ev['master_failures'], master_samples=ev['master_verified_samples']),
                     conditions={c: {k: v for k, v in m.items() if k != 'per_scene'} for c, m in conds.items()}, E1=E1, E2=E2, controls=C, composition=CO, downstream=D, natural_panel=NP)
            for c, m in conds.items(): T3.append(dict(model=model, task=m['task'], condition=c, scenes=m['n_scenes'], queries=m['n_queries'], accuracy=m['accuracy'], changed=m['changed'], invariant_harm=m['invariant_harm'], conditional_harm=m['conditional_harm'], all_correct=m['all_correct'], false_nonaddressed_change=m['false_nonaddressed_change'], changed_dir0=m['changed_dir0'], changed_dir1=m['changed_dir1'], by_family=m['by_family'], by_draw=m['by_draw'], changed_by_draw=m['changed_by_draw'], harm_by_draw=m['harm_by_draw'], by_actors=m['by_actors'], changed_by_actors=m['changed_by_actors'], harm_by_actors=m['harm_by_actors'], answer_mass=m['answer_mass'], ties=m['ties'], argmax_outside_labels=m['argmax_outside_labels'], generation_parse_ok=m['generation_parse_ok'], generation_agrees=m['generation_agrees_with_score'], realized_norm=m['mean_realized_norm'], edited_tokens=m['edited_tokens'], energy=m['energy'], cache_bytes=m['cache_bytes'], coherence_ratio=m.get('coherence_ratio_vs_counterfactual')))
            for c, m in NP.items(): T3.append(dict(model=model, task=m['task'], condition='NATURAL_'+c, scenes=m['n_scenes'], queries=m['n_queries'], accuracy=m['accuracy'], changed=m['changed'], invariant_harm=m['invariant_harm'], all_correct=m['all_correct'], by_family=m['by_family'], by_actors=m['by_actors'], by_options=m.get('by_options')))
            if C.get('status') == 'MEASURED':
                for k in ('learned', 'random_matched', 'wrong_address_vs_intended', 'wrong_address_vs_its_own_world'): T3.append(dict(model=model, task=C[k]['task'], condition='CONTROL_'+k, scenes=C[k]['n_scenes'], queries=C[k]['n_queries'], accuracy=C[k]['accuracy'], changed=C[k]['changed'], invariant_harm=C[k]['invariant_harm'], all_correct=C[k]['all_correct'], realized_norm=C[k]['mean_realized_norm']))
            if CO.get('status') == 'MEASURED':
                for c, v in CO['conditions'].items(): T4.append(dict(model=model, panel='composition', condition=c, **(v if isinstance(v, dict) else dict(order_disagreement=v))))
            if D.get('status') == 'MEASURED':
                for c, v in D['conditions'].items(): T4.append(dict(model=model, panel='downstream', condition=c, **v))
                for k, v in D['intervals'].items(): T4.append(dict(model=model, panel='downstream_interval', condition=k, **(v or {})))
        elif (fp/'NATURAL_PANEL.json').is_file():
            npn = load(fp/'NATURAL_PANEL.json'); NP = {c: {k: v for k, v in (metrics(rows(fp/'natural'/f'{c}.jsonl')) or {}).items() if k != 'per_scene'} for c in npn['rows']}
            F = dict(status='NATURAL_PANEL_ONLY', natural_panel=NP)
            for c, m in NP.items(): T3.append(dict(model=model, task=m['task'], condition='NATURAL_'+c, scenes=m['n_scenes'], queries=m['n_queries'], accuracy=m['accuracy'], changed=m['changed'], invariant_harm=m['invariant_harm'], all_correct=m['all_correct'], by_family=m['by_family'], by_actors=m['by_actors'], by_options=m.get('by_options')))
        M['final'] = F; S['models'][model] = M
        term = load(d/'MODEL_TERMINAL.json')
        if term: T5.append(dict(model=model, unit='r3', forwards=term['forwards'], backwards=term['backwards'], backbone_unchanged=term['backbone']['unchanged']))
    for unit in ('r1', 'r2'):
        t = load(run.parent/unit/'MEASUREMENTS_TERMINAL.json')
        if t: T5.append(dict(model=t['models'], unit=unit, forwards=t['model_forwards'], backwards=t['backward_passes'], status=t['status'], seconds=t['elapsed_seconds']))
    write_csv(out/'T1_native_qualification.csv', T1); write_csv(out/'T2_development_curves.csv', T2); write_csv(out/'T3_fresh_scope_coherence.csv', T3); write_csv(out/'T4_composition_downstream.csv', T4); write_csv(out/'T5_timing_custody.csv', T5)
    (out/'SUMMARY.json').write_text(json.dumps(Q.json_safe(S), indent=1)+'\n')
    return S

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--q', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); S = analyze(a.run, a.q, a.output)
    g = S['models']['gemma']; print(json.dumps(Q.json_safe(dict(final=g['final'].get('status'), E1=g['final'].get('E1'), E2={k: v for k, v in (g['final'].get('E2') or {}).items() if k != 'intervals'})), indent=1)[:3000])

if __name__ == '__main__': main()
