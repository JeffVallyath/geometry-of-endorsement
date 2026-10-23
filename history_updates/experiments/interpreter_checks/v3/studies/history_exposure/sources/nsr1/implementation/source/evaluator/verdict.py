"""NSR1 flags and terminal interpretation. The mapping below was committed before any final output was pulled.

Flags are reported separately (protocol 8); one headline never replaces them.
Terminal mapping (strict, in this order):
  TECHNICAL_STOP                 a required primary panel did not execute, or S0 fails for a technical reason
  FULL_ON_DECLARED_EVALUATION    S0 and S3 pass and every supported answer on every declared main panel and on the
                                 preserved A10 cohort is valid and correct (no new checked error)
  FORMAL_CONTRACT_ONLY           S0 passes but no real-model language measurement exists
  LANGUAGE_COMPETENCE_FAILURE    S0 passes and at least one S1 component fails (the failing components are listed)
  SIMPLE_BASELINE_SUFFICIENT     S1 passes and FLAT_SYMBOLIC passes every S1 component too (no comparative gain needed)
  PRESERVATION_FAILURE           S1 passes and S2 fails (repair rate with >= 20 witness roots, or newly-wrong bound)
  BOUNDED_RELIABLE_REMEDY        S0, S1, S3 pass, S2 passes or its guard is too sparse for a rate (counts reported)
Ordinary SQLite matching the custom executor is reported in the novelty verdict; it never changes the terminal label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

C = json.loads((Path(__file__).resolve().parents[1] / 'governing_inputs' / 'NSR1_CONFIG.json').read_text())['criteria']
MAIN = ('core', 'factorial', 'temporal', 'graph', 'public')


def s1(model, arm='ISOLATED_SYMBOLIC'):
    P = model['panels']
    comp, missing = {}, [n for n in MAIN if P.get(n, {}).get('status') != 'EXECUTED']
    if missing:
        return dict(status='NOT_MEASURED', missing=missing)
    fam = P['core']['families'].get(arm, {})
    for f in ('direct', 'opposes', 'joint', 'historical'):
        a = fam.get(f, {}).get('accuracy')
        comp[f'core_{f}'] = dict(value=a, floor=C['primary_boolean_valid_accuracy_min'], ok=a is not None and a >= C['primary_boolean_valid_accuracy_min'])
    fc = P['factorial'].get('factorial', {}).get(arm, {})
    comp['factorial_min_cell'] = dict(value=fc.get('min_cell_accuracy'), floor=C['primary_each_boolean_truth_cell_min'],
                                      ok=fc.get('min_cell_accuracy') is not None and fc['min_cell_accuracy'] >= C['primary_each_boolean_truth_cell_min'],
                                      cells_below=[k for k, v in fc.get('cells', {}).items() if v['accuracy'] < C['primary_each_boolean_truth_cell_min']])
    wb = P['core']['whole_root_bundle'].get(arm, {})
    comp['core_whole_root_bundles'] = dict(value=wb.get('fraction'), floor=C['primary_whole_root_correct_min'],
                                           ok=wb.get('fraction') is not None and wb['fraction'] >= C['primary_whole_root_correct_min'])
    for name, floor in (('temporal', C['primary_temporal_valid_accuracy_min']), ('graph', C['primary_graph_valid_accuracy_min']),
                        ('public', C['primary_public_valid_accuracy_min'])):
        a = P[name]['families'].get(arm, {}).get('ALL', {}).get('accuracy')
        comp[name] = dict(value=a, floor=floor, ok=a is not None and a >= floor)
    n = sum(P[x]['families'].get(arm, {}).get('ALL', {}).get('n', 0) for x in MAIN)
    inv = sum(P[x]['families'].get(arm, {}).get('ALL', {}).get('invalid', 0) for x in MAIN)
    comp['in_scope_abstention'] = dict(value=inv / n if n else None, ceiling=C['max_in_scope_abstention_fraction'], no_answer_rows=inv, rows=n,
                                       ok=n > 0 and inv / n <= C['max_in_scope_abstention_fraction'])
    failing = [k for k, v in comp.items() if not v['ok']]
    return dict(status='PASS' if not failing else 'FAIL', failing=failing, components=comp, arm=arm)


def s2(model, a10):
    w = model['panels'].get('core', {}).get('witness')
    if not w:
        return dict(status='NOT_MEASURED')
    out = dict(fresh_core=dict(witnesses=w['baseline_witnesses'], witness_roots=w['witness_roots'], target_opportunities=w['target_opportunities'],
                               baseline_targets_with_any_history_error=w['baseline_targets_with_any_history_error']))
    c = w.get('ISOLATED_SYMBOLIC', {})
    rate = c.get('complete_repair_rate') or {}
    harm = c.get('newly_wrong_all_scheduled') or {}
    enough = w['witness_roots'] >= C['min_witness_roots_for_repair_rate']
    out['fresh_core'].update(complete_preserved_repair=c.get('complete_preserved_repair'), target_repaired=c.get('target_repaired'),
                             repair_point=rate.get('point'), repair_lower_one_sided95=rate.get('lower'),
                             newly_wrong=c.get('newly_wrong_count'), newly_wrong_upper_one_sided95=harm.get('upper'), rate_claim_allowed=enough)
    harm_ok = harm.get('upper') is not None and harm['upper'] <= C['new_harm_upper_bound_max']
    if not enough:
        status = 'GUARD_SPARSE_COUNTS_ONLY' if harm_ok else 'FAIL'
    else:
        status = 'PASS' if (rate['point'] >= C['repair_point_min'] and rate['lower'] >= C['repair_lower_bound_min'] and harm_ok) else 'FAIL'
    out['status'] = status
    if a10:
        out['old_a10'] = {arm: {k: v['totals'].get(k) for k in ('baseline_W', 'A10_B_repairs', 'W_target_repaired', 'W_fully_preserved_repair',
                                                                'W_retained_as_candidate_witness', 'W_lost_qualification_without_repair',
                                                                'new_candidate_witnesses', 'candidate_bundle_all_correct', 'target_opportunities',
                                                                'joint_old_error', 'joint_new_error', 'joint_corrected', 'joint_introduced',
                                                                'direct_old_error', 'direct_new_error', 'direct_introduced',
                                                                'opposes_old_error', 'opposes_new_error', 'opposes_introduced')}
                          for arm, v in a10.items()}
    return out


def s3(model):
    P = model['panels']
    checks, ok = {}, True
    for name, p in P.items():
        if p.get('status') != 'EXECUTED' or 'isolation' not in p:
            continue
        iso, sq = p['isolation'], p.get('sql_shared_parse', {})
        rc = (p['run'].get('recheck') or {})
        mism = sum(v.get('mismatch', 0) for v in rc.values())
        checks[name] = dict(unexplained_answer_changes=iso['unexplained'], explained_by_parsed_state=iso['explained_by_parsed_state'],
                            archive_reads_in_current_programs=iso['archive_reads_in_current_programs'], sql_all_agree=sq.get('all_agree'),
                            sql_executed_rows=sq.get('executed'), current_reread_after_history_mismatch=mism)
        ok = ok and iso['unexplained'] == 0 and iso['archive_reads_in_current_programs'] == 0 and bool(sq.get('all_agree')) and mism == 0
    return dict(status='PASS' if ok and checks else ('NOT_MEASURED' if not checks else 'FAIL'), panels=checks,
                ablations=P.get('boundary', {}).get('boundary'))


def s4(timing):
    if not timing:
        return dict(status='NOT_MEASURED')
    comp = timing['competence']
    cand_ok = comp['ISOLATED_SYMBOLIC']['competent']
    competent = [b for b, v in timing['speedups'].items() if v['baseline_competent']]
    strongest = min(competent, key=lambda b: timing['speedups'][b]['geometric_mean_ratio_baseline_over_candidate']) if competent else None
    out = dict(candidate_competent=cand_ok, competence=comp, strongest_competent_baseline=strongest, speedups=timing['speedups'])
    if not cand_ok:
        out['status'] = 'FAIL_NOT_COMPETENT'
    elif strongest is None:
        out['status'] = 'NO_COMPETENT_COMPARABLE_BASELINE'
    else:
        s = timing['speedups'][strongest]
        out['status'] = 'PASS' if (s['geometric_mean_ratio_baseline_over_candidate'] >= C['practical_speedup_point_min'] and s['lower95_one_sided'] > 1) else 'FAIL'
        out['point'], out['lower'] = s['geometric_mean_ratio_baseline_over_candidate'], s['lower95_one_sided']
    return out


def verdict(analysis, cpu_tests, a10, timing, secondary=None):
    model = analysis
    S0 = dict(cpu_tests_passed=cpu_tests['passed'], cpu_tests_total=cpu_tests['total'],
              typed_oracle=next((r for r in cpu_tests['results'] if r['test'] == 'B_typed_oracle_three_way_all_panels'), {}),
              sql_shared_parse={n: dict(executed=p['sql_shared_parse']['executed'], all_agree=p['sql_shared_parse']['all_agree'])
                                for n, p in model['panels'].items() if p.get('status') == 'EXECUTED' and 'sql_shared_parse' in p})
    S0['status'] = 'PASS' if (cpu_tests['passed'] == cpu_tests['total'] and all(v['all_agree'] for v in S0['sql_shared_parse'].values())) else 'FAIL'
    S1, S1F = s1(model), s1(model, 'FLAT_SYMBOLIC')
    S2, S3, S4 = s2(model, a10), s3(model), s4(timing)
    P = model['panels']
    executed = all(P.get(n, {}).get('status') == 'EXECUTED' for n in MAIN)
    errors = {n: P[n]['families']['ISOLATED_SYMBOLIC']['ALL']['n'] - P[n]['families']['ISOLATED_SYMBOLIC']['ALL']['correct'] for n in MAIN if P.get(n, {}).get('status') == 'EXECUTED'}
    a10_ok = bool(a10) and all(v['totals']['candidate_bundle_all_correct'] == v['totals']['target_opportunities'] and
                               v['totals']['joint_new_error'] == v['totals']['direct_new_error'] == v['totals']['opposes_new_error'] == 0 for v in a10.values())
    full = executed and S0['status'] == 'PASS' and S3['status'] == 'PASS' and sum(errors.values()) == 0 and a10_ok
    if not executed or S0['status'] != 'PASS':
        term = 'TECHNICAL_STOP' if S0['status'] != 'PASS' or any(P.get(n, {}).get('status') == 'EXECUTED' for n in MAIN) else 'FORMAL_CONTRACT_ONLY'
        if S0['status'] == 'PASS' and not any(P.get(n, {}).get('status') == 'EXECUTED' for n in MAIN):
            term = 'FORMAL_CONTRACT_ONLY'
    elif full:
        term = 'FULL_ON_DECLARED_EVALUATION'
    elif S1['status'] != 'PASS':
        term = 'LANGUAGE_COMPETENCE_FAILURE'
    elif S1F['status'] == 'PASS':
        term = 'SIMPLE_BASELINE_SUFFICIENT'
    elif S2['status'] == 'FAIL':
        term = 'PRESERVATION_FAILURE'
    elif S3['status'] == 'PASS':
        term = 'BOUNDED_RELIABLE_REMEDY'
    else:
        term = 'TECHNICAL_STOP'
    return dict(study_id='NSR1_NEUROSYMBOLIC_REPAIR_CLOSURE', terminal_interpretation=term,
                flags=dict(S0_EXECUTOR_CONTRACT=S0, S1_LANGUAGE_COMPETENCE=S1, S2_PRESERVED_REPAIR=S2, S3_END_TO_END_BOUNDARY=S3, S4_PRACTICAL=S4,
                           FULL_ON_DECLARED_EVALUATION=dict(status=full, isolated_errors_by_panel=errors, a10_preserved_and_error_free=a10_ok)),
                comparators=dict(S1_FLAT_SYMBOLIC=S1F, S1_NATIVE_NEURAL=s1(model, 'NATIVE_NEURAL'),
                                 primary_contrasts=dict(core_whole_root=P.get('core', {}).get('primary_contrast_whole_root_boolean_isolated_minus_flat'),
                                                        graph_end_to_end=P.get('graph', {}).get('primary_contrast_graph_end_to_end_isolated_minus_flat'))),
                novelty=dict(custom_executor_necessary=False,
                             reason='ordinary SQLite executes the same parsed state and programs and agrees on every executed row' if S0['status'] == 'PASS' else 'S0 not established',
                             statement='existing-method construction (language-to-program + versioned state + exact execution); no new architecture claimed'),
                secondary_backend=secondary, typed_oracle_never_pooled=True)


if __name__ == '__main__':
    rec = Path(sys.argv[1])
    A = json.loads((rec / 'NSR1_ANALYSIS.json').read_text())
    v = verdict(A['primary'], json.loads((rec / 'CPU_TESTS.json').read_text()), A.get('a10'), A.get('timing'), A.get('secondary_summary'))
    (rec / 'NSR1_VERDICT.json').write_text(json.dumps(v, indent=1, default=str))
    print(v['terminal_interpretation'], {k: (x.get('status') if isinstance(x, dict) else x) for k, x in v['flags'].items()})
