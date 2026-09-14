"""Root/scene-paired V6 analyses. Never replaces primary outputs with gold."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import rfr5_metrics as A
import rsi6_common as S


def rate(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def valid(row, source=False):
    return bool(row.get('source_valid' if source else 'valid', False))


def correct(row):
    return valid(row) and row['prediction'] == row['gold']


def origin_rates(rows):
    invariant = [r for r in rows if r['gold'] == r['source_gold']]
    changed = [r for r in rows if r['gold'] != r['source_gold']]
    source_correct = [r for r in invariant if valid(r, True) and r['source_prediction'] == r['source_gold']]
    errors = sum(not correct(r) for r in source_correct)
    return {'accuracy': rate(correct(r) for r in rows),
            'source_world_accuracy': rate(valid(r) and r['prediction'] == r['source_gold'] for r in rows),
            'changed_accuracy': rate(correct(r) for r in changed),
            'harm': errors / len(invariant) if invariant else None,
            'conditional_harm': errors / len(source_correct) if source_correct else None,
            'questions': len(rows), 'changed_questions': len(changed),
            'invariant_questions': len(invariant), 'source_correct_invariant_questions': len(source_correct),
            'new_invariant_errors': errors, 'invalids': sum(not valid(r) for r in rows)}


def root_statistics(rows):
    """One root, one condition/seed/reader; fail closed on missing origins/queries."""
    if not rows or len({(r['actor'], r['root_id'], r['condition'], r['reader']) for r in rows}) != 1:
        raise ValueError('Expected one matched root group')
    by = defaultdict(dict)
    for row in rows:
        key = row['query_id']
        if key in by[row['origin']]:
            raise ValueError('Duplicate origin/question')
        by[row['origin']][key] = row
    if set(by) != {0, 1, 2, 3}:
        raise ValueError('All four origins required, not four independent samples')
    ids = sorted(by[0])
    if len(ids) != 34 or any(set(v) != set(ids) for v in by.values()):
        raise ValueError('Complete matched 34-question bundle required')
    for qid in ids:
        reference = by[0][qid]
        for origin in (1, 2, 3):
            r = by[origin][qid]
            if any(r[k] != reference[k] for k in ('gold', 'labels', 'suffix_ids', 'spec', 'additional')):
                raise ValueError('Origin-dependent terminal labels or reader interface')
    original_ids = [i for i in ids if not by[0][i]['additional']]
    if len(original_ids) != 24:
        raise ValueError('Original-24 membership changed')
    def metrics(keys):
        predictions = [[by[o][i]['prediction'] for i in keys] for o in range(4)]
        validity = [[valid(by[o][i]) for i in keys] for o in range(4)]
        return S.R.origin_metrics(predictions, [by[0][i]['gold'] for i in keys], validity)
    row = rows[0]
    result = {k: row[k] for k in ('actor', 'root_id', 'condition', 'reader', 'seed')}
    result.update(all34=metrics(ids), original24=metrics(original_ids),
                  by_origin={str(o): origin_rates(list(by[o].values())) for o in range(4)})
    result['by_family'] = {family: metrics([i for i in ids if by[0][i]['family'] == family])
                           for family in sorted({by[0][i]['family'] for i in ids})}
    result['joint_only'] = metrics([i for i in ids if by[0][i]['family'] in ('same', 'both', 'either')])
    # This is a root summary, not independent origins or answer-code draws.
    result['mean_origin_harm'] = rate(v['harm'] for v in result['by_origin'].values() if v['harm'] is not None)
    result['mean_origin_conditional_harm'] = rate(v['conditional_harm'] for v in result['by_origin'].values()
                                                if v['conditional_harm'] is not None)
    result['prediction_probability_spread'] = float(np.mean([
        max(np.exp(by[o][i]['logps'][1]) for o in range(4)) -
        min(np.exp(by[o][i]['logps'][1]) for o in range(4)) for i in ids]))
    return result


def source_independence(rows, *, sizes=S.SIZES['B']):
    grouped = defaultdict(list)
    for r in rows:
        if r['panel'] != 'B':
            raise ValueError('Panel B analysis cannot pool Panel A/C')
        grouped[(r['actor'], r['root_id'], r['condition'], r['reader'])].append(r)
    stats = {k: root_statistics(v) for k, v in grouped.items()}
    roots = {actor: {k[1] for k in stats if k[0] == actor} for actor in sizes}
    for actor, n in sizes.items():
        if len(roots[actor]) != n:
            raise ValueError('Incomplete fixed root population')
        required = {(actor, root, cond, reader) for root in roots[actor]
                    for cond in S.CONDITIONS for reader in S.C.READERS}
        if {k for k in stats if k[0] == actor} != required:
            raise ValueError('Missing actor/seed/condition/reader; no favorable intersection')
    if set(k[0] for k in stats) != set(sizes):
        raise ValueError('Unexpected actor')

    def vector(actor, cond, reader, bundle='all34', metric='all_origin_question_correct'):
        return {r: stats[(actor, r, cond, reader)][bundle][metric] for r in sorted(roots[actor])}

    def interval(terms, level=.9875):
        paired = A.paired_vectors(terms)
        return {**S.R.paired_interval(list(paired.values()), level=level), 'per_root': paired}

    primaries, secondary, absolute = [], [], []
    for actor in sizes:
        for cond in S.CONDITIONS:
            for reader in S.C.READERS:
                record = {'actor': actor, 'condition': cond, 'reader': reader, 'roots': len(roots[actor])}
                for bundle in ('all34', 'original24'):
                    record[bundle] = {metric: interval([(1, vector(actor, cond, reader, bundle, metric))], .95)
                                      for metric in ('mean_accuracy', 'all_origin_question_correct',
                                                     'all_origin_bundle_correct', 'origin_disagreement',
                                                     'all_origins_same_but_wrong')}
                m = record['all34']
                record['prospective_operating_point'] = (
                    m['all_origin_question_correct']['effect'] >= .9 and
                    m['origin_disagreement']['effect'] <= .05 and
                    m['all_origin_bundle_correct']['effect'] >= .7)
                record['joint_only'] = {metric: interval([(1, vector(actor, cond, reader, 'joint_only', metric))], .95)
                    for metric in ('mean_accuracy', 'all_origin_question_correct', 'all_origin_bundle_correct', 'origin_disagreement')}
                record['probability_spread'] = interval([(1, {r: stats[(actor, r, cond, reader)]['prediction_probability_spread']
                    for r in sorted(roots[actor])})], .95)
                absolute.append(record)
        for arm in S.C.COMPLETE:
            for reader in ('R1', 'R2'):
                seed_specific = {}
                terms = []
                for seed in (0, 1):
                    cond = f'{arm}_s{seed}'
                    terms.extend([(.5, vector(actor, cond, reader)), (-.5, vector(actor, cond, 'R0'))])
                    seed_specific[str(seed)] = interval([(1, vector(actor, cond, reader)),
                                                         (-1, vector(actor, cond, 'R0'))])
                result = {'actor': actor, 'architecture': arm, 'reader': reader, 'baseline': 'R0',
                    'endpoint': 'all34.all_origin_question_correct', 'seed_specific': seed_specific,
                    'classification': 'primary' if reader == 'R2' else 'secondary', **interval(terms)}
                (primaries if reader == 'R2' else secondary).append(result)
                # Native reader change subtracted only as a declared secondary analysis.
                did = terms + [(-1, vector(actor, 'NATIVE_FINAL', reader)),
                               (1, vector(actor, 'NATIVE_FINAL', 'R0'))]
                secondary.append({'actor': actor, 'architecture': arm, 'reader': reader,
                    'contrast': 'native-difference-in-differences', **interval(did)})
                for bundle in ('joint_only', 'original24'):
                    terms = [term for seed in (0, 1) for term in (
                        (.5, vector(actor, f'{arm}_s{seed}', reader, bundle)),
                        (-.5, vector(actor, f'{arm}_s{seed}', 'R0', bundle)))]
                    secondary.append({'actor': actor, 'architecture': arm, 'reader': reader,
                        'contrast': bundle + '-reader-effect', 'classification': 'secondary', **interval(terms)})
    return {'per_root': list(stats.values()), 'absolute': absolute, 'primary': primaries,
            'secondary': secondary, 'independence_unit': 'root; origins/questions/draws/seeds grouped',
            'original24_is_separate': True, 'finite_question_family_only': True}


def program_retention(rows):
    """Per-scene repeat/restoration evidence, not the unrerun comparative F2."""
    groups = defaultdict(list)
    for r in rows:
        if r['panel'] != 'C':
            raise ValueError('Panel C required')
        groups[(r['actor'], r['scene_id'], r['condition'], r['reader'], r['program'])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        actor, scene, condition, reader, program = key
        if program not in ('repeat2', 'repeat4', 'repeat8', 'restore', 'restore_after4'):
            continue
        base = {'actor': actor, 'scene_id': scene, 'condition': condition, 'reader': reader, 'program': program}
        original = [r for r in rs if not r['additional']]
        if len(original) != 24:
            raise ValueError('Original question bundle incomplete')
        if program.startswith('repeat'):
            single = {r['query_id']: r for r in groups[(actor, scene, condition, reader, 'single')]
                      if not r['additional']}
            if set(single) != {r['query_id'] for r in original}:
                raise ValueError('Repeat/SINGLE question pairing incomplete')
            base['disagreement'] = rate(not (valid(r) and valid(single[r['query_id']])
                and r['prediction'] == single[r['query_id']]['prediction']) for r in original)
            base['single_accuracy'] = rate(correct(r) for r in single.values())
        else:
            if any(r['gold'] != r['source_gold'] for r in original):
                raise ValueError('Restoration final world is not the original')
            keep = [r for r in original if valid(r, True) and r['source_prediction'] == r['source_gold']]
            base.update(preservation=rate(correct(r) for r in keep), source_correct_questions=len(keep),
                        absolute_accuracy=rate(correct(r) for r in original))
        out.append(base)
    return out


def grouped_rates(rows, keys, *, unit='root_id'):
    """Macro by independent unit; truth-cell conditioning never filters primary rows."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(tuple(r[k]) if isinstance(r[k], list) else r[k] for k in keys)
        groups[key][r[unit]].append(r)
    result = []
    for key, units in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        rates = {sid: origin_rates(rs) for sid, rs in units.items()}
        record = dict(zip(keys, key))
        record.update(independence_unit=unit, units=len(units), questions=sum(len(rs) for rs in units.values()),
                      per_unit=rates)
        for metric in ('accuracy', 'source_world_accuracy', 'changed_accuracy', 'harm', 'conditional_harm'):
            values = [v[metric] for v in rates.values() if v[metric] is not None]
            record[metric] = ({**A.bootstrap(values, level=.95), 'undefined_units': len(units) - len(values)}
                              if values else {'status': 'UNDEFINED_DENOMINATOR', 'undefined_units': len(units)})
        result.append(record)
    return result


def native_agreement(rows):
    native = {(r['actor'], r['root_id'], r['origin'], r['reader'], r['query_id']): r
              for r in rows if r['condition'] == 'NATIVE_FINAL'}
    groups = defaultdict(list)
    for row in rows:
        n = native[(row['actor'], row['root_id'], row['origin'], row['reader'], row['query_id'])]
        key = tuple(row[k] for k in ('actor', 'condition', 'reader', 'root_id'))
        groups[key].append({'agreement': valid(row) and valid(n) and row['prediction'] == n['prediction'],
                           'truth_correct': correct(row), 'native_correct': correct(n),
                           'agrees_with_native_wrong': valid(row) and valid(n) and
                            row['prediction'] == n['prediction'] and not correct(n)})
    per_root = [{**dict(zip(('actor', 'condition', 'reader', 'root_id'), key)),
                 **{m: rate(v[m] for v in values) for m in values[0]}} for key, values in groups.items()]
    summaries = defaultdict(list)
    for r in per_root:
        summaries[tuple(r[k] for k in ('actor', 'condition', 'reader'))].append(r)
    return {'per_root': per_root, 'summaries': [
        {**dict(zip(('actor', 'condition', 'reader'), key)), **{m: A.bootstrap([r[m] for r in rs], level=.95)
         for m in ('agreement', 'truth_correct', 'native_correct', 'agrees_with_native_wrong')}}
        for key, rs in summaries.items()], 'agreement_is_not_truth_correctness': True}


def functional_progress(a_rows, c_rows, *, prior_a=None):
    """Prospective reader-qualified conjunctions; historical comparative F2/F4 absent."""
    retention = program_retention(c_rows) if c_rows else []
    def summary(rows, actor, condition, reader, program='single'):
        rs = [r for r in rows if (r['actor'], r['condition'], r['reader'], r['program']) ==
              (actor, condition, reader, program)]
        # Keep the raw forced-choice values untouched; invalids fail prospective
        # quality conjunctions rather than becoming successful stable answers.
        view = [dict(r, prediction=r['prediction'] if valid(r) else -1,
                     source_prediction=r['source_prediction'] if valid(r, True) else -1) for r in rs]
        return S.C.Q.summarize(view) if view else None
    def f1(m, native):
        if m is None or native is None:
            return None
        rates = m['changed_by_direction']
        directions = list(rates.values())
        return (m['changed'] is not None and m['changed'] >= .85 and m['harm'] <= .05
                and len(directions) == 2 and all(v is not None and v >= .8 for v in directions)
                and m['all24'] >= .7 * native['all24'])
    cached = None
    if prior_a is not None:
        if a_rows:
            raise ValueError('Use raw A rows or verified prior A summaries, not both')
        cached = {(r['actor'], r['architecture'], r['reader']): r for r in prior_a}
        expected = {(a, arm, reader) for a in S.C.SIZES for arm in S.C.COMPLETE
                    for reader in S.C.READERS}
        if len(cached) != len(prior_a) or set(cached) != expected:
            raise ValueError('Retained A summary inventory changed')
        for row in cached.values():
            if set(row['seeds']) != {'0', '1'}:
                raise ValueError('Both original A seeds are required')
            for seed in row['seeds'].values():
                if seed['main_F1'] != f1(seed['main_metrics'], seed['main_native_metrics']):
                    raise ValueError('Retained A F1 disagrees with its frozen summary')
    out = []
    for actor in S.C.SIZES:
        for arm in S.C.COMPLETE:
            for reader in S.C.READERS:
                seeds = {}
                for seed in (0, 1):
                    cond = f'{arm}_s{seed}'
                    previous = cached[(actor, arm, reader)]['seeds'][str(seed)] if cached is not None else None
                    main = previous['main_metrics'] if previous is not None else summary(a_rows, actor, cond, reader)
                    main_native = (previous['main_native_metrics'] if previous is not None
                                   else summary(a_rows, actor, 'NATIVE_FINAL', reader))
                    single_c = summary(c_rows, actor, cond, reader)
                    native_c = summary(c_rows, actor, 'NATIVE_FINAL', reader)
                    record = {'main_F1': f1(main, main_native), 'main_metrics': main,
                        'main_native_metrics': main_native, 'program_SINGLE_F1_quality': f1(single_c, native_c),
                        'program_SINGLE_metrics': single_c, 'program_SINGLE_native_metrics': native_c,
                        'repeat': {}, 'restoration': {}, 'joint': {}}
                    for program in ('repeat2', 'repeat4', 'repeat8', 'restore', 'restore_after4'):
                        rs = [r for r in retention if (r['actor'], r['condition'], r['reader'], r['program']) ==
                              (actor, cond, reader, program)]
                        metric = 'disagreement' if program.startswith('repeat') else 'preservation'
                        values = [r[metric] for r in rs if r[metric] is not None]
                        stat = A.bootstrap(values, level=.95) if values else None
                        passed = (stat['effect'] <= .05 if metric == 'disagreement' else stat['effect'] >= .95) if stat else None
                        record['repeat' if metric == 'disagreement' else 'restoration'][program] = {
                            'pass': passed, 'interval': stat, 'undefined_scenes': len(rs) - len(values), 'per_scene': rs}
                    for program in ('AB', 'ABC'):
                        m = summary(c_rows, actor, cond, reader, program)
                        record['joint'][program] = {'metrics': m,
                            'pass': m['changed'] >= .8 and m['harm'] <= .05 if m else None}
                    repeat_pass = record['program_SINGLE_F1_quality'] and all(v['pass'] for v in record['repeat'].values())
                    restore_pass = all(v['pass'] for v in record['restoration'].values())
                    record['functional_F1_repeat_restoration'] = bool(record['main_F1'] and repeat_pass and restore_pass) if c_rows else None
                    record['G2_full_functional_conjunction'] = bool(record['functional_F1_repeat_restoration'] and
                        all(v['pass'] for v in record['joint'].values())) if c_rows else None
                    seeds[str(seed)] = record
                out.append({'actor': actor, 'architecture': arm, 'reader': reader, 'seeds': seeds,
                    'two_seed_functional_replication': all(v['functional_F1_repeat_restoration'] for v in seeds.values()) if c_rows else None,
                    'two_seed_G2': all(v['G2_full_functional_conjunction'] for v in seeds.values()) if c_rows else None,
                    'historical_V4_unchanged': True, 'old_comparative_F2_rerun': False, 'fresh_F4': False})
    return out
