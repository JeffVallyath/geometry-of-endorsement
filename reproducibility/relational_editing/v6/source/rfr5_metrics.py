"""Scene-paired inference and saved-direct-answer diagnostics; no model imports."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import rfr5_common as C


def bootstrap(values, *, level=C.PRIMARY_LEVEL, draws=C.BOOT_DRAWS, seed=C.BOOT_SEED):
    x = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Need finite whole-scene observations')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(draws, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return {'effect': float(x.mean()), 'lower': float(np.quantile(means, alpha)),
            'upper': float(np.quantile(means, 1 - alpha)), 'scenes': len(x),
            'level': level, 'draws': draws, 'bootstrap_seed': seed,
            'unit': 'whole scene', 'discordant_scenes': int(np.count_nonzero(x)),
            'zero_crossing_is_equivalence': False}


def vector(rows, *, actor, condition, reader, program='ABC', family='same', order='early'):
    grouped = defaultdict(list)
    for r in rows:
        if (r['actor'], r['condition'], r['reader'], r['program'], r['order']) == (actor, condition, reader, program, order) and (family is None or r['family'] == family):
            grouped[r['scene_id']].append(float(r['prediction'] == r['gold']))
    if not grouped:
        raise ValueError('Missing required comparison: ' + str((actor, condition, reader)))
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def paired_vectors(terms):
    """Linear contrasts paired before resampling; no unmatched-scene intersection."""
    keys = set(terms[0][1])
    if any(set(v) != keys for _, v in terms):
        raise ValueError('Missing paired scenes, not permission to drop them')
    return {k: sum(coef * v[k] for coef, v in terms) for k in sorted(keys)}


def seed_average(rows, actor, condition, reader):
    vec = [vector(rows, actor=actor, condition=condition + '_s' + str(seed), reader=reader) for seed in (0, 1)]
    return paired_vectors([(.5, v) for v in vec])


def primary_contrasts(rows):
    out = []
    for test in C.primary_inventory():
        a, c, reader = test['actor'], test['architecture'], test['reader']
        diff = paired_vectors([(1, seed_average(rows, a, c, reader)), (-1, seed_average(rows, a, c, 'R0'))])
        if len(diff) != C.SIZES[a]:
            raise ValueError('Primary scene coverage incomplete')
        seeds = {}
        for seed in (0, 1):
            d = paired_vectors([(1, vector(rows, actor=a, condition=c + '_s' + str(seed), reader=reader)),
                                (-1, vector(rows, actor=a, condition=c + '_s' + str(seed), reader='R0'))])
            seeds[str(seed)] = bootstrap(d.values())
        out.append({**test, **bootstrap(diff.values()), 'seed_specific': seeds, 'per_scene': diff})
    return out


def secondary_contrasts(rows):
    out = []
    for actor in C.SIZES:
        refs = ('SOURCE', 'NATIVE_FINAL', 'TEXT_CORRECTION', 'CANONICAL_CURRENT')
        for condition in refs:
            get = (lambda reader: seed_average(rows, actor, condition, reader)) if condition == 'CANONICAL_CURRENT' else (lambda reader: vector(rows, actor=actor, condition=condition, reader=reader))
            for reader in C.READERS[1:]:
                d = paired_vectors([(1, get(reader)), (-1, get('R0'))])
                out.append({'actor': actor, 'contrast': condition + ':' + reader + '-R0', **bootstrap(d.values())})
        for condition in C.COMPLETE:
            old = 'FROZEN_V3_' + ('INVARIANT_SET' if condition.startswith('INV') else 'FREE_OVERWRITE')
            for reader in C.READERS:
                edit = seed_average(rows, actor, condition, reader)
                native = vector(rows, actor=actor, condition='NATIVE_FINAL', reader=reader)
                for name, ref in [('edited-minus-native', native), ('COMPLETE-minus-V3', seed_average(rows, actor, old, reader))]:
                    d = paired_vectors([(1, edit), (-1, ref)])
                    out.append({'actor': actor, 'condition': condition, 'reader': reader, 'contrast': name, **bootstrap(d.values())})
                if reader != 'R0':
                    d = paired_vectors([(1, edit), (-1, native), (-1, seed_average(rows, actor, condition, 'R0')),
                                        (1, vector(rows, actor=actor, condition='NATIVE_FINAL', reader='R0'))])
                    out.append({'actor': actor, 'condition': condition, 'reader': reader, 'contrast': 'edited-native-difference-in-differences', **bootstrap(d.values())})
    return out


def summaries(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['actor'], r['condition'], r['reader'], r['program'], r['order'])].append(r)
    return [{'actor': k[0], 'condition': k[1], 'reader': k[2], 'program': k[3], 'order': k[4], **C.Q.summarize(v)} for k, v in sorted(groups.items())]


def full_family_contrasts(summaries_):
    indexed={(r['actor'],r['condition'],r['reader'],r['program'],r['order']):r for r in summaries_}
    result=[]
    for actor in C.SIZES:
        for condition in C.COMPLETE:
            old='FROZEN_V3_'+('INVARIANT_SET' if condition.startswith('INV') else 'FREE_OVERWRITE')
            for program,order in [('single','early'),('single','late'),('AB','early'),('ABC','early')]:
                for reader in C.READERS:
                    for metric in ('changed','harm','conditional_harm','all24','all_program_questions','accuracy'):
                        def get(cond):
                            r=indexed[(actor,cond,reader,program,order)]
                            return {s:v[metric] for s,v in r['per_scene'].items()}
                        e=[get(condition+'_s'+str(seed)) for seed in (0,1)]
                        for name,ref in [('COMPLETE-minus-V3',[get(old+'_s'+str(seed)) for seed in (0,1)]),('edited-minus-native',[get('NATIVE_FINAL')]*2)]:
                            keys=set(e[0])
                            if any(set(v)!=keys for v in e+ref):raise ValueError('Full-family scene pairing mismatch')
                            valid=[s for s in sorted(keys) if all(v[s] is not None for v in e+ref)]
                            base={'actor':actor,'condition':condition,'reader':reader,'program':program,'order':order,'metric':metric,'contrast':name,
                                  'total_scenes':len(keys),'undefined_denominator_scenes':len(keys)-len(valid)}
                            if valid:
                                d=[sum(e[i][s]-ref[i][s] for i in (0,1))/2 for s in valid]
                                base.update(bootstrap(d))
                            else:base['status']='UNDEFINED_DENOMINATOR'
                            result.append(base)
    return result


def recomposition_summaries(diagnostic):
    groups=defaultdict(list)
    for r in diagnostic:groups[tuple(r[k] for k in ('actor','condition','reader','program','order'))].append(r)
    result=[]
    for key,rs in groups.items():
        by_scene=defaultdict(list)
        for r in rs:by_scene[r['scene_id']].append(r)
        complete=[sid for sid,v in by_scene.items() if all(r['operand_coverage'] for r in v)]
        converted=[]
        for sid in complete:
            converted.extend(dict(r,prediction=r['recomposed_prediction'],source_prediction=r['recomposed_source_prediction']) for r in by_scene[sid])
        result.append({'actor':key[0],'condition':key[1],'reader':key[2],'program':key[3],'order':key[4],
                       'diagnostic_only':True,'total_scenes':len(by_scene),'fully_covered_scenes':len(complete),
                       'missing_operand_queries':sum(not r['operand_coverage'] for r in rs),
                       'direct_queries_used':len({(r['scene_id'],r['draw'],qid) for r in rs for qid in r['direct_query_ids']}),
                       'summary_on_fully_covered_scenes':C.Q.summarize(converted) if converted else None,
                       'never_internal_F2_or_native_reasoning':True})
    return result


def exact_logic(kind, x, y=None):
    if kind == 'direct': return int(x)
    if kind == 'opposes': return 1 - int(x)
    if y is None: raise ValueError('Missing directly measured operand')
    if kind == 'same': return int(x == y)
    if kind == 'both': return int(bool(x) and bool(y))
    if kind == 'either': return int(bool(x) or bool(y))
    raise ValueError(kind)


def recomposition(rows):
    """Never substitute gold. Missing operand queries remain explicitly missing."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ('actor', 'scene_id', 'condition', 'reader', 'program', 'order', 'draw'))].append(r)
    result = []
    for key, rs in groups.items():
        direct = {(r['spec']['a'], r['spec']['p']): r for r in rs if r['family'] == 'direct'}
        for r in rs:
            spec = r['spec']; kind = r['family']; x = direct.get((spec['a'], spec['p']))
            y = direct.get((spec['b'], spec['p'])) if kind in ('same', 'both', 'either') else None
            present = x is not None and (kind in ('direct', 'opposes') or y is not None)
            pred = exact_logic(kind, x['prediction'], y['prediction'] if y else None) if present else None
            source_pred = exact_logic(kind, x['source_prediction'], y['source_prediction'] if y else None) if present else None
            result.append({**r, 'native_prediction': r['prediction'], 'native_source_prediction': r['source_prediction'],
                           'recomposed_prediction': pred, 'recomposed_source_prediction': source_pred,
                           'operand_coverage': present, 'direct_query_ids': [v['query_id'] for v in (x, y) if v is not None],
                           'diagnostic_only': True, 'symbolic_access': 'declared query kind and operand addresses; extra saved direct outputs',
                           'agreement_with_direct': r['prediction'] == pred if present else None})
    return result


def truth_cells(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['family'] == 'same':
            key = tuple(r[k] for k in ('actor', 'condition', 'reader', 'program', 'order')) + tuple(r['truth_cell'])
            groups[key].append(r)
    result = []
    for key, rs in sorted(groups.items()):
        by = defaultdict(list)
        for r in rs: by[r['scene_id']].append(float(r['prediction'] == r['gold']))
        result.append({'actor': key[0], 'condition': key[1], 'reader': key[2], 'program': key[3], 'order': key[4],
                       'source_a': key[5], 'source_b': key[6], 'final_a': key[7], 'final_b': key[8],
                       'queries': len(rs), 'correct': sum(r['prediction'] == r['gold'] for r in rs),
                       'scene_macro': bootstrap([np.mean(v) for v in by.values()]), 'conditional_diagnostic': True})
    return result
