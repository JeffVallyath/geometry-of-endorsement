"""Fixed-denominator reporting layered on the unchanged independent A/C/D code."""
from __future__ import annotations
import math
import numpy as np
from text_scope.common import METHODS


def decode_score(row):
    """Permit invalid/out-of-label responses; never discard or force an answer."""
    lp = [float(v) for v in row['logps']]
    finite = len(lp) == 2 and all(math.isfinite(v) for v in lp)
    assert bool(row['finite']) == finite
    if finite:
        assert max(lp) <= 1e-6
        mass = sum(math.exp(v) for v in lp)
        assert 0 <= mass <= 1 + 1e-5
        assert abs(mass - float(row['answer_mass'])) <= 1e-6
        maximum = max(lp)
        weights = [math.exp(v - maximum) for v in lp]
        probabilities = [v / sum(weights) for v in weights]
        prediction = int(lp[1] > lp[0])
        assert prediction == row['prediction']
        tie = lp[0] == lp[1]
        assert tie == bool(row['tie'])
    else:
        mass = None
        probabilities = [None, None]
        prediction = row['prediction']
        tie = bool(row['tie'])
    if 'full_vocab_argmax_token_id' in row:
        member = row['full_vocab_argmax_token_id'] in row['label_token_ids']
        assert member == bool(row['argmax_in_labels'])
    valid = finite and not tie and bool(row['argmax_in_labels'])
    if 'valid' in row:
        assert row['valid'] == valid
    return dict(prediction=prediction, valid=bool(valid), probabilities=probabilities, mass=mass)


def root_metrics(evaluator, design, index, method):
    counts, details = evaluator.classify(design, index, method)
    questions = design['questions']
    values = design['root']['terminal']['values']
    def correct(key, q):
        value = index[key, q['query_id']]
        return bool(value['valid'] and value['prediction'] == evaluator.truth(values, q['spec']))
    direct = [q for q in questions if q['spec']['kind'] == 'direct']
    joint = [q for q in questions if q['spec']['kind'] in ('same', 'both', 'either')]
    all_scores = [index[f'o{o}/{method}', q['query_id']] for o in range(8) for q in questions]
    counts.update(
        A_only=sum(d['A'] and not d['C'] for d in details),
        C_only=sum(d['C'] and not d['A'] for d in details),
        neither=sum(not d['A'] and not d['C'] for d in details),
        direct_correct=sum(correct(f'o{o}/{method}', q) for o in range(8) for q in direct),
        direct_denominator=8*len(direct),
        joint_correct=sum(correct(f'o{o}/{method}', q) for o in range(8) for q in joint),
        joint_denominator=8*len(joint),
        battery_correct=sum(correct(f'o{o}/{method}', q) for o in range(8) for q in questions),
        battery_denominator=8*len(questions),
        native_correct=sum(correct('NATIVE_FINAL', q) for q in questions),
        noop_correct=sum(correct(f'o0/{method}', q) for q in questions),
        reference_denominator=len(questions),
        all_origin_joint_correct=sum(all(correct(f'o{o}/{method}', q) for o in range(8)) for q in joint),
        complete_case=all(correct(f'o{o}/{method}', q) for o in range(8) for q in questions),
        all_origin_valid_joint=sum(all(index[f'o{o}/{method}', q['query_id']]['valid'] for o in range(8)) for q in joint),
        invalid=sum(not r['valid'] for r in all_scores),
        finite_mass_count=sum(r['mass'] is not None for r in all_scores),
        candidate_masses=[r['mass'] for r in all_scores],
    )
    for d in details:
        q = next(q for q in joint if q['query_id'] == d['query_id'])
        d['all_origin_correct'] = all(correct(f'o{o}/{method}', q) for o in range(8))
        d['all_origin_valid'] = all(index[f'o{o}/{method}', q['query_id']]['valid'] for o in range(8))
        probabilities = d['correct_probabilities']
        d['finite_history_count'] = sum(p is not None for p in probabilities)
        d['half_probability_range'] = (max(probabilities) - min(probabilities))/2 if all(p is not None for p in probabilities) else None
        d['answer_code'] = d['query_id'].split('-d')[1][0]
    return counts, details


def root_draws():
    return np.random.Generator(np.random.PCG64(2609141002)).integers(0, 64, (10000, 64))


def interval(values, level, draws=None):
    values = np.asarray(values, dtype=float)
    assert values.shape == (64,)
    samples = values[root_draws() if draws is None else draws].mean(axis=1)
    tail = (1-level)/2
    return np.quantile(samples, [tail, 1-tail]).tolist()


def ratio(numerator, denominator, draws=None):
    num, den = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    assert num.shape == den.shape == (64,)
    draws = root_draws() if draws is None else draws
    sampled_num, sampled_den = num[draws].sum(axis=1), den[draws].sum(axis=1)
    valid = sampled_den > 0
    samples = sampled_num[valid] / sampled_den[valid]
    return dict(estimate=float(num.sum()/den.sum()) if den.sum() else None,
                interval95=np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                valid_draw_fraction=float(valid.mean()), numerator=float(num.sum()), denominator=float(den.sum()))


def summarize(roots, *, new_row=True):
    assert len(roots) == 64
    roots = sorted(roots, key=lambda r: r['root_id'])
    assert [r['root_id'] for r in roots] == [f'SSC1-FINAL-{i:04d}' for i in range(64)]
    assert all(r['questions'] == 18 for r in roots)
    draws = root_draws()
    values = np.array([r['W']/18 for r in roots])
    result = dict(roots=64, opportunities=1152,
                  W=sum(r['W'] for r in roots), strong=sum(r['strong'] for r in roots),
                  witness_rate=float(values.mean()), interval95=interval(values, .95, draws),
                  roots_with_witness=sum(r['W'] > 0 for r in roots),
                  roots_with_strong=sum(r['strong'] > 0 for r in roots))
    if new_row:
        result['primary_family8_interval99375'] = interval(values, .99375, draws)
    result['conditional_witness'] = ratio([r['W'] for r in roots], [r['AC'] for r in roots], draws)
    for name, key, denominator in [('A_coverage', 'A', 18), ('C_coverage', 'C', 18),
        ('E_coverage', 'AC', 18), ('A_only', 'A_only', 18), ('C_only', 'C_only', 18),
        ('neither', 'neither', 18), ('all_origin_joint_accuracy', 'all_origin_joint_correct', 18),
        ('complete_case_success', 'complete_case', 1), ('disagreement', 'D', 18),
        ('all_origin_joint_validity', 'all_origin_valid_joint', 18)]:
        array = np.array([r[key]/denominator for r in roots])
        result[name] = dict(estimate=float(array.mean()), interval95=interval(array, .95, draws))
    for name, numerator, denominator in [('direct_accuracy', 'direct_correct', 'direct_denominator'),
        ('joint_accuracy', 'joint_correct', 'joint_denominator'),
        ('battery_accuracy', 'battery_correct', 'battery_denominator'),
        ('native_accuracy', 'native_correct', 'reference_denominator'),
        ('noop_accuracy', 'noop_correct', 'reference_denominator')]:
        result[name] = ratio([r[numerator] for r in roots], [r[denominator] for r in roots], draws)
    mass = [m for r in roots for m in r['candidate_masses'] if m is not None]
    result['candidate_mass'] = dict(finite_count=len(mass), total=64*8*34,
        quantiles=dict(zip(('min','q05','q25','median','q75','q95','max'),
                          np.quantile(mass,[0,.05,.25,.5,.75,.95,1]).tolist())) if mass else None)
    result['invalid_responses'] = sum(r['invalid'] for r in roots)
    if result['W'] == 0:
        result['zero_count_warning'] = 'A zero-width bootstrap is not proof of invariance or a population upper bound.'
        result['zero_root_occurrence_upper_bound'] = dict(independent_root_sampling_assumption=True,
            roots_with_witness=0, roots=64, one_sided95=1-.05**(1/64),
            one_sided_family8=1-(.05/8)**(1/64))
    return result


def paired(left, right):
    left, right = [sorted(v, key=lambda r: r['root_id']) for v in (left, right)]
    assert len(left) == len(right) == 64
    assert [r['root_id'] for r in left] == [r['root_id'] for r in right]
    delta = np.array([(a['W']-b['W'])/18 for a,b in zip(left,right,strict=True)])
    draws = root_draws()
    result = dict(estimate=float(delta.mean()), interval95=interval(delta,.95,draws),
                  contrast_family6_interval=interval(delta,1-.05/6,draws))
    result['conditional_W_difference'] = paired_ratio(
        [r['W'] for r in left],[r['AC'] for r in left],
        [r['W'] for r in right],[r['AC'] for r in right],draws)
    for key, denominator in [('A',18),('C',18),('AC',18),('all_origin_joint_correct',18),('complete_case',1),
        ('direct_correct',96),('joint_correct',144),('battery_correct',272),('native_correct',34),('noop_correct',34)]:
        array = np.array([(a[key]-b[key])/denominator for a,b in zip(left,right,strict=True)])
        result[key] = dict(estimate=float(array.mean()), interval95=interval(array,.95,draws))
    return result


def paired_ratio(left_num,left_den,right_num,right_den,draws=None):
    arrays=[np.asarray(v,dtype=float) for v in (left_num,left_den,right_num,right_den)]
    assert all(v.shape==(64,) for v in arrays)
    draws=root_draws() if draws is None else draws
    ln,ld,rn,rd=[v[draws].sum(axis=1) for v in arrays]
    valid=(ld>0)&(rd>0)
    values=ln[valid]/ld[valid]-rn[valid]/rd[valid]
    totals=[v.sum() for v in arrays]
    estimate=float(totals[0]/totals[1]-totals[2]/totals[3]) if totals[1] and totals[3] else None
    return dict(estimate=estimate,interval95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,
                valid_draw_fraction=float(valid.mean()))


def jointly_certified(details_by_row):
    """Two/three-way E intersections; descriptive ratios, never primary filtering."""
    lookups = {row:{(v['root_id'],v['query_id']):v for v in details} for row,details in details_by_row.items()}
    assert len(lookups) in (2,3)
    keys = set(next(iter(lookups.values())))
    assert len(keys) == 1152 and all(set(v) == keys for v in lookups.values())
    roots = [f'SSC1-FINAL-{i:04d}' for i in range(64)]
    eligible = {key:all(v[key]['AC'] for v in lookups.values()) for key in keys}
    denominators = [sum(eligible[k] for k in keys if k[0]==rid) for rid in roots]
    result = dict(common_eligible=sum(denominators), full_opportunities=1152,
                  subset_is_secondary_not_causal_adjustment=True, rows={})
    for row, lookup in lookups.items():
        result['rows'][row] = {}
        for metric in ('W','all_origin_correct'):
            numerators = [sum(eligible[k] and lookup[k][metric] for k in keys if k[0]==rid) for rid in roots]
            result['rows'][row][metric] = ratio(numerators, denominators)
    result['paired_differences'] = {}
    from itertools import combinations
    for left,right in combinations(lookups,2):
        result['paired_differences'][left+' minus '+right] = {}
        for metric in ('W','all_origin_correct'):
            ln=[sum(eligible[k] and lookups[left][k][metric] for k in keys if k[0]==rid) for rid in roots]
            rn=[sum(eligible[k] and lookups[right][k][metric] for k in keys if k[0]==rid) for rid in roots]
            result['paired_differences'][left+' minus '+right][metric] = paired_ratio(ln,denominators,rn,denominators)
    return result
