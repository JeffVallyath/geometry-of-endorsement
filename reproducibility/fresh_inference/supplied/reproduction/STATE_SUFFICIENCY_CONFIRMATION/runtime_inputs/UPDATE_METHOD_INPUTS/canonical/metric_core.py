from collections import defaultdict
from types import SimpleNamespace
import numpy as np
from . import origins
S = SimpleNamespace(R=origins)

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

