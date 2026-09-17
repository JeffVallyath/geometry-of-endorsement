from collections import defaultdict

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

