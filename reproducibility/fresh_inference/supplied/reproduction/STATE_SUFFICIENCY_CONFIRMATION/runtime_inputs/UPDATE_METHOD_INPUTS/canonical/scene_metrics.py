import numpy as np

def per_scene(rows):
    """Per scene: changed accuracy, invariant new-error fraction (all invariant rows), source-correct conditional fraction, all-24 (mean over
    prefix orders of every question correct), accuracy, touched-record accuracy/error, other-address error rate, preservation of originally
    correct answers, gold accuracy."""
    by = {}
    for r in rows: by.setdefault(r['scene_id'], []).append(r)
    out = {}
    for sid, rs in by.items():
        ch = [r for r in rs if r['gold'] != r['source_gold']]; inv = [r for r in rs if r['gold'] == r['source_gold']]
        src_ok = [r for r in inv if r['source_prediction'] == r['gold']]
        arts = {}
        for r in rs: arts.setdefault((r.get('order'), r.get('variant'), r.get('draw')), []).append(r)
        all_by_art = {k: float(all(r['prediction'] == r['gold'] for r in v)) for k, v in arts.items()}
        # full-scene all-24: every question (both draws) of one compiled artifact (scene x order x variant) correct
        art24 = {}
        for (o, v, d), val in all_by_art.items(): art24.setdefault((o, v), []).append(val)
        touched = [r for r in rs if r.get('touches_target')]; other = [r for r in rs if not r.get('touches_target') and r['gold'] == r['source_gold']]
        originally_correct = [r for r in rs if r['source_prediction'] == r['source_gold']]
        out[sid] = dict(scene_id=sid, queries=len(rs), direction=rs[0]['direction'], cell=rs[0].get('cell'),
                        changed=float(np.mean([r['prediction'] == r['gold'] for r in ch])) if ch else None,
                        harm=float(np.mean([r['source_prediction'] == r['gold'] and r['prediction'] != r['gold'] for r in inv])) if inv else None,
                        conditional_harm=float(np.mean([r['prediction'] != r['gold'] for r in src_ok])) if src_ok else None,
                        all24=float(np.mean([min(v) for v in art24.values()])),
                        accuracy=float(np.mean([r['prediction'] == r['gold'] for r in rs])),
                        touched_accuracy=float(np.mean([r['prediction'] == r['gold'] for r in touched])) if touched else None,
                        other_address_error=float(np.mean([r['prediction'] != r['gold'] for r in other])) if other else None,
                        preservation=float(np.mean([r['prediction'] == r['gold'] for r in originally_correct if r['gold'] == r['source_gold']])) if any(r['gold'] == r['source_gold'] for r in originally_correct) else None,
                        nonaddressed_change=float(np.mean([r['prediction'] != r['source_prediction'] for r in rs if not r.get('mentions_target') and r['gold'] == r['source_gold']])) if any(not r.get('mentions_target') and r['gold'] == r['source_gold'] for r in rs) else None)
    return out


def summarize(rows):
    ps = per_scene(rows)
    def avg(k, sub=None):
        xs = [v[k] for v in ps.values() if v[k] is not None and (sub is None or sub(v))]; return float(np.mean(xs)) if xs else None
    return dict(n_scenes=len(ps), n_queries=len(rows), changed=avg('changed'), harm=avg('harm'), conditional_harm=avg('conditional_harm'), all24=avg('all24'), accuracy=avg('accuracy'),
                touched_accuracy=avg('touched_accuracy'), other_address_error=avg('other_address_error'), preservation=avg('preservation'), nonaddressed_change=avg('nonaddressed_change'),
                changed_by_direction={d: avg('changed', lambda v, d=d: v['direction'] == d) for d in (0, 1)},
                by_cell={str(c): dict(changed=avg('changed', lambda v, c=c: tuple(v['cell']) == c), harm=avg('harm', lambda v, c=c: tuple(v['cell']) == c), all24=avg('all24', lambda v, c=c: tuple(v['cell']) == c)) for c in sorted({tuple(v['cell']) for v in ps.values() if v['cell']})},
                by_family={f: float(np.mean([r['prediction'] == r['gold'] for r in rows if r['family'] == f])) for f in sorted({r['family'] for r in rows})},
                per_scene=ps)

