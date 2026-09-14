"""Independent CPU verification of the RCC4 evaluator-side labels with integer truth tables (not protocol.evaluate): every FIT/CAL
question of both coverages (single flip and declared no-op), every FINAL/WORKFLOW program question (original + coverage additions), gold /
source gold / changed flags, plus the coverage structure (sparse = first six specifications of the complete bundle, same query ids).
Writes a receipt; exits nonzero on any discrepancy."""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcc4_common as C

def truth(values, spec):
    a, p = spec['a'], spec['p']; x = values[a][p]; k = spec['kind']
    if k == 'direct': return x
    if k == 'opposes': return 1-x
    y = values[spec['b']][p]
    return {'same': 1-(x-y)**2, 'both': x*y, 'either': min(1, x+y)}[k]

def edited_values(s, edits):
    v = [list(r) for r in s.values]
    for e in edits: v[e.actor][e.project] = e.value
    return v

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', required=True); a = ap.parse_args(); rep = dict(at=C.now(), seed=C.SEED, problems=[], counts={}); n = 0; fam = Counter()
    for split in ('FIT', 'CAL'):
        for s in C.scenes(split):
            for program in ('single', 'noop'):
                ev = edited_values(s, C.programs(s)[program]); src = [list(r) for r in s.values]
                for draw in (0, 1):
                    sparse = C.questions(s, split, draw, program, 'sparse'); complete = C.questions(s, split, draw, program, 'complete')
                    if len(sparse) != 6 or len(complete) != 12 or [q['query_id'] for q in sparse] != [q['query_id'] for q in complete[:6]] or [q['text'] for q in sparse] != [q['text'] for q in complete[:6]]: rep['problems'].append(dict(split=split, scene=s.scene_id, draw=draw, issue='coverage structure'))
                    if {q['family'] for q in sparse} - {'direct', 'opposes', 'same'} or not {'both', 'either'} <= {q['family'] for q in complete}: rep['problems'].append(dict(split=split, scene=s.scene_id, draw=draw, issue='families'))
                    for q in complete:
                        n += 1; fam[q['family']] += 1; g0 = truth(src, q['spec']); g1 = truth(ev, q['spec'])
                        if q['source_gold'] != g0 or q['gold'] != g1 or q['changed'] != (g0 != g1): rep['problems'].append(dict(split=split, scene=s.scene_id, program=program, query=q['query_id'], expected=(g0, g1), got=(q['source_gold'], q['gold'], q['changed'])))
                    if program == 'noop' and any(q['changed'] for q in complete): rep['problems'].append(dict(split=split, scene=s.scene_id, issue='no-op has a changed question'))
    for split in ('FINAL', 'WORKFLOW'):
        for s in C.scenes(split):
            for program in C.PROGRAMS:
                ev = edited_values(s, C.programs(s)[program]); src = [list(r) for r in s.values]
                for draw in (0, 1):
                    qs = C.questions(s, split, draw, program)
                    if sum(1 for q in qs if not q['additional']) != 12: rep['problems'].append(dict(split=split, scene=s.scene_id, program=program, issue='original bundle size'))
                    for q in qs:
                        n += 1; g0 = truth(src, q['spec']); g1 = truth(ev, q['spec'])
                        if q['source_gold'] != g0 or q['gold'] != g1 or q['changed'] != (g0 != g1): rep['problems'].append(dict(split=split, scene=s.scene_id, program=program, query=q['query_id'], expected=(g0, g1), got=(q['source_gold'], q['gold'], q['changed'])))
    rep['counts'] = dict(questions_checked=n, fit_cal_families=dict(fam)); rep['pass_'] = not rep['problems']
    C.dump(a.receipt, rep); print(C.canonical(dict(pass_=rep['pass_'], checked=n, problems=len(rep['problems']))))
    return 0 if rep['pass_'] else 1

if __name__ == '__main__': raise SystemExit(main())
