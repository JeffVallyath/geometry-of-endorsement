"""Independent CPU verification of evaluator-side labels (NATIVE_QUALIFICATION_AMENDMENT_V1 condition 1).  Uses integer truth tables written
here, not protocol.evaluate, to check: (a) every qualification row's gold against the gold of the world actually rendered (source rows vs
source world, counterfactual rows vs the single-edit world); (b) equality's four truth cases present and correctly labelled in the battery;
(c) both answer mappings (No/Yes and A/B either order, 'labels[1] = yes'); (d) every FIT/CAL/FINAL/WORKFLOW question's changed/invariant
flag for the single edit (and every program on FINAL).  Writes a receipt; exits nonzero on any discrepancy."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import srs2_common as Q

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
    ap = argparse.ArgumentParser(); ap.add_argument('--qdir', help='phase Q output dir of the actor (qualification_*.jsonl)'); ap.add_argument('--receipt', required=True); a = ap.parse_args()
    rep = dict(at=Q.now(), problems=[], counts={})
    # (d) changed/invariant flags and gold for every split/program from independent truth tables
    n = 0
    for split in ('FIT', 'CAL', 'FINAL', 'WORKFLOW'):
        for s in Q.scenes(split):
            progs = Q.PROGRAMS if split == 'FINAL' else ('single', 'noop')
            for program in progs:
                ev = edited_values(s, Q.P.programs(s)[program]); src = [list(r) for r in s.values]
                for draw in (0, 1):
                    for q in Q.questions(s, split, draw, program=program):
                        n += 1; g0 = truth(src, q['spec']); g1 = truth(ev, q['spec'])
                        if q['source_gold'] != g0 or q['gold'] != g1 or q['changed'] != (g0 != g1): rep['problems'].append(dict(split=split, scene=s.scene_id, program=program, query=q['query_id'], expected=(g0, g1), got=(q['source_gold'], q['gold'], q['changed'])))
                        lab = q['labels']
                        if not ((lab == ['No', 'Yes'] and draw % 2 == 0) or (sorted(lab) == ['A', 'B'] and draw % 2 == 1)): rep['problems'].append(dict(query=q['query_id'], bad_labels=lab))
                        if draw % 2 == 1 and f'Use {lab[1]} for yes and {lab[0]} for no' not in q['text']: rep['problems'].append(dict(query=q['query_id'], mapping_text_mismatch=q['text'][-80:]))
    rep['counts']['questions_checked'] = n
    if a.qdir:
        qd = Q.load(Path(a.qdir)/'QUALIFICATION.json'); rep['counts']['procedures'] = [x['procedure'] for x in qd['attempts']]; cases = Counter(); maps = Counter(); nrows = 0
        scenes = {s.scene_id: s for s in Q.scenes('CAL')}
        for att in qd['attempts']:
            for r in Q.read_rows(Path(a.qdir)/f"qualification_{att['procedure']}.jsonl"):
                s = scenes[r['scene_id']]; nrows += 1; spec = next(q for q in Q.questions(s, 'CAL', r['draw'], program='single') if q['query_id'] == r['query_id'])['spec']
                world = [list(x) for x in s.values] if r['world'] == 'source' else edited_values(s, Q.P.programs(s)['single'])
                g = truth(world, spec)
                if r['target_index'] != g: rep['problems'].append(dict(row=r['query_id'], world=r['world'], procedure=att['procedure'], expected_gold=g, got=r['target_index']))
                if r['generation_prediction'] is not None and (r['generation_correct'] != (r['generation_prediction'] == g)): rep['problems'].append(dict(row=r['query_id'], world=r['world'], bad_generation_correct=True))
                if spec['kind'] == 'same' and att['procedure'] == qd['attempts'][0]['procedure']: cases[(r['world'], world[spec['a']][spec['p']], world[spec['b']][spec['p']], g)] += 1
                maps[(r['world'], tuple(r['labels']))] += 1
        rep['counts']['qualification_rows'] = nrows; rep['counts']['equality_truth_cases'] = {str(k): v for k, v in sorted(cases.items())}; rep['counts']['label_mappings'] = {str(k): v for k, v in sorted(maps.items())}
        four = {(x, y) for (w, x, y, g) in cases if w == 'source'}
        if four != {(0, 0), (0, 1), (1, 0), (1, 1)}: rep['problems'].append(dict(equality_truth_cases_missing=sorted({(0, 0), (0, 1), (1, 0), (1, 1)}-four)))
        if any((g != (1 if x == y else 0)) for (w, x, y, g) in cases): rep['problems'].append(dict(equality_gold_inconsistent=True))
        rep['recomputed_summaries'] = {}
        for att in qd['attempts']:
            rows = Q.read_rows(Path(a.qdir)/f"qualification_{att['procedure']}.jsonl"); out = {}
            for w in ('source', 'counterfactual'):
                rs = [r for r in rows if r['world'] == w]; dn = [r for r in rs if r['family'] in ('direct', 'opposes')]; eq = [r for r in rs if r['family'] == 'same']
                out[w] = dict(parse=sum(r['parse_status'].startswith('OK') for r in rs)/len(rs), direct_negation=sum(r['generation_correct'] for r in dn)/len(dn), equality=sum(r['generation_correct'] for r in eq)/len(eq))
                for k, v in (('parse_rate', out[w]['parse']), ('direct_negation_accuracy', out[w]['direct_negation']), ('equality_accuracy', out[w]['equality'])):
                    if abs(att['summary'][w][k]-v) > 1e-9: rep['problems'].append(dict(procedure=att['procedure'], world=w, summary_mismatch=k, recorded=att['summary'][w][k], recomputed=v))
            rep['recomputed_summaries'][att['procedure']] = out
    rep['pass_'] = not rep['problems']; Path(a.receipt).write_text(json.dumps(Q.json_safe(rep), indent=1)+'\n'); print(json.dumps(Q.json_safe(dict(pass_=rep['pass_'], counts=rep['counts'], problems=rep['problems'][:10])), indent=1))
    return 0 if rep['pass_'] else 1

if __name__ == '__main__': raise SystemExit(main())
