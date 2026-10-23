"""Independent recomputation of the frozen A10 baseline-witness cohort W and of the B repairs.

Written from the rule text (ANALYSIS_RULES.md / A10_refresh_accounting.md) and the stored score fields only;
it does NOT import or execute the original replay module. Compared against the supplied per-target ledger.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path


def decode(row):
    lp = row['logps']
    finite = all(math.isfinite(x) for x in lp)
    valid = bool(finite and not row['tie'] and row['argmax_in_labels'])
    pred = int(lp[1] > lp[0]) if finite else None
    return pred, valid


def truth(values, spec):
    x = values[spec['a']][spec['p']]
    if spec['kind'] == 'direct':
        return int(x)
    if spec['kind'] == 'opposes':
        return int(not x)
    y = values[spec['b']][spec['p']]
    return int({'same': x == y, 'both': bool(x and y), 'either': bool(x or y)}[spec['kind']])


def main(archive, package_csv, out):
    base = 'reproducibility/question_blind_repair/scores/'
    rows, tot = [], Counter()
    with zipfile.ZipFile(archive) as z:
        designs = {}
        for line in gzip.decompress(z.read(base + 'designs.jsonl.gz')).splitlines():
            d = json.loads(line)
            designs[d['root']['root_id']] = d
        files = sorted(n for n in z.namelist() if n.startswith(base + 'primary-seed') and n.endswith('.jsonl.gz'))
        for name in files:
            for line in gzip.decompress(z.read(name)).splitlines():
                panel = json.loads(line)
                d, seed = designs[panel['root_id']], panel['seed']
                gold = {q['query_id']: truth(d['root']['terminal']['values'], q['spec']) for q in d['questions']}
                rec = {(r['context'], r['query_id']): decode(r) for r in panel['physical']}

                def good(ctx, q):
                    pred, valid = rec[ctx, q]
                    return valid and pred == gold[q]
                A = lambda o: f'o{o}/INV_PAIR_NLL_s{seed}'
                Bc = lambda o: f'B/o{o}'
                for q, m in d['plan']['gates']['targets'].items():
                    controls = d['plan']['gates']['direct'] + m['same_pair'] + m['cross_pair']
                    native_ok = good('NATIVE_FINAL', q)
                    disagree = any(rec[A(a), q][1] and rec[A(b), q][1] and rec[A(a), q][0] != rec[A(b), q][0]
                                   for a, b in d['plan']['structural_pairs'])
                    w = bool(native_ok and good(A(0), q) and all(good(A(o), c) for o in range(8) for c in controls) and disagree)
                    b_rep = bool(w and all(good(Bc(o), c) for o in range(8) for c in [q] + controls))
                    rows.append(dict(root_id=panel['root_id'], seed=seed, query_id=q, W=w, B_repair=b_rep))
                    tot['targets'] += 1
                    tot['W'] += w
                    tot['B_repair'] += b_rep
                tot['blocks'] += 1
    supplied = {(r['root_id'], int(r['seed']), r['query_id']): r for r in csv.DictReader(open(package_csv, newline=''))}
    mism = [r for r in rows if (str(r['W']) != supplied[(r['root_id'], r['seed'], r['query_id'])]['baseline_W'] or
                                str(r['B_repair']) != supplied[(r['root_id'], r['seed'], r['query_id'])]['A10_B_repair'])]
    res = dict(schema='NSR1_INDEPENDENT_W_V1', independent_of_original_replay_module=True, blocks=tot['blocks'],
               target_opportunities=tot['targets'], baseline_W=tot['W'], A10_B_repairs=tot['B_repair'],
               witness_designs=len({r['root_id'] for r in rows if r['W']}),
               W_by_seed={s: sum(r['W'] for r in rows if r['seed'] == s) for s in (0, 1)},
               mismatches_vs_supplied_ledger=len(mism), supplied_rows=len(supplied))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    w = csv.DictWriter(open(str(out).replace('.json', '_targets.csv'), 'w', newline=''), fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
