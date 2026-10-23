"""Evaluator-side scoring: join candidate rows with gold. Separate process/file tree from the candidate.

Frozen answer rule: a Boolean answer is correct iff the produced label equals the gold label exactly (the
candidate-side rule already removed only chat controls, surrounding whitespace and at most one terminal
period). An entity/value answer is correct iff it equals the canonical label or a declared alias exactly;
only if that fails is one terminal period removed and the comparison repeated.
Every scheduled question is in the denominator: parser failures, abstentions and invalid outputs are errors.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path


def load_gold(data, panel):
    g = {}
    for line in open(Path(data) / 'evaluator' / panel / 'gold.jsonl', encoding='utf-8'):
        r = json.loads(line)
        g[(r['root_id'], r['history_id'], r['question_id'])] = r
    return g


def load_rows(out, panel):
    with gzip.open(Path(out) / panel / 'rows.jsonl.gz', 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def correct(answer, gold):
    if answer is None:
        return False
    ok = {gold['gold']} | set(gold.get('aliases') or [])
    if answer in ok:
        return True
    if isinstance(gold['semantic'], bool):
        return False
    return answer.endswith('.') and answer[:-1] in ok


def score_panel(data, out, panel):
    gold = load_gold(data, panel)
    scored = []
    for r in load_rows(out, panel):
        g = gold[(r['root_id'], r['history_id'], r['question_id'])]
        scored.append(dict(r, family=g['family'], kind=g['kind'], gold=g['gold'], semantic=g['semantic'],
                           valid=r['answer'] is not None, correct=correct(r['answer'], g)))
    return scored


def summarize(scored):
    agg = collections.OrderedDict()
    for r in scored:
        for key in ((r['arm'], 'ALL'), (r['arm'], r['family'])):
            a = agg.setdefault(key, dict(n=0, correct=0, invalid=0, errors=collections.Counter()))
            a['n'] += 1
            a['correct'] += r['correct']
            a['invalid'] += not r['valid']
            if not r['correct']:
                a['errors'][r.get('error') or 'WRONG_ANSWER'] += 1
    return agg


if __name__ == '__main__':
    data, out = sys.argv[1], sys.argv[2]
    for panel in sys.argv[3].split(','):
        s = summarize(score_panel(data, out, panel))
        for (arm, fam), a in s.items():
            print(f'{panel:14s} {arm:32s} {fam:11s} n={a["n"]:6d} acc={a["correct"] / a["n"]:.4f} invalid={a["invalid"]:5d} '
                  f'{dict(a["errors"].most_common(4)) if a["errors"] else ""}')
