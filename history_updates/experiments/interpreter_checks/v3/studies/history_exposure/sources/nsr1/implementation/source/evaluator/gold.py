"""Evaluator-side gold labels. Separate file tree; imports nothing from the candidate (nsr/).

A deliberately small, independently written interpreter of the formal spec (docs/FORMAL_SPEC.md):
replays typed events from the initial records and evaluates the INTENDED program of each question.
"""
from __future__ import annotations

import json
from pathlib import Path


class GoldError(Exception):
    pass


def snapshots(initial, events):
    """List of dict snapshots [ORIGINAL, after event 1, ...]; duplicate initial addresses -> '<AMBIGUOUS>'."""
    s0, seen = {}, set()
    for w in initial:
        k = (w['entity'], w['field'], w['topic'])
        s0[k] = '<AMBIGUOUS>' if k in seen else w['value']
        seen.add(k)
    snaps = [s0]
    for ev in events:
        cur = dict(snaps[-1])
        new = {}
        for w in ev.get('writes', []):
            new[(w['entity'], w['field'], w['topic'])] = w['value']
        for r in ev.get('restores', []):
            src = snaps[0] if r['snapshot'] == 'ORIGINAL' else snaps[int(r['snapshot'].split(':')[1])]
            new[(r['entity'], r['field'], r['topic'])] = src[(r['entity'], r['field'], r['topic'])]
        cur.update(new)
        snaps.append(cur)
    return snaps


def _snap(snaps, label, upto):
    if label == 'CURRENT':
        return snaps[upto]
    if label == 'ORIGINAL':
        return snaps[0]
    k = int(label.split(':')[1])
    if not 1 <= k <= upto:
        raise GoldError('UNSUPPORTED_SNAPSHOT')
    return snaps[k]


def ev(node, snaps, upto):
    op = node['op']
    if op == 'READ':
        s = _snap(snaps, node['snapshot'], upto)
        k = (node['entity'], node['field'], node['topic'])
        if k not in s:
            raise GoldError('MISSING')
        if s[k] == '<AMBIGUOUS>':
            raise GoldError('AMBIGUOUS')
        return s[k]
    if op == 'FOLLOW':
        s = _snap(snaps, node['snapshot'], upto)
        cur = node['entity']
        for r in node['path']:
            if (cur, r, None) not in s:
                raise GoldError('MISSING')
            cur = s[(cur, r, None)]
            if cur == '<AMBIGUOUS>':
                raise GoldError('AMBIGUOUS')
        return cur
    if op == 'TEST':
        return ev(node['arg'], snaps, upto) == node['value']
    if op == 'SAME':
        return ev(node['left'], snaps, upto) == ev(node['right'], snaps, upto)
    if op == 'NOT':
        return not ev(node['arg'], snaps, upto)
    a, b = ev(node['left'], snaps, upto), ev(node['right'], snaps, upto)
    return (a and b) if op == 'BOTH' else (a or b)


def answer_string(value, mapping):
    if isinstance(value, bool):
        return mapping['true'] if value else mapping['false']
    return value


def gold_for_root(eroot):
    """Yield one gold row per (history, question) for an evaluator root."""
    rows = []
    for th in eroot['typed']:
        snaps = snapshots(th['initial'], th['events'])
        for q in eroot['questions']:
            upto = q.get('checkpoint_events', q.get('after_updates', len(th['events'])))
            val = ev(q['program'], snaps, upto)
            rows.append(dict(root_id=eroot['root_id'], history_id=th['history_id'], question_id=q['question_id'],
                             family=q.get('family'), kind=q.get('kind'), semantic=val,
                             gold=answer_string(val, q.get('mapping') or {}), aliases=q.get('aliases')))
    return rows


def build(panel_dir, out_path):
    rows = []
    for line in open(Path(panel_dir) / 'roots.jsonl', encoding='utf-8'):
        rows += gold_for_root(json.loads(line))
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    return rows


if __name__ == '__main__':
    import sys
    base = Path(sys.argv[1])
    for panel in sorted(p.name for p in (base / 'evaluator').iterdir() if (p / 'roots.jsonl').exists()):
        rows = build(base / 'evaluator' / panel, base / 'evaluator' / panel / 'gold.jsonl')
        print(panel, len(rows))
