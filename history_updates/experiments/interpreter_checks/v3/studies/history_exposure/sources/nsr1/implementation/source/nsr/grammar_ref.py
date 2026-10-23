"""Grammar-limited exact text parser (rule-based reference; NOT the neural candidate).

Uses: (1) CPU dry runs of the whole pipeline before any GPU spend, (2) the additional labelled
A10 reference track (never swapped in after neural failures), (3) an independent check that the
visible text of every fresh panel determines the evaluator's gold answer.
It emulates a Gen backend by reading the last user turn and emitting the JSON / answer a perfect
template parser would produce. It only knows the fixed surface templates; anything else -> UNSUPPORTED.
"""
from __future__ import annotations

import json
import re

from . import core, render
from .pipeline import FlatStore, Gen

N = r'(.+?)'
NL = chr(10)
Q = [  # (regex, builder)
    (rf'^Is {N} currently in favor of the {N} proposal\?$', lambda a, p: _fav(a, p)),
    (rf'^Is {N} currently against the {N} proposal\?$', lambda a, p: _test(a, p, 'oppose')),
    (rf'^Are {N} and {N} on the same side of the {N} proposal\?$', lambda a, b, p: _same(a, b, p)),
    (rf'^Do {N} and {N} both support the {N} proposal\?$', lambda a, b, p: _bin('BOTH', a, b, p)),
    (rf'^Does {N} or {N} \(or both\) support the {N} proposal\?$', lambda a, b, p: _bin('EITHER', a, b, p)),
    (rf'^In the original records, before any correction, was {N} in favor of the {N} proposal\?$',
     lambda a, p: _fav(a, p, 'ORIGINAL')),
    (rf'^At this point, is {N} in favor of the {N} proposal\?$', lambda a, p: _fav(a, p)),
    (rf'^At this point, do {N} and {N} both support the {N} proposal\?$', lambda a, b, p: _bin('BOTH', a, b, p)),
    (rf'^At this point, does {N} or {N} \(or both\) support the {N} proposal\?$', lambda a, b, p: _bin('EITHER', a, b, p)),
    (rf'^At this point, are {N} and {N} on the same side of the {N} proposal\?$', lambda a, b, p: _same(a, b, p)),
    (rf'^Right after correction (\d+), was {N} in favor of the {N} proposal\?$', lambda k, a, p: _fav(a, p, f'EVENT:{k}')),
    # original A10 wording
    (rf'^Does {N} favor the {N} proposal\?$', lambda a, p: _fav(a, p)),
    (rf'^Does {N} oppose the {N} proposal\?$', lambda a, p: _test(a, p, 'oppose')),
    (rf'^Do {N} and {N} take the same side on {N}\?$', lambda a, b, p: _same(a, b, p)),
    (rf'^Is the {N} proposal supported by both {N} and {N}\?$', lambda p, a, b: _bin('BOTH', a, b, p)),
    (rf'^Does at least one of {N} and {N} support {N}\?$', lambda a, b, p: _bin('EITHER', a, b, p)),
    # workload
    (rf"^What is the current status of {N}'s {N} record\?$", lambda a, p: _read(a, p, 'status')),
    (rf"^What is the current shipment day of {N}'s {N} record\?$", lambda a, p: _read(a, p, 'day')),
    (rf"^In the original records, before any correction, what was the status of {N}'s {N} record\?$",
     lambda a, p: _read(a, p, 'status', 'ORIGINAL')),
    (rf'^Do {N} and {N} currently have the same shipment day on the {N} proposal\?$',
     lambda a, b, p: dict(op='SAME', left=_read(a, p, 'day'), right=_read(b, p, 'day'))),
    # graph
    (rf'^Which team is {N} on\?$', lambda s: _follow(s, ['team'])),
    (rf'^Name the team that {N} belongs to\.$', lambda s: _follow(s, ['team'])),
    (rf'^Which organization does {N} work for\?$', lambda s: _follow(s, ['organization'])),
    (rf'^Name the employer of {N}\.$', lambda s: _follow(s, ['organization'])),
    (rf"^Which hub is {N}'s team based at\?$", lambda s: _follow(s, ['team', 'hub'])),
    (rf'^Name the hub where the team of {N} is based\.$', lambda s: _follow(s, ['team', 'hub'])),
    (rf'^Who founded the organization that {N} works for\?$', lambda s: _follow(s, ['organization', 'founder'])),
    (rf"^Name the founder of {N}'s employer\.$", lambda s: _follow(s, ['organization', 'founder'])),
    (rf"^Which region is the hub of {N}'s team in\?$", lambda s: _follow(s, ['team', 'hub', 'region'])),
    (rf'^{N} is on a team based at some hub\. Name the region of that hub\.$', lambda s: _follow(s, ['team', 'hub', 'region'])),
    (rf"^Which city does the founder of {N}'s organization live in\?$", lambda s: _follow(s, ['organization', 'founder', 'location'])),
    (rf'^Name the home city of the person who founded the organization {N} works for\.$',
     lambda s: _follow(s, ['organization', 'founder', 'location'])),
    (rf"^Which hub is the team of the founder of {N}'s organization based at\?$",
     lambda s: _follow(s, ['organization', 'founder', 'team', 'hub'])),
    (rf"^Name the hub where the founder of {N}'s employer has their team based\.$",
     lambda s: _follow(s, ['organization', 'founder', 'team', 'hub'])),
]
Q = [(re.compile(rx), fn) for rx, fn in Q]
HIST_PREFIX = 'In the original records, before any correction: '


def _read(a, p, field='stance', snap='CURRENT'):
    return dict(op='READ', entity=a, field=field, topic=p, snapshot=snap)


def _test(a, p, v, snap='CURRENT'):
    return dict(op='TEST', arg=_read(a, p, 'stance', snap), value=v)


def _fav(a, p, snap='CURRENT'):
    return _test(a, p, 'favor', snap)


def _same(a, b, p):
    return dict(op='SAME', left=_read(a, p), right=_read(b, p))


def _bin(op, a, b, p):
    return dict(op=op, left=_fav(a, p), right=_fav(b, p))


def _follow(s, path, snap='CURRENT'):
    return dict(op='FOLLOW', entity=s, path=path, snapshot=snap)


def compile_question(text):
    first = text.split('\n')[0]
    snap = None
    if first.startswith(HIST_PREFIX):
        rest = first[len(HIST_PREFIX):]
        first, snap = rest, 'ORIGINAL'
        if not any(rx.match(first) for rx, _ in Q):
            first = rest[0].upper() + rest[1:]
    for rx, fn in Q:
        m = rx.match(first)
        if m:
            prog = fn(*m.groups())
            if snap:
                prog = dict(prog, snapshot=snap)
            return prog
    return dict(op='UNSUPPORTED', reason='no grammar rule')


STX = {'in favor': 'favor', 'opposed': 'oppose'}
REC = [
    (re.compile(rf'^Record \S+: {N} on the {N} proposal: (in favor|opposed); status (\w+); shipment day (\w+)\.$'),
     lambda a, p, s, st, d: dict(kind='member', person=a, project=p, stance=STX[s], status=st, day=d)),
    (re.compile(rf'^{N} is in favor of the {N} proposal\.$'), lambda a, p: dict(kind='member', person=a, project=p, stance='favor')),
    (re.compile(rf'^{N} is opposed to the {N} proposal\.$'), lambda a, p: dict(kind='member', person=a, project=p, stance='oppose')),
    (re.compile(rf'^Record \S+: {N} is on the {N} team\.$'), lambda s, o: dict(kind='relation', entity=s, relation='team', value=o)),
    (re.compile(rf'^Record \S+: The {N} team is based at the {N} hub\.$'), lambda s, o: dict(kind='relation', entity=s, relation='hub', value=o)),
    (re.compile(rf'^Record \S+: The {N} hub is in the {N} region\.$'), lambda s, o: dict(kind='relation', entity=s, relation='region', value=o)),
    (re.compile(rf'^Record \S+: {N} works for {N}\.$'), lambda s, o: dict(kind='relation', entity=s, relation='organization', value=o)),
    (re.compile(rf'^Record \S+: {N} was founded by {N}\.$'), lambda s, o: dict(kind='relation', entity=s, relation='founder', value=o)),
    (re.compile(rf'^Record \S+: {N} lives in {N}\.$'), lambda s, o: dict(kind='relation', entity=s, relation='location', value=o)),
]


def parse_record(line):
    for rx, fn in REC:
        m = rx.match(line)
        if m:
            return fn(*m.groups())
    return dict(kind='none')


FLD = {'stance': 'stance', 'status': 'status', 'shipment day': 'day'}
CLAUSE = [
    (re.compile(rf'^{N} now favors the {N} proposal$', re.I), lambda a, p: ('w', a, 'stance', p, 'favor')),
    (re.compile(rf'^{N} now opposes the {N} proposal$', re.I), lambda a, p: ('w', a, 'stance', p, 'oppose')),
    (re.compile(rf"^the status of {N}'s {N} record is now (\w+)$", re.I), lambda a, p, v: ('w', a, 'status', p, v)),
    (re.compile(rf"^{N}'s {N} shipment day is now (\w+)$", re.I), lambda a, p, v: ('w', a, 'day', p, v)),
    (re.compile(rf"^{N}'s {N} (stance|status|shipment day) is restored to its original value$", re.I),
     lambda a, p, f: ('r', a, FLD[f], p, 'ORIGINAL')),
    (re.compile(rf"^{N}'s {N} (stance|status|shipment day) is restored to the value it had right after correction (\d+)$", re.I),
     lambda a, p, f, k: ('r', a, FLD[f], p, f'EVENT:{k}')),
    (re.compile(rf'^{N} has moved to the {N} team$'), lambda s, o: ('w', s, 'team', None, o)),
    (re.compile(rf'^The {N} team is now based at the {N} hub$'), lambda s, o: ('w', s, 'hub', None, o)),
    (re.compile(rf'^The {N} hub is now in the {N} region$'), lambda s, o: ('w', s, 'region', None, o)),
    (re.compile(rf'^{N} now works for {N}$'), lambda s, o: ('w', s, 'organization', None, o)),
    (re.compile(rf'^{N} is now recorded as founded by {N}$'), lambda s, o: ('w', s, 'founder', None, o)),
    (re.compile(rf'^{N} now lives in {N}$'), lambda s, o: ('w', s, 'location', None, o)),
    (re.compile(rf'^the (\w+) of {N} is restored to its original value$'), lambda f, s: ('r', s, f, None, 'ORIGINAL')),
]


def parse_update(stmt):
    m = re.match(r'^Correction \S+(?: \(applied together\))?: (.*)\.$', stmt)
    if not m:
        return {}
    ev = {}
    for clause in m.group(1).split('; '):
        hit = None
        for rx, fn in CLAUSE:
            mm = rx.match(clause)
            if mm:
                hit = fn(*mm.groups())
                break
        if hit is None:
            return {}
        kind, e, f, t, v = hit
        if kind == 'w':
            if e[0].islower() and f in core.ENUMS:
                e = e[0].upper() + e[1:]
            ev.setdefault('writes', []).append(dict(entity=_fix(e, stmt), field=f, topic=t, value=v))
        else:
            ev.setdefault('restores', []).append(dict(entity=_fix(e, stmt), field=f, topic=t, snapshot=v))
    return ev


def _fix(name, stmt):
    """Sentence-initial capitalization: recover the exact name as it appears in the statement."""
    if name in stmt:
        return name
    alt = name[0].upper() + name[1:]
    return alt if alt in stmt else name


class GrammarGen(Gen):
    """Emulates every arm with exact template parsing (perfect within the grammar, UNSUPPORTED outside)."""
    model_id = 'grammar-reference'

    def run(self, reqs):
        return [self._one(r) for r in reqs]

    def _one(self, req):
        tag, user = req.get('tag', ''), req['messages'][1]['content']
        kind = tag.split(':')[1] if ':' in tag else tag
        if kind == 'warm':
            text = ''
        elif kind == 'exposed':
            text = json.dumps(compile_question(user.rsplit('\n\nQuestion: ', 1)[1][:-len('\nProgram:')]))
        elif kind == 'query':
            text = json.dumps(compile_question(user[len('Question: '):-len('\nProgram:')]))
        elif kind == 'record':
            text = json.dumps(parse_record(user[len('Line: '):-len('\nJSON:')]))
        elif kind == 'update':
            text = json.dumps(parse_update(user[len('Statement: '):-len('\nJSON:')]))
        else:
            text = self._solve(kind, user)
        return dict(text=text, token_ids=[], finish='stop', n_prompt=len(user) // 4, n_out=len(text) // 4, stop_token=None)

    def _solve(self, kind, user):
        head, question = user.rsplit('\n\nQuestion: ', 1)
        question = question.rsplit('\n', 1)[0] if kind != 'renderer' else question
        if kind == 'renderer':
            question, computed = question.split('\nComputed answer: ')
            return computed.rsplit('\nFinal answer:', 1)[0]
        prog = compile_question(question)
        if kind == 'current':
            cur, orig = head[len('CURRENT records:' + NL):].split(NL + 'ORIGINAL records:' + NL)
            store = FlatStore([dict(f, snapshot='CURRENT') for f in _table(cur)] +
                              [dict(f, snapshot='ORIGINAL') for f in _table(orig)])
            res = core.execute(prog, store) if prog['op'] != 'UNSUPPORTED' else dict(status='ERROR', error='UNSUPPORTED')
            ans, err = render.render(res, question)
            return ans if ans is not None else 'UNKNOWN'
        recs, ups = head[len('Records:\n'):].split('\nCorrections:\n')
        writes, src = [], {}
        for line in recs.split('\n'):
            rid = 'Record ' + line.split(':')[0].split(' ')[1] if line.startswith('Record ') else None
            obj = parse_record(line)
            if obj['kind'] == 'member':
                ws = [dict(entity=obj['person'], field=f, topic=obj['project'], value=obj[f]) for f in ('stance', 'status', 'day') if f in obj]
            elif obj['kind'] == 'relation':
                ws = [dict(entity=obj['entity'], field=obj['relation'], topic=None, value=obj['value'])]
            else:
                ws = []
            for w in ws:
                src[(0, w['entity'], w['field'], w['topic'])] = rid
            writes += ws
        store = core.Store(writes)
        last = {(k[1], k[2], k[3]): (0, v) for k, v in src.items()}
        hist = [dict(last)]
        if ups != '(none)':
            for line in ups.split('\n'):
                ev = parse_update(line)
                out = store.apply(ev)
                if out['status'] == 'ACCEPTED':
                    cid = 'Correction ' + line.split(' ')[1].rstrip(':')
                    for w in ev.get('writes', []) + ev.get('restores', []):
                        last[(w['entity'], w['field'], w['topic'])] = (out['index'], cid)
                hist.append(dict(last))
        if prog['op'] == 'UNSUPPORTED':
            return json.dumps(prog) if kind == 'flat' else 'UNKNOWN'
        res = core.execute(prog, store)
        if kind == 'flat':
            facts = []
            leaves = core._leaves(prog, [])
            labels = {}
            for leaf in leaves:
                labels[store.resolve(leaf['snapshot']).index] = leaf['snapshot']
            for idx, e, f, t in res['read_set']:
                snap = store.snapshots[idx]
                if (e, f, t) not in snap.values:
                    continue
                facts.append(dict(source=hist[idx][(e, f, t)][1], entity=e, field=f, topic=t, value=snap.values[(e, f, t)],
                                  snapshot=labels[idx]))
            return json.dumps(dict(program=prog, facts=facts))
        ans, err = render.render(res, question)
        return ans if ans is not None else 'UNKNOWN'


def _table(text):
    facts = []
    for line in text.split(NL):
        obj = parse_record(line)
        if obj['kind'] == 'member':
            facts += [dict(entity=obj['person'], field=f, topic=obj['project'], value=obj[f]) for f in ('stance', 'status', 'day') if f in obj]
        elif obj['kind'] == 'relation':
            facts.append(dict(entity=obj['entity'], field=obj['relation'], topic=None, value=obj['value']))
    return facts
