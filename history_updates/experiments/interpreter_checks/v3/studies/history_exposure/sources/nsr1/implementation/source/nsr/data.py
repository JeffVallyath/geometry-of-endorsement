"""NSR1 fresh panel generators (seeds from NSR1_CONFIG.json).

Every root is written as VISIBLE inputs (record/update/question text only) and EVALUATOR-only material
(typed records/events, intended programs, labels, cells) in separate trees. The runtime candidate reads
only the visible tree. Gold answers are computed later by evaluator/gold.py, not here.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

FIRST = ['Abel', 'Brisa', 'Calder', 'Delphine', 'Emrys', 'Fenna', 'Galen', 'Hesper', 'Idris', 'Jorun',
         'Kasimir', 'Linnea', 'Marek', 'Noemi', 'Osric', 'Perpetua', 'Quillon', 'Rosalind', 'Severin', 'Thea',
         'Ulric', 'Valka', 'Wendel', 'Xanthe', 'Yorick', 'Zelie', 'Anselm', 'Bettany', 'Corvin', 'Dagny',
         'Eamon', 'Florian', 'Gwendolyn', 'Halvard', 'Ilse', 'Jasper', 'Katrin', 'Leander', 'Mireille', 'Nils']
LAST = ['Achterberg', 'Bellweather', 'Castellane', 'Drummond', 'Eckhart', 'Fairweather', 'Galloway',
        'Holloway', 'Ingersoll', 'Jaramillo', 'Kavanagh', 'Lindqvist', 'Marchetti', 'Northcott', 'Oyelaran',
        'Pemberton', 'Quintero', 'Rasmussen', 'Sandoval', 'Thorsby', 'Underhill', 'Vasquez', 'Whitlock',
        'Yarborough', 'Zielinski', 'Abernathy', 'Blackwood', 'Carrow', 'Delacroix', 'Esterhazy']
PROJECTS = ['Kestrel', 'Lantern', 'Meridian', 'Nimbus', 'Obsidian', 'Pinnacle', 'Quarry', 'Rampart', 'Sapphire',
            'Tributary', 'Umbra', 'Vantage', 'Willow', 'Zephyr', 'Aurora', 'Basalt', 'Cobalt', 'Driftwood',
            'Ember', 'Fjord', 'Granite', 'Hollow', 'Isthmus', 'Juniper', 'Keystone', 'Lagoon', 'Monsoon',
            'Nettle', 'Orchid', 'Pylon', 'Quartz', 'Riverbend', 'Saltmarsh', 'Thicket', 'Upland', 'Verdant']
TEAMS = ['Falcon', 'Heron', 'Osprey', 'Kingfisher', 'Plover', 'Lapwing', 'Merlin', 'Shrike', 'Tern', 'Wren',
         'Avocet', 'Bittern', 'Curlew', 'Dunlin', 'Egret', 'Fulmar', 'Gannet', 'Harrier', 'Ibis', 'Jacana']
HUBS = ['Delta', 'Sigma', 'Kappa', 'Lambda', 'Omicron', 'Upsilon', 'Theta', 'Zeta', 'Rho', 'Tau', 'Phi',
        'Chi', 'Psi', 'Iota', 'Epsilon', 'Gamma']
REGIONS = ['North', 'South', 'East', 'West', 'Highland', 'Lowland', 'Coastal', 'Central', 'Frontier', 'Valley']
ORGS = ['Lumen Labs', 'Cobalt Works', 'Halcyon Group', 'Brightwater Company', 'Ironleaf Holdings', 'Northwind Partners',
        'Silverpine Studio', 'Redfern Systems', 'Bluestone Guild', 'Amberline Industries', 'Greyhaven Trust',
        'Copperfield Ventures', 'Starling Collective', 'Ravenmoor Holdings', 'Westgate Foundry', 'Oakhurst Media']
CITIES = ['Porto', 'Tallinn', 'Ghent', 'Bergen', 'Lyon', 'Krakow', 'Bilbao', 'Turku', 'Graz', 'Aarhus', 'Utrecht',
          'Bologna', 'Brno', 'Cork', 'Leipzig', 'Malmo']
STANCE_TXT = {'favor': 'in favor', 'oppose': 'opposed'}
STATUS = ['pending', 'approved', 'rejected', 'deferred']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def rng(seed, *parts):
    return random.Random(hashlib.sha256('|'.join(map(str, (seed,) + parts)).encode()).hexdigest())


def people(r, n):
    out = set()
    while len(out) < n:
        out.add(f'{r.choice(FIRST)} {r.choice(LAST)}')
    return sorted(out, key=lambda _: r.random())


# ----------------------------- rendering -----------------------------
def record_text(rid, w):
    """Compact consistent member-record line (all three fields present at every size)."""
    return (f'Record {rid}: {w["person"]} on the {w["project"]} proposal: {STANCE_TXT[w["stance"]]}; '
            f'status {w["status"]}; shipment day {w["day"]}.')


def member_writes(m):
    return [dict(entity=m['person'], field=f, topic=m['project'], value=m[f]) for f in ('stance', 'status', 'day')]


def update_text(uid, ops):
    """ops: list of ('set', person, project, field, value) | ('restore', person, project, field, snapshot)."""
    parts = []
    for op in ops:
        kind, p, proj, field, val = op
        if kind == 'set':
            if field == 'stance':
                parts.append(f'{p} now {"favors" if val == "favor" else "opposes"} the {proj} proposal')
            elif field == 'status':
                parts.append(f'the status of {p}\'s {proj} record is now {val}')
            else:
                parts.append(f'{p}\'s {proj} shipment day is now {val}')
        else:
            when = 'its original value' if val == 'ORIGINAL' else f'the value it had right after correction {val[6:]}'
            what = {'stance': 'stance', 'status': 'status', 'day': 'shipment day'}[field]
            parts.append(f'{p}\'s {proj} {what} is restored to {when}')
    body = '; '.join(parts)
    return f'Correction {uid}: {body[0].upper() + body[1:]}.' if len(ops) == 1 else \
        f'Correction {uid} (applied together): {body[0].upper() + body[1:]}.'


def typed_event(ops):
    ev = dict(writes=[], restores=[])
    for kind, p, proj, field, val in ops:
        if kind == 'set':
            ev['writes'].append(dict(entity=p, field=field, topic=proj, value=val))
        else:
            ev['restores'].append(dict(entity=p, field=field, topic=proj, snapshot=val))
    return {k: v for k, v in ev.items() if v}


def fmt_suffix(r, draw):
    """Answer instruction. draw 0: No/Yes; draw 1: A/B with a hashed swap (A10-shaped)."""
    if draw == 0:
        return '\nAnswer with exactly one of No, Yes.', {True: 'Yes', False: 'No'}
    yes, no = ('A', 'B') if r.randrange(2) else ('B', 'A')
    labels = sorted([yes, no])
    return f'\nUse {yes} for yes and {no} for no.\nAnswer with exactly one of {", ".join(labels)}.', {True: yes, False: no}


# Stance question wordings for fresh panels (new phrasings; A10 wording is not reused).
WORD = {
    'direct': 'Is {a} currently in favor of the {p} proposal?',
    'opposes': 'Is {a} currently against the {p} proposal?',
    'same': 'Are {a} and {b} on the same side of the {p} proposal?',
    'both': 'Do {a} and {b} both support the {p} proposal?',
    'either': 'Does {a} or {b} (or both) support the {p} proposal?',
    'hist_direct': 'In the original records, before any correction, was {a} in favor of the {p} proposal?',
}


def rd(person, project, snap='CURRENT', field='stance'):
    return dict(op='READ', entity=person, field=field, topic=project, snapshot=snap)


def intended(kind, a, b, p, snap='CURRENT'):
    fav = lambda x: dict(op='TEST', arg=rd(x, p, snap), value='favor')
    if kind == 'direct':
        return fav(a)
    if kind == 'opposes':
        return dict(op='TEST', arg=rd(a, p, snap), value='oppose')
    if kind == 'same':
        return dict(op='SAME', left=rd(a, p, snap), right=rd(b, p, snap))
    if kind == 'both':
        return dict(op='BOTH', left=fav(a), right=fav(b))
    if kind == 'either':
        return dict(op='EITHER', left=fav(a), right=fav(b))
    raise ValueError(kind)


def random_member(r, person, project):
    return dict(person=person, project=project, stance=r.choice(['favor', 'oppose']),
                status=r.choice(STATUS), day=r.choice(DAYS))


# ----------------------------- core / boundary / dev -----------------------------
def core_root(seed, ns, i, combo):
    """32 actor-project records (8 people x 4 projects); 3 addresses vary over 8 origins and receive
    identical 3 SET commands; every origin ends at the same intended current state."""
    r = rng(seed, ns, i)
    ppl = people(r, 8)
    projs = r.sample(PROJECTS, 4)
    p, o = projs[0], projs[1]
    a, b, c, d = ppl[0], ppl[1], ppl[2], ppl[3]
    terminal = {}
    for x in ppl:
        for y in projs:
            terminal[(x, y)] = random_member(r, x, y)
    for who, bit in zip((a, b, c), combo):  # balanced AND/OR/SAME cells over roots, fixed before execution
        terminal[(who, p)]['stance'] = 'favor' if bit else 'oppose'
    order = list(terminal)
    r.shuffle(order)
    rid = {k: f'{n + 1:02d}' for n, k in enumerate(order)}
    commands = [('set', who, p, 'stance', terminal[(who, p)]['stance']) for who in (a, b, c)]
    flip = {'favor': 'oppose', 'oppose': 'favor'}
    histories, typed = [], []
    for h in range(8):
        src = {k: dict(v) for k, v in terminal.items()}
        for j, who in enumerate((a, b, c)):
            if (h >> j) & 1:
                src[(who, p)]['stance'] = flip[src[(who, p)]['stance']]
        recs = [dict(record_id=rid[k], text=record_text(rid[k], src[k])) for k in order]
        ups = [dict(update_id=str(n + 1), text=update_text(str(n + 1), [cmd])) for n, cmd in enumerate(commands)]
        histories.append(dict(history_id=f'h{h}', records=recs, updates=ups))
        typed.append(dict(history_id=f'h{h}', initial=[w for k in order for w in member_writes(src[k])],
                          events=[typed_event([cmd]) for cmd in commands]))
    specs = [('direct', a, b, p), ('opposes', a, b, p), ('same', a, b, p), ('direct', b, a, p),
             ('opposes', b, a, p), ('direct', a, b, o), ('both', a, b, p), ('either', a, b, p),
             ('same', c, a, p), ('both', c, a, p), ('either', c, a, p), ('direct', c, a, o),
             ('direct', c, a, p), ('same', b, c, p), ('both', b, c, p), ('either', b, c, p), ('direct', d, a, p)]
    qs, ev = [], []
    for draw in (0, 1):
        for j, (kind, x, y, pp) in enumerate(specs):
            suffix, mapping = fmt_suffix(rng(seed, ns, i, 'fmt', draw, j), draw)
            qid = f'{ns}-{i:04d}-d{draw}-{j:02d}'
            qs.append(dict(question_id=qid, text=WORD[kind].format(a=x, b=y, p=pp) + suffix))
            fam = 'joint' if kind in ('same', 'both', 'either') else kind
            ev.append(dict(question_id=qid, family=fam, kind=kind, draw=draw, mapping=_m(mapping),
                           program=intended(kind, x, y, pp), operands=[x, y] if fam == 'joint' else [x],
                           target_pair=_pair(x, y, a, b, c) if fam == 'joint' else None))
        for j, x in enumerate((a, b)):
            suffix, mapping = fmt_suffix(rng(seed, ns, i, 'hfmt', draw, j), draw)
            qid = f'{ns}-{i:04d}-d{draw}-h{j}'
            qs.append(dict(question_id=qid, text=WORD['hist_direct'].format(a=x, p=p) + suffix))
            ev.append(dict(question_id=qid, family='historical', kind='direct', draw=draw, mapping=_m(mapping),
                           program=intended('direct', x, None, p, 'ORIGINAL'), operands=[x], target_pair=None))
    return dict(root_id=f'{ns}-{i:04d}', histories=histories, questions=qs), \
        dict(root_id=f'{ns}-{i:04d}', typed=typed, questions=ev, combo=list(combo),
             roles=dict(a=a, b=b, c=c, d=d, p=p, o=o))


def _m(mapping):
    return {'true': mapping[True], 'false': mapping[False]}


def _pair(x, y, a, b, c):
    s = {x, y}
    return 'ab' if s == {a, b} else 'ca' if s == {a, c} else 'bc'


def core_panel(seed, ns, n):
    combos = [((k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1) for k in range(8)]
    r = rng(seed, ns, 'combo-order')
    seq = []
    while len(seq) < n:
        block = combos[:]
        r.shuffle(block)
        seq += block
    return [core_root(seed, ns, i, seq[i]) for i in range(n)]


# ----------------------------- factorial / locality -----------------------------
def factorial_panel(seed, n_roots, sizes=(8, 32, 128), ns='NSR1-FACT'):
    vis, ev = [], []
    for i in range(n_roots):
        r = rng(seed, ns, i)
        ppl = people(r, 40)
        projs = r.sample(PROJECTS, 4)
        x, y, p = ppl[0], ppl[1], projs[0]
        base_x, base_y = random_member(r, x, p), random_member(r, y, p)
        pool = [(u, v) for u in ppl[2:] for v in projs][:max(sizes) - 2]
        r.shuffle(pool)
        irr = {k: random_member(r, *k) for k in pool}
        questions, qev = [], []
        ops = ['both', 'either', 'same']
        for draw in (0, 1):
            for o_i, (u, v) in enumerate(((x, y), (y, x))):
                for kind in ops:
                    suffix, mapping = fmt_suffix(rng(seed, ns, i, 'fmt', draw, o_i, kind), draw)
                    qid = f'{ns}-{i:04d}-d{draw}-{"xy" if o_i == 0 else "yx"}-{kind}'
                    questions.append(dict(question_id=qid, text=WORD[kind].format(a=u, b=v, p=p) + suffix))
                    qev.append(dict(question_id=qid, kind=kind, draw=draw, order=o_i, mapping=_m(mapping),
                                    program=intended(kind, u, v, p), family='joint'))
        dbs, tdbs = [], []
        for N in sizes:
            keys = pool[:N - 2]
            for state in range(4):
                vx, vy = ('favor' if state & 2 else 'oppose'), ('favor' if state & 1 else 'oppose')
                mx, my = dict(base_x, stance=vx), dict(base_y, stance=vy)
                rows = [(x, p, mx), (y, p, my)] + [(k[0], k[1], irr[k]) for k in keys]
                pos = rng(seed, ns, i, 'order', N)  # same layout for all four states of one N
                idx = list(range(len(rows)))
                pos.shuffle(idx)
                rows = [rows[j] for j in idx]
                recs = [dict(record_id=f'{n + 1:03d}', text=record_text(f'{n + 1:03d}', m)) for n, (_, _, m) in enumerate(rows)]
                db_id = f'N{N}-s{state}'
                dbs.append(dict(history_id=db_id, records=recs, updates=[]))
                tdbs.append(dict(history_id=db_id, N=N, state=state, operands=dict(x=vx, y=vy),
                                 initial=[w for (_, _, m) in rows for w in member_writes(m)], events=[]))
        vis.append(dict(root_id=f'{ns}-{i:04d}', histories=dbs, questions=questions))
        ev.append(dict(root_id=f'{ns}-{i:04d}', typed=tdbs, questions=qev, roles=dict(x=x, y=y, p=p)))
    return list(zip(vis, ev))


# ----------------------------- temporal -----------------------------
FAMILIES = ['duplicates', 'restore', 'disjoint_order', 'repeat_overwrite', 'historical_return',
            'old_length_variation', 'irrelevant_archive_growth', 'atomic_multi_address']
TWORD = {
    'cur_x': 'At this point, is {a} in favor of the {p} proposal?',
    'cur_y': 'At this point, is {a} in favor of the {p} proposal?',
    'and': 'At this point, do {a} and {b} both support the {p} proposal?',
    'or': 'At this point, does {a} or {b} (or both) support the {p} proposal?',
    'same': 'At this point, are {a} and {b} on the same side of the {p} proposal?',
    'unchanged': 'At this point, is {a} in favor of the {p} proposal?',
    'hist': 'Right after correction {k}, was {a} in favor of the {p} proposal?',
    'hist0': 'In the original records, before any correction, was {a} in favor of the {p} proposal?',
}


def temporal_panel(seed, n_programs, ns='NSR1-TEMP'):
    out = []
    flip = {'favor': 'oppose', 'oppose': 'favor'}
    for i in range(n_programs):
        fam = FAMILIES[i % len(FAMILIES)]
        r = rng(seed, ns, i)
        ppl = people(r, 10)
        projs = r.sample(PROJECTS, 3)
        p = projs[0]
        x, y, z = ppl[0], ppl[1], ppl[2]
        others = [(u, v) for u in ppl[3:] for v in projs]
        r.shuffle(others)
        base = {(u, v): random_member(r, u, v) for u in ppl for v in projs}
        # Target current (x, y) stance at checkpoints after events 2, 4, 6, 8: all four cells cycle.
        cells = [(1, 1), (1, 0), (0, 1), (0, 0)]
        r.shuffle(cells)
        target = [{x: 'favor' if cx else 'oppose', y: 'favor' if cy else 'oppose'} for cx, cy in cells]
        trajs = []
        for t in range(2):  # matched trajectories: same current (x,y) at each checkpoint, different history
            tr = rng(seed, ns, i, 'traj', t)
            init = {k: dict(v) for k, v in base.items()}
            if t == 1:  # different obsolete facts
                init[(x, p)]['stance'] = flip[init[(x, p)]['stance']]
                init[(y, p)]['stance'] = flip[init[(y, p)]['stance']]
            events, cur = [], {k: dict(v) for k, v in init.items()}
            for cp in range(4):
                tx, ty = target[cp][x], target[cp][y]
                pair = _family_events(fam, tr, t, cp, x, y, p, tx, ty, cur, others, init)
                for ops in pair:
                    events.append(ops)
                    for kind, u, v, f, val in ops:
                        if kind == 'set':
                            cur[(u, v)][f] = val
                        else:
                            cur[(u, v)][f] = _value_at(init, events[:-1], (u, v), f, val)
                assert cur[(x, p)]['stance'] == tx and cur[(y, p)]['stance'] == ty, (fam, t, cp)
            assert len(events) == 8
            order = list(init)
            rr = rng(seed, ns, i, 'layout')
            rr.shuffle(order)
            rid = {k: f'{n + 1:02d}' for n, k in enumerate(order)}
            trajs.append(dict(order=order, rid=rid, init=init, events=events))
        vis_h, typ_h = [], []
        for t, tr in enumerate(trajs):
            recs = [dict(record_id=tr['rid'][k], text=record_text(tr['rid'][k], tr['init'][k])) for k in tr['order']]
            ups = [dict(update_id=str(n + 1), text=update_text(str(n + 1), ops)) for n, ops in enumerate(tr['events'])]
            vis_h.append(dict(history_id=f't{t}', records=recs, updates=ups))
            typ_h.append(dict(history_id=f't{t}', initial=[w for k in tr['order'] for w in member_writes(tr['init'][k])],
                              events=[typed_event(ops) for ops in tr['events']]))
        qs, qev = [], []
        for cp in range(4):
            k_event = 2 * (cp + 1)
            hist_k = r.choice([0, 1]) if cp == 0 else r.randrange(1, k_event)
            for draw in (0, 1):
                specs = [('cur_x', 'direct', x, None, 'CURRENT'), ('cur_y', 'direct', y, None, 'CURRENT'),
                         ('and', 'both', x, y, 'CURRENT'), ('or', 'either', x, y, 'CURRENT'),
                         ('same', 'same', x, y, 'CURRENT'), ('unchanged', 'direct', z, None, 'CURRENT'),
                         ('hist0' if hist_k == 0 else 'hist', 'direct', x, None,
                          'ORIGINAL' if hist_k == 0 else f'EVENT:{hist_k}')]
                for j, (wk, kind, u, v, snap) in enumerate(specs):
                    suffix, mapping = fmt_suffix(rng(seed, ns, i, 'fmt', cp, draw, j), draw)
                    qid = f'{ns}-{i:04d}-c{cp}-d{draw}-{wk}'
                    qs.append(dict(question_id=qid, checkpoint_events=k_event,
                                   text=TWORD[wk].format(a=u, b=v, p=p, k=hist_k) + suffix))
                    famq = 'joint' if kind in ('both', 'either', 'same') else ('historical' if snap != 'CURRENT' else
                                                                               ('unchanged' if wk == 'unchanged' else 'direct'))
                    qev.append(dict(question_id=qid, checkpoint_events=k_event, family=famq, kind=kind, draw=draw,
                                    mapping=_m(mapping), program=intended(kind, u, v, p, snap)))
        out.append((dict(root_id=f'{ns}-{i:04d}', family=fam, histories=vis_h, questions=qs),
                    dict(root_id=f'{ns}-{i:04d}', family=fam, typed=typ_h, questions=qev, roles=dict(x=x, y=y, z=z, p=p))))
    return out


def _value_at(init, events, addr, field, snap):
    if snap == 'ORIGINAL':
        return init[addr][field]
    k = int(snap[6:])
    val = init[addr][field]
    for ops in events[:k]:
        for kind, u, v, f, value in ops:
            if (u, v) == addr and f == field and kind == 'set':
                val = value
    return val


def _family_events(fam, r, t, cp, x, y, p, tx, ty, cur, others, init):
    """Two events per checkpoint that reach the target current (x,y) values; family-specific history."""
    sx = ('set', x, p, 'stance', tx)
    sy = ('set', y, p, 'stance', ty)
    irr = others[(cp * 2 + t) % len(others)]
    irr2 = others[(cp * 2 + t + 7) % len(others)]
    noise = ('set', irr[0], irr[1], 'status', r.choice(STATUS))
    noise2 = ('set', irr2[0], irr2[1], 'day', r.choice(DAYS))
    flip = {'favor': 'oppose', 'oppose': 'favor'}
    if fam == 'duplicates':
        return [[sx], [sy]] if t == 0 else [[sx, sy], [sx, sy]]
    if fam == 'restore':
        if init[(x, p)]['stance'] == tx and cp % 2 == 1:
            return [[('set', x, p, 'stance', flip[tx])], [('restore', x, p, 'stance', 'ORIGINAL'), sy]]
        return [[sx], [sy]]
    if fam == 'disjoint_order':
        return [[sx], [sy]] if t == 0 else [[sy], [sx]]
    if fam == 'repeat_overwrite':
        return [[('set', x, p, 'stance', flip[tx])], [sx, sy]] if t == 0 else [[sx], [('set', x, p, 'stance', tx), sy]]
    if fam == 'historical_return':
        return [[('set', x, p, 'stance', flip[tx]), sy], [sx]]
    if fam == 'old_length_variation':
        return [[noise], [sx, sy]] if t == 0 else [[sx, sy], [noise]]
    if fam == 'irrelevant_archive_growth':
        return [[noise, sx], [noise2, sy]] if t == 0 else [[sx], [sy]]
    if fam == 'atomic_multi_address':
        return [[sx, sy], [noise]] if t == 0 else [[noise2], [sx, sy]]
    raise ValueError(fam)


# ----------------------------- synthetic graph -----------------------------
GW = {  # two disjoint wordings per query kind and hop pattern
    ('team', 'hub'): ('Which hub is {s}\'s team based at?', 'Name the hub where the team of {s} is based.'),
    ('organization', 'founder'): ('Who founded the organization that {s} works for?',
                                  'Name the founder of {s}\'s employer.'),
    ('team', 'hub', 'region'): ('Which region is the hub of {s}\'s team in?',
                                '{s} is on a team based at some hub. Name the region of that hub.'),
    ('organization', 'founder', 'location'): ('Which city does the founder of {s}\'s organization live in?',
                                              'Name the home city of the person who founded the organization {s} works for.'),
    ('organization', 'founder', 'team', 'hub'): ('Which hub is the team of the founder of {s}\'s organization based at?',
                                                 'Name the hub where the founder of {s}\'s employer has their team based.'),
}
FIRST_EDGE = {'team': ('Which team is {s} on?', 'Name the team that {s} belongs to.'),
              'organization': ('Which organization does {s} work for?', 'Name the employer of {s}.')}
EDGE_TXT = {'team': '{s} is on the {o} team.', 'hub': 'The {s} team is based at the {o} hub.',
            'region': 'The {s} hub is in the {o} region.', 'organization': '{s} works for {o}.',
            'founder': '{s} was founded by {o}.', 'location': '{s} lives in {o}.'}
UPD_TXT = {'team': '{s} has moved to the {o} team.', 'hub': 'The {s} team is now based at the {o} hub.',
           'region': 'The {s} hub is now in the {o} region.', 'organization': '{s} now works for {o}.',
           'founder': '{s} is now recorded as founded by {o}.', 'location': '{s} now lives in {o}.'}
GPATHS = [('team', 'hub'), ('organization', 'founder'), ('team', 'hub', 'region'),
          ('organization', 'founder', 'location'), ('organization', 'founder', 'team', 'hub')]


def graph_panel(seed, n_roots, ns='NSR1-GRAPH'):
    out = []
    hop_cycle = [2, 3, 4]
    for i in range(n_roots):
        r = rng(seed, ns, i)
        hops = hop_cycle[i % 3]
        cands = [pth for pth in GPATHS if len(pth) == hops]
        path = cands[(i // 3) % len(cands)]
        ppl = people(r, 6)
        s, u = ppl[0], ppl[1]  # u: unrelated subject with its own chain
        pools = {'team': r.sample(TEAMS, 4), 'hub': r.sample(HUBS, 4), 'region': r.sample(REGIONS, 3),
                 'organization': r.sample(ORGS, 4), 'founder': ppl[2:5], 'location': r.sample(CITIES, 3)}

        def chain(start, k):
            nodes, edges = [start], []
            for n_, rel in enumerate(path):
                val = pools[rel][k % len(pools[rel])] if rel != 'founder' else pools['founder'][k % 3]
                edges.append((nodes[-1], rel, val))
                nodes.append(val)
            return nodes, edges
        n_s, e_s = chain(s, 0)
        n_u, e_u = chain(u, 1)
        # alternative values for first and intermediate edges
        alt_first = pools[path[0]][2] if path[0] != 'founder' else pools['founder'][2]
        alt_mid = (pools[path[1]][3] if path[1] != 'founder' else pools['founder'][2])
        current = e_s + e_u
        distract = []
        for rel in ('team', 'organization', 'location'):
            who = ppl[5]
            val = (pools[rel] if rel != 'founder' else pools['founder'])[-1]
            distract.append((who, rel, val))
        # need edges defined for alt values too (so alternative chains resolve)
        extra = []

        def onward(node, rels, k):
            prev = node
            for rel in rels:
                val = pools[rel][k % len(pools[rel])] if rel != 'founder' else pools['founder'][k % 3]
                extra.append((prev, rel, val))
                prev = val
        onward(alt_first, path[1:], 3)
        onward(alt_mid, path[2:], 2)
        histories = []
        # h0: original == current; h1: first edge originally alt then corrected; h2: intermediate edge
        # originally alt then corrected; h3: first edge changed to alt then restored to original.
        for h in range(4):
            init = list(current) + distract + extra
            events = []
            if h == 1:
                init = [(a_, b_, alt_first) if (a_, b_) == (s, path[0]) else (a_, b_, c_) for a_, b_, c_ in init]
                events = [[('set', s, path[0], n_s[1])]]
            elif h == 2:
                init = [(a_, b_, alt_mid) if (a_, b_) == (n_s[1], path[1]) else (a_, b_, c_) for a_, b_, c_ in init]
                events = [[('set', n_s[1], path[1], n_s[2])]]
            elif h == 3:
                events = [[('set', s, path[0], alt_first)], [('restore', s, path[0], 'ORIGINAL')]]
            # dedupe while keeping edge functional
            seen, uniq = set(), []
            for e in init:
                if (e[0], e[1]) not in seen:
                    seen.add((e[0], e[1]))
                    uniq.append(e)
            rr = rng(seed, ns, i, 'layout', h)
            rr.shuffle(uniq)
            recs = [dict(record_id=f'{n + 1:02d}', text=f'Record {n + 1:02d}: ' + EDGE_TXT[b_].format(s=a_, o=c_))
                    for n, (a_, b_, c_) in enumerate(uniq)]
            ups = []
            for n, ops in enumerate(events):
                kind, a_, b_, c_ = ops[0]
                if kind == 'set':
                    ups.append(dict(update_id=str(n + 1), text=f'Correction {n + 1}: ' + UPD_TXT[b_].format(s=a_, o=c_)))
                else:
                    ups.append(dict(update_id=str(n + 1), text=f'Correction {n + 1}: the {b_} of {a_} is restored to its original value.'))
            typed_init = [dict(entity=a_, field=b_, topic=None, value=c_) for a_, b_, c_ in uniq]
            typed_ev = []
            for ops in events:
                kind, a_, b_, c_ = ops[0]
                typed_ev.append(dict(writes=[dict(entity=a_, field=b_, topic=None, value=c_)]) if kind == 'set'
                                else dict(restores=[dict(entity=a_, field=b_, topic=None, snapshot='ORIGINAL')]))
            histories.append((dict(history_id=f'g{h}', records=recs, updates=ups),
                              dict(history_id=f'g{h}', initial=typed_init, events=typed_ev)))
        qs, qev = [], []
        kinds = [('edge', s, (path[0],), 'CURRENT'), ('dest', s, path, 'CURRENT'),
                 ('hist_dest', s, path, 'ORIGINAL'), ('unrelated', u, path, 'CURRENT')]
        for kind, subj, pth, snap in kinds:
            for w in range(2):
                if kind == 'edge':
                    text = FIRST_EDGE[pth[0]][w].format(s=subj)
                else:
                    text = GW[tuple(pth)][w].format(s=subj)
                    if kind == 'hist_dest':  # never alter the case of a name that starts the sentence
                        body = text if text.startswith(subj) else text[0].lower() + text[1:]
                        text = 'In the original records, before any correction: ' + body
                qid = f'{ns}-{i:04d}-{kind}-w{w}'
                qs.append(dict(question_id=qid, text=text))
                qev.append(dict(question_id=qid, kind=kind, wording=w, hops=len(pth), family='graph',
                                program=dict(op='FOLLOW', entity=subj, path=list(pth), snapshot=snap)))
        out.append((dict(root_id=f'{ns}-{i:04d}', hops=hops, histories=[v for v, _ in histories], questions=qs),
                    dict(root_id=f'{ns}-{i:04d}', hops=hops, path=list(path), typed=[t for _, t in histories],
                         questions=qev)))
    return out


# ----------------------------- workloads (cost panel) -----------------------------
WQ = {
    'status': 'What is the current status of {a}\'s {p} record?',
    'day': 'What is the current shipment day of {a}\'s {p} record?',
    'hist_status': 'In the original records, before any correction, what was the status of {a}\'s {p} record?',
    'same_day': 'Do {a} and {b} currently have the same shipment day on the {p} proposal?',
}


def workload_panel(seed, n, sizes, ns='NSR1-WORK'):
    out = []
    for i in range(n):
        N = sizes[i % len(sizes)]
        r = rng(seed, ns, i)
        ppl = people(r, max(8, N // 4))
        projs = r.sample(PROJECTS, 4)
        keys = [(u, v) for u in ppl for v in projs][:N]
        init = {k: random_member(r, *k) for k in keys}
        order = list(init)
        r.shuffle(order)
        rid = {k: f'{n_ + 1:03d}' for n_, k in enumerate(order)}
        recs = [dict(record_id=rid[k], text=record_text(rid[k], init[k])) for k in order]
        events, ups, cur = [], [], {k: dict(v) for k, v in init.items()}
        qs, qev = [], []
        for u_i in range(8):
            k = r.choice(keys)
            field = r.choice(['status', 'day'])
            val = r.choice([v for v in (STATUS if field == 'status' else DAYS) if v != cur[k][field]])
            ops = [('set', k[0], k[1], field, val)]
            cur[k][field] = val
            events.append(ops)
            ups.append(dict(update_id=str(u_i + 1), text=update_text(str(u_i + 1), ops)))
            for q_i in range(4):
                kind = ['status', 'day', 'hist_status', 'same_day'][q_i]
                a_ = k if q_i < 2 else r.choice(keys)
                if kind == 'same_day':
                    b_ = r.choice([kk for kk in keys if kk[1] == a_[1] and kk != a_] or [a_])
                    text = WQ[kind].format(a=a_[0], b=b_[0], p=a_[1]) + '\nAnswer with exactly one of No, Yes.'
                    prog = dict(op='SAME', left=rd(a_[0], a_[1], field='day'), right=rd(b_[0], b_[1], field='day'))
                    mapping = {'true': 'Yes', 'false': 'No'}
                else:
                    f = 'day' if kind == 'day' else 'status'
                    snap = 'ORIGINAL' if kind == 'hist_status' else 'CURRENT'
                    text = WQ[kind].format(a=a_[0], p=a_[1])
                    prog = rd(a_[0], a_[1], snap, field=f)
                    mapping = None
                qid = f'{ns}-{i:04d}-u{u_i + 1}-q{q_i}'
                qs.append(dict(question_id=qid, after_updates=u_i + 1, text=text))
                qev.append(dict(question_id=qid, after_updates=u_i + 1, kind=kind, program=prog, mapping=mapping,
                                family='workload'))
        vis = dict(root_id=f'{ns}-{i:04d}', N=N, histories=[dict(history_id='w', records=recs, updates=ups)], questions=qs)
        tv = dict(root_id=f'{ns}-{i:04d}', N=N, typed=[dict(history_id='w', initial=[w for k in order for w in member_writes(init[k])],
                                                            events=[typed_event(o) for o in events])], questions=qev)
        out.append((vis, tv))
    return out


# ----------------------------- negative contract cases (typed + language where expressible) ----------
def negative_cases():
    P = dict(person='Abel Drummond', project='Kestrel')
    base = [dict(entity='Abel Drummond', field='stance', topic='Kestrel', value='favor'),
            dict(entity='Abel Drummond', field='status', topic='Kestrel', value='approved'),
            dict(entity='Abel Drummond', field='day', topic='Kestrel', value='Monday'),
            dict(entity='Brisa Holloway', field='stance', topic='Kestrel', value='oppose'),
            dict(entity='Brisa Holloway', field='team', topic=None, value='Falcon'),
            dict(entity='Falcon', field='hub', topic=None, value='Delta'),
            dict(entity='Delta', field='region', topic=None, value='North')]
    R = lambda e, f='stance', t='Kestrel', s='CURRENT': dict(op='READ', entity=e, field=f, topic=t, snapshot=s)
    cases = []
    add = lambda cat, initial=base, events=(), program=None, text=None, expect='REJECT': cases.append(
        dict(case_id=f'NSR1-NEG-{len(cases):02d}', category=cat, initial=initial, events=list(events),
             program=program, question_text=text, expect=expect))
    # missing keys (6)
    add('missing_key', program=dict(op='TEST', arg=R('Calder Galloway'), value='favor'),
        text='Is Calder Galloway currently in favor of the Kestrel proposal?\nAnswer with exactly one of No, Yes.')
    add('missing_key', program=dict(op='TEST', arg=R('Abel Drummond', t='Nimbus'), value='favor'),
        text='Is Abel Drummond currently in favor of the Nimbus proposal?\nAnswer with exactly one of No, Yes.')
    add('missing_key', program=dict(op='FOLLOW', entity='Abel Drummond', path=['team', 'hub'], snapshot='CURRENT'),
        text='Which hub is Abel Drummond\'s team based at?')
    add('missing_key', initial=base + [dict(entity='Corvin Whitlock', field='team', topic=None, value='Heron')],
        program=dict(op='FOLLOW', entity='Corvin Whitlock', path=['team', 'hub'], snapshot='CURRENT'),
        text="Which hub is Corvin Whitlock's team based at?")
    add('missing_key', program=R('Brisa Holloway', f='day'),
        text='What is the current shipment day of Brisa Holloway\'s Kestrel record?')
    add('missing_key', events=[dict(restores=[dict(entity='Brisa Holloway', field='day', topic='Kestrel', snapshot='ORIGINAL')])])
    # duplicate names / ambiguous (6)
    dup = base + [dict(entity='Abel Drummond', field='stance', topic='Kestrel', value='oppose')]
    add('duplicate_name', initial=dup, program=dict(op='TEST', arg=R('Abel Drummond'), value='favor'))
    add('duplicate_name', initial=base + [dict(base[0])], program=R('Abel Drummond'))
    add('duplicate_name', initial=base + [dict(entity='Falcon', field='team', topic=None, value='Heron')],
        expect='REJECT_STATE')  # Falcon is a team and a person
    add('duplicate_name', initial=base + [dict(entity='Delta', field='organization', topic=None, value='Lumen Labs')],
        expect='REJECT_STATE')
    add('duplicate_name', initial=dup, program=dict(op='SAME', left=R('Abel Drummond'), right=R('Brisa Holloway')))
    add('duplicate_name', initial=base + [dict(entity='Brisa Holloway', field='team', topic=None, value='Heron')],
        program=dict(op='FOLLOW', entity='Brisa Holloway', path=['team', 'hub'], snapshot='CURRENT'))
    # unsupported temporal ambiguity (5)
    for s in ('EVENT:0', 'EVENT:99', 'YESTERDAY', 'LAST_WEEK', 'EVENT:-1'):
        add('unsupported_temporal', program=dict(op='TEST', arg=R('Abel Drummond', s=s), value='favor'))
    # invalid types (5)
    add('invalid_type', program=dict(op='TEST', arg=R('Abel Drummond'), value='maybe'))
    add('invalid_type', program=dict(op='TEST', arg=R('Abel Drummond'), value='approved'))
    add('invalid_type', program=dict(op='SAME', left=R('Abel Drummond'), right=R('Abel Drummond', f='day')))
    add('invalid_type', events=[dict(writes=[dict(entity='Abel Drummond', field='stance', topic='Kestrel', value='neutral')])])
    add('invalid_type', program=dict(op='FOLLOW', entity='Brisa Holloway', path=['team', 'region'], snapshot='CURRENT'))
    # conflicting multi-writes (5)
    w = lambda v, f='stance': dict(entity='Abel Drummond', field=f, topic='Kestrel', value=v)
    add('conflicting_writes', events=[dict(writes=[w('favor'), w('oppose')])])
    add('conflicting_writes', events=[dict(writes=[w('approved', 'status'), w('rejected', 'status')])])
    add('conflicting_writes', events=[dict(writes=[w('oppose')]), dict(writes=[w('favor')], restores=[dict(entity='Abel Drummond', field='stance', topic='Kestrel', snapshot='EVENT:1')])])
    add('conflicting_writes', events=[dict(writes=[w('oppose'), dict(entity='Brisa Holloway', field='stance', topic='Kestrel', value='banana')])])
    add('conflicting_writes', events=[dict(deletes=[dict(entity='Abel Drummond', field='stance', topic='Kestrel')])])
    # nonfunctional / unsupported paths (5)
    add('nonfunctional_path', program=dict(op='FOLLOW', entity='Brisa Holloway', path=['team', 'hub', 'region', 'team', 'hub'], snapshot='CURRENT'))
    add('nonfunctional_path', program=dict(op='FOLLOW', entity='Brisa Holloway', path=['manager'], snapshot='CURRENT'))
    add('nonfunctional_path', program=dict(op='FOLLOW', entity='Kestrel', path=['founder'], snapshot='CURRENT'))
    add('nonfunctional_path', program=dict(op='TEST', arg=dict(op='READ', entity='Brisa Holloway', field='team', topic='Kestrel', snapshot='CURRENT'), value='Falcon'))
    add('nonfunctional_path', program=dict(op='SAME', left=R('Abel Drummond'), right=R('Abel Drummond')))  # forged constant
    assert len(cases) == 32 and all(c['expect'] != 'OK' for c in cases), len(cases)
    return cases


def positive_controls():
    """Supported counterparts of the negative cases; must succeed (always-reject must fail)."""
    neg = negative_cases()
    base = neg[0]['initial']
    R = lambda e, f='stance', t='Kestrel', s='CURRENT': dict(op='READ', entity=e, field=f, topic=t, snapshot=s)
    w = lambda v, f='stance': dict(entity='Abel Drummond', field=f, topic='Kestrel', value=v)
    dup = base + [dict(entity='Abel Drummond', field='stance', topic='Kestrel', value='oppose')]
    return [
        dict(case_id='NSR1-POS-00', initial=base, events=[], program=dict(op='TEST', arg=R('Abel Drummond'), value='favor'), expect_value=True),
        dict(case_id='NSR1-POS-01', initial=base, events=[], program=dict(op='FOLLOW', entity='Brisa Holloway', path=['team', 'hub', 'region'], snapshot='ORIGINAL'), expect_value='North'),
        dict(case_id='NSR1-POS-02', initial=dup, events=[dict(writes=[w('oppose')])], program=dict(op='TEST', arg=R('Abel Drummond'), value='oppose'), expect_value=True),
        dict(case_id='NSR1-POS-03', initial=base, events=[dict(writes=[w('oppose')]), dict(restores=[dict(entity='Abel Drummond', field='stance', topic='Kestrel', snapshot='ORIGINAL')])],
             program=dict(op='TEST', arg=R('Abel Drummond'), value='favor'), expect_value=True),
        dict(case_id='NSR1-POS-04', initial=base, events=[dict(writes=[w('oppose')])], program=dict(op='TEST', arg=R('Abel Drummond', s='EVENT:1'), value='favor'), expect_value=False),
        dict(case_id='NSR1-POS-05', initial=base, events=[], program=dict(op='NOT', arg=dict(op='SAME', left=R('Abel Drummond'), right=R('Brisa Holloway'))), expect_value=True),
    ]


def write_panel(root_dir, name, pairs):
    vis_dir = Path(root_dir) / 'visible' / name
    ev_dir = Path(root_dir) / 'evaluator' / name
    vis_dir.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    with open(vis_dir / 'roots.jsonl', 'w', encoding='utf-8', newline='\n') as fv, \
            open(ev_dir / 'roots.jsonl', 'w', encoding='utf-8', newline='\n') as fe:
        for v, e in pairs:
            fv.write(json.dumps(v, sort_keys=True) + '\n')
            fe.write(json.dumps(e, sort_keys=True) + '\n')


def build_all(config_path, out_dir):
    cfg = json.loads(Path(config_path).read_text())
    S, N = cfg['seeds'], cfg['samples']
    # 16 development roots in total: 5 core-shaped, 2 factorial, 3 temporal, 3 graph (2/3/4 hops), 1 workload,
    # plus 2 public cases (hash ranks 17-18; built in nsr/panels.py).
    d = S['development']
    write_panel(out_dir, 'dev_core', core_panel(d, 'NSR1-DEVC', 5))
    write_panel(out_dir, 'dev_factorial', factorial_panel(d, 2, (8, 32), ns='NSR1-DEVF'))
    write_panel(out_dir, 'dev_temporal', temporal_panel(d, 3, ns='NSR1-DEVT'))
    write_panel(out_dir, 'dev_graph', graph_panel(d, 3, ns='NSR1-DEVG'))
    write_panel(out_dir, 'dev_workload', workload_panel(d, 1, (32,), ns='NSR1-DEVW'))
    write_panel(out_dir, 'core', core_panel(S['core'], 'NSR1-CORE', N['core_roots']))
    write_panel(out_dir, 'boundary', core_panel(S['boundary'], 'NSR1-BOUND', N['boundary_roots']))
    write_panel(out_dir, 'factorial', factorial_panel(S['factorial'], N['factorial_roots'], tuple(N['factorial_records'])))
    write_panel(out_dir, 'temporal', temporal_panel(S['temporal'], N['temporal_programs']))
    write_panel(out_dir, 'graph', graph_panel(S['graph'], N['synthetic_graph_roots']))
    write_panel(out_dir, 'workload', workload_panel(S['timing'], N['timing_workloads'], tuple(N['timing_records'])))
    neg = negative_cases()
    p = Path(out_dir) / 'evaluator' / 'negative'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'cases.json').write_text(json.dumps(neg, indent=1), encoding='utf-8')
    (p / 'positive_controls.json').write_text(json.dumps(positive_controls(), indent=1), encoding='utf-8')


if __name__ == '__main__':
    import sys
    build_all(sys.argv[1], sys.argv[2])
