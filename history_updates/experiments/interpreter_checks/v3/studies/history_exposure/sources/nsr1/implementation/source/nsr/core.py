"""NSR1 candidate exact core: fixed schema, versioned typed store, whitelisted read-only DSL.

Contract: docs/FORMAL_SPEC.md. No evaluator code is imported here; no generated code is executed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from types import MappingProxyType

EXECUTOR_REVISION = 'NSR1-CORE-2'
ENUMS = {
    'stance': ('favor', 'oppose'),
    'status': ('pending', 'approved', 'rejected', 'deferred'),
    'day': ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'),
}
MEMBER_FIELDS = tuple(ENUMS)
RELATIONS = {  # relation -> (domain type, range type)
    'team': ('person', 'team'), 'hub': ('team', 'hub'), 'region': ('hub', 'region'),
    'organization': ('person', 'organization'), 'founder': ('organization', 'person'),
    'location': ('person', 'city'),
}
MAX_HOPS = 4
AMBIG = object()


class NSRError(Exception):
    def __init__(self, code, detail=''):
        super().__init__(f'{code}: {detail}' if detail else code)
        self.code, self.detail = code, detail


def register_public_relations(names):
    """Public-bridge relations: generic entity -> entity, declared from the source-level vocabulary."""
    for n in names:
        if n in RELATIONS and RELATIONS[n] != ('entity', 'entity'):
            raise NSRError('MALFORMED', f'public relation collides with fixed schema: {n}')
        RELATIONS[n] = ('entity', 'entity')


def _name(x):
    if not isinstance(x, str) or not x or len(x) > 80 or x != x.strip():
        raise NSRError('MALFORMED', f'bad name {x!r}')
    return x


def check_write(w):
    """Type-check one write; returns (address, value, entity-type assignments)."""
    if not isinstance(w, dict) or set(w) != {'entity', 'field', 'topic', 'value'}:
        raise NSRError('MALFORMED', f'write keys {sorted(w) if isinstance(w, dict) else type(w).__name__}')
    e, f, t, v = _name(w['entity']), w['field'], w['topic'], w['value']
    if f in ENUMS:
        _name(t)
        if v not in ENUMS[f]:
            raise NSRError('TYPE_ERROR', f'{f} value {v!r}')
        return (e, f, t), v, ((e, 'person'), (t, 'project'))
    if f in RELATIONS:
        if t is not None:
            raise NSRError('TYPE_ERROR', f'relation {f} takes no topic')
        dom, rng = RELATIONS[f]
        return (e, f, None), _name(v), ((e, dom), (v, rng))
    raise NSRError('TYPE_ERROR', f'undeclared field {f!r}')


def _assign_types(types, pairs):
    out = dict(types)
    for name, typ in pairs:
        if typ == 'entity':
            continue
        prior = out.get(name)
        if prior is not None and prior != typ:
            raise NSRError('TYPE_CONFLICT', f'{name!r} is {prior} and {typ}')
        out[name] = typ
    return out


def content_hash(mapping):
    rows = sorted([list(k), '<AMBIGUOUS>' if v is AMBIG else v] for k, v in mapping.items())
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), default=str).encode()).hexdigest()


class Snapshot:
    """Immutable published snapshot: values, per-address versions, entity types, identities."""
    __slots__ = ('index', 'values', 'versions', 'types', 'content', 'history')

    def __init__(self, index, values, versions, types, history):
        self.index = index
        self.values = MappingProxyType(dict(values))
        self.versions = MappingProxyType(dict(versions))
        self.types = MappingProxyType(dict(types))
        self.content = content_hash(values)
        self.history = history  # hash chain of accepted events (differs from content identity)


class Store:
    """Versioned authoritative state with atomic transactions, archives and a derived-view cache."""

    def __init__(self, initial_records):
        values, versions, types, seen = {}, {}, {}, set()
        for w in initial_records:
            addr, v, pairs = check_write(w)
            types = _assign_types(types, pairs)
            if addr in seen:
                values[addr] = AMBIG
            else:
                values[addr] = v
                seen.add(addr)
            versions[addr] = (0, 0)
        h0 = hashlib.sha256(b'NSR1-ORIGINAL' + content_hash(values).encode()).hexdigest()
        self.snapshots = [Snapshot(0, values, versions, types, h0)]
        self.event_log = []      # accepted and rejected events, in order
        self.views = {}          # view_id -> program
        self.cache = {}          # view_id -> entry
        self.stats = dict(view_hits=0, view_rebuilds=0, view_invalidations=0, stale_detected=0)
        self._dependents = {}    # address -> set(view_id)

    # ----- snapshots -----
    @property
    def current(self):
        return self.snapshots[-1]

    def resolve(self, label):
        if label == 'CURRENT':
            return self.current
        if label == 'ORIGINAL':
            return self.snapshots[0]
        if isinstance(label, str) and label.startswith('EVENT:') and label[6:].isdigit() and label[6] != '0':
            k = int(label[6:])
            if 1 <= k < len(self.snapshots):
                return self.snapshots[k]
        raise NSRError('UNSUPPORTED_SNAPSHOT', repr(label))

    # ----- transactions -----
    def apply(self, event):
        """Validate every operation, then publish one coherent snapshot; on error publish nothing."""
        cur = self.current
        try:
            if not isinstance(event, dict) or not set(event) <= {'writes', 'restores'} or not event:
                raise NSRError('UNSUPPORTED_OPERATION', f'event keys {sorted(event) if isinstance(event, dict) else event!r}')
            ops, types = {}, dict(cur.types)
            for w in event.get('writes', []):
                addr, v, pairs = check_write(w)
                types = _assign_types(types, pairs)
                if addr in ops and ops[addr] != v:
                    raise NSRError('CONFLICTING_WRITES', repr(addr))
                ops[addr] = v
            for r in event.get('restores', []):
                if not isinstance(r, dict) or set(r) != {'entity', 'field', 'topic', 'snapshot'}:
                    raise NSRError('MALFORMED', 'restore keys')
                src = self.resolve(r['snapshot'])
                addr = (r['entity'], r['field'], r['topic'])
                if addr not in src.values:
                    raise NSRError('MISSING', f'restore source {addr}')
                if src.values[addr] is AMBIG:
                    raise NSRError('AMBIGUOUS', f'restore source {addr}')
                if addr in ops and ops[addr] != src.values[addr]:
                    raise NSRError('CONFLICTING_WRITES', repr(addr))
                ops[addr] = src.values[addr]
            if not ops:
                raise NSRError('MALFORMED', 'empty event')
        except NSRError as exc:
            self.event_log.append(dict(event=event, status='REJECTED', error=exc.code))
            return dict(status='REJECTED', error=exc.code, detail=exc.detail)
        k = len(self.snapshots)
        values, versions = dict(cur.values), dict(cur.versions)
        changed = []
        for addr, v in ops.items():
            if addr not in values or values[addr] is AMBIG or values[addr] != v:
                changed.append(addr)
            values[addr] = v
            versions[addr] = (k, versions.get(addr, (0, 0))[1] + 1)
        hist = hashlib.sha256((cur.history + json.dumps(sorted([list(a), v] for a, v in ops.items()),
                                                         default=str)).encode()).hexdigest()
        self.snapshots.append(Snapshot(k, values, versions, types, hist))
        self.event_log.append(dict(event=event, status='ACCEPTED', index=k))
        for addr in ops:  # selective invalidation: every view whose dependency address was written
            for vid in self._dependents.pop(addr, set()):
                if vid in self.cache:
                    self.stats['view_invalidations'] += 1
                    del self.cache[vid]
        return dict(status='ACCEPTED', index=k, changed=len(changed))

    # ----- derived views -----
    def register_view(self, program, question_id=''):
        validate_program(program)
        vid = hashlib.sha256((EXECUTOR_REVISION + canonical(program)).encode()).hexdigest()[:24]
        self.views[vid] = program
        return vid

    def _entry_digest(self, vid, deps, result):
        return hashlib.sha256(canonical([EXECUTOR_REVISION, vid, sorted(deps), result]).encode()).hexdigest()

    def read_view(self, vid):
        """Serve a CURRENT-snapshot view; reuse only after every declared dependency validates."""
        prog, cur = self.views[vid], self.current
        entry = self.cache.get(vid)
        if entry is not None:
            ok = all(_dep_now(cur, a) == ver for a, ver in entry['deps'])
            ok = ok and entry['digest'] == self._entry_digest(vid, entry['deps'], entry['result'])
            if ok:
                self.stats['view_hits'] += 1
                # Every dependency address has the identical version in the current snapshot, so the served
                # provenance is restated against the snapshot it was validated on (not the one it was built on).
                res = dict(entry['result'], read_set=[[cur.index] + list(r[1:]) for r in entry['result']['read_set']])
                return dict(res, served='cache', computed_at=entry['computed_at'], validated_at=cur.index)
            self.stats['stale_detected'] += 1
            del self.cache[vid]
        result = execute(prog, self, default_snapshot='CURRENT')
        if any(r[0] != cur.index for r in result['read_set']):
            return dict(result, served='uncached_historical')  # historical reads are not cached views
        deps = [(tuple(r[1:]), _dep_now(cur, tuple(r[1:]))) for r in result['read_set']]
        self.cache[vid] = dict(deps=deps, result=result, digest=self._entry_digest(vid, deps, result),
                               computed_at=cur.index)
        for a, _ in deps:
            self._dependents.setdefault(a, set()).add(vid)
        self.stats['view_rebuilds'] += 1
        return dict(result, served='rebuilt', computed_at=cur.index, validated_at=cur.index)


def _dep_now(snapshot, addr):
    """Dependency token: per-address version; missing keys are dependencies too."""
    return list(snapshot.versions[addr]) if addr in snapshot.versions else ['MISSING']


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


# ----------------------------- DSL -----------------------------
BOOL_OPS = {'TEST', 'SAME', 'BOTH', 'EITHER', 'NOT'}
KEYS = {
    'READ': {'op', 'entity', 'field', 'topic', 'snapshot'},
    'FOLLOW': {'op', 'entity', 'path', 'snapshot'},
    'TEST': {'op', 'arg', 'value'}, 'SAME': {'op', 'left', 'right'},
    'BOTH': {'op', 'left', 'right'}, 'EITHER': {'op', 'left', 'right'}, 'NOT': {'op', 'arg'},
}


def _snap_ok(s):
    return s in ('CURRENT', 'ORIGINAL') or (isinstance(s, str) and s.startswith('EVENT:') and s[6:].isdigit()
                                            and s[6] != '0')


def static_type(node, depth=0):
    """Return the static type ('bool', an enum field name, or 'entity:<type>'); raise on violations."""
    if depth > 12:
        raise NSRError('MALFORMED', 'nesting too deep')
    if not isinstance(node, dict) or node.get('op') not in KEYS or set(node) != KEYS[node['op']]:
        raise NSRError('MALFORMED', f'node {node!r}'[:160])
    op = node['op']
    if op in ('READ', 'FOLLOW'):
        _name(node['entity'])
        if not _snap_ok(node['snapshot']):
            raise NSRError('UNSUPPORTED_SNAPSHOT', repr(node['snapshot']))
    if op == 'READ':
        f, t = node['field'], node['topic']
        if f in ENUMS:
            _name(t)
            return f
        if f in RELATIONS:
            if t is not None:
                raise NSRError('TYPE_ERROR', 'relation READ takes topic null')
            return 'entity:' + RELATIONS[f][1]
        raise NSRError('TYPE_ERROR', f'undeclared field {f!r}')
    if op == 'FOLLOW':
        path = node['path']
        if not isinstance(path, list) or not path:
            raise NSRError('MALFORMED', 'path')
        if len(path) > MAX_HOPS:
            raise NSRError('HOP_LIMIT', str(len(path)))
        prev = None
        for r in path:
            if r not in RELATIONS:
                raise NSRError('TYPE_ERROR', f'unknown relation {r!r}')
            dom, rng = RELATIONS[r]
            if prev is not None and 'entity' not in (prev, dom) and prev != dom:
                raise NSRError('TYPE_ERROR', f'path does not chain at {r}')
            prev = rng
        return 'entity:' + prev
    if op == 'TEST':
        t = static_type(node['arg'], depth + 1)
        if node['arg']['op'] not in ('READ', 'FOLLOW'):
            raise NSRError('TYPE_ERROR', 'TEST needs READ/FOLLOW')
        if t in ENUMS:
            if node['value'] not in ENUMS[t]:
                raise NSRError('TYPE_ERROR', f'literal {node["value"]!r} not in {t}')
        else:
            _name(node['value'])
        return 'bool'
    if op == 'SAME':
        a, b = static_type(node['left'], depth + 1), static_type(node['right'], depth + 1)
        if node['left']['op'] not in ('READ', 'FOLLOW') or node['right']['op'] not in ('READ', 'FOLLOW'):
            raise NSRError('TYPE_ERROR', 'SAME needs READ/FOLLOW operands')
        if a.startswith('entity:') and b.startswith('entity:'):
            return 'bool'
        if a != b:
            raise NSRError('TYPE_ERROR', f'SAME {a} vs {b}')
        return 'bool'
    if op == 'NOT':
        if static_type(node['arg'], depth + 1) != 'bool':
            raise NSRError('TYPE_ERROR', 'NOT needs bool')
        return 'bool'
    for side in ('left', 'right'):  # BOTH / EITHER
        if static_type(node[side], depth + 1) != 'bool':
            raise NSRError('TYPE_ERROR', f'{op} needs bool operands')
    return 'bool'


def _leaves(node, out):
    op = node['op']
    if op in ('READ', 'FOLLOW'):
        out.append(node)
    for k in ('arg', 'left', 'right'):
        if k in node and isinstance(node[k], dict):
            _leaves(node[k], out)
    return out


def constant_program(node):
    """True when the program's answer cannot depend on any source value (forged constant answer)."""
    if static_type(node) != 'bool':
        return False
    leaves, keys = _leaves(node, []), []
    for leaf in leaves:
        c = canonical(leaf)
        if c not in keys:
            keys.append(c)
    domains = []
    for c in keys:
        leaf = json.loads(c)
        t = static_type(leaf)
        domains.append(list(ENUMS[t]) if t in ENUMS else ['<e1>', '<e2>', '<e3>'])
    if sum(len(d) for d in domains) > 64 or len(keys) > 6:
        return False
    outs = set()
    for combo in itertools.product(*domains):
        env = dict(zip(keys, combo))
        outs.add(_eval_abstract(node, env))
        if len(outs) > 1:
            return False
    return True


def _eval_abstract(node, env):
    op = node['op']
    if op in ('READ', 'FOLLOW'):
        return env[canonical(node)]
    if op == 'TEST':
        v = _eval_abstract(node['arg'], env)
        return v == node['value'] if not str(v).startswith('<e') else (v == '<e1>')
    if op == 'SAME':
        return _eval_abstract(node['left'], env) == _eval_abstract(node['right'], env)
    if op == 'NOT':
        return not _eval_abstract(node['arg'], env)
    a, b = _eval_abstract(node['left'], env), _eval_abstract(node['right'], env)
    return (a and b) if op == 'BOTH' else (a or b)


def validate_program(node):
    """Whitelist + static types + forbid constant-answer programs. Returns the answer type."""
    t = static_type(node)
    if constant_program(node):
        raise NSRError('CONSTANT_PROGRAM', 'answer independent of every source read')
    return t


def execute(program, store, default_snapshot='CURRENT'):
    """Exact strict evaluation over immutable snapshots. Historical reads use a disposable workspace
    (a read-only view of the archived snapshot); the current snapshot identity is re-checked after."""
    before = (store.current.index, store.current.content, store.current.history)
    reads = []
    try:
        t = validate_program(program)
        value = _eval(program, store, reads)
        res = dict(status='OK', value=value, type='bool' if t == 'bool' else
                   (t if t in ENUMS else 'entity'))
    except NSRError as exc:
        res = dict(status='ERROR', error=exc.code, detail=exc.detail[:200])
    after = (store.current.index, store.current.content, store.current.history)
    if before != after:
        raise RuntimeError('read path mutated the current snapshot')
    res['read_set'] = sorted({tuple(r) for r in reads}, key=lambda r: (r[0], r[1], r[2], str(r[3])))
    res['read_set'] = [list(r) for r in res['read_set']]
    return res


def _lookup(store, snap_label, addr, reads):
    snap = store.resolve(snap_label)
    workspace = snap.values  # MappingProxy: disposable read-only view, never written back
    reads.append((snap.index,) + addr)
    if addr not in workspace:
        raise NSRError('MISSING', repr(addr))
    v = workspace[addr]
    if v is AMBIG:
        raise NSRError('AMBIGUOUS', repr(addr))
    return v


def _eval(node, store, reads):
    op = node['op']
    if op == 'READ':
        return _lookup(store, node['snapshot'], (node['entity'], node['field'], node['topic']), reads)
    if op == 'FOLLOW':
        cur = node['entity']
        for r in node['path']:
            cur = _lookup(store, node['snapshot'], (cur, r, None), reads)
        return cur
    if op == 'TEST':
        return _eval(node['arg'], store, reads) == node['value']
    if op == 'SAME':
        a = _eval(node['left'], store, reads)
        b = _eval(node['right'], store, reads)
        return a == b
    if op == 'NOT':
        return not _eval(node['arg'], store, reads)
    a = _eval(node['left'], store, reads)
    b = _eval(node['right'], store, reads)
    return (a and b) if op == 'BOTH' else (a or b)


def replay(initial_records, events):
    """Fresh independent replay from the initial state (used for differential checks)."""
    s = Store(initial_records)
    for e in events:
        s.apply(e)
    return s
