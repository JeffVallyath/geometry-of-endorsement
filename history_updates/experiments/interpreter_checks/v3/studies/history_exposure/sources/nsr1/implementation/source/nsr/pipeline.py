"""NSR1 candidate-side pipeline: payload rule, validators, one syntax/type repair, the four main arms and
the diagnostics. Reads only visible inputs; never imports evaluator code. Runs with any `Gen` backend.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from . import core, prompts, render
from .core import NSRError

CAPS = dict(query=192, record=512, update=512, flat=384, answer=32)
FENCE = re.compile(r'```(?:json)?[ \t]*\n(.*?)\n?```', re.S)


# ----------------------------- payload + validators -----------------------------
def payload(text):
    """The one frozen complete-payload rule: the whole output (after whitespace strip) must be one JSON
    object, optionally wrapped in exactly one code fence. No substring search, no trailing prose."""
    t = text.strip()
    m = FENCE.fullmatch(t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
    except ValueError as exc:
        raise NSRError('JSON_INVALID', str(exc)[:120])
    if not isinstance(obj, dict):
        raise NSRError('JSON_NOT_OBJECT', type(obj).__name__)
    return obj


# Fixed source-level value-alias dictionary, shared by every program/fact-generating arm (never answer-derived):
# the surface forms that the records and corrections themselves use for the two stance values.
VALUE_ALIASES = {'stance': {'in favor': 'favor', 'favors': 'favor', 'supports': 'favor', 'opposed': 'oppose',
                            'opposes': 'oppose', 'against': 'oppose'}}


def norm_value(field, value):
    return VALUE_ALIASES.get(field, {}).get(value, value) if isinstance(value, str) else value


def norm_program(node):
    """Apply the value-alias dictionary to TEST literals on enum reads (in place); structure is untouched."""
    if isinstance(node, dict):
        if node.get('op') == 'TEST' and isinstance(node.get('arg'), dict) and node['arg'].get('op') == 'READ':
            node['value'] = norm_value(node['arg'].get('field'), node.get('value'))
        for k in ('arg', 'left', 'right'):
            norm_program(node.get(k))
    return node


def _names(node, out):
    if node['op'] in ('READ', 'FOLLOW'):
        out.append(node['entity'])
        if node.get('topic') is not None and node['op'] == 'READ':
            out.append(node['topic'])
    if node['op'] == 'TEST' and core.static_type(node['arg']) not in core.ENUMS:
        out.append(node['value'])
    for k in ('arg', 'left', 'right'):
        if isinstance(node.get(k), dict):
            _names(node[k], out)
    return out


def validate_query(obj, question):
    """Static, question-only validation (no state, no gold). Returns ('PROGRAM', ast) or ('UNSUPPORTED', reason)."""
    if obj.get('op') == 'UNSUPPORTED':
        return 'UNSUPPORTED', str(obj.get('reason', ''))[:200]
    norm_program(obj)
    core.validate_program(obj)
    if prompts.PUBLIC_WORDING:  # declared public schema active: only its relations are type-valid
        for leaf in core._leaves(obj, []):
            for rel in ([leaf['field']] if leaf['op'] == 'READ' else leaf['path']):
                if rel not in prompts.PUBLIC_WORDING and rel not in core.ENUMS:
                    raise NSRError('TYPE_ERROR', f'relation {rel!r} is not in the declared schema')
    for n in _names(obj, []):
        if n not in question:
            raise NSRError('NAME_NOT_IN_QUESTION', repr(n))
    return 'PROGRAM', obj


def validate_record(obj, line):
    kind = obj.get('kind')
    if kind == 'none' and set(obj) == {'kind'}:
        return []
    if kind == 'member':
        if not {'kind', 'person', 'project'} <= set(obj) or not set(obj) <= {'kind', 'person', 'project', 'stance', 'status', 'day'}:
            raise NSRError('MALFORMED', f'member keys {sorted(obj)}')
        writes = [dict(entity=obj['person'], field=f, topic=obj['project'], value=norm_value(f, obj[f]))
                  for f in ('stance', 'status', 'day') if f in obj]
        if not writes:
            raise NSRError('MALFORMED', 'member record without a field')
    elif kind == 'relation':
        if set(obj) != {'kind', 'entity', 'relation', 'value'}:
            raise NSRError('MALFORMED', f'relation keys {sorted(obj)}')
        writes = [dict(entity=obj['entity'], field=obj['relation'], topic=None, value=obj['value'])]
    else:
        raise NSRError('MALFORMED', f'kind {kind!r}')
    for w in writes:
        core.check_write(w)
        for n in (w['entity'], w['topic']) + ((w['value'],) if w['field'] in core.RELATIONS else ()):
            if n is not None and n not in line:
                raise NSRError('NAME_NOT_IN_SOURCE', repr(n))
    return writes


def validate_update(obj, statement):
    if not obj or not set(obj) <= {'writes', 'restores'}:
        raise NSRError('MALFORMED', f'transaction keys {sorted(obj)}')
    ev = {}
    for w in obj.get('writes', []):
        if isinstance(w, dict) and 'field' in w and 'value' in w:
            w['value'] = norm_value(w['field'], w['value'])
        core.check_write(w)
        for n in (w['entity'], w['topic']) + ((w['value'],) if w['field'] in core.RELATIONS else ()):
            if n is not None and n not in statement:
                raise NSRError('NAME_NOT_IN_SOURCE', repr(n))
        ev.setdefault('writes', []).append(w)
    for r in obj.get('restores', []):
        if not isinstance(r, dict) or set(r) != {'entity', 'field', 'topic', 'snapshot'}:
            raise NSRError('MALFORMED', 'restore keys')
        if r['field'] not in core.ENUMS and r['field'] not in core.RELATIONS:
            raise NSRError('TYPE_ERROR', f'undeclared field {r["field"]!r}')
        if not core._snap_ok(r['snapshot']) or r['snapshot'] == 'CURRENT':
            raise NSRError('UNSUPPORTED_SNAPSHOT', repr(r['snapshot']))
        for n in (r['entity'], r['topic']):
            if n is not None and n not in statement:
                raise NSRError('NAME_NOT_IN_SOURCE', repr(n))
        ev.setdefault('restores', []).append(r)
    if not ev:
        raise NSRError('MALFORMED', 'empty transaction')
    return ev


def validate_flat(obj, question, source_ids):
    if set(obj) != {'facts', 'program'} or not isinstance(obj['facts'], list):
        raise NSRError('MALFORMED', f'flat keys {sorted(obj)}')
    kind, prog = validate_query(obj['program'], question)
    facts = []
    for f in obj['facts']:
        if not isinstance(f, dict) or set(f) != {'source', 'entity', 'field', 'topic', 'value', 'snapshot'}:
            raise NSRError('MALFORMED', 'fact keys')
        if f['source'] not in source_ids:
            raise NSRError('UNKNOWN_SOURCE', repr(f['source']))
        if not core._snap_ok(f['snapshot']):
            raise NSRError('UNSUPPORTED_SNAPSHOT', repr(f['snapshot']))
        f['value'] = norm_value(f['field'], f['value'])
        core.check_write({k: f[k] for k in ('entity', 'field', 'topic', 'value')})
        facts.append(f)
    # Structural well-formedness of the self-contained problem (no state, no gold): the fact base must be a function,
    # and every READ of the program must be covered by a listed fact. Both are validator errors (one repair).
    seen = {}
    for f in facts:
        k = (f['snapshot'], f['entity'], f['field'], f['topic'])
        if k in seen and seen[k] != f['value']:
            raise NSRError('CONFLICTING_FACTS', f'{f["entity"]} / {f["field"]} / {f["topic"]} has two different values in snapshot {f["snapshot"]}')
        seen[k] = f['value']
    if kind == 'PROGRAM':
        for leaf in core._leaves(prog, []):
            first = leaf['field'] if leaf['op'] == 'READ' else leaf['path'][0]
            k = (leaf['snapshot'], leaf['entity'], first, leaf.get('topic') if leaf['op'] == 'READ' else None)
            if k not in seen:
                raise NSRError('MISSING_FACT', f'the program reads {leaf["entity"]} / {first} in snapshot {leaf["snapshot"]} but no fact lists it')
    return kind, prog, facts


# ----------------------------- generation with one syntax/type repair -----------------------------
class Gen:
    """Backend interface. `run(reqs)` -> list of dict(text, token_ids, finish, n_prompt, n_out, seconds_batch)."""
    model_id = 'abstract'

    def run(self, reqs):
        raise NotImplementedError


class CachedGen:
    """Exact input-keyed generation cache (a real runtime feature; every hit is logged)."""

    def __init__(self, backend, log=None, enabled=True):
        self.backend, self.cache, self.enabled = backend, {}, enabled
        self.log = log
        self.stats = dict(calls=0, hits=0, generated=0, prompt_tokens=0, output_tokens=0, seconds=0.0)

    @staticmethod
    def key(req):
        return hashlib.sha256(json.dumps([req['messages'], req['max_new']], separators=(',', ':')).encode()).hexdigest()

    def run(self, reqs):
        keys = [self.key(r) for r in reqs]
        if self.enabled:
            todo, seen = [], set()
            for k, r in zip(keys, reqs):
                if k not in self.cache and k not in seen:
                    todo.append((k, r))
                    seen.add(k)
        else:  # cache-skipping cold mode: every request is generated, even exact duplicates
            todo = list(zip(keys, reqs))
        t0 = time.perf_counter()
        outs = self.backend.run([r for _, r in todo]) if todo else []
        fresh = []
        for (k, r), o in zip(todo, outs):
            o = dict(o, key=k, tag=r.get('tag', ''))
            fresh.append(o)
            self.stats['generated'] += 1
            self.stats['prompt_tokens'] += o['n_prompt']
            self.stats['output_tokens'] += o['n_out']
            if self.log is not None:
                self.log.write(json.dumps(dict(o, messages=r['messages'], max_new=r['max_new'])) + '\n')
        self.stats['seconds'] += time.perf_counter() - t0
        self.stats['calls'] += len(reqs)
        self.stats['hits'] += len(reqs) - len(todo)
        if not self.enabled:
            return fresh
        for o in fresh:
            self.cache[o['key']] = o
        return [self.cache[k] for k in keys]


def msgs(system, user):
    return [dict(role='system', content=system), dict(role='user', content=user)]


def gen_validated(gen, items, system, cap, validator, tag):
    """items: list of (user_text, validator_args). One syntax/type-only repair for invalid payloads.
    The repair turn sees only the prior output and the validator error; never a label or gold."""
    reqs = [dict(messages=msgs(system, u), max_new=cap, tag=tag) for u, _ in items]
    first = gen.run(reqs)
    results, repair_idx, repair_reqs = [], [], []
    for i, ((u, vargs), o) in enumerate(zip(items, first)):
        res = dict(first_key=o['key'], first_text=o['text'], first_finish=o['finish'], used_repair=False,
                   repair_key=None, value=None, error=None, first_error=None)
        try:
            if o['finish'] != 'stop':
                raise NSRError('TRUNCATED', o['finish'])
            res['value'] = validator(payload(o['text']), *vargs)
        except NSRError as exc:
            res['first_error'] = res['error'] = exc.code
            repair_idx.append(i)
            repair_reqs.append(dict(messages=msgs(system, u) + [
                dict(role='assistant', content=o['text']),
                dict(role='user', content=prompts.REPAIR.format(error=f'{exc.code}: {exc.detail}'[:300]))],
                max_new=cap, tag=tag + ':repair'))
        results.append(res)
    if repair_reqs:
        second = gen.run(repair_reqs)
        for i, o in zip(repair_idx, second):
            res, (u, vargs) = results[i], items[i]
            res.update(used_repair=True, repair_key=o['key'], repair_text=o['text'], repair_finish=o['finish'])
            try:
                if o['finish'] != 'stop':
                    raise NSRError('TRUNCATED', o['finish'])
                res['value'] = validator(payload(o['text']), *vargs)
                res['error'] = None
            except NSRError as exc:
                res['error'] = exc.code
    return results


# ----------------------------- flat solver adapter -----------------------------
class _FlatSnap:
    def __init__(self, index, values):
        self.index, self.values, self.content, self.history = index, values, core.content_hash(values), 'flat'


class FlatStore:
    """Self-contained symbolic problem: the model-selected facts, grouped by the snapshot they claim."""

    def __init__(self, facts):
        self.snaps, self.order = {}, []
        for f in facts:
            lab = f['snapshot']
            if lab not in self.snaps:
                self.snaps[lab] = {}
                self.order.append(lab)
            k = (f['entity'], f['field'], f['topic'])
            self.snaps[lab][k] = core.AMBIG if k in self.snaps[lab] and self.snaps[lab][k] != f['value'] else f['value']
        self._objs = {lab: _FlatSnap(-(i + 1) if lab != 'ORIGINAL' else 0, v) for i, (lab, v) in enumerate(self.snaps.items())}
        self.current = self._objs.get('CURRENT', _FlatSnap(-99, {}))

    def resolve(self, label):
        if not core._snap_ok(label):
            raise NSRError('UNSUPPORTED_SNAPSHOT', repr(label))
        return self._objs.get(label, _FlatSnap(-98, {}))


# ----------------------------- canonical table rendering -----------------------------
CANON = {'team': '{s} is on the {o} team.', 'hub': 'The {s} team is based at the {o} hub.',
         'region': 'The {s} hub is in the {o} region.', 'organization': '{s} works for {o}.',
         'founder': '{s} was founded by {o}.', 'location': '{s} lives in {o}.'}
STANCE_WORD = {'favor': 'in favor', 'oppose': 'opposed'}


def table(values):
    """Canonical rendering of a parsed snapshot as record sentences (the dataset's own record style), in a fixed
    sorted order that carries no trace of the original record order or of any correction."""
    members, rels = {}, []
    for (e, f, t), v in values.items():
        v = 'AMBIGUOUS' if v is core.AMBIG else v
        if f in core.ENUMS:
            members.setdefault((e, t), {})[f] = v
        elif f in CANON:
            rels.append((e, f, CANON[f].format(s=e, o=v)))
        else:  # declared public relation: its own record wording, as a plain sentence
            rels.append((e, f, prompts.PUBLIC_WORDING[f].replace('{S}', e).replace('{O}', v)))
    rows = []
    for (e, t), d in sorted(members.items()):
        parts = [STANCE_WORD.get(d['stance'], d['stance'])] if 'stance' in d else []
        parts += [f'status {d["status"]}'] if 'status' in d else []
        parts += [f'shipment day {d["day"]}'] if 'day' in d else []
        rows.append(f'{e} on the {t} proposal: ' + '; '.join(parts) + '.')
    rows += [x[2] for x in sorted(rels)]
    if prompts.PUBLIC_WORDING:
        return rows
    return [f'Record {i + 1:02d}: {r}' for i, r in enumerate(rows)]


# ----------------------------- native-answer extraction (frozen exact rule) -----------------------------
def native_answer(o, labels=None):
    """After removing terminal chat controls (already excluded by the stop rule) and surrounding whitespace:
    exact label match first; then one terminal period may be removed. Anything else is invalid."""
    if o['finish'] != 'stop':
        return None, 'TRUNCATED'
    t = o['text'].strip()
    if labels is not None:
        if t in labels:
            return t, None
        if t.endswith('.') and t[:-1] in labels:
            return t[:-1], None
        return None, 'INVALID_LABEL'
    if not t or '\n' in t:
        return None, 'INVALID_ENTITY_OUTPUT'
    return t, None  # entity/value output: exact comparison (and one terminal period) is done by the evaluator


def labels_of(question):
    m = render.answer_mapping(question)
    return None if m is None else (m[True], m[False])


# ----------------------------- runtime input allowlist -----------------------------
FORBIDDEN = ('terminal', 'gold', 'spec', 'answer', 'eligib', 'witness', 'semantic', 'mapping', 'alias', 'program',
             'evaluator', 'label', 'chain', 'path', 'decomposition', 'mask', 'plan')
ALLOW = dict(root=('root_id', 'histories', 'questions'),
             history=('history_id', 'records', 'updates', 'typed_commands', 'typed_initial'),
             record=('record_id', 'text'), update=('update_id', 'text'),
             question=('question_id', 'text', 'checkpoint_events', 'after_updates'))


def sanitize_root(root):
    """Enforce the runtime input schema before any candidate/model call: evaluator-style fields are REJECTED,
    any other non-allowlisted metadata is stripped. Returns a new object holding only allowlisted keys."""
    def check(obj, kind):
        for k in obj:
            if any(f in k.lower() for f in FORBIDDEN):
                raise NSRError('POISONED_INPUT', f'{kind} field {k!r}')
        return {k: obj[k] for k in ALLOW[kind] if k in obj}
    out = check(root, 'root')
    out['histories'] = []
    for h in root['histories']:
        hh = check(h, 'history')
        hh['records'] = [check(r, 'record') for r in h['records']]
        hh['updates'] = [check(u, 'update') for u in h['updates']]
        out['histories'].append(hh)
    out['questions'] = [check(q, 'question') for q in root['questions']]
    return out


# ----------------------------- units -----------------------------
def units_of(root):
    """Yield (history, k_updates, questions) evaluation units for a visible root."""
    for h in root['histories']:
        keyf = 'checkpoint_events' if any('checkpoint_events' in q for q in root['questions']) else \
            ('after_updates' if any('after_updates' in q for q in root['questions']) else None)
        if keyf is None:
            yield h, len(h['updates']), root['questions']
        else:
            for k in sorted({q[keyf] for q in root['questions']}):
                yield h, k, [q for q in root['questions'] if q[keyf] == k]


def row(panel, arm, root, h, q, **kw):
    return dict(panel=panel, arm=arm, root_id=root['root_id'], history_id=h['history_id'],
                question_id=q['question_id'], **kw)


# ----------------------------- ISOLATED_SYMBOLIC -----------------------------
class IsolatedState:
    """Per-history candidate state built from independently parsed records and updates."""

    def __init__(self, record_results, typed_initial=None):
        self.intake_errors = []
        writes = list(typed_initial or [])
        for rid, res in record_results:
            if res['error'] is not None:
                self.intake_errors.append(('record', rid, res['error']))
            else:
                writes += res['value']
        self.failed_from = None  # first update index whose intake failed (fail-closed for later states)
        try:
            self.store = core.Store(writes)
            self.state_error = None
        except NSRError as exc:
            self.store, self.state_error = None, exc.code
        self.applied = 0

    def apply(self, k, res=None, typed_event=None):
        if self.store is None:
            return
        if typed_event is None:
            if res['error'] is not None:
                self.intake_errors.append(('update', k, res['error']))
                self.failed_from = self.failed_from or k
                return
            typed_event = res['value']
        if self.failed_from is not None:
            return
        out = self.store.apply(typed_event)
        if out['status'] != 'ACCEPTED':
            self.intake_errors.append(('update', k, 'REJECTED_' + out['error']))
            self.failed_from = k
        else:
            self.applied += 1

    def answer(self, program_res, question, use_view=True):
        """Exact execution + deterministic rendering. Returns dict(answer, error, result)."""
        if self.store is None:
            return dict(answer=None, error='STATE_' + self.state_error, result=None)
        if program_res['error'] is not None:
            return dict(answer=None, error='COMPILE_' + program_res['error'], result=None)
        kind, prog = program_res['value']
        if kind == 'UNSUPPORTED':
            return dict(answer=None, error='ABSTAIN_UNSUPPORTED', result=None)
        if self.failed_from is not None:
            reads_current = 'CURRENT' in json.dumps(prog)
            if reads_current or any(f'EVENT:{j}' in json.dumps(prog) for j in range(self.failed_from, 200)):
                return dict(answer=None, error='INTAKE_INCOMPLETE', result=None)
        if use_view:
            vid = self.store.register_view(prog)
            result = self.store.read_view(vid)
        else:
            result = core.execute(prog, self.store)
        ans, err = render.render(result, question)
        return dict(answer=ans, error=err, result=result)


def run_isolated(gen, panel, roots, out, typed_commands=None, arm='ISOLATED_SYMBOLIC', structured=None):
    """History-blind query compilation (question + fixed schema only), independent record/update intake,
    versioned exact state, validated exact execution, deterministic rendering."""
    rsys, usys, csys = prompts.record_system(), prompts.update_system(), prompts.compiler_system()
    lines = sorted({r['text'] for root in roots for h in root['histories'] for r in h['records']}) if structured is None else []
    stmts = sorted({u['text'] for root in roots for h in root['histories'] for u in h['updates']})
    quests = sorted({q['text'] for root in roots for q in root['questions']})
    rec = dict(zip(lines, gen_validated(gen, [(prompts.record_user(l), (l,)) for l in lines], rsys, CAPS['record'],
                                        validate_record, f'{panel}:record')))
    upd = dict(zip(stmts, gen_validated(gen, [(prompts.update_user(s), (s,)) for s in stmts], usys, CAPS['update'],
                                        validate_update, f'{panel}:update')))
    cmp_ = dict(zip(quests, gen_validated(gen, [(prompts.compiler_user(q), (q,)) for q in quests], csys, CAPS['query'],
                                          validate_query, f'{panel}:query')))
    rows, states, tables, recheck = [], {}, {}, dict(units=0, current_reread=0, mismatch=0)
    for root in roots:
        for h in root['histories']:
            if structured is not None:
                st = IsolatedState([], typed_initial=structured[(root['root_id'], h['history_id'])])
            else:
                st = IsolatedState([(r['record_id'], rec[r['text']]) for r in h['records']])
            done = 0
            tcmd = typed_commands.get((root['root_id'], h['history_id'])) if typed_commands else None
            n_up = len(tcmd) if tcmd is not None else len(h['updates'])
            unit_iter = units_of(dict(root, histories=[h])) if tcmd is None else [(h, n_up, root['questions'])]
            for hh, k, qs in unit_iter:
                while done < k:
                    if tcmd is not None:
                        st.apply(done + 1, typed_event=tcmd[done])
                    else:
                        st.apply(done + 1, res=upd[h['updates'][done]['text']])
                    done += 1
                ok = st.store is not None and st.failed_from is None
                tables[(root['root_id'], h['history_id'], k)] = (
                    table(st.store.current.values), table(st.store.snapshots[0].values)) if ok else None
                before = st.store.current.content if st.store else None
                first = []
                for q in qs:
                    a = st.answer(cmp_[q['text']], q['text'])
                    first.append(a)
                    res = a['result'] or {}
                    rows.append(row(panel, arm, root, h, q, answer=a['answer'], error=a['error'],
                                    program=cmp_[q['text']]['value'][1] if cmp_[q['text']]['error'] is None else None,
                                    compile_key=cmp_[q['text']]['first_key'], compile_repair=cmp_[q['text']]['used_repair'],
                                    read_set=res.get('read_set'), served=res.get('served'), value=res.get('value'),
                                    exec_status=res.get('status'), exec_error=res.get('error'),
                                    snapshot_index=st.store.current.index if st.store else None,
                                    content=st.store.current.content if st.store else None,
                                    intake_errors=len(st.intake_errors)))
                # after every historical read: fresh current reads must be unchanged (disposable workspace)
                recheck['units'] += 1
                for q, a in zip(qs, first):
                    b = st.answer(cmp_[q['text']], q['text'], use_view=False)
                    recheck['current_reread'] += 1
                    if (a['answer'], a['error']) != (b['answer'], b['error']):
                        recheck['mismatch'] += 1
                if st.store and st.store.current.content != before:
                    raise RuntimeError('a read changed the current snapshot')
            states[(root['root_id'], h['history_id'])] = st
    out['rows'] += rows
    out.setdefault('recheck', {})[panel + ':' + arm] = recheck
    out.setdefault('intake', {})[panel + ':' + arm] = dict(
        records={l: _slim(r) for l, r in rec.items()}, updates={s: _slim(r) for s, r in upd.items()},
        queries={q: _slim(r) for q, r in cmp_.items()})
    return states, cmp_, tables


def _slim(r):
    v = r['value']
    return dict(value=v if not isinstance(v, tuple) else list(v), error=r['error'], first_error=r['first_error'],
                used_repair=r['used_repair'], first_key=r['first_key'], repair_key=r['repair_key'])


# ----------------------------- NATIVE / CURRENT / FLAT -----------------------------
def run_native(gen, panel, roots, out):
    sysm = prompts.native_system()
    reqs, meta = [], []
    for root in roots:
        for h, k, qs in units_of(root):
            recs, ups = [r['text'] for r in h['records']], [u['text'] for u in h['updates'][:k]]
            for q in qs:
                reqs.append(dict(messages=msgs(sysm, prompts.native_user(recs, ups, q['text'])), max_new=CAPS['answer'],
                                 tag=f'{panel}:native'))
                meta.append((root, h, q))
    for (root, h, q), o in zip(meta, gen.run(reqs)):
        ans, err = native_answer(o, labels_of(q['text']))
        out['rows'].append(row(panel, 'NATIVE_NEURAL', root, h, q, answer=ans, error=err, raw=o['text'], gen_key=o['key']))


def run_current(gen, panel, roots, out, tables, arm='CURRENT_NEURAL'):
    """Neural answering from the candidate-parsed (or, diagnostically, typed-reference) canonical tables.
    tables: {(root_id, history_id, k): (current rows, original rows) or None}."""
    sysm = prompts.current_system()
    reqs, meta = [], []
    for root in roots:
        for h, k, qs in units_of(root):
            tb = tables.get((root['root_id'], h['history_id'], k))
            if tb is None:
                for q in qs:
                    out['rows'].append(row(panel, arm, root, h, q, answer=None, error='STATE_UNAVAILABLE'))
                continue
            for q in qs:
                reqs.append(dict(messages=msgs(sysm, prompts.current_user(tb[0], tb[1], q['text'])), max_new=CAPS['answer'],
                                 tag=f'{panel}:current'))
                meta.append((root, h, q))
    for (root, h, q), o in zip(meta, gen.run(reqs)):
        ans, err = native_answer(o, labels_of(q['text']))
        out['rows'].append(row(panel, arm, root, h, q, answer=ans, error=err, raw=o['text'], gen_key=o['key']))


def run_flat(gen, panel, roots, out):
    sysm = prompts.flat_system()
    items, meta = [], []
    for root in roots:
        for h, k, qs in units_of(root):
            recs, ups = [r['text'] for r in h['records']], [u['text'] for u in h['updates'][:k]]
            ids = {f'Record {r["record_id"]}' for r in h['records']} | {f'Correction {u["update_id"]}' for u in h['updates'][:k]}
            for q in qs:
                items.append((prompts.flat_user(recs, ups, q['text']), (q['text'], ids)))
                meta.append((root, h, q))
    for (root, h, q), res in zip(meta, gen_validated(gen, items, sysm, CAPS['flat'], validate_flat, f'{panel}:flat')):
        ans = err = prog = facts = result = None
        if res['error'] is not None:
            err = 'TRANSLATE_' + res['error']
        else:
            kind, prog, facts = res['value']
            if kind == 'UNSUPPORTED':
                err, prog = 'ABSTAIN_UNSUPPORTED', None
            else:
                result = core.execute(prog, FlatStore(facts))
                ans, err = render.render(result, q['text'])
        out['rows'].append(row(panel, 'FLAT_SYMBOLIC', root, h, q, answer=ans, error=err, program=prog, facts=facts,
                               gen_key=res['first_key'], used_repair=res['used_repair'], first_error=res['first_error'],
                               exec_error=(result or {}).get('error')))


# ----------------------------- diagnostics (boundary panel) -----------------------------
def run_exposed_compiler(gen, panel, roots, out, states):
    """Same compiler prompt + the obsolete history; same parsed current state and executor."""
    csys = prompts.compiler_system()
    items, meta = [], []
    for root in roots:
        for h, k, qs in units_of(root):
            recs, ups = [r['text'] for r in h['records']], [u['text'] for u in h['updates'][:k]]
            for q in qs:
                items.append((prompts.exposed_compiler_user(recs, ups, q['text']), (q['text'],)))
                meta.append((root, h, q))
    for (root, h, q), res in zip(meta, gen_validated(gen, items, csys, CAPS['query'], validate_query, f'{panel}:exposed')):
        st = states[(root['root_id'], h['history_id'])]
        a = st.answer(res, q['text'], use_view=False)
        out['rows'].append(row(panel, 'HISTORY_EXPOSED_QUERY_COMPILER', root, h, q, answer=a['answer'], error=a['error'],
                               program=res['value'][1] if res['error'] is None else None, gen_key=res['first_key'],
                               used_repair=res['used_repair']))


def run_neural_renderer(gen, panel, roots, out, isolated_rows):
    """Neural renderer that sees the exact computed answer and the full history (diagnostic only)."""
    sysm = prompts.renderer_system()
    ans = {(r['root_id'], r['history_id'], r['question_id']): r['answer'] for r in isolated_rows}
    reqs, meta = [], []
    for root in roots:
        for h, k, qs in units_of(root):
            recs, ups = [r['text'] for r in h['records']], [u['text'] for u in h['updates'][:k]]
            for q in qs:
                a = ans.get((root['root_id'], h['history_id'], q['question_id']))
                if a is None:
                    out['rows'].append(row(panel, 'NEURAL_RENDERER_OVERRIDE', root, h, q, answer=None, error='NO_COMPUTED_ANSWER'))
                    continue
                reqs.append(dict(messages=msgs(sysm, prompts.renderer_user(recs, ups, q['text'], a)), max_new=CAPS['answer'],
                                 tag=f'{panel}:renderer'))
                meta.append((root, h, q, a))
    for (root, h, q, a), o in zip(meta, gen.run(reqs)):
        got, err = native_answer(o, labels_of(q['text']))
        out['rows'].append(row(panel, 'NEURAL_RENDERER_OVERRIDE', root, h, q, answer=got, error=err, computed=a,
                               raw=o['text'], gen_key=o['key']))
