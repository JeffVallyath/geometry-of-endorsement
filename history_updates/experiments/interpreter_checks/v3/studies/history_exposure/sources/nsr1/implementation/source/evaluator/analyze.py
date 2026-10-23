"""NSR1 frozen analysis (evaluator side). Reads pod outputs + evaluator gold; writes NSR1_ANALYSIS.json.

Statistics: 10,000 paired ROOT bootstrap draws (seed config.seeds.bootstrap), every history, variant, mapping and
question of a root kept together. All scheduled questions are in every denominator; parser failures, abstentions and
invalid outputs are errors. Typed-oracle results are never pooled with natural-input results.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import score, sqlcheck  # noqa: E402
from nsr import core  # noqa: E402

CFG = json.loads((ROOT / 'governing_inputs' / 'NSR1_CONFIG.json').read_text())
DRAWS, SEED = CFG['criteria']['interval_bootstraps'], CFG['seeds']['bootstrap']
PAIRS = [(1, 2), (1, 4), (2, 4), (3, 5), (3, 6), (5, 6)]  # structurally matched histories (equal number of obsolete facts)


# ----------------------------- bootstrap -----------------------------
def boot_ratio(num, den, lo=0.025, hi=0.975):
    """Root-cluster bootstrap of sum(num)/sum(den); num/den are dicts root -> value."""
    roots = sorted(den)
    n, d = np.array([num.get(r, 0) for r in roots], float), np.array([den[r] for r in roots], float)
    if d.sum() == 0:
        return dict(point=None, lower=None, upper=None, roots=len(roots))
    rng = np.random.default_rng(SEED)
    stats = []
    for i in range(0, DRAWS, 500):
        idx = rng.integers(len(roots), size=(min(500, DRAWS - i), len(roots)))
        dd = d[idx].sum(1)
        stats.append(np.where(dd > 0, n[idx].sum(1) / np.where(dd > 0, dd, 1), np.nan))
    s = np.concatenate(stats)
    s = s[~np.isnan(s)]
    return dict(point=float(n.sum() / d.sum()), lower=float(np.quantile(s, lo)), upper=float(np.quantile(s, hi)),
                roots=len(roots), level=[lo, hi], draws=DRAWS)


def boot_diff(a, b, lo=0.0125, hi=0.9875):
    """Paired root bootstrap of mean(a_root - b_root)."""
    roots = sorted(a)
    x = np.array([a[r] - b[r] for r in roots], float)
    rng = np.random.default_rng(SEED)
    means = np.concatenate([x[rng.integers(len(x), size=(min(500, DRAWS - i), len(x)))].mean(1) for i in range(0, DRAWS, 500)])
    return dict(point=float(x.mean()), lower=float(np.quantile(means, lo)), upper=float(np.quantile(means, hi)),
                roots=len(roots), level=[lo, hi], draws=DRAWS)


# ----------------------------- program equivalence (semantic, not syntactic) -----------------------------
def _norm_leaf(leaf):
    if leaf['op'] == 'READ' and leaf['field'] in core.RELATIONS:
        return dict(op='FOLLOW', entity=leaf['entity'], path=[leaf['field']], snapshot=leaf['snapshot'])
    return leaf


def _norm(node):
    if node['op'] in ('READ', 'FOLLOW'):
        return _norm_leaf(node)
    return {k: (_norm(v) if isinstance(v, dict) else v) for k, v in node.items()}


def equivalent(p, q):
    """Joint function equivalence: equal on every assignment of the union of their grounded reads."""
    if p is None or q is None:
        return False
    try:
        p, q = _norm(p), _norm(q)
        tp, tq = core.static_type(p), core.static_type(q)
    except core.NSRError:
        return False
    if (tp == 'bool') != (tq == 'bool'):
        return False
    if tp != 'bool':
        return core.canonical(p) == core.canonical(q)
    keys = []
    for leaf in core._leaves(p, []) + core._leaves(q, []):
        c = core.canonical(leaf)
        if c not in keys:
            keys.append(c)
    doms = []
    for c in keys:
        t = core.static_type(json.loads(c))
        doms.append(list(core.ENUMS[t]) if t in core.ENUMS else ['<e1>', '<e2>', '<e3>'])
    import itertools
    if np.prod([len(d) for d in doms]) > 20000:
        return core.canonical(p) == core.canonical(q)
    for combo in itertools.product(*doms):
        env = dict(zip(keys, combo))
        if core._eval_abstract(p, env) != core._eval_abstract(q, env):
            return False
    return True


def program_diff(p, q):
    """Which stage-level elements differ between a compiled program and the intended one."""
    if p is None:
        return ['no_program']
    lp, lq = [_norm_leaf(x) for x in core._leaves(p, [])], [_norm_leaf(x) for x in core._leaves(q, [])]
    out = []
    if {x['entity'] for x in lp} != {x['entity'] for x in lq}:
        out.append('entity')
    if {x.get('topic') for x in lp} != {x.get('topic') for x in lq}:
        out.append('topic')
    if {x.get('field') or tuple(x.get('path', [])) for x in lp} != {x.get('field') or tuple(x.get('path', [])) for x in lq}:
        out.append('field_or_path')
    if {x['snapshot'] for x in lp} != {x['snapshot'] for x in lq}:
        out.append('snapshot')
    ops = lambda n, acc: (acc.append(n['op']) if n['op'] not in ('READ', 'FOLLOW') else None,
                          [ops(n[k], acc) for k in ('arg', 'left', 'right') if isinstance(n.get(k), dict)], acc)[2]
    if sorted(ops(p, [])) != sorted(ops(q, [])):
        out.append('operator')
    if not out and [x['entity'] for x in lp] != [x['entity'] for x in lq]:
        out.append('operand_order')
    return out or ['other']


# ----------------------------- loading -----------------------------
def ev_roots(data, panel):
    p = Path(data) / 'evaluator' / panel / 'roots.jsonl'
    return [json.loads(l) for l in open(p, encoding='utf-8')] if p.exists() else []


def intended_map(data, panel):
    return {q['question_id']: q for r in ev_roots(data, panel) for q in r['questions']}


def fam_table(scored):
    t = collections.OrderedDict()
    for r in scored:
        for fam in ('ALL', r['family']):
            a = t.setdefault(r['arm'], collections.OrderedDict()).setdefault(fam, dict(n=0, correct=0, invalid=0, abstain=0, errors=collections.Counter()))
            a['n'] += 1
            a['correct'] += r['correct']
            a['invalid'] += not r['valid']
            a['abstain'] += (r.get('error') or '').startswith('ABSTAIN')
            if not r['correct']:
                a['errors'][r.get('error') or 'WRONG_ANSWER'] += 1
    for arm in t.values():
        for a in arm.values():
            a['accuracy'] = a['correct'] / a['n']
            a['errors'] = dict(a['errors'].most_common(8))
    return t


def per_root(scored, arm, pred=lambda r: True):
    num, den = collections.Counter(), collections.Counter()
    for r in scored:
        if r['arm'] == arm and pred(r):
            den[r['root_id']] += 1
            num[r['root_id']] += r['correct']
    return num, den


def bundles(scored, arm, pred=lambda r: True):
    """Whole-root bundle: every scheduled answer of the root (all histories, questions, mappings) correct."""
    num, den = per_root(scored, arm, pred)
    return {r: float(num[r] == den[r]) for r in den}


# ----------------------------- panel analyses -----------------------------
def core_cells(data, panel, scored, arm):
    ev = {r['root_id']: r for r in ev_roots(data, panel)}
    qmeta = {q['question_id']: (q, r) for r in ev.values() for q in r['questions']}
    cells = {}
    for r in scored:
        if r['arm'] != arm or r['family'] != 'joint':
            continue
        q, er = qmeta[r['question_id']]
        bit = dict(a=er['combo'][0], b=er['combo'][1], c=er['combo'][2])
        first, second = {'ab': ('a', 'b'), 'ca': ('c', 'a'), 'bc': ('b', 'c')}[q['target_pair']]
        key = f'{q["kind"]}|{"T" if bit[first] else "F"}{"T" if bit[second] else "F"}|draw{q["draw"]}'
        c = cells.setdefault(key, [0, 0])
        c[0] += r['correct']
        c[1] += 1
    return {k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for k, v in sorted(cells.items())}


def factorial_analysis(data, panel, scored, arm):
    ev = {r['root_id']: r for r in ev_roots(data, panel)}
    qmeta = {q['question_id']: q for r in ev.values() for q in r['questions']}
    dbs = {(r['root_id'], t['history_id']): t for r in ev.values() for t in r['typed']}
    rows = [r for r in scored if r['arm'] == arm]
    cells, byq = {}, {}
    for r in rows:
        q, db = qmeta[r['question_id']], dbs[(r['root_id'], r['history_id'])]
        vx, vy = db['operands']['x'] == 'favor', db['operands']['y'] == 'favor'
        first, second = (vx, vy) if q['order'] == 0 else (vy, vx)
        key = f'{q["kind"]}|{"T" if first else "F"}{"T" if second else "F"}|draw{q["draw"]}'
        c = cells.setdefault(key, [0, 0])
        c[0] += r['correct']
        c[1] += 1
        byq[(r['root_id'], db['N'], db['state'], r['question_id'])] = r
    out = dict(cells={k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for k, v in sorted(cells.items())})
    out['min_cell_accuracy'] = min(v['accuracy'] for v in out['cells'].values()) if out['cells'] else None
    by_n = collections.defaultdict(lambda: [0, 0])
    for (root, N, state, qid), r in byq.items():
        by_n[N][0] += r['correct']
        by_n[N][1] += 1
    out['by_N'] = {str(N): dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for N, v in sorted(by_n.items())}
    # coherent triples, commutativity, answer-change / answer-invariance edges, locality across N
    tri = comm = ch = inv = loc = 0
    tri_n = comm_n = ch_n = inv_n = loc_n = 0
    sem = lambda r: None if r['answer'] is None else (r['answer'] == r['gold']) == bool(r['semantic'])  # produced truth value
    groups = collections.defaultdict(dict)
    for (root, N, state, qid), r in byq.items():
        q = qmeta[qid]
        groups[(root, N, state, q['draw'], q['order'])][q['kind']] = sem(r)
    for key, g in groups.items():
        if len(g) == 3:
            tri_n += 1
            ok = any(g['both'] == (a and b) and g['either'] == (a or b) and g['same'] == (a == b) for a in (0, 1) for b in (0, 1))
            tri += bool(ok and None not in g.values())
    for (root, N, state, draw, order), g in groups.items():
        if order == 0 and (root, N, state, draw, 1) in groups:
            for kind in g:
                comm_n += 1
                comm += g[kind] is not None and g[kind] == groups[(root, N, state, draw, 1)].get(kind)
    ids = collections.defaultdict(dict)
    for (root, N, state, qid), r in byq.items():
        ids[(root, N, qid)][state] = r
    for (root, N, qid), st in ids.items():
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                if s1 in st and s2 in st:
                    a, b = st[s1], st[s2]
                    both_ok = a['correct'] and b['correct']
                    if a['semantic'] != b['semantic']:
                        ch_n += 1
                        ch += both_ok
                    else:
                        inv_n += 1
                        inv += both_ok
    across = collections.defaultdict(dict)
    for (root, N, state, qid), r in byq.items():
        across[(root, state, qid)][N] = r['answer']
    for k, d in across.items():
        if len(d) > 1:
            loc_n += 1
            loc += len(set(d.values())) == 1 and None not in d.values()
    out.update(coherent_triples=dict(ok=tri, n=tri_n), commutativity=dict(ok=comm, n=comm_n),
               required_change_edges=dict(ok=ch, n=ch_n), invariance_edges=dict(ok=inv, n=inv_n),
               irrelevant_record_locality=dict(identical_answer_across_N=loc, n=loc_n))
    return out


def witness_analysis(scored, baseline='NATIVE_NEURAL', candidates=('ISOLATED_SYMBOLIC', 'FLAT_SYMBOLIC', 'CURRENT_NEURAL')):
    """Fresh-core analogue of the frozen A10 rule. Eligibility uses ONLY the baseline arm: every control (12 direct,
    2 same-pair and 6 other-pair functions in the target's format) valid and correct in all eight histories, the
    target correct at the clean/no-op reference history h0, and a valid disagreement on the target between two
    structurally matched histories. Complete repair = target and every one of those controls correct in all eight
    histories for the candidate."""
    idx = collections.defaultdict(dict)
    for r in scored:
        idx[r['arm']][(r['root_id'], r['history_id'], r['question_id'])] = r
    roots = sorted({r['root_id'] for r in scored})
    qids = collections.defaultdict(set)
    meta = {}
    for r in scored:
        qids[r['root_id']].add(r['question_id'])
        meta[r['question_id']] = r
    hs = [f'h{i}' for i in range(8)]
    out = {}
    W = []
    for root in roots:
        qs = sorted(qids[root])
        direct = [q for q in qs if meta[q]['family'] == 'direct']
        for draw in ('d0', 'd1'):
            joint = [q for q in qs if meta[q]['family'] == 'joint' and f'-{draw}-' in q]
            for t in joint:
                controls = direct + [q for q in joint if q != t]
                b = idx[baseline]
                ok = lambda arm, h, q: idx[arm][(root, h, q)]['correct']
                elig = all(ok(baseline, h, c) for h in hs for c in controls) and ok(baseline, 'h0', t)
                dis = any(b[(root, f'h{x}', t)]['valid'] and b[(root, f'h{y}', t)]['valid'] and
                          b[(root, f'h{x}', t)]['answer'] != b[(root, f'h{y}', t)]['answer'] for x, y in PAIRS)
                any_wrong = any(not ok(baseline, h, t) for h in hs)
                W.append(dict(root=root, target=t, W=bool(elig and dis), history_dependent_error_no_guard=bool(any_wrong),
                              controls=controls))
    wit = [w for w in W if w['W']]
    out['baseline'] = baseline
    out['target_opportunities'] = len(W)
    out['baseline_witnesses'] = len(wit)
    out['witness_roots'] = len({w['root'] for w in wit})
    out['baseline_targets_with_any_history_error'] = sum(w['history_dependent_error_no_guard'] for w in W)
    for arm in candidates:
        if arm not in idx:
            continue
        ok = lambda h, q, root: idx[arm][(root, h, q)]['correct']
        num_t, num_b, den = collections.Counter(), collections.Counter(), collections.Counter()
        for w in wit:
            den[w['root']] += 1
            tgt = all(ok(h, w['target'], w['root']) for h in hs)
            bun = tgt and all(ok(h, c, w['root']) for h in hs for c in w['controls'])
            num_t[w['root']] += tgt
            num_b[w['root']] += bun
        harm_n, harm_d = collections.Counter(), collections.Counter()
        for key, br in idx[baseline].items():
            cr = idx[arm].get(key)
            if cr is None:
                continue
            harm_d[key[0]] += 1
            harm_n[key[0]] += bool(br['correct'] and not cr['correct'])
        out[arm] = dict(target_repaired=sum(num_t.values()), complete_preserved_repair=sum(num_b.values()), witnesses=len(wit),
                        complete_repair_rate=boot_ratio(num_b, den, 0.05, 0.95) if wit else None,
                        newly_wrong_all_scheduled=boot_ratio(harm_n, harm_d, 0.05, 0.95),
                        newly_wrong_count=sum(harm_n.values()), scheduled=sum(harm_d.values()))
    return out


def isolation_analysis(scored, arm='ISOLATED_SYMBOLIC'):
    """S3: for current-only programs the canonical answer must be a function of (parsed program, parsed current state)."""
    g = collections.defaultdict(list)
    for r in scored:
        if r['arm'] == arm and r.get('program') is not None:
            prog = json.dumps(r['program'], sort_keys=True)
            if 'ORIGINAL' in prog or 'EVENT:' in prog:
                continue
            # Analysis-only correction (2026-10-01, after the final outputs existed; documented in DEV_RECORD.md): the group key
            # no longer includes the snapshot index, so histories with different numbers of events are compared too.
            g[(r['root_id'], r['question_id'])].append(r)
    out = dict(current_question_groups=len(g), groups_with_answer_change=0, explained_by_parsed_state=0, unexplained=0,
               archive_reads_in_current_programs=0, examples=[])
    for key, rows in g.items():
        if len({(r['answer'], r['error']) for r in rows}) > 1:
            out['groups_with_answer_change'] += 1
            by_content = collections.defaultdict(set)
            for r in rows:
                by_content[r['content']].add((r['answer'], r['error']))
            if all(len(v) == 1 for v in by_content.values()):
                out['explained_by_parsed_state'] += 1
            else:
                out['unexplained'] += 1
                out['examples'].append(key)
        for r in rows:
            if r.get('read_set') and r.get('snapshot_index') is not None and any(x[0] != r['snapshot_index'] for x in r['read_set']):
                out['archive_reads_in_current_programs'] += 1
    return out


def history_dependence(scored):
    """Descriptive only (added after the final outputs existed; no gate uses it): for questions about the CURRENT
    state, how often does an arm's answer differ across the matched histories of one root, although every history
    ends in the same current state?"""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in scored:
        if r['family'] != 'historical' and r.get('kind') != 'hist_dest':
            g[r['arm']][(r['root_id'], r['question_id'])].append(r)
    out = {}
    for arm, groups in g.items():
        varied = sum(len({r['answer'] for r in v}) > 1 for v in groups.values())
        allc = sum(all(r['correct'] for r in v) for v in groups.values())
        consistent_wrong = sum(len({r['answer'] for r in v}) == 1 and not v[0]['correct'] for v in groups.values())
        out[arm] = dict(question_groups=len(groups), answer_varies_across_histories=varied, correct_in_every_history=allc,
                        same_wrong_or_missing_answer_in_every_history=consistent_wrong)
    return out


def intake_agreement(data, out_dir, panel, arm='ISOLATED_SYMBOLIC'):
    """Stage-level semantic agreement of the intake (evaluator-side): each distinct record line and each distinct
    correction statement, parsed value versus the typed reference. Descriptive; added after the final outputs existed."""
    ev = {r['root_id']: r for r in ev_roots(data, panel)}
    if not ev:
        return None
    intake = sqlcheck.load_intake(out_dir, panel, arm)
    rec_ok, upd_ok, bad = {}, {}, []
    for line in open(Path(data) / 'visible' / panel / 'roots.jsonl', encoding='utf-8'):
        root = json.loads(line)
        for h, th in zip(root['histories'], ev[root['root_id']]['typed']):
            i = 0
            for rec in h['records']:
                if rec['text'] not in intake['records']:
                    i += 3 if ' proposal: ' in rec['text'] else 1
                    continue
                n = 3 if ' proposal: ' in rec['text'] else 1
                want = th['initial'][i:i + n]
                i += n
                got = intake['records'][rec['text']]['value']
                ok = got is not None and sorted(map(core.canonical, got)) == sorted(map(core.canonical, want))
                rec_ok[rec['text']] = ok
                if not ok and len(bad) < 12 and rec['text'] not in [b_['text'] for b_ in bad]:
                    bad.append(dict(text=rec['text'], parsed=got, intended=want))
            for u, want in zip(h['updates'], th['events']):
                if u['text'] not in intake['updates']:
                    continue
                got = intake['updates'][u['text']]['value']
                norm = lambda e: None if e is None else {k: sorted(map(core.canonical, v)) for k, v in e.items() if v}
                # a restore and the literal write of the same value are different events (the restore reads an archive)
                upd_ok[u['text']] = norm(got) == norm(want)
                if not upd_ok[u['text']] and len(bad) < 12 and u['text'] not in [b_['text'] for b_ in bad]:
                    bad.append(dict(text=u['text'], parsed=got, intended=want))
    return dict(distinct_record_lines=len(rec_ok), record_lines_parsed_as_intended=sum(rec_ok.values()),
                distinct_update_statements=len(upd_ok), update_statements_parsed_as_intended=sum(upd_ok.values()), mismatches=bad)


def boundary_analysis(scored):
    """Diagnostics: history-exposed query compiler and neural renderer vs the trusted path (same parsed state)."""
    idx = collections.defaultdict(dict)
    for r in scored:
        idx[r['arm']][(r['root_id'], r['history_id'], r['question_id'])] = r
    iso, exp, ren = idx['ISOLATED_SYMBOLIC'], idx.get('HISTORY_EXPOSED_QUERY_COMPILER', {}), idx.get('NEURAL_RENDERER_OVERRIDE', {})
    out = {}
    cur = [k for k, r in iso.items() if r['family'] != 'historical']
    for name, arm in (('blind_compiler', iso), ('exposed_compiler', exp), ('neural_renderer', ren)):
        if not arm:
            continue
        num, den = collections.Counter(), collections.Counter()
        for k in cur:
            den[k[0]] += 1
            num[k[0]] += arm[k]['correct']
        out[name] = dict(accuracy=boot_ratio(num, den), n=sum(den.values()), correct=sum(num.values()))
    if exp:
        groups = collections.defaultdict(list)
        for k in cur:
            groups[(k[0], k[2])].append(exp[k])
        varied_prog = sum(len({json.dumps(r.get('program'), sort_keys=True) for r in v}) > 1 for v in groups.values())
        varied_ans = sum(len({r['answer'] for r in v}) > 1 for v in groups.values())
        differs = sum(exp[k]['answer'] != iso[k]['answer'] for k in cur)
        out['exposed_compiler'].update(question_groups=len(groups), groups_with_history_dependent_program=varied_prog,
                                       groups_with_history_dependent_answer=varied_ans, rows_differing_from_blind=differs,
                                       invalid=sum(not exp[k]['valid'] for k in cur))
        num, den = collections.Counter(), collections.Counter()
        a = {k[0]: 0 for k in cur}
        pa, pb = collections.Counter(), collections.Counter()
        for k in cur:
            pa[k[0]] += iso[k]['correct']
            pb[k[0]] += exp[k]['correct']
            den[k[0]] += 1
        out['blind_minus_exposed'] = boot_diff({r: pa[r] / den[r] for r in den}, {r: pb[r] / den[r] for r in den}, 0.025, 0.975)
    if ren:
        changed = sum(ren[k]['answer'] != iso[k]['answer'] for k in cur if iso[k]['answer'] is not None)
        groups = collections.defaultdict(list)
        for k in cur:
            groups[(k[0], k[2])].append(ren[k])
        out['neural_renderer'].update(rows_changed_from_computed=changed, rows_with_computed=sum(iso[k]['answer'] is not None for k in cur),
                                      groups_with_history_dependent_answer=sum(len({r['answer'] for r in v}) > 1 for v in groups.values()),
                                      invalid=sum(not ren[k]['valid'] for k in cur))
        pa, pb, den = collections.Counter(), collections.Counter(), collections.Counter()
        for k in cur:
            pa[k[0]] += iso[k]['correct']
            pb[k[0]] += ren[k]['correct']
            den[k[0]] += 1
        out['deterministic_minus_neural_renderer'] = boot_diff({r: pa[r] / den[r] for r in den}, {r: pb[r] / den[r] for r in den}, 0.025, 0.975)
    return out


def localize(data, panel, scored, arm, intended):
    """Stage-level error localization for a program-producing arm (semantic comparisons are evaluator-side)."""
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    diffs = collections.Counter()
    for r in scored:
        if r['arm'] != arm:
            continue
        q = intended.get(r['question_id'])
        prog = r.get('program')
        eq = equivalent(prog, q['program']) if (q and prog is not None) else None
        stats['rows'] += 1
        if prog is not None:
            stats['programs'] += 1
            stats['program_semantically_equivalent'] += bool(eq)
            stats['program_exact_canonical'] += core.canonical(prog) == core.canonical(q['program']) if q else 0
        if r['correct']:
            stats['correct'] += 1
            if eq is False:
                stats['correct_despite_nonequivalent_program'] += 1
            continue
        err = r.get('error') or ''
        if err.startswith('ABSTAIN'):
            stage = 'abstention'
        elif err.startswith(('COMPILE_', 'TRANSLATE_')):
            stage = 'program_or_translation_invalid'
        elif eq is False:
            stage = 'program_semantic'
            for d in program_diff(prog, q['program']):
                diffs[d] += 1
        elif err.startswith(('STATE_', 'INTAKE_', 'EXECUTION_')) or eq:
            stage = 'intake_or_fact_selection'
        else:
            stage = 'answer_reading' if prog is None else 'other'
        if r.get('exec_status') == 'OK' and r.get('value') == r['semantic'] and r['answer'] != r['gold'] and eq:
            stage = 'rendering'
        stats['error_' + stage] += 1
        if len(examples[stage]) < 5:
            examples[stage].append(dict(root=r['root_id'], history=r['history_id'], question=r['question_id'], error=r.get('error'),
                                        answer=r['answer'], gold=r['gold'], program=prog))
    return dict(stats=dict(stats), semantic_diff_elements=dict(diffs), examples=dict(examples))


# ----------------------------- old A10 cohort -----------------------------
def a10_analysis(source_zip, scored, package_csv):
    """Fixed original cohort (83 W), per saved editor seed; candidate answers are seed-independent (no editor)."""
    import zipfile
    sys.path.insert(0, str(ROOT / 'audit'))
    import independent_w as IW
    base = 'reproducibility/question_blind_repair/scores/'
    cand = collections.defaultdict(dict)
    for r in scored:
        cand[r['arm']][(r['root_id'], r['history_id'], r['question_id'])] = r['correct']
    valid = collections.defaultdict(dict)
    for r in scored:
        valid[r['arm']][(r['root_id'], r['history_id'], r['question_id'])] = (r['valid'], r['answer'])
    supplied = {(r['root_id'], int(r['seed']), r['query_id']): r for r in csv.DictReader(open(package_csv, newline=''))}
    out = {}
    src = Path(source_zip)
    if src.is_dir():  # extracted score files (the delivered package carries them; no source ZIP needed)
        names = sorted(str(x.relative_to(src)).replace(chr(92), '/') for x in src.rglob('*.jsonl.gz'))
        read = lambda n: (src / n).read_bytes()
        base = ''
    else:
        z = zipfile.ZipFile(src)
        names, read = z.namelist(), z.read
    designs = {}
    for line in gzip.decompress(read(base + 'designs.jsonl.gz')).splitlines():
        d = json.loads(line)
        designs[d['root']['root_id']] = d
    files = sorted(n for n in names if n.startswith(base + 'primary-seed') and n.endswith('.jsonl.gz'))
    assert len(files) == 16, files
    panels = []
    for name in files:
        for line in gzip.decompress(read(name)).splitlines():
            panels.append(json.loads(line))
    for arm in cand:
        tot = collections.Counter()
        by_seed = collections.defaultdict(collections.Counter)
        roots_rep, targets = set(), []
        for panel in panels:
            d, seed, rid = designs[panel['root_id']], panel['seed'], panel['root_id']
            gold = {q['query_id']: IW.truth(d['root']['terminal']['values'], q['spec']) for q in d['questions']}
            rec = {(r['context'], r['query_id']): IW.decode(r) for r in panel['physical']}
            old_good = lambda o, q: rec[f'o{o}/INV_PAIR_NLL_s{seed}', q][1] and rec[f'o{o}/INV_PAIR_NLL_s{seed}', q][0] == gold[q]
            new_good = lambda o, q: cand[arm][(rid, f'o{o}', q)]
            for q in d['questions']:
                kind = q['spec']['kind']
                fam = 'joint' if kind in ('same', 'both', 'either') else kind
                for o in range(8):
                    og, ng = old_good(o, q['query_id']), new_good(o, q['query_id'])
                    for c in (tot, by_seed[seed]):
                        c[f'{fam}_scheduled'] += 1
                        c[f'{fam}_old_error'] += not og
                        c[f'{fam}_new_error'] += not ng
                        c[f'{fam}_corrected'] += (not og) and ng
                        c[f'{fam}_introduced'] += og and not ng
                        c[f'{fam}_invalid'] += not valid[arm][(rid, f'o{o}', q['query_id'])][0]
            for q, m in d['plan']['gates']['targets'].items():
                controls = d['plan']['gates']['direct'] + m['same_pair'] + m['cross_pair']
                sup = supplied[(rid, seed, q)]
                w, brep = sup['baseline_W'] == 'True', sup['A10_B_repair'] == 'True'
                tgt = all(new_good(o, q) for o in range(8))
                bun = tgt and all(new_good(o, c) for o in range(8) for c in controls)
                va = [valid[arm][(rid, f'o{o}', q)] for o in range(8)]
                dis = any(va[a][0] and va[b][0] and va[a][1] != va[b][1] for a, b in d['plan']['structural_pairs'])
                cw = bool(new_good(0, q) and all(new_good(o, c) for o in range(8) for c in controls) and dis)
                whole34 = None
                for c in (tot, by_seed[seed]):
                    c['target_opportunities'] += 1
                    c['baseline_W'] += w
                    c['A10_B_repairs'] += brep
                    c['candidate_target_all_correct'] += tgt
                    c['candidate_bundle_all_correct'] += bun
                    c['W_target_repaired'] += w and tgt
                    c['W_fully_preserved_repair'] += w and bun
                    c['W_retained_as_candidate_witness'] += w and cw
                    c['W_lost_qualification_without_repair'] += w and not bun and not cw
                    c['new_candidate_witnesses'] += (not w) and cw
                if w and bun:
                    roots_rep.add(rid)
                targets.append(dict(root_id=rid, seed=seed, query_id=q, baseline_W=w, A10_B_repair=brep,
                                    candidate_target_all_correct=tgt, candidate_bundle_all_correct=bun))
            draw_ids = [q['query_id'] for q in d['questions']]
            for c in (tot, by_seed[seed]):
                c['case_seed_blocks'] += 1
                c['whole_34_question_blocks_all_correct'] += all(new_good(o, q) for o in range(8) for q in draw_ids)
        out[arm] = dict(totals=dict(tot), by_seed={str(k): dict(v) for k, v in by_seed.items()},
                        repaired_witness_designs=len(roots_rep), targets=targets)
    return out


# ----------------------------- workload cost -----------------------------
def timing_analysis(data, out_dir, panel='workload'):
    p = Path(out_dir) / panel / 'timing_rows.jsonl.gz'
    if not p.exists():
        return None
    rows = [json.loads(l) for l in gzip.open(p, 'rt', encoding='utf-8')]
    gold = {(g['root_id'], g['question_id']): g for g in map(json.loads, open(Path(data) / 'evaluator' / panel / 'gold.jsonl', encoding='utf-8'))}
    arms = sorted({r['arm'] for r in rows})
    comp, med, cold, stages = {}, collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(lambda: collections.defaultdict(list))
    for arm in arms:
        num, den = collections.Counter(), collections.Counter()
        for r in rows:
            if r['arm'] == arm and r['mode'] == 'rep' and r['rep'] == 0:
                for qid, a in r['answers'].items():
                    den[r['root_id']] += 1
                    num[r['root_id']] += score.correct(a['answer'], gold[(r['root_id'], qid)])
        comp[arm] = dict(correct=sum(num.values()), n=sum(den.values()), accuracy=sum(num.values()) / max(1, sum(den.values())),
                         competent=sum(num.values()) / max(1, sum(den.values())) >= 0.99)
        for root in sorted({r['root_id'] for r in rows}):
            reps = [r['total_seconds'] for r in rows if r['arm'] == arm and r['root_id'] == root and r['mode'] == 'rep']
            med[arm][root] = float(np.median(reps))
            c = [r['total_seconds'] for r in rows if r['arm'] == arm and r['root_id'] == root and r['mode'] == 'cold']
            cold[arm][root] = float(c[0]) if c else None
        for r in rows:
            if r['arm'] == arm and r['mode'] == 'rep':
                for k, v in r['stages'].items():
                    if isinstance(v, (int, float)):
                        stages[arm][k].append(v)
    N = {r['root_id']: r['N'] for r in rows}
    out = dict(competence=comp, arms=arms, workloads=len(N),
               median_seconds={a: dict(by_workload=med[a], mean=float(np.mean(list(med[a].values()))),
                                       by_N={str(n): float(np.mean([v for k, v in med[a].items() if N[k] == n])) for n in sorted(set(N.values()))})
                               for a in arms},
               cold_seconds={a: dict(mean=float(np.mean([v for v in cold[a].values() if v is not None])) if any(cold[a].values()) else None) for a in arms},
               stage_means={a: {k: float(np.mean(v)) for k, v in stages[a].items()} for a in arms}, speedups={})
    cand = 'ISOLATED_SYMBOLIC'
    rng = np.random.default_rng(SEED)
    for b in arms:
        if b == cand:
            continue
        roots = sorted(med[cand])
        logr = np.array([np.log(med[b][r] / med[cand][r]) for r in roots])
        bs = np.exp(np.array([logr[rng.integers(len(roots), size=len(roots))].mean() for _ in range(DRAWS)]))
        out['speedups'][b] = dict(geometric_mean_ratio_baseline_over_candidate=float(np.exp(logr.mean())), lower95_one_sided=float(np.quantile(bs, 0.05)),
                                  upper95_one_sided=float(np.quantile(bs, 0.95)), per_workload={r: float(np.exp(x)) for r, x in zip(roots, logr)},
                                  baseline_competent=comp[b]['competent'])
    out['sql_agrees_with_isolated_all_rows'] = all(r.get('agrees_with_isolated', True) for r in rows if r['arm'] == 'SQL_SHARED_PARSE')
    one_time = ('intake_parse', 'state_build', 'update_parse', 'apply', 'compile')
    out['isolated_one_time_vs_recurring'] = dict(
        one_time_parse_seconds_mean=float(sum(np.mean(stages[cand][k]) for k in one_time if k in stages[cand])),
        recurring_execute_seconds_mean=float(np.mean(stages[cand]['execute'])) if 'execute' in stages[cand] else None,
        archive_bytes_mean=float(np.mean(stages[cand]['archive_bytes'])) if 'archive_bytes' in stages[cand] else None)
    return out


# ----------------------------- main per-model analysis -----------------------------
def analyze_model(data, out_dir, panels, model, public_vocab=None):
    res = dict(model=model, panels={})
    for name, panel in panels.items():
        if not (Path(out_dir) / panel / 'rows.jsonl.gz').exists():
            res['panels'][name] = dict(status='NOT_EXECUTED')
            continue
        if name == 'public' and public_vocab:
            core.register_public_relations(public_vocab)
        scored = score.score_panel(data, out_dir, panel)
        stats = json.loads((Path(out_dir) / panel / 'stats.json').read_text())
        P = dict(status='EXECUTED', panel=panel, rows=len(scored), families=fam_table(scored), run=dict(seconds=stats['seconds'], gen=stats['gen'], stages=stats['stages'], recheck=stats.get('recheck')))
        arms = sorted({r['arm'] for r in scored})
        intended = intended_map(data, panel)
        iso_arm = 'ISOLATED_SYMBOLIC'
        for arm in arms:
            num, den = per_root(scored, arm)
            P.setdefault('root_accuracy', {})[arm] = boot_ratio(num, den)
            b = bundles(scored, arm)
            P.setdefault('whole_root_bundle', {})[arm] = dict(roots=len(b), all_correct=int(sum(b.values())), fraction=float(np.mean(list(b.values()))))
        if iso_arm in arms:
            iso = [r for r in scored if r['arm'] == iso_arm]
            sq, parsed = sqlcheck.check(data, out_dir, panel, iso, public_relations=public_vocab if name == 'public' else None)
            P['sql_shared_parse'] = sq
            P['isolation'] = isolation_analysis(scored)
            if intended:
                P['localization'] = {arm: localize(data, panel, scored, arm, intended) for arm in arms if arm in (iso_arm, 'FLAT_SYMBOLIC', 'HISTORY_EXPOSED_QUERY_COMPILER')}
            P['intake_agreement_descriptive'] = intake_agreement(data, out_dir, panel)
            intake = sqlcheck.load_intake(out_dir, panel, iso_arm)
            P['intake'] = {k: dict(n=len(v), errors=sum(x['error'] is not None for x in v.values()), used_repair=sum(x['used_repair'] for x in v.values()),
                                   first_pass_invalid=sum(x['first_error'] is not None for x in v.values())) for k, v in intake.items()}
        if name in ('core', 'boundary'):
            for arm in arms:
                P.setdefault('truth_cells', {})[arm] = core_cells(data, panel, scored, arm)
                cur = lambda r: r['family'] != 'historical'
                b = bundles(scored, arm, cur)
                P.setdefault('whole_root_current_bundle', {})[arm] = dict(all_correct=int(sum(b.values())), roots=len(b))
        if name in ('core', 'boundary', 'graph', 'public'):
            P['history_dependence_descriptive'] = history_dependence(scored)
        if name == 'core':
            P['witness'] = witness_analysis(scored)
            if 'FLAT_SYMBOLIC' in arms:
                a, b = bundles(scored, iso_arm), bundles(scored, 'FLAT_SYMBOLIC')
                P['primary_contrast_whole_root_boolean_isolated_minus_flat'] = boot_diff(a, b)
        if name == 'factorial':
            P['factorial'] = {arm: factorial_analysis(data, panel, scored, arm) for arm in arms}
        if name == 'graph':
            ev = {r['root_id']: r for r in ev_roots(data, panel)}
            for arm in arms:
                by_h, by_k = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
                for r in scored:
                    if r['arm'] == arm:
                        for dct, key in ((by_h, ev[r['root_id']]['hops']), (by_k, r['kind'])):
                            dct[key][0] += r['correct']
                            dct[key][1] += 1
                P.setdefault('by_hops', {})[arm] = {str(k): dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for k, v in sorted(by_h.items())}
                P.setdefault('by_kind', {})[arm] = {k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for k, v in sorted(by_k.items())}
            if 'FLAT_SYMBOLIC' in arms:
                e2e = lambda r: r['kind'] == 'dest'
                na, da = per_root(scored, iso_arm, e2e)
                nb, db = per_root(scored, 'FLAT_SYMBOLIC', e2e)
                P['primary_contrast_graph_end_to_end_isolated_minus_flat'] = boot_diff({r: na[r] / da[r] for r in da}, {r: nb[r] / db[r] for r in db})
        if name == 'boundary':
            P['boundary'] = boundary_analysis(scored)
        if name == 'temporal':
            ev = {r['root_id']: r for r in ev_roots(data, panel)}
            for arm in arms:
                by_f = collections.defaultdict(lambda: [0, 0])
                for r in scored:
                    if r['arm'] == arm:
                        by_f[ev[r['root_id']]['family']][0] += r['correct']
                        by_f[ev[r['root_id']]['family']][1] += 1
                P.setdefault('by_lifecycle_family', {})[arm] = {k: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for k, v in sorted(by_f.items())}
        res['panels'][name] = P
    return res


def secondary_summary(sec):
    out = {}
    for name, p in sec['panels'].items():
        if p.get('status') == 'EXECUTED':
            out[name] = {arm: dict(accuracy=v['ALL']['accuracy'], n=v['ALL']['n'], invalid=v['ALL']['invalid']) for arm, v in p['families'].items()}
            out[name]['sql_all_agree'] = p.get('sql_shared_parse', {}).get('all_agree')
            out[name]['unexplained_isolation_changes'] = p.get('isolation', {}).get('unexplained')
    return out


def full(study, dest):
    """Complete replay: primary (Qwen) panels, old A10 cohort, workload cost, secondary (Gemma), CPU references."""
    study = Path(study)
    import os
    data = study / 'data'
    pod = Path(os.environ.get('NSR1_POD_OUT', str(study / 'runs' / 'pod_out')))  # override: run on an extracted package
    vocab = [r['relation'] for r in json.loads((data / 'visible' / 'PUBLIC_SCHEMA.json').read_text(encoding='utf-8'))]
    names = ['core', 'factorial', 'temporal', 'graph', 'public', 'boundary']
    res = dict(schema='NSR1_ANALYSIS_V1', bootstrap=dict(draws=DRAWS, seed=SEED),
               primary=analyze_model(data, pod / 'final' / 'qwen', {n: n for n in names}, 'Qwen/Qwen2.5-7B-Instruct', public_vocab=vocab))
    a10_src = ROOT_A10(data)
    a10_rows = pod / 'final' / 'qwen' / 'a10' / 'rows.jsonl.gz'
    if a10_rows.exists():
        scored = score.score_panel(data, pod / 'final' / 'qwen', 'a10')
        res['a10'] = a10_analysis(a10_src, scored, study / 'governing_inputs' / 'audit' / 'A10_RECOMPOSITION_TARGETS.csv')
        for arm in res['a10']:
            res['a10'][arm].pop('targets')
        res['a10_families'] = fam_table(scored)
        sq = {}
        for arm in ('A10_TEXT_SOURCE', 'A10_STRUCTURED_SOURCE'):
            sq[arm] = sqlcheck.check(data, pod / 'final' / 'qwen', 'a10', [r for r in scored if r['arm'] == arm], arm=arm)[0]
        res['a10_sql_shared_parse'] = sq
        intended = {}
        for r in ev_a10(data):
            intended[r['question_id']] = r
        res['a10_localization'] = {arm: localize(data, 'a10', scored, arm, intended) for arm in ('A10_TEXT_SOURCE', 'A10_STRUCTURED_SOURCE')}
        intake = sqlcheck.load_intake(pod / 'final' / 'qwen', 'a10', 'A10_TEXT_SOURCE')
        res['a10_intake'] = {k: dict(n=len(v), errors=sum(x['error'] is not None for x in v.values()), used_repair=sum(x['used_repair'] for x in v.values()))
                             for k, v in intake.items()}
    res['timing'] = timing_analysis(data, pod / 'final' / 'qwen_timing')
    neg = pod / 'final' / 'qwen_negative_nl' / 'rows.jsonl.gz'
    if neg.exists():  # language-path negative contract cases: separate denominator, never pooled with in-scope efficacy
        exp = {e['question_id']: e for e in json.loads((data / 'evaluator' / 'negative_nl' / 'expected.json').read_text())}
        rows = [json.loads(l) for l in gzip.open(neg, 'rt', encoding='utf-8')]
        res['negative_language'] = dict(
            negative_questions=sum(e['expect'] == 'REJECT' for e in exp.values()),
            rejected_with_explicit_error=sum(exp[r['question_id']]['expect'] == 'REJECT' and r['answer'] is None and bool(r['error']) for r in rows),
            answered_instead_of_rejected=sum(exp[r['question_id']]['expect'] == 'REJECT' and r['answer'] is not None for r in rows),
            controls=sum(e['expect'] == 'ANSWER' for e in exp.values()),
            controls_answered_correctly=sum(exp[r['question_id']]['expect'] == 'ANSWER' and r['answer'] == exp[r['question_id']]['gold'] for r in rows),
            rows=[dict(question_id=r['question_id'], answer=r['answer'], error=r['error'], program=r.get('program')) for r in rows])
    gem = pod / 'final' / 'gemma'
    if gem.exists():
        res['secondary'] = analyze_model(data, gem, {'core': 'core', 'graph': 'graph'}, 'google/gemma-2-9b-it', public_vocab=vocab)
        res['secondary_summary'] = secondary_summary(res['secondary'])
    Path(dest).write_text(json.dumps(res, indent=1, default=str))
    return res


def ev_a10(data):
    """Intended programs of the old A10 questions, built evaluator-side from the exposed query specs."""
    out = []
    for line in gzip.decompress((ROOT_A10(data) / 'designs.jsonl.gz').read_bytes()).splitlines():
        d = json.loads(line)
        t = d['root']['terminal']
        for q in d['questions']:
            s = q['spec']
            rd = lambda a: dict(op='READ', entity=t['actors'][a], field='stance', topic=t['projects'][s['p']], snapshot='CURRENT')
            fav = lambda a: dict(op='TEST', arg=rd(a), value='favor')
            prog = {'direct': lambda: fav(s['a']), 'opposes': lambda: dict(op='TEST', arg=rd(s['a']), value='oppose'),
                    'same': lambda: dict(op='SAME', left=rd(s['a']), right=rd(s['b'])),
                    'both': lambda: dict(op='BOTH', left=fav(s['a']), right=fav(s['b'])),
                    'either': lambda: dict(op='EITHER', left=fav(s['a']), right=fav(s['b']))}[s['kind']]()
            out.append(dict(question_id=q['query_id'], program=prog))
    return out


def ROOT_A10(data):
    import os
    return Path(os.environ.get('NSR1_A10_SCORES', str(Path(data).parent / 'inputs_a10_scores')))


if __name__ == '__main__':
    if sys.argv[1] == 'full':
        r = full(sys.argv[2], sys.argv[3])
        for n, p in r['primary']['panels'].items():
            if p['status'] == 'EXECUTED':
                print(n, {a: round(v['ALL']['accuracy'], 4) for a, v in p['families'].items()}, 'sql_all_agree', p.get('sql_shared_parse', {}).get('all_agree'))
    else:
        data, out_dir, model, dest = sys.argv[1:5]
        prefix = sys.argv[5] if len(sys.argv) > 5 else ''
        names = ['core', 'factorial', 'temporal', 'graph', 'public', 'boundary']
        vocab = [r['relation'] for r in json.loads((Path(data) / 'visible' / 'PUBLIC_SCHEMA.json').read_text(encoding='utf-8'))]
        r = analyze_model(data, out_dir, {n: prefix + n for n in names}, model, public_vocab=vocab)
        Path(dest).write_text(json.dumps(r, indent=1, default=str))
        for n, p in r['panels'].items():
            if p['status'] == 'EXECUTED':
                print(n, {a: round(v['ALL']['accuracy'], 4) for a, v in p['families'].items()}, 'sql_all_agree', p.get('sql_shared_parse', {}).get('all_agree'))
