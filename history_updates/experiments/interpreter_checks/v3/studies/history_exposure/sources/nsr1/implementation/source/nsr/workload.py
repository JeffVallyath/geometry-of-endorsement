"""NSR1 workload cost panel (protocol 7G): complete timed workflows, one workload/arm at a time, alone on the GPU.

Arms: ISOLATED_SYMBOLIC (full pipeline), SQL_SHARED_PARSE (the same parsed inputs executed by ordinary SQLite; charged
the shared parser cost), FLAT_SYMBOLIC, CURRENT_NEURAL. Every arm's cost includes cold intake, updates, neural
parsing/repair calls, dependency invalidation, execution, rendering and archive storage. Three paired repetitions
(input-keyed program cache enabled, empty at the start of each workload/arm) plus one cache-skipping cold replicate.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

from . import core, pipeline as P, prompts, render

T = time.perf_counter


def isolated(gen, root, want_tables=False):
    """Full ISOLATED_SYMBOLIC workflow. Returns (answers, timing, parsed) ."""
    h = root['histories'][0]
    tm = dict(intake_parse=0.0, state_build=0.0, update_parse=0.0, apply=0.0, compile=0.0, execute=0.0, render=0.0)
    t = T()
    rec = P.gen_validated(gen, [(prompts.record_user(r['text']), (r['text'],)) for r in h['records']],
                          prompts.record_system(), P.CAPS['record'], P.validate_record, 'work:record')
    tm['intake_parse'] = T() - t
    t = T()
    st = P.IsolatedState([(r['record_id'], x) for r, x in zip(h['records'], rec)])
    tm['state_build'] = T() - t
    answers, events, programs, tables = {}, [], {}, {}
    for k, u in enumerate(h['updates'], 1):
        t = T()
        res = P.gen_validated(gen, [(prompts.update_user(u['text']), (u['text'],))], prompts.update_system(),
                              P.CAPS['update'], P.validate_update, 'work:update')[0]
        tm['update_parse'] += T() - t
        t = T()
        st.apply(k, res=res)
        tm['apply'] += T() - t
        events.append(res['value'] if res['error'] is None else None)
        qs = [q for q in root['questions'] if q['after_updates'] == k]
        t = T()
        cmp_ = P.gen_validated(gen, [(prompts.compiler_user(q['text']), (q['text'],)) for q in qs], prompts.compiler_system(),
                               P.CAPS['query'], P.validate_query, 'work:query')
        tm['compile'] += T() - t
        if want_tables and st.store is not None and st.failed_from is None:
            tables[k] = (P.table(st.store.current.values), P.table(st.store.snapshots[0].values))
        for q, c in zip(qs, cmp_):
            t = T()
            a = st.answer(c, q['text'])
            tm['execute'] += T() - t
            answers[q['question_id']] = dict(answer=a['answer'], error=a['error'])
            programs[q['question_id']] = c['value'][1] if c['error'] is None and c['value'][0] == 'PROGRAM' else None
    archive = sum(len(json.dumps(sorted([list(k_), str(v)] for k_, v in s.values.items()))) for s in st.store.snapshots) if st.store else 0
    tm['archive_bytes'] = archive
    tm['view_stats'] = dict(st.store.stats) if st.store else {}
    parsed = dict(initial=[w for x in rec if x['error'] is None for w in x['value']], events=events, programs=programs)
    return answers, tm, parsed, tables


def sql_shared(parsed, root, sqlmod):
    """Ordinary SQLite executing ISOLATED_SYMBOLIC's exact parsed state and programs."""
    tm = dict(load=0.0, apply=0.0, execute=0.0, render=0.0)
    t = T()
    try:
        st = sqlmod.SQLStore(parsed['initial'])
    except sqlmod.SQLRefError as exc:
        return {q['question_id']: dict(answer=None, error='STATE_' + exc.code) for q in root['questions']}, tm
    tm['load'] = T() - t
    answers, failed = {}, False
    for k, ev in enumerate(parsed['events'], 1):
        t = T()
        if ev is None or failed or st.apply(ev)['status'] != 'ACCEPTED':
            failed = True
        tm['apply'] += T() - t
        for q in [q for q in root['questions'] if q['after_updates'] == k]:
            prog = parsed['programs'][q['question_id']]
            if prog is None:
                answers[q['question_id']] = dict(answer=None, error='NO_PROGRAM')
                continue
            if failed and ('CURRENT' in json.dumps(prog)):
                answers[q['question_id']] = dict(answer=None, error='INTAKE_INCOMPLETE')
                continue
            t = T()
            res = st.execute(prog)
            tm['execute'] += T() - t
            t = T()
            ans, err = render.render(res, q['text'])
            tm['render'] += T() - t
            answers[q['question_id']] = dict(answer=ans, error=err)
    return answers, tm


def flat(gen, root):
    h = root['histories'][0]
    tm = dict(translate=0.0, solve=0.0)
    answers = {}
    recs = [r['text'] for r in h['records']]
    for k in range(1, len(h['updates']) + 1):
        ups = [u['text'] for u in h['updates'][:k]]
        ids = {f'Record {r["record_id"]}' for r in h['records']} | {f'Correction {u["update_id"]}' for u in h['updates'][:k]}
        qs = [q for q in root['questions'] if q['after_updates'] == k]
        t = T()
        res = P.gen_validated(gen, [(prompts.flat_user(recs, ups, q['text']), (q['text'], ids)) for q in qs],
                              prompts.flat_system(), P.CAPS['flat'], P.validate_flat, 'work:flat')
        tm['translate'] += T() - t
        t = T()
        for q, r in zip(qs, res):
            if r['error'] is not None:
                answers[q['question_id']] = dict(answer=None, error='TRANSLATE_' + r['error'])
                continue
            kind, prog, facts = r['value']
            if kind == 'UNSUPPORTED':
                answers[q['question_id']] = dict(answer=None, error='ABSTAIN_UNSUPPORTED')
                continue
            ans, err = render.render(core.execute(prog, P.FlatStore(facts)), q['text'])
            answers[q['question_id']] = dict(answer=ans, error=err)
        tm['solve'] += T() - t
    return answers, tm


def current(gen, root):
    """CURRENT_NEURAL end to end: the same candidate intake/update parsing and state maintenance, then neural
    answering from the canonical tables."""
    h = root['histories'][0]
    tm = dict(intake_parse=0.0, update_parse=0.0, apply=0.0, neural_read=0.0)
    t = T()
    rec = P.gen_validated(gen, [(prompts.record_user(r['text']), (r['text'],)) for r in h['records']],
                          prompts.record_system(), P.CAPS['record'], P.validate_record, 'work:record')
    st = P.IsolatedState([(r['record_id'], x) for r, x in zip(h['records'], rec)])
    tm['intake_parse'] = T() - t
    answers = {}
    for k, u in enumerate(h['updates'], 1):
        t = T()
        res = P.gen_validated(gen, [(prompts.update_user(u['text']), (u['text'],))], prompts.update_system(),
                              P.CAPS['update'], P.validate_update, 'work:update')[0]
        tm['update_parse'] += T() - t
        t = T()
        st.apply(k, res=res)
        tm['apply'] += T() - t
        qs = [q for q in root['questions'] if q['after_updates'] == k]
        if st.store is None or st.failed_from is not None:
            for q in qs:
                answers[q['question_id']] = dict(answer=None, error='STATE_UNAVAILABLE')
            continue
        cur, orig = P.table(st.store.current.values), P.table(st.store.snapshots[0].values)
        t = T()
        outs = gen.run([dict(messages=P.msgs(prompts.current_system(), prompts.current_user(cur, orig, q['text'])),
                             max_new=P.CAPS['answer'], tag='work:current') for q in qs])
        tm['neural_read'] += T() - t
        for q, o in zip(qs, outs):
            ans, err = P.native_answer(o, P.labels_of(q['text']))
            answers[q['question_id']] = dict(answer=ans, error=err, raw=o['text'])
    return answers, tm


def run(backend, data, out_dir, panel, sqlref_dir, reps=3, seed=610171):
    sys.path.insert(0, sqlref_dir)
    import sqlref as sqlmod
    sqlmod.SQLStore.evaluate_all = False
    out_dir = Path(out_dir) / panel
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [P.sanitize_root(json.loads(l)) for l in open(Path(data) / 'visible' / panel / 'roots.jsonl', encoding='utf-8')]
    meta = {json.loads(l)['root_id']: json.loads(l).get('N') for l in open(Path(data) / 'visible' / panel / 'roots.jsonl', encoding='utf-8')}
    prompts.use_fixed_schema()
    # engine warm-up on the fixed system prompts only (no workload content)
    warm = P.CachedGen(backend, enabled=False)
    warm.run([dict(messages=P.msgs(s, 'Question: ?'), max_new=4, tag='work:warm') for s in
              (prompts.record_system(), prompts.update_system(), prompts.compiler_system(), prompts.flat_system(), prompts.current_system())])
    rng = random.Random(seed)
    log = gzip.open(out_dir / 'generations.jsonl.gz', 'at', encoding='utf-8')
    rows = []
    schedule = [('rep', r) for r in range(reps)] + [('cold', 0)]
    for mode, rep in schedule:
        for root in roots:
            arms = ['ISOLATED_SYMBOLIC', 'FLAT_SYMBOLIC', 'CURRENT_NEURAL']
            rng.shuffle(arms)
            for arm in arms:
                gen = P.CachedGen(backend, log=_Log(log, mode, rep, root['root_id'], arm) if (mode, rep) == ('rep', 0) else None,
                                  enabled=(mode != 'cold'))
                t0 = T()
                if arm == 'ISOLATED_SYMBOLIC':
                    answers, tm, parsed, _ = isolated(gen, root)
                elif arm == 'FLAT_SYMBOLIC':
                    answers, tm = flat(gen, root)
                else:
                    answers, tm = current(gen, root)
                total = T() - t0
                rows.append(dict(panel=panel, mode=mode, rep=rep, root_id=root['root_id'], N=meta[root['root_id']], arm=arm,
                                 total_seconds=total, stages=tm, gen=dict(gen.stats), answers=answers))
                if arm == 'ISOLATED_SYMBOLIC':
                    t0 = T()
                    a2, tm2 = sql_shared(parsed, root, sqlmod)
                    sql_exec = T() - t0
                    shared = tm['intake_parse'] + tm['update_parse'] + tm['compile']
                    rows.append(dict(panel=panel, mode=mode, rep=rep, root_id=root['root_id'], N=meta[root['root_id']],
                                     arm='SQL_SHARED_PARSE', total_seconds=shared + sql_exec, stages=dict(tm2, shared_parse=shared, sql_only=sql_exec),
                                     gen=dict(gen.stats), answers=a2,
                                     agrees_with_isolated=all(a2[q]['answer'] == answers[q]['answer'] for q in answers)))
            print(json.dumps(dict(mode=mode, rep=rep, root=root['root_id'], N=meta[root['root_id']],
                                  seconds={r['arm']: round(r['total_seconds'], 2) for r in rows[-4:]})), flush=True)
    log.close()
    with gzip.open(out_dir / 'timing_rows.jsonl.gz', 'wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    (out_dir / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


class _Log:
    def __init__(self, f, mode, rep, root, arm):
        self.f, self.meta = f, dict(mode=mode, rep=rep, root=root, arm=arm)

    def write(self, line):
        o = json.loads(line)
        o.pop('messages', None)
        self.f.write(json.dumps(dict(o, **self.meta)) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['vllm', 'grammar'], required=True)
    ap.add_argument('--model-path')
    ap.add_argument('--model-id', default='qwen')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--panel', default='workload')
    ap.add_argument('--sqlref', required=True)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    if a.backend == 'vllm':
        from .llm_vllm import VLLMGen
        backend = VLLMGen(a.model_path, a.model_id, max_model_len=20000)
    else:
        from .grammar_ref import GrammarGen
        backend = GrammarGen()
    run(backend, a.data, a.out, a.panel, a.sqlref, reps=a.reps)
