"""Build the exposed-data panels (public MQuAKE-derived bridge, old A10 cohort) in the common unit format.

Visible tree: literal records / SET lines / question text (plus, for A10, the original typed commands and the
structured source scene, both legitimate typed inputs). Evaluator tree: typed states, intended programs, gold
labels and aliases. Nothing from `terminal`, gold chains or query specs is written to the visible tree.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


# ----------------------------- public bridge -----------------------------
def template_regexes(vocab):
    out = []
    for r in vocab['relations']:
        rx = re.escape(r['record_template']).replace(r'\{S\}', '(?P<S>.+)').replace(r'\{O\}', '(?P<O>.+)')
        out.append((r['relation_label'], re.compile('^' + rx + '$')))
    return out


def typed_public(line, regs):
    hits = [(lab, m.group('S'), m.group('O')) for lab, rx in regs for m in [rx.match(line)] if m]
    if len(hits) != 1:
        raise ValueError(f'{len(hits)} template matches: {line!r}')
    lab, s, o = hits[0]
    return dict(entity=s, field=lab, topic=None, value=o)


def build_public(study, public_order_seed):
    cohort = json.loads((study / 'data_public' / 'public_cohort_all.json').read_text(encoding='utf-8'))
    vocab = json.loads((study / 'data_public' / 'public_relation_vocabulary.json').read_text(encoding='utf-8'))
    regs = template_regexes(vocab)
    # Selection rule (protocol 7E): ascending SHA256(str(config.public_order) || str(case_id)); first 16 are the
    # final panel; ranks 17-18 are development cases (never in the final panel).
    ranked = sorted(cohort, key=lambda c: hashlib.sha256((str(public_order_seed) + str(c['case_id'])).encode()).hexdigest())
    sel = dict(rule='ascending sha256(str(610151) + str(case_id)).hexdigest(); ranks 1-16 final, 17-18 development',
               final=[c['case_id'] for c in ranked[:16]], development=[c['case_id'] for c in ranked[16:18]],
               all_ranked=[[c['case_id'], hashlib.sha256((str(public_order_seed) + str(c['case_id'])).encode()).hexdigest()] for c in ranked])
    (study / 'data' / 'evaluator').mkdir(parents=True, exist_ok=True)
    (study / 'data' / 'PUBLIC_SELECTION.json').write_text(json.dumps(sel, indent=1), encoding='utf-8')
    for name, cases in (('public', ranked[:16]), ('dev_public', ranked[16:18])):
        vis, ev, gold = [], [], []
        for c in cases:
            rid = f'NSR1-PUB-{c["case_id"]}'
            eo = c['evaluator_only']
            chain = eo['chain']
            prog = dict(op='FOLLOW', entity=chain['subject']['label'], path=chain['relation_labels_verbatim'], snapshot='CURRENT')
            aliases = sorted(set(eo['answer_aliases'][eo['gold_terminal']['entity_id']]) | {eo['gold_terminal']['label']})
            hs, ths = [], []
            for h in c['histories']:
                recs = [dict(record_id=str(i + 1), text=t) for i, t in enumerate(h['visible_records'])]
                ups = [dict(update_id=str(i + 1), text=t) for i, t in enumerate(h['visible_updates'])]
                hs.append(dict(history_id=h['history_id'], records=recs, updates=ups))
                ths.append(dict(history_id=h['history_id'], initial=[typed_public(t, regs) for t in h['visible_records']],
                                events=[dict(writes=[typed_public(t[4:], regs)]) for t in h['visible_updates']]))
                assert all(t.startswith('SET ') for t in h['visible_updates'])
            qs = [dict(question_id=f'{rid}-{q["question_id"]}', text=q['text']) for q in c['questions']]
            vis.append(dict(root_id=rid, case_id=c['case_id'], histories=hs, questions=qs))
            ev.append(dict(root_id=rid, case_id=c['case_id'], hops=chain['hops'], typed=ths,
                           questions=[dict(question_id=q['question_id'], family='public', kind=f'hop{chain["hops"]}',
                                           program=prog, aliases=aliases, mapping=None) for q in qs],
                           gold_label=eo['gold_terminal']['label'], obsolete_label=eo['obsolete_answer']['label']))
        w(study / 'data' / 'visible' / name / 'roots.jsonl', vis)
        w(study / 'data' / 'evaluator' / name / 'roots.jsonl', ev)
    (study / 'data' / 'visible' / 'PUBLIC_SCHEMA.json').write_text(json.dumps(
        [dict(relation=r['relation_label'], record_wording=r['record_template']) for r in vocab['relations']], indent=1), encoding='utf-8')
    return sel


# ----------------------------- old A10 cohort -----------------------------
def render_scene(s):
    """The original renderer (reproducibility/learned_editors/canonical/v2_protocol.py: render, order='original',
    variant=0), reimplemented verbatim; every output is checked against the retained source hash."""
    pairs = [(a, p) for a in s['layout'] for p in range(len(s['projects']))]
    lines = [f'This is a fictional {s["setting"]}. Use only the records below.',
             'Each named member has independent views. Updating one record leaves the other records unchanged.']
    for a, p in pairs:
        lines.append(f'{s["actors"][a]} is ' + ('in favor of ' if s['values'][a][p] else 'opposed to ') + f'the {s["projects"][p]} proposal.')
    lines.append('End of records.\n')
    return '\n'.join(lines)


def truth(values, spec):
    x = int(values[spec['a']][spec['p']])
    kind = spec['kind']
    if kind == 'direct':
        return x
    if kind == 'opposes':
        return 1 - x
    y = int(values[spec['b']][spec['p']])
    return {'same': 1 - (x ^ y), 'both': x * y, 'either': min(1, x + y)}[kind]


def build_a10(study, source_root):
    path = Path(source_root) / 'reproducibility' / 'question_blind_repair' / 'scores' / 'designs.jsonl.gz'
    vis, ev, gold, checked = [], [], [], 0
    sv = {1: 'favor', 0: 'oppose'}
    for line in gzip.decompress(path.read_bytes()).splitlines():
        d = json.loads(line)
        root = d['root']
        hs = []
        for o, s in enumerate(root['sources']):
            text = render_scene(s)
            if hashlib.sha256(text.encode()).hexdigest() != root['source_hashes'][o]:
                raise ValueError('source text does not reproduce the retained hash')
            checked += 1
            lines = text.split('\n')[:-1]  # the trailing newline after 'End of records.'
            recs = [dict(record_id=f'{i:02d}', text=t) for i, t in enumerate(lines)]
            cmds = [dict(writes=[dict(entity=s['actors'][c['actor']], field='stance', topic=s['projects'][c['project']],
                                      value=sv[c['value']])]) for c in root['commands']]
            init = [dict(entity=s['actors'][a], field='stance', topic=s['projects'][p], value=sv[s['values'][a][p]])
                    for a in s['layout'] for p in range(len(s['projects']))]
            hs.append(dict(history_id=f'o{o}', records=recs, updates=[], typed_commands=cmds, typed_initial=init,
                           source_sha256=root['source_hashes'][o]))
        vis.append(dict(root_id=root['root_id'], histories=hs,
                        questions=[dict(question_id=q['query_id'], text=q['text']) for q in d['questions']]))
        for q in d['questions']:
            g = truth(root['terminal']['values'], q['spec'])
            kind = q['spec']['kind']
            for o in range(8):
                gold.append(dict(root_id=root['root_id'], history_id=f'o{o}', question_id=q['query_id'], kind=kind,
                                 family='joint' if kind in ('same', 'both', 'either') else kind,
                                 semantic=bool(g), gold=q['labels'][g]))
    w(study / 'data' / 'visible' / 'a10' / 'roots.jsonl', vis)
    w(study / 'data' / 'evaluator' / 'a10' / 'gold.jsonl', gold)
    return dict(designs=len(vis), source_texts_hash_verified=checked, gold_rows=len(gold))


if __name__ == '__main__':
    study = Path(sys.argv[1])
    cfg = json.loads((study / 'governing_inputs' / 'NSR1_CONFIG.json').read_text())
    print(json.dumps(dict(public=build_public(study, cfg['seeds']['public_order'])['final'],
                          a10=build_a10(study, sys.argv[2]))))
