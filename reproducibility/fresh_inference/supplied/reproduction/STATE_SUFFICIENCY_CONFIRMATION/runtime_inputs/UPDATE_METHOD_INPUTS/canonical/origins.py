from contextlib import contextmanager
from dataclasses import replace
from typing import Any
import hashlib, json
import numpy as np
SEED = 260912601
BOOT_SEED = 260913902
READERS = ("R0", "R1", "R2")
ORIGIN_MASKS = ((0,0,0),(1,0,0),(0,1,1),(1,1,1))

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def bind_seed(old):
    prior = old.SEED
    old.SEED = SEED
    try:
        yield
    finally:
        old.SEED = prior


def source_digest(old, scene) -> str:
    return hashlib.sha256(old.render(scene)[0].encode()).hexdigest()


def commands(old, terminal):
    """All three requested final assignments execute, including no-ops.

    Do NOT call the historical `programs(source)['ABC']` here: that flips source
    values and would produce different final worlds in the four origins.
    """
    a, p, n = terminal.target_actor, terminal.target_project, len(terminal.actors)
    return [old.Edit((a + j) % n, p, terminal.values[(a + j) % n][p]) for j in range(3)]


def origins(old, terminal):
    edits = commands(old, terminal)
    out = []
    for i, mask in enumerate(ORIGIN_MASKS):
        matrix = [list(row) for row in terminal.values]
        for e, toggle in zip(edits, mask):
            matrix[e.actor][e.project] ^= toggle
        source = replace(terminal, values=tuple(map(tuple, matrix)),
                         scene_id=terminal.scene_id + f'-origin{i}')
        reached = old.apply_program(source, edits)
        if reached.values != terminal.values:
            raise ValueError('Source does not reach declared terminal world')
        if old.render(reached, 'early')[0] != old.render(terminal, 'early')[0]:
            raise ValueError('Terminal rendering depends on origin')
        out.append(source)
    return out


def truth(values, spec):
    """Independent integer truth implementation, not the production evaluator."""
    x = int(values[spec['a']][spec['p']])
    kind = spec['kind']
    if kind == 'direct': return x
    if kind == 'opposes': return 1 - x
    y = int(values[spec['b']][spec['p']])
    if kind == 'same': return 1 - (x ^ y)
    if kind == 'both': return x * y
    if kind == 'either': return min(1, x + y)
    raise ValueError(kind)


def questions(C, terminal, source, reader, draw):
    """34 questions per origin/reader: original 24, plus ten marked additions.

    Query text and answer mappings are determined by the terminal root ID and
    query specification, not source values/origin ID. Actual terminal bits never
    enter the wording. The existing three V5 readers remain unchanged.
    """
    if reader not in READERS or draw not in (0, 1):
        raise ValueError('Undeclared reader/draw')
    old = C.Q.OLD
    with bind_seed(old):
        rows = [dict(q) for q in old.questions(terminal, terminal, True, draw)]
    for q in rows:
        q['additional'] = False
    a, p, n = terminal.target_actor, terminal.target_project, len(terminal.actors)
    b, c, d = (a + 1) % n, (a + 2) % n, (a + 3) % n
    extras = [('direct', c, a), ('same', b, c), ('both', b, c),
              ('either', b, c), ('direct', d, a)]
    for j, (kind, aa, bb) in enumerate(extras):
        an, bn, project = terminal.actors[aa], terminal.actors[bb], terminal.projects[p]
        text = {'direct': f'Does {an} favor the {project} proposal?',
                'same': f'Do {an} and {bn} take the same side on {project}?',
                'both': f'Is the {project} proposal supported by both {an} and {bn}?',
                'either': f'Does at least one of {an} and {bn} support {project}?'}[kind]
        swap = int(stable_hash((SEED, terminal.scene_id, draw, j))[-1], 16) % 2
        labels = ['No', 'Yes'] if draw == 0 else (['A', 'B'] if swap else ['B', 'A'])
        if draw:
            text += f'\nUse {labels[1]} for yes and {labels[0]} for no.'
        text += '\nAnswer with exactly one of ' + ', '.join(labels) + '.'
        rows.append({'query_id': f'{terminal.scene_id}-d{draw}-extra{j}',
                     'scene_id': terminal.scene_id, 'family': kind, 'text': text,
                     'labels': labels, 'additional': True,
                     'spec': {'kind': kind, 'a': aa, 'b': bb, 'p': p}})
    for q in rows:
        spec = q['spec']
        q['source_gold'] = truth(source.values, spec)
        q['gold'] = truth(terminal.values, spec)
        q['changed'] = q['source_gold'] != q['gold']
        q['text'] = C.apply_reader(q['text'], q['family'], terminal.actors[spec['a']],
                                   terminal.actors[spec['b']], terminal.projects[spec['p']], reader)
        q.update(reader=reader, draw=draw, root_id=terminal.scene_id,
                 origin_id=source.scene_id, program_id='SET_TERMINAL_ABC')
    if len(rows) != 17:
        raise ValueError('Query cardinality drift')
    return rows


def origin_metrics(predictions, gold, valid=None):
    """Per-root metrics for [4 origins, Q questions] semantic predictions.

    No averaging across training seeds occurs here. The caller reports each seed
    and only then forms paired within-root seed averages for comparisons.
    Invalid rows count as incorrect and never count as stable successes.
    """
    p = np.asarray(predictions)
    y = np.asarray(gold)
    if p.ndim != 2 or p.shape[0] != 4 or y.shape != (p.shape[1],):
        raise ValueError('Expected [4,Q] predictions and [Q] gold')
    if not np.isin(p, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Semantic bits required; invalids belong in valid mask')
    v = np.ones(p.shape, bool) if valid is None else np.asarray(valid, bool)
    if v.shape != p.shape:
        raise ValueError('Validity shape mismatch')
    correct = (p == y[None, :]) & v
    unanimous = (p == p[0:1, :]).all(axis=0) & v.all(axis=0)
    return {'mean_accuracy': float(correct.mean()),
            'all_origin_question_correct': float(correct.all(axis=0).mean()),
            'all_origin_bundle_correct': float(correct.all()),
            'origin_disagreement': float((~unanimous).mean()),
            'all_origins_same_but_wrong': float((unanimous & ~correct[0]).mean()),
            'questions': int(p.shape[1]), 'origins': 4}


def paired_interval(root_differences, *, level=.9875, draws=10000, seed=BOOT_SEED):
    x = np.asarray(root_differences, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite paired ROOT differences required')
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(len(x), size=(min(500, draws-i), len(x)))].mean(1)
                            for i in range(0, draws, 500)])
    lo, hi = np.quantile(means, [(1-level)/2, (1+level)/2])
    return {'effect': float(x.mean()), 'lower': float(lo), 'upper': float(hi),
            'level': level, 'roots': len(x), 'draws': draws,
            'zero_crossing_is_equivalence': False}

