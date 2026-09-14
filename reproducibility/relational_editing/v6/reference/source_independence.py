"""Model-free V6 source-independence protocol and conservative metrics.

This module does not implement a model runner and never loads a backbone. The
scientific implementation inherits the prepared fixed-readout model routines.
Input objects below belong to the evaluator. Only serialized source text,
validated token spans, and requested bits may reach the existing writer API.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

SEED = 260912601
BOOT_SEED = 260912602
READERS = ('R0', 'R1', 'R2')
# These describe differences from the final values, NOT commands to the model.
# Relative to a terminal world, the four sources require 0, 1, 2, or 3 flips.
ORIGIN_MASKS = ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
N_GEMMA_ROOTS = 64
N_QWEN_ROOTS = 32


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


def make_population(C):
    """64 independent terminal roots; 32-root Qwen subset.

    Every actor-size x project-size x 3-bit terminal combination is present twice
    for Gemma and once for Qwen. Names/layout/untouched records vary across roots.
    Source identity is deduplicated against prior V2--V5 pools and across roots.
    The terminal world equals origin 0 within a root by design; not extra data.
    """
    old = C.Q.OLD
    historical = set(C.earlier_sources())
    prepared, _ = C.make_population()
    historical.update(source_digest(old, s) for s in prepared['gemma'])
    with bind_seed(old):
        candidates = old.make_scenes('FINAL', 1024)
    roots = []
    used = set(historical)
    rejections = []
    for size_index, (members, projects) in enumerate(((4, 2), (4, 3), (6, 2), (6, 3))):
        available = [s for s in candidates if len(s.actors) == members and len(s.projects) == projects]
        for bits_code in range(8):
            bits = tuple((bits_code >> j) & 1 for j in range(3))
            for replica in range(2):
                cell = (members, projects, bits_code, replica)
                ranked = sorted(available, key=lambda s: stable_hash((SEED, cell, s.scene_id)))
                selected = None
                for base in ranked:
                    values = [list(row) for row in base.values]
                    for j, bit in enumerate(bits):
                        values[(base.target_actor + j) % members][base.target_project] = bit
                    root_id = f'FINAL-RSI6-{len(roots):04d}'
                    t = replace(base, scene_id=root_id, values=tuple(map(tuple, values)))
                    branches = origins(old, t)
                    hashes = [source_digest(old, x) for x in branches]
                    if len(set(hashes)) != 4:
                        raise ValueError('Origins are not distinct')
                    if any(h in used for h in hashes):
                        rejections.append({'cell': cell, 'candidate': base.scene_id})
                        continue
                    selected = {'root_id': root_id, 'terminal': asdict(t),
                                'sources': [asdict(x) for x in branches],
                                'commands': [asdict(e) for e in commands(old, t)],
                                'origin_masks': [list(x) for x in ORIGIN_MASKS],
                                'cell': list(cell), 'source_hashes': hashes,
                                'qwen': replica == 0}
                    used.update(hashes)
                    break
                if selected is None:
                    raise ValueError('Fixed candidate population exhausted; no silent resizing')
                roots.append(selected)
    if len(roots) != N_GEMMA_ROOTS or sum(r['qwen'] for r in roots) != N_QWEN_ROOTS:
        raise ValueError('Population size drift')
    return roots, {'seed': SEED, 'historical_source_count': len(historical),
                   'roots': len(roots), 'gemma_sources': 4 * len(roots),
                   'qwen_roots': sum(r['qwen'] for r in roots),
                   'independence_unit': 'terminal root, not source or query or seed',
                   'duplicate_candidate_rejections': rejections,
                   'outcome_access': False}


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


def export(C, directory):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    roots, audit = make_population(C)
    for actor in ('gemma', 'qwen'):
        selected = roots if actor == 'gemma' else [r for r in roots if r['qwen']]
        p = destination / f'{actor}_source_roots.jsonl'
        p.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in selected))
    (destination / 'SOURCE_POPULATION_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit
