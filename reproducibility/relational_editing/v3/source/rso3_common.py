# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""RELATIONAL_SET_OPERATIONS_V3 (RSO3) fixed contract: bundle custody, V3 scenes/programs/questions, prompts (exact V2 C1/C1S/T4
demonstrations), records and metrics.  No model imports.

Reuses the SRS2 (V2) contract module read-only for prompts, demonstrations, metrics, selection rules, bootstraps and model identities.
The V3 bundle's `reference/protocol.py` (new seed, twelve declared edit programs, coverage questions) is the semantic authority for the
V3 scenes; the compiler never receives a question, label mapping, gold, affected flag or target world.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/relational_set_operations_v3'
BUNDLE = Path(os.environ.get('RSO3_SUPPLIED', str(REPORT/'supplied')))
DATA = BUNDLE/'data'
FACTORS = BUNDLE/'provenance/factors'
ANCHORS = Path(os.environ.get('RSO3_ANCHORS', str(REPORT/'anchors')))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import srs2_common as S                   # noqa: E402  V2 contract (prompts/demos/metrics/models), read-only
sys.path.insert(0, str(BUNDLE))
import reference.protocol as P3           # noqa: E402  V3 scenes / programs / coverage questions (new seed)
OLD = P3.old                              # the V3 copy of the V2 protocol (render/evaluate/questions) bound to the V3 seed

now = S.now; digest = S.digest; sha = S.sha; json_safe = S.json_safe; canonical = S.canonical; dump = S.dump; load = S.load; read_rows = S.read_rows; write_rows = S.write_rows
hash_rank = S.hash_rank; budget = S.budget
NL = chr(10)

BUNDLE_ZIP_SHA256 = None
CONTRACT_SHA256 = {}
DATA_SHA256 = {'fit_scenes.jsonl': 'a57b52c69eeddf31e103ddd73f08c413548538101e9caf6852b24fb72827954a',
               'cal_scenes.jsonl': 'd25637f0b38acb0228a793a174e91e4631d62d647d6868b060e6f97e83ba832d',
               'final_scenes.jsonl': '88709ec7ab0fb192d56503af0fb0a7ae408cc7eee45a777a556572cf6c26f76a',
               'workflow_scenes.jsonl': '200bbc5bc3f11348c1bb122d000d1404ace15f684a552b033e70a1a593d5a17d'}

GEMMA = dict(S.GEMMA); QWEN = dict(S.QWEN); MODELS = {'gemma': GEMMA, 'qwen': QWEN}
GEMMA_CAP = S.GEMMA_CAP
RANK = 16
SEEDS = (0, 1)
LRS = (0.0003, 0.001)
LR_EPOCHS = 2; MAX_EPOCHS = 4; PATIENCE = 2; GROUP = 4
STRENGTHS_AFFINE = (0.5, 0.75, 1.0, 1.25); STRENGTHS_SETTER = (1.0,)
BETA = 0.01                       # training regularizer beta * mean(||delta||^2 / cap_ref^2) for BOTH overwrite methods (not a runtime cap)
RIDGE = 1e-3
LEARNED_ARMS = ('INVARIANT_SET', 'FREE_OVERWRITE', 'CONTINUED_AFFINE')          # Gemma: CONTINUED_AFFINE = exact V2 SHARED_CLAUSE continued
QWEN_ARMS = ('INVARIANT_SET', 'FREE_OVERWRITE', 'AFFINE_CLAUSE')                # Qwen: independently trained affine clause comparator
KIND = {'INVARIANT_SET': 'invariant', 'FREE_OVERWRITE': 'free'}
FROZEN = {'FROZEN_V2_CLAUSE': dict(arm='SHARED_CLAUSE', strength=1.25, workspace='legacy', footprint='clause'),
          'FROZEN_V2_CLAUSE_FP32': dict(arm='SHARED_CLAUSE', strength=1.25, workspace='fp32', footprint='clause'),
          'FROZEN_V2_TAIL': dict(arm='SHARED_TAIL', strength=1.0, workspace='legacy', footprint='tail'),
          'CANONICAL_V2': dict(arm='SHARED_CLAUSE', strength=1.25, workspace='legacy', footprint='clause', canonical=True)}
PROCEDURES = {'gemma': ('C1',), 'qwen': ('T4', 'C1S', 'C1')}           # exact V2 C1 for Gemma; Qwen T4 -> C1S -> C1 under the new policy
QUAL_N = 96; QUAL_PARSE = .95; QUAL_DIRECT = .90; QUAL_EQUALITY_REPORTED = .85   # equality is measured and reported, never a veto
MAX_NEW_TOKENS = 32
FEASIBLE_HARM = .05; FEASIBLE_DIR = .80
BOOT_SEED = 260918; BOOT_DRAWS = 10000
PROGRAMS = ('single', 'noop', 'repeat2', 'repeat4', 'repeat8', 'restore', 'restore_after4', 'overwrite', 'AB', 'BA', 'ABC', 'CBA', 'AB_cross')
EQUAL_TERMINAL = (('single', 'repeat2'), ('single', 'repeat4'), ('single', 'repeat8'), ('single', 'overwrite'), ('AB', 'BA'), ('ABC', 'CBA'), ('noop', 'restore'), ('noop', 'restore_after4'))
REPEAT_RESTORE_PANEL = ('repeat2', 'repeat4', 'repeat8', 'restore', 'restore_after4')
WORKFLOW_PROGRAMS = ('single', 'repeat2', 'restore', 'AB')
ORDERS = ('early', 'late')
SIZES = {'gemma': {'FULL': dict(final=128, program=64, workflow=128), 'SMALL': dict(final=64, program=32, workflow=64), 'QUARTER': dict(final=32, program=16, workflow=32)},
         'qwen': {'FULL': dict(final=64, program=32, workflow=64), 'SMALL': dict(final=32, program=16, workflow=32), 'QUARTER': dict(final=16, program=8, workflow=16)}}
PANELS = dict(control=16, witness=4, generation=8)
BUDGET = {}  # Caller-supplied runtime policy for optional model execution; no private spending defaults.
INSTRUCTION = S.INSTRUCTION
CONTRAST_LEVEL = .9875
NONINFERIORITY = dict(margin=.05, level=.975)
CHUNK = 12                        # uniform batched clone-serving shape (identical to V2; validated against single asks in E0)

# ---------------------------------------------------------------- bundle custody
def verify_bundle():
    raise RuntimeError("Original mixed source bundle is not distributed. Public artifact identities are verified with python -m repro verify.")

def v2_factor_dir(arm, seed): return BUNDLE/'provenance/factors'/arm/f'seed{seed}'
def v2_strength(arm): return float(load(BUNDLE/'provenance/INDEX.json')['models'][arm]['strength'])

# ---------------------------------------------------------------- scenes (V3 data through the V3 protocol)
def scene_from_dict(d):
    d = dict(d); d['actors'] = tuple(d['actors']); d['projects'] = tuple(d['projects']); d['layout'] = tuple(d['layout']); d['values'] = tuple(tuple(x) for x in d['values'])
    return OLD.validate_scene(P3.Scene(**d))

def scenes(split):
    rows = read_rows(DATA/(split.lower()+'_scenes.jsonl')); out = [scene_from_dict(r) for r in rows]
    if any(s.split != split for s in out) or any('RSO3' not in s.scene_id for s in out): raise ValueError('split mismatch')
    return out

def cell(s): return S.cell(s)
def subset(scene_list, n, salt): return S.subset(scene_list, n, salt)
def noop_scenes(scene_list): return S.noop_scenes(scene_list)

# ---------------------------------------------------------------- programs (declared semantic SET assignments; evaluator side)
def programs(s):
    """The twelve supplied programs plus one cross-project two-edit variant (B on the next project), reported separately."""
    out = dict(P3.programs(s)); a = out['single'][0]; b = (a.actor+1) % len(s.actors); o = (a.project+1) % len(s.projects)
    out['AB_cross'] = [a, P3.Edit(b, o, 1-s.values[b][o])]
    if tuple(out) != PROGRAMS: raise ValueError('program set drift')
    return out

def final_world(s, program): return OLD.apply_program(s, programs(s)[program])

def program_queries(s, name, draw=0):
    """FINAL/WORKFLOW question bundle of a program: the original 24-question bundle (12 per draw) of the single edit's specs evaluated in the
    program's true final world, plus one additional direct question per touched address not already covered (flag `additional`)."""
    if name in P3.programs(s): out = P3.program_queries(s, name, draw)
    else:
        prog = programs(s)[name]; world = final_world(s, name); out = OLD.questions(s, world, True, draw)
        covered = {(q['spec']['a'], q['spec']['p']) for q in out if q['family'] == 'direct'}
        for e in prog:
            if (e.actor, e.project) in covered: continue
            spec = {'kind': 'direct', 'a': e.actor, 'b': (e.actor+1) % len(s.actors), 'p': e.project}
            labels = ['No', 'Yes'] if not draw % 2 else ['B', 'A']
            text = f'Does {s.actors[e.actor]} favor the {s.projects[e.project]} proposal?'
            if draw % 2: text += f'\nUse {labels[1]} for yes and {labels[0]} for no.'
            text += '\nAnswer with exactly one of '+', '.join(labels)+'.'
            v0 = OLD.evaluate(s, spec); v1 = OLD.evaluate(world, spec)
            out.append({'query_id': f'{s.scene_id}-d{draw}-extra{len(out)}', 'scene_id': s.scene_id, 'family': 'direct', 'text': text, 'labels': labels, 'spec': spec, 'source_gold': v0, 'gold': v1, 'changed': v0 != v1, 'additional': True})
            covered.add((e.actor, e.project))
        for q in out: q['program_id'] = name
    for q in out: q.setdefault('additional', False)
    return out

def questions(s, split, draw, program='single'):
    """FIT/CAL: the six direct/negation/equality/nonaddressed questions per draw in the program's final world (single flip or declared no-op).
    FINAL/WORKFLOW: program_queries (24 original + coverage additions)."""
    if split in ('FINAL', 'WORKFLOW'): return program_queries(s, program, draw)
    world = final_world(s, program); out = OLD.questions(s, edited=world, challenge=False, draw=draw)
    for q in out: q['additional'] = False; q['program_id'] = program
    return out

def query_record(s, q, split, draw, order='original', variant=0, program='single'):
    rec = S.query_record(s, q, split, draw, order, variant, program); rec['additional'] = bool(q.get('additional', False)); return rec

def parse_symbol(text, labels): return S.parse_symbol(text, labels)

# ---------------------------------------------------------------- prompts (compiler side: source text + clause spans only; exact V2 demonstrations)
def demo_block(procedure): return S.demo_block(procedure)

def correction_line(s, e):
    v = int(e.value)
    return f"Correction: the record for {s.actors[e.actor]} on the {s.projects[e.project]} proposal is updated. {s.actors[e.actor]} is now "+('in favor of' if v else 'opposed to')+f' the {s.projects[e.project]} proposal.'

def prefix_text(s, procedure, order='original', variant=0, world=None, corrections=None):
    """(text, spans): V2 demonstrations + rendered records of `world` (default the source scene) in the given order/wording; optional
    correction lines (one per declared command, in command order) inserted before 'End of records.'.  Never contains a question or answer."""
    w = s if world is None else world
    body, spans = OLD.render(w, order, variant); demos = demo_block(procedure)
    if corrections:
        tail = 'End of records.\n'
        if not body.endswith(tail): raise ValueError('unexpected render tail')
        body = body[:-len(tail)]+''.join(correction_line(s, e)+'\n' for e in corrections)+tail
    off = len(demos)
    return demos+body, {k: (off+a, off+b) for k, (a, b) in spans.items()}

def suffix_text(text): return text+INSTRUCTION

# ---------------------------------------------------------------- metrics (scene-level units; orders/draws are repeated conditions)
def summarize(rows):
    """V2 summary on the ORIGINAL question bundle (all24 = every original question of an artifact correct); plus `all_program_questions`
    (every question including coverage additions correct) and the additions' accuracy when additions exist."""
    orig = [r for r in rows if not r.get('additional')]; m = S.summarize(orig)
    if any(r.get('additional') for r in rows):
        full = S.summarize(rows); m['all_program_questions'] = full['all24']; m['accuracy_all_program_questions'] = full['accuracy']
        adds = [r for r in rows if r.get('additional')]; m['additional_accuracy'] = float(np.mean([r['prediction'] == r['gold'] for r in adds])); m['additional_rows'] = len(adds)
        for sid, v in m['per_scene'].items(): v['all_program_questions'] = full['per_scene'][sid]['all24']; v['accuracy_all_program_questions'] = full['per_scene'][sid]['accuracy']
    else:
        m['all_program_questions'] = m['all24']; m['accuracy_all_program_questions'] = m['accuracy']
        for v in m['per_scene'].values(): v['all_program_questions'] = v['all24']; v['accuracy_all_program_questions'] = v['accuracy']
    return m

per_scene = S.per_scene; selection_score = S.selection_score; feasible = S.feasible; frontier_score = S.frontier_score; select_recipe = S.select_recipe
def bootstrap_mean(values, seed=BOOT_SEED, n=BOOT_DRAWS, level=.95): return S.bootstrap_mean(values, seed=seed, n=n, level=level)
def paired_bootstrap(a_ps, b_ps, key, level=CONTRAST_LEVEL, seed=BOOT_SEED, n=BOOT_DRAWS): return S.paired_bootstrap(a_ps, b_ps, key, level=level, seed=seed, n=n)

def one_sided_lower(a_ps, b_ps, key, level=NONINFERIORITY['level'], seed=BOOT_SEED, n=BOOT_DRAWS):
    """One-sided lower confidence bound (level) of the within-scene mean difference a-b on the EXACT named field (e.g. all24)."""
    ks = sorted(k for k in a_ps if k in b_ps and a_ps[k][key] is not None and b_ps[k][key] is not None)
    if not ks: return None
    d = np.asarray([a_ps[k][key]-b_ps[k][key] for k in ks]); rng = np.random.default_rng(seed); idx = rng.integers(len(d), size=(n, len(d))); out = d[idx].mean(1)
    return dict(effect=float(d.mean()), lower=float(np.quantile(out, 1-level)), scenes=len(ks), level=level, field=key, discordant=int(np.sum(d != 0)))

def token_mask(cp, n_prefix, footprint): return list(OLD.token_mask(cp, n_prefix, footprint))
