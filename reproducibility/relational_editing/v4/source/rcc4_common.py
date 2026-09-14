# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""RELATIONAL_COMPOSITION_COVERAGE_V4 (RCC4) fixed contract: supplied-document custody, fresh RCC4 scenes (seed 260911398), the four matched
coverage conditions, V3 warm-start checkpoint identities, the declared complete-single CAL selection rule, and metric re-exports.

Reuses the RSO3 (V3) contract module read-only for prompts (exact V2 demonstrations), programs, coverage questions, records, metrics,
bootstraps and model identities; `rcc4_protocol` binds the V4 seed to the shared protocol module before anything else is imported.
No model imports.  The compiler never receives a question, label mapping, gold, affected flag or target world.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcc4_protocol as PR                       # noqa: E402  binds SEED 260911398 to the shared protocol module (must precede other use)
import rso3_common as Q                          # noqa: E402  V3 contract (prompts/programs/metrics/models), read-only

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/relational_composition_coverage_v4'
SUPPLIED = Path(os.environ.get('RCC4_SUPPLIED', str(REPORT/'supplied')))
DATA = SUPPLIED/'data'
V3_STORE = Path(os.environ.get('RCC4_V3_STORE', 'external-artifacts'))

now = Q.now; digest = Q.digest; sha = Q.sha; json_safe = Q.json_safe; canonical = Q.canonical; dump = Q.dump; load = Q.load; read_rows = Q.read_rows; write_rows = Q.write_rows
hash_rank = Q.hash_rank; budget = Q.budget
SEED = PR.SEED
BUNDLE_ZIP_SHA256 = None
CONTRACT_SHA256 = {}
DATA_SHA256 = {'fit_scenes.jsonl': '27c9bce66077af63be4f9de2a6f468a560d4333298a032e64a3c332fa129e8dd',
               'cal_scenes.jsonl': '4ea419b7cbe9ca2891ffe1a7455e0c10bdc161f858a05b191eb1edb8c7997892',
               'final_scenes.jsonl': 'b8f996543a3e172b82810095e668f01cbfea27d6fedc92623317b2e675a2aab3',
               'workflow_scenes.jsonl': '083d671eae02f29b8a9a16c331ab992c259e8a14af0f8037de5b7ff41d93163f'}

MODELS = Q.MODELS; GEMMA = Q.GEMMA; QWEN = Q.QWEN; RANK = Q.RANK; SEEDS = Q.SEEDS; GROUP = Q.GROUP; BETA = Q.BETA
EPOCHS = 2; CHECKPOINTS = (0, 1, 2); SLOTS = 24          # question-forward slots per scene per epoch, identical in every condition
ARMS = ('INVARIANT_SET', 'FREE_OVERWRITE')
CONDITIONS = {'INV_SPARSE_CONTINUE': dict(arm='INVARIANT_SET', coverage='sparse'), 'INV_COMPLETE_SINGLE': dict(arm='INVARIANT_SET', coverage='complete'),
              'FREE_SPARSE_CONTINUE': dict(arm='FREE_OVERWRITE', coverage='sparse'), 'FREE_COMPLETE_SINGLE': dict(arm='FREE_OVERWRITE', coverage='complete')}
CONDITION_ORDER = tuple(CONDITIONS)                       # declared tie order (spec listing order)
MATCHED = (('INV_COMPLETE_SINGLE', 'INV_SPARSE_CONTINUE'), ('FREE_COMPLETE_SINGLE', 'FREE_SPARSE_CONTINUE'))   # COMPLETE minus matched SPARSE
FROZEN_V3 = {'FROZEN_V3_INVARIANT_SET': 'INVARIANT_SET', 'FROZEN_V3_FREE_OVERWRITE': 'FREE_OVERWRITE'}
CANONICAL = 'CANONICAL_CURRENT'
ANCHOR_AFFINE = 'FROZEN_V2_CLAUSE'                         # Gemma-only small frozen-affine anchor panel (numerical continuity, no efficacy claim)
PROCEDURES = {'gemma': 'C1', 'qwen': 'T4'}                 # the V3-selected procedures; V4 re-qualifies each actor on the fresh CAL with that procedure only
Q.PROCEDURES = {k: (v,) for k, v in PROCEDURES.items()}
V3_SELECTED = {'gemma': {'INVARIANT_SET': dict(lr=0.0003, epoch=2), 'FREE_OVERWRITE': dict(lr=0.001, epoch=3)},
               'qwen': {'INVARIANT_SET': dict(lr=0.0003, epoch=3), 'FREE_OVERWRITE': dict(lr=0.001, epoch=2)}}
PROGRAMS = Q.PROGRAMS; EQUAL_TERMINAL = Q.EQUAL_TERMINAL; WORKFLOW_PROGRAMS = Q.WORKFLOW_PROGRAMS; ORDERS = Q.ORDERS; SIZES = Q.SIZES
PANELS = dict(witness=4, generation=8, anchor=16)
COVERAGE_PROGRAMS = ('AB', 'ABC')                          # the new primary endpoint programs (order-equivalent duplicates BA/CBA are not double counted)
BOOT_SEED = 260919; BOOT_DRAWS = 10000; CONTRAST_LEVEL = .9875
FEASIBLE_HARM = Q.FEASIBLE_HARM; FEASIBLE_DIR = Q.FEASIBLE_DIR
MAX_NEW_TOKENS = Q.MAX_NEW_TOKENS; CHUNK = Q.CHUNK; INSTRUCTION = Q.INSTRUCTION
BUDGET = {}  # Caller-supplied runtime policy for optional model execution; no private spending defaults.

# ---------------------------------------------------------------- custody
def verify_bundle():
    raise RuntimeError("Original mixed source bundle is not distributed. Public artifact identities are verified with python -m repro verify.")

def v3_index(): return load(SUPPLIED/'v3_selected/INDEX.json')

def v3_checkpoint_dir(actor, arm, seed, store=None):
    sel = V3_SELECTED[actor][arm]; return Path(store or V3_STORE)/actor/'arms'/arm/f'lr{sel["lr"]}_s{seed}'/f'epoch{sel["epoch"]}'

def verify_v3_checkpoint(actor, arm, seed, store=None):
    """The retained V3 selected checkpoint (setter.npz + META.json) must hash-match the index recorded from the local mirror at packaging."""
    d = v3_checkpoint_dir(actor, arm, seed, store); rec = v3_index()['checkpoints'][actor][arm][str(seed)]; got = dict(setter=sha(d/'setter.npz'), meta=sha(d/'META.json'))
    if got != rec['sha256']: raise ValueError(f'V3 checkpoint hash mismatch: {actor}/{arm}/s{seed}')
    return dict(dir=str(d), sha256=got, v3_lr=rec['lr'], v3_epoch=rec['epoch'])

# ---------------------------------------------------------------- scenes and questions (V4 seed)
def scenes(split):
    rows = read_rows(DATA/(split.lower()+'_scenes.jsonl')); out = [Q.scene_from_dict(r) for r in rows]
    if any(s.split != split for s in out) or any(PR.TAG not in s.scene_id for s in out): raise ValueError('split mismatch')
    return out

cell = Q.cell; subset = Q.subset; noop_scenes = Q.noop_scenes; programs = Q.programs; final_world = Q.final_world; program_queries = Q.program_queries
prefix_text = Q.prefix_text; suffix_text = Q.suffix_text; demo_block = Q.demo_block; query_record = Q.query_record; parse_symbol = Q.parse_symbol; token_mask = Q.token_mask

def questions(s, split, draw, program='single', coverage='sparse'):
    """FIT/CAL: the six original specifications (sparse) or all twelve including both/either and the second comparison pair (complete),
    per draw, in the program's final world (single flip or declared no-op).  FINAL/WORKFLOW: the V3 program bundles (24 original +
    coverage additions), coverage argument ignored."""
    if split in ('FINAL', 'WORKFLOW'): return program_queries(s, program, draw)
    if coverage not in ('sparse', 'complete'): raise ValueError(coverage)
    if program not in ('single', 'noop'): raise ValueError('only single flips and declared no-ops enter FIT/CAL')
    world = final_world(s, program); out = Q.OLD.questions(s, edited=world, challenge=True, draw=draw)
    if coverage == 'sparse': out = out[:SPARSE_SPECS]          # the original six specifications: an exact subset (same text, labels, ids) of the complete bundle
    for q in out: q['additional'] = False; q['program_id'] = program; q['coverage'] = coverage
    return out

SPARSE_SPECS = 6          # query ids -dX-00 .. -dX-05 are the sparse subset of the complete bundle

# ---------------------------------------------------------------- metrics / selection
summarize = Q.summarize; per_scene = Q.per_scene; feasible = Q.feasible; selection_score = Q.selection_score; frontier_score = Q.frontier_score
def bootstrap_mean(values, seed=BOOT_SEED, n=BOOT_DRAWS, level=.95): return Q.S.bootstrap_mean(values, seed=seed, n=n, level=level)
def paired_bootstrap(a_ps, b_ps, key, level=CONTRAST_LEVEL, seed=BOOT_SEED, n=BOOT_DRAWS): return Q.S.paired_bootstrap(a_ps, b_ps, key, level=level, seed=seed, n=n)
def one_sided_lower(a_ps, b_ps, key, level=.975, seed=BOOT_SEED, n=BOOT_DRAWS): return Q.one_sided_lower(a_ps, b_ps, key, level=level, seed=seed, n=n)

def rank_key(summ, checkpoint, order_index=0):
    """Declared complete-single CAL rule (feasibility first): (infeasible, -all24, -changed, harm, checkpoint, declared order)."""
    f = feasible(summ); a = summ['all24'] if summ['all24'] is not None else 0.; c = summ['changed'] if summ['changed'] is not None else 0.; h = summ['harm'] if summ['harm'] is not None else 1.
    return (0 if f else 1, -round(a, 12), -round(c, 12), round(h, 12), int(checkpoint), int(order_index))

def select_checkpoint(cands):
    """cands: [dict(checkpoint, cal)] for checkpoints 0/1/2 of one condition and seed, evaluated on complete single-edit CAL bundles.
    Feasible (<=5% harm, >=80% each direction) first; then highest whole-bundle (all-24) correctness, higher changed accuracy, lower harm,
    earlier checkpoint.  If none is feasible: best by the V3 frontier rule (changed - 2*harm, same tie order), reported as infeasible."""
    if not cands: raise ValueError('no candidates')
    feas = [c for c in cands if feasible(c['cal'])]
    if feas: best = min(feas, key=lambda c: rank_key(c['cal'], c['checkpoint'])); return dict(checkpoint=best['checkpoint'], feasible=True, label='capable_candidate', rule='complete_single_cal_feasibility_first')
    best = min(cands, key=lambda c: (-round(selection_score(c['cal']), 12), -round(c['cal']['all24'] or 0, 12), round(c['cal']['harm'] or 1, 12), c['checkpoint']))
    return dict(checkpoint=best['checkpoint'], feasible=False, label='diagnostic_not_capable', rule='v3_frontier_changed_minus_2harm (no feasible checkpoint)')

def select_canonical(selected):
    """selected: {condition: dict(checkpoint, cal)} for one actor and seed (the selected checkpoint of each condition).  Same rule; ties by
    the declared condition order.  Returns the condition name whose selected editor CANONICAL_CURRENT replays."""
    best = min(selected, key=lambda c: rank_key(selected[c]['cal'], selected[c]['checkpoint'], CONDITION_ORDER.index(c)))
    return dict(condition=best, feasible=feasible(selected[best]['cal']), rule='same complete-single CAL rule over the four selected editors; ties by declared condition order')

def select_workflow_candidate(selected_by_seed):
    """One condition per actor for the receiver/aggregator workflow, frozen by CAL: best worse-seed rank under the same rule."""
    conds = sorted(set.intersection(*[set(v) for v in selected_by_seed.values()]), key=CONDITION_ORDER.index)
    def worse(c): return max(rank_key(selected_by_seed[s][c]['cal'], selected_by_seed[s][c]['checkpoint'], CONDITION_ORDER.index(c)) for s in selected_by_seed)
    best = min(conds, key=worse); return dict(condition=best, rule='worse-seed rank under the complete-single CAL rule; ties by declared condition order')
