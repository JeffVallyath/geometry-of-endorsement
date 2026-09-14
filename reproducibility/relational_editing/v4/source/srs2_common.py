# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""SHARED_RELATION_STATE_V2 (SRS2) fixed contract: bundle custody, scenes, prompts, questions, records, metrics, selection rules.

No model imports.  The compiler interface (`protocol.CompileRequest`) receives prefix token ids, explicit clause spans, desired values
and a checkpoint identity only; every question, label mapping, gold index, affected flag and target world is evaluator-side.
The historical QBRC196 modules (task/metrics/adapter/train) are imported read-only from the supplied `historical/` folder for the
anchor replay and the frozen V1 references; nothing here launches the old study.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/shared_relation_state_v2'
BUNDLE = Path(os.environ.get('SRS2_SUPPLIED', str(REPORT/'supplied')))
DATA = BUNDLE/'data'
HIST = BUNDLE/'historical'
ANCHORS = Path(os.environ.get('SRS2_ANCHORS', str(REPORT/'anchors')))
sys.path.insert(0, str(BUNDLE/'reference'))
import protocol as P                      # noqa: E402  new fixed scenes / semantic SET algebra / questions / masks / compile request
os.environ.setdefault('QBRC_SUPPLIED', str(HIST))
sys.path.insert(0, str(HIST/'scripts'))
import qbrc_common as QB                  # noqa: E402  historical prompts/questions (old FINAL anchors) and metrics; read-only reference code
METRICS = QB.METRICS

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def json_safe(value):
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return QB.json_safe(value)
def canonical(value): return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
NL = chr(10)
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline=NL) as f: f.write(json.dumps(json_safe(value), indent=2, allow_nan=False)+NL)
load = QB.load; read_rows = QB.read_rows
def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline=NL) as f:
        for row in rows: f.write(canonical(row)+NL)
def hash_rank(*parts): return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
def budget(deadline, reserve=0): return QB.budget(deadline, reserve)

BUNDLE_ZIP_SHA256 = None
CONTRACT_SHA256 = {}
DATA_SHA256 = {'fit_scenes.jsonl': '43c0ceb4a0f32ce82cceca24b72f4a00bb7fb7f04a151cc5cd1c7af807434141',
               'cal_scenes.jsonl': '712599f1adcafe09fd1b41d9e5afd4b459753dea7f82eeeaa3192fe4d4365f19',
               'final_scenes.jsonl': '07b1da3d2cc6513fb41f7448e0e071f9a2a1f685c0e0674045e4d7b568854318',
               'workflow_scenes.jsonl': '65c5cb0e49f6177291254145729adde7587c538a38efd6b897a4600dac22929a'}

GEMMA = {'key': 'gemma', 'id': 'google/gemma-2-9b-it', 'revision': '11c9b309abf73637e4b6f9a3fa1e92e615547819', 'layers': 42, 'cap': 30.0, 'hidden': 3584,
         'site': 15, 'late_site': 27, 'sites': (15,), 'template_kwargs': {}, 'cap_rule': 'inherited V1 site-15 relation cap 103.18592071533203 (clip THEN strength)',
         'site_convention': 'zero-based post-block index of the decoder layer whose output is edited'}
QWEN = {'key': 'qwen', 'id': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218', 'layers': 36, 'cap': None, 'hidden': 4096,
        'site': int(round(16*36/42))-1, 'late_site': None, 'sites': (int(round(16*36/42))-1,), 'template_kwargs': {'enable_thinking': False},
        'cap_rule': '2 x 95th percentile of FIT ||z_dst - z_src|| at the mapped site, fixed before any effect',
        'site_rule': 'round(16 * Qwen_layers / 42) - 1 = round(13.71) - 1 = 13; actual module depth verified at load'}
MODELS = {'gemma': GEMMA, 'qwen': QWEN}
GEMMA_CAP = 103.18592071533203
ARMS = ('SHARED_CLAUSE', 'SHARED_TAIL', 'QUERY_CLAUSE', 'QUERY_TAIL')
FOOTPRINT = {'SHARED_CLAUSE': 'clause', 'SHARED_TAIL': 'tail', 'QUERY_CLAUSE': 'clause', 'QUERY_TAIL': 'tail'}
AWARE = {'SHARED_CLAUSE': False, 'SHARED_TAIL': False, 'QUERY_CLAUSE': True, 'QUERY_TAIL': True}
RANK = 16
LRS = (0.0003, 0.001)
STRENGTHS = (0.5, 0.75, 1.0, 1.25)
LR_EPOCHS = 2          # complete epochs per LR under seed 0 before the LR choice
MAX_EPOCHS = 6         # ceiling for the extension of the chosen LR
PATIENCE = 2           # stop extension after two complete epochs without CAL frontier improvement
GROUP = 4              # optimizer step accumulates four questions from one scene
SEEDS = (0, 1)
PROCEDURES = ('C1', 'C1S', 'T4')
QUAL_N = 96; QUAL_PARSE = .95; QUAL_DIRECT = .90; QUAL_EQUALITY = .85
MAX_NEW_TOKENS = 32
FEASIBLE_HARM = .05; FEASIBLE_DIR = .80
BOOT_SEED = 260917; BOOT_DRAWS = 10000
PROGRAMS = ('single', 'noop', 'repeat', 'restore', 'AB', 'BA', 'ABC', 'CBA')
EQUAL_TERMINAL = (('single', 'repeat'), ('AB', 'BA'), ('ABC', 'CBA'))
ORDERS = ('early', 'late')
SIZES = {'gemma': {'FULL': dict(final=128, algebra=64, workflow=128), 'SMALL': dict(final=64, algebra=32, workflow=64)},
         'qwen': {'FULL': dict(final=64, algebra=32, workflow=64), 'SMALL': dict(final=32, algebra=16, workflow=32)}}
PANELS = dict(energy=32, control=32, variant1=32, witness=4, generation=8)
BUDGET = {}  # Caller-supplied runtime policy for optional model execution; no private spending defaults.
INSTRUCTION = QB.INSTRUCTION      # '\nReply with only that answer on the first line.'
CONTRAST_LEVEL = .9875             # two-sided scene-bootstrap level for the four matched contrasts
NONINFERIORITY = dict(scene=.05, harm=.02, level=.975)

# ---------------------------------------------------------------- bundle custody
def verify_bundle():
    raise RuntimeError("Original mixed source bundle is not distributed. Public artifact identities are verified with python -m repro verify.")

# ---------------------------------------------------------------- scenes
def scene_from_dict(d):
    d = dict(d); d['actors'] = tuple(d['actors']); d['projects'] = tuple(d['projects']); d['layout'] = tuple(d['layout']); d['values'] = tuple(tuple(x) for x in d['values'])
    return P.validate_scene(P.Scene(**d))

def scenes(split):
    rows = read_rows(DATA/(split.lower()+'_scenes.jsonl')); out = [scene_from_dict(r) for r in rows]
    if any(s.split != split for s in out): raise ValueError('split mismatch')
    return out

def old_scenes(split='FINAL', task='relation'): return QB.by_task(QB.scenes(split), task)

def cell(s): return (len(s.actors), len(s.projects), s.values[s.target_actor][s.target_project])

def subset(scene_list, n, salt):
    """Fixed hash-ordered subset chosen by scene ID before any outcome, preserving every (size, projects, direction) cell equally."""
    cells = {}
    for s in scene_list: cells.setdefault(cell(s), []).append(s)
    if n >= len(scene_list): return list(scene_list)
    keys = sorted(cells); per = n//len(keys); extra = sorted(keys, key=lambda c: hash_rank(salt, 'cell', c))[:n-per*len(keys)]; out = []
    for c in keys: out.extend(sorted(cells[c], key=lambda s: hash_rank(salt, s.scene_id))[:per+(1 if c in extra else 0)])
    return sorted(out, key=lambda s: s.scene_id)

def noop_scenes(scene_list): return [s for i, s in enumerate(scene_list) if i % 4 == 0]

def edited_world(s, program): return P.apply_program(s, P.programs(s)[program])

# ---------------------------------------------------------------- prompts (compiler side: source text + clause spans only)
_DEMOS = {}
def _pick(fit, salt, k): return sorted(fit, key=lambda s: hash_rank(salt, s.scene_id))[:k]

def _example(s, q, i):
    text, _ = P.render(s); return f'Example {i+1}:\n{text}{q["text"]}\nAnswer: {q["labels"][q["source_gold"]]}\n'

def demo_block(procedure, fit=None):
    """FIT-only demonstrations, identical across methods and worlds. C1: the V1 two demonstrations (direct/opposes, alternating answers).
    C1S: C1 plus one explicit same-side example.  T4: four examples listing equality's four truth cases.  No both/either example anywhere."""
    if procedure not in PROCEDURES: raise ValueError(procedure)
    if procedure in _DEMOS: return _DEMOS[procedure]
    fit = scenes('FIT') if fit is None else fit; lines = []
    if procedure in ('C1', 'C1S'):
        for i, s in enumerate(_pick(fit, 'demo', 2)):
            q = next(x for x in P.questions(s, edited=s, draw=0) if x['family'] in ('direct', 'opposes') and x['source_gold'] == i % 2)
            lines.append(_example(s, q, i))
    if procedure == 'C1S':
        s = _pick(fit, 'demo-same', 3)[-1]; q = next(x for x in P.questions(s, edited=s, draw=0) if x['family'] == 'same'); lines.append(_example(s, q, 2))
    if procedure == 'T4':
        want = [(0, 0), (0, 1), (1, 0), (1, 1)]; i = 0
        for s in sorted(fit, key=lambda s: hash_rank('demo-table', s.scene_id)):
            a, p = s.target_actor, s.target_project; b = (a+1) % len(s.actors); pair = (s.values[a][p], s.values[b][p])
            if pair not in want: continue
            want.remove(pair); q = next(x for x in P.questions(s, edited=s, draw=0) if x['family'] == 'same'); lines.append(_example(s, q, i)); i += 1
            if not want: break
    _DEMOS[procedure] = '\n'.join(lines)+'\nNow the actual records:\n'
    return _DEMOS[procedure]

def correction_line(s, world):
    a, p = s.target_actor, s.target_project; v = world.values[a][p]
    return f"Correction: the record for {s.actors[a]} on the {s.projects[p]} proposal is updated. {s.actors[a]} is now "+('in favor of' if v else 'opposed to')+f' the {s.projects[p]} proposal.'

def prefix_text(s, procedure, order='original', variant=0, world=None, correction=False):
    """(text, spans) : demonstrations + rendered records of `world` (default the source scene) in the given prefix order/wording.
    spans map (actor, project) -> character interval of that record inside `text`.  Never contains a question or answer."""
    w = s if world is None else world
    body, spans = P.render(w, order, variant); demos = demo_block(procedure)
    if correction:
        tail = 'End of records.\n'
        if not body.endswith(tail): raise ValueError('unexpected render tail')
        body = body[:-len(tail)]+correction_line(s, world if world is not None else P.set_value(s, P.primary_edit(s)))+'\n'+tail
    off = len(demos)
    return demos+body, {k: (off+a, off+b) for k, (a, b) in spans.items()}

def suffix_text(text): return text+INSTRUCTION

def questions(s, split, draw, world=None, program='single'):
    edited = edited_world(s, program) if world is None else world
    return P.questions(s, edited=edited, challenge=(split in ('FINAL', 'WORKFLOW')), draw=draw)

def query_record(s, q, split, draw, order='original', variant=0, program='single'):
    a, p = s.target_actor, s.target_project; spec = q['spec']
    return dict(scene_id=s.scene_id, split=split, draw=draw, order=order, variant=variant, program=program, query_id=q['query_id'], family=q['family'], labels=list(q['labels']),
                gold=int(q['gold']), source_gold=int(q['source_gold']), changed=bool(q['changed']), n_actors=len(s.actors), n_projects=len(s.projects),
                holder=s.actors[spec['a']], project=s.projects[spec['p']], touches_target=(spec['a'] == a and spec['p'] == p),
                mentions_target=((spec['a'] == a or spec.get('b') == a) and spec['p'] == p), addressed_project=(spec['p'] == p),
                code_reversed=list(q['labels']) != ['No', 'Yes'], source_stance=int(s.values[a][p]), direction=1-int(s.values[a][p]), cell=list(cell(s)))

def parse_symbol(text, labels): return QB.parse_symbol(text, labels)

# ---------------------------------------------------------------- metrics (scene-level units; orders/draws are repeated conditions)
def per_scene(rows):
    """Per scene: changed accuracy, invariant new-error fraction (all invariant rows), source-correct conditional fraction, all-24 (mean over
    prefix orders of every question correct), accuracy, touched-record accuracy/error, other-address error rate, preservation of originally
    correct answers, gold accuracy."""
    by = {}
    for r in rows: by.setdefault(r['scene_id'], []).append(r)
    out = {}
    for sid, rs in by.items():
        ch = [r for r in rs if r['gold'] != r['source_gold']]; inv = [r for r in rs if r['gold'] == r['source_gold']]
        src_ok = [r for r in inv if r['source_prediction'] == r['gold']]
        arts = {}
        for r in rs: arts.setdefault((r.get('order'), r.get('variant'), r.get('draw')), []).append(r)
        all_by_art = {k: float(all(r['prediction'] == r['gold'] for r in v)) for k, v in arts.items()}
        # full-scene all-24: every question (both draws) of one compiled artifact (scene x order x variant) correct
        art24 = {}
        for (o, v, d), val in all_by_art.items(): art24.setdefault((o, v), []).append(val)
        touched = [r for r in rs if r.get('touches_target')]; other = [r for r in rs if not r.get('touches_target') and r['gold'] == r['source_gold']]
        originally_correct = [r for r in rs if r['source_prediction'] == r['source_gold']]
        out[sid] = dict(scene_id=sid, queries=len(rs), direction=rs[0]['direction'], cell=rs[0].get('cell'),
                        changed=float(np.mean([r['prediction'] == r['gold'] for r in ch])) if ch else None,
                        harm=float(np.mean([r['source_prediction'] == r['gold'] and r['prediction'] != r['gold'] for r in inv])) if inv else None,
                        conditional_harm=float(np.mean([r['prediction'] != r['gold'] for r in src_ok])) if src_ok else None,
                        all24=float(np.mean([min(v) for v in art24.values()])),
                        accuracy=float(np.mean([r['prediction'] == r['gold'] for r in rs])),
                        touched_accuracy=float(np.mean([r['prediction'] == r['gold'] for r in touched])) if touched else None,
                        other_address_error=float(np.mean([r['prediction'] != r['gold'] for r in other])) if other else None,
                        preservation=float(np.mean([r['prediction'] == r['gold'] for r in originally_correct if r['gold'] == r['source_gold']])) if any(r['gold'] == r['source_gold'] for r in originally_correct) else None,
                        nonaddressed_change=float(np.mean([r['prediction'] != r['source_prediction'] for r in rs if not r.get('mentions_target') and r['gold'] == r['source_gold']])) if any(not r.get('mentions_target') and r['gold'] == r['source_gold'] for r in rs) else None)
    return out

def summarize(rows):
    ps = per_scene(rows)
    def avg(k, sub=None):
        xs = [v[k] for v in ps.values() if v[k] is not None and (sub is None or sub(v))]; return float(np.mean(xs)) if xs else None
    return dict(n_scenes=len(ps), n_queries=len(rows), changed=avg('changed'), harm=avg('harm'), conditional_harm=avg('conditional_harm'), all24=avg('all24'), accuracy=avg('accuracy'),
                touched_accuracy=avg('touched_accuracy'), other_address_error=avg('other_address_error'), preservation=avg('preservation'), nonaddressed_change=avg('nonaddressed_change'),
                changed_by_direction={d: avg('changed', lambda v, d=d: v['direction'] == d) for d in (0, 1)},
                by_cell={str(c): dict(changed=avg('changed', lambda v, c=c: tuple(v['cell']) == c), harm=avg('harm', lambda v, c=c: tuple(v['cell']) == c), all24=avg('all24', lambda v, c=c: tuple(v['cell']) == c)) for c in sorted({tuple(v['cell']) for v in ps.values() if v['cell']})},
                by_family={f: float(np.mean([r['prediction'] == r['gold'] for r in rows if r['family'] == f])) for f in sorted({r['family'] for r in rows})},
                per_scene=ps)

def selection_score(summ):
    ch = summ['changed'] if summ['changed'] is not None else 0.; harm = summ['harm'] if summ['harm'] is not None else 0.
    return ch-2*harm

def feasible(summ):
    return (summ['harm'] is not None and summ['harm'] <= FEASIBLE_HARM and all(v is not None and v >= FEASIBLE_DIR for v in summ['changed_by_direction'].values()))

def frontier_score(summ):
    """CAL frontier value used for the LR choice and the extension stopping rule: feasible candidates rank above infeasible ones by changed
    accuracy; infeasible ones by changed - 2*harm.  Uses only the training families (direct/negation/equality/nonaddressed)."""
    return (1 if feasible(summ) else 0, summ['changed'] if feasible(summ) else selection_score(summ))

def select_recipe(candidates):
    """candidates: list of dict(lr, epoch, strength, seeds={0: summ, 1: summ}, norm) for recipes evaluated in BOTH seeds.
    Declared rule: maximize worse-seed changed accuracy among recipes feasible in EACH seed (<=5% harm, >=80% each direction);
    ties -> higher worse-seed all-24, lower worse-seed harm, smaller norm, earlier epoch.  Otherwise (diagnostic, not capable):
    maximize worse-seed (changed - 2*harm) with the same tie order."""
    if not candidates: raise ValueError('no candidates')
    def worse(c, k, sign=1): return sign*min(sign*(c['seeds'][s][k] if c['seeds'][s][k] is not None else (0. if sign > 0 else 1.)) for s in c['seeds'])
    def key(c): return (-round(worse(c, 'changed'), 12), -round(worse(c, 'all24'), 12), round(worse(c, 'harm', -1), 12), round(c.get('norm', 0.), 6), c['epoch'], c['strength'])
    feas = [c for c in candidates if all(feasible(c['seeds'][s]) for s in c['seeds']) and len(c['seeds']) == 2]
    if feas: best = min(feas, key=key); return dict({k: v for k, v in best.items() if k != 'seeds'}, feasible=True, label='capable_candidate')
    def key2(c): return (-round(min(selection_score(c['seeds'][s]) for s in c['seeds']), 12), -round(worse(c, 'all24'), 12), round(worse(c, 'harm', -1), 12), round(c.get('norm', 0.), 6), c['epoch'], c['strength'])
    best = min(candidates, key=key2); return dict({k: v for k, v in best.items() if k != 'seeds'}, feasible=False, label='diagnostic_not_capable')

def bootstrap_mean(values, seed=BOOT_SEED, n=BOOT_DRAWS, level=.95):
    d = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(d): return None
    rng = np.random.default_rng(seed); idx = rng.integers(len(d), size=(n, len(d))); out = d[idx].mean(1); q = (1-level)/2
    return dict(mean=float(d.mean()), lower=float(np.quantile(out, q)), upper=float(np.quantile(out, 1-q)), scenes=int(len(d)), level=level)

def paired_bootstrap(a_ps, b_ps, key, level=CONTRAST_LEVEL, seed=BOOT_SEED, n=BOOT_DRAWS):
    ks = sorted(k for k in a_ps if k in b_ps and a_ps[k][key] is not None and b_ps[k][key] is not None)
    if not ks: return None
    d = np.asarray([a_ps[k][key]-b_ps[k][key] for k in ks]); rng = np.random.default_rng(seed); idx = rng.integers(len(d), size=(n, len(d))); out = d[idx].mean(1); q = (1-level)/2
    return dict(effect=float(d.mean()), lower=float(np.quantile(out, q)), upper=float(np.quantile(out, 1-q)), scenes=len(ks), level=level, discordant=int(np.sum(d != 0)))
