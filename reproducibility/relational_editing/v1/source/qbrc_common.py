# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""QUESTION_BLIND_RELATION_CONTROL_V1 (QBRC196) fixed contract, bundle custody, scenes, prompts and metrics. No model imports.

The supplied reference task module renders prefixes/questions and holds the executable gold evaluator.  The compiler interface
(`CompileRequest`) receives only the source prefix text, the explicit addressed edit request and the clause character span; every
question, label mapping, option table, changed/invariant flag or gold index is evaluator-side only.
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
REPORT = ROOT/'reports/question_blind_relation_control_v1'
BUNDLE = Path(os.environ.get('QBRC_SUPPLIED', str(REPORT/'supplied')))
DATA = BUNDLE/'data'
sys.path.insert(0, str(BUNDLE/'reference'))
import task as TASK                    # noqa: E402  deterministic scenes/questions/gold (evaluator side)
import metrics as METRICS              # noqa: E402  scene-level summaries + paired bootstrap
from access_contract import CompileRequest  # noqa: E402

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def json_safe(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value): return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value
def canonical(value): return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f: f.write(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n')
def load(path): return json.loads(Path(path).read_text(encoding='utf8'))
def read_rows(path):
    with Path(path).open(encoding='utf8') as f: return [json.loads(line) for line in f if line.strip()]
def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        for row in rows: f.write(canonical(row)+'\n')

BUNDLE_ZIP_SHA256 = None
CONTRACT_SHA256 = {}
DATA_SHA256 = {'fit_scenes.jsonl': '486b45e771c7aceba4a689ca4a386d5a84bfa9cd1bd69e80238e240ae49016a5',
               'cal_scenes.jsonl': 'c19b13129c91c55db794d7ce5f99b45d70a37bdacb81143f56647040a7fa87f3',
               'final_scenes.jsonl': '8e2b189db41b319eadc6de1befa2ee69bd66ccccb39801c84bbcfbd67f145fbf'}

GEMMA = {'key': 'gemma', 'id': 'google/gemma-2-9b-it', 'revision': '11c9b309abf73637e4b6f9a3fa1e92e615547819', 'layers': 42, 'cap': 30.0, 'hidden': 3584,
         'sites': (15, 27), 'late_site': 27, 'site_convention': 'zero-based post-block index of the decoder layer whose output is edited'}
def llama_site(gemma_site, L_llama=32, L_gemma=42): return int(round((gemma_site+1)*L_llama/L_gemma))-1
LLAMA = {'key': 'llama', 'id': 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'revision': '0e9e39f249a16976918f6564b8830bc894c89659', 'layers': 32, 'cap': None, 'hidden': 4096,
         'sites': tuple(llama_site(s) for s in (15, 27)), 'late_site': llama_site(27),
         'site_rule': 'round((site+1)*32/42)-1 per the goal: 15->11, 27->20 (NOT the historical proportional 21); actual module depth verified at load'}
MODELS = {'gemma': GEMMA, 'llama': LLAMA}
SPLITS = ('FIT', 'CAL', 'FINAL')
TASKS = ('relation', 'priority')
RANK = 16
LRS = (0.0003, 0.001)
STRENGTHS = (0.5, 1.0, 2.0)
MEAN_STRENGTHS = (0.25, 0.5, 1.0, 2.0, 4.0)
MAX_EPOCHS = 4
GROUP = 4
SEEDS = (0, 1)
FAMILIES = ('LATE', 'PREFIX_LAST', 'PREFIX_DISTRIBUTED')
PROCEDURES = ('C0', 'C1', 'C2')
MAX_NEW_TOKENS = 32
QUAL_REQUESTS = 48
QUAL_PARSE_MIN = .95
QUAL_ACC_MIN = .90
FEASIBLE_HARM = .05
FEASIBLE_CHANGED = .70
BOOT_SEED = 260913
BOOT_DRAWS = 10000
CONTROL_SCENES_PER_TASK = 16
COMPOSITION_SCENES_PER_TASK = 16
DOWNSTREAM_SCENES = 32
GENERATION_SCENES_PER_TASK = 8
FINAL_FALLBACK_SCENES_PER_TASK = 32
LLAMA_MIN_FINAL_SCENES = 32
BUDGET = {}  # Caller-supplied runtime policy for optional model execution; no private spending defaults.
INSTRUCTION = '\nReply with only that answer on the first line.'

def budget(deadline, reserve=0):
    remaining = (dt.datetime.fromisoformat(deadline)-dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining <= reserve: raise TimeoutError('RESOURCE_STOP: insufficient authorized remaining instance time')
    return remaining

def hash_rank(*parts): return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()

# ---------------------------------------------------------------- bundle custody
def verify_bundle():
    raise RuntimeError("Original mixed source bundle is not distributed. Public artifact identities are verified with python -m repro verify.")

# ---------------------------------------------------------------- scenes
def scenes(split):
    rows = read_rows(DATA/(split.lower()+'_scenes.jsonl'))
    out = [TASK.from_dict(r) for r in rows]
    if any(s.split != split for s in out): raise ValueError('split mismatch')
    return out

def by_task(scene_list, task): return [s for s in scene_list if s.task == task]

def scene_by_id(scene_list): return {s.scene_id: s for s in scene_list}

def subset_by_id(scene_list, n_per_task, salt):
    """Fixed hash-ordered subset per task, chosen by scene ID before any outcome."""
    out = []
    for task in TASKS:
        out.extend(sorted(by_task(scene_list, task), key=lambda s: hash_rank(salt, s.scene_id))[:n_per_task])
    return out

def edit_request(scene): return TASK.edit_request(scene)
def apply_edit(scene, **kw): return TASK.apply_edit(scene, **kw)

def second_edit(scene):
    """The independent second addressed edit for composition: another member (never the addressed one)."""
    n = len(scene.actors); b = (scene.target_actor+2) % n if scene.task == 'relation' else (scene.target_actor+1) % n
    return dict(actor=b, project=scene.target_project if scene.task == 'relation' else None)

def wrong_address(scene):
    """Same task/magnitude, different addressed record (another member; relation keeps the project)."""
    n = len(scene.actors); b = (scene.target_actor+1) % n
    return dict(actor=b, project=scene.target_project if scene.task == 'relation' else None)

# ---------------------------------------------------------------- demonstrations (FIT-only, identical across methods)
def _demo_block(task, k):
    fit = by_task(scenes('FIT'), task)
    chosen = sorted(fit, key=lambda s: hash_rank('demo', task, s.scene_id))[:k]
    lines = []
    for i, s in enumerate(chosen):
        text, _ = TASK.render_prefix(s)
        qs = TASK.questions(s, s, challenge=False, draw=0)   # source world questions; answers are source gold
        # alternate positive/negative (relation) or distinct chosen options (priority): pick the first query whose source gold matches the parity
        want = i % 2
        if task == 'relation':
            q = next(x for x in qs if x.family in ('direct', 'opposes') and x.source_gold_index == want)
        else:
            q = next(x for x in qs if x.family.startswith('recommend') and (x.source_gold_index % 2) == want)
        lines.append(f'Example {i+1}:\n{text}{q.text}\nAnswer: {q.labels[q.source_gold_index]}\n')
    return '\n'.join(lines)+'\nNow the actual records:\n'

_DEMOS = {}
def demo_block(task, procedure):
    k = {'C0': 0, 'C1': 2, 'C2': 4}[procedure]
    if k == 0: return ''
    key = (task, k)
    if key not in _DEMOS: _DEMOS[key] = _demo_block(task, k)
    return _DEMOS[key]

# ---------------------------------------------------------------- prompts
def prefix_text(scene, procedure='C0', target=None):
    """Compiler-side text: demonstrations (if any) + rendered records of `scene` (or of `target`, the natural counterfactual reference).
    Returns (text, clause_char_span) where the span addresses the scene's edited record inside `text`."""
    s = scene if target is None else target
    body, (a, b) = TASK.render_prefix(s)
    demos = demo_block(scene.task, procedure)
    return demos+body, (len(demos)+a, len(demos)+b)

def suffix_text(query, procedure='C0'):
    return query.text+INSTRUCTION+('\nAnswer:' if procedure == 'C2' else '')

def suffix_text_plain(text, procedure='C0'):
    return text+INSTRUCTION+('\nAnswer:' if procedure == 'C2' else '')

def questions(scene, split, draw, target=None):
    challenge = split == 'FINAL'
    return TASK.questions(scene, target, challenge=challenge, draw=draw)

def compile_request(scene, procedure, model_revision, method_digest, actor=None, project=None):
    """The ONLY inputs a learned/native compiler receives: source prefix text, the explicit addressed edit request, the clause span."""
    a = scene.target_actor if actor is None else actor; p = scene.target_project if project is None else project
    if scene.task == 'relation': old = scene.values[a][p]
    else: old = scene.priorities[a]
    # clause span: render_prefix addresses the scene's own target; for another address rebuild the span from a re-targeted view
    view = TASK.replace(scene, target_actor=a, target_project=p if scene.task == 'relation' else scene.target_project)
    text, span = prefix_text(view, procedure)
    req = CompileRequest(source_prefix=text, addressed_actor=scene.actors[a], addressed_project=scene.targets[p] if scene.task == 'relation' else None,
                         task=scene.task, new_value=1-old, clause_char_span=tuple(span), model_revision=model_revision, method_digest=method_digest)
    req.validate(); return req

# ---------------------------------------------------------------- query records
def query_record(scene, q, split, draw):
    return dict(scene_id=scene.scene_id, task=scene.task, split=split, draw=draw, query_id=q.query_id, family=q.family, labels=list(q.labels),
                gold=int(q.gold_index), source_gold=int(q.source_gold_index), affected=bool(q.affected), n_actors=len(scene.actors), n_targets=len(scene.targets),
                n_options=len(q.case['scores']) if q.case.get('type') == 'recommend' else None, holder=scene.actors[q.case['a']] if 'a' in q.case else None,
                addressed_holder=q.case.get('a') == scene.target_actor, project=scene.targets[q.case['p']] if 'p' in q.case and scene.task == 'relation' else None,
                addressed_project=(q.case.get('p') == scene.target_project) if scene.task == 'relation' else None,
                code_reversed=(list(q.labels) != ['No', 'Yes'] and scene.task == 'relation') or (scene.task == 'priority' and q.case.get('type') == 'recommend' and list(q.labels) != sorted(q.labels)),
                source_stance=(scene.values[scene.target_actor][scene.target_project] if scene.task == 'relation' else scene.priorities[scene.target_actor]),
                direction=1-(scene.values[scene.target_actor][scene.target_project] if scene.task == 'relation' else scene.priorities[scene.target_actor]))

def parse_symbol(text, labels):
    """First non-empty line must be exactly one allowed label (case-sensitive after stripping punctuation/quotes)."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), '')
    cleaned = first.strip(' \t"\'*`.:;,!)(').split()
    if not cleaned: return None, 'EMPTY'
    tok = cleaned[0].rstrip('.,:;')
    if tok in labels and len(cleaned) == 1: return tok, 'OK'
    if tok in labels: return tok, 'OK_EXTRA_WORDS'
    low = {l.lower(): l for l in labels}
    if tok.lower() in low: return low[tok.lower()], 'OK_CASE'
    return None, 'UNPARSEABLE'

# ---------------------------------------------------------------- metrics
def score_rows(rows):
    """rows: dicts with scene_id, gold, source_gold, prediction, source_prediction (indices)."""
    return METRICS.summarize(rows)

def selection_score(summary):
    ch = summary['changed']; harm = summary['invariant_harm']
    return (ch if ch is not None else 0.)-(harm if harm is not None else 0.)

def frontier_rows(rows):
    """Per task scene-macro changed accuracy, invariant harm (all invariant rows), conditional harm, all-correct, false nonaddressed changes."""
    out = {}
    for task in TASKS:
        rs = [r for r in rows if r['task'] == task]
        if not rs: out[task] = None; continue
        summ = score_rows(rs)
        inv = [r for r in rs if r['gold'] == r['source_gold']]
        src_ok = [r for r in inv if r['source_prediction'] == r['gold']]
        cond = float(np.mean([r['prediction'] != r['gold'] for r in src_ok])) if src_ok else None
        nonaddr = [r for r in rs if not r.get('addressed_holder', True) and r['gold'] == r['source_gold']]
        false_change = float(np.mean([r['prediction'] != r['source_prediction'] for r in nonaddr])) if nonaddr else None
        by_dir = {}
        for d in (0, 1):
            sub = [r for r in rs if r['direction'] == d]
            by_dir[d] = score_rows(sub)['changed'] if sub else None
        out[task] = dict(changed=summ['changed'], invariant_harm=summ['invariant_harm'], conditional_harm=cond, all_correct=summ['all_correct'], accuracy=summ['accuracy'],
                         n_scenes=summ['n_scenes'], n_queries=summ['n_queries'], false_nonaddressed_change=false_change, changed_by_direction=by_dir,
                         score=selection_score(summ))
    valid = [v for v in out.values() if v]
    out['mean_score'] = float(np.mean([v['score'] for v in valid])) if valid else None
    return out

def choose_setting(curve):
    """Declared rule: prefer feasible (<=5% harm, changed>=70%) settings by score; otherwise best score (diagnostic nonfeasible).
    Ties: lower harm, lower realized norm, earlier checkpoint. Scores are per-task means unless a task is excluded."""
    if not curve: raise ValueError('empty curve')
    def key(c): return (-round(c['score'], 12), round(c['harm'], 12), round(c.get('norm', 0.), 6), c['epoch'], c['strength'])
    feasible = [c for c in curve if c['harm'] <= FEASIBLE_HARM and c['changed'] >= FEASIBLE_CHANGED]
    if feasible: best = min(feasible, key=key); return dict(best, feasible=True)
    best = min(curve, key=key); return dict(best, feasible=False)

def paired_interval(a_rows, b_rows, key='all_correct', level=.975, seed=BOOT_SEED):
    return METRICS.paired_interval(score_rows(a_rows), score_rows(b_rows), key=key, seed=seed, n=BOOT_DRAWS, level=level)

def bootstrap_mean(rows_summary_key, per_scene_values, seed=BOOT_SEED, n=BOOT_DRAWS, level=.95):
    d = np.asarray([v for v in per_scene_values if v is not None], dtype=float)
    if not len(d): return None
    rng = np.random.default_rng(seed); out = np.empty(n)
    for i in range(n): out[i] = d[rng.integers(len(d), size=len(d))].mean()
    q = (1-level)/2
    return dict(mean=float(d.mean()), lower=float(np.quantile(out, q)), upper=float(np.quantile(out, 1-q)), scenes=int(len(d)), level=level)
