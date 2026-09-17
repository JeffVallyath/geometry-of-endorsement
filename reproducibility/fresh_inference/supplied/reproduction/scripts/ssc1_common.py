"""SSC1 immutable data and metadata boundary. Import performs no file reads."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'STATE_SUFFICIENCY_CONFIRMATION'
PUBLIC = BUNDLE / 'runtime_inputs/UPDATE_METHOD_INPUTS'
REPORT = ROOT / 'reports/state_sufficiency_confirmation'
sys.path.insert(0, str(PUBLIC))
ACTORS = ('gemma', 'qwen')
CHUNKS = {'gemma': 1, 'qwen': 1}
RECIPES = ('INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY')
CONDITIONS = ('SOURCE', *(f'{m}_s{s}' for m in RECIPES for s in (0, 1)),
              'EXISTING_CORRECTION', 'LATEST_SAME_WORDING')

def sha(path):
    from ssc1_boundary import hash_file
    return hash_file(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def rows(path):
    with Path(path).open(encoding='utf8') as f:
        for line in f: yield json.loads(line)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def method_seed(condition):
    if condition not in CONDITIONS: raise ValueError('Undeclared condition')
    return (condition[:-3], int(condition[-1])) if condition.endswith(('_s0', '_s1')) else (condition, None)

def coverage_inventory():
    """Runtime inventory from precomputed metadata ONLY, including transitive calls."""
    meta = load(BUNDLE/'frozen_data/MANIFEST.json')
    expected = dict(terminal_roots=64, technical_roots=8, origins_per_root=8, question_count=34)
    if any(meta[k] != v for k,v in expected.items()): raise ValueError('Frozen metadata differs')
    return dict(actors=list(ACTORS), terminal_roots_per_actor=64, technical_roots_per_actor=8,
        origins=8, conditions=list(CONDITIONS), contexts_per_root=57,
        ordinary_questions_per_context=34, total_logical_responses=248064,
        aliases_before_model_validation=0)

def verify_bundle():
    from ssc1_intake import verify_manifest
    manifest=load(BUNDLE/'BUNDLE_MANIFEST.json')
    scope_path=BUNDLE/'SCIENTIFIC_INPUT_EXPORT.json'
    omitted=load(scope_path)['omitted_non_scientific_documents'] if scope_path.exists() else {}
    for name,rec in omitted.items():
        allowed=name.startswith('review/') or name in ('GOAL.md','PROGRESS_CONTRACT.md','LAUNCHER.txt','START_HERE.md','VERIFICATION.json')
        if not allowed or rec!=manifest['files'].get(name): raise ValueError('Unsafe scientific export omission')
    for name,rec in manifest['files'].items():
        path=(BUNDLE/name).resolve()
        if name in omitted and not path.exists(): continue
        if not path.is_relative_to(BUNDLE.resolve()) or not path.is_file() or path.stat().st_size!=rec['bytes'] or sha(path)!=rec['sha256']:
            raise ValueError('Missing/changed supplied member: '+name)
    count=len(manifest['files'])
    public_count = verify_manifest(PUBLIC, 'ARTIFACT_MANIFEST.json')
    expected = load(BUNDLE/'inputs/INPUTS.json')['small_factor_export']
    if sha(BUNDLE/'inputs/UPDATE_METHOD_INPUTS.zip') != expected['sha256']:
        raise ValueError('Factor export changed')
    # Hashing bytes does not parse evaluation content.
    from inputs import inventory
    factors = [r for r in inventory()['selected'] if r['recipe'] in RECIPES]
    if len(factors) != 8: raise ValueError('Eight selected learned factors required')
    return dict(bundle_members=count, public_members=public_count, factors=factors,
        selected_manifest_sha256=sha(PUBLIC/'weights/SELECTED.json'))

def prepare_shards(destination):
    """CPU pre-outcome packaging. No evaluator dictionaries go into compiler shards."""
    destination = Path(destination)
    grouped = {name: defaultdict(list) for name in ('questions','labels','source_inputs')}
    for name, collection in grouped.items():
        for row in rows(BUNDLE/f'frozen_data/{name}.jsonl'): collection[row['root_id']].append(row)
    index = []
    for root in rows(BUNDLE/'frozen_data/roots.jsonl'):
        rid = root['root_id']; folder = destination/rid
        if len(grouped['questions'][rid]) != 34 or len(grouped['labels'][rid]) != 272:
            raise ValueError('Question/label inventory')
        for name in grouped:
            value = grouped[name][rid]
            if name == 'source_inputs':
                value = [{k:r[k] for k in ('actor','origin','source_text','commands')} for r in value]
            save(folder/(name+'.json'), value)
        save(folder/'root.json',root)
        index.append(dict(root_id=rid,split=root['split'],question_count=34,
            source_count=16,files={p.name:sha(p) for p in sorted(folder.iterdir())}))
    # Chosen without outcomes; stable hash order, identical design across models.
    repeats = sorted((r['root_id'] for r in index if r['split']=='FINAL'),
        key=lambda rid: hashlib.sha256(rid.encode()).hexdigest())[:8]
    first_q = grouped['questions'][index[0]['root_id']]
    selected = []
    for draw in (0,1):
        for kind in ('direct','same','both','either'):
            matches = [q for q in first_q if q['spec']['kind']==kind and f'-d{draw}-' in q['query_id']]
            if len(matches)==0: raise ValueError('Repeatability query family absent')
            selected.append(matches[0]['query_id'].removeprefix(index[0]['root_id']))
    save(destination/'INDEX.json', index)
    save(destination/'REPEATABILITY.json', dict(roots=repeats,query_suffixes=selected,
        root_selection='lowest sha256 of UTF8 root_id',question_selection='first original query per direct/same/both/either kind and answer-code draw',
        conditions=['NATIVE_FINAL',*[c for c in CONDITIONS if c.endswith(('_s0','_s1'))]],
        origins=list(range(8)),passes=2,batch_size=1,selected_before_outcomes=True))
    return index

def scenes_in(value):
    if isinstance(value, dict):
        if {'actors','projects','values','target_actor','target_project','layout','scene_id','split'} <= set(value):
            yield value
        else:
            for item in value.values(): yield from scenes_in(item)
    elif isinstance(value, list):
        for item in value: yield from scenes_in(item)

def collision_check(paths):
    from canonical.candidate import semantic_hash
    from canonical.scenes import scene_from_dict
    history = set(load(BUNDLE/'inputs/PRIOR_SOURCE_HASHES.json')['sha256'])
    initial = len(history); inspected = []
    for path in sorted(set(map(Path, paths))):
        n = 0
        objects = rows(path) if path.suffix == '.jsonl' else [load(path)]
        for obj in objects:
            for scene in scenes_in(obj):
                history.add(semantic_hash(scene_from_dict(scene))); n += 1
        inspected.append(dict(path=str(path),sha256=sha(path),source_records=n))
    fresh = [semantic_hash(scene_from_dict(s)) for r in rows(BUNDLE/'frozen_data/roots.jsonl') for s in r['sources']]
    if len(set(fresh)) != 576 or set(fresh)&history: raise ValueError('Source collision; deterministic replacement required before outcomes')
    return dict(status='PASS', initial_historical_sources=initial, union_historical_sources=len(history),
        fresh_sources=576,collisions=[],inspected=inspected,historical_union_sha256=digest(sorted(history)),model_outcomes_used=False)
