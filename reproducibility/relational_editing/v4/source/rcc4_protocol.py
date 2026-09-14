"""RCC4 fresh scene pools (FIT 96 / CAL 48 / FINAL 128 / WORKFLOW 128) under the explicit fresh namespace/seed 260911398.

The renderer, semantic SET algebra, question specifications and edit programs are the immutable V3 copies (`reference/v2_protocol.py`,
`reference/protocol.py` of the RSO3 supplied bundle).  Exactly as the V3 protocol bound its own seed at runtime (`old.SEED = SEED`), this
module binds the V4 seed to the same module object so that scene draws and answer-mapping draws of every RCC4 scene come from the V4
namespace.  Export performs exact source-text deduplication against every supplied earlier pool (V2 SRS2 and V3 RSO3, all four splits)
BEFORE any model outcome exists; it is a data check, never an outcome filter.  No model inference here.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

SEED = 260911398
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rso3_common as Q                          # noqa: E402  binds the V3 supplied protocol; its `OLD` is the shared v2_protocol module object
OLD = Q.OLD; P3 = Q.P3
OLD.SEED = SEED; P3.SEED = SEED                  # runtime seed binding, identical mechanism to the V3 protocol
Scene = OLD.Scene; Edit = OLD.Edit; render = OLD.render; apply_semantics = OLD.apply_program

TAG = 'RCC4'
SPLITS = (('FIT', 96), ('CAL', 48), ('FINAL', 128), ('WORKFLOW', 128))

def check_seed():
    if OLD.SEED != SEED or P3.SEED != SEED: raise RuntimeError('RCC4 seed binding lost')
    return SEED

def scenes(split, n):
    check_seed(); return [replace(s, scene_id=s.scene_id.replace('SRS2', TAG)) for s in OLD.make_scenes(split, n)]

def source_hash(s): return hashlib.sha256(render(s)[0].encode()).hexdigest()

def earlier_pools():
    """Rendered-source hashes of every supplied earlier scene pool: V2 (SRS2 supplied data) and V3 (RSO3 supplied data)."""
    pools = {}
    for name, sc in (('V2_SRS2', Q.S.scenes), ('V3_RSO3', Q.scenes)):
        for split in ('FIT', 'CAL', 'FINAL', 'WORKFLOW'):
            for s in sc(split): pools[source_hash(s)] = f'{name}:{s.scene_id}'
    return pools

def export(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); files = {}; seen = {}; earlier = earlier_pools(); collisions = []
    for split, n in SPLITS:
        data = scenes(split, n)
        for s in data:
            sig = source_hash(s)
            if sig in seen: raise ValueError(f'duplicate source inside RCC4: {s.scene_id} == {seen[sig]}')
            if sig in earlier: collisions.append(dict(scene_id=s.scene_id, earlier=earlier[sig]))
            seen[sig] = s.scene_id
        p = root/f'{split.lower()}_scenes.jsonl'; p.write_text(''.join(json.dumps(asdict(s), sort_keys=True)+'\n' for s in data), encoding='utf8', newline='\n')
        files[p.name] = {'scenes': n, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    if collisions: raise ValueError('RCC4 scene collides with an earlier pool: '+json.dumps(collisions[:5]))
    dedup = dict(seed=SEED, rcc4_scenes=len(seen), earlier_pool_scenes=len(earlier), earlier_pools=sorted({v.split(':')[0] for v in earlier.values()}), collisions=0,
                 rule='exact rendered-source-text sha256 equality; checked within RCC4 and against every V2/V3 split before any model outcome; no outcome-based filtering',
                 rcc4_source_hashes={sid: h for h, sid in seen.items()})
    (root/'MANIFEST.json').write_text(json.dumps({'seed': SEED, 'files': files, 'programs': list(Q.PROGRAMS), 'question_specs': dict(sparse=6, complete=12, per_draw=True, draws=2)}, indent=2)+'\n', encoding='utf8', newline='\n')
    (root/'DEDUP.json').write_text(json.dumps(dedup, indent=1, sort_keys=True)+'\n', encoding='utf8', newline='\n')
    return files

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--export', required=True); print(json.dumps(export(p.parse_args().export), indent=1))
