"""Portable, explicit one-root neural reproduction entry point.

Default: CPU source/input/checkpoint smoke only. --execute is an optional fresh
forward path, not exercised or certified by the saved-score integration pass.
The original scientific compile/seal/serve implementation is reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
SUPPLIED = HERE / 'supplied'
REPRODUCTION = SUPPLIED / 'reproduction'
PUBLIC = REPRODUCTION / 'STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS'
SHARDS = REPRODUCTION / 'reports/state_sufficiency_confirmation/shards'


def verify():
    manifest = json.loads((SUPPLIED / 'MANIFEST.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        path = (SUPPLIED / row['path']).resolve()
        if not path.is_relative_to(SUPPLIED.resolve()):
            raise ValueError('Escaping scientific input')
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Scientific input changed: ' + row['path'])
    return manifest


def modules():
    # Only this process receives the preserved scientific import layout.
    sys.path[:0] = [str(REPRODUCTION / 'scripts'), str(PUBLIC)]
    return importlib.import_module('ssc1_eval'), importlib.import_module('inputs')


def smoke(root_id='SSC1-FINAL-0006'):
    manifest = verify()
    evaluation, inputs = modules()
    import numpy as np
    from canonical.config import MODELS
    from ssc1_boundary import ReadBoundary
    sources = json.loads((SHARDS / root_id / 'source_inputs.json').read_text(encoding='utf8'))
    for actor in ('gemma', 'qwen'):
        if not re.fullmatch('[0-9a-f]{40}', MODELS[actor]['revision']):
            raise ValueError('Unpinned model')
        rows = sorted((r for r in sources if r['actor'] == actor), key=lambda r:r['origin'])
        if [r['origin'] for r in rows] != list(range(8)):
            raise ValueError('Missing source history')
        rebuilt = []
        for row in rows:
            commands = inputs.commands_from_json(row['commands'])
            inputs.validate(row['source_text'], commands)
            rebuilt.append(inputs.rewrite(row['source_text'], commands, 'REBUILD'))
            for method in ('SOURCE', 'EXISTING_CORRECTION', 'LATEST_SAME_WORDING'):
                inputs.prepare(row['source_text'], commands, method=method, actor=actor)
        if len(set(rebuilt)) != 1 or rebuilt[0] != rows[0]['source_text']:
            raise ValueError('Current facts do not match across histories')
    selected = inputs.inventory()['selected']
    for record in selected:
        folder = PUBLIC / record['path']
        for name, key in (('setter.npz', 'setter_sha256'), ('META.json', 'meta_sha256')):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != record[key]:
                raise ValueError('Selected checkpoint changed')
        with np.load(folder / 'setter.npz', allow_pickle=False) as weights:
            for key in ('basis_raw', 'basis_realized', 'head', 'bias', 'center'):
                if not np.isfinite(weights[key]).all():
                    raise ValueError('Nonfinite learned parameters')
            basis = weights['basis_realized'].astype('float64')
            if not np.allclose(basis.T @ basis, np.eye(16), atol=2e-6, rtol=0):
                raise ValueError('Saved realized basis invalid')
    if 'transformers' in sys.modules or ('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()):
        raise RuntimeError('CPU smoke must not initialize CUDA or import model loaders')
    return dict(status='CPU_SCIENTIFIC_PACKAGE_SMOKE_PASS', bound_files=len(manifest['files']),
        checkpoints=len(selected), root_id=root_id, origins_per_model=8,
        source_imports=True, neural_forward_passes=0,
        neural_readiness='NOT_VERIFIED: no pretrained smoke was run')


def execute(actor, root_id, cache, output):
    verify()
    if not output or output.exists():
        raise ValueError('Fresh output directory required')
    repository = HERE.parents[1].resolve()
    output = output.resolve()
    if output.is_relative_to(repository) and not output.is_relative_to(repository / 'reproduced'):
        raise ValueError('In-repository neural outputs must be under reproduced')
    # Preserve the actual completed runtime pins, rather than silently running
    # with the unrelated public CPU replay environment.
    import torch
    import transformers
    import numpy as np
    if (sys.version.split()[0], torch.__version__, transformers.__version__) != ('3.12.3', '2.11.0+cu128', '4.56.2'):
        raise ValueError('Original neural Python/torch/transformers versions required')
    if np.__version__ != '2.2.6':
        raise ValueError('Original neural numpy version required')
    evaluation, inputs = modules()
    from canonical.adapter import Backbone
    from canonical.config import MODELS
    from ssc1_boundary import ReadBoundary
    from ssc1_common import save
    from canonical.base_adapter import validate_cache
    validate_cache(cache, MODELS[actor])  # exact revision, local-only retrieval
    output.mkdir(parents=True)
    with ReadBoundary(output / 'ENTRY_EVENTS.jsonl'):
        backbone = Backbone(cache=cache, cfg=MODELS[actor])
        try:
            result = evaluation.evaluate_root(backbone, actor, SHARDS, root_id, output / 'roots', chunk_size=1)
            save(output / 'UNCHANGED.json', backbone.finish())
        finally:
            for handle in backbone.handles:
                handle.remove()
    save(output / 'REPRODUCTION_SCOPE.json', dict(
        scope='Single completed-study root regenerated; not a new prospective experiment or full primary aggregate.',
        actor=actor,root_id=root_id,model=MODELS[actor],chunk_size=1,
        historical_scientific_code='ssc1_eval.evaluate_root, unchanged',
        original_full_study_technical_qualification='Not repeated by this one-root interface; no full-study certification.'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-id', default='SSC1-FINAL-0006')
    p.add_argument('--actor', choices=('gemma','qwen'), default='qwen')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not re.fullmatch(r'SSC1-(FINAL|DEV)-\d{4}', args.root_id) or not (SHARDS / args.root_id).is_dir():
        p.error('Choose an included frozen root ID')
    if args.execute and (args.cache is None or args.output is None):
        p.error('--execute requires a local --cache and fresh --output')
    result = execute(args.actor,args.root_id,args.cache,args.output) if args.execute else smoke(args.root_id)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
