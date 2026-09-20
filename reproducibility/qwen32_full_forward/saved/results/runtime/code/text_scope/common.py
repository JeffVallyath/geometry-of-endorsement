"""Study-local paths and immutable JSON; no provider or inference side effects."""
from __future__ import annotations
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

DEFAULT_RUNTIME = Path(os.environ.get('V10_TEXT_SCOPE_RUNTIME', 'D:/geometry-of-endorsement/v10-text-scope-runtime-20260918'))
OPERATOR_LIVE = Path(os.environ.get('V10_TEXT_SCOPE_LIVE', str(DEFAULT_RUNTIME/'operator/live')))
METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')
CONTEXTS = ('NATIVE_FINAL', *(f'o{o}/{m}' for o in range(8) for m in METHODS))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('IMMUTABLE_ARTIFACT_DRIFT:' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def source_root(runtime):
    return Path(runtime) / 'source' / 'geometry-of-endorsement'


def reproduction(runtime):
    return source_root(runtime) / 'reproducibility/fresh_inference/supplied/reproduction'


def shards(runtime):
    return reproduction(runtime) / 'reports/state_sufficiency_confirmation/shards'


def setup_imports(runtime):
    root = reproduction(runtime)
    paths = (root / 'scripts', root / 'STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS')
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return paths


def independent_evaluator(runtime):
    path = source_root(runtime) / 'reproducibility/state_sufficiency/independent_verification/reconstruct.py'
    spec = importlib.util.spec_from_file_location('text_scope_inherited_reconstruct', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(runtime):
    result = read(Path(runtime) / 'handoff/SCHEDULE.json')
    assert result['methods'] == list(METHODS)
    assert result['counts']['all_planned_score_evaluations'] == 160136
    assert result['counts']['new_final_scores'] == 147968
    assert len(result['final_root_ids']) == 64 and len(CONTEXTS) == 17
    return result
