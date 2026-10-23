"""Small, source-bound adapter around the submitted CPU analysis functions."""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / 'history_updates'
LEGACY = 'studies/original_public_history/'
FRESH = LEGACY + 'reproducibility/fresh_history_validation/'
CASE = 'FHV1-FINAL-0017'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def release_api():
    with path_context(RELEASE):
        return importlib.import_module('experiments._shared.release')


@contextlib.contextmanager
def path_context(path):
    original = sys.path[:]
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


def verify_bundle(folder, source=None, native_source=None):
    """Closed compact inventory; optionally compare every export to original bytes."""
    folder = Path(folder).resolve()
    api = release_api()
    binding = load(folder / 'BINDINGS.json')
    if binding['schema'] != 'COLLABORATOR_EXAMPLES_V1':
        raise ValueError('Unknown example binding')
    if digest(RELEASE / 'metadata/study_index.json') != binding['study_index_sha256']:
        raise ValueError('Changed submitted resolver')
    resolver = load(RELEASE / 'metadata/study_index.json')
    seen = set()
    for name, row in binding['records'].items():
        if name.casefold() in seen:
            raise ValueError('Duplicate example member')
        seen.add(name.casefold())
        path = api.safe(folder, name)
        if not path.is_file() or digest(path) != row['export_sha256']:
            raise ValueError('Missing or changed example input: ' + name)
        if row.get('source_kind') != 'native':
            namespace, logical = row['logical_source'].split('/', 1)
            declared = resolver['mounts'][namespace][logical]
            if row['source'] != declared:
                raise ValueError('Example source differs from submitted resolver')
            expected = declared.get('materialized_sha256', declared['sha256'])
            if row['export_sha256'] != expected:
                raise ValueError('Export differs from exact source bytes')
        origin = native_source if row.get('source_kind') == 'native' else source
        if origin is not None:
            original = api.safe(Path(origin).resolve(), row['source']['included'])
            if not original.is_file() or digest(original) != row['source']['sha256']:
                raise ValueError('Missing or changed original input: ' + row['source']['included'])
            if 'selection' in row:
                selection = row['selection']
                projected = [u for u in load(original)[selection['field']]
                             if u['case_id'] in selection['case_ids']]
                if load(path) != projected:
                    raise ValueError('Compact projection differs from original records')
    for name, row in binding.get('derived', {}).items():
        path = api.safe(folder, name)
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError('Missing or changed derived input: ' + name)
    actual = set(api.inventory(folder))
    if actual != set(binding['records']) | set(binding.get('derived', {})) | {'BINDINGS.json'}:
        raise ValueError('Undeclared example input')
    return binding


def mount_code(destination):
    """Restore only the eleven small modules used here, not the full evidence tree."""
    index = load(RELEASE / 'metadata/study_index.json')
    names = [FRESH + 'source/' + name for name in
             ('analyze.py', 'bindings.py', 'panels.py', 'replay.py', 'decision.py')]
    names += [LEGACY + 'src/repro/' + name for name in
              ('mqmh.py', 'common.py', 'release_inputs.py', '__init__.py')]
    names += ['src/repro/example.py',
              'studies/repair_qwen7/input/PPR1_Same_Case_Public_Repair/vendor/ranker_contract.py']
    selected = {'mounts': {'v3': {name: index['mounts']['v3'][name] for name in names}}}
    return release_api().mount(selected, destination, RELEASE)


@contextlib.contextmanager
def scoring_modules(source):
    """Isolate legacy absolute imports from the older repository's repro package."""
    names = ('analyze', 'bindings', 'panels', 'replay', 'decision')
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    package = types.ModuleType('collaborator_public')
    package.__path__ = [str(source / 'v3' / LEGACY / 'src/repro')]
    sys.modules['collaborator_public'] = package
    try:
        with path_context(source / 'v3' / FRESH / 'source'):
            yield (importlib.import_module('analyze'), importlib.import_module('panels'),
                   importlib.import_module('replay'),
                   importlib.import_module('collaborator_public.mqmh'))
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for name in list(sys.modules):
            if name == 'collaborator_public' or name.startswith('collaborator_public.'):
                del sys.modules[name]


def forbid_inference_and_network():
    import importlib.abc

    class NoModels(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub',
                                         'lambda_api', 'operator_adapter'}:
                raise ImportError('Saved-output examples do not load model libraries')

    def offline(event, args):
        if event.startswith('socket.') and event != 'socket.gethostname':
            raise RuntimeError('Saved-output examples do not use the network')

    sys.meta_path.insert(0, NoModels())
    sys.addaudithook(offline)
