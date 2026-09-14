# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""Frozen V5 protocol. CPU-only; historical code is imported, never modified.

Only equality question wording changes. Labels and worlds are evaluator-only.
The writer receives a separate immutable prefix/addresses/values request.
"""
from __future__ import annotations

import contextlib
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcc4_common as V4

Q = V4.Q
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/relational_frozen_readout_v5'
INPUT = REPORT / 'supplied/V4_REVIEW'
SEED = 260912501
READERS = ('R0', 'R1', 'R2')
PROGRAMS = ('single', 'AB', 'ABC')
COMPLETE = ('INV_COMPLETE_SINGLE', 'FREE_COMPLETE_SINGLE')
PROCEDURES = {'gemma': 'C1', 'qwen': 'T4'}
SIZES = {'gemma': 64, 'qwen': 32}
BOOT_SEED = 260912502
BOOT_DRAWS = 10000
PRIMARY_LEVEL = .99375
FREEZE_HASHES = {
    'gemma': '5f8dcccda4c21a0e1f83f9450b800eccf8285e9c1d382b0b2af95adbf0fac454',
    'qwen': '28c533cbd067982f8efcdd2eb8924b4d60c0eac4c7c59495f659f1a3debc55ae',
}
V4_STORE = Path(os.environ.get('RFR5_V4_STORE', 'external-artifacts'))
V3_STORE = Path(os.environ.get('RFR5_V3_STORE', 'external-artifacts'))
TEMPLATES = {
    'R0': 'Do {A} and {B} take the same side on {project}?',
    'R1': 'Do {A} and {B} hold matching positions regarding {project}?',
    'R2': 'Do {A} and {B} hold matching positions regarding {project}? The answer is yes if both support it or both oppose it, and no if one supports it and the other opposes it.',
}
sha = Q.sha
dump = Q.dump
load = Q.load
write_rows = Q.write_rows
read_rows = Q.read_rows
now = Q.now


@contextlib.contextmanager
def fresh_seed():
    previous = Q.OLD.SEED
    Q.OLD.SEED = SEED
    try:
        yield
    finally:
        Q.OLD.SEED = previous


def source_hash(scene):
    return hashlib.sha256(Q.OLD.render(scene)[0].encode('utf8')).hexdigest()


def earlier_sources():
    """Read all supplied V2/V3/V4 scene pools, never their model outcomes."""
    sources = {}
    for version, getter in [('V2', Q.S.scenes), ('V3', Q.scenes), ('V4', V4.scenes)]:
        for split in ('FIT', 'CAL', 'FINAL', 'WORKFLOW'):
            for scene in getter(split):
                sources.setdefault(source_hash(scene), []).append(version + ':' + scene.scene_id)
    return sources


def make_population():
    # Fixed candidate count and hash rule, before model outcomes. Never grow a
    # population to seek successful answers. Duplicate-source removal is allowed.
    with fresh_seed():
        candidates = [replace(s, scene_id=s.scene_id.replace('SRS2', 'RFR5'))
                      for s in Q.OLD.make_scenes('FINAL', 128)]
    earlier = earlier_sources()
    seen = set(earlier)
    kept, removed = [], []
    for s in candidates:
        h = source_hash(s)
        if h in seen:
            removed.append({'scene_id': s.scene_id, 'source_sha256': h})
        else:
            kept.append(s)
            seen.add(h)
    gemma = Q.subset(kept, 64, 'RFR5-GEMMA-FIXED')
    qwen = Q.subset(gemma, 32, 'RFR5-QWEN-FIXED')
    for actor, selected in [('gemma', gemma), ('qwen', qwen)]:
        if len(selected) != SIZES[actor]:
            raise ValueError('Insufficient unique balanced scenes; do not reduce coverage')
        counts = {}
        for s in selected:
            counts[Q.cell(s)] = counts.get(Q.cell(s), 0) + 1
        if len(counts) != 8 or len(set(counts.values())) != 1:
            raise ValueError('Scene size/direction balance failed')
    return {'gemma': gemma, 'qwen': qwen}, {
        'seed': SEED, 'namespace': 'RFR5', 'candidate_count': 128,
        'historical_unique_sources': len(earlier), 'duplicate_removals': removed,
        'selection': 'fixed SHA256 rank within eight size/project/direction cells; Qwen subset of Gemma',
        'outcome_filtering': False,
    }


def equality_text(reader, A, B, project):
    """The reader interface has names only: no scene, values, gold or program."""
    return TEMPLATES[reader].format(A=A, B=B, project=project)


def apply_reader(text, family, A, B, project, reader):
    if reader not in READERS:
        raise ValueError('Undeclared reader')
    if family != 'same':
        return text
    first, sep, tail = text.partition('\n')
    if first != equality_text('R0', A, B, project):
        raise ValueError('Original equality wording drift')
    return equality_text(reader, A, B, project) + sep + tail


def queries(scene, program, reader, *, include_additions=True):
    if program not in PROGRAMS:
        raise ValueError('Undeclared V5 program')
    out = []
    with fresh_seed():
        for draw in (0, 1):
            for original in Q.program_queries(scene, program, draw):
                q = dict(original)
                if q['additional'] and not include_additions:
                    continue
                spec = q['spec']
                q['text'] = apply_reader(q['text'], q['family'], scene.actors[spec['a']],
                                         scene.actors[spec['b']], scene.projects[spec['p']], reader)
                q['reader'] = reader
                q['draw'] = draw
                out.append(q)
    return out


@dataclass(frozen=True)
class WriterRequest:
    prefix_ids: tuple[int, ...]
    addresses: tuple[tuple[int, ...], ...]
    desired_values: tuple[int, ...]

    def __post_init__(self):
        if len(self.addresses) != len(self.desired_values):
            raise ValueError('Unmatched commands')
        for address, value in zip(self.addresses, self.desired_values):
            Q.OLD.token_mask(address, len(self.prefix_ids), 'clause')
            if value not in (0, 1):
                raise ValueError('SET needs a binary requested value')


def primary_inventory():
    return [{'actor': actor, 'architecture': arm, 'reader': reader, 'baseline': 'R0',
             'program': 'ABC', 'family': 'same', 'seeds': [0, 1], 'level': PRIMARY_LEVEL,
             'draws': BOOT_DRAWS, 'unit': 'whole scene; average both seeds inside scene'}
            for actor in SIZES for arm in COMPLETE for reader in READERS[1:]]


def verify_inputs(v4_store=V4_STORE, v3_store=V3_STORE):
    historical = V4.verify_bundle()
    manifest = load(INPUT / 'MANIFEST.json')
    if len(manifest['files']) != 39:
        raise ValueError('Review manifest count changed')
    for row in manifest['files']:
        p = INPUT / row['path']
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Supplied review mismatch: ' + row['path'])
    freezes = {}
    for actor, expected in FREEZE_HASHES.items():
        p = Path(v4_store) / f'outputs/rcc4/r1/{actor}/FREEZE.json'
        if sha(p) != expected or sha(INPUT / f'audit/{actor}_FREEZE.json') != expected:
            raise ValueError('Original FREEZE changed: ' + actor)
        freezes[actor] = load(p)
    checked = []
    with (INPUT / 'next_step/SELECTED_CHECKPOINTS.csv').open(encoding='utf-8-sig', newline='') as f:
        selections = list(csv.DictReader(f))
    if len(selections) != 16:
        raise ValueError('Need all 16 V4 selections')
    for row in selections:
        path = Path(v4_store) / row['remote_relative']
        meta = load(path.with_name('META.json'))
        if path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
            raise ValueError('Original setter mismatch: ' + str(path))
        freeze = freezes[row['actor']]
        if int(freeze['selection'][row['method']][row['seed']]['checkpoint']) != int(row['checkpoint']):
            raise ValueError('Original CAL selection mismatch')
        if meta['files']['setter'] != row['sha256']:
            raise ValueError('Metadata setter digest mismatch')
        checked.append({'actor': row['actor'], 'condition': row['method'], 'seed': int(row['seed']),
                        'path': str(path.parent), 'setter_sha256': row['sha256'],
                        'meta_sha256': sha(path.with_name('META.json')), 'origin': 'V4'})
    for actor, freeze in freezes.items():
        for arm, seeds in freeze['v3_warm_starts'].items():
            for seed, ref in seeds.items():
                relative = ref['dir'][ref['dir'].index('outputs/'):]
                path = Path(v3_store) / relative
                if sha(path / 'setter.npz') != ref['sha256']['setter'] or sha(path / 'META.json') != ref['sha256']['meta']:
                    raise ValueError('Historical V3 factor mismatch')
                checked.append({'actor': actor, 'condition': 'FROZEN_V3_' + arm, 'seed': int(seed),
                                'path': str(path), 'setter_sha256': ref['sha256']['setter'],
                                'meta_sha256': ref['sha256']['meta'], 'origin': 'V3'})
    return {'review_payloads': 39, 'historical_bundle': historical, 'checkpoints': checked, 'freezes': freezes,
            'freeze_hashes': FREEZE_HASHES, 'protocol_sha256': sha(INPUT / 'next_step/RELATIONAL_FROZEN_READOUT_V5.md')}


def prepare(output, v4_store=V4_STORE, v3_store=V3_STORE):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Prepared protocol is immutable; choose a new revision')
    binding = verify_inputs(v4_store, v3_store)
    populations, dedup = make_population()
    output.mkdir(parents=True)
    for actor, scenes in populations.items():
        write_rows(output / (actor + '_scenes.jsonl'), [asdict(s) for s in scenes])
    dump(output / 'INPUT_BINDING.json', binding)
    dump(output / 'DEDUP.json', dedup)
    dump(output / 'READERS.json', TEMPLATES)
    dump(output / 'PRIMARY_CONTRASTS.json', primary_inventory())
    dump(output / 'PROTOCOL.json', {
        'status': 'PREPARED_NOT_EXECUTED', 'seed': SEED, 'sizes': SIZES,
        'programs': list(PROGRAMS), 'orders': ['early', 'late'], 'readers': list(READERS),
        'singles_original_questions': 24, 'program_touched_coverage': True,
        'native_procedures': PROCEDURES, 'models': V4.MODELS, 'rank': 16,
        'fitting': False, 'reader_adaptation': False, 'checkpoint_selection': False,
        'compile_before_queries': True, 'future_query_files': [],
        'claim_boundary': 'assisted reader when R2 succeeds; historical V4 unchanged; no full F2/F3/F4 award',
    })
    dump(output / 'MANIFEST.json', {'files': {p.name: sha(p) for p in sorted(output.iterdir())}})
    return {'status': 'PREPARED_NOT_EXECUTED', 'directory': str(output), 'manifest_sha256': sha(output / 'MANIFEST.json')}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepare), indent=2))
