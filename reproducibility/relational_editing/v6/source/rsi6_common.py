"""V6 evaluator data; the production writer remains V5's minimal WriterRequest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import rfr5_common as C
import rfr5_forward as V5

ROOT = Path(__file__).resolve().parents[1]
SUPPLIED = ROOT
sys.path.insert(0, str(SUPPLIED / 'reference'))
import source_independence as R

PANELS = ('A', 'B', 'C')
PROGRAMS = ('single', 'noop', 'repeat2', 'repeat4', 'repeat8', 'restore',
            'restore_after4', 'overwrite', 'AB', 'BA', 'ABC', 'CBA', 'AB_cross')
SIZES = {'A': {'gemma': 64, 'qwen': 32}, 'B': {'gemma': 64, 'qwen': 32},
         'C': {'gemma': 32, 'qwen': 16}}
CONDITIONS = tuple(['SOURCE', 'NATIVE_FINAL', 'TEXT_CORRECTION'] +
                   [f'{arm}_s{seed}' for arm in (*C.COMPLETE, 'CANONICAL_CURRENT')
                    for seed in (0, 1)])


@dataclass(frozen=True)
class Case:
    """Evaluator-only. Never passed to compile_writer or SetterBank."""
    panel: str
    scene: object
    terminal: object
    commands: tuple
    program: str
    order: str = 'early'
    root_id: str | None = None
    origin: int | None = None

    @property
    def key(self):
        return f'{self.scene.scene_id}-{self.program}-{self.order}'

    @property
    def identity(self):
        return R.stable_hash({'panel': self.panel, 'scene': asdict(self.scene),
                              'terminal': asdict(self.terminal),
                              'commands': [asdict(e) for e in self.commands],
                              'program': self.program, 'order': self.order,
                              'root_id': self.root_id, 'origin': self.origin})


def conditions(panel):
    if panel not in PANELS:
        raise ValueError('Undeclared panel')
    return tuple(V5.condition_names()) if panel == 'A' else CONDITIONS


def cases_for_scene(panel, scene):
    if panel == 'A':
        blocks = V5.blocks(scene)
    elif panel == 'C':
        if tuple(C.Q.programs(scene)) != PROGRAMS:
            raise ValueError('Inherited program inventory drift')
        blocks = [(p, 'early') for p in PROGRAMS]
    else:
        raise ValueError('Panel B requires explicit terminal commands')
    return [Case(panel, scene, C.Q.final_world(scene, program),
                 tuple(C.Q.programs(scene)[program]), program, order)
            for program, order in blocks]


def cases_for_root(record):
    terminal = C.Q.scene_from_dict(record['terminal'])
    sources = [C.Q.scene_from_dict(s) for s in record['sources']]
    commands = tuple(C.Q.OLD.Edit(**e) for e in record['commands'])
    expected = R.commands(C.Q.OLD, terminal)
    if commands != tuple(expected) or record['origin_masks'] != [list(m) for m in R.ORIGIN_MASKS]:
        raise ValueError('Explicit terminal commands/origin masks changed')
    if record['root_id'] != terminal.scene_id or len(sources) != 4:
        raise ValueError('Root identity or origin cardinality changed')
    expected_sources = R.origins(C.Q.OLD, terminal)
    for source, expected_source in zip(sources, expected_sources, strict=True):
        if source != expected_source:
            raise ValueError('Declared source does not match fixed relative mask')
    return [Case('B', s, terminal, commands, 'SET_TERMINAL_ABC',
                 root_id=terminal.scene_id, origin=i) for i, s in enumerate(sources)]


def load_cases(panel, actor, prepared=ROOT / 'data'):
    if actor not in C.SIZES or panel not in PANELS:
        raise ValueError('Undeclared actor/panel')
    if panel == 'B':
        records = C.read_rows(SUPPLIED / f'data/{actor}_source_roots.jsonl')
        units = [cases_for_root(r) for r in records]
    else:
        path = (Path(prepared) / f'{actor}_scenes.jsonl' if panel == 'A'
                else SUPPLIED / f'data/{actor}_program_subset.jsonl')
        scenes = [C.Q.scene_from_dict(r) for r in C.read_rows(path)]
        units = [cases_for_scene(panel, s) for s in scenes]
        if panel == 'C':
            original = {r['scene_id']: r for r in C.read_rows(Path(prepared) / f'{actor}_scenes.jsonl')}
            if any(asdict(s) != asdict(C.Q.scene_from_dict(original[s.scene_id])) for s in scenes):
                raise ValueError('Panel C scene is not the unchanged prepared A scene')
    if len(units) != SIZES[panel][actor]:
        raise ValueError('Complete panel population required')
    flat = [c for unit in units for c in unit]
    if len({c.key for c in flat}) != len(flat):
        raise ValueError('Duplicate case identity')
    return flat


def queries(case, reader):
    """Constructed only AFTER the actor/panel master seal has been committed."""
    if case.panel == 'A':
        return C.queries(case.scene, case.program, reader)
    if case.panel == 'B':
        return [q for draw in (0, 1)
                for q in R.questions(C, case.terminal, case.scene, reader, draw)]
    out = []
    with C.fresh_seed():
        for draw in (0, 1):
            for original in C.Q.program_queries(case.scene, case.program, draw):
                q = dict(original)
                spec = q['spec']
                q['text'] = C.apply_reader(q['text'], q['family'],
                    case.scene.actors[spec['a']], case.scene.actors[spec['b']],
                    case.scene.projects[spec['p']], reader)
                q.update(reader=reader, draw=draw)
                out.append(q)
    return out


def query_record(case, q):
    for field, world in (('gold', case.terminal), ('source_gold', case.scene)):
        if q[field] != R.truth(world.values, q['spec']):
            raise ValueError('Production labels disagree with independent truth')
    if bool(q['changed']) != (q['gold'] != q['source_gold']):
        raise ValueError('Changed flag drift')
    rec = C.Q.query_record(case.scene, q, 'FINAL', q['draw'], case.order, 0, case.program)
    rec.update(panel=case.panel, root_id=case.root_id or case.scene.scene_id,
               origin=case.origin, case_identity=case.identity,
               commands=[asdict(e) for e in case.commands], spec=dict(q['spec']))
    if case.panel == 'B':
        # Historical `direction` assumes a single source-relative flip. B has
        # three terminal SETs including no-ops; do not silently use that field.
        rec['direction'] = None
        rec['command_directions'] = [[int(case.scene.values[e.actor][e.project]), int(e.value)]
                                     for e in case.commands]
    if q['family'] in ('same', 'both', 'either'):
        a, b, p = (q['spec'][k] for k in ('a', 'b', 'p'))
        rec['truth_cell'] = [case.scene.values[a][p], case.scene.values[b][p],
                             case.terminal.values[a][p], case.terminal.values[b][p]]
    return rec


def verify_data(prepared=ROOT / 'data'):
    manifest = C.load(SUPPLIED / 'DATA_MANIFEST.json')
    for row in manifest['files']:
        path = SUPPLIED / row['path']
        if path.stat().st_size != row['bytes'] or C.sha(path) != row['sha256']:
            raise ValueError('Supplied V6 population digest mismatch')
    frozen = C.load(Path(prepared) / 'MANIFEST.json')
    for name, digest in frozen['files'].items():
        if C.sha(Path(prepared) / name) != digest:
            raise ValueError('Original prepared V5 digest mismatch')
    inventories = {p: {a: len(load_cases(p, a, prepared)) for a in C.SIZES} for p in PANELS}
    return {'data_manifest_sha256': C.sha(SUPPLIED / 'DATA_MANIFEST.json'),
            'prepared_manifest_sha256': C.sha(Path(prepared) / 'MANIFEST.json'),
            'case_counts': inventories, 'fresh_outcomes': 0}
