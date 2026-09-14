"""Forward-only execution core. No training module is imported.

Two passes: seal all evaluation prefixes before constructing any future query;
then recompile/hash-check each master and serve fixed readers on clones.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import rfr5_common as C


def condition_names():
    return ['SOURCE', 'NATIVE_FINAL', 'TEXT_CORRECTION'] + [c + '_s' + str(s) for c in (*C.COMPLETE, 'FROZEN_V3_INVARIANT_SET', 'FROZEN_V3_FREE_OVERWRITE', 'CANONICAL_CURRENT') for s in (0, 1)]


def blocks(scene):
    # Early SINGLE is measured once, retaining its main/program interpretations.
    return [('single', 'early'), ('single', 'late'), ('AB', 'early'), ('ABC', 'early')]


def prefix_request(bb, scene, actor, program, order):
    text, spans = C.Q.prefix_text(scene, C.PROCEDURES[actor], order)
    commands = C.Q.programs(scene)[program]
    addresses = tuple(tuple(bb.clause_positions(text, spans[(e.actor, e.project)])) for e in commands)
    request = C.WriterRequest(tuple(bb.prefix_ids(text)), addresses, tuple(int(e.value) for e in commands))
    return text, request, commands


def compile_writer(bb, request, bank, *, canonical=False):
    """Only source tokens, addressed clause spans and requested bits enter."""
    addresses, values = request.addresses, request.desired_values
    if canonical:
        # Same normalizer as V3/V4, source backup + full command history only.
        from rso3_editor import canonical_program
        addresses, values = canonical_program(addresses, values)
    maps, union = [], set()
    for cp, value in zip(addresses, values, strict=True):
        idx = (bb.t.zeros(len(cp), dtype=bb.t.long, device=bb.device), bb.t.as_tensor(cp, dtype=bb.t.long, device=bb.device))
        maps.append({'kind': 'setter', 'fn': bank.fn(value, 1.0, record=False), 'index': idx, 'clause_positions': list(cp)})
        union.update(cp)
    return bb.compile(request.prefix_ids, maps=maps, index=sorted(union), site=bank.site, workspace='fp32', grad=False)


def load_banks(bb, actor, binding, v4_store=C.V4_STORE, v3_store=C.V3_STORE):
    from rso3_editor import SetterBank
    banks = {}
    for record in binding['checkpoints']:
        if record['actor'] != actor or (record['origin'] == 'V4' and record['condition'] not in C.COMPLETE):
            continue
        # Resolve from frozen original relative identities, not stale Windows
        # absolute paths in the prepared protocol on the remote host.
        if record['origin'] == 'V4':
            sel = binding['freezes'][actor]['selection'][record['condition']][str(record['seed'])]
            relative = sel['path'][sel['path'].index('outputs/'):]
            path = Path(v4_store) / relative
        else:
            arm = record['condition'].removeprefix('FROZEN_V3_')
            ref = binding['freezes'][actor]['v3_warm_starts'][arm][str(record['seed'])]
            path = Path(v3_store) / ref['dir'][ref['dir'].index('outputs/'):]
        if C.sha(path/'setter.npz') != record['setter_sha256'] or C.sha(path/'META.json') != record['meta_sha256']:
            raise ValueError('Runtime factor identity mismatch')
        bank = SetterBank.load(bb, path)
        bank.model.eval()
        for p in bank.model.parameters(): p.requires_grad_(False)
        if bank.actor != actor or bank.site != C.V4.MODELS[actor]['site'] or bank.seed != record['seed']:
            raise ValueError('Runtime factor actor/site/seed mismatch')
        # Preserve saved basis_realized used in V4; do not recompute it.
        banks[record['condition'] + '_s' + str(record['seed'])] = bank
    if len(banks) != 8:
        raise ValueError('Need both constructions and both V3 references, two seeds')
    for seed in (0, 1):
        chosen = binding['freezes'][actor]['canonical_current'][str(seed)]['condition']
        banks['CANONICAL_CURRENT_s' + str(seed)] = banks[chosen + '_s' + str(seed)]
    return banks


def compile_condition(bb, scene, actor, program, order, name, banks):
    text, request, commands = prefix_request(bb, scene, actor, program, order)
    if name == 'NATIVE_FINAL':
        world = C.Q.final_world(scene, program)
        text, _ = C.Q.prefix_text(scene, C.PROCEDURES[actor], order, world=world)
    elif name == 'TEXT_CORRECTION':
        text, _ = C.Q.prefix_text(scene, C.PROCEDURES[actor], order, corrections=commands)
    if name in ('SOURCE', 'NATIVE_FINAL', 'TEXT_CORRECTION'):
        art = bb.compile(bb.prefix_ids(text), grad=False)
    else:
        art = compile_writer(bb, request, banks[name], canonical=name.startswith('CANONICAL_CURRENT'))
    return text, art


def seal_all(bb, actor, scenes, banks, output, check_time):
    output = Path(output)
    if (output/'SEAL.json').exists():
        seal = C.load(output/'SEAL.json')
        if C.sha(output/'SEAL.jsonl') != seal['sha256']:
            raise ValueError('Seal digest mismatch')
        return C.read_rows(output/'SEAL.jsonl')
    rows = []
    for scene in scenes:
        for program, order in blocks(scene):
            for name in condition_names():
                check_time()
                text, art = compile_condition(bb, scene, actor, program, order, name, banks)
                rows.append({'scene_id': scene.scene_id, 'program': program, 'order': order,
                             'condition': name, 'artifact_hash': art['hash'], 'prefix_sha256': art['prefix_sha256'],
                             'prefix_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'cache_bytes': art['bytes'], 'prefix_tokens': art['prefix_len']})
                del art
    C.write_rows(output/'SEAL.jsonl', rows)
    C.dump(output/'SEAL.json', {'at': C.now(), 'sha256': C.sha(output/'SEAL.jsonl'), 'artifacts': len(rows),
                              'questions_constructed_before_seal': False})
    return rows


def measurement_key(art, suffix_text, suffix_ids, label_ids):
    return (art['hash'], art['prefix_sha256'], suffix_text, tuple(suffix_ids), tuple(label_ids))


def score_reader(bb, scene, actor, program, order, reader, name, text, art, cache, source, aliases):
    queries = C.queries(scene, program, reader, include_additions=True)
    encoded = []
    pending = {}
    for q in queries:
        suffix = C.Q.suffix_text(q['text'])
        enc = bb.encode(text, suffix)
        if list(enc['prefix_ids']) != list(art['prefix_ids']):
            raise ValueError('Reader changed the sealed prefix token boundary')
        labels = bb.label_ids(enc['full_text'], q['labels'])
        key = measurement_key(art, suffix, enc['suffix_ids'], labels)
        item = (q, suffix, enc, labels, key)
        encoded.append(item)
        if key not in cache: pending.setdefault(key, item)
    work = list(pending.values())
    for start in range(0, len(work), C.Q.CHUNK):
        chunk = work[start:start+C.Q.CHUNK]
        results = bb.ask_batch(art, [v[2]['suffix_ids'] for v in chunk], [v[3] for v in chunk], verify_master=True)
        if len(results) != len(chunk): raise ValueError('Scoring row count mismatch')
        for item, result in zip(chunk, results, strict=True):
            if not result.get('master_unchanged') or not result.get('finite', True):
                raise ValueError('Mutable master or nonfinite score')
            # Cache lifetime is one scene/program/order. Identical inputs
            # remeasured in another block are distinct executions (batch
            # composition may alter low-order floating-point scores).
            mid = hashlib.sha256(repr((actor, scene.scene_id, program, order, item[4])).encode()).hexdigest()
            cache[item[4]] = (dict(result), mid, (name, reader, item[0]['query_id']))
    rows = []
    world = C.Q.final_world(scene, program)  # evaluator-only; compilation finished
    for q, suffix, enc, labels, key in encoded:
        result, mid, original = cache[key]
        rec = C.Q.query_record(scene, q, 'FINAL', q['draw'], order, 0, program)
        source_key = (reader, q['query_id'])
        if name == 'SOURCE': source[source_key] = result
        sp = source[source_key]
        rec.update(actor=actor, condition=name, reader=reader, spec=dict(q['spec']),
                   question_text=q['text'], suffix_text=suffix, suffix_ids=enc['suffix_ids'], label_token_ids=labels,
                   prefix_text=text, prefix_sha256=art['prefix_sha256'], artifact_hash=art['hash'],
                   cache_bytes=art['bytes'], measurement_id=mid, prediction=result['prediction'], logps=result['logps'],
                   answer_mass=result['answer_mass'], argmax_in_labels=result['argmax_in_labels'], tie=result['tie'],
                   master_verified=result['master_unchanged'], source_prediction=sp['prediction'], source_logps=sp['logps'],
                   gold_logp=result['logps'][rec['gold']], canonical=name.startswith('CANONICAL_CURRENT'),
                   seed=int(name[-1]) if name.endswith(('_s0','_s1')) else None,
                   panel_aliases=['main','program'] if program == 'single' and order == 'early' else ['main' if order == 'late' else 'program'])
        if q['family'] == 'same':
            a,b,p = (q['spec'][k] for k in ('a','b','p'))
            rec['truth_cell'] = [scene.values[a][p], scene.values[b][p], world.values[a][p], world.values[b][p]]
        if original != (name, reader, q['query_id']):
            rec['measurement_alias_of'] = list(original)
            aliases.append({'scene_id': scene.scene_id, 'program': program, 'order': order, 'condition': name,
                            'reader': reader, 'query_id': q['query_id'], 'measurement_id': mid,
                            'original': list(original), 'artifact_hash': art['hash'], 'suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest()})
        if name.startswith('CANONICAL_CURRENT'):
            rec['privileges'] = 'source backup and command history; last requested bit per address; no gold'
        if art.get('realized'):
            rec['realized'] = art['realized']
        rows.append(rec)
    return rows


def evaluate(bb, actor, scenes, banks, output, check_time):
    output = Path(output)
    sealed = seal_all(bb, actor, scenes, banks, output, check_time)
    expected = {(r['scene_id'],r['program'],r['order'],r['condition']): r for r in sealed}
    for scene in scenes:
        for program, order in blocks(scene):
            tag = scene.scene_id + '-' + program + '-' + order
            done = output/'blocks'/ (tag + '.DONE.json')
            if done.exists():
                receipt = C.load(done)
                for fn, h in receipt['files'].items():
                    if C.sha(done.parent/fn) != h: raise ValueError('Saved block integrity failure')
                continue
            cache, source, aliases, rows = {}, {}, [], []
            started = time.monotonic()
            for name in condition_names():
                check_time()
                text, art = compile_condition(bb, scene, actor, program, order, name, banks)
                seal = expected[(scene.scene_id,program,order,name)]
                if art['hash'] != seal['artifact_hash'] or art['prefix_sha256'] != seal['prefix_sha256']:
                    raise ValueError('Sealed compilation changed')
                for reader in C.READERS:
                    rows.extend(score_reader(bb, scene, actor, program, order, reader, name, text, art, cache, source, aliases))
                del art
            path = output/'blocks'/(tag + '.jsonl')
            alias_path = output/'blocks'/(tag + '.ALIASES.jsonl')
            C.write_rows(path, rows)
            C.write_rows(alias_path, aliases)
            C.dump(done, {'at':C.now(), 'rows':len(rows), 'measurements':len(cache), 'aliases':len(aliases),
                          'seconds':time.monotonic()-started, 'files':{path.name:C.sha(path),alias_path.name:C.sha(alias_path)}})
    C.dump(output/'EVALUATION_COMPLETE.json', {'at':C.now(),'actor':actor,'scenes':len(scenes),'blocks':len(scenes)*4,'conditions':condition_names(),'readers':list(C.READERS)})
