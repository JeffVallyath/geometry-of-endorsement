"""Explicit-command V6 integration with the unmodified V5 neural writer.

Preserves V5 first-occurrence clone-serving: CHUNK=12, condition then reader
order, exact-input aliases within ONE case only. No cross-origin/program
aliasing is assumed. Every alias retains its canonical physical batch/slot.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

import rsi6_common as S

C, V5, R = S.C, S.V5, S.R
SERVING = {'schema': 'RSI6_SERVING_V1', 'scheduler': 'V5-first-occurrence',
           'chunk': C.Q.CHUNK, 'workspace': 'fp32', 'backbone': 'bf16',
           'readout': 'inherited-logits32-full-vocabulary-log-softmax',
           'alias_scope': 'one-case-canonical-physical-batch-slot'}


def serving_mode(bb):
    return json.loads(json.dumps({**SERVING, 'actor_config': dict(getattr(bb, 'cfg', {})),
            'pad_id': getattr(bb, 'pad_id', None),
            'device': str(getattr(bb, 'device', 'synthetic-cpu')),
            'runtime_binding': getattr(bb, 'rsi6_runtime_binding', None)}))


def tagged_nonfinite(value):
    """Lossless JSON evidence for failed raw floats, never primary answers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': value.hex()}
    if isinstance(value, dict):
        return {k: tagged_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tagged_nonfinite(v) for v in value]
    return value


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def physical_batch(bb, art, batch, *, journal=None, event=None):
    """Record actual outputs before validation; reuse only verified same-attempt bytes.

    An interrupted intent without a result receipt is ambiguous exposure, not
    permission to repeat an example. Such cases stop for technical adjudication.
    """
    if journal is None:
        return bb.ask_batch(art, batch['suffix_ids'], batch['label_ids'], verify_master=True)
    identity = {'event': event, 'batch': batch}
    event_id = R.stable_hash(identity)
    base = Path(journal) / event_id
    intent, raw, done = (base.with_suffix(s) for s in ('.INTENT.json', '.RAW.json', '.DONE.json'))
    if done.exists():
        receipt = C.load(done)
        if (receipt['event_id'] != event_id or C.sha(intent) != receipt['intent_sha256']
                or C.sha(raw) != receipt['raw_sha256'] or C.load(intent) != identity):
            raise ValueError('Physical journal identity/hash mismatch')
        result = C.load(raw)
        if not result['accepted']:
            raise ValueError('Preserved failed physical batch; no outcome retry')
        return result['results']
    if intent.exists() or raw.exists():
        raise ValueError('Ambiguous incomplete physical batch; do not rerun measured examples')
    durable_json(intent, identity)
    try:
        results = bb.ask_batch(art, batch['suffix_ids'], batch['label_ids'], verify_master=True)
    except Exception as error:
        durable_json(raw, {'accepted': False, 'exception_type': type(error).__name__,
                          'exception': str(error), 'results': None})
        durable_json(done, {'event_id': event_id, 'intent_sha256': C.sha(intent), 'raw_sha256': C.sha(raw)})
        raise
    accepted = len(results) == batch['batch_size'] and all(
        r.get('master_unchanged') and r.get('finite', True) and
        all(math.isfinite(v) for v in r.get('logps', [])) for r in results)
    durable_json(raw, {'accepted': accepted, 'results': tagged_nonfinite(results)})
    durable_json(done, {'event_id': event_id, 'intent_sha256': C.sha(intent), 'raw_sha256': C.sha(raw)})
    if not accepted:
        raise ValueError('Invalid physical batch preserved; no efficacy retry')
    return results


def explicit_request(bb, source_text, supplied_spans, commands):
    """Narrow bridge from supplied addresses to V5's immutable writer schema."""
    addresses = tuple(tuple(bb.clause_positions(source_text, supplied_spans[(e.actor, e.project)]))
                      for e in commands)
    return C.WriterRequest(tuple(bb.prefix_ids(source_text)), addresses,
                           tuple(int(e.value) for e in commands))


def compile_case(bb, case, actor, name, banks):
    if name not in S.conditions(case.panel):
        raise ValueError('Undeclared condition')
    text, spans = C.Q.prefix_text(case.scene, C.PROCEDURES[actor], case.order)
    if name == 'NATIVE_FINAL':
        text, _ = C.Q.prefix_text(case.scene, C.PROCEDURES[actor], case.order, world=case.terminal)
    elif name == 'TEXT_CORRECTION':
        text, _ = C.Q.prefix_text(case.scene, C.PROCEDURES[actor], case.order, corrections=case.commands)
    if name in ('SOURCE', 'NATIVE_FINAL', 'TEXT_CORRECTION'):
        return text, bb.compile(bb.prefix_ids(text), grad=False)
    # No scene, terminal world, query, origin, gold, or donor crosses this call.
    request = explicit_request(bb, text, spans, case.commands)
    art = V5.compile_writer(bb, request, banks[name], canonical=name.startswith('CANONICAL_CURRENT'))
    art['writer_request'] = {'schema': ['prefix_ids', 'addresses', 'desired_values'],
        'prefix_ids_sha256': R.stable_hash(request.prefix_ids),
        'addresses': [list(a) for a in request.addresses], 'desired_values': list(request.desired_values)}
    realized = art.get('realized')
    if realized:
        expected = len(case.commands)
        if name.startswith('CANONICAL_CURRENT'):
            from rso3_editor import canonical_program
            expected = len(canonical_program(request.addresses, request.desired_values)[0])
        if realized['maps'] != expected or realized['workspace'] != 'fp32':
            raise ValueError('Commands skipped/normalized or workspace changed')
    return text, art


def seal_all(bb, actor, cases, banks, output, check_time, run_identity):
    output = Path(output)
    binding = {'run_identity': run_identity, 'actor': actor,
               'cases': [c.identity for c in cases], 'serving': serving_mode(bb)}
    binding_hash = R.stable_hash(binding)
    if (output / 'SEAL.json').exists():
        seal = C.load(output / 'SEAL.json')
        if seal['binding_hash'] != binding_hash or C.sha(output / 'SEAL.jsonl') != seal['sha256']:
            raise ValueError('Resume seal or run identity mismatch')
        for name, digest in seal.get('witness_files', {}).items():
            if C.sha(output / name) != digest:
                raise ValueError('Physical witness integrity failure')
        rows = C.read_rows(output / 'SEAL.jsonl')
        required = {(c.key, n) for c in cases for n in S.conditions(c.panel)}
        if len(rows) != len(required) or {(r['case'], r['condition']) for r in rows} != required:
            raise ValueError('Seal inventory incomplete or duplicated')
        return rows
    rows, native_by_root, witnesses = [], {}, {}
    witness_roots = set(sorted({c.root_id or c.scene.scene_id for c in cases})[:4])
    for case in cases:
        for name in S.conditions(case.panel):
            check_time()
            text, art = compile_case(bb, case, actor, name, banks)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if case.panel == 'B' and name == 'NATIVE_FINAL':
                previous = native_by_root.setdefault(case.root_id, text_hash)
                if previous != text_hash:
                    raise ValueError('Different origins have different native terminal text')
            rows.append({'case': case.key, 'case_identity': case.identity, 'condition': name,
                         'artifact_hash': art['hash'], 'prefix_sha256': art['prefix_sha256'],
                         'prefix_text_sha256': text_hash, 'cache_bytes': art['bytes'],
                         'prefix_tokens': art['prefix_len'], 'commands': len(case.commands),
                         'realized': art.get('realized'), 'writer_request': art.get('writer_request')})
            if (case.root_id or case.scene.scene_id) in witness_roots and '_workspace_states' in art:
                path = output / 'witnesses' / f'{case.key}-{name}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as handle:
                    np.savez_compressed(handle, after_states=art['_after_states'],
                        workspace_states=art['_workspace_states'], positions=np.asarray(art['_index']),
                        prefix_ids=np.asarray(art['prefix_ids'], dtype='<i8'))
                witnesses[path.relative_to(output).as_posix()] = C.sha(path)
            del art
    C.write_rows(output / 'SEAL.jsonl', rows)
    C.dump(output / 'SEAL.json', {'at': C.now(), 'sha256': C.sha(output / 'SEAL.jsonl'),
        'binding': binding, 'binding_hash': binding_hash, 'artifacts': len(rows),
        'native_text_hashes_by_root': native_by_root, 'questions_constructed_before_seal': False,
        'witness_roots': sorted(witness_roots), 'witness_files': witnesses})
    return rows


def measurement_key(bb, art, suffix, suffix_ids, labels):
    return (R.stable_hash(serving_mode(bb)), *V5.measurement_key(art, suffix, suffix_ids, labels))


def score_reader(bb, case, actor, reader, name, text, art, cache, source, aliases, *, journal=None, binding=None):
    encoded, pending = [], {}
    for q in S.queries(case, reader):
        suffix = C.Q.suffix_text(q['text'])
        enc = bb.encode(text, suffix)
        if list(enc['prefix_ids']) != list(art['prefix_ids']):
            raise ValueError('Reader changed the sealed token boundary')
        labels = bb.label_ids(enc['full_text'], q['labels'])
        key = measurement_key(bb, art, suffix, enc['suffix_ids'], labels)
        item = (q, suffix, enc, labels, key)
        encoded.append(item)
        if key not in cache:
            pending.setdefault(key, item)
    work = list(pending.values())
    for start in range(0, len(work), C.Q.CHUNK):
        chunk = work[start:start + C.Q.CHUNK]
        batch = {'mode': serving_mode(bb), 'artifact_hash': art['hash'],
                 'prefix_sha256': art['prefix_sha256'], 'prefix_tokens': art['prefix_len'],
                 'suffix_ids': [v[2]['suffix_ids'] for v in chunk],
                 'label_ids': [v[3] for v in chunk], 'batch_size': len(chunk),
                 'padded_suffix_length': max(len(v[2]['suffix_ids']) for v in chunk)}
        batch_hash = R.stable_hash(batch)
        results = physical_batch(bb, art, batch, journal=journal,
            event={'binding': binding, 'actor': actor, 'case_identity': case.identity,
                   'condition': name, 'reader': reader, 'batch_start': start})
        if len(results) != len(chunk):
            raise ValueError('Scoring row count mismatch')
        for slot, (item, result) in enumerate(zip(chunk, results, strict=True)):
            if not result.get('master_unchanged') or not result.get('finite', True):
                raise ValueError('Mutable master or nonfinite scores; preserve technical failure')
            mid = R.stable_hash((actor, case.panel, case.key, item[4], batch_hash, slot))
            cache[item[4]] = {'result': dict(result), 'measurement_id': mid,
                              'original': (name, reader, item[0]['query_id']),
                              'serving_batch': batch, 'serving_batch_sha256': batch_hash,
                              'serving_slot': slot}
    rows = []
    for q, suffix, enc, labels, key in encoded:
        physical = cache[key]
        result, original = physical['result'], physical['original']
        batch, slot = physical['serving_batch'], physical['serving_slot']
        if (batch['mode'] != serving_mode(bb) or batch['artifact_hash'] != art['hash']
                or batch['prefix_sha256'] != art['prefix_sha256']
                or batch['suffix_ids'][slot] != enc['suffix_ids'] or batch['label_ids'][slot] != labels
                or R.stable_hash(batch) != physical['serving_batch_sha256']):
            raise ValueError('Alias serving configuration/byte identity mismatch')
        source_key = (reader, q['query_id'])
        if name == 'SOURCE':
            source[source_key] = physical
        source_physical = source[source_key]  # no old-reader fallback
        sp = source_physical['result']
        rec = S.query_record(case, q)
        rec.update(actor=actor, condition=name, reader=reader, question_text=q['text'],
            suffix_text=suffix, suffix_ids=enc['suffix_ids'], label_token_ids=labels,
            prefix_text=text, prefix_ids=list(art['prefix_ids']), prefix_sha256=art['prefix_sha256'],
            artifact_hash=art['hash'], cache_bytes=art['bytes'], measurement_id=physical['measurement_id'],
            prediction=result['prediction'], logps=result['logps'], answer_mass=result['answer_mass'],
            argmax_in_labels=result['argmax_in_labels'], tie=result['tie'], finite=result.get('finite', True),
            valid=not result['tie'] and result.get('finite', True),
            master_verified=result['master_unchanged'], source_prediction=sp['prediction'],
            source_logps=sp['logps'], source_valid=not sp['tie'] and sp.get('finite', True),
            source_measurement_id=source_physical['measurement_id'],
            gold_logp=result['logps'][rec['gold']], canonical=name.startswith('CANONICAL_CURRENT'),
            seed=int(name[-1]) if name.endswith(('_s0', '_s1')) else None,
            serving_batch=batch, serving_batch_sha256=physical['serving_batch_sha256'], serving_slot=slot,
            score_convention='original forced-choice label argmax; vocabulary argmax separately recorded')
        if original != (name, reader, q['query_id']):
            rec['measurement_alias_of'] = list(original)
            aliases.append({'case': case.key, 'condition': name, 'reader': reader,
                'query_id': q['query_id'], 'original': list(original),
                'measurement_id': physical['measurement_id'], 'artifact_hash': art['hash'],
                'serving_batch_sha256': physical['serving_batch_sha256'], 'serving_slot': slot})
        if name.startswith('CANONICAL_CURRENT'):
            rec['privileges'] = 'source backup and command history; no gold; original canonical_program'
        if art.get('realized'):
            rec['realized'] = art['realized']
        rows.append(rec)
    return rows


def evaluate(bb, actor, cases, banks, output, check_time, run_identity):
    if not cases or len({c.panel for c in cases}) != 1:
        raise ValueError('One nonempty actor/panel per evaluation')
    output = Path(output)
    sealed = seal_all(bb, actor, cases, banks, output, check_time, run_identity)
    expected = {(r['case'], r['condition']): r for r in sealed}
    seal_hash = C.sha(output / 'SEAL.json')
    sealed_mode = C.load(output / 'SEAL.json')['binding']['serving']
    for case in cases:
        done = output / 'blocks' / (case.key + '.DONE.json')
        if done.exists():
            receipt = C.load(done)
            if receipt['seal_sha256'] != seal_hash or receipt['case_identity'] != case.identity:
                raise ValueError('Completed case belongs to another sealed attempt')
            for fn, digest in receipt['files'].items():
                if C.sha(done.parent / fn) != digest:
                    raise ValueError('Completed case integrity failure')
            continue
        cache, source, aliases, rows = {}, {}, [], []
        started = time.monotonic()
        for name in S.conditions(case.panel):
            check_time()
            if serving_mode(bb) != sealed_mode:
                raise ValueError('Serving mode changed after seal')
            text, art = compile_case(bb, case, actor, name, banks)
            seal = expected[(case.key, name)]
            if (art['hash'] != seal['artifact_hash'] or art['prefix_sha256'] != seal['prefix_sha256']
                    or hashlib.sha256(text.encode()).hexdigest() != seal['prefix_text_sha256']):
                raise ValueError('Sealed compilation changed')
            for reader in C.READERS:
                rows.extend(score_reader(bb, case, actor, reader, name, text, art, cache, source, aliases,
                    journal=output / 'physical' / case.key, binding=seal_hash))
            del art
        path = done.parent / (case.key + '.jsonl')
        alias_path = done.parent / (case.key + '.ALIASES.jsonl')
        for member, content in ((path, rows), (alias_path, aliases)):
            if member.exists():
                if C.read_rows(member) != content:
                    raise ValueError('Existing derived block differs from verified physical journal')
            else:
                C.write_rows(member, content)
        C.dump(done, {'at': C.now(), 'rows': len(rows), 'measurements': len(cache),
            'aliases': len(aliases), 'seconds': time.monotonic() - started,
            'case_identity': case.identity, 'seal_sha256': seal_hash,
            'files': {path.name: C.sha(path), alias_path.name: C.sha(alias_path)},
            'physical_receipts': {p.relative_to(output).as_posix(): C.sha(p)
                                  for p in sorted((output / 'physical' / case.key).glob('*.DONE.json'))}})
    complete = output / 'EVALUATION_COMPLETE.json'
    if not complete.exists():
        C.dump(complete, {'at': C.now(), 'actor': actor, 'panel': cases[0].panel,
            'cases': len(cases), 'units': len({c.root_id or c.scene.scene_id for c in cases}),
            'seal_sha256': seal_hash, 'conditions': S.conditions(cases[0].panel), 'readers': C.READERS})
