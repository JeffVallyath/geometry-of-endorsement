"""Raw-to-receipt verifier; completion is proved against fixed cases, not flags."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import rsi6_common as S

C, R = S.C, S.R


def safe_member(root, name):
    if not isinstance(name, str) or not name or '\\' in name or PureWindowsPath(name).drive:
        raise ValueError('Artifact paths must be relative canonical POSIX names')
    p = PurePosixPath(name)
    target = Path(root) / p
    if (p.is_absolute() or p.as_posix() != name or '..' in p.parts
            or not target.resolve().is_relative_to(Path(root).resolve())):
        raise ValueError('Unsafe artifact member')
    return target


def verify_case(rows, case, actor, seals, physical=None):
    expected = {(name, reader, q['query_id']): q for name in S.conditions(case.panel)
                for reader in C.READERS for q in S.queries(case, reader)}
    observed, measurements = {}, {}
    for row in rows:
        key = tuple(row[k] for k in ('condition', 'reader', 'query_id'))
        if key not in expected or key in observed:
            raise ValueError('Duplicate/undeclared scientific row')
        q = expected[key]
        record = S.query_record(case, q)
        if row['actor'] != actor or any(row.get(k) != v for k, v in record.items()):
            raise ValueError('Fixed case identity/independent labels drift')
        if row['question_text'] != q['text'] or row['suffix_text'] != C.Q.suffix_text(q['text']):
            raise ValueError('Frozen reader wording drift')
        seal = seals[(case.key, row['condition'])]
        if any(row[k] != seal[k] for k in ('artifact_hash', 'prefix_sha256', 'cache_bytes')):
            raise ValueError('Raw row does not match master seal')
        text, _ = C.Q.prefix_text(case.scene, C.PROCEDURES[actor], case.order,
            world=case.terminal if row['condition'] == 'NATIVE_FINAL' else None,
            corrections=case.commands if row['condition'] == 'TEXT_CORRECTION' else None)
        if row['prefix_text'] != text or hashlib.sha256(text.encode()).hexdigest() != seal['prefix_text_sha256']:
            raise ValueError('Prefix is not the declared source/native/correction text')
        lp = row['logps']
        if (len(lp) != 2 or not all(math.isfinite(v) for v in lp) or row['prediction'] != (0 if lp[0] >= lp[1] else 1)
                or row['tie'] != (lp[0] == lp[1]) or row['valid'] != (not row['tie'] and row['finite'])
                or not row['master_verified'] or row['gold_logp'] != lp[row['gold']]
                or not math.isclose(row['answer_mass'], sum(math.exp(v) for v in lp), abs_tol=1e-7)):
            raise ValueError('Raw score/validity/immutable-master inconsistency')
        batch, slot = row['serving_batch'], row['serving_slot']
        if (R.stable_hash(batch) != row['serving_batch_sha256'] or batch['artifact_hash'] != row['artifact_hash']
                or batch['prefix_sha256'] != row['prefix_sha256'] or batch['suffix_ids'][slot] != row['suffix_ids']
                or batch['label_ids'][slot] != row['label_token_ids']):
            raise ValueError('Batch/slot alias identity drift')
        logical_key = (R.stable_hash(batch['mode']), row['artifact_hash'], row['prefix_sha256'],
                       row['suffix_text'], tuple(row['suffix_ids']), tuple(row['label_token_ids']))
        mid = R.stable_hash((actor, case.panel, case.key, logical_key, row['serving_batch_sha256'], slot))
        if row['measurement_id'] != mid:
            raise ValueError('Measurement ID not bound to exact case and serving bytes')
        identity = (logical_key, row['serving_batch_sha256'], slot, lp, row['prediction'])
        if measurements.setdefault(mid, identity) != identity:
            raise ValueError('Nonidentical execution mislabeled as an alias')
        if physical is not None:
            actual = physical[(row['serving_batch_sha256'], slot)]
            if any(row[k] != actual[k] for k in ('prediction', 'logps', 'answer_mass', 'tie', 'argmax_in_labels')):
                raise ValueError('Derived row differs from actual physical model output')
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError(f'Incomplete case: {len(observed)}/{len(expected)} rows')
    for key, row in observed.items():
        source = observed[('SOURCE', key[1], key[2])]
        if (row['source_prediction'] != source['prediction'] or row['source_logps'] != source['logps']
                or row['source_valid'] != source['valid'] or row['source_measurement_id'] != source['measurement_id']):
            raise ValueError('Same-reader SOURCE denominator mismatch')
        if 'measurement_alias_of' in row:
            origin = observed[tuple(row['measurement_alias_of'])]
            if origin['measurement_id'] != row['measurement_id']:
                raise ValueError('Alias points to a different physical execution')
    return {'rows': len(rows), 'measurements': len(measurements)}


def load_panel(root, panel, prepared=S.ROOT / 'prepared', *, case_sets=None):
    """Production uses complete supplied case_sets; injection is for CPU fixtures."""
    root = Path(root)
    case_sets = {a: S.load_cases(panel, a, prepared) for a in C.SIZES} if case_sets is None else case_sets
    output, checks = [], {}
    for actor, cases in case_sets.items():
        folder = root / panel / actor
        seal = C.load(folder / 'SEAL.json')
        if C.sha(folder / 'SEAL.jsonl') != seal['sha256'] or seal['binding']['cases'] != [c.identity for c in cases]:
            raise ValueError('Seal does not cover exact supplied population')
        for name, digest in seal['witness_files'].items():
            if C.sha(safe_member(folder, name)) != digest:
                raise ValueError('Witness custody mismatch')
        seals = {(r['case'], r['condition']): r for r in C.read_rows(folder / 'SEAL.jsonl')}
        required_seals = {(c.key, n) for c in cases for n in S.conditions(panel)}
        if set(seals) != required_seals:
            raise ValueError('Missing/extra master seal')
        expected_done = {c.key + '.DONE.json' for c in cases}
        if {p.name for p in (folder / 'blocks').glob('*.DONE.json')} != expected_done:
            raise ValueError('Complete case receipt inventory missing/extra')
        total = 0
        for case in cases:
            done = C.load(folder / 'blocks' / (case.key + '.DONE.json'))
            if done['seal_sha256'] != C.sha(folder / 'SEAL.json') or done['case_identity'] != case.identity:
                raise ValueError('Case receipt seal mismatch')
            physical = {}
            for name, digest in done['physical_receipts'].items():
                path = safe_member(folder, name)
                if C.sha(path) != digest:
                    raise ValueError('Physical receipt hash mismatch')
                receipt = C.load(path)
                stem = path.name.removesuffix('.DONE.json')
                intent_path, raw_path = path.with_name(stem + '.INTENT.json'), path.with_name(stem + '.RAW.json')
                if C.sha(intent_path) != receipt['intent_sha256'] or C.sha(raw_path) != receipt['raw_sha256']:
                    raise ValueError('Raw physical model output custody mismatch')
                intent, raw = C.load(intent_path), C.load(raw_path)
                if R.stable_hash(intent) != receipt['event_id'] or not raw['accepted']:
                    raise ValueError('Physical batch is failed or identity changed')
                if intent['event']['binding'] != done['seal_sha256'] or intent['event']['case_identity'] != case.identity:
                    raise ValueError('Physical output belongs to a different scientific attempt')
                bh = R.stable_hash(intent['batch'])
                for slot, result in enumerate(raw['results']):
                    if (bh, slot) in physical:
                        raise ValueError('Multiple physical executions share an identity')
                    physical[(bh, slot)] = result
            rows, aliases = [], []
            for name, digest in done['files'].items():
                member = safe_member(folder / 'blocks', name)
                if C.sha(member) != digest:
                    raise ValueError('Derived output custody mismatch')
                if name.endswith('.ALIASES.jsonl'):
                    aliases = C.read_rows(member)
                else:
                    rows.extend(C.read_rows(member))
            verified = verify_case(rows, case, actor, seals, physical)
            if len(rows) != done['rows'] or len(aliases) != done['aliases']:
                raise ValueError('Receipt row/alias counts inconsistent')
            expected_aliases = {(r['condition'], r['reader'], r['query_id'], r['measurement_id'])
                                for r in rows if 'measurement_alias_of' in r}
            if {(r['condition'], r['reader'], r['query_id'], r['measurement_id']) for r in aliases} != expected_aliases:
                raise ValueError('Alias inventory incomplete')
            # Exact prompts/IDs/batches stay in raw custody. Avoid repeating their
            # bulk in every derived diagnostic and in-memory analysis copy.
            omit = {'prefix_text', 'prefix_ids', 'serving_batch', 'realized'}
            output.extend({k: v for k, v in r.items() if k not in omit} for r in rows)
            total += verified['rows']
        checks[actor] = {'cases': len(cases), 'units': len({c.root_id or c.scene.scene_id for c in cases}), 'rows': total}
    return output, {'panel': panel, 'status': 'EXACT_RAW_PHYSICAL_CUSTODY_AND_COVERAGE_PASS', 'actors': checks}
