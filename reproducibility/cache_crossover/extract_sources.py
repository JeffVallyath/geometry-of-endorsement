"""Verify a completed local delivery and export compact scientific evidence only.

No archive paths are extracted to the filesystem. No model code is imported.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile

ARCHIVE_SHA256 = 'fc6f8c3eed7bfca435fc1042453bfb2e44874a3179fda4e950ba95416d18b5e2'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or p.as_posix() != name:
        raise ValueError('Unsafe archive path: ' + name)


def extract(archive, receipt_path, output):
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    with archive.open('rb') as stream:
        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert archive_hash == receipt['sha256'] == ARCHIVE_SHA256
    assert archive.stat().st_size == receipt['bytes'] == 420599025
    assert receipt['status'] == 'VERIFIED_FULL_DELIVERY'
    payloads = {}
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        assert len(members) == receipt['members'] == 70
        assert len({m.filename.casefold() for m in members}) == len(members)
        manifest = json.loads(z.read('RESULTS_MANIFEST.json'))
        declared = {r['path']: r for r in manifest['files']}
        assert len(declared) == len(manifest['files'])
        assert set(z.namelist()) == set(declared) | {'RESULTS_MANIFEST.json'}
        for entry in members:
            safe(entry.filename)
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            data = z.read(entry)
            if entry.filename in declared:
                row = declared[entry.filename]
                assert len(data) == row['bytes'] and digest(data) == row['sha256']
            payloads[entry.filename] = data
    capsule = json.loads(payloads['provenance/RUN_CAPSULE.json'])
    capsule_receipt = json.loads(payloads['provenance/CAPSULE_RECEIPT.json'])
    assert digest(payloads['provenance/RUN_CAPSULE.json']) == capsule_receipt['capsule_manifest_sha256']
    assert digest(payloads['provenance/run-payload.tar.gz']) == capsule_receipt['payload_sha256']
    assert len(payloads['provenance/run-payload.tar.gz']) == capsule_receipt['payload_bytes']
    bound = {r['path']: r for r in capsule['files']}
    assert len(bound) == len(capsule['files'])
    selected, seen = {}, set()
    with tarfile.open(fileobj=io.BytesIO(payloads['provenance/run-payload.tar.gz']), mode='r:gz') as tar:
        for entry in tar:
            name = entry.name.removeprefix('./')
            safe(name)
            assert entry.isfile() and name.casefold() not in seen
            seen.add(name.casefold())
            data = tar.extractfile(entry).read()
            if name == 'RUN_CAPSULE.json':
                assert data == payloads['provenance/RUN_CAPSULE.json']
            else:
                assert name in bound and len(data) == bound[name]['bytes'] and digest(data) == bound[name]['sha256']
            if (name in ('handoff/CASE_PLAN.json', 'continuation/gates.py',
                         'continuation/resume02/prior/results/raw.jsonl')
                    or '/independent_verification/inputs/designs/' in name):
                selected[name] = data
        assert seen == {n.casefold() for n in bound} | {'run_capsule.json'}
        assert len(seen) == capsule_receipt['members'] == 747
    terminal = json.loads(payloads['combined/terminal.json'])
    prior = selected['continuation/resume02/prior/results/raw.jsonl']
    new = payloads['pretrained2/results/raw.jsonl']
    assert payloads['combined/raw.jsonl'] == prior + new
    assert digest(prior) == terminal['original_raw_sha256']
    assert digest(new) == terminal['new_raw_sha256']
    assert terminal['status'] == 'SCIENTIFIC_EXECUTED'
    assert (terminal['real_forwards'], terminal['prior_forwards_reused'], terminal['new_forwards']) == (1992, 332, 1660)
    for name in ('prior/first-instance-terminated.json', 'watchdog2/terminated.json'):
        r = json.loads(payloads[name])
        assert r['status'] == 'CONFIRMED_TERMINATED' and r['response']['http_status'] == 200
        assert r['response']['body']['data']['id'] == r['instance_id']
        assert r['response']['body']['data']['status'] == 'terminated'
    assert json.loads(payloads['watchdog2/TASK_CLEANUP.json'])['state'] == 'Disabled'

    output.mkdir(parents=True, exist_ok=False)
    files = []

    def write(name, data, sources, derivation):
        (output / name).write_bytes(data)
        files.append(dict(path=name, sha256=digest(data), bytes=len(data),
                          sources=sources, derivation=derivation))

    def source(name, nested=False):
        row = (bound if nested else declared)[name]
        return dict(archive='V10_CACHE_CROSSOVER_RESULTS.zip', member=name,
                    container='provenance/run-payload.tar.gz' if nested else None,
                    sha256=row['sha256'], bytes=row['bytes'], source_manifest_verified=True)

    def encoded(value):
        return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()

    write('raw.jsonl.gz', gzip.compress(payloads['combined/raw.jsonl'], mtime=0),
          [source('combined/raw.jsonl')], 'Lossless gzip of the complete scientific journal; no rows or fields omitted.')
    for target, name in [('saved_summaries.json', 'combined/summaries.json'),
                         ('terminal.json', 'combined/terminal.json')]:
        write(target, payloads[name], [source(name)], 'Byte-for-byte scientific output.')
    plan = json.loads(selected['handoff/CASE_PLAN.json'])
    write('case_plan.json', selected['handoff/CASE_PLAN.json'], [source('handoff/CASE_PLAN.json', True)],
          'Byte-for-byte pre-execution plan; its proposed status describes planning, not the completed run.')
    write('frozen_gates.py.txt', selected['continuation/gates.py'], [source('continuation/gates.py', True)],
          'Byte-for-byte classification specification, retained for inspection, not imported by replay.')
    worlds, sources = {}, []
    for case in plan['records']:
        rid = case['root_id']
        name = next(n for n in selected if n.endswith('/inputs/designs/' + rid + '.json'))
        design = json.loads(selected[name])
        worlds[rid] = design['root']['terminal']
        sources.append(source(name, True))
    write('terminal_worlds.json', encoded(worlds), sources,
          'Select the complete terminal world for each of the twelve planned roots; no numeric rounding.')
    # Host paths, provider identifiers and operational logs stay outside the public package.
    public_receipt = {k: v for k, v in receipt.items() if k not in ('archive', 'cumulative_compute_cent_ceiling_usd')}
    public_receipt['archive'] = 'V10_CACHE_CROSSOVER_RESULTS.zip'
    write('delivery_receipt.json', encoded(public_receipt),
          [dict(member='DELIVERY_RECEIPT.json', sha256=digest(receipt_bytes), bytes=len(receipt_bytes))],
          'Receipt projection: local archive path replaced by basename; cost field omitted; other fields unchanged.')
    provenance = dict(schema_version=1, archives={'V10_CACHE_CROSSOVER_RESULTS.zip': dict(
        sha256=archive_hash, bytes=archive.stat().st_size, members=len(members),
        public_url=None, status='unavailable_to_external_reproducer')},
        capsule=capsule_receipt, verification=dict(outer_members=70, capsule_members=747,
        exact_prior_plus_new_journal=True, recorded_both_instances_terminated=True,
        recorded_watchdog_disabled=True, scope='Local delivered bytes and saved receipts; no live provider check or neural execution'),
        files=files)
    (output / 'provenance.json').write_bytes(encoded(provenance))
    return dict(status='verified_scientific_evidence_extracted', files=len(files), output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory only')
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.receipt, args.output), indent=2))
