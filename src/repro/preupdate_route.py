"""Check compact historical deletion evidence, without model execution."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .common import ROOT, load_json, safe_path, sha256

SOURCE = ROOT / 'reproducibility/preupdate_route'
ARCHIVE_SHA256 = '0d968769fb62335991d365086ed50902bd2a47dfe5bb6881e87652d36d8047a8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(source=SOURCE, archive=None):
    source = Path(source).resolve()
    binding = load_json(source / 'SOURCE_ARCHIVE.json')
    manifest = load_json(source / 'MANIFEST.json')
    require(binding['sha256'] == manifest['archive_sha256'] == ARCHIVE_SHA256,
            'Wrong external archive binding')
    require(manifest['schema'] == 'PREUPDATE_COMPACT_V1', 'Wrong compact schema')
    records = manifest['files']
    actual = {p.relative_to(source).as_posix() for p in (source / 'supplied').rglob('*') if p.is_file()}
    require(actual == set(records), 'Undeclared or missing supplied evidence')
    for name, record in records.items():
        path = safe_path(name, source)
        require(path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'],
                'Changed supplied evidence: ' + name)
    original = load_json(source / 'supplied/MANIFEST.json')
    require(sha256(source / 'supplied/MANIFEST.json') == binding['archive_manifest_sha256'],
            'Changed historical manifest')
    require(original['capsules'] == binding['capsules'], 'Changed capsule binding')
    for name, record in records.items():
        member = record['archive_member']
        if member != 'MANIFEST.json':
            require(original['files'][member] == {k: record[k] for k in ('bytes', 'sha256')},
                    'Compact member disagrees with original manifest: ' + name)
    if archive is not None:
        archive = Path(archive)
        require(archive.stat().st_size == binding['bytes'] and sha256(archive) == binding['sha256'],
                'Changed full external archive')
    return len(records)


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def reconstruct(source=SOURCE):
    """Recompute tabular totals and gates; raw-score replay is a separate command."""
    source = Path(source)
    supplied = source / 'supplied'
    screen = load_json(supplied / 'replay/SCREEN.json')
    proof = load_json(supplied / 'replay/CPU_REPLAY.json')
    full_manifest = load_json(supplied / 'MANIFEST.json')
    topology = load_json(supplied / 'external_review_packet_v1.json')['evidence_topology']
    totals = defaultdict(lambda: dict(scheduled=0, errors=0, new_errors=0, invalid=0))
    damage = rows(supplied / 'SCREEN_DAMAGE.csv')
    identities = {(r['root_id'], int(r['seed']), r['arm'], r['family']) for r in damage}
    cases = {r['root_id'] for r in damage}
    expected_identities = {(case, seed, arm, family) for case in cases for seed in (0, 1)
                           for arm in ('N', 'K', 'SCTRL') for family in ('direct', 'joint', 'opposes')}
    require(len(damage) == len(identities) and identities == expected_identities,
            'Missing or duplicate case/seed/arm/family rows')
    for row in damage:
        values = {key: int(row[key]) for key in totals[row['arm'], row['family']]}
        require(0 <= values['new_errors'] <= values['errors'] <= values['scheduled']
                and 0 <= values['invalid'] <= values['scheduled'], 'Invalid error count')
        for key, value in values.items():
            totals[row['arm'], row['family']][key] += value
    canonical_totals = sorted([dict(arm=arm, family=family, **value)
                               for (arm, family), value in totals.items()], key=lambda r: (r['arm'], r['family']))
    require(canonical_totals == sorted(load_json(supplied / 'SCREEN_TOTALS.json'), key=lambda r: (r['arm'], r['family'])),
            'Damage rows disagree with supplied totals')
    corrections = []
    for row in rows(supplied / 'SCREEN_CORRECTIONS.csv'):
        require(row['corrected'] in ('True', 'False') and row['positive'] in ('True', 'False'),
                'Invalid correction Boolean')
        corrections.append(dict(row, seed=int(row['seed']), corrected=row['corrected'] == 'True',
                                positive=row['positive'] == 'True', pair=[int(v) for v in row['pair'].split(',')]))
    key = lambda r: (r['root_id'], r['seed'], r['arm'])
    require(len({key(r) for r in corrections}) == len(corrections), 'Duplicate correction row')
    require({key(r) for r in corrections} == {(case, seed, arm) for case in cases
            for seed in (0, 1) for arm in ('K', 'SCTRL')}, 'Incomplete correction panel')
    require(all(not r['corrected'] or r['positive'] for r in corrections),
            'Correction credited outside the positive panel')
    require(sorted(corrections, key=key) == sorted(screen['details'], key=key),
            'Correction rows disagree with screen')
    corrected = {arm: sorted({r['root_id'] for r in corrections if r['arm'] == arm and r['corrected']})
                 for arm in ('K', 'SCTRL')}
    by_seed = {str(seed): sorted({r['root_id'] for r in corrections if r['arm'] == 'K'
                                and r['seed'] == seed and r['corrected']}) for seed in (0, 1)}
    require(corrected == screen['corrected_cases'] and by_seed == screen['K_corrected_by_seed'],
            'Correction case sets differ')
    joint = {arm: totals[arm, 'joint']['errors'] for arm in ('N', 'K', 'SCTRL')}
    direct = totals['K', 'direct']
    require(len({totals[arm, 'joint']['scheduled'] for arm in joint}) == 1, 'Unequal joint denominators')
    gates = dict(three_distinct_positive=len(corrected['K']) >= 3,
                 each_seed=all(by_seed.values()),
                 two_more_than_control=len(corrected['K']) - len(corrected['SCTRL']) >= 2,
                 lower_mean_joint_error=joint['K'] < joint['N'],
                 direct_new_errors_at_most_2pp=100 * direct['new_errors'] <= 2 * direct['scheduled'])
    require(gates == screen['gates'], 'Recomputed admission gates differ')
    require(screen['joint_errors'] == {arm: joint[arm] for arm in ('K', 'N')}
            and screen['new_direct_errors'] == direct['new_errors']
            and screen['scheduled_direct'] == direct['scheduled']
            and screen['scheduled_joint'] == totals['K', 'joint']['scheduled'], 'Screen counts differ')
    require(proof['screen_sha256'] == sha256(supplied / 'replay/SCREEN.json'), 'Screen receipt hash differs')
    require(proof['final_cases'] == full_manifest['final_cases'] == topology['fresh_final_cases_run']
            == topology['complete_final_repeats_run'] == 0, 'Unexpected final outcomes')
    require(proof['status'] == screen['status'] == full_manifest['scientific_status']
            == 'NO_ACTIONABLE_DIRECT_PREUPDATE_ROUTE' and not all(gates.values()), 'Wrong terminal branch')
    return dict(status=proof['status'], scope='Fixed historical screen only',
                statistical_unit='case; seeds, histories and questions are repeated measurements',
                historical_cases=len(cases), historical_positive_cases=len({r['root_id'] for r in corrections if r['positive']}),
                seeds=sorted({int(r['seed']) for r in damage}),
                joint_errors=joint, joint_responses=totals['K', 'joint']['scheduled'],
                direct_new_errors=direct['new_errors'], direct_responses=direct['scheduled'],
                new_joint_errors=totals['K', 'joint']['new_errors'],
                corrected_joint_errors=joint['N'] + totals['K', 'joint']['new_errors'] - joint['K'],
                gates=gates, failed_gates=sum(not value for value in gates.values()),
                total_gates=len(gates), corrected_cases=corrected, corrected_by_seed=by_seed,
                final_cases_run=proof['final_cases'], final_cases_planned=topology['fresh_final_inputs_frozen'],
                source_archive_sha256=ARCHIVE_SHA256)


def check(source=SOURCE, archive=None):
    count = verify(source, archive)
    result = reconstruct(source)
    require(result == load_json(Path(source) / 'EXPECTED.json'), 'Compact expected output differs')
    return dict(status='PASS', checked_members=count, external_archive_checked=archive is not None,
                reconstruction_level='Saved tables and gate consistency; not raw-score or tensor replay', result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.archive), indent=2))


if __name__ == '__main__':
    main()
