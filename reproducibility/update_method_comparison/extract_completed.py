"""Closed-inventory intake of the completed ORIGIN panel; no inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ARCHIVE_SHA256 = '42163f6c18385140ab10797e1bf54de69e9eb019c3744d507332862e2b787cad'
CONDITIONS = ('SOURCE','REBUILD','EXISTING_CORRECTION','LATEST_SAME_WORDING',
    'LATEST_ERRATUM','FIELD_PLUS_LATEST_ERRATUM','INV_PAIR_NLL_s0','INV_PAIR_NLL_s1',
    'FREE_PAIR_CONSISTENCY_s0','FREE_PAIR_CONSISTENCY_s1','CANONICAL_REPLAY_s0','CANONICAL_REPLAY_s1')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def extract(archive, output):
    if output.exists():
        raise FileExistsError('Fresh destination required')
    assert sha(archive.read_bytes()) == ARCHIVE_SHA256, 'Unexpected delivery bytes'
    with zipfile.ZipFile(archive) as z:
        prefix = 'UPDATE_METHOD_RESULTS/'
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        assert len({n.casefold() for n in names}) == len(names)
        assert all(n.startswith(prefix) and '..' not in PurePosixPath(n).parts and ':' not in n and '\\' not in n for n in names)
        original = json.loads(z.read(prefix+'MANIFEST.json'))
        for name, rec in original['files'].items():
            raw = z.read(prefix+name)
            assert len(raw) == rec['bytes'] and sha(raw) == rec['sha256'], name
        # The old manifest is incomplete. Admit only this independently declared
        # finite inventory, not arbitrary unmanifested archive members.
        expected = {'MANIFEST.json','raw/PREFLIGHT_MODEL_gemma.json','raw/PREFLIGHT_MODEL_qwen.json'}
        for actor in ('gemma','qwen'):
            for panel, folder in (('ORIGIN','raw/'),('DEV','raw/dev/')):
                for condition in CONDITIONS:
                    stem = folder + '__'.join((actor,panel,condition))
                    expected.update(stem+ext for ext in ('.jsonl','.telemetry.jsonl','.done.json'))
                expected.add(folder+'__'.join((actor,panel,'SOURCE'))+'.answers.json')
        expected.update('tables/'+name for name in ('primary_contrasts.json','headline_results.csv',
            'per_scene_results.csv','per_root_results.csv','timing_memory.csv',
            'seed_diag_selection.csv','seed_diag_geometry.csv','seed_diag_behavior.csv'))
        assert {n.removeprefix(prefix) for n in names} == expected, 'Unexpected/missing delivered member'
        rows=[]
        for member in sorted(names):
            relative=member.removeprefix(prefix)
            raw=z.read(member)
            name='SOURCE_MANIFEST.json' if relative=='MANIFEST.json' else relative
            data=raw
            if name.endswith(('.jsonl','.answers.json')):
                name += '.gz'; data=gzip.compress(raw,mtime=0)
            dest=output/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(data)
            rows.append(dict(path=name,member=member,sha256=sha(data),bytes=len(data),
                source_sha256=sha(raw),source_bytes=len(raw),
                original_manifest_bound=relative in original['files'],
                transformation='gzip-lossless' if data!=raw else 'none'))
        result=dict(schema_version=1,archive=archive.name,archive_sha256=ARCHIVE_SHA256,
            archive_bytes=archive.stat().st_size,files=rows,
            original_manifest_members_verified=len(original['files']),
            additional_members_bound_by_closed_intake_inventory=len(expected)-1-len(original['files']),
            integrity_limit='Original manifest predates ORIGIN scores and analysis tables. Added members are bound by the supplied outer archive and this explicit intake inventory, not retroactively attested by the original manifest.',
            provenance_limit='Delivery contains no executed code snapshot, environment receipt or external worked-example output. Earlier supplied implementation is retained separately; exact executed-code identity is not attested.')
        (output/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();extract(a.archive,a.output)
