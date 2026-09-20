"""Offline verification/replay from delivered bytes, with no machine-local dependencies."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode=True


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf8'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main(bundle,work):
    bundle=bundle.resolve();manifest=load(bundle/'MANIFEST.json')
    if manifest['schema']!='PREUPDATE_ROUTE_RESULTS_V1':raise ValueError('Wrong result package')
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual!=set(manifest['files'])|{'MANIFEST.json'}:raise ValueError('Undeclared/missing delivered file')
    for name,record in manifest['files'].items():
        rel=PurePosixPath(name);path=bundle.joinpath(*rel.parts)
        if rel.is_absolute() or '..' in rel.parts or ':' in name or '\\' in name or path.is_symlink() or not path.resolve().is_relative_to(bundle):raise ValueError('Unsafe package member')
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Changed delivered file: '+name)
    work.mkdir(parents=True,exist_ok=False)
    capsule=bundle/'capsules/puf1-r2.zip';reader=module('_puf_capsule_reader',bundle/'tools/run_capsule.py')
    reader.extract(capsule,work/'capsule',manifest['capsules']['r2'])
    exporter=module('_puf_export_reader',bundle/'tools/artifact_export.py')
    exports=[]
    for attempt in ('r1','r2'):
        for seed in (0,1):
            archive=bundle/f'raw/{attempt}_seed{seed}.zip'
            dest=work/f'seed{seed}' if attempt=='r2' else None
            receipt=exporter.verify(archive,manifest['capsules'][attempt],dest)
            if not receipt['quiescent_snapshot']:raise ValueError('Non-terminal export')
            exports.append(dict(attempt=attempt,seed=seed,**receipt))
    sys.path.insert(0,str(work/'capsule/src'))
    import puf_replay
    result=puf_replay.replay({s:work/f'seed{s}/scientific' for s in (0,1)},work/'analysis')
    if result!=load(bundle/'replay/CPU_REPLAY.json') or load(work/'analysis/SCREEN.json')!=load(bundle/'replay/SCREEN.json'):
        raise ValueError('Bundled replay differs')
    figures=module('_puf_figures',bundle/'tools/screen_figures.py')
    regenerated=figures.create(bundle,work/'figures')
    for name,digest in regenerated['figures'].items():
        if sha(bundle/'figures'/name)!=digest:raise ValueError('Figure regeneration differs')
    import numpy
    receipt=dict(status='PASS',package_members=len(manifest['files']),no_external_paths_or_gpu=True,numpy=numpy.__version__,
        result=result,exports=exports,figures_identical=True,analysis_screen_sha256=sha(work/'analysis/SCREEN.json'))
    with (work/'BUNDLE_REPLAY.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='exports'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();main(a.bundle,a.work)
