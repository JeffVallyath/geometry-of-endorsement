"""Portable release integrity and compatibility paths for saved-output analyzers."""
from __future__ import annotations
import hashlib
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SECTIONS=('matched_histories','cache_interventions','external_state','interpreter_checks','gmello_comparison')

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def safe(root,name):
    relative=PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name
        or relative.as_posix()!=name or any(p.endswith((' ','.')) for p in relative.parts)
        or any(p.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*('COM'+str(i) for i in range(1,10)),*('LPT'+str(i) for i in range(1,10))} for p in relative.parts)):
        raise ValueError('Nonportable relative path: '+name)
    path=Path(root).joinpath(*relative.parts)
    for item in (path,*path.parents):
        if item==Path(root).parent:break
        if item.is_symlink() or getattr(item,'is_junction',lambda:False)():raise ValueError('Linked path: '+name)
    if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Escaping path')
    return path

def inventory(root):
    result={}
    for path in Path(root).rglob('*'):
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('Linked release member')
        if path.is_file():result[path.relative_to(root).as_posix()]=path
    return result

def verify(root=ROOT):
    root=Path(root).resolve(); manifest=root/'metadata/files.sha256'
    entries={}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        value,name=line.split('  ',1)
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid SHA-256')
        if name.casefold() in {n.casefold() for n in entries}:raise ValueError('Duplicate release member')
        path=safe(root,name)
        if not path.is_file() or digest(path)!=value:raise ValueError('Missing or changed release member: '+name)
        entries[name]=value
    if set(inventory(root))!=set(entries)|{'metadata/files.sha256'}:raise ValueError('Unexpected release file inventory')
    return {'status':'PASS','files':len(entries),'bytes':sum(safe(root,n).stat().st_size for n in entries)}

def fresh_output(path,root=ROOT):
    path=Path(path).absolute();root=Path(root).resolve()
    if path.exists():raise FileExistsError('Choose a new output directory')
    if path.is_relative_to(root) or root.is_relative_to(path):raise ValueError('Output must be outside the input repository')
    for parent in path.parents:
        if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('Linked output parent')
    path.mkdir(parents=True)
    return path.resolve()

def mount(index,destination,root=ROOT):
    """Reconstruct legacy paths from one included copy; no symlinks or network."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Compatibility directory already exists')
    destination.mkdir(parents=True)
    for namespace,files in index['mounts'].items():
        for relative,record in files.items():
            target=safe(destination,namespace+'/'+relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            if 'included' in record:
                source=safe(root,record['included'])
                if digest(source)!=record['sha256']:raise ValueError('Changed resolver input')
                if 'codec' in record:
                    from .transport import restore
                    restore(source,target,record)
                else:shutil.copyfile(source,target)
            else:
                value=record['inline_utf8'].encode('utf-8')
                if hashlib.sha256(value).hexdigest()!=record['sha256']:raise ValueError('Changed inline compatibility metadata')
                target.write_bytes(value)
    return destination

def environment(source,output,hash_seed):
    allowed={'systemroot','windir','path','comspec','pathext','lang','lc_all'}
    env={k:v for k,v in os.environ.items() if k.lower() in allowed}
    runtime=Path(output)/'_runtime';runtime.mkdir(exist_ok=True)
    guard=source/'v3/src/repro/offline_site'
    env.update(PYTHONPATH=os.pathsep.join([str(guard),str(source/'v3/src')]),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONHASHSEED=str(hash_seed),
        REPRO_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        REPRO_RELEASE_ROOT=str(ROOT),
        HF_HUB_DISABLE_TELEMETRY='1',HF_HOME=str(runtime/'hf'),CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MPLBACKEND='Agg',MPLCONFIGDIR=str(runtime/'matplotlib'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TEMP=str(runtime),TMP=str(runtime),TMPDIR=str(runtime))
    if 'REPRO_DEADLINE_UTC' in os.environ:env['REPRO_DEADLINE_UTC']=os.environ['REPRO_DEADLINE_UTC']
    return env

def child_environment(cwd,env):
    result=dict(env)
    if (Path(cwd)/'src/repro/common.py').is_file():
        guard=result['PYTHONPATH'].split(os.pathsep)[0]
        result['PYTHONPATH']=os.pathsep.join([guard,str(Path(cwd)/'src')])
    return result

def run_step(label,python,args,cwd,output,env,timeout=1800,keep_log=False):
    start=time.monotonic()
    log=Path(output)/'_runtime'/(label+'.log')
    child=child_environment(cwd,env)
    deadline=time.time()+timeout
    if child.get('REPRO_DEADLINE_UTC'):
        deadline=min(deadline,datetime.datetime.fromisoformat(child['REPRO_DEADLINE_UTC']).timestamp())
    child['REPRO_DEADLINE_UTC']=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat()
    error=None
    try:
        with log.open('xb') as stream:
            process=subprocess.run([str(python),'-B','-X','utf8',*map(str,args)],cwd=cwd,env=child,
                stdout=stream,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.time()),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        code=process.returncode
    except subprocess.TimeoutExpired:
        code=124;error='Step timeout; the same finite deadline is inherited by Python descendants.'
    row={'step':label,'status':'PASS' if code==0 else 'FAIL',
         'exit_code':code,'seconds':round(time.monotonic()-start,3)}
    if code or keep_log:row['diagnostics']=log.relative_to(output).as_posix()
    else:log.unlink()  # Only this function's successful diagnostic file.
    if error:row['error']=error
    return row
