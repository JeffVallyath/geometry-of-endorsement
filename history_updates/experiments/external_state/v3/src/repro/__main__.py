"""Reconstruct the paper's main results from included, visible recorded evidence."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT
from .file_manifest import InputStore,EvidenceError,MissingEvidence,verify_release

QUICK=('original_public_history','paired_native_scale','repair_qwen7','repair_qwen32','repair_gemma27')
UPDATES=('history_exposure','same_case_baselines')
INCLUDED=(*QUICK,'historical_paper',*UPDATES)


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')


def fresh_output(path):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('Output already exists; use a fresh directory')
    if path==Path(path.anchor) or ROOT.is_relative_to(path):raise ValueError('Output cannot be a repository or filesystem ancestor')
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'reproduced'):
        raise ValueError('In-repository output must be under reproduced/')
    current=path.parent
    while current!=current.parent:
        if current.is_symlink() or getattr(current,'is_junction',lambda:False)():raise ValueError('Linked output parent')
        current=current.parent
    path.mkdir(parents=True)
    return path


def environment(output):
    allowed={'systemroot','windir','path','comspec','pathext','lang','lc_all'}
    env={k:v for k,v in os.environ.items() if k.lower() in allowed}
    runtime=output/'_runtime';runtime.mkdir(exist_ok=True)
    env.update(PYTHONPATH=os.pathsep.join([str(Path(__file__).parent/'offline_site'),str(ROOT/'src')]),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',
        PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED','0'),REPRO_OFFLINE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',HF_HOME=str(runtime/'hf'),CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MPLBACKEND='Agg',MPLCONFIGDIR=str(runtime/'matplotlib'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TEMP=str(runtime),TMP=str(runtime),TMPDIR=str(runtime))
    if 'REPRO_DEADLINE_UTC' in os.environ:env['REPRO_DEADLINE_UTC']=os.environ['REPRO_DEADLINE_UTC']
    return env


def portable(text,output):
    for path,label in ((output,'<output>'),(ROOT,'<repository>'),(Path(sys.prefix),'<python-env>'),(Path(sys.base_prefix),'<base-python>')):
        for value in (str(path),path.as_posix()):
            text=text.replace(json.dumps(value)[1:-1],label).replace(value,label)
    return text


def step(identifier,output,store,historical_python=None):
    if identifier not in INCLUDED:
        return {'study':identifier,'status':'NOT_RUN','reason':'Outside this reviewer bundle. Additional research-history studies remain in the separate full archive.'}
    if identifier=='historical_paper' and historical_python is None:
        raise MissingEvidence('The original manuscript route requires --historical-python from requirements-cpu.lock; it performs saved-result CPU analysis, not pretrained inference.')
    start=time.monotonic();source,input_receipt=store.verify(identifier)
    if identifier=='history_exposure':store.verify('original_public_history')
    if identifier=='same_case_baselines':store.verify('repair_qwen7')
    driver=('update_driver.py' if identifier in UPDATES else
            'legacy_driver.py' if identifier in {'original_public_history','historical_paper'} else 'source_driver.py')
    scientific_identifier='repository' if identifier=='historical_paper' else identifier
    interpreter=str(historical_python) if identifier=='historical_paper' else sys.executable
    destination=output/identifier
    arguments=[str(Path(__file__).with_name(driver)),scientific_identifier,'--source',str(source),'--output',str(destination)]
    result=subprocess.run([interpreter,'-B',*arguments],cwd=ROOT,env=environment(output),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',
        timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    log=identifier+'.log';(output/log).write_text(portable(result.stdout,output),encoding='utf8',newline='\n')
    # The scientific adapters are unchanged. Check that direct use of the
    # visible input tree also leaves every included byte and path untouched.
    store.verify(identifier)
    if identifier=='history_exposure':store.verify('original_public_history')
    if identifier=='same_case_baselines':store.verify('repair_qwen7')
    child=destination/'RECONSTRUCTION_RECEIPT.json'
    child_status=json.loads(child.read_bytes()).get('status') if child.is_file() else 'MISSING'
    passed=result.returncode==0 and child_status=='PASS'
    return {'study':identifier,'status':'PASS' if passed else 'FAIL','returncode':result.returncode,
            'scientific_receipt_status':child_status,'elapsed_seconds':round(time.monotonic()-start,3),
            'log':log,'input_verification':input_receipt,'inputs_unchanged':'PASS'}


def execute(identifiers,output,historical_python=None):
    output=fresh_output(output);records=[]
    try:
        store=InputStore(ROOT) if any(s in INCLUDED for s in identifiers) else None
        for identifier in identifiers:
            print('Reconstructing '+identifier,flush=True)
            row=step(identifier,output,store,historical_python);records.append(row)
            if row['status'] in {'FAIL','BLOCKED'}:break
    except MissingEvidence as error:
        records.append({'status':'BLOCKED','reason':str(error)})
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        records.append({'status':'FAIL','reason':type(error).__name__+': '+str(error)})
    passed=len(records)==len(identifiers) and all(r['status']=='PASS' for r in records)
    tables=None
    if passed and set(QUICK)<=set(identifiers):
        try:
            from .tables import reconstruct
            tables=reconstruct(output,ROOT)
        except (OSError,ValueError,KeyError) as error:
            passed=False;tables={'status':'FAIL','reason':type(error).__name__+': '+str(error)}
    if passed and set(UPDATES)<=set(identifiers):
        try:
            from .update_tables import reconstruct
            tables=reconstruct(output,ROOT)
        except (OSError,ValueError,KeyError) as error:
            passed=False;tables={'status':'FAIL','reason':type(error).__name__+': '+str(error)}
    attempted={r.get('study') for r in records}
    for identifier in identifiers:
        if identifier not in attempted:records.append({'study':identifier,'status':'NOT_RUN','reason':'Stopped after an earlier execution or integrity failure'})
    packages={}
    for name in ('numpy','transformers','tokenizers','Jinja2'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    receipt={'status':'PASS' if passed else 'FAIL','requested_studies':list(identifiers),'steps':records,
        'python':sys.version,'packages':packages,'pretrained_inference':False,'training':False,'offline':True,
        'separate_evidence_packs_required':False,'scientific_negative_results_count_as_execution_success':True,
        'unsupported_or_failed_execution_counts_as_success':False,'canonical_table_reconstruction':tables}
    write(output/'RUN_RECEIPT.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('list')
    quick=commands.add_parser('quick');quick.add_argument('--output',type=Path,required=True)
    updates=commands.add_parser('updates');updates.add_argument('--output',type=Path,required=True)
    paper=commands.add_parser('paper');paper.add_argument('--output',type=Path,required=True)
    paper.add_argument('--historical-python',type=Path,required=True)
    study=commands.add_parser('study');study.add_argument('study_id');study.add_argument('--output',type=Path,required=True)
    study.add_argument('--historical-python',type=Path)
    check=commands.add_parser('check-release');check.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    try:
        if args.command=='list':
            for row in json.loads((ROOT/'provenance/STUDY_CATALOG.json').read_bytes()):
                print(row['id']+'\t'+row['title']+'\t'+row['status'])
            return 0
        if args.command=='check-release':
            if args.root.resolve()!=ROOT:raise ValueError('Run from the extracted repository being checked')
            print(json.dumps(verify_release(ROOT),indent=2));return 0
        requested=(QUICK if args.command=='quick' else UPDATES if args.command=='updates' else ('historical_paper',*QUICK) if args.command=='paper' else (args.study_id,))
        result=execute(requested,args.output,getattr(args,'historical_python',None));print(json.dumps({'status':result['status'],'steps':len(result['steps'])}))
        return 0 if result['status']=='PASS' else 1
    except (OSError,ValueError,KeyError) as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
