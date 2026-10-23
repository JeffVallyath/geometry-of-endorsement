"""Run retained historical CPU analyzers in isolated subprocesses.

Original analyzers, populations and scientific comparisons remain authoritative.
This adapter changes output locations and supervision, not measured inputs.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

SEMANTIC={'neurosymbolic_v1','semantic_compilation_v2','role_preservation_v3',
          'language_canonicalization_v4','contract_selection_v5','bound_references_v6',
          'name_invariant_compilation','supervised_compiler','compound_updates'}
MODULAR={'versioned_kv_v1','materialized_views_v2','materialized_views_v3','frozen_reader_boundary'}


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')


def environment(source,output,pythonpath,extra=None):
    allowed={'systemroot','windir','path','comspec','pathext','lang','lc_all'}
    env={k:v for k,v in os.environ.items() if k.lower() in allowed}
    runtime=output/'_runtime';runtime.mkdir(exist_ok=True)
    env.update(PYTHONPATH=os.pathsep.join([str(Path(__file__).parent/'offline_site'),str(pythonpath)]),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',
        PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED','0'),REPRO_OFFLINE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',HF_HOME=str(runtime/'hf'),CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MPLBACKEND='Agg',MPLCONFIGDIR=str(runtime/'matplotlib'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TEMP=str(runtime),TMP=str(runtime),TMPDIR=str(runtime))
    if 'REPRO_DEADLINE_UTC' in os.environ:env['REPRO_DEADLINE_UTC']=os.environ['REPRO_DEADLINE_UTC']
    env.update(extra or {})
    return env


def run(source,output,commands,*,pythonpath,extra=None):
    env=environment(source,output,pythonpath,extra)
    receipts=[]
    for label,cwd,args in commands:
        started=time.monotonic();print('RUN '+label,flush=True)
        result=subprocess.run([sys.executable,'-B',*map(str,args)],cwd=cwd,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
            timeout=1200,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        text=result.stdout
        for path,alias in [(source,'<source>'),(output,'<output>'),(Path(sys.prefix),'<python-env>')]:
            for value in (str(path),path.as_posix()):
                text=text.replace(json.dumps(value)[1:-1],alias).replace(value,alias)
        (output/(label+'.log')).write_text(text,encoding='utf-8',newline='\n')
        receipts.append({'step':label,'status':'PASS' if result.returncode==0 else 'FAIL',
            'returncode':result.returncode,'seconds':round(time.monotonic()-started,3),'log':label+'.log'})
        if result.returncode:break
    return receipts


def baseline(source,output):
    sys.path.insert(0,str(source/'src'))
    from repro.paper_results import jobs
    from repro.release_manifest import verify
    integrity=verify()
    commands=[(label,source,args) for label,args in jobs(output/'analyses')]
    receipts=run(source,output,commands,pythonpath=source/'src')
    return {'status':'PASS' if len(receipts)==len(commands) and all(r['status']=='PASS' for r in receipts) else 'FAIL',
        'kind':'historical_saved_score_reconstruction','steps':receipts,'input_integrity':integrity,
        'fresh_pretrained_execution':False,'retraining':False}


def original_public_history(source,output):
    receipts=run(source,output,[('original-public-history',source,
        ['-m','repro.mqmh','--output',output/'analysis'])],pythonpath=source/'src')
    return {'status':'PASS' if all(r['status']=='PASS' for r in receipts) else 'FAIL',
        'kind':'recorded_call_reconstruction','steps':receipts,'original_rules_unchanged':True}


def differences(actual,expected,tolerance=0,path='$'):
    if isinstance(expected,dict) and isinstance(actual,dict):
        if set(actual)!=set(expected):return [path+': keys differ']
        found=[]
        for key in expected:
            found.extend(differences(actual[key],expected[key],tolerance,path+'.'+key))
            if len(found)>=12:break
        return found[:12]
    if isinstance(expected,list) and isinstance(actual,list):
        if len(actual)!=len(expected):return [path+': lengths differ']
        found=[]
        for number,(a,b) in enumerate(zip(actual,expected)):
            found.extend(differences(a,b,tolerance,path+'['+str(number)+']'))
            if len(found)>=12:break
        return found[:12]
    if isinstance(expected,float) and isinstance(actual,(int,float)) and tolerance:
        return [] if abs(actual-expected)<=tolerance*max(1,abs(actual),abs(expected)) else [path+': float differs']
    return [] if actual==expected else [path+': value differs']


def semantic(identifier,source,output):
    impl=source/'implementation/source';rec=source/'records';commands=[];compare=[];extra={};reused=[]
    def add(label,args):commands.append((label,impl,args))
    def expected(name):compare.append((output/name,rec/name))
    if identifier=='neurosymbolic_v1':
        extra={'NSR1_POD_OUT':str(source/'runs/pod_out'),'NSR1_A10_SCORES':str(source/'inputs/a10_scores')}
        shutil.copyfile(rec/'CPU_TESTS.json',output/'CPU_TESTS.json');reused=['CPU_TESTS.json']
        add('analyze',['evaluator/analyze.py','full','.',output/'NSR1_ANALYSIS.json'])
        add('verdict',['evaluator/verdict.py',output])
        expected('NSR1_ANALYSIS.json');expected('NSR1_VERDICT.json')
    elif identifier=='semantic_compilation_v2':
        extra={'NSR2_POD_OUT':str(source/'runs/pod_out/final'),'NSR2_DATA':str(impl/'data'),
               'NSR2_RECORDS':str(rec),'NSR1_ROOT':str(source/'inputs/nsr1_subset')}
        add('analyze',['evaluator2/analyze2.py','full','.',output/'NSR2_ANALYSIS.json'])
        add('verdict',['evaluator2/verdict2.py',output/'NSR2_ANALYSIS.json',output/'NSR2_VERDICT.json'])
        expected('NSR2_ANALYSIS.json');expected('NSR2_VERDICT.json');reused=['source records read by frozen analyzer']
    elif identifier=='role_preservation_v3':
        extra={'NSR3_POD_OUT':str(source/'runs/pod_out/final'),'NSR3_RECORDS':str(rec)}
        add('analyze',['evaluator3/analyze3.py','full','.',output/'NSR3_ANALYSIS.json'])
        add('verdict',['evaluator3/verdict3.py',output/'NSR3_ANALYSIS.json',output/'NSR3_VERDICT.json'])
        add('saved-generation-replay',['tools/self_replay.py','data3',source/'runs/pod_out/final/qwen/fresh',
            'micro,core,graph,temporal,factorial,negative_nl',output/'raw_replay',output/'SELF_REPLAY.json'])
        expected('NSR3_ANALYSIS.json');expected('NSR3_VERDICT.json');reused=['source gate receipts read by frozen analyzer']
    elif identifier=='language_canonicalization_v4':
        add('records',['tools/export_delivery4.py','records','--runs',source/'runs','--out',output])
        add('saved-generation-replay',['tools/self_replay4.py','--data','data4','--run',source/'runs/pod_out/final/qwen/fresh',
            '--panels','core,graph,temporal,controls','--out',output/'SELF_REPLAY.json'])
        expected('NSR4_ANALYSIS.json');expected('NSR4_VERDICT.json')
        reused=['FORMAL_QUALIFICATION.json','A10_COMPAT_COVERAGE.json']
    elif identifier=='contract_selection_v5':
        for label,mod in [('analyze','evaluator5.analyze5'),('exactness','evaluator5.exact5')]:
            add(label,['-m',mod,'--data','data5','--runs',source/'runs/pod_out/final/qwen',
                       '--jev',source/'runs/jev/final','--out',output])
        add('saved-generation-replay',['tools/self_replay5.py','--runs',source/'runs/pod_out/final/qwen','--out',output/'SELF_REPLAY.json'])
        expected('NSR5_ANALYSIS.json');expected('EXACTNESS.json')
    elif identifier=='bound_references_v6':
        for label,mod in [('analyze','evaluator6.analyze6'),('exactness','evaluator6.exact6')]:
            add(label,['-m',mod,'--data','data6','--runs',source/'runs/pod_out/final','--out',output])
        add('saved-generation-replay',['tools/self_replay6.py','--runs',source/'runs/pod_out/final','--out',output/'SELF_REPLAY.json'])
        add('request-regeneration',['tools/dump_requests6.py','--runs',source/'runs/pod_out/final',
                                  '--out',output/'requests','--record',output/'REQUESTS_REGENERATED.json'])
        for name in ['NSR6_ANALYSIS.json','EXACTNESS.json','SELF_REPLAY.json','REQUESTS_REGENERATED.json']:expected(name)
    elif identifier=='name_invariant_compilation':
        for label,mod in [('analyze','evaluator1.analyze1'),('exactness','evaluator1.exact1'),('equivariance','evaluator1.equivariance1')]:
            add(label,['-m',mod,'--data','data1','--runs',source/'runs/pod_out/final','--out',output])
        add('saved-generation-replay',['tools/self_replay1.py','--runs',source/'runs/pod_out/final','--out',output/'SELF_REPLAY.json'])
        for name in ['NIC1_ANALYSIS.json','EXACTNESS.json','EQUIVARIANCE.json','SELF_REPLAY.json']:expected(name)
    elif identifier=='supervised_compiler':
        add('analyze',['-m','analysis.analyze','--run',source/'runs/pod_out/final','--data','data',
            '--config',source/'governing_inputs/SCA1_CONFIG.json','--nic1-data','ref_nic1/data1',
            '--nic1-final',source/'nic1_reference/final','--train-out',source/'runs/pod_out/train','--out',output/'SCA1_ANALYSIS.json'])
        add('verdict',['-m','analysis.verdict','--analysis',output/'SCA1_ANALYSIS.json','--records',rec,'--out',output/'SCA1_VERDICT.json'])
        expected('SCA1_ANALYSIS.json');expected('SCA1_VERDICT.json');reused=['source gate receipts read by frozen verdict']
    elif identifier=='compound_updates':
        add('analyze',['-m','tanalysis.analyze','--run',source/'runs/pod_out/final','--data','data_tca1','--sca1-data','data',
            '--sca1-final',source/'sca1_reference/final','--failed-inputs',source/'governing_inputs/audit/FAILED_MAIN_INPUTS.json',
            '--train-out',source/'runs/pod_out/train','--config',source/'governing_inputs/TCA1_CONFIG.json','--out',output/'TCA1_ANALYSIS.json'])
        add('verdict',['-m','tanalysis.verdict','--analysis',output/'TCA1_ANALYSIS.json','--records',rec,'--out',output/'TCA1_VERDICT.json'])
        expected('TCA1_ANALYSIS.json');expected('TCA1_VERDICT.json');reused=['source gate receipts read by frozen verdict']
    receipts=run(source,output,commands,pythonpath=impl,extra=extra)
    passed=len(receipts)==len(commands) and all(r['status']=='PASS' for r in receipts)
    comparisons=[]
    if passed:
        for left,right in compare:
            if not left.is_file() or not right.is_file():
                comparisons.append({'file':left.name,'status':'FAIL','reason':'Missing expected or reproduced output'});continue
            tolerance=1e-9 if identifier=='language_canonicalization_v4' else 0
            diff=differences(json.loads(left.read_text(encoding='utf8')),json.loads(right.read_text(encoding='utf8')),tolerance)
            comparisons.append({'file':left.name,'status':'FAIL' if diff else 'PASS','differences':diff,'float_tolerance':tolerance})
        passed=all(r['status']=='PASS' for r in comparisons)
    return {'status':'PASS' if passed else 'FAIL','kind':'historical_saved_observation_analysis',
            'steps':receipts,'comparisons':comparisons,'historical_gate_receipts_reused_not_new_tests':reused,
            'original_verdicts_not_reclassified':True}


def modular(identifier,source,output):
    impl=source/'implementation/source';rec=source/'records';commands=[]
    ignored={}
    if identifier=='versioned_kv_v1':
        impl=impl/'versioned_kv';analysis='PILOT_ANALYSIS.json';verdict='PILOT_VERDICT.json'
        limits=impl/'package/LIMITS_AND_GATES.json'
        # The delivered top-level ledgers include later shutdown accounting.
        # Reuse the exact administrative snapshot bound into the old verdict;
        # the analysis and every scientific gate are still recomputed.
        original=json.loads((rec/verdict).read_bytes())
        dump(output/'VERDICT_CLOCK_SNAPSHOT.json',original['clock'])
        dump(output/'VERDICT_COST_SNAPSHOT.json',original['cost'])
        commands=[('analyze',impl,['-m','vkv.analyze','--out',source/'runs/pod_out/pilot',
            '--stage','pilot','--limits',limits,'--result',output/analysis]),
            ('verdict',impl,['-m','vkv.verdict','--analysis',output/analysis,'--dev',source/'runs/pod_out/dev',
            '--limits',limits,'--freeze',rec/'receipts/FREEZE.json','--cost',output/'VERDICT_COST_SNAPSHOT.json',
            '--clock',output/'VERDICT_CLOCK_SNAPSHOT.json','--out',output/verdict])]
        ignored={verdict:['written_utc']}
    elif identifier=='materialized_views_v2':
        analysis='PILOT_ANALYSIS.json';verdict='PILOT_VERDICT.json'
        commands=[('analyze',impl,['-m','mnv.analyze',source/'runs/pod_out',output/analysis]),
            ('verdict',impl,['-m','mnv.verdict',output/analysis,output/verdict])]
    elif identifier=='materialized_views_v3':
        analysis='MNV3_ANALYSIS.json';verdict='MNV3_VERDICT.json'
        commands=[('analyze',impl,['-m','mnv.analyze3',source/'runs/pod_out',output/analysis]),
            ('verdict',impl,['-m','mnv.verdict3',output/analysis,output/verdict])]
    elif identifier=='frozen_reader_boundary':
        analysis='MNV4_ANALYSIS.json';verdict='MNV4_VERDICT.json'
        commands=[('analyze',impl,['-m','mnv.analyze4',source/'runs/pod_out/eval',output/analysis,
            '--dev-dir',source/'runs/pod_out/dev']),
            ('verdict',impl,['-m','mnv.verdict4',output/analysis,output/verdict,'--candidate','R_BATCHED'])]
    else:raise ValueError('Unsupported modular route')
    receipts=run(source,output,commands,pythonpath=impl)
    passed=len(receipts)==len(commands) and all(r['status']=='PASS' for r in receipts)
    comparisons=[]
    if passed:
        for name in (analysis,verdict):
            actual=json.loads((output/name).read_bytes());expected=json.loads((rec/name).read_bytes())
            for key in ignored.get(name,[]):
                if key not in actual or key not in expected:raise ValueError('Expected clock field missing')
                del actual[key];del expected[key]
            diff=differences(actual,expected)
            comparisons.append({'file':name,'status':'FAIL' if diff else 'PASS','differences':diff,
                'ignored_non_scientific_clock_fields':ignored.get(name,[]),'float_tolerance':0})
        passed=all(r['status']=='PASS' for r in comparisons)
    return {'status':'PASS' if passed else 'FAIL','kind':'historical_saved_observation_analysis',
        'steps':receipts,'comparisons':comparisons,
        'historical_gate_receipts_reused_not_new_tests':['recorded dev, freeze and training checks'],
        'administrative_snapshot_policy':'original verdict-embedded clock and cost' if identifier=='versioned_kv_v1' else 'source records',
        'original_verdicts_not_reclassified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['repository','original_public_history',*sorted(SEMANTIC|MODULAR)])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    if output.is_relative_to(source):raise ValueError('Output must be outside source')
    output.mkdir(parents=True)
    try:
        if args.study=='repository':result=baseline(source,output)
        elif args.study=='original_public_history':result=original_public_history(source,output)
        elif args.study in MODULAR:result=modular(args.study,source,output)
        else:result=semantic(args.study,source,output)
    except Exception as error:
        result={'status':'FAIL','error':type(error).__name__+': '+str(error)}
    packages={}
    for name in ['numpy','scipy','torch','scikit-learn','pandas','transformers']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    result.update(study=args.study,python=sys.version,packages=packages,
        offline=True,pretrained_inference=False,training=False)
    dump(output/'RECONSTRUCTION_RECEIPT.json',result)
    print(json.dumps({'study':args.study,'status':result['status']}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
