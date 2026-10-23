"""Reconstruct paper measurements from the included recorded observations."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
from experiments._shared.release import ROOT,SECTIONS,verify,fresh_output,mount,environment,run_step

HISTORICAL={
 'matched_histories': [('primary','state_sufficiency'),('controls','cpu_robustness'),
                       ('fresh','fresh_history'),('qwen32','qwen32_full_forward'),('editors','learned_editors')],
 'cache_interventions': [('crossover','cache_crossover'),('crossover-extension','cache_crossover_extension'),
                        ('refresh','question_blind_repair'),('mechanism','completed_followups')],
}

def commands(sections,source,out,core,historical,gmello):
    v3=source/'v3'; legacy=v3/'studies/original_public_history'; shared=v3/'src/repro'
    result=[]
    for section in ('matched_histories','cache_interventions'):
        if section not in sections:continue
        if not historical:raise ValueError('This section needs --historical-python; see docs/reproduction.md')
        for label,module in HISTORICAL[section]:
            result.append((label,historical,['-m','repro.'+module,'--output',out/'analyses'/label],legacy))
        if section=='matched_histories':
            for mode in ('analyze','compare'):
                result.append(('independent-'+mode,historical,
                    ['reproducibility/state_sufficiency/independent_verification/reconstruct.py',mode,'--output',out/'analyses/independent'],legacy))
            result.extend([
                ('original_public_history',core,[shared/'legacy_driver.py','original_public_history','--source',legacy,'--output',out/'analyses/original_public_history'],v3),
                ('paired_native_scale',core,[shared/'source_driver.py','paired_native_scale','--source',v3/'studies/paired_native_scale','--output',out/'analyses/paired_native_scale'],v3)])
    if 'external_state' in sections:
        if 'matched_histories' not in sections:
            result.extend([
                ('original_public_history',core,[shared/'legacy_driver.py','original_public_history','--source',legacy,'--output',out/'analyses/original_public_history'],v3),
                ('paired_native_scale',core,[shared/'source_driver.py','paired_native_scale','--source',v3/'studies/paired_native_scale','--output',out/'analyses/paired_native_scale'],v3)])
        if 'cache_interventions' not in sections:
            if not historical:raise ValueError('The Table 6 cache-refresh comparator needs --historical-python')
            result.append(('refresh',historical,['-m','repro.question_blind_repair','--output',out/'analyses/refresh'],legacy))
        for study in ('repair_qwen7','repair_qwen32','repair_gemma27'):
            result.append((study,core,[shared/'source_driver.py',study,'--source',v3/'studies'/study,'--output',out/'analyses'/study],v3))
    elif 'matched_histories' in sections:
        result.append(('repair_gemma27',core,[shared/'source_driver.py','repair_gemma27','--source',v3/'studies/repair_gemma27','--output',out/'analyses/repair_gemma27'],v3))
    if sections & {'external_state','interpreter_checks'}:
        result.append(('history_exposure',core,[shared/'update_driver.py','history_exposure','--source',v3/'studies/history_exposure','--output',out/'analyses/history_exposure'],v3))
    if 'interpreter_checks' in sections:
        result.append(('same_case_baselines',core,[shared/'update_driver.py','same_case_baselines','--source',v3/'studies/same_case_baselines','--output',out/'analyses/same_case_baselines'],v3))
    if 'gmello_comparison' in sections:
        target=out/'analyses/gmello_comparison'
        for script,name in [('replay.py','CPU_REPLAY.json'),('verify_diagnostics.py','DIAGNOSTIC_VERIFICATION.json'),('evaluate.py',None),('report_results.py',None)]:
            result.append(('gmello-'+script[:-3],gmello or core,[script,'--raw',source/'gmello/raw','--prior',source/'prior','--out',target/name if name else target],source/'gmello'))
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all',action='store_true')
    mode.add_argument('--section',choices=SECTIONS)
    mode.add_argument('--verify',action='store_true')
    mode.add_argument('--test',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--historical-python',type=Path)
    parser.add_argument('--gmello-python',type=Path)
    parser.add_argument('--hash-seed',type=int,default=0)
    parser.add_argument('--diagnostics',action='store_true',help='Keep successful stage logs in the output directory')
    args=parser.parse_args(argv)
    for field,name in [('historical_python','historical'),('gmello_python','gmello')]:
        if getattr(args,field) is None:
            candidate=ROOT.parent/('.venv-'+name)/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
            if candidate.is_file():setattr(args,field,candidate)
    if args.historical_python:args.historical_python=args.historical_python.resolve()
    if args.gmello_python:args.gmello_python=args.gmello_python.resolve()
    integrity=verify()
    if args.verify:print(json.dumps(integrity));return 0
    if args.output is None:parser.error('--output is required for replay or tests')
    out=fresh_output(args.output)
    index=json.loads((ROOT/'metadata/study_index.json').read_bytes())
    source=mount(index,out/'_runtime/source')
    env=environment(source,out,args.hash_seed)
    sections=set(SECTIONS if args.all else [args.section] if args.section else [])
    todo=commands(sections,source,out,sys.executable,args.historical_python,args.gmello_python)
    if args.test:
        todo=[('release-protections',sys.executable,['-m','unittest','discover','-s','tests','-v'],ROOT),
              ('v3-protections',sys.executable,['-m','unittest','discover','-s','tests','-v'],source/'v3')]
    status={'status':'INCOMPLETE','sections':sorted(sections),'input_integrity':integrity,
            'steps':[],'saved_output_replay':True,'new_model_inference':False,'offline':True}
    try:
        for label,python,cmd,cwd in todo:
            print('Reconstructing '+label,flush=True)
            row=run_step(label,python,cmd,cwd,out,env,keep_log=args.diagnostics);status['steps'].append(row)
            if row['status']=='PASS' and label in {'original_public_history','paired_native_scale','repair_qwen7',
                                                   'repair_qwen32','repair_gemma27','history_exposure','same_case_baselines'}:
                receipt=out/'analyses'/label/'RECONSTRUCTION_RECEIPT.json'
                if not receipt.is_file() or json.loads(receipt.read_bytes()).get('status')!='PASS':
                    row.update(status='FAIL',reason='Missing or failed scientific reconstruction receipt')
            (out/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
            if row['status']!='PASS':raise RuntimeError('Replay failed: '+label)
        # Table/figure bindings are implemented in the paper-level exporter.
        if not args.test:
            from experiments._shared.paper_outputs import export
            status['paper_outputs']=export(source,out,sections)
            if args.historical_python and sections & {'matched_histories','cache_interventions','external_state'}:
                row=run_step('source-figures',args.historical_python,
                    [ROOT/'experiments/_shared/render_figures.py','--legacy',source/'v3/studies/original_public_history',
                     '--tables',out/'tables','--output',out/'figures'],ROOT,out,env,keep_log=args.diagnostics)
                status['steps'].append(row)
                if row['status']!='PASS':raise RuntimeError('Source figure rendering failed')
        status['inputs_after']=verify()
        status['status']='PASS'
    except Exception as error:
        status.update(status='FAIL',error=type(error).__name__+': '+str(error))
    (out/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    (out/'RESULTS.md').write_text('# Recorded-output reconstruction\n\n'+
        ('All requested analyses completed. See the generated tables and RUN_STATUS.json.\n' if status['status']=='PASS'
         else 'Reconstruction is incomplete. See RUN_STATUS.json and the named diagnostic log.\n'),encoding='utf-8')
    print(status['status'])
    return 0 if status['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
