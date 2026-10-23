"""Subprocess adapters around original scientific analyzers, without cloud imports.

Run as a script so studies with legacy module names remain isolated from each
other. The recorded source implementations perform the scientific computation.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def scientific_equal(actual, expected, name):
    """Only source-declared newly measured CPU durations are variable."""
    def scrub(value):
        if isinstance(value,dict):
            ignored={'cpu_query_seconds','cpu_store_and_query_seconds'}
            if name=='TIMINGS.json':ignored.add('complete_cpu_evaluator_seconds')
            if name=='EIGHT_FIXED_TRACES.json':ignored |= {'cpu_intake_seconds','cpu_execution_seconds'}
            return {k:scrub(v) for k,v in value.items() if k not in ignored}
        if isinstance(value,list):return [scrub(v) for v in value]
        return value
    if scrub(actual)!=scrub(expected):
        raise ValueError('SCIENTIFIC_REPLAY_MISMATCH: '+name)


def compare_directory(actual,expected):
    names={p.name for p in Path(expected).iterdir() if p.is_file()}
    produced={p.name for p in Path(actual).iterdir() if p.is_file()}
    if names != produced:
        raise ValueError('Scientific output inventory mismatch: '+str(sorted(names^produced)))
    for name in sorted(names):
        left,right=Path(actual)/name,Path(expected)/name
        a,b=(rows(left),rows(right)) if name.endswith(('.jsonl','.jsonl.gz')) else (load(left),load(right))
        scientific_equal(a,b,name)
    return sorted(names)


def qwen7(source,output):
    import importlib.metadata
    if importlib.metadata.version('transformers')!='4.56.2':
        raise RuntimeError('TOKENIZER_REPLAY_REQUIRES_TRANSFORMERS_4.56.2')
    sys.path.insert(0,str(source/'controller'))
    import replay_result
    original_dump=replay_result.dump
    def receipt_only(path,value):
        if Path(path)!=source/'REPLAY_VERIFICATION.json':
            raise ValueError('Unexpected source write')
        original_dump(output/'REQUEST_REPLAY.json',value)
    replay_result.dump=receipt_only
    result=replay_result.replay(source,output/'analysis')
    # The original request replay verifies every call, recomputes metrics, and
    # compares both the complete summary and subject vectors to recorded values.
    return {'status':'PASS','kind':'recorded_call_reconstruction',**result}


def qwen32(source,output):
    sys.path.insert(0,str(source))
    import analyze_run
    import complete_reporting
    analyze_run.ROOT=output
    analyze_run.analyze(source/'results')
    complete_reporting.build_reports(source/'results',output/'analysis')
    compared=compare_directory(output/'analysis',source/'analysis')
    return {'status':'PASS','kind':'recorded_call_reconstruction','analysis_files_compared':compared}


def gemma(source,output):
    impl=source/'source/Gemma27_Same_Case_Repair/implementation'
    sys.path.insert(0,str(impl))
    import gemma_evaluate
    import gemma_validate_controls
    evidence=source/'remote-evidence'
    gemma_evaluate.evaluate(evidence/'base/responses.jsonl',evidence/'seed0/responses.jsonl',
                            evidence/'seed1/responses.jsonl',output/'analysis')
    for arm,folder in [('BASE','base'),('0','seed0'),('1','seed1')]:
        kwargs={} if arm=='BASE' else {
            'loader_expected':evidence/('train'+arm)/'LOADER_EXPECTED.json',
            'loader_actual':evidence/(folder+'_loader')/'responses.jsonl'}
        result=gemma_validate_controls.validate(evidence/folder/'responses.jsonl',
            evidence/(folder+'_repeat')/'responses.jsonl',evidence/(folder+'_return')/'responses.jsonl',**kwargs)
        if arm!='BASE' and not result['loader']['pass']:
            raise ValueError('Recorded loader agreement failed')
        dump(output/'analysis'/('controls_'+arm+'.json'),result)
    # The original supplementary analyzer expects these raw observations below
    # its output root. Copies are new local products, never source overwrites.
    for folder in ['base','seed0','seed1','base_return','seed0_return','seed1_return']:
        target=output/'remote-evidence'/folder/'responses.jsonl'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(evidence/folder/'responses.jsonl',target)
    supplement=module(source/'complete_analysis.py','frozen_gemma_supplement')
    supplement.RUN=output
    supplement.main()
    compared=compare_directory(output/'analysis',source/'analysis')
    return {'status':'PASS','kind':'recorded_call_reconstruction','analysis_files_compared':compared,
            'main_logical_outcomes':len(rows(output/'analysis/ALL_OUTCOMES.jsonl.gz'))}


def paired(source,output):
    sys.path.insert(0,str(source/'MQMH_SCALE_V1/implementation'))
    import study
    native=source/'evidence/native'
    study.analyze(native/'small-primary/raw.jsonl',native/'large-primary/raw.jsonl',output/'analysis')
    # Exact integer identities and original floating policies are checked by
    # the original endpoint implementation; source-hash path keys are metadata.
    scientific_equal(load(output/'analysis/SUMMARY.json'),load(source/'evidence/analysis/SUMMARY.json'),'SUMMARY.json')
    sys.path.insert(0,str(source/'execution'))
    import audit_results
    result=audit_results.supplementary(native/'small-primary/raw.jsonl',native/'large-primary/raw.jsonl',
        native/'small-repeat/raw.jsonl',native/'large-repeat/raw.jsonl',output/'supplementary')
    expected=load(source/'evidence/supplementary/SUPPLEMENTARY_SUMMARY.json')
    actual=dict(result);expected=dict(expected)
    if sorted(actual['source_hashes'].values()) != sorted(expected['source_hashes'].values()):
        raise ValueError('Supplementary raw-input hashes differ')
    actual.pop('source_hashes');expected.pop('source_hashes')
    scientific_equal(actual,expected,'SUPPLEMENTARY_SUMMARY.json')
    token_output=output/'tokenization';token_output.mkdir()
    token_receipt=paired_tokenization(source,token_output)
    return {'status':'PASS','kind':'recorded_call_reconstruction','primary_and_repeats':True,
            'exact_token_and_request_verification':token_receipt,
            'path_dependent_metadata_excluded':['SUPPLEMENTARY_SUMMARY.json: source_hashes path keys']}


def selection(source,output):
    sys.path.insert(0,str(source/'implementation'))
    import evaluate
    evaluate.evaluate(source,output/'analysis')
    scientific_equal(load(output/'analysis/SUMMARY.json'),load(source/'results/SUMMARY.json'),'SUMMARY.json')
    return {'status':'PASS','kind':'retrospective_saved_score_reconstruction',
            'receipt':load(output/'analysis/RUN_RECEIPT.json')}


def paired_tokenization(source,output):
    sys.path.insert(0,str(source/'execution'))
    import verify_native
    receipts={}
    for size in ['small','large']:
        for phase in ['primary','repeat']:
            requests=source/'evidence/runtime'/('runtime_requests.jsonl' if phase=='primary' else size+'_repeat_requests.jsonl')
            key=size+'-'+phase
            receipt=verify_native.verify(source/'evidence/native'/key,requests,
                source/'evidence/tokenization-v2'/size,source/'evidence/SCIENTIFIC_FREEZE.json')
            scientific_equal(receipt,load(source/'evidence/native'/key/'TOKEN_VERIFICATION.json'),'TOKEN_VERIFICATION.json')
            receipts[key]=receipt
            dump(output/(key+'.json'),receipt)
    return {'status':'PASS','kind':'recorded_token_and_request_reconstruction','checks':receipts}


def answer_type(source,output):
    sys.path.insert(0,str(source/'implementation'))
    import evaluate
    evaluate.evaluate(output/'analysis')
    import independent_check
    independent_check.main(output/'analysis')
    names=['SUMMARY.json','ROOTS.json','TYPE_COVERAGE.json','REFERENCE_PRUNED.json','SEED_DISAGREEMENT.json',
           'ALL_DECISIONS.jsonl.gz','ALL_CHANGES.jsonl.gz']
    for name in names:
        reader=rows if name.endswith('.gz') else load
        scientific_equal(reader(output/'analysis'/name),reader(source/'results_v2'/name),name)
    return {'status':'PASS','kind':'retrospective_saved_score_reconstruction','compared':names,
        'receipt':load(output/'analysis/RUN_RECEIPT.json'),'independent':load(output/'analysis/INDEPENDENT_CHECK.json')}


def external_validation(source,output):
    original=module(source/'external_evaluate.py','frozen_external_evaluation')
    supplied=source/'external/transport_v2'
    result=original.analyze(supplied/'raw/attempt_001',load(supplied/'evaluator/SOURCE_MAPPING.json'))
    scientific_equal(result,load(supplied/'RESULTS.json'),'EXTERNAL_RESULTS.json')
    dump(output/'RESULTS.json',result)
    return {'status':'PASS','kind':'historical_saved_observation_analysis',
        'roots_per_arm':4,'arms':len(result['summary']),'source_wording_amendment_preserved':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['repair_qwen7','repair_qwen32','repair_gemma27','paired_native_scale','selection_admission','paired_tokenization','answer_type_filter','public_external_validation'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh analysis output directory')
    if args.output.resolve().is_relative_to(args.source.resolve()):raise ValueError('Output must not modify input tree')
    args.output.mkdir(parents=True)
    adapters={'repair_qwen7':qwen7,'repair_qwen32':qwen32,'repair_gemma27':gemma,
              'paired_native_scale':paired,'selection_admission':selection,'paired_tokenization':paired_tokenization,
              'answer_type_filter':answer_type,'public_external_validation':external_validation}
    result=adapters[args.study](args.source.resolve(),args.output.resolve())
    result.update(study=args.study,python=sys.version,pretrained_inference=False,training=False)
    dump(args.output/'RECONSTRUCTION_RECEIPT.json',result)
    print(json.dumps({'study':args.study,'status':result['status']}))


if __name__=='__main__':main()
