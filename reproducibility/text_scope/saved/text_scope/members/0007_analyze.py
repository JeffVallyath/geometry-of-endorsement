"""CPU-only complete-row analysis; missing rows are not zero-effect results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from text_scope.common import (DEFAULT_RUNTIME, CONTEXTS, METHODS, read, save, sha,
    source_root, shards, schedule, independent_evaluator)
from text_scope.metrics import decode_score, root_metrics, summarize, paired, jointly_certified
from text_scope import sensitivities


def load_design(runtime, rid):
    folder = shards(runtime) / rid
    return {key:read(folder/(filename+'.json')) for key,filename in
            [('root','root'),('questions','questions'),('source_inputs','source_inputs')]}


def historical(runtime, evaluator, actor, rid, design, protocol):
    base = source_root(runtime)/'reproducibility/state_sufficiency/independent_verification'
    panel = read(base/f'inputs/responses/{actor}/{rid}/core.json.gz')
    qindex = evaluator.validate_design(design,rid)
    index, _ = evaluator.index_panel(panel,design,qindex,'core',protocol)
    return index,{key:len(value['prefix_ids']) for key,value in panel['contexts'].items()}


def new_root(runtime, row, rid, design):
    folder = Path(runtime)/'results'/row/'final'/rid
    done = read(folder/'DONE.json')
    assert done['logical_scores']==17*34 and done['row_id']==row and done['root_id']==rid
    for filename, hashed in done['files'].items():
        assert sha(folder/filename)==hashed
    scores = read(folder/'SCORES.json')
    assert set(scores)==set(CONTEXTS)
    qids={q['query_id'] for q in design['questions']}
    assert all(set(v)==qids for v in scores.values())
    seal=read(folder/'CONTEXT_SEAL.json')
    return ({(c,q):decode_score(record) for c,values in scores.items() for q,record in values.items()},
            {key:len(value['cache_identity']['prefix_ids']) for key,value in seal['contexts'].items()})


def comparison_blocker(runtime,historical_row,complete):
    if not complete:
        return 'MISSING_COMPLETE_ROW'
    bridge=Path(runtime)/'checks'/('BRIDGE_'+historical_row+'.json')
    if not bridge.exists() or read(bridge).get('status')!='PASS':
        return 'HISTORICAL_BRIDGE_NOT_PASSED'
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=args.runtime
    assert not args.output.exists(), 'Use a fresh analysis output; preserve earlier results'
    spec=schedule(runtime)
    evaluator=independent_evaluator(runtime)
    base=source_root(runtime)/'reproducibility/state_sufficiency/independent_verification'
    protocol=read(base/'inputs/protocol.json')
    historical_counts=read(source_root(runtime)/'reproducibility/state_sufficiency/v10/root_counts.json')
    historical_primary=read(source_root(runtime)/'reproducibility/state_sufficiency/v10/primary_results.json')
    table=[]
    all_roots={}
    all_details={}
    dispositions=[]
    for row in spec['historical_rows']+spec['new_rows']:
        is_historical=row in spec['historical_rows']
        row_id=row['id']
        complete=[rid for rid in spec['final_root_ids'] if (runtime/'results'/row_id/'final'/rid/'DONE.json').exists()]
        missing_repeats=[phase for phase in ('repeat0','repeat1') if not (runtime/'results'/row_id/phase/'PHASE_DONE.json').exists()]
        if not is_historical and (len(complete)!=64 or missing_repeats):
            disposition=dict(row_id=row_id,status='NOT_MEASURED' if not complete else 'INCOMPLETE_ROW_NO_AGGREGATE',
                completed_roots=len(complete),required_roots=64,missing_roots=[rid for rid in spec['final_root_ids'] if rid not in complete],
                full_denominator=1152,aggregate_permitted=False,missing_repeat_passes=missing_repeats)
            dispositions.append(disposition)
            for method in METHODS:
                table.append(dict(row_id=row_id,method=method,historical=False,status=disposition['status'],
                    roots_measured=len(complete),roots_required=64,full_opportunities=1152,
                    witness_rate=None,conditional_witness=None,A_coverage=None,C_coverage=None,E_coverage=None,
                    strong=None,complete_case_success=None))
            continue
        roots_by_method={m:[] for m in METHODS}
        details_by_method={m:[] for m in METHODS}
        sensitivities_by_method={m:[] for m in METHODS}
        for rid in spec['final_root_ids']:
            design=load_design(runtime,rid)
            actor='gemma' if row_id=='gemma9_c1' else 'qwen'
            index,lengths=historical(runtime,evaluator,actor,rid,design,protocol) if is_historical else new_root(runtime,row_id,rid,design)
            for method in METHODS:
                root,details=root_metrics(evaluator,design,index,method)
                if is_historical:
                    expected=next(v for v in historical_counts if (v['actor'],v['root_id'],v['condition'])==(actor,rid,method))
                    assert all(root[k]==expected[k] for k in ('questions','A','C','AC','D','W','strong'))
                roots_by_method[method].append(root)
                details_by_method[method].extend(details)
                sensitivities_by_method[method].extend(sensitivities.evaluate(runtime,evaluator,design,index,method,lengths,details))
        for method in METHODS:
            all_roots[row_id,method]=roots_by_method[method]
            all_details[row_id,method]=details_by_method[method]
            result=summarize(roots_by_method[method],new_row=not is_historical)
            if is_historical:
                original=next(v for v in historical_primary if (v['actor'],v['group'])==(actor,method))
                assert abs(result['witness_rate']-original['mean'])<1e-12
                result['historical_original_family8_interval99375']=original['corrected_interval']
                result['historical_original_primary_family_unchanged']=True
            table.append(dict(row_id=row_id,method=method,historical=is_historical,
                status='HISTORICAL_RECONSTRUCTED' if is_historical else 'COMPLETE_FINAL_ROW',**result))
            save(args.output/'rows'/row_id/(method+'.json'),dict(roots=roots_by_method[method],queries=details_by_method[method],
                sensitivities=sensitivities_by_method[method],sensitivity_summary=sensitivities.summarize(sensitivities_by_method[method])))
    contrasts=[]
    for left,right in spec['paired_witness_contrasts']['pairs']:
        for method in METHODS:
            available=all((row,method) in all_roots for row in (left,right))
            blocker=comparison_blocker(runtime,right,available)
            if blocker:
                contrasts.append(dict(left=left,right=right,method=method,status='NOT_ASSESSABLE',
                    reason=blocker))
            else:
                contrasts.append(dict(left=left,right=right,method=method,status='MEASURED_BRIDGED',
                    witness_contrast=paired(all_roots[left,method],all_roots[right,method]),
                    jointly_certified=jointly_certified({row:all_details[row,method] for row in (left,right)})))
    common_three=[]
    for method in METHODS:
        rows=('qwen4_t4','qwen8_t4','qwen32_t4')
        blocker=comparison_blocker(runtime,'qwen8_t4',all((row,method) in all_details for row in rows))
        if not blocker:
            common_three.append(dict(method=method,**jointly_certified({row:all_details[row,method] for row in rows})))
        else:
            common_three.append(dict(method=method,status='NOT_ASSESSABLE',reason=blocker))
    save(args.output/'MAIN_TABLE.json',table)
    save(args.output/'PLANNED_COMPARISONS.json',contrasts)
    save(args.output/'QWEN_THREE_WAY_INTERSECTION.json',common_three)
    save(args.output/'ROW_DISPOSITIONS.json',dispositions)
    save(args.output/'ANALYSIS_BINDING.json',dict(schedule_sha256=sha(runtime/'handoff/SCHEDULE.json'),
         original_primary_not_modified=True,source_archive_sha256=read(runtime/'INTAKE.json')['source_sha256'],
         historical_primary_family_unchanged=True,new_primary_family_size=8,paired_contrast_family_size=6,
         no_new_inference_in_analysis=True))
    print(json.dumps(dict(status='CPU_TABLES_WRITTEN',rows=len(table),contrasts=len(contrasts),
                         not_measured_rows=len(dispositions),output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
