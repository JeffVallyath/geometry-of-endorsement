"""CPU-only adapters to the frozen two-study analyzers."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(Path(path).read_bytes())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def compare(actual, expected):
    """Compare all scientific fields; never round or drop adverse outcomes."""
    actual, expected = Path(actual), Path(expected)
    if actual.name.endswith(('.jsonl', '.jsonl.gz')):
        from itertools import zip_longest
        opener = gzip.open if actual.suffix == '.gz' else open
        missing = object()
        with opener(actual, 'rt', encoding='utf8') as left, opener(expected, 'rt', encoding='utf8') as right:
            for index, (a, b) in enumerate(zip_longest(left, right, fillvalue=missing), 1):
                if a is missing or b is missing or json.loads(a) != json.loads(b):
                    raise ValueError('Scientific row mismatch: ' + actual.name + ':' + str(index))
    elif actual.suffix == '.json':
        if load(actual) != load(expected):
            raise ValueError('Scientific output mismatch: ' + actual.name)
    elif actual.read_bytes() != expected.read_bytes():
        raise ValueError('Output bytes differ: ' + actual.name)
    return actual.name


def baselines(source, output):
    sys.path.insert(0, str(source))
    import evaluate
    import report_behavior
    import report_work
    from update_tables import check_baseline_coverage, matched_snapshots, rows, training_exposure
    analysis = output / 'analysis'
    scientific = evaluate.evaluate(source/'raw', source/'evaluator_data', analysis)
    check_baseline_coverage(scientific)
    if scientific['comparison_validity'] != 'PASS':
        raise ValueError('Frozen baseline validity gate failed')
    behavior = report_behavior.report_behavior(source/'raw', analysis)
    work = report_work.report_work(source/'raw', source/'evaluator_data', analysis)
    if behavior['status'] != 'PASS' or work['status'] != 'PASS':
        raise ValueError('Component or workload replay failed')
    compared = [compare(path, source/'evaluation'/path.name) for path in sorted(analysis.iterdir())]
    snapshots = matched_snapshots(rows(analysis/'SCORED_LOGICAL.jsonl'))
    for endpoint, result in snapshots.items():
        saved = behavior['matched_target_snapshot_correctness'][endpoint]
        if (result['original_correct'], result['current_correct'], result['matched_reads']) != (
                saved['pre_target_correct'], saved['current_target_correct'], saved['pre_target_reads']):
            raise ValueError('Exact snapshot join differs from original component report')
    with (analysis/'MATCHED_SNAPSHOTS.json').open('x', encoding='utf8') as stream:
        json.dump(snapshots, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    exposure_audit=training_exposure(source,source.parent/'repair_qwen7/input/PPR1_Same_Case_Public_Repair/audit/SUPERVISED_EXPOSURE.json')
    with (analysis/'TRAINING_EXPOSURE_JOIN.json').open('x',encoding='utf8') as stream:
        json.dump(exposure_audit,stream,indent=2,ensure_ascii=False);stream.write('\n')
    return {'status':'PASS', 'kind':'complete_frozen_cpu_analyzer_and_component_replays',
            'logical_reads':55536, 'endpoints':8, 'matched_reads_per_snapshot_per_endpoint':1536,
            'compared_outputs':compared, 'exploratory_snapshot_join':True}


def exposure(source, output):
    sys.path[:0] = [str(source/'runtime/payload/implementation'), str(source/'runtime/payload/vendor')]
    original = module(source/'tools/analyze_exposure.py', 'frozen_exposure_analysis')
    original.ROOT = source
    original.PAYLOAD = source/'runtime/payload'
    original.INPUT = source/'sources/ppr1'
    original.PREPARED = source/'data/schedule'
    original.PROTOCOL = source/'protocol/exposure_v1.json'
    analysis = output/'analysis'
    original.evaluate(source/'raw/exposure', analysis)
    compared = [compare(path, source/'evidence/exposure'/path.name) for path in sorted(analysis.iterdir())]
    from update_tables import exposure_errors, rows
    categories = exposure_errors(rows(analysis/'scored.jsonl.gz'))
    metrics = load(analysis/'METRICS.json')
    for key, result in categories.items():
        if result['robust_question_harms'] != metrics['paired_harms'][key]['introduced']:
            raise ValueError('New error classification disagrees with frozen harm ledger')
    with (analysis/'ERROR_CATEGORIES.json').open('x', encoding='utf8') as stream:
        json.dump(categories, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    recovered = bridge(source, output/'bridge')
    return {'status':'PASS', 'kind':'complete_frozen_exposure_cpu_analyzer',
            'logical_reads':metrics['logical_reads'], 'saved_calls':metrics['raw_calls'],
            'compared_outputs':compared, 'error_categories_recomputed':True,
            'recovered_bridge':recovered}


def bridge(source, output):
    output.mkdir()
    nsr=source/'sources/nsr1'
    follow=source.parent/'original_public_history'
    env=dict(os.environ,NSR1_POD_OUT=str(nsr/'runs/pod_out'),NSR1_A10_SCORES=str(nsr/'inputs/a10_scores'))
    tasks=[('a10',[str(Path(__file__).with_name('a10_recompose.py')),str(source),'--output',str(output/'a10')],source),
           ('nsr1',['evaluator/analyze.py','full','.',str(output/'NSR1_ANALYSIS.json')],nsr/'implementation/source'),
           ('followups',['-m','src.repro.completed_followups','--output',str(output/'followups')],follow)]
    for name,args,cwd in tasks:
        with (output/(name+'.log')).open('xb') as stream:
            process=subprocess.run([sys.executable,'-B','-X','utf8',*args],cwd=cwd,env=env,stdout=stream,
                stderr=subprocess.STDOUT,timeout=900,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:
            raise ValueError('Recovered bridge analyzer failed: '+name)
    compared=[]
    for actual,expected in [(output/'a10/A10_RECOMPOSITION.json',source/'evidence/A10_RECOMPOSITION.json'),
                            (output/'a10/A10_RECOMPOSITION_TARGETS.csv',source/'evidence/A10_RECOMPOSITION_TARGETS.csv'),
                            (output/'NSR1_ANALYSIS.json',source/'evidence/NSR1_ANALYSIS.json'),
                            (output/'followups/VERIFICATION.json',source/'evidence/FOLLOWUP_VERIFICATION.json')]:
        compared.append(compare(actual,expected))
    audit=module(source/'tools/audit_evidence.py','frozen_bridge_audit')
    audit.NSR=nsr;audit.FOLLOW=follow/'reproducibility/completed_followups';audit.OUT=output
    for name,value in [('a10_schedule_reconciliation.json',audit.a10()),
                       ('carrier_mediation_recomputed.json',audit.followups())]:
        if value!=load(source/'evidence'/name):raise ValueError('Bridge output mismatch: '+name)
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
        compared.append(name)
    compared.append(compare(output/'a10_exact_id_join.jsonl.gz',source/'evidence/a10_exact_id_join.jsonl.gz'))
    return {'status':'PASS','compared_outputs':compared,'original_zip_replaced_by_verified_member_mapping':True,
            'shared_native_cache_source':'studies/original_public_history',
            'deterministic_compatibility':'Retained source report only; not a new replay or pooled with the neural compiler'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['same_case_baselines', 'history_exposure'])
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a fresh output directory outside the source tree')
    output.mkdir(parents=True)
    result = (baselines if args.study == 'same_case_baselines' else exposure)(source, output)
    result.update(study=args.study, python=sys.version, pretrained_inference=False, training=False)
    with (output/'RECONSTRUCTION_RECEIPT.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'study':args.study, 'status':result['status']}))


if __name__ == '__main__':
    main()
