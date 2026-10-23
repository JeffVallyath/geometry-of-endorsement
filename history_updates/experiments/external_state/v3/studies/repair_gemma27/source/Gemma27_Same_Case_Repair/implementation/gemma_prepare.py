"""Source-faithful CPU preparation. No model calls; gold remains evaluator-only."""
from pathlib import Path
import argparse, collections, json, shutil, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
INHERITED=ROOT/'inherited/PPR32_Repair_128'
sys.path.insert(0,str(INHERITED/'implementation'))
import prepare_data as inherited
from prepare_data import load,gzrows,dump,write_rows,canonical,digest,HISTORIES,ALL
from ppr_requests import answer_request
LEGACY=INHERITED/'legacy_ppr'

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        old=inherited.build(Path(temp))
        for p in (INHERITED/'prepared').iterdir():
            if p.is_file() and (Path(temp)/p.name).read_bytes()!=p.read_bytes():
                raise ValueError('INHERITED_PREPARATION_CHANGED: '+p.name)
    bank=gzrows(INHERITED/'prepared/REQUESTS.jsonl.gz')
    lookup={r['request_id']:r for r in bank}
    logical=gzrows(INHERITED/'prepared/LOGICAL_ROWS.jsonl.gz')
    cases={r['case_id']:r for r in load(LEGACY/'prepared/RUNTIME_CASES.json')}
    new_ids=[]
    for row in logical:
        if row['phase']!='post':continue
        req={'kind':'generate',**answer_request(cases[row['case_id']],row['history'],'post',row['question'],'HISTORY')}
        rid=digest(req)
        lookup.setdefault(rid,{'request_id':rid,**req})
        row['NATIVE_GEMMA']=rid;new_ids.append(rid)
    for name in ('EVALUATOR_ONLY.json','STATE_RECEIPTS.json','CONTROLS.json','REPEAT_REQUEST_IDS.json','RETURN_REQUEST_IDS.json','TRAIN_CANARY_REQUESTS.jsonl.gz'):
        shutil.copyfile(INHERITED/'prepared'/name,out/name)
    shutil.copyfile(INHERITED/'prepared/KNOWN_32B_WITNESSES.json',out/'QWEN9_CHALLENGE.json')
    write_rows(out/'REQUESTS.jsonl.gz',[lookup[k] for k in sorted(lookup)])
    write_rows(out/'LOGICAL_ROWS.jsonl.gz',logical)
    kinds=collections.Counter(r['kind'] for r in lookup.values())
    report={k:old[k] for k in ('subjects','post_questions','targets','controls','independently_built_source_conditions','six_way_identical_current_inputs','equal_current_query_groups','gold_executor_diagnostic_checks','candidate_max','training_draws','training_distinct_pairs','exact_training_question_overlap','repeat_cases','repeat_logical_reads_per_scorer','return_programs_per_scorer')}
    report.update(study='GEMMA27_SAME_CASE_REPAIR_V1',logical_rows=6942,new_logical_reads=40116,
        native_post_reads=5406,native_pre='NOT_MEASURED',unique_bank=len(lookup),unique_by_kind=dict(kinds),
        unique_native_requests=len(set(new_ids)),base_input_computations=len(lookup),
        trained_input_computations_each=kinds['score'],known_Qwen_challenge=9,own_Gemma_witnesses='NOT_MEASURED',
        tokenizer_or_GPU_validated=False,neural_inference_performed=False)
    dump(out/'PREPARATION.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(build(a.out),indent=2))
