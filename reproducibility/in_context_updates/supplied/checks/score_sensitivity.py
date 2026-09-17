"""Score the SAME saved model responses with LP1 and original v2 registries.
No new inference. Only the primary LP1 report may be used for the amended gate;
union sensitivity is descriptive and must not replace it after seeing outcomes.
"""
import argparse,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from icmh.analyze import analyze
from icmh.common import read_json,write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=['development','qualification','final','repeat_670','repeat_618'])
    p.add_argument('--journal',action='append',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seconds-available',type=float)
    p.add_argument('--remaining-units',type=int,default=0)
    p.add_argument('--remaining-processes',type=int,default=0)
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    inp=ROOT/'inputs/built'
    pa=args.out_dir/'PRIMARY_LP1.json';sa=args.out_dir/'SENSITIVITY_UNION_V2.json'
    for f in [pa,sa,args.out_dir/'PARSER_SENSITIVITY_COMPARISON.json']:
        if f.exists():raise FileExistsError(f'Use a fresh output directory; preserving {f}')
    common=(args.seconds_available,args.remaining_units,args.remaining_processes)
    analyze(inp,args.phase,args.journal,pa,*common)
    with tempfile.TemporaryDirectory(prefix='icmh-union-sensitivity-') as tmp:
        tmp=Path(tmp)
        for name in [f'PLAN_{args.phase}.json',f'SCORING_{args.phase}.json','ROOTS.json']:
            shutil.copyfile(inp/name,tmp/name)
        shutil.copyfile(inp/'REGISTRY_UNION.json',tmp/'REGISTRY.json')
        analyze(tmp,args.phase,args.journal,sa,*common)
    primary,sensitivity=read_json(pa),read_json(sa)
    changes=[]
    for qid,row in primary['per_question'].items():
        old=sensitivity['per_question'][qid]
        delta={k:{'LP1':row[k],'union_v2':old[k]} for k in ['A','C','D','W','A_and_C'] if row[k]!=old[k]}
        if delta:changes.append({'question_id':qid,'root_id':row['root_id'],'differences':delta})
    result={'phase':args.phase,'same_model_responses':True,'primary_rule':'LP1',
        'sensitivity_rule':'union_v2','changed_question_count':len(changes),'changes':changes,
        'limitations':['Label-priority is a stipulated response convention, not an independent inference of model intent.',
         'Sensitivity disagreement does not justify retrospective parser or gate selection.',
         'No extra inference or additional independent roots are supplied by this comparison.']}
    write_json(args.out_dir/'PARSER_SENSITIVITY_COMPARISON.json',result)
    print(json.dumps({'phase':args.phase,'changed_questions':len(changes)}))
if __name__=='__main__':main()
