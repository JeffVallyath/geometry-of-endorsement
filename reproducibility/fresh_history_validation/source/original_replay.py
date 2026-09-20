"""CPU-only complete reconstruction; no model, GPU, provider or credential access."""
import argparse
import gzip
import json
import math
from pathlib import Path
from bindings import ROOT,load,save,sha,digest
import analyze as A
import decision as D
from neural import load_panel,panel_index

def metadata():
    return [(load(ROOT/'generated'/r['root_id']/'root.json'),load(ROOT/'generated'/r['root_id']/'questions.json'),load(ROOT/'generated'/r['root_id']/'PLAN.json'))
            for r in load(ROOT/'generated/INDEX.json') if r['split']=='FINAL']

def equivalent(actual,expected):
    """Inherited saved-score CPU reconstruction tolerance; classification exact."""
    if isinstance(actual,dict) and isinstance(expected,dict):
        return actual.keys()==expected.keys() and all(equivalent(actual[k],expected[k]) for k in actual)
    if isinstance(actual,list) and isinstance(expected,list):
        return len(actual)==len(expected) and all(equivalent(a,b) for a,b in zip(actual,expected))
    if type(actual) is float and type(expected) is float:
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=1e-12
    return type(actual) is type(expected) and actual==expected

def check_case(folder,design):
    root,questions,plan=design;panel=load_panel(folder);index=panel_index(panel)
    expected={q['query_id']:q for q in questions}
    for row in panel['physical']:
        q=expected[row['query_id']];tokens=panel['tokens'][row['context']][row['query_id']]
        if digest(tokens)!=row['tokens_sha256'] or tokens['question_sha256']!=__import__('hashlib').sha256(q['text'].encode()).hexdigest():
            raise ValueError('Response/token/question identity failure')
    for key,context in panel['contexts'].items():
        origin=0 if key=='NATIVE_FINAL' else int(key.split('/')[0][1:])
        if context['cache_identity']['prefix_length']!=plan['prefix_lengths'][origin]:raise ValueError('Frozen matching lengths changed')
    return panel,index

def main(run,out):
    if out.exists():raise FileExistsError('New replay output directory required')
    lock=load(run/'DECISION_LOCK.json')
    if sha(run/'RANKINGS_LOCKED.json')!=lock['rankings_sha256']:raise ValueError('Rankings changed')
    for rec in lock['receipts']:
        if sha(run/f"DECISIONS_{rec['procedure']}.json")!=rec['output_sha256']:raise ValueError('Decision lock changed')
    data=metadata();decoded={};details=[]
    for design in data:
        rid=design[0]['root_id'];panel,index=check_case(run/'final'/rid,design);decoded[rid]=index
        details.extend(A.primary(*design,index))
    decisions=[]
    for procedure in ('A','B'):
        packet=A.decision_packet(procedure,data,decoded);reproduced=D.decide(packet,load(ROOT/'DECISION_INPUT_SCHEMAS.json')[procedure])
        if reproduced!=load(run/f'DECISIONS_{procedure}.json'):raise ValueError('Decision replay mismatch')
        decisions.extend(reproduced)
    if A.rankings(decisions)!=load(run/'RANKINGS_LOCKED.json'):raise ValueError('Ranking replay mismatch')
    ids=[x[0]['root_id'] for x in data]
    summary=A.bootstrap_primary(details,ids)
    if not equivalent(details,load(run/'PRIMARY_DETAILS.json')):raise ValueError('Primary full-row CPU reconstruction mismatch')
    if not equivalent(summary,load(run/'PRIMARY_SUMMARY.json')):raise ValueError('Primary bootstrap reconstruction mismatch')
    losses=A.withheld_losses(data,decoded);utility=A.utility_summary(decisions,losses,ids)
    tokens={}
    for procedure in ('A','B'):
        packet=A.decision_packet(procedure,data,decoded);total_common=0;total_validation=0
        for case in packet['cases']:
            panel=load_panel(run/'final'/case['root_id'])
            for field in ('common','validation'):
                for row in case[field]:
                    c=panel['aliases'].get(row['context'],row['context']);tok=panel['tokens'][c][row['query_id']]
                    token_count=panel['contexts'][c]['cache_identity']['prefix_length']+len(tok['suffix_ids'])
                    if field=='common':total_common+=token_count
                    else:total_validation+=token_count
        tokens[procedure]=dict(common_full_prompt_token_sum=total_common,validation_full_prompt_token_sum=total_validation,
            semantics='Hypothetical uncached prompt token sum; actual execution shared compiled contexts; query equality is not GPU-cost equality')
    repeat=load(run/'REPEAT_SELECTION.json');from worker import compare
    repeated=[]
    for rid in repeat['root_ids']:
        result=compare(load_panel(run/'final'/rid),load_panel(run/'repeat'/rid))
        if result!=load(run/'repeat'/rid/'COMPARISON.json'):raise ValueError('Repeat reconstruction mismatch')
        repeated.append(dict(root_id=rid,**result))
    physical=0;compilation=0;prior_reads=0
    for line in (run/'READOUT_LEDGER.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['event']=='SCORED_READ_RESERVED':physical+=r['count']
        elif r['event']=='COMPILATION_FORWARDS':compilation+=r['count']
        elif r['event']=='PRIOR_ATTEMPT_WORK':
            prior_reads+=r['scored_readouts'];physical+=r['scored_readouts'];compilation+=r['compilation_forwards']
    if physical>50000 or len(details)!=2304 or len(repeated)!=4:raise ValueError('Completion/count contract failed')
    save(out/'PRIMARY.json',summary);save(out/'UTILITY.json',utility);save(out/'WITHHELD_LOSSES.json',losses)
    save(out/'REPEATS.json',repeated);save(out/'READ_COSTS.json',tokens)
    save(out/'CPU_REPLAY.json',dict(status='PASS',final_cases=64,learned_seeds=[0,1],coded_target_rows=2304,
        final_logical_readouts=36992,all_actual_scored_readout_reservations=physical,prior_attempt_scored_readouts=prior_reads,compilation_forwards=compilation,
        locked_decisions_and_rankings_reproduced=True,full_primary_rows_reproduced=True,repeat_cases=4))
    print(json.dumps(dict(status='CPU_REPLAY_PASS',final_cases=64,physical_readouts=physical,compilation_forwards=compilation)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run,a.output)
