"""Reader tables from verified frozen results; no inference or new estimators."""
import argparse
import csv
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main(source,run,output):
    proof=read(source/'CPU_REPLAY.json')
    if proof['status']!='PASS' or proof['final_cases']!=64 or proof['learned_seeds']!=[0,1]:
        raise ValueError('Complete two-seed CPU replay required')
    output.mkdir(parents=True,exist_ok=False)
    primary=read(source/'PRIMARY.json');utility=read(source/'UTILITY.json')
    losses={(r['root_id'],r['seed'],r['draw']):r for r in read(source/'WITHHELD_LOSSES.json')}
    tables={}
    def write(name,rows):
        if not rows:raise ValueError('Unexpected empty required table: '+name)
        with (output/name).open('x',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tables[name]=len(rows)
    write('PRIMARY_BY_SEED.csv',[dict(r,rate=r['matched']/r['denominator']) for r in primary['per_seed']])
    gates=[]
    for row in primary['seed_code_decomposition']:
        for gate,counts in row['gates'].items():
            gates.append(dict(seed=row['seed'],draw=row['draw'],gate=gate,denominator=row['denominator'],
                              contributing_cases=row['contributing_cases'][gate],**counts))
    write('PRIMARY_GATE_COUNTS.csv',gates)
    witnesses=[]
    for row in read(run/'PRIMARY_DETAILS.json'):
        if row['both_matched']:
            witnesses.append({k:row[k] for k in ('root_id','seed','draw','query_id','gold','structural_pairs','both_strong')}|
                {k:json.dumps(row[k],separators=(',',':')) for k in ('matched_pairs','strong_pairs','predictions','correct_probabilities','valid')})
    if len(witnesses)!=sum(r['matched'] for r in primary['per_seed']):raise ValueError('Witness rows differ')
    if witnesses:write('PRIMARY_WITNESSES.csv',witnesses)
    accepted=[];retained=[];exclusive=[];contrasts=[]
    for row in utility['per_seed_format']:
        base=dict(seed=row['seed'],draw=row['draw'],interface='literal No/Yes' if row['draw']==0 else 'A/B sensitivity',common_cases=row['common_cases'])
        for policy,counts in row['policies'].items():
            if counts['accepted']['count']+counts['rejected']!=row['common_cases']:raise ValueError('Acceptance coverage mismatch')
            accepted.append(dict(base,policy=policy,rejected=counts['rejected'],**counts['accepted']))
            for fraction,stats in counts['equal_retention'].items():
                if stats['count']!=math.floor(float(fraction)*row['common_cases']):raise ValueError('Retained denominator mismatch')
                retained.append(dict(base,policy=policy,retained_fraction=fraction,**stats))
        for direction in ('A_only','B_only'):
            if len(row[direction]['cases'])!=row[direction]['stats']['count']:raise ValueError('Exclusive coverage mismatch')
            for rid in row[direction]['cases']:
                loss=losses[rid,row['seed'],row['draw']]
                exclusive.append(dict(base,accepted_only_by=direction[0],root_id=rid,
                    **{k:loss[k] for k in ('task_error','any_error','valid_disagreement','native_error')}))
        lo,hi=row['interval95_50pct'] if row['interval95_50pct'] is not None else (None,None)
        contrasts.append(dict(base,B_minus_A_50pct=row['contrasts']['0.5'],interval95_low=lo,interval95_high=hi,
                              undefined_bootstrap_draws=row['undefined_bootstrap_draws']))
    write('UTILITY_ACCEPTANCE.csv',accepted);write('UTILITY_EQUAL_RETENTION.csv',retained)
    if exclusive:write('POLICY_DISAGREEMENT_CASES.csv',exclusive)
    write('UTILITY_SEED_CONTRASTS.csv',contrasts)
    repeat=[{k:r[k] for k in ('root_id','status','rows','tolerance','max_logp_delta')}|
            dict(score_discrepancies=len(r['all_discrepancies']),failed_rows=len(r['failed_rows'])) for r in read(source/'REPEATS.json')]
    write('REPEATS.csv',repeat)
    if (tables['PRIMARY_BY_SEED.csv'],tables['PRIMARY_GATE_COUNTS.csv'],tables['UTILITY_ACCEPTANCE.csv'],
        tables['UTILITY_EQUAL_RETENTION.csv'],tables['UTILITY_SEED_CONTRASTS.csv'],tables['REPEATS.csv'])!=(2,20,8,24,4,4):
        raise ValueError('Required table coverage differs')
    with (output/'TABLE_COVERAGE.json').open('x',encoding='utf8') as handle:
        json.dump(dict(status='PASS',rates_are_fractions_not_percent=True,analysis_unchanged=True,rows=tables),handle,indent=2)
    print(json.dumps(dict(status='PRESENTATION_TABLES_PASS',rows=tables)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.run,a.output)
