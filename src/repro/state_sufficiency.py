"""Reconstruct compact saved summaries, not witness eligibility or neural inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, load_json, new_output, sha256

BASE = ROOT / 'reproducibility/state_sufficiency'
GROUPS = ('INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY', 'EXISTING_CORRECTION', 'LATEST_SAME_WORDING')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(np.allclose(actual, expected, rtol=0, atol=1e-12), 'Saved summary mismatch')


def reconstruct(base=BASE):
    read = lambda name: load_json(base / name)
    spec = read('v10/specification_summary.json')
    require(spec['joint_questions_per_root'] == 18 and spec['root_denominator'] == 64,
            'Fixed denominator changed')
    require(spec['bootstrap_draws'] == 10000 and spec['bootstrap_seed'] == 2609141002
            and spec['interval_level'] == .99375, 'Frozen bootstrap changed')
    counts = read('v10/root_counts.json')
    primaries = read('v10/primary_results.json')
    witnesses = read('v10/witness_summary.json')
    expected_roots = {f'SSC1-FINAL-{i:04d}' for i in range(64)}
    conditions = {'SOURCE','NATIVE_FINAL', 'INV_PAIR_NLL_s0', 'INV_PAIR_NLL_s1',
                  'FREE_PAIR_CONSISTENCY_s0', 'FREE_PAIR_CONSISTENCY_s1',
                  'EXISTING_CORRECTION','LATEST_SAME_WORDING'}
    # NATIVE_FINAL is not a separate condition in the source root table.
    conditions.remove('NATIVE_FINAL')
    identities = {(r['actor'], r['condition'], r['root_id']) for r in counts}
    require(len(counts) == len(identities) == 896, 'Missing or duplicate root records')
    require(identities == {(a,c,r) for a in ('gemma','qwen') for c in conditions for r in expected_roots},
            'Model/seed/root inventory changed')
    for r in counts:
        require(r['questions'] == 18, 'Ineligible questions must remain in fixed denominator')
        require(all(type(r[k]) is int and 0 <= r[k] <= 18 for k in ('A','C','AC','D','W','strong','all_direct_witness')),
                'Invalid witness count')
        require(r['strong'] <= r['W'] <= min(r['AC'], r['D']) and r['AC'] <= min(r['A'],r['C'])
                and r['all_direct_witness'] <= r['W'], 'Witness subset violated')
        close(r['primary_score'], r['W']/18)
        require(r['any_witness'] == (r['W'] > 0), 'Root event mismatch')
    require(len(primaries) == 8 and {(r['actor'],r['group']) for r in primaries} ==
            {(a,g) for a in ('gemma','qwen') for g in GROUPS}, 'Eight primary cells required')
    primary_rows, strong_rows = [], []
    for p in primaries:
        names = [p['group']+suffix for suffix in ('_s0','_s1')] if p['group'] in GROUPS[:2] else [p['group']]
        rows = [r for r in counts if r['actor']==p['actor'] and r['condition'] in names]
        ordered = sorted(expected_roots)
        scores = np.array([sum(r['W'] for r in rows if r['root_id']==rid)/(18*len(names)) for rid in ordered])
        require(p['root_ids'] == ordered and p['roots'] == 64 and p['full_coverage'], 'Primary root inventory changed')
        close(scores,p['root_scores'])
        rng=np.random.default_rng(spec['bootstrap_seed'])
        boot=scores[rng.integers(0,64,size=(spec['bootstrap_draws'],64))].mean(axis=1)
        tail=(1-spec['interval_level'])/2
        interval=np.quantile(boot,[tail,1-tail])
        close(interval,p['corrected_interval']); close(scores.mean(),p['mean'])
        affected=sum(scores>0); close(affected/64,p['root_witness_prevalence'])
        primary_rows.append({'model':p['actor'],'group':p['group'],'rate':float(scores.mean()),
                             'lower':float(interval[0]),'upper':float(interval[1]),'roots':int(affected),'total_roots':64})
        strong_roots=len({r['root_id'] for r in rows if r['strong']>0})
        strong_rows.append({'model':p['actor'],'group':p['group'],'count':sum(r['strong'] for r in rows),
                            'opportunities':len(rows)*18,'roots':strong_roots,'total_roots':64})
        for name in names:
            seeds=[r for r in rows if r['condition']==name]
            require(sum(r['strong'] for r in seeds)>0 and sum(r['all_direct_witness'] for r in seeds)>0,
                    'Individual-condition recurrence missing')
            w=[r for r in witnesses if r['actor']==p['actor'] and r['condition']==name and r['draw']=='all']
            require(len(w)==1,'Witness summary identity mismatch')
            for key in ('A','C','AC','D','W','strong','all_direct_witness'):
                require(sum(r[key] for r in seeds)==w[0][key], 'Per-seed count mismatch')
            require(w[0]['fixed_questions']==1152,'Per-seed denominator changed')
    v9=[]
    for r in read('v9/primary_summary.json'):
        require(r['view']=='HIGH_DIRECT_MARGIN_090' and r['variant']=='d0' and r['cohort']=='ALL_HISTORIES',
                'V9 strong view must include direct margin and both reference gates')
        row={'model':r['model'],'condition':r['condition'],
             **{out:int(float(r[src])) for out,src in [('count','count_ge10'),('eligible','n'),('full','full_denominator'),('roots','roots_any_ge10'),('total_roots','full_roots')]}}
        require(row['count']<=row['eligible']<=row['full']==1088 and row['roots']<=row['total_roots']==64,'V9 denominator mismatch')
        v9.append(row)
    require(len(v9)==12 and len({(r['model'],r['condition']) for r in v9})==12,'V9 inventory mismatch')
    v6=[]
    for r in read('v6/fixed_certificate_summary.json'):
        require(r['view']=='E_SAME_METHOD_NOOP' and r['interface']=='ALL_34','V6 fixed interface changed')
        row={'model':r['model'],'condition':r['condition'],
             **{out:int(r[src]) for out,src in [('count','count_ge10'),('eligible','eligible_questions'),('full','total_question_opportunities'),('roots','roots_ge10'),('total_roots','total_roots')]}}
        require(row['count']<=row['eligible']<=row['full']==34*row['total_roots'] and row['roots']<=row['total_roots'],'V6 denominator mismatch')
        v6.append(row)
    require(len(v6)==10,'V6 inventory mismatch')
    coherence=[{'model':r['model'],'condition':r['condition'],'cohort':r['cohort'],'mean':float(r['mean'])} for r in read('v9/coherence_summary.json')]
    for model in ('gemma','qwen'):
        native=next(r['mean'] for r in coherence if r['model']==model and r['condition']=='NATIVE_FINAL')
        require(all(r['mean']<native for r in coherence if r['model']==model and r['cohort']=='ACTUAL_EDIT'),'Adverse coherence control changed')
    example=read('v10/selected_repeat_checked_example.json')
    close(np.ptp(example['correct_probabilities'])/2,example['half_probability_range'])
    require(example['predictions']==[0,0,0,0,1,1,0,1] and example['W'] and example['all_direct_witness'],'Illustrative witness changed')
    v7=read('v7/constructive_summary.json')
    for model in ('gemma','qwen'):
        for group in ('INV_PAIR_NLL','INV_PAIR_CONSISTENCY','FREE_PAIR_CONSISTENCY'):
            rows=[r for r in v7['gates'] if r['model']==model and r['condition'] in (group+'_s0',group+'_s1')]
            require(len(rows)==2 and all(r['H1'] and r['H2'] and not r['H3'] for r in rows),'Constructive gate boundary changed')
    return {'v7':v7['three_edit'],'v9':v9,'v6':v6,'coherence':coherence,'v10':primary_rows,'strong':strong_rows,
            'example':{'minimum_direct_probability':min(example['direct_correct_probabilities']),
                       'half_range':example['half_probability_range']}}


def verify_provenance(base=BASE):
    for row in load_json(base/'provenance.json')['files']:
        require(sha256(base/row['path'])==row['sha256'],'Compact derivative hash mismatch')


def replay(base=BASE):
    verify_provenance(base)
    result=reconstruct(base)
    require(result == load_json(base/'headline_values.json'),'Derived headline values changed')
    return result


def export(output):
    result=replay()
    destination=new_output(Path(output)/'state_sufficiency')
    for key in ('v7','v9','v6','coherence','v10','strong'):
        with (destination/(key+'.csv')).open('x',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result[key][0]))
            writer.writeheader();writer.writerows(result[key])
    return {'status':'compact_aggregate_consistency_verified','primary_cells':len(result['v10']),
            'output':str(destination),'scope':'saved root-count bootstrap and aggregate checks; no raw-score or neural replay'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reproduced'))
    print(json.dumps(export(parser.parse_args().output),indent=2))
