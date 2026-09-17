"""Reconstruct the completed matched-origin comparison from saved scores."""
from collections import defaultdict
import argparse
import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

from .common import ROOT, load_json, sha256, safe_path, new_output

BASE = ROOT / 'reproducibility/update_method_comparison'
INPUTS = ROOT / 'reproducibility/fresh_inference/supplied/reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS'
RECIPES = ('INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY')
BASELINE = 'FIELD_PLUS_LATEST_ERRATUM'
CONDITIONS = ('SOURCE','REBUILD','EXISTING_CORRECTION','LATEST_SAME_WORDING','LATEST_ERRATUM',BASELINE,
              'INV_PAIR_NLL_s0','INV_PAIR_NLL_s1','FREE_PAIR_CONSISTENCY_s0','FREE_PAIR_CONSISTENCY_s1',
              'CANONICAL_REPLAY_s0','CANONICAL_REPLAY_s1')


def need(value, message):
    if not value:
        raise ValueError(message)


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def truth(values, spec):
    a = int(values[spec['a']][spec['p']]); b = int(values[spec['b']][spec['p']])
    return {'direct':a, 'opposes':1-a, 'same':int(a==b), 'both':a*b, 'either':int(bool(a or b))}[spec['kind']]


def interval(values):
    x = np.array(values, dtype=float)
    rng = np.random.default_rng(260913902)
    means = np.concatenate([x[rng.integers(len(x), size=(500,len(x)))].mean(1) for _ in range(20)])
    low, high = np.quantile(means, [.00625,.99375])
    return dict(effect=float(x.mean()),lower=float(low),upper=float(high),level=.9875,
                roots=len(x),draws=10000,zero_crossing_is_equivalence=False)


def verify(base=BASE):
    for row in load_json(base/'supplied/completed/MANIFEST.json')['files']:
        p=safe_path(row['path'],base/'supplied/completed')
        need(p.stat().st_size==row['bytes'] and sha256(p)==row['sha256'], 'Changed comparison member: '+row['path'])
    # These are the exact fixed inputs already distributed with the main-study
    # scientific dependencies. Do not create a second divergent case inventory.
    for name, rec in load_json(INPUTS/'ARTIFACT_MANIFEST.json')['files'].items():
        if name.startswith('cases/fixed/'):
            p=safe_path(name,INPUTS)
            need(p.stat().st_size==rec['bytes'] and sha256(p)==rec['sha256'], 'Changed fixed input: '+name)


def validate_score(r, reference, origin, sources):
    for key, value in reference.items():
        actual = r['input_panel'] if key=='panel' else r[key]
        need(actual==value, 'Score differs from frozen evaluation field: '+key)
    need(r['input_panel']=='ORIGIN' and r['reader']=='R0', 'Unexpected comparison panel/reader')
    need(r['gold']==truth(origin['terminal']['values'],r['spec']), 'Incorrect terminal semantic label')
    need(r['source_gold']==truth(origin['sources'][r['origin']]['values'],r['spec']), 'Incorrect source semantic label')
    lp=r['logps']; need(len(lp)==2 and all(math.isfinite(x) for x in lp),'Invalid score')
    need(r['prediction']==int(lp[1]>lp[0]), 'Prediction differs from scores')
    need(r['tie']==(lp[0]==lp[1]) and r['valid']==bool(r['argmax_in_labels'] and not r['tie']), 'Validity mismatch')
    need(math.isclose(r['answer_mass'],sum(math.exp(x) for x in lp),abs_tol=1e-8,rel_tol=1e-6), 'Candidate mass differs')
    need(r['correct']==bool(r['valid'] and r['prediction']==r['gold']), 'Correctness differs')
    need(r['master_unchanged'], 'Recorded master mutation')
    key=(r['input_id'],r['query_id'])
    if r['condition']=='SOURCE':
        sources[key]=(r['prediction'],r['valid'],r['logps'])
    need((r['source_prediction'],r['source_valid'],r['source_logps'])==sources[key], 'Changed SOURCE reference')


def reconstructed(base=BASE):
    verify(base)
    evaluation={a:{} for a in ('gemma','qwen')}
    for row in rows(INPUTS/'cases/fixed/evaluation.jsonl'):
        if row['panel']=='ORIGIN':
            key=(row['input_id'],row['query_id'])
            need(key not in evaluation[row['actor']], 'Duplicate frozen query')
            evaluation[row['actor']][key]=row
    origins={r['root_id']:r for r in rows(INPUTS/'cases/fixed/origin.jsonl')}
    need(len(origins)==64, 'Expected 64 frozen roots')
    compact=[]; telemetry=[]; vectors={}; absolute=[]; fallback=[]
    fields=('actor','condition','root_id','scene_id','input_panel','origin','correct','valid','prediction','source_prediction','source_valid')
    for actor in ('gemma','qwen'):
        need(len(evaluation[actor])==64*4*34, 'Fixed question inventory differs')
        sources={}
        for condition in CONDITIONS:
            stem=actor+'__ORIGIN__'+condition
            seen=set(); grouped=defaultdict(list)
            for row in rows(base/'supplied/completed/raw'/(stem+'.jsonl.gz')):
                key=(row['input_id'],row['query_id'])
                need(key not in seen and key in evaluation[actor], 'Missing/duplicate/extra score identity')
                need(row['actor']==actor and row['condition']==condition, 'Wrong condition')
                validate_score(row,evaluation[actor][key],origins[row['root_id']],sources)
                seen.add(key); slim={k:row[k] for k in fields}; compact.append(slim); grouped[row['root_id']].append(slim)
            need(seen==set(evaluation[actor]),'Incomplete fixed score coverage')
            vector={}
            for rid, rr in grouped.items():
                need(len(rr)==136 and {r['origin'] for r in rr}=={0,1,2,3},'Incomplete root bundle')
                vector[rid]=int(all(r['correct'] for r in rr))
            need(set(vector)==set(origins),'Missing root')
            vectors[actor,condition]=vector
            absolute.append(dict(actor=actor,condition=condition,whole_bundle_roots=sum(vector.values()),roots=64,
                                 rate=sum(vector.values())/64,question_accuracy=sum(r['correct'] for rr in grouped.values() for r in rr)/len(seen)))
            tt=list(rows(base/'supplied/completed/raw'/(stem+'.telemetry.jsonl.gz')))
            need(len(tt)==256 and {r['input_id'] for r in tt}=={k[0] for k in seen},'Incomplete telemetry coverage')
            need(all(r['actor']==actor and r['condition']==condition for r in tt),'Wrong telemetry condition')
            telemetry.extend(tt)
            if condition==BASELINE:
                fallback.append(dict(actor=actor,contexts=len(tt),fallback=sum(bool(t['refresh_fallback']) for t in tt)))
    contrasts=[]
    for actor in ('gemma','qwen'):
        baseline=vectors[actor,BASELINE]
        for recipe in RECIPES:
            seed_d={str(s):{r:vectors[actor,recipe+f'_s{s}'][r]-baseline[r] for r in sorted(origins)} for s in (0,1)}
            delta={r:(seed_d['0'][r]+seed_d['1'][r])/2 for r in sorted(origins)}
            contrasts.append(dict(actor=actor,recipe=recipe,baseline=BASELINE,
                endpoint='complete 34-question bundle correct from all four origins',
                unit='terminal root; seed mean inside each root; baseline once',bootstrap_seed=260913902,
                per_root=delta,per_seed={s:interval(list(v.values())) for s,v in seed_d.items()},**interval(list(delta.values()))))
    result=dict(status='saved_origin_comparison_reconstructed',score_rows=len(compact),telemetry_rows=len(telemetry),
                questions_per_history=len(evaluation['gemma'])//len(origins)//4,
                contrasts=contrasts,absolute_rates=absolute,fallback=fallback,
                scope='Matched-origin comparison; 64 roots per model, four histories, 34 questions, both learned seeds.')
    # Original tabulation code is reused only after strict independent coverage,
    # semantic-label and score checks; its weak len(origins)-only guard is not used.
    spec=importlib.util.spec_from_file_location('_supplied_comparison_analysis',base/'supplied/later_delivery/UPDATE_METHOD_RESULTS/code/analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tables={'headline_results':module.headline(compact,contrasts),'per_root_results':module.per_root(compact),
            'per_scene_results':module.per_scene(compact),'timing_memory':module.timing_memory(telemetry)}
    need(contrasts==load_json(base/'supplied/completed/tables/primary_contrasts.json')['contrasts'],'Primary comparison differs')
    for name, records in tables.items():
        with (base/'supplied/completed/tables'/(name+'.csv')).open(encoding='utf8',newline='') as f:
            saved=list(csv.DictReader(f))
        need(len(saved)==len(records),'Table row count differs: '+name)
        for actual, expected in zip(records,saved,strict=True):
            for key,string in expected.items():
                value=actual[key]
                need((string=='' and value is None) or
                     (isinstance(value,(float,int)) and string!='' and math.isclose(float(string),value,rel_tol=1e-12,abs_tol=1e-12)) or
                     string==str(value),'Table differs: '+name+'/'+key)
    return result,tables


def export(output):
    result,tables=reconstructed()
    destination=new_output(Path(output))
    with (destination/'results.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    for name,records in tables.items():
        with (destination/(name+'.csv')).open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('reproduced/update-methods'))
    result=export(p.parse_args().output)
    print(json.dumps({k:result[k] for k in ('status','score_rows','telemetry_rows','fallback')} |
        {'contrasts':[{k:r[k] for k in ('actor','recipe','effect','lower','upper')} for r in result['contrasts']],
         'absolute_rates':result['absolute_rates']},indent=2))


if __name__=='__main__':main()
