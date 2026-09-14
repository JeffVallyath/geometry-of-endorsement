# Public adaptation: provider launch policy and private custody paths omitted; scientific functions retained.
"""Fail-closed full-coverage V5 analysis. No partial efficacy interpretation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import rfr5_common as C
import rfr5_forward as F
import rfr5_metrics as M


def verify_rows(rows, populations):
    """Independently verify cardinality, labels, scoring and matched SOURCE."""
    expected={}
    for actor,scenes in populations.items():
        for s in scenes:
            for program,order in F.blocks(s):
                for reader in C.READERS:
                    for q in C.queries(s,program,reader):
                        for name in F.condition_names():
                            expected[(actor,s.scene_id,program,order,reader,name,q['query_id'])]=(s,q)
    observed={}
    measurements={}
    for r in rows:
        key=tuple(r[k] for k in ('actor','scene_id','program','order','reader','condition','query_id'))
        if key in observed or key not in expected:raise ValueError('Duplicate or undeclared scientific row')
        s,q=expected[key]; spec=q['spec']
        # Independent SET and Boolean label implementation, not gold lookup.
        world=[list(v) for v in s.values]; n={'single':1,'AB':2,'ABC':3}[r['program']]
        for offset in range(n):
            a=(s.target_actor+offset)%len(s.actors);p=s.target_project;world[a][p]=1-world[a][p]
        def truth(matrix):
            a,b,p=(spec[k] for k in ('a','b','p'));x,y=matrix[a][p],matrix[b][p]
            return {'direct':x,'opposes':1-x,'same':int(x==y),'both':x*y,'either':max(x,y)}[spec['kind']]
        if r['gold']!=truth(world) or r['source_gold']!=truth(s.values) or r['changed']!=(truth(world)!=truth(s.values)):
            raise ValueError('Independent labels disagree')
        if r['labels']!=q['labels'] or r['question_text']!=q['text'] or r['additional']!=q['additional'] or r['spec']!=q['spec']:
            raise ValueError('Query contract drift')
        if len(r['logps'])!=2 or not all(__import__('math').isfinite(v) for v in r['logps']):raise ValueError('Invalid score')
        pred=0 if r['logps'][0]>=r['logps'][1] else 1
        if r['prediction']!=pred or not r['master_verified']:raise ValueError('Scoring or immutable master mismatch')
        identity=(r['artifact_hash'],r['prefix_sha256'],r['suffix_text'],tuple(r['suffix_ids']),tuple(r['label_token_ids']))
        stored=measurements.setdefault(r['measurement_id'],(identity,r['logps']))
        if stored!=(identity,r['logps']):raise ValueError('Illegal measurement sharing')
        observed[key]=r
    if set(observed)!=set(expected):raise ValueError(f'Incomplete coverage: {len(observed)} / {len(expected)} rows')
    for key,r in observed.items():
        src=observed[key[:5]+('SOURCE',key[-1])]
        if r['source_prediction']!=src['prediction'] or r['source_logps']!=src['logps']:
            raise ValueError('Wrong-reader SOURCE denominator')
    return {'status':'FULL_COVERAGE_AND_INDEPENDENT_LABELS_PASS','rows':len(rows),'unique_measurements':len(measurements),
            'independent_scenes':{a:len(s) for a,s in populations.items()},'actors_share_scene_designs':True}


def source_reader_changes(rows):
    result=[]
    for actor in C.SIZES:
        for reader in C.READERS[1:]:
            groups={}
            for r in rows:
                if r['actor']==actor and r['condition']=='SOURCE' and r['reader'] in ('R0',reader):
                    groups.setdefault((r['scene_id'],r['program'],r['order'],r['query_id']),{})[r['reader']]=r
            per=defaultdict(list)
            for (sid,program,order,qid),pair in groups.items():
                a,b=pair['R0'],pair[reader]
                per[(program,order,sid)].append((int(b['prediction']==b['source_gold'])-int(a['prediction']==a['source_gold']),a['prediction']!=b['prediction']))
            for program,order in [('single','early'),('single','late'),('AB','early'),('ABC','early')]:
                values=[sum(v[0] for v in x)/len(x) for (p,o,s),x in per.items() if (p,o)==(program,order)]
                result.append({'actor':actor,'reader':reader,'program':program,'order':order,
                               'metric':'SOURCE source-world accuracy reader difference',**M.bootstrap(values)})
    return result


def run(input_dir, prepared, output):
    input_dir,prepared,output=map(Path,(input_dir,prepared,output))
    terminal=C.load(input_dir/'TERMINAL.json')
    if terminal['status']!='FINAL_MEASUREMENTS_COMPLETE':raise ValueError('No efficacy analysis of incomplete experiment')
    if output.exists():raise FileExistsError('Analysis output immutable; choose new revision')
    rows=[]
    for actor in C.SIZES:
        complete=C.load(input_dir/actor/'EVALUATION_COMPLETE.json')
        if complete['scenes']!=C.SIZES[actor]:raise ValueError('Missing actor coverage')
        for receipt in sorted((input_dir/actor/'blocks').glob('*.DONE.json')):
            done=C.load(receipt)
            for fn,h in done['files'].items():
                path=receipt.parent/fn
                if C.sha(path)!=h:raise ValueError('Block member digest mismatch')
                if not fn.endswith('.ALIASES.jsonl'):rows.extend(C.read_rows(path))
    populations={a:[C.Q.scene_from_dict(v) for v in C.read_rows(prepared/(a+'_scenes.jsonl'))] for a in C.SIZES}
    checks=verify_rows(rows,populations)
    primary=M.primary_contrasts(rows);secondary=M.secondary_contrasts(rows)
    output.mkdir(parents=True)
    C.dump(output/'INDEPENDENT_CHECKS.json',checks)
    C.dump(output/'PRIMARY_CONTRASTS.json',primary)
    C.dump(output/'SECONDARY_CONTRASTS.json',secondary)
    C.dump(output/'SOURCE_READER_EFFECTS.json',source_reader_changes(rows))
    summary=M.summaries(rows)
    C.dump(output/'FULL_FAMILY_METRICS.json',summary)
    C.dump(output/'FULL_FAMILY_CONTRASTS.json',M.full_family_contrasts(summary))
    C.dump(output/'EQUALITY_TRUTH_TABLE.json',M.truth_cells(rows))
    diagnostic=M.recomposition(rows)
    # Summarize only fully covered artifact bundles. Missing operands are
    # explicitly counted, never quietly filled or treated as correct.
    C.write_rows(output/'RECOMPOSITION_DIAGNOSTIC.jsonl',diagnostic)
    C.dump(output/'RECOMPOSITION_SUMMARY.json',M.recomposition_summaries(diagnostic))
    C.write_rows(output/'ERROR_EXAMPLES.jsonl',[r for r in rows if r['prediction']!=r['gold']])
    text=['# RESULTS — RELATIONAL_FROZEN_READOUT_V5','',
          'Fresh frozen-reader inference; historical V4 results and failures remain unchanged.',
          'R2 gains are assisted-reader results. No full new F2/F3/F4 claim is available because repeat/restoration/workflow panels were not rerun.',
          'Intervals crossing zero do not establish equivalence. Qwen direct-edit directions remain separately measured.','',
          '| Actor | Frozen construction | Reader contrast | Effect (pp) | 99.375% interval (pp) |',
          '|---|---|---|---:|---|']
    for row in primary:
        text.append(f"| {row['actor']} | {row['architecture']} | {row['reader']}-R0 | {100*row['effect']:.2f} | [{100*row['lower']:.2f}, {100*row['upper']:.2f}] |")
    text+=['','Independent units: 64 Gemma scenes and 32 Qwen scenes; Qwen shares designs as a hash-balanced subset. Seeds are averaged within scene before 10,000 paired resamples.',
           'Seed-specific intervals and per-scene effects are in PRIMARY_CONTRASTS.json. Native reader changes, edited-minus-native, difference-in-differences and COMPLETE-minus-V3 uncertainty are in SECONDARY_CONTRASTS.json.',
           'FULL_FAMILY_METRICS.json preserves original/additional bundles, all-program correctness, directions, family accuracy and same-reader SOURCE harm/conditional-harm denominators, by program and order.',
           'CANONICAL_CURRENT retains source backup and command history. Byte-identical artifact/suffix measurements carry explicit aliases and are not extra observations.',
           'Exact logic diagnostics use saved directly measured operands and known symbolic query structure. They are not gold substitution, native reasoning, an internal F2 pass or a speedup.',
           'external-artifacts']
    (output/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    (output/'DELTA_FROM_V4.md').write_text('# Delta from V4\n\nHistorical selected weights exported without retraining. No writer, layer, rank, strength, fitting or checkpoint-selection change. Only two fixed alternative equality readers added to the original reader on fresh RFR5 scenes. Native/text/source, both V3 references, both seeds and CAL-selected replay retained.\n\nThis result cannot retroactively change V4 or claim unmeasured repeat/restoration/workflow requirements.\n',encoding='utf8')
    C.dump(output/'MANIFEST.json',{'files':{p.name:C.sha(p) for p in sorted(output.iterdir())}})
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.input,a.prepared,a.output),indent=2))
