"""Independent V10 reconstruction from saved scores and semantic records.

Only Python's standard library and NumPy are used. The analyze command never
reads expected/ or imports the original experimental analysis or aggregate replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

HERE=Path(__file__).resolve().parent
MODELS=('gemma','qwen')
GROUPS=('INV_PAIR_NLL','FREE_PAIR_CONSISTENCY','EXISTING_CORRECTION','LATEST_SAME_WORDING')
METHODS=('SOURCE','INV_PAIR_NLL_s0','INV_PAIR_NLL_s1','FREE_PAIR_CONSISTENCY_s0','FREE_PAIR_CONSISTENCY_s1','EXISTING_CORRECTION','LATEST_SAME_WORDING')
ROOTS=tuple(f'SSC1-FINAL-{i:04d}' for i in range(64))
FLAGS=('A','C','AC','D','W','AD','strong','all_direct_eligible','all_direct_witness','stable_valid_agreement')


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(obj): return digest(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def load(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def check_manifest(base,role='input'):
    manifest=load(base/'MANIFEST.json');seen=set()
    for row in manifest['files']:
        name=row['path'];p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name,'Unsafe input path')
        require(name not in seen,'Duplicate manifest path');seen.add(name)
        if row['role']!=role: continue
        prefix={'input':'inputs/','expected':'expected/','example':'examples/'}[role]
        require(name.startswith(prefix),'Unexpected role/path')
        path=base/name
        require(path.resolve().is_relative_to(base.resolve()),'Escaping input path')
        raw=path.read_bytes()
        require(len(raw)==row['bytes'] and digest(raw)==row['sha256'],'Input hash mismatch: '+name)
    return manifest


def truth(values,spec):
    a=values[spec['a']][spec['p']];b=values[spec['b']][spec['p']]
    kind=spec['kind']
    if kind=='direct':return a
    if kind=='opposes':return 1-a
    if kind=='same':return int(a==b)
    if kind=='both':return int(a and b)
    if kind=='either':return int(a or b)
    raise ValueError('Unknown question function')


def validate_design(design,rid):
    world=design['root'];terminal=world['terminal'];commands=world['commands'];sources=world['sources']
    require(world['root_id']==rid and world['split']=='FINAL','Wrong root identity')
    require(len(sources)==8 and len(commands)==3,'History/command denominator')
    addresses=[(c['actor'],c['project']) for c in commands]
    require(len(set(addresses))==3,'Repeated addressed record')
    patterns=set()
    for source in sources:
        require(source['actors']==terminal['actors'] and source['projects']==terminal['projects'],'History entity mismatch')
        current=[row[:] for row in source['values']]
        require(len(current)==len(terminal['actors']) and all(len(r)==len(terminal['projects']) and all(type(v) is int and v in (0,1) for v in r) for r in current),'Invalid relation table')
        patterns.add(tuple(current[a][p] for a,p in addresses))
        for c in commands:current[c['actor']][c['project']]=c['value']
        require(current==terminal['values'],'Commands do not converge to complete final table')
    require(patterns==set(itertools.product((0,1),repeat=3)),'Incomplete starting assignments')
    require(sources[0]['values']==terminal['values'],'Origin zero is not already correct')
    questions=design['questions'];qindex={q['query_id']:q for q in questions}
    require(len(questions)==len(qindex)==34,'Duplicate/missing questions')
    for q in questions:
        require(q['root_id']==rid and q['family']=='familiar' and q['query_id'].startswith(rid+'-d'),'Question identity mismatch')
        spec=q['spec'];a=terminal['actors'][spec['a']];b=terminal['actors'][spec['b']];p=terminal['projects'][spec['p']]
        wording={'direct':f'Does {a} favor the {p} proposal?', 'opposes':f'Does {a} oppose the {p} proposal?',
                 'same':f'Do {a} and {b} take the same side on {p}?', 'both':f'Is the {p} proposal supported by both {a} and {b}?',
                 'either':f'Does at least one of {a} and {b} support {p}?'}[spec['kind']]
        lines=q['text'].splitlines();labels=q['labels'];draw=q['query_id'].split('-d')[1][0]
        require(lines[0]==wording and lines[-1]==f'Answer with exactly one of {labels[0]}, {labels[1]}.','Rendered question/semantic mismatch')
        if draw=='0':require(labels==['No','Yes'] and len(lines)==2,'Literal answer mapping changed')
        else:require(draw=='1' and set(labels)=={'A','B'} and len(lines)==3 and lines[1]==f'Use {labels[1]} for yes and {labels[0]} for no.','Code answer mapping changed')
    require(Counter(q['query_id'].split('-d')[1][0] for q in questions)=={'0':17,'1':17},'Answer-code coverage changed')
    inputs=design['source_inputs'];keys={(s['actor'],s['origin']) for s in inputs}
    require(len(inputs)==len(keys)==16 and keys==set(itertools.product(MODELS,range(8))),'Rendered-source inventory')
    for s in inputs:
        source=sources[s['origin']]
        actual=s['source_text'].split('Now the actual records:\n')
        require(len(actual)==2,'Missing source boundary')
        records=re.findall(r'^(.+) is (in favor of|opposed to) the (.+) proposal\.$',actual[1],re.M)
        parsed={}
        for name,stance,project in records:
            key=(source['actors'].index(name),source['projects'].index(project))
            require(key not in parsed,'Duplicate rendered relation');parsed[key]=int(stance=='in favor of')
        expected={(a,p):v for a,row in enumerate(source['values']) for p,v in enumerate(row)}
        require(parsed==expected,'Rendered source differs from starting relations')
        require(len(s['commands'])==3,'Rendered command count')
        for rendered,c in zip(s['commands'],commands):
            sentence=s['source_text'][rendered['start']:rendered['end']]
            a,p=c['actor'],c['project'];stance='in favor of' if source['values'][a][p] else 'opposed to'
            require(sentence==f"{source['actors'][a]} is {stance} the {source['projects'][p]} proposal." and rendered['value']==c['value'],'Addressed command/span mismatch')
    return qindex


def decode(row,question,slack=1e-6):
    """Derive validity, prediction, mass and probabilities without trusting flags.

The two saved scores are full-vocabulary log probabilities. If the larger
candidate probability exceeds ALL omitted vocabulary mass, the global maximum
must be a candidate. All retained records satisfy this certificate with slack.
"""
    lp=row['logps']
    require(len(lp)==2 and all(isinstance(v,(float,int)) and math.isfinite(v) for v in lp),'Nonfinite/two-candidate score violation')
    ps=[math.exp(v) for v in lp];mass=sum(ps)
    require(0<mass<=1+1e-5,'Invalid full-vocabulary candidate mass')
    require(max(ps)>max(0,1-mass)+slack,'Cannot independently establish global argmax membership from saved scores')
    pred=int(lp[1]>lp[0]);tie=lp[0]==lp[1];valid=not tie
    maximum=max(lp);weights=[math.exp(v-maximum) for v in lp];probs=[w/sum(weights) for w in weights]
    recorded=row['recorded_checks']
    derived={'prediction':pred,'finite':True,'tie':tie,'argmax_in_labels':True,'valid':valid}
    require(all(recorded[k]==v for k,v in derived.items()),'Observed validity/choice disagrees with score reconstruction')
    require(abs(recorded['answer_mass']-mass)<1e-12,'Candidate mass reconstruction mismatch')
    tok=row['tokens'];require(canonical(tok)==recorded['tokens_sha256'],'Token object hash mismatch')
    require(tok['question_sha256']==digest(question['text'].encode()),'Question hash mismatch')
    require(len(tok['label_token_ids'])==2 and len(set(tok['label_token_ids']))==2,'Label-token mapping invalid')
    return {'prediction':pred,'valid':valid,'probabilities':probs,'mass':mass,
            'argmax_certificate_gap':max(ps)-max(0,1-mass)}


def index_panel(panel,design,questions,phase,protocol):
    model=panel['model'];rid=design['root']['root_id']
    require(model in MODELS and panel['root_id']==rid and panel['phase']==phase,'Response shard identity mismatch')
    contexts=panel['contexts'];aliases=panel['aliases']
    expected_contexts={'NATIVE_FINAL'}|{f'o{o}/{m}' for o in range(8) for m in METHODS}
    require(set(contexts)==expected_contexts,'Context inventory mismatch')
    source_inputs={s['origin']:s for s in design['source_inputs'] if s['actor']==model}
    for key,context in contexts.items():
        origin=0 if key=='NATIVE_FINAL' else int(key.split('/')[0][1:])
        method='REBUILD' if key=='NATIVE_FINAL' else key.split('/')[1].split('_s')[0]
        require(context['method']==method and context['command_count']==3,'Context method/command identity mismatch')
        require(context['source_sha256']==digest(source_inputs[origin]['source_text'].encode()),'Context source hash mismatch')
        require(context['updated_text_sha256']==digest(context['text'].encode()),'Updated context text hash mismatch')
        require(context['cache_hash']==context['artifact_sha256'],'Cache/artifact identity mismatch')
        if method in ('REBUILD','SOURCE','INV_PAIR_NLL','FREE_PAIR_CONSISTENCY'):
            require(context['text']==source_inputs[origin]['source_text'],'Native/source/learned prefix text mismatch')
    for alias,target in aliases.items():
        require(alias in contexts and target in contexts and target not in aliases,'Invalid/cyclic alias')
        require(contexts[alias]['cache_hash']==contexts[target]['cache_hash'] and contexts[alias]['prefix_ids']==contexts[target]['prefix_ids'],'Alias not bound to same recorded cache/prefix')
    if phase=='core':expected={(c,q) for c in expected_contexts for q in questions}
    else:
        repeat=protocol['repeatability'];require(rid in repeat['roots'],'Unplanned repeat root')
        selected={'NATIVE_FINAL'}|{f'o{o}/{m}' for o in range(8) for m in METHODS[1:5]}
        expected={(c,rid+suffix) for c in selected for suffix in repeat['query_suffixes']}
    physical={};decoded={};suffixes={};label_ids={}
    for row in panel['physical']:
        key=(row['context'],row['query_id']);require(key not in physical,'Duplicate physical response ID')
        require(key in expected and key[0] not in aliases,'Unexpected physical response')
        physical[key]=row;decoded[key]=decode(row,questions[key[1]],protocol['argmax_certificate_slack'])
        tok=row['tokens'];old=suffixes.setdefault(key[1],tok['suffix_ids'])
        require(old==tok['suffix_ids'],'Downstream token suffix varies across histories/methods')
        for label,token in zip(questions[key[1]]['labels'],tok['label_token_ids']):
            require(label_ids.setdefault(label,token)==token,'Token/answer-code mapping changes within model/root')
    index={};raw_index={}
    for binding in panel['bindings']:
        key=(binding['context'],binding['query_id']);require(key not in index,'Duplicate logical response ID')
        target=aliases.get(key[0],key[0]);pk=(target,key[1])
        require(binding['physical_id']==target+'|'+key[1] and binding['alias']==(target!=key[0]),'Alias binding mismatch')
        require(pk in physical,'Missing physical response join')
        index[key]=decoded[pk];raw_index[key]=physical[pk]
    require(set(index)==expected,'Missing/extra fixed logical opportunities')
    require(set(physical)=={(aliases.get(c,c),q) for c,q in expected},'Unused/missing physical rows')
    return index,raw_index


def classify(design,index,condition):
    rid=design['root']['root_id'];values=design['root']['terminal']['values'];qs=design['questions']
    direct={(q['spec']['a'],q['spec']['p'],q['query_id'].split('-d')[1][0]):q for q in qs if q['spec']['kind']=='direct'}
    require(len(direct)==12,'Measured direct-query inventory')
    def response(o,q):return index[f'o{o}/{condition}',q['query_id']]
    def correct(r,g):return r['valid'] and r['prediction']==g
    all_direct=all(correct(response(o,q),truth(values,q['spec'])) for q in direct.values() for o in range(8))
    details=[]
    for q in qs:
        spec=q['spec']
        if spec['kind'] not in ('same','both','either'):continue
        qid=q['query_id'];draw=qid.split('-d')[1][0];gold=truth(values,spec)
        operands=[direct[a,spec['p'],draw] for a in (spec['a'],spec['b'])]
        reads=[(response(o,d),truth(values,d['spec'])) for o in range(8) for d in operands]
        joint=[response(o,q) for o in range(8)];native=index['NATIVE_FINAL',qid]
        a=all(correct(r,g) for r,g in reads);native_ok=correct(native,gold);noop_ok=correct(joint[0],gold)
        c=native_ok and noop_ok;d=len({r['prediction'] for r in joint if r['valid']})>1;w=a and c and d
        jp=[r['probabilities'][gold] for r in joint];dp=[r['probabilities'][g] for r,g in reads]
        strong=w and all(r['valid'] for r in joint) and min(dp)>=.95 and min(r['mass'] for r in joint+[r for r,g in reads]+[native])>=.90 and max(jp)>=.95 and min(jp)<=.05
        details.append({'root_id':rid,'condition':condition,'query_id':qid,'gold':gold,'A':a,'C':c,'AC':a and c,'D':d,'W':w,'AD':a and d,'strong':strong,
                        'native_correct':native_ok,'noop_correct':noop_ok,'operand_ids':[q['query_id'] for q in operands],
                        'all_direct_eligible':all_direct and c,'all_direct_witness':all_direct and c and d,
                        'stable_valid_agreement':all(r['valid'] for r in joint) and len({r['prediction'] for r in joint})==1,
                        'predictions':[r['prediction'] for r in joint],'correct_probabilities':jp,'direct_correct_probabilities':dp})
    require(len(details)==18,'Primary denominator must include all 18 opportunities')
    counts={k:sum(int(r[k]) for r in details) for k in FLAGS}
    return {'root_id':rid,'condition':condition,'questions':18,**counts,'primary_score':counts['W']/18,'any_witness':counts['W']>0},details


def summarize(roots,protocol):
    bootstrap=protocol['bootstrap'];require(bootstrap=={'draws':10000,'seed':2609141002,'level':.99375},'Frozen inference specification changed')
    primary=[];conditions=[]
    for model in MODELS:
        for method in METHODS:
            subset=[r for r in roots if r['actor']==model and r['condition']==method]
            require(len(subset)==64 and {r['root_id'] for r in subset}==set(ROOTS),'Per-condition root coverage')
            counts={k:sum(r[k] for r in subset) for k in FLAGS}
            conditions.append({'actor':model,'condition':method,'roots':64,'opportunities':1152,**counts,
                               'rate':counts['W']/1152,'conditional_rate':counts['W']/counts['AC'] if counts['AC'] else None,
                               'roots_with_witness':sum(r['W']>0 for r in subset),'roots_with_strong':sum(r['strong']>0 for r in subset)})
        for group in GROUPS:
            methods=[group+'_s0',group+'_s1'] if group in GROUPS[:2] else [group]
            selected=[r for r in roots if r['actor']==model and r['condition'] in methods]
            scores=np.array([sum(r['primary_score'] for r in selected if r['root_id']==rid)/len(methods) for rid in ROOTS])
            events=np.array([any(r['W'] for r in selected if r['root_id']==rid) for rid in ROOTS])
            rng=np.random.default_rng(bootstrap['seed']);draws=rng.integers(0,64,(bootstrap['draws'],64))
            means=scores[draws].mean(axis=1);prevalence=events[draws].mean(axis=1)
            p={'actor':model,'group':group,'roots':64,'root_ids':list(ROOTS),'root_scores':scores.tolist(),
               'mean':float(scores.mean()),'root_witness_prevalence':float(events.mean()),'roots_with_witness':int(events.sum()),
               'strong':sum(r['strong'] for r in selected),'strong_opportunities':len(selected)*18,
               'roots_with_strong':len({r['root_id'] for r in selected if r['strong']})}
            for level,label in ((bootstrap['level'],'corrected'),(.95,'descriptive')):
                tail=(1-level)/2;p[label+'_interval']=np.quantile(means,[tail,1-tail]).tolist()
                p[label+'_root_prevalence_interval']=np.quantile(prevalence,[tail,1-tail]).tolist()
            primary.append(p)
    return primary,conditions


def analyze(base=HERE):
    check_manifest(base);protocol=load(base/'inputs/protocol.json')
    require(protocol['inventory']['origins']==8 and protocol['inventory']['terminal_roots_per_actor']==64,'Protocol inventory changed')
    root_rows=[];query_rows=[];repeats={'fresh_score_rows':0,'pairwise_comparisons':0,'max_logp_difference':0.0}
    audit={'physical_rows':0,'logical_rows':0,'aliases':0,'semantic_final_tables':0,'argmax_certified':0,'minimum_argmax_certificate_gap':1.0}
    for rid in ROOTS:
        design=load(base/f'inputs/designs/{rid}.json');questions=validate_design(design,rid);audit['semantic_final_tables']+=8
        for model in MODELS:
            panel=load(base/f'inputs/responses/{model}/{rid}/core.json.gz')
            require(panel['model']==model,'Wrong model shard')
            index,raw_index=index_panel(panel,design,questions,'core',protocol)
            audit['physical_rows']+=len(panel['physical']);audit['logical_rows']+=len(index)
            audit['aliases']+=sum(b['alias'] for b in panel['bindings']);audit['argmax_certified']+=len(panel['physical'])
            audit['minimum_argmax_certificate_gap']=min(audit['minimum_argmax_certificate_gap'],min(r['argmax_certificate_gap'] for r in index.values()))
            for method in METHODS:
                root,details=classify(design,index,method)
                root_rows.append({'actor':model,**root});query_rows.extend({'actor':model,**q} for q in details)
            if rid in protocol['repeatability']['roots']:
                panels=[panel];indices=[index];raws=[raw_index]
                for phase in ('pass0','pass1'):
                    repeat=load(base/f'inputs/responses/{model}/{rid}/{phase}.json.gz')
                    require(repeat['model']==model,'Wrong repeat model shard')
                    ri,rr=index_panel(repeat,design,questions,phase,protocol)
                    repeats['fresh_score_rows']+=len(ri);panels.append(repeat);indices.append(ri);raws.append(rr)
                for key in indices[1]:
                    for left,right in ((0,1),(0,2),(1,2)):
                        lr,rr=raws[left][key],raws[right][key]
                        delta=max(abs(a-b) for a,b in zip(lr['logps'],rr['logps']))
                        repeats['max_logp_difference']=max(repeats['max_logp_difference'],delta)
                        require(lr['tokens']==rr['tokens'] and indices[left][key]==indices[right][key],'Repeat score/token mismatch')
                        require(panels[left]['contexts'][key[0]]['cache_hash']==panels[right]['contexts'][key[0]]['cache_hash'],'Recorded repeat cache identity mismatch')
                        repeats['pairwise_comparisons']+=1
    require((audit['physical_rows'],audit['logical_rows'],audit['aliases'])==(230656,248064,17408),'Saved core inventory mismatch')
    require(repeats['fresh_score_rows']==8448 and repeats['pairwise_comparisons']==12672,'Repeat coverage mismatch')
    primary,conditions=summarize(root_rows,protocol)
    return {'primary':primary,'conditions':conditions,'root_rows':root_rows,'query_rows':query_rows,'repeatability':repeats,'audit':audit}


def compare(result,base=HERE):
    """Explicitly separate expected-output access from independent calculation."""
    check_manifest(base,role='expected')
    for saved in load(base/'expected/PRIMARY.json'):
        computed=next(r for r in result['primary'] if (r['actor'],r['group'])==(saved['actor'],saved['group']))
        for k in ('roots','root_ids'):require(computed[k]==saved[k],'Primary identity mismatch: '+k)
        for k in ('mean','root_scores','root_witness_prevalence','corrected_interval','descriptive_interval','corrected_root_prevalence_interval','descriptive_root_prevalence_interval'):
            require(np.allclose(computed[k],saved[k],atol=1e-12,rtol=0),'Primary comparison mismatch: '+k)
    for filename,rows,fields in [('ROOT_TABLE.json',result['root_rows'],FLAGS+('questions','primary_score','any_witness')),
                                 ('QUERY_TABLE.json.gz',result['query_rows'],FLAGS+('gold','predictions','correct_probabilities','direct_correct_probabilities'))]:
        identity=('actor','root_id','condition')+(('query_id',) if filename.startswith('QUERY') else ())
        saved=load(base/'expected'/filename);lookup={tuple(r[k] for k in identity):r for r in saved}
        require(len(rows)==len(saved)==len(lookup),'Expected row inventory mismatch')
        for r in rows:
            s=lookup[tuple(r[k] for k in identity)]
            for k in fields:
                require(np.allclose(r[k],s[k],atol=1e-12,rtol=0),'Expected per-example/count mismatch: '+k)
    for s in load(base/'expected/WITNESS_SUMMARY.json'):
        if s['draw']!='all':continue
        r=next(r for r in result['conditions'] if (r['actor'],r['condition'])==(s['actor'],s['condition']))
        for k in ('A','C','AC','D','W','strong','all_direct_eligible','all_direct_witness'):require(r[k]==s[k],'Condition count mismatch')
        require(r['opportunities']==s['fixed_questions'],'Condition denominator mismatch')
    return {'status':'independent_reconstruction_matches','primary_cells':8,'root_rows':len(result['root_rows']),
            'joint_query_rows':len(result['query_rows']),'repeatability':result['repeatability'],'audit':result['audit']}


def write_result(result,output):
    require(not output.exists(),'Output exists; use a fresh directory')
    output.mkdir(parents=True)
    for key,data in result.items():
        with (output/(key+'.json')).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    fields=('actor','group','mean','corrected_interval','roots_with_witness','strong','strong_opportunities','roots_with_strong')
    with (output/'primary.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['primary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['analyze','compare'])
    p.add_argument('--package',type=Path,default=HERE);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='analyze':
        result=analyze(a.package);write_result(result,a.output)
        print(json.dumps({'status':'independently_reconstructed_without_expected_access','primary_cells':len(result['primary']),'audit':result['audit'],'repeatability':result['repeatability']},indent=2))
    else:
        result={p.stem:load(p) for p in a.output.glob('*.json')}
        print(json.dumps(compare(result,a.package),indent=2))
