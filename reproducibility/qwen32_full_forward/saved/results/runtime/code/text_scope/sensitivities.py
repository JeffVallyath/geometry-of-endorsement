"""Preserve the published count/length/logical/cross-code sensitivity rules."""
from itertools import combinations
import sys
from text_scope.common import source_root, read


def evaluate(runtime,evaluator,design,index,method,prefix_lengths,primary_details):
    src=source_root(runtime)/'src'
    if str(src) not in sys.path:
        sys.path.insert(0,str(src))
    from repro.cpu_robustness import certification,common_pairs,sem,GATES
    rid=design['root']['root_id']
    world=design['root']
    values=world['terminal']['values']
    queries={q['query_id']:q for q in design['questions']}
    details={r['query_id']:r for r in primary_details}
    joint=[q for q in queries.values() if q['spec']['kind'] in ('same','both','either')]
    direct=[q['query_id'] for q in queries.values() if q['spec']['kind']=='direct']
    changed=[sum(s['values'][c['actor']][c['project']]!=c['value'] for c in world['commands']) for s in world['sources']]
    assert changed==[0,1,1,2,1,2,2,3]
    lengths=[prefix_lengths[f'o{o}/{method}'] for o in range(8)]
    allowed=[list(pair) for pair in combinations(range(8),2) if changed[pair[0]]==changed[pair[1]] and lengths[pair[0]]==lengths[pair[1]]]
    dec={}
    for key,score in index.items():
        gold=evaluator.truth(values,queries[key[1]]['spec'])
        dec[key]=(score['prediction'],score['valid'],score['valid'] and score['prediction']==gold,
                  score['probabilities'][gold],score['mass'])
    local={}
    for q in joint:
        qid=q['query_id']
        pair,project,kind=sem(q)
        draw=qid.split('-d')[1][0]
        operands=[i for i in direct if queries[i]['spec']['a'] in pair and queries[i]['spec']['p']==project and i.split('-d')[1][0]==draw]
        same=[x['query_id'] for x in joint if sem(x)[:2]==(pair,project) and sem(x)[2]!=kind and x['query_id'].split('-d')[1][0]==draw]
        cross=[x['query_id'] for x in joint if sem(x)[0]!=pair and sem(x)[1]==project and x['query_id'].split('-d')[1][0]==draw]
        assert len(operands)==2 and len(same)==2 and len(cross)==6
        gates=certification(dec,method,qid,operands,direct,same,cross)
        assert gates['G0']==details[qid]['AC']
        js=[dec[f'o{o}/{method}',qid] for o in range(8)]
        matched=[p for p in allowed if gates['G0'] and all(js[o][1] for o in p) and js[p[0]][0]!=js[p[1]][0]]
        extremes=[p for p in matched if details[qid]['strong'] and min(js[o][3] for o in p)<=.05 and max(js[o][3] for o in p)>=.95]
        flags={}
        for gate in GATES:
            flags[gate+'_pass']=gates[gate]
            flags[gate+'_witness']=gates[gate] and details[qid]['D']
            flags[gate+'_M_witness']=gates[gate] and bool(matched)
            flags[gate+'_strong_M']=gates[gate] and bool(extremes)
        local[qid]=dict(root_id=rid,query_id=qid,method=method,answer_code=draw,semantic=sem(q),
            changed_fact_counts=changed,prefix_token_counts=lengths,pair_availability=bool(allowed),
            allowed_pairs=allowed,matched=matched,extremes=extremes,joint=js,gates=gates,**flags)
    for qid,row in local.items():
        twin=next(v for v in local.values() if v['semantic']==row['semantic'] and v['answer_code']!=row['answer_code'])
        common=common_pairs(row,twin)
        extremes=[p for p in common if p in row['extremes'] and p in twin['extremes']]
        row['crosscode_pairs']=common
        row['crosscode_strong_pairs']=extremes
        for gate in GATES:
            eligible=row['gates'][gate] and twin['gates'][gate]
            row[gate+'_crosscode_pass']=eligible
            row['crosscode_'+gate+'_M_witness']=eligible and bool(common)
            row['crosscode_'+gate+'_strong_M']=eligible and bool(extremes)
    assert len(local)==18
    return list(local.values())


def summarize(rows):
    assert len(rows)==1152
    result=dict(coded_denominator=1152,semantic_denominator=576,roots=64,
        pair_available=sum(v['pair_availability'] for v in rows),controls={})
    for gate in ('G0','G1','G2a','G2b','G2ab'):
        result['controls'][gate]=dict(eligible=sum(v[gate+'_pass'] for v in rows),
            matched_witnesses=sum(v[gate+'_M_witness'] for v in rows),
            matched_strong=sum(v[gate+'_strong_M'] for v in rows),
            roots_with_matched=len({v['root_id'] for v in rows if v[gate+'_M_witness']}),
            crosscode_eligible=sum(v[gate+'_crosscode_pass'] for v in rows if v['answer_code']=='0'),
            crosscode_matched=sum(v['crosscode_'+gate+'_M_witness'] for v in rows if v['answer_code']=='0'),
            crosscode_strong=sum(v['crosscode_'+gate+'_strong_M'] for v in rows if v['answer_code']=='0'))
    result['each_answer_code']={draw:{key:sum(v[key] for v in rows if v['answer_code']==draw)
        for key in ('G0_pass','G0_witness','G0_M_witness','G0_strong_M')} for draw in ('0','1')}
    return result
