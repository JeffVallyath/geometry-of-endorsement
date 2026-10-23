"""Independent raw-record reconstruction: no experiment classification imports."""
import argparse
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def truth(spec, values):
    a,b = values[spec['a']][spec['p']],values[spec['b']][spec['p']]
    if spec['kind'] == 'direct': return a
    if spec['kind'] == 'opposes': return 1-a
    if spec['kind'] == 'same': return int(a == b)
    if spec['kind'] == 'both': return int(a == b == 1)
    if spec['kind'] == 'either': return int(a == 1 or b == 1)
    raise ValueError('UNKNOWN_FROZEN_QUESTION')


def decode(raw):
    s = raw['score']
    lp = [float(v) for v in s['logps']]
    finite = all(math.isfinite(x) for x in lp)
    pred = int(lp[1] > lp[0])
    tie = lp[0] == lp[1]
    if not finite:
        assert not s['finite'] and not s['valid'] and s['prediction']==pred
        return dict(pred=pred,valid=False,mass=None,prob=[None,None])
    candidate = min((int(token) for token,value in zip(s['label_token_ids'],lp) if value == max(lp)),default=-1)
    outside = float(s['maximum_non_candidate_logp'])
    winner = candidate if max(lp) > outside else int(s['maximum_non_candidate_token_id'])
    if max(lp) == outside:
        winner = min(candidate,int(s['maximum_non_candidate_token_id']))
    member = winner in s['label_token_ids']
    valid = finite and not tie and member
    assert (finite,pred,tie,valid,member,winner) == (s['finite'],s['prediction'],s['tie'],s['valid'],s['argmax_in_labels'],s['full_vocab_argmax_token_id'])
    mass = sum(math.exp(x) for x in lp)
    assert abs(mass-float(s['answer_mass'])) <= 1e-6
    if finite:
        w = [math.exp(x-max(lp)) for x in lp]
        prob = [x/sum(w) for x in w]
    else:
        prob = [None,None]
    return dict(pred=pred, valid=valid, mass=mass, prob=prob)


def reconstruct_case(runtime, rid, phase):
    design = load(runtime/'cpu/designs'/(rid+'.json'))
    questions = {q['query_id']:q for q in design['questions']}
    world = design['root']
    values = world['terminal']['values']
    folder = runtime/'results/qwen32_t4'/phase/rid
    done = load(folder/'DONE.json')
    for name,digest in done['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    index = {}
    with (folder/'RAW_SCORES.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            raw = json.loads(line)
            key = raw['context'],raw['query_id']
            assert key not in index
            q = questions[key[1]]
            assert raw['tokens']['question_sha256'] == hashlib.sha256(q['text'].encode()).hexdigest()
            assert raw['tokens']['label_token_ids'] == raw['score']['label_token_ids']
            assert raw['semantic_index_mapping'] == ['no','yes']
            index[key] = decode(raw)
    assert len(index) == 578
    seal = load(folder/'CONTEXT_SEAL.json')['contexts']
    changed = [sum(s['values'][c['actor']][c['project']] != c['value'] for c in world['commands']) for s in world['sources']]
    direct = [q for q in questions if questions[q]['spec']['kind']=='direct']
    joints = [q for q in questions if questions[q]['spec']['kind'] in ('same','both','either')]
    def semantic(qid):
        s=questions[qid]['spec']
        return tuple(sorted((s['a'],s['b']))),s['p'],s['kind']
    def draw(qid): return qid.split('-d')[1][0]
    result = {}
    for method in ('EXISTING_CORRECTION','LATEST_SAME_WORDING'):
        def r(o,q): return index[f'o{o}/{method}',q]
        def correct(score,q): return score['valid'] and score['pred'] == truth(questions[q]['spec'],values)
        primary = {}
        for q in joints:
            pair,p,kind=semantic(q)
            operands=[d for d in direct if questions[d]['spec']['a'] in pair and questions[d]['spec']['p']==p and draw(d)==draw(q)]
            reads=[(r(o,d),d) for o in range(8) for d in operands]
            targets=[r(o,q) for o in range(8)]
            native=index['NATIVE_FINAL',q]
            A=all(correct(row,d) for row,d in reads)
            C=correct(native,q) and correct(targets[0],q)
            D=len({x['pred'] for x in targets if x['valid']}) > 1
            W=A and C and D
            gold=truth(questions[q]['spec'],values)
            strong=W and all(x['valid'] for x in targets)
            if strong:
                strong = all(row['prob'][truth(questions[d]['spec'],values)] >= .95 for row,d in reads) \
                    and min(x['mass'] for x in targets+[row for row,d in reads]+[native]) >= .9 \
                    and min(x['prob'][gold] for x in targets) <= .05 and max(x['prob'][gold] for x in targets) >= .95
            primary[q]=dict(A=A,C=C,AC=A and C,D=D,W=W,strong=bool(strong))
        allowed=[list(pair) for pair in combinations(range(8),2) if changed[pair[0]]==changed[pair[1]]
                 and len(seal[f'o{pair[0]}/{method}']['prefix_ids'])==len(seal[f'o{pair[1]}/{method}']['prefix_ids'])]
        semantic_rows={}
        for q in joints:
            if draw(q)!='0':continue
            twin=next(x for x in joints if semantic(x)==semantic(q) and draw(x)=='1')
            logical=[x for x in joints if x not in (q,twin)]
            assert len(logical)==16
            direct_ok=all(correct(r(o,d),d) for o in range(8) for d in direct)
            reference=all(correct(index[k,x],x) for k in ('NATIVE_FINAL',f'o0/{method}') for x in (q,twin))
            logical_ok=all(correct(r(o,d),d) for o in range(8) for d in logical)
            pairs=[]
            for a,b in allowed:
                observations=[r(o,x) for o in (a,b) for x in (q,twin)]
                if all(x['valid'] for x in observations) and r(a,q)['pred']!=r(b,q)['pred'] \
                    and r(a,twin)['pred']!=r(b,twin)['pred'] and all(r(o,q)['pred']==r(o,twin)['pred'] for o in (a,b)):
                    pairs.append([a,b])
            strongest=pairs if direct_ok and reference and logical_ok else []
            high=[]
            for pair in strongest:
                if all(primary[x]['strong'] for x in (q,twin)) and all(
                    min(r(o,x)['prob'][truth(questions[x]['spec'],values)] for o in pair)<=.05 and
                    max(r(o,x)['prob'][truth(questions[x]['spec'],values)] for o in pair)>=.95 and
                    all(r(o,x)['mass']>=.9 for o in pair) for x in (q,twin)):
                    high.append(pair)
            semantic_rows[q]=dict(all_direct=direct_ok,reference=reference,all_logical=logical_ok,
                allowed_pairs=allowed,matched_crossformat_G0_pairs=pairs if primary[q]['AC'] and primary[twin]['AC'] else [],
                strongest_pairs=strongest,high_margin_pairs=high,
                all_history_targets_correct=all(correct(r(o,x),x) for o in range(8) for x in (q,twin)))
            required=[r(o,x) for o in range(8) for x in direct+logical]
            required += [index[k,x] for k in ('NATIVE_FINAL',f'o0/{method}') for x in (q,twin)]
            valid_pair=any(all(r(o,x)['valid'] for o in pair for x in (q,twin))
                for pair in (allowed or list(combinations(range(8),2))))
            failures=dict(invalid_answer=not all(x['valid'] for x in required) or not valid_pair,
                failed_direct_readout=not direct_ok,failed_native_noop_target=not reference,
                failed_other_logical_question=not logical_ok,unavailable_matched_pair=not allowed,no_target_disagreement=not pairs)
            semantic_rows[q]['failures']=failures
            semantic_rows[q]['attrition_reason']=next((k for k,v in failures.items() if v),'survivor')
        counts={k:sum(row[k] for row in primary.values()) for k in ('A','C','AC','D','W','strong')}
        counts.update(direct_correct=sum(correct(r(o,q),q) for o in range(8) for q in direct),
            joint_correct=sum(correct(r(o,q),q) for o in range(8) for q in joints),
            battery_correct=sum(correct(r(o,q),q) for o in range(8) for q in questions),
            native_correct=sum(correct(index['NATIVE_FINAL',q],q) for q in questions),
            noop_correct=sum(correct(r(0,q),q) for q in questions),
            all_origin_joint_correct=sum(all(correct(r(o,q),q) for o in range(8)) for q in joints),
            complete_case=all(correct(r(o,q),q) for o in range(8) for q in questions),
            invalid=sum(not r(o,q)['valid'] for o in range(8) for q in questions))
        result[method]=dict(primary=primary,semantic=semantic_rows,counts=counts)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    a=p.parse_args();runtime=a.runtime
    roots=load(runtime/'ANALYSIS_FREEZE.json')['case_order']
    checked=0;all_counts={m:[] for m in ('EXISTING_CORRECTION','LATEST_SAME_WORDING')};final_reconstruction={}
    for rid in roots:
        reconstructed=reconstruct_case(runtime,rid,'final')
        final_reconstruction[rid]=reconstructed
        expected=load(a.analysis/'cases'/(rid+'.json'))
        for method,row in reconstructed.items():
            assert all(expected[method]['metrics'][k]==v for k,v in row['counts'].items())
            all_counts[method].append(row['counts']);checked+=len(row['counts'])
            original={r['query_id']:r for r in expected[method]['primary']}
            for q,flags in row['primary'].items():
                assert all(original[q][k]==v for k,v in flags.items()),(rid,method,q)
                checked+=len(flags)
            original={r['target_ids'][0]:r for r in expected[method]['semantic']}
            for q,flags in row['semantic'].items():
                assert all(original[q][k]==v for k,v in flags.items()),(rid,method,q)
                checked+=len(flags)
    import numpy as np
    summary=load(a.analysis/'SUMMARY.json')
    draw=np.random.Generator(np.random.PCG64(2609141002)).integers(0,64,(10000,64))
    for method,counts in all_counts.items():
        values=np.array([r['W']/18 for r in counts])
        assert abs(float(values.mean())-summary[method]['primary']['witness_rate'])<1e-15
        samples=values[draw].mean(axis=1)
        for level,name in ((.95,'interval95'),(.99375,'primary_family8_interval99375')):
            tail=(1-level)/2
            assert np.array_equal(np.quantile(samples,[tail,1-tail]),summary[method]['primary'][name])
    repeated=[]
    selection_path=runtime/'REPEAT_SELECTION.json'
    if selection_path.exists():
        for rid in load(selection_path)['roots']:
            observed=reconstruct_case(runtime,rid,'repeat_full')
            repeated.append(dict(root_id=rid,classification_identical=observed==final_reconstruction[rid]))
        assert len(repeated)==4
    # This writer emits only the independently verified receipt, never labels.
    result=dict(status='PASS',roots=64,raw_final_rows=36992,classification_fields_checked=checked,
                imports_experiment_classification=False,source='RAW_SCORES.jsonl + source design schema',
                full_repeat_reconstruction=repeated)
    path=a.analysis/'INDEPENDENT_REPLAY.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
