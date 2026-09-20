"""Frozen primary gates and paired equal-retention utility on the same scores."""
from collections import defaultdict
import hashlib
import math
import numpy as np

CONDITIONS = ('INV_PAIR_NLL_s0', 'INV_PAIR_NLL_s1')
GATES = ('G0', 'G1', 'G2a', 'G2b', 'both')

def decode(row):
    logps = [float(x) for x in row['logps']]
    finite = all(math.isfinite(x) for x in logps)
    if len(logps) != 2 or finite != row['finite']: raise ValueError('Invalid score/finite binding')
    valid = finite and not row['tie'] and row['argmax_in_labels']
    if valid != row['valid']: raise ValueError('Invalid response validity binding')
    if finite:
        prediction = int(logps[1] > logps[0])
        if prediction != row['prediction'] or (logps[0] == logps[1]) != row['tie']:
            raise ValueError('Score/hard-answer disagreement')
        maximum = max(logps); weights = [math.exp(x-maximum) for x in logps]
        probability = [x/sum(weights) for x in weights]
        mass = sum(math.exp(x) for x in logps)
        if abs(mass-float(row['answer_mass'])) > 1e-12: raise ValueError('Candidate mass mismatch')
        if mass > 1+1e-5 or mass <= 0: raise ValueError('Invalid full-vocabulary log probabilities')
        if max(math.exp(x) for x in logps) > max(0, 1-mass)+1e-6 and not row['argmax_in_labels']:
            raise ValueError('Recorded full-vocabulary argmax contradicts the score certificate')
    else:
        prediction, probability, mass = int(row['prediction']), [None, None], None
    return dict(prediction=prediction, valid=valid, probabilities=probability, mass=mass)

def truth(values, spec):
    x, y = values[spec['a']][spec['p']], values[spec['b']][spec['p']]
    return {'direct':x, 'opposes':1-x, 'same':int(x == y), 'both':x & y, 'either':x | y}[spec['kind']]

def primary(root, questions, plan, index):
    qs = {q['query_id']:q for q in questions}; values = root['terminal']['values']
    gold = {qid:truth(values,q['spec']) for qid,q in qs.items()}
    pairs = [tuple(x) for x in plan['structural_pairs']]
    details = []
    for seed, condition in enumerate(CONDITIONS):
        def response(o, qid): return index[(f'o{o}/{condition}', qid)]
        def correct(r, qid): return r['valid'] and r['prediction'] == gold[qid]
        def all_correct(ids): return all(correct(response(o,qid),qid) for qid in ids for o in range(8))
        all_direct = all_correct(plan['gates']['direct'])
        local = {}
        for qid, mapping in plan['gates']['targets'].items():
            rows = [response(o,qid) for o in range(8)]
            native = index[('NATIVE_FINAL',qid)]
            references = correct(native,qid) and correct(rows[0],qid)
            g0 = references and all_correct(mapping['operands'])
            g1 = references and all_direct
            gates = dict(G0=g0, G1=g1, G2a=g1 and all_correct(mapping['same_pair']), G2b=g1 and all_correct(mapping['cross_pair']))
            gates['both'] = gates['G2a'] and gates['G2b']
            matched = [pair for pair in pairs if all(rows[i]['valid'] for i in pair) and rows[pair[0]]['prediction'] != rows[pair[1]]['prediction']]
            any_disagreement = len({r['prediction'] for r in rows if r['valid']}) > 1
            operands = [(response(o,k), gold[k]) for k in mapping['operands'] for o in range(8)]
            strong = (g0 and any_disagreement and all(r['valid'] for r in rows)
                and all(r['valid'] and r['probabilities'][g] >= .95 for r,g in operands)
                and all(r['mass'] is not None and r['mass'] >= .90 for r in rows+[r for r,g in operands]+[native]))
            strong_pairs = [pair for pair in matched if strong and max(rows[i]['probabilities'][gold[qid]] for i in pair) >= .95
                            and min(rows[i]['probabilities'][gold[qid]] for i in pair) <= .05]
            record = dict(root_id=root['root_id'],seed=seed,query_id=qid,draw=mapping['draw'],gold=gold[qid],
                semantic_representative=mapping['draw']==0, structural_pairs=len(pairs),
                matched_pairs=[list(x) for x in matched], strong_pairs=[list(x) for x in strong_pairs],
                predictions=[r['prediction'] for r in rows], valid=[r['valid'] for r in rows],
                correct_probabilities=[r['probabilities'][gold[qid]] for r in rows])
            for gate, passing in gates.items():
                record[gate+'_eligible'] = bool(passing)
                record[gate+'_witness'] = bool(passing and any_disagreement)
                record[gate+'_matched'] = bool(passing and matched)
                record[gate+'_strong'] = bool(passing and strong_pairs)
            local[qid] = record
        for qid, record in local.items():
            twin_id = next(x for x in plan['gates']['targets'][qid]['twins'] if x != qid)
            twin = local[twin_id]
            common = set(map(tuple,record['matched_pairs'])) & set(map(tuple,twin['matched_pairs']))
            common = {pair for pair in common if all(record['predictions'][i] == twin['predictions'][i] for i in pair)}
            common_strong = common & set(map(tuple,record['strong_pairs'])) & set(map(tuple,twin['strong_pairs']))
            for gate in GATES:
                eligible = record[gate+'_eligible'] and twin[gate+'_eligible']
                record['cross_'+gate+'_eligible'] = bool(eligible)
                record['cross_'+gate+'_matched'] = bool(eligible and common)
                record['cross_'+gate+'_strong'] = bool(eligible and common_strong)
        details.extend(local.values())
    if len(details) != 36: raise ValueError('Changed fixed primary denominator')
    return details

def bootstrap_primary(details, root_ids):
    if len(root_ids) != 64 or len(set(root_ids)) != 64: raise ValueError('All 64 final cases required')
    matrix = np.asarray([[sum(r['both_matched'] for r in details if r['root_id']==rid and r['seed']==seed)/18
                          for seed in (0,1)] for rid in root_ids])
    rng = np.random.default_rng(2609182102)
    means = matrix.mean(1)[rng.integers(64,size=(10000,64))].mean(1)
    per_seed = []
    for seed in (0,1):
        rows = [r for r in details if r['seed']==seed]
        if len(rows)!=1152: raise ValueError('Per-seed denominator changed')
        per_seed.append(dict(seed=seed, denominator=1152, matched=sum(r['both_matched'] for r in rows),
            eligible=sum(r['both_eligible'] for r in rows), contributing_cases=len({r['root_id'] for r in rows if r['both_matched']})))
    decomposition=[]
    for seed in (0,1):
        for draw in (0,1):
            rows=[r for r in details if r['seed']==seed and r['draw']==draw]
            decomposition.append(dict(seed=seed,draw=draw,denominator=576,
                gates={g:{s:sum(r[g+'_'+s] for r in rows) for s in ('eligible','witness','matched','strong')} for g in GATES},
                contributing_cases={g:len({r['root_id'] for r in rows if r[g+'_matched']}) for g in GATES}))
    cross=[]
    for seed in (0,1):
        rows=[r for r in details if r['seed']==seed and r['semantic_representative']]
        cross.append(dict(seed=seed,denominator=576,
            gates={g:{s:sum(r['cross_'+g+'_'+s] for r in rows) for s in ('eligible','matched','strong')} for g in GATES}))
    return dict(rate=float(matrix.mean()), interval95=list(map(float,np.quantile(means,[.025,.975]))),
        cases=64, denominator_per_seed=1152, per_seed=per_seed,
        contributing_cases=len({r['root_id'] for r in details if r['both_matched']}), bootstrap_draws=10000, bootstrap_seed=2609182102,
        seed_code_decomposition=decomposition,cross_code_common_pair=cross,
        structural_opportunity_coverage=sum(r['structural_pairs']>0 for r in details)/2304)

def decision_packet(procedure, designs, decoded):
    cases = []
    for root, questions, plan in designs:
        rid = root['root_id']; qs={q['query_id']:q for q in questions}
        for seed, condition in enumerate(CONDITIONS):
            for draw in (0,1):
                partition=plan['utility'][str(draw)]; selected=plan['selected_anchor_pair']
                common_keys=[]; validation_keys=[]
                if selected is not None:
                    anchor, other=selected
                    contexts=[f'o{anchor}/{condition}', f'o{other}/{condition}', 'NATIVE_FINAL', f'o0/{condition}']
                    common_keys=sorted({(context,qid) for context in contexts for qid in partition['direct']})
                    if procedure=='A': validation_keys=[(f'o{anchor}/{condition}',q) for q in partition['accuracy_queries']]
                    else: validation_keys=[(f'o{o}/{condition}',q) for q in partition['history_queries'] for o in selected]
                forbidden=set(partition['withheld_queries'])
                if any(q in forbidden for c,q in common_keys+validation_keys): raise ValueError('Withheld input contamination')
                def safe(keys):
                    return [dict(context=c,query_id=q,prediction=decoded[rid][c,q]['prediction'],valid=decoded[rid][c,q]['valid'],
                                 gold=truth(root['terminal']['values'],qs[q]['spec'])) for c,q in keys]
                cases.append(dict(root_id=rid,seed=seed,draw=draw,tie_breaker=hashlib.sha256(('retention|'+rid).encode()).hexdigest(),
                    applicable=selected is not None,common=safe(common_keys),validation=safe(validation_keys),expected_keys=common_keys+validation_keys))
    return dict(schema='FHV1_DECISION_INPUT_V1',procedure=procedure,cases=cases)

def rankings(decisions):
    result = []
    for seed in (0,1):
        for draw in (0,1):
            for procedure in ('A','B'):
                eligible=[r for r in decisions if r['seed']==seed and r['draw']==draw and r['procedure']==procedure and r['common_eligible']]
                ranked=sorted(eligible,key=lambda r:(r['wrong_validation'],r['tie_breaker']))
                result.append(dict(seed=seed,draw=draw,procedure=procedure,common_count=len(eligible),ranking=[r['root_id'] for r in ranked],
                    retained={str(f):[r['root_id'] for r in ranked[:math.floor(f*len(ranked))]] for f in (.25,.5,.75)}))
    return result

def withheld_losses(designs,index):
    rows=[]
    for root,questions,plan in designs:
        pair=plan['selected_anchor_pair']
        if pair is None: continue
        qs={q['query_id']:q for q in questions};rid=root['root_id']
        for seed,condition in enumerate(CONDITIONS):
            for draw in (0,1):
                ids=plan['utility'][str(draw)]['withheld_queries']; failures=[]; disagreements=[]; native_failures=[]
                for qid in ids:
                    gold=truth(root['terminal']['values'],qs[qid]['spec'])
                    answers=[index[rid][f'o{o}/{condition}',qid] for o in pair]
                    failures.extend(not r['valid'] or r['prediction']!=gold for r in answers)
                    disagreements.append(all(r['valid'] for r in answers) and answers[0]['prediction']!=answers[1]['prediction'])
                    native=index[rid]['NATIVE_FINAL',qid]
                    native_failures.append(not native['valid'] or native['prediction']!=gold)
                rows.append(dict(root_id=rid,seed=seed,draw=draw,task_error=sum(failures)/6,any_error=any(failures),
                    valid_disagreement=sum(disagreements)/3,native_error=sum(native_failures)/3))
    return rows

def utility_summary(decisions, losses, root_ids, draws=10000):
    dec={(r['root_id'],r['seed'],r['draw'],r['procedure']):r for r in decisions}
    loss={(r['root_id'],r['seed'],r['draw']):r for r in losses}
    def summarize(ids,seed,draw):
        common=[rid for rid in ids if dec[rid,seed,draw,'A']['common_eligible']]
        if any(dec[rid,seed,draw,'A']['common_eligible']!=dec[rid,seed,draw,'B']['common_eligible'] for rid in ids):
            raise ValueError('Comparison common eligibility differs')
        def stats(selected):
            n=len(selected)
            return dict(count=n, task_error=sum(loss[rid,seed,draw]['task_error'] for rid in selected)/n if n else None,
                any_error=sum(loss[rid,seed,draw]['any_error'] for rid in selected)/n if n else None,
                valid_disagreement=sum(loss[rid,seed,draw]['valid_disagreement'] for rid in selected)/n if n else None,
                successful=sum(not loss[rid,seed,draw]['any_error'] for rid in selected),
                native_error=sum(loss[rid,seed,draw]['native_error'] for rid in selected)/n if n else None)
        out=dict(seed=seed,draw=draw,common_cases=len(common),common=stats(common),policies={},contrasts={})
        for procedure in ('A','B'):
            accepted=[rid for rid in common if dec[rid,seed,draw,procedure]['accept']]
            ranked=sorted(common,key=lambda rid:(dec[rid,seed,draw,procedure]['wrong_validation'],dec[rid,seed,draw,procedure]['tie_breaker']))
            out['policies'][procedure]=dict(accepted=stats(accepted),rejected=len(common)-len(accepted),
                equal_retention={str(f):stats(ranked[:math.floor(f*len(common))]) for f in (.25,.5,.75)})
        for f in (.25,.5,.75):
            a=out['policies']['A']['equal_retention'][str(f)]['task_error'];b=out['policies']['B']['equal_retention'][str(f)]['task_error']
            out['contrasts'][str(f)]=None if a is None or b is None else b-a
        for left,right in (('A','B'),('B','A')):
            selected=[rid for rid in common if dec[rid,seed,draw,left]['accept'] and not dec[rid,seed,draw,right]['accept']]
            out[left+'_only']=dict(stats=stats(selected),cases=selected)
        return out
    result={'per_seed_format':[summarize(root_ids,s,d) for d in (0,1) for s in (0,1)],'combined':[]}
    # Every resample keeps the paired root IDs, seeds, and formats; duplicates remain weighted observations.
    rng=np.random.default_rng(2609182102)
    resamples=rng.integers(64,size=(draws,64))
    for draw in (0,1):
        originals=[summarize(root_ids,s,draw) for s in (0,1)]
        endpoint=[r['contrasts']['0.5'] for r in originals]
        values=[];undefined=0;seed_values=[[],[]];seed_undefined=[0,0]
        for sample in resamples:
            ids=[root_ids[i] for i in sample]
            contrasts=[summarize(ids,s,draw)['contrasts']['0.5'] for s in (0,1)]
            for seed,contrast in enumerate(contrasts):
                if contrast is None:seed_undefined[seed]+=1
                else:seed_values[seed].append(contrast)
            if any(x is None for x in contrasts): undefined+=1
            else: values.append(sum(contrasts)/2)
        for row in result['per_seed_format']:
            if row['draw']==draw:
                seed=row['seed'];row['interval95_50pct']=None if seed_undefined[seed] else list(map(float,np.quantile(seed_values[seed],[.025,.975])))
                row['undefined_bootstrap_draws']=seed_undefined[seed]
        result['combined'].append(dict(draw=draw,interface='literal No/Yes' if draw==0 else 'A/B sensitivity',
            quantitative_headline_available=all(r['common_cases']>=20 for r in originals),
            B_minus_A_50pct=None if any(x is None for x in endpoint) else sum(endpoint)/2,
            interval95=None if undefined else list(map(float,np.quantile(values,[.025,.975]))),
            undefined_bootstrap_draws=undefined,bootstrap_draws=draws))
    return result
