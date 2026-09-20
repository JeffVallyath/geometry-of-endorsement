"""Replay the post-hoc V10 CPU audit from frozen scores, without inference.

Adapted from the supplied audit workspace check_audit.py. Source identity and
unchanged audit artifacts are recorded in cpu_audit/MANIFEST.json.
"""
from pathlib import Path
import argparse,csv,gzip,itertools,json,math
import importlib.util
from repro.common import ROOT, new_output, safe_path, sha256

BASE = ROOT / 'reproducibility/state_sufficiency/cpu_audit'
INPUTS = BASE.parent / 'independent_verification'
GATES = ('G0', 'G1', 'G2a', 'G2b', 'G2ab')


def load(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)
def truth(q,values):
    s=q['spec'];a=values[s['a']][s['p']];b=values[s['b']][s['p']]
    return {'direct':a,'opposes':1-a,'same':int(a==b),'both':a&b,'either':a|b}[s['kind']]
def sem(q):
    s=q['spec'];return tuple(sorted((s['a'],s['b']))),s['p'],s['kind']

def compare_value(actual, expected, *, probabilities=False):
    """Only probabilities use the original absolute 1e-12 tolerance."""
    if probabilities:
        if len(actual) != len(expected) or not all(
            math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
            for a, b in zip(actual, expected)
        ):
            raise ValueError('Probability mismatch')
    elif actual != expected:
        raise ValueError('Exact classification or identity mismatch')


def check_integrity(audit=BASE, package=INPUTS):
    manifest = load(audit / 'MANIFEST.json')
    for record in manifest['files']:
        path = safe_path(record['path'].replace('\\', '/'), audit)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Audit artifact hash mismatch: ' + record['path'])
    spec = importlib.util.spec_from_file_location('audit_v10_inputs', package / 'reconstruct.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.check_manifest(package)
    return verifier


def certification(dec, condition, qid, operands, direct, same_pair, cross_pair):
    """Target histories contribute only the prespecified native/no-op references."""
    def correct(ids):
        assert qid not in ids
        return all(dec[f'o{o}/{condition}', i][2] for i in ids for o in range(8))
    reference = dec['NATIVE_FINAL', qid][2] and dec[f'o0/{condition}', qid][2]
    g0 = reference and correct(operands)
    g1 = reference and correct(direct)
    a = g1 and correct(same_pair)
    b = g1 and correct(cross_pair)
    return dict(zip(GATES, (g0, g1, a, b, a and b)))


def common_pairs(left, right):
    """Both answer codes must disagree on the same pair in the same direction."""
    return [p for p in left['matched'] if p in right['matched']
            and [left['joint'][o][0] for o in p] == [right['joint'][o][0] for o in p]]


def summarize(rows):
    """Coded opportunities are pooled descriptively; designs are shared unions."""
    result = {'coded_opportunities': len(rows),
              'shared_designs': len({r['root_id'] for r in rows}),
              'original_witnesses': sum(r['G0_witness'] for r in rows), 'controls': {}}
    for gate in GATES:
        selected = [r for r in rows if r[gate + '_M_witness']]
        cross = [r for r in rows if r['answer_code_draw'] == '0'
                 and r['crosscode_' + gate + '_M_witness']]
        result['controls'][gate] = {
            'matched': len(selected), 'designs': len({r['root_id'] for r in selected}),
            'learned': sum(r['method'] in ('INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY') for r in selected),
            'textual': sum(r['method'] in ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING') for r in selected),
            'crosscode_semantic_matched': len(cross),
            'crosscode_designs': len({r['root_id'] for r in cross})}
    return result


def reconstruct(audit=BASE, package=INPUTS):
    if not __debug__:
        raise RuntimeError('Replay requires Python assertions; do not use -O')
    verifier = check_integrity(audit, package)
    spec=load(audit/'GATE_SPEC.json')
    assert set(spec['mappings']) == set(verifier.ROOTS)
    with (audit/'RESULTS.csv').open(encoding='utf-8', newline='') as stream:
        rows=list(csv.DictReader(stream))
    ri={(r['model'],r['root_id'],r['condition'],r['query_id']):r for r in rows};assert len(ri)==len(rows)==13824
    flags_checked=0;mapping_checks=0;results=[];conditions=load(package/'inputs/protocol.json')['inventory']['conditions'][1:]
    gates=('G0','G1','G2a','G2b','G2ab')
    for rid in spec['mappings']:
        design=load(package/f'inputs/designs/{rid}.json');world=design['root'];values=world['terminal']['values'];qs=verifier.validate_design(design,rid)
        qs_draw={qid:int(qid.split('-d')[1][0]) for qid in qs}
        joints=[q for q in qs.values() if q['spec']['kind'] in ('same','both','either')];direct=[q['query_id'] for q in qs.values() if q['spec']['kind']=='direct']
        changed=[sum(s['values'][c['actor']][c['project']]!=c['value'] for c in world['commands']) for s in world['sources']]
        assert changed==[0,1,1,2,1,2,2,3]
        for model in ('gemma','qwen'):
            panel=load(package/f'inputs/responses/{model}/{rid}/core.json.gz');raw={r['context']+'|'+r['query_id']:r for r in panel['physical']};dec={}
            for b in panel['bindings']:
                rr=raw[b['physical_id']];lp=rr['logps'];pred=int(lp[1]>lp[0]);valid=lp[0]!=lp[1] and all(math.isfinite(x) for x in lp)
                mass=sum(math.exp(x) for x in lp);assert max(math.exp(x) for x in lp)>max(0,1-mass)+1e-6
                pg=1/(1+math.exp(-abs(lp[1]-lp[0])));gold=truth(qs[b['query_id']],values)
                dec[b['context'],b['query_id']]=(pred,valid,valid and pred==gold,pg if pred==gold else 1-pg,mass)
            for condition in conditions:
                lengths=[len(panel['contexts'][f'o{o}/{condition}']['prefix_ids']) for o in range(8)]
                allowed=[list(x) for x in itertools.combinations(range(8),2) if changed[x[0]]==changed[x[1]] and lengths[x[0]]==lengths[x[1]]]
                local={}
                for q in joints:
                    qid=q['query_id'];pair,project,kind=sem(q);draw=qs_draw[qid]
                    operands=[i for i in direct if qs[i]['spec']['a'] in pair and qs[i]['spec']['p']==project and qs_draw[i]==draw]
                    sa=[x['query_id'] for x in joints if sem(x)[:2]==(pair,project) and sem(x)[2]!=kind and qs_draw[x['query_id']]==draw]
                    sb=[x['query_id'] for x in joints if sem(x)[0]!=pair and sem(x)[1]==project and qs_draw[x['query_id']]==draw]
                    m=spec['mappings'][rid]['targets'][qid]
                    assert set(sa)==set(m['G2a_same_draw_checks']) and len(sa)==2
                    assert set(sb)==set(m['G2b_same_draw_checks']) and len(sb)==6;mapping_checks+=1
                    js=[dec[f'o{o}/{condition}',qid] for o in range(8)];native=dec['NATIVE_FINAL',qid]
                    gp=certification(dec,condition,qid,operands,direct,sa,sb);g0=gp['G0']
                    disagree=len({x[0] for x in js if x[1]})>1
                    reads=[dec[f'o{o}/{condition}',i] for i in operands for o in range(8)]
                    strong=g0 and disagree and all(x[1] for x in js) and min(x[3] for x in reads)>=.95 and min(x[4] for x in reads+js+[native])>=.9 and min(x[3] for x in js)<=.05 and max(x[3] for x in js)>=.95
                    matched=[p for p in allowed if g0 and all(js[o][1] for o in p) and js[p[0]][0]!=js[p[1]][0]]
                    extremes=[p for p in matched if strong and min(js[o][3] for o in p)<=.05 and max(js[o][3] for o in p)>=.95]
                    obs=ri[model,rid,condition,qid]
                    compare_value(json.loads(obs['target_predictions']), [x[0] for x in js])
                    compare_value(json.loads(obs['target_valid']), [x[1] for x in js])
                    compare_value(json.loads(obs['target_correct_probabilities']), [x[3] for x in js], probabilities=True)
                    assert obs['method'] == (condition[:-3] if condition.endswith(('_s0','_s1')) else condition)
                    assert obs['seed'] == (condition[-1] if condition.endswith(('_s0','_s1')) else '')
                    assert obs['answer_code_draw'] == str(draw) and (obs['semantic_representative']=='True') == (draw==0)
                    assert int(obs['gold']) == truth(q,values) and json.loads(obs['actor_pair']) == list(pair)
                    assert obs['function'] == kind and obs['semantic_id'] == m['semantic_id']
                    assert json.loads(obs['count_matched_pairs']) == [list(p) for p in itertools.combinations(range(8),2) if changed[p[0]]==changed[p[1]]]
                    assert json.loads(obs['changed_fact_counts'])==changed and json.loads(obs['prefix_token_counts'])==lengths
                    assert json.loads(obs['matched_pairs'])==allowed and json.loads(obs['matched_disagreeing_pairs'])==matched and json.loads(obs['matched_strong_pairs'])==extremes
                    derived={'original_strong':strong,'original_D':disagree,'pair_availability':bool(allowed)}
                    for gate in gates:
                        derived[gate+'_pass']=gp[gate];derived[gate+'_witness']=gp[gate] and disagree
                        derived[gate+'_M_witness']=gp[gate] and bool(matched);derived[gate+'_strong_M']=gp[gate] and bool(extremes)
                    for field,value in derived.items():assert (obs[field]=='True')==value,(field,model,rid,condition,qid)
                    flags_checked+=len(derived)
                    local[qid]={'obs':obs,'flags':derived,'gates':gp,'joint':js,'matched':matched,'extremes':extremes,'draw':draw,'semantic':sem(q)}
                for qid,x in local.items():
                    twin=next(y for i,y in local.items() if y['semantic']==x['semantic'] and y['draw']!=x['draw'])
                    same=common_pairs(x,twin)
                    extreme=[p for p in same if p in x['extremes'] and p in twin['extremes']]
                    obs=x['obs'];assert same==json.loads(obs['crosscode_matched_pairs']) and extreme==json.loads(obs['crosscode_strong_pairs'])
                    check={'original_both_code_W':x['flags']['G0_witness'] and twin['flags']['G0_witness']}
                    for gate in gates:
                        ok=x['gates'][gate] and twin['gates'][gate]
                        check[gate+'_crosscode_pass']=ok;check['crosscode_'+gate+'_witness']=ok and bool(same)
                        check['crosscode_'+gate+'_M_witness']=ok and bool(same);check['crosscode_'+gate+'_strong_M']=ok and bool(extreme)
                    for field,value in check.items():assert (obs[field]=='True')==value,(field,model,rid,condition,qid)
                    flags_checked+=len(check)
                    results.append({**obs,**x['flags'],**check})
    survivors=load(audit/'SURVIVORS.json')
    selected=[r for r in results if r['G2a_M_witness'] or r['G2b_M_witness']]
    identity=lambda r:(r['model'],r['root_id'],r['condition'],r['query_id'])
    assert {identity(r) for r in selected} == {identity(r) for r in survivors['cases']}
    assert len(selected) == len(survivors['cases'])
    return {'summary': summarize(results), 'rows': results,
            'verification': {'status':'PASS', 'opportunities':len(results),
                'independent_boolean_flag_checks':flags_checked,
                'independent_question_mapping_checks':mapping_checks,
                'probability_tolerance': {'atol': 1e-12, 'rtol': 0},
                'classification_equality': 'exact', 'inference': False}}


def export(output, audit=BASE, package=INPUTS):
    if output.exists():
        raise ValueError('Output exists; use a fresh directory')
    result = reconstruct(audit, package)
    output = new_output(output)
    for name in ('summary', 'verification'):
        with (output / (name + '.json')).open('x', encoding='utf8', newline='\n') as stream:
            json.dump(result[name], stream, indent=2, sort_keys=True)
            stream.write('\n')
    records = []
    for (model, condition) in sorted({(r['model'], r['condition']) for r in result['rows']}):
        rows = [r for r in result['rows'] if (r['model'], r['condition']) == (model, condition)]
        for gate, counts in summarize(rows)['controls'].items():
            records.append({'model': model, 'condition': condition, 'control': gate,
                            'coded_opportunities': len(rows), **counts})
    with (output / 'by_condition.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return result['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=BASE)
    parser.add_argument('--package', type=Path, default=INPUTS)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.audit, args.package), indent=2))


if __name__ == '__main__':
    main()
