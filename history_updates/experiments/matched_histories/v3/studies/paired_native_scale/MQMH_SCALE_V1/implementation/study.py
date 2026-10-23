"""CPU preparation and analysis for the fixed paired MQMH study.

This module does not rent GPUs or execute models. Original parser/endpoint
functions are imported without modification. A separate implementation checks
the central guard/witness/relevance arithmetic.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from repro import mqmh
from repro.release_inputs import verify, payload


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fresh(path: Path) -> Path:
    if path.exists():
        raise ValueError('Output already exists; preserve prior evidence and choose a fresh directory')
    path.mkdir(parents=True)
    return path


def inputs():
    manifest = verify(mqmh.SOURCE, 'mqmh')
    data = {r['member']: json.loads(payload(mqmh.SOURCE, r)) for r in manifest['files']}
    return data['cohort.json'], {m: data[m+'_responses.json'] for m in mqmh.MODELS}


def messages(context: str, question: str, cfg: dict) -> list[dict]:
    return [{'role': 'system', 'content': cfg['inference']['system_prompt']},
            {'role': 'user', 'content': cfg['inference']['user_template'].format(context=context, question=question)}]


def prepare(out: Path) -> dict:
    cohort, _ = inputs()
    cfg = json.loads((ROOT/'SCALE_CONFIG.json').read_text())
    output = fresh(out)
    rows = []
    cases = {c['case_id']:c for c in cohort['cases']}
    for cid in cohort['case_ids']:
        case = cases[cid]
        for arm in mqmh.ALL_ARMS:
            for query in case['queries']:
                msg = messages(case['contexts'][arm], query['question'], cfg)
                rows.append(dict(case_id=cid, arm=arm, query_id=query['query_id'], messages=msg))
    if len(rows) != 5406 or len({(r['case_id'],r['arm'],r['query_id']) for r in rows}) != 5406:
        raise ValueError('Frozen request inventory mismatch')
    raw = ''.join(json.dumps(r, ensure_ascii=False, separators=(',',':'))+'\n' for r in rows).encode()
    (output/'runtime_requests.jsonl').write_bytes(raw)
    canaries = json.loads((ROOT/'TECHNICAL_CANARIES.json').read_text())
    # Only the canary checker, not the inference worker, needs the expected field.
    canary_runtime = [dict(canary_id=c['canary_id'], messages=messages(c['context'],c['question'],cfg)) for c in canaries]
    dump(output/'canary_requests.json', canary_runtime)
    receipt = dict(requests=len(rows), subjects=128, targets=384, long_targets=195,
                   sha256=hashlib.sha256(raw).hexdigest(), model_visible_fields='messages only',
                   no_gold_or_registry_fields=True, wrapper='prospectively fixed new common wrapper')
    dump(output/'PREPARATION.json', receipt)
    return receipt


def load_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    required = {'case_id','arm','query_id','text','truncated','control_tokens_valid'}
    for r in rows:
        if not required <= r.keys() or type(r['truncated']) is not bool or type(r['control_tokens_valid']) is not bool:
            raise ValueError('Raw row schema mismatch; require real Boolean validity flags')
    return rows


def scored_rows(rows: list[dict], cases: dict) -> dict:
    parsed = []
    for row in rows:
        if row['case_id'] not in cases:
            raise ValueError('Unknown case ID')
        entity, strict = mqmh.parse_response(row, cases[row['case_id']])
        fields = dict(entity_id=entity, valid=entity is not None, strict_entity_id=strict, strict_valid=strict is not None)
        # Refuse contradictory saved classifications instead of silently overwriting.
        for key,value in fields.items():
            if key in row and row[key] != value:
                raise ValueError('Saved classification differs from frozen parser')
        parsed.append(dict(row, **fields))
    return mqmh.score(parsed, cases)


def independent_endpoint(case: dict, scored: dict, strict: bool=False) -> list[dict]:
    """No call to the original endpoint code; independently reproduce its logic."""
    cid=case['case_id']; ek='strict_entity_id' if strict else 'entity_id'
    gold={q['query_id']:q['gold'] for q in case['queries']}
    def ok(a,q):return scored[(cid,a,q)][ek] is not None and scored[(cid,a,q)][ek]==gold[q]
    facts=all(ok(a,f'F{i}') for i in range(case['hops']) for a in ('H0','H1','S0','S1'))
    suffix=all(ok(a,'K') for a in mqmh.ARMS) if case['hops']>2 else None
    out=[]
    for q in ('J0','J1','J2'):
        guard=facts and ok('C',q) and ok('C+',q)
        answers=[scored[cid,a,q][ek] for a in mqmh.ARMS]
        correct=[ok(a,q) for a in mqmh.ARMS]
        # At least one correct answer and a distinct valid incorrect answer
        # is equivalent to one of the six ordered-by-arm pairs qualifying.
        witnessed=bool(guard and any(correct) and any(e is not None and not c for e,c in zip(answers,correct)))
        err=[int(not c) for c in correct]
        out.append(dict(query_id=q,common_guard=guard,qualified=witnessed,
                        relevance_difference=(err[0]+err[1]-err[2]-err[3])/2 if guard else 0,
                        long_chain_qualified=bool(witnessed and suffix) if suffix is not None else None))
    return out


def validate_independent(case: dict, scored: dict, original: dict, strict: bool=False) -> None:
    for got, expected in zip(independent_endpoint(case,scored,strict), original['targets']):
        for key, value in got.items():
            if expected[key] != value:
                raise AssertionError(f'Independent endpoint mismatch {case["case_id"]} {key}')


def csv_file(path:Path, rows:list[dict]) -> None:
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(small:Path, large:Path, out:Path) -> dict:
    cohort,historical=inputs();cases={c['case_id']:c for c in cohort['cases']};ids=cohort['case_ids']
    scored={name:scored_rows(load_rows(path),cases) for name,path in [('small',small),('large',large)]}
    summaries={};all_roots={};targets=[];attrition=[];raw_stats={}
    for name,sc in scored.items():
        all_roots[name]={};summaries[name]={}
        for strict,label in [(False,'primary'),(True,'strict_canonical')]:
            roots=[mqmh.endpoints(cases[cid],sc,strict=strict) for cid in ids]
            for cid,root in zip(ids,roots): validate_independent(cases[cid],sc,root,strict)
            all_roots[name][label]=roots;summaries[name][label]=mqmh.summarize(roots)
            s=summaries[name][label];s['conditional_witness_rate']=s['qualified']/s['common_guard'] if s['common_guard'] else None
            targets.extend(dict(model=name,parser=label,**row) for r in roots for row in r['targets'])
        for cid in ids:
            c=cases[cid]
            for q in ('J0','J1','J2'):
                failed_facts=[f'{a}:F{i}' for a in mqmh.ARMS for i in range(c['hops']) if not sc[cid,a,f'F{i}']['correct']]
                ans={a:sc[cid,a,q]['entity_id'] for a in mqmh.ARMS}
                attrition.append(dict(model=name,case_id=cid,query_id=q,failed_facts='|'.join(failed_facts),
                    C_correct=sc[cid,'C',q]['correct'],Cplus_correct=sc[cid,'C+',q]['correct'],
                    all_history_answers_valid=all(v is not None for v in ans.values()),
                    correct_consensus=all(sc[cid,a,q]['correct'] for a in mqmh.ARMS),
                    wrong_consensus=all(v is not None for v in ans.values()) and len(set(ans.values()))==1 and not sc[cid,'H0',q]['correct']))
        stats={}
        for a in mqmh.ALL_ARMS:
            rows=[r for k,r in sc.items() if k[1]==a and k[2] in ('J0','J1','J2')]
            stats[a]=dict(n=len(rows),correct=sum(r['correct'] for r in rows),invalid=sum(not r['valid'] for r in rows),
                truncated=sum(r['truncated'] for r in rows),extra_prose_lines=sum(len(r['text'].strip().splitlines())>1 for r in rows),
                wrong_old_terminal=sum(r['valid'] and not r['correct'] and r['entity_id']==cases[r['case_id']]['old_terminal'] for r in rows))
        raw_stats[name]=stats
    paired={};transition=[]
    for label in ('primary','strict_canonical'):
        ra,rb=all_roots['small'][label],all_roots['large'][label]
        point={}
        for field in ('qualified','relevance_difference'):
            diff=[sum(float(b[field])-float(a[field]) for a,b in zip(sa['targets'],lb['targets']))/3 for sa,lb in zip(ra,rb)]
            point[field]=dict(large_minus_small=float(np.mean(diff)),interval_98_75=mqmh.interval(diff))
        shared=[(a,b) for x,y in zip(ra,rb) for a,b in zip(x['targets'],y['targets']) if a['common_guard'] and b['common_guard']]
        point['shared_qualified_descriptive']=dict(n=len(shared),small_witnesses=sum(a['qualified'] for a,b in shared),large_witnesses=sum(b['qualified'] for a,b in shared),note='Selected by both models; not a population causal effect')
        paired[label]=point
        for x,y in zip(ra,rb):
            for a,b in zip(x['targets'],y['targets']):
                transition.append(dict(parser=label,case_id=a['case_id'],query_id=a['query_id'],small_guard=a['common_guard'],large_guard=b['common_guard'],small_witness=a['qualified'],large_witness=b['qualified']))
    old_sc=mqmh.score(historical['qwen']['primary'],cases)
    old=[mqmh.endpoints(cases[cid],old_sc) for cid in ids]
    oldset={(x['case_id'],q['query_id']) for x in old for q in x['targets'] if q['qualified']}
    newset={(x['case_id'],q['query_id']) for x in all_roots['small']['primary'] for q in x['targets'] if q['qualified']}
    overlap=dict(old=len(oldset),fresh_small=len(newset),intersection=len(oldset&newset),old_only=len(oldset-newset),fresh_only=len(newset-oldset))
    result=dict(models=summaries,paired=paired,raw_target_stats=raw_stats,historical_small_overlap=overlap,
        completed_primary_reads={k:len(v) for k,v in scored.items()},independent_endpoint_check=True,
        repeat_checks_included=False,note='Complete required repeat audit and operational closeout before final study verdict. Bootstrap-zero intervals are not population upper bounds.')
    output=fresh(out);dump(output/'SUMMARY.json',result);csv_file(output/'TARGETS.csv',targets);csv_file(output/'GUARD_ATTRITION.csv',attrition);csv_file(output/'PAIRED_TRANSITIONS.csv',transition)
    dump(output/'RAW_INPUT_HASHES.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (small,large)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    for name in ('historical','prepare'):
        q=sp.add_parser(name);q.add_argument('--out',type=Path,required=True)
    q=sp.add_parser('analyze');q.add_argument('--small',type=Path,required=True);q.add_argument('--large',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='historical':result=mqmh.run(a.out)
    elif a.command=='prepare':result=prepare(a.out)
    else:result=analyze(a.small,a.large,a.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
