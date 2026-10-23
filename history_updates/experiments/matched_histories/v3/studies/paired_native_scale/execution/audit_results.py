"""Mandatory supplementary ledgers; original parser/endpoints remain unchanged."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from native_worker import digest, save

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'MQMH_SCALE_V1'/'implementation'))
import study
from repro import mqmh


def repeated_selection(roots,case_ids,fixed):
    positive=sorted(r['case_id'] for r in roots if r['case_id'] not in fixed and any(q['qualified'] for q in r['targets']))[:4]
    fill=sorted(set(case_ids)-set(fixed)-set(positive))[:4-len(positive)]
    return positive,fill


def guard_summary(rows):
    overlaps=Counter()
    for row in rows:
        failures=tuple(name for name in ('Fi','C','Cplus') if row[name+'_failed'])
        overlaps['+'.join(failures) or 'none']+=1
    return dict(scheduled_targets=len(rows),overlapping_failure_cells=dict(sorted(overlaps.items())),
        Fi_failed=sum(r['Fi_failed'] for r in rows),C_failed=sum(r['C_failed'] for r in rows),
        Cplus_failed=sum(r['Cplus_failed'] for r in rows),
        qualified=sum(not any(r[f+'_failed'] for f in ('Fi','C','Cplus')) for r in rows))


def supplementary(small,large,small_repeat,large_repeat,out):
    out=study.fresh(out)
    cohort,historical=study.inputs()
    cases={c['case_id']:c for c in cohort['cases']}
    ids=cohort['case_ids']
    cfg=json.loads((ROOT/'MQMH_SCALE_V1'/'SCALE_CONFIG.json').read_text(encoding='utf-8'))
    primary_paths={'small':small,'large':large}
    repeat_paths={'small':small_repeat,'large':large_repeat}
    scored={m:study.scored_rows(study.load_rows(p),cases) for m,p in primary_paths.items()}
    repeats={m:study.scored_rows(study.load_rows(p),cases) for m,p in repeat_paths.items()}
    roots={}
    subjects=[];guards=[];observations=[];repeat_observations=[];repeat_targets=[];overlaps=[]
    summaries={};repeat_summaries={}
    for model,sc in scored.items():
        roots[model]={};summaries[model]={}
        for strict,label in ((False,'primary'),(True,'strict')):
            ek='strict_entity_id' if strict else 'entity_id'
            vk='strict_valid' if strict else 'valid'
            ck='strict_correct' if strict else 'correct'
            current_roots=[mqmh.endpoints(cases[cid],sc,strict=strict) for cid in ids]
            roots[model][label]=current_roots
            for cid,root in zip(ids,current_roots):
                study.validate_independent(cases[cid],sc,root,strict)
                subjects.append(dict(model=model,parser=label,case_id=cid,subject_id=cases[cid]['subject_id'],
                    hops=cases[cid]['hops'],guarded_targets=sum(q['common_guard'] for q in root['targets']),
                    witnesses=sum(q['qualified'] for q in root['targets']),
                    relevance_difference=sum(q['relevance_difference'] for q in root['targets'])/3,
                    cross_phrasing=root['cross_phrasing'],K_pass=root['suffix'],
                    strong_long_cross_phrasing=root['cross_phrasing'] and root['suffix'] is True))
                for q in ('J0','J1','J2'):
                    failures=[f'{a}:F{i}' for a in mqmh.ARMS for i in range(cases[cid]['hops']) if not sc[cid,a,f'F{i}'][ck]]
                    guards.append(dict(model=model,parser=label,case_id=cid,query_id=q,
                        Fi_failed=bool(failures),Fi_failure_details='|'.join(failures),
                        C_failed=not sc[cid,'C',q][ck],Cplus_failed=not sc[cid,'C+',q][ck]))
            for (cid,arm,q),row in sc.items():
                entity=row[ek];correct=row[ck]
                category='correct' if correct else ('invalid' if entity is None else
                    'old_terminal_wrong' if entity==cases[cid]['old_terminal'] else 'other_valid_wrong')
                observations.append(dict(model=model,parser=label,case_id=cid,arm=arm,query_id=q,
                    query_family='target' if q.startswith('J') else 'K' if q=='K' else 'Fdecoy' if q=='Fdecoy' else 'Fi',
                    entity_id=entity,valid=row[vk],correct=correct,error=not correct,error_category=category,
                    truncated=row['truncated'],control_tokens_valid=row['control_tokens_valid'],
                    extra_lines=len(row['text'].strip().splitlines())>1,
                    text=row['text']))
            guard_rows=[r for r in guards if r['model']==model and r['parser']==label]
            subset=[r for r in observations if r['model']==model and r['parser']==label]
            summary=guard_summary(guard_rows)
            summary['raw_by_arm_and_family']={}
            for arm in mqmh.ALL_ARMS:
                summary['raw_by_arm_and_family'][arm]={}
                for family in ('target','Fi','Fdecoy','K'):
                    selected=[r for r in subset if r['arm']==arm and r['query_family']==family]
                    summary['raw_by_arm_and_family'][arm][family]=dict(n=len(selected),
                        correct=sum(r['correct'] for r in selected),invalid=sum(not r['valid'] for r in selected),
                        errors=sum(r['error'] for r in selected),truncated=sum(r['truncated'] for r in selected),
                        extra_lines=sum(r['extra_lines'] for r in selected),
                        error_categories=dict(Counter(r['error_category'] for r in selected)))
            summary['cross_phrasing_subjects']=sum(r['cross_phrasing'] for r in current_roots)
            summary['long_K_denominator']=195
            summaries[model][label]=summary
        positive,fill=repeated_selection(roots[model]['primary'],ids,cfg['evaluation']['prespecified_repeat_cases'])
        repeat_ids=set(cfg['evaluation']['prespecified_repeat_cases']+positive+fill)
        rp=repeats[model]
        if len(repeat_ids)!=12 or {k[0] for k in rp}!=repeat_ids:
            raise ValueError('Repeats do not follow the fixed eight plus positive/fill rule')
        if {r['recorded_process_label'] for r in rp.values()} & {r['recorded_process_label'] for r in sc.values()}:
            raise ValueError('Repeats were not measured in new processes')
        stable_cases={}
        target_counts={}
        for strict,label in ((False,'primary'),(True,'strict')):
            ek='strict_entity_id' if strict else 'entity_id'
            vk='strict_valid' if strict else 'valid'
            ck='strict_correct' if strict else 'correct'
            stable_cases[label]={}
            for cid in sorted(repeat_ids):
                repeat_root=mqmh.endpoints(cases[cid],rp,strict=strict)
                primary_root=next(r for r in roots[model][label] if r['case_id']==cid)
                keys=[k for k in rp if k[0]==cid]
                stable_cases[label][str(cid)]=all(tuple(rp[k][f] for f in (vk,ek,ck))==tuple(sc[k][f] for f in (vk,ek,ck)) for k in keys)
                for k in keys:
                    old,new=sc[k],rp[k]
                    repeat_observations.append(dict(model=model,parser=label,case_id=cid,arm=k[1],query_id=k[2],
                        primary_entity=old[ek],repeat_entity=new[ek],primary_valid=old[vk],repeat_valid=new[vk],
                        primary_correct=old[ck],repeat_correct=new[ck],
                        semantic_stable=tuple(old[f] for f in (vk,ek,ck))==tuple(new[f] for f in (vk,ek,ck)),
                        exact_tokens_stable=old['output_token_ids']==new['output_token_ids']))
                for old,new in zip(primary_root['targets'],repeat_root['targets']):
                    repeat_targets.append(dict(model=model,parser=label,case_id=cid,query_id=old['query_id'],
                        primary_witness=old['qualified'],repeat_witness=new['qualified'],
                        primary_guard=old['common_guard'],repeat_guard=new['common_guard'],
                        retained_counterexample=old['qualified'] and new['qualified'],
                        complete_case_semantically_stable=stable_cases[label][str(cid)]))
            rows=[r for r in repeat_targets if r['model']==model and r['parser']==label]
            target_counts[label]=dict(primary_witnesses_in_repeated_cases=sum(r['primary_witness'] for r in rows),
                retained_counterexamples=sum(r['retained_counterexample'] for r in rows),
                lost_counterexamples=sum(r['primary_witness'] and not r['repeat_witness'] for r in rows),
                newly_appearing_repeat_witnesses=sum(not r['primary_witness'] and r['repeat_witness'] for r in rows),
                witnesses_in_completely_stable_cases=sum(r['primary_witness'] and r['complete_case_semantically_stable'] for r in rows))
        repeat_summaries[model]=dict(cases=12,rows=len(rp),fixed=cfg['evaluation']['prespecified_repeat_cases'],
            selected_positive=positive,fill=fill,stable_cases=stable_cases,target_stability=target_counts,
            inference_scope='Selected repeats do not certify nonrepeated targets; primary rows never replaced')
    common={}
    for label in ('primary','strict'):
        a=[q for r in roots['small'][label] for q in r['targets']]
        b=[q for r in roots['large'][label] for q in r['targets']]
        pairs=[(x,y) for x,y in zip(a,b) if x['common_guard'] and y['common_guard']]
        n=len(pairs)
        common[label]=dict(fixed_overlap_denominator=n,
            small_witnesses=sum(x['qualified'] for x,y in pairs),large_witnesses=sum(y['qualified'] for x,y in pairs),
            witness_rate_difference=sum(y['qualified']-x['qualified'] for x,y in pairs)/n if n else None,
            relevance_difference_contrast=sum(y['relevance_difference']-x['relevance_difference'] for x,y in pairs)/n if n else None,
            note='Descriptive subset selected by both models; not a population causal effect')
        for x,y in zip(a,b):
            overlaps.append(dict(comparison='fresh_small_vs_large',parser=label,case_id=x['case_id'],query_id=x['query_id'],
                first_witness=x['qualified'],second_witness=y['qualified'],both=x['qualified'] and y['qualified']))
    old_sc=mqmh.score(historical['qwen']['primary'],cases)
    old=[q for cid in ids for q in mqmh.endpoints(cases[cid],old_sc)['targets']]
    new=[q for r in roots['small']['primary'] for q in r['targets']]
    for x,y in zip(old,new):
        overlaps.append(dict(comparison='historical_vs_fresh_small',parser='primary',case_id=x['case_id'],query_id=x['query_id'],
            first_witness=x['qualified'],second_witness=y['qualified'],both=x['qualified'] and y['qualified']))
    for name,rows in [('SUBJECTS',subjects),('GUARD_FAILURES',guards),('OBSERVATIONS',observations),
                     ('REPEAT_OBSERVATIONS',repeat_observations),('REPEAT_TARGETS',repeat_targets),('WITNESS_OVERLAPS',overlaps)]:
        study.csv_file(out/(name+'.csv'),rows)
    result=dict(guard_and_raw_accounting=summaries,repeat_stability=repeat_summaries,
        common_qualified_descriptive=common,
        source_hashes={str(p):digest(p) for p in [small,large,small_repeat,large_repeat]},
        all_primary_rows_retained=True,all_384_primary_denominator=True,all_195_K_denominator=True)
    save(out/'SUPPLEMENTARY_SUMMARY.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('small','large','small-repeat','large-repeat','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=supplementary(a.small,a.large,a.small_repeat,a.large_repeat,a.out)
    print(json.dumps({'complete':True,'models':list(result['repeat_stability'])}))


if __name__=='__main__': main()
