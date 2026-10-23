"""Paper-level access to the retained carrier and reader measurements.

This exposes the existing measured-score fields and checks the recorded reader
fractions against their original log-odds definition. It performs no inference
or tensor intervention and does not change the source selection rule.
"""
import collections
import importlib
import math
from pathlib import Path
import sys
import types


def reconstruct(legacy):
    name='_paper_legacy_score_accounting'
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(Path(legacy)/'src/repro')]
        sys.modules[name]=package
    frozen=importlib.import_module(name+'.completed_followups')
    source=Path(legacy)/'reproducibility/completed_followups'
    output={'reproduction_level':'Exact retained score accounting; no tensor/model execution','studies':{}}
    for study in frozen.STUDIES:
        binding,saved=frozen.read_records(source,study)
        scores=saved('ALL_SCORES_AND_EFFECTS.csv' if study=='carrier' else 'ALL_UNIQUE_FORWARD_SCORES.csv')
        groups=collections.defaultdict(list)
        for row in scores:
            groups[row['stage'],row['condition']].append(row)
            if study=='mediation':
                margin=float(row['logp_yes'])-float(row['logp_no'])
                if not math.isclose(margin,float(row['margin']),rel_tol=0,abs_tol=1e-12):
                    raise ValueError('Recorded log-odds difference does not match raw log probabilities')
        accounting=[]
        for (stage,condition),rows in sorted(groups.items()):
            direct=[r for r in rows if r['kind']=='direct' and r['direct_preserved']!='']
            joint=[r for r in rows if r['kind']!='direct']
            accounting.append({'stage':stage,'condition':condition,'joint_rows':len(joint),
                'donor_answers':sum(r.get('hard_transfer')=='True' for r in joint),
                'direct_checks':len(direct),'direct_preserved':sum(frozen.direct_preservation(r) for r in direct)})
        record={'score_rows':len(scores),'condition_accounting':accounting,
                'adverse_outcomes':saved('ADVERSE_AND_OMISSIONS.json')}
        if study=='carrier':record['recorded_condition_details']=saved('ALL_CONDITION_SUMMARIES.json')
        else:
            refs={}
            for row in scores:
                if row['stage']=='baseline':refs[row['condition'],row['recipient'],row['query_id']]=row
            measured={(r['stage'],r['condition'],r['recipient'],r['query_id']):r for r in scores}
            readers=saved('READER_UNION_EFFECTS.json')
            for row in readers:
                recipient=str(row['recipient']);question=row['query_id']
                baseline=refs['NATIVE',recipient,question];carrier=refs['CARRIER',recipient,question]
                base=float(baseline['logp_yes'])-float(baseline['logp_no'])
                full=float(carrier['logp_yes'])-float(carrier['logp_no'])
                fractions={}
                for arm in ('block','inject'):
                    condition=row['set']+'_'+arm.upper()
                    key=('readers',condition,recipient,question)
                    if key not in measured and row['set']=='TOP1':
                        block,head=row['sites'][0]
                        key=('readers',f'L{block}Q{head:02d}_'+arm.upper(),recipient,question)
                    retained=measured[key]
                    margin=float(retained['logp_yes'])-float(retained['logp_no'])
                    fraction=(margin-base)/(full-base)
                    fractions[arm]=1-fraction if arm=='block' else fraction
                    if not math.isclose(fractions[arm],row[arm+'_fraction'],rel_tol=0,abs_tol=1e-12):
                        raise ValueError('Reader fraction differs from recorded log-odds accounting')
                if min(fractions.values())!=row['mediation_score']:
                    if not math.isclose(min(fractions.values()),row['mediation_score'],rel_tol=0,abs_tol=1e-12):
                        raise ValueError('Reader minimum differs')
            record.update(reader_unions=readers,selection=saved('NUMERICAL_AUDIT.json'),
                          chain_effects=saved('CHAIN_EFFECTS.json'),writer_effects=saved('WRITER_COMPONENT_MEDIATION.json'),
                          log_odds_rows_recomputed=len(scores),reader_fraction_rows_recomputed=len(readers))
        output['studies'][study]=record
    return output
