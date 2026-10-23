"""Frozen QSA1 scheduling; no evaluator or reference-label imports."""
from pathlib import Path
import datetime as dt, json, time
from csg1_core import ROOT, digest, current_view, graph_options, Store, execute_program
from csg1_runner import publish, controls_plan
from ranker_contract import request, choose

CFG=json.loads((ROOT/'QSA1_CONFIG.json').read_bytes())
CLOCK=json.loads((ROOT/'CLOCK.json').read_bytes())
DEADLINE=dt.datetime.fromisoformat(CLOCK['science_deadline_utc'])

def check_time():
    if dt.datetime.now(dt.timezone.utc)>=DEADLINE: raise TimeoutError('SCIENCE_CUTOFF')

def roots(group='main'):
    return json.loads((ROOT/f'candidate/{group}.json').read_bytes())

def jobs(root,phase,arms):
    result=[]
    for kind,field in [('target','questions')]+([('direct','control_questions')] if phase=='post' else []):
        for i,q in enumerate(root[field]):
            for arm in arms:
                result.append({'kind':kind,'index':i,'question':q,'arm':arm,
                               'condition':'H0' if arm=='HISTORY_PROGRAM' else 'C','primary':True,'phase':phase})
    return result

def state_for(root,phase):
    return root if phase=='post' else dict(root,edits=[])

def rank_unit(root,joblist,scorer,namespace,reconstruction='H0',reverse=False):
    check_time(); start=time.perf_counter()
    ctx=current_view(root,reconstruction)
    store=Store(root['initial']); store.apply(root['edits'])
    requests=[]; candidate_groups=[]
    for job in joblist:
        options=[v for v in graph_options(root,job['question'],reverse,reconstruction).values() if v is not None]
        candidate_groups.append(options)
        requests.extend(request(job['question'],ctx,c) for c in options)
    prepared=time.perf_counter()
    scores=scorer.run(requests,namespace)
    assert len(scores)==len(requests)
    generated=time.perf_counter(); offset=0; rows=[]
    for job,options in zip(joblist,candidate_groups):
        local=scores[offset:offset+len(options)]; offset+=len(options)
        margins=[s['margin'] for s in local]
        selected=choose(options,margins)
        program=None if selected is None else {'op':'FOLLOW','entity':selected['entity'],'path':selected['path'],'snapshot':'CURRENT'}
        result={'error':'UNSUPPORTED','answer':None} if selected is None else execute_program(root,job['question'],program,store,selected['start_id'])
        if selected is not None and not result['error']:
            assert result['endpoint']==selected['endpoint'] and result['dependencies']==selected['edges']
        rows.append(dict(job,key=root['key'],namespace=namespace,candidate_set=options,scores=local,
                         selected=selected,program=program,result=result,retried=False,
                         requests=[request(job['question'],ctx,c) for c in options]))
    snapshot=sorted([s,p,o] for (s,p),o in store.view().items())
    return {'namespace':namespace,'key':root['key'],'root_candidate_sha256':digest(root),
            'schedule_sha256':digest(joblist),'rows':rows,'snapshots':[root['initial'],snapshot,snapshot],
            'timing':{'total_seconds':time.perf_counter()-start,'transport_state_prompt_enumeration_seconds':prepared-start,
                      'score_seconds':generated-prepared,'parse_execution_codec_seconds':time.perf_counter()-generated,
                      'candidate_calls':len(requests),'prompt_tokens':sum(s['n_prompt'] for s in scores)}}

def unit_path(arm,phase,key): return ROOT/'raw'/arm/phase/(key.replace('/','__')+'.json')

def run_ranker(arm,scorer,group='main',tag=None):
    for root in roots(group):
        for phase in (['post','pre'] if group=='main' else ['post']):
            namespace=f'{arm}/{tag or group}/{phase}/{root["key"]}'
            path=unit_path(arm,tag or phase,root['key']) if group!='main' else unit_path(arm,phase,root['key'])
            if path.exists():
                existing=json.loads(path.read_bytes())
                assert existing['namespace']==namespace
                continue
            publish(path,rank_unit(state_for(root,phase),jobs(root,phase,[arm]),scorer,namespace))
    if group=='main':
        index={r['key']:r for r in roots()}
        plan=controls_plan(roots())['same_current_pairs']
        for kind in ['current','permutation']:
            for i,job in enumerate(plan):
                root=index[job['key']]
                for side in [0,1]:
                    namespace=f'{arm}/controls/{kind}/{i}/{side}'
                    path=ROOT/'raw'/arm/'controls'/f'{kind}_{i}_{side}.json'
                    if path.exists(): continue
                    j=dict(job,arm=arm,phase='post',condition='C',primary=False)
                    publish(path,rank_unit(root,[j],scorer,namespace,
                            reconstruction='H1' if kind=='current' and side else 'H0',reverse=kind=='permutation' and bool(side)))
    publish(ROOT/'raw'/f'{arm}_{tag or group}_COMPLETE.json',{'status':'COMPLETE','arm':arm,'group':group,'tag':tag})
