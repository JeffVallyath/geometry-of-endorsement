"""CPU-only source transport + evaluator separation for PPR1.
No model is run, and candidate generation never receives expected paths/answers.
"""
from __future__ import annotations
from pathlib import Path
import argparse,gzip,hashlib,json,sqlite3,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from transport import parse_context,materialize,current_text,TransportError
from state_grounding import current_payload,enumerate_paths
from ranker_contract import request,key
from public_graph import VersionStore

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def sha(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def follow(state,start,path):
    s={(a,b):c for a,b,c in state};edges=[]
    for p in path:
        end=s[start,p];edges.append([start,p,end]);start=end
    return start,edges

def candidates(edges,labels,q):
    # Identical prefix set/all-literal expansion used by CSG1 graph_options.
    paths=enumerate_paths(edges,[],labels,q,4,64)
    expanded=[]
    for p in paths:
        for lit in labels[p['start_id']]:
            if lit in q:expanded.append(dict(p,entity=lit))
    if len(expanded)>64:raise TransportError('CANDIDATE_BUDGET_EXCEEDED')
    return sorted(expanded,key=key)

def sql_check(initial,edits):
    c=sqlite3.connect(':memory:')
    c.execute('CREATE TABLE t(s TEXT,p TEXT,o TEXT,PRIMARY KEY(s,p))')
    for s,p,o in initial:
        prev=c.execute('SELECT o FROM t WHERE s=? AND p=?',(s,p)).fetchone()
        if prev is not None and prev[0]!=o:raise TransportError('SQL_CONFLICT')
        c.execute('INSERT OR IGNORE INTO t VALUES (?,?,?)',(s,p,o))
    c.commit()
    with c:
        for t in edits:c.execute('INSERT INTO t VALUES (?,?,?) ON CONFLICT(s,p) DO UPDATE SET o=excluded.o',t)
    out=[list(t) for t in c.execute('SELECT s,p,o FROM t ORDER BY s,p')];c.close();return out

def prepare(output):
    cohort=json.loads(gzip.decompress((ROOT/'source/cohort.json.gz').read_bytes()))
    registry=json.loads((ROOT/'mapping/RELATION_REGISTRY.json').read_bytes())
    runtime=[];evals=[];receipts=[];cq=Counter();preq=0;maxcand=0;missing=[];nodeclash=[];byte_equal=0;source_sql=0;c_reconstruct=0
    postscore=0;prescore=0;candidate_counts=[]
    for case in cohort['cases']:
        labels=dict(case['canonical_labels']);labels[case['decoy_id']]=case['decoy_label']
        lablist={k:[v] for k,v in labels.items()}
        arms={a:parse_context(c,labels) for a,c in case['contexts'].items()}
        current=[materialize(d['initial'],d['edits']) for d in arms.values()]
        assert all(s==current[0] for s in current),'CURRENT_DIVERGENCE'
        assert current[0]==materialize(arms['C']['initial'],[])
        # Same global correction sequence in all actual histories and C+.
        assert len({sha(arms[a]['edits']) for a in ['H0','H1','S0','S1','C+']})==1
        assert not arms['C']['edits']
        # Render the computed view in source language; sorted C reference agrees.
        rendered=current_text(current[0],labels)
        assert sorted(rendered.splitlines()[1:])==sorted(case['contexts']['C'].splitlines()[1:]);c_reconstruct+=1
        for a,d in arms.items():
            v=VersionStore(d['initial'],set(registry));v.apply(d['edits'])
            assert sorted([s,p,o] for (s,p),o in v.view().items())==current[0]==sql_check(d['initial'],d['edits']);source_sql+=1
            receipts.extend(dict(case_id=case['case_id'],history=a,**r) for r in d['receipts'])
        payloads=[current_payload(d['initial'],d['edits'],lablist,registry) for d in arms.values()]
        assert len({sha(p) for p in payloads})==1;byte_equal+=1
        for q in case['queries']:
            cq[q['branch']]+=1
            cs=candidates(current[0],lablist,q['question']);maxcand=max(maxcand,len(cs));postscore+=len(cs);candidate_counts.append(len(cs))
            # Expected programs derived only here, after source-only candidates.
            if q['query_id'].startswith('J'):start=case['subject_id'];path=case['relations']
            elif q['query_id']=='K':start=case['current_bridge'];path=case['relations'][1:]
            elif q['query_id']=='Fdecoy':start=case['decoy_id'];path=case['relations'][:1]
            else:
                idx=int(q['query_id'][1:]);start=case['subject_id'] if idx==0 else follow(current[0],case['subject_id'],case['relations'][:idx])[0];path=case['relations'][idx:idx+1]
            ans,_=follow(current[0],start,path);assert ans==q['gold']
            recall=any(p['start_id']==start and p['path']==path for p in cs)
            if not recall:missing.append({'case_id':case['case_id'],'query_id':q['query_id'],'question':q['question']})
            for payload in payloads:
                for candidate in cs:request(q['question'],payload,candidate)
            pre={}
            if q['query_id'].startswith('J'):
                for a in ['H0','H1','S0','S1']:
                    initial=arms[a]['initial'];state=materialize(initial,[])
                    expected,_=follow(state,start,path)
                    opts=candidates(state,lablist,q['question']);prescore+=len(opts);preq+=1;maxcand=max(maxcand,len(opts))
                    assert any(p['start_id']==start and p['path']==path for p in opts)
                    pre[a]={'entity':expected,'start':start,'path':path}
            evals.append({'case_id':case['case_id'],'query_id':q['query_id'],'question':q['question'],'branch':q['branch'],'post':{'entity':q['gold'],'start':start,'path':path},'pre':pre,'candidate_recall':recall})
        runtime.append({'case_id':case['case_id'],'labels':labels,'histories':{a:{k:d[k] for k in ['initial','edits','source_text']} for a,d in arms.items()},'queries':[{'query_id':q['query_id'],'question':q['question']} for q in case['queries']]})
    manifest={'case_ids':cohort['case_ids'],'main_subjects':len(runtime),'queries_by_role':dict(cq),'post_semantic_queries':sum(cq.values()),'target_questions':cq['J'],'pre_target_history_reads_per_arm':preq,'actual_histories':['H0','H1','S0','S1'],'reference_histories':['C','C+'],'parser_templates':32,'full_relation_registry':len(registry),'sql_endpoints_checked':source_sql,'equal_six_way_current_inputs':byte_equal,'computed_C_source_sentence_agreement':c_reconstruct,'post_expected_programs_available':sum(cq.values())-len(missing),'missing_programs':missing,'post_candidate_score_requests_per_scorer_no_replication':postscore,'pre_candidate_score_requests_per_scorer_all_four_histories':prescore,'max_candidates':maxcand,'model_inference':False,'source_transport_privilege':'Exact inverse of the existing MQMH generated statement language; not general raw-language understanding.'}
    dump(output/'RUNTIME_CASES.json',runtime);dump(output/'EVALUATOR_ONLY.json',evals);dump(output/'TRANSPORT_RECEIPTS.json',receipts);dump(output/'CENSUS.json',manifest)
    from schedule_plan import make_schedule
    arms=['HISTORY','REFLECT','CURRENT_TEXT','CURRENT_JSON','STATE_BASE','STATE_0','STATE_1']
    rows,notes=make_schedule(runtime,arms)
    dump(output/'SCHEDULE.json',{'arms':arms,'logical_answer_reads':rows,'reflection_preparations':notes})
    query_keys=[(c['case_id'],q['query_id']) for c in runtime for q in c['queries']]
    pairs=sorted(query_keys,key=lambda k:sha(['uncached',*k]))[:16]
    timings=sorted([c['case_id'] for c in runtime],key=lambda k:sha(['timing',k]))[:8]
    dump(output/'CONTROL_SELECTION.json',{'uncached_pairs':pairs,'timed_subjects':timings,'demo_first':[1617,'J2'],'illustration_sort':'sha256 canonical [illustration,case_id,query_id]'})
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();prepare(args.out)
