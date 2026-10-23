"""Descriptive post-run audit; saved outputs only, with frozen component replay."""
from pathlib import Path
import argparse
import collections
import json
from baseline_core import ENDPOINTS, digest, parse_chain, sql_chain
from evaluate import read, write

HISTORIES={'H0','H1','S0','S1'}

def report_behavior(raw,evaluation):
    raw,evaluation=Path(raw),Path(evaluation)
    scientific=read(evaluation/'SCIENTIFIC_RESULTS.json')
    snapshots={a:collections.Counter() for a in ENDPOINTS}
    agreement=collections.defaultdict(dict)
    gmello_ids=set()
    gmello_logical_routes=collections.defaultdict(collections.Counter)
    with (evaluation/'SCORED_LOGICAL.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line); arm=row['arm']; count=snapshots[arm]
            if row['branch']=='J' and row['history'] in HISTORIES:
                prefix='pre_target' if row['phase']=='pre' else 'current_target'
                count[prefix+'_reads']+=1;count[prefix+'_correct']+=row['correct']
            if row['phase']=='post' and row['history'] in HISTORIES:
                key=(arm,row['case_id'],row['query_id'])
                value={k:row.get(k) for k in ['outcome','answer_text','program','selected']}
                agreement[key][row['history']]=digest(value)
            if arm=='GMELLO_LOCAL':
                gmello_ids.add(row['method_id'])
                route_count=gmello_logical_routes[row['route']]
                route_count['reads']+=1;route_count['correct']+=row['correct']
                route_count[row['phase']+'_reads']+=1;route_count[row['phase']+'_correct']+=row['correct']
    agreement_counts=collections.Counter()
    for (arm,_,_),answers in agreement.items():
        assert set(answers)==HISTORIES
        agreement_counts[arm]+=len(set(answers.values()))==1
    for arm in ENDPOINTS:
        assert snapshots[arm]['pre_target_reads']==snapshots[arm]['current_target_reads']==1536
        assert sum(key[0]==arm for key in agreement)==901
    parse_counts=collections.Counter();graph_counts=collections.Counter();routes=collections.Counter()
    examples={};replayed=set();cap_hits=0
    with (raw/'results/method.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            method=json.loads(line)
            if method['method_id'] not in gmello_ids:continue
            assert method['method_id'] not in replayed
            replayed.add(method['method_id'])
            result=method['result'];payload=method['payload'];trace=result['trace']
            saved_parse=trace['parse'];normalizations=saved_parse.get('normalization',[])
            options=[(key,label) for key,values in sorted(payload['relation_registry'].items()) for label in values]
            def retained_normalization(query,labels,k):
                # Replay the recorded neural-retriever decision, not a fresh embedding/model call.
                assert labels==[label for _,label in options] and k==1
                selected={n['normalized'] for n in normalizations if n['generated']==query}
                assert len(selected)==1
                return [{'index':next(i for i,(key,_) in enumerate(options) if key in selected)}]
            generated=trace['chain_generation']
            program,parsing=parse_chain(generated['text'],payload,retained_normalization)
            assert program==result['program']==trace['program'] and parsing==saved_parse
            graph={'status':'INVALID_QUERY','entity':None} if program is None or generated['cap_hit'] else sql_chain(payload,program)
            assert graph==trace['graph_only']
            expected_route='graph' if graph['entity'] is not None else 'fallback'
            assert result['route']==expected_route
            assert (trace['fallback'] is not None)==(expected_route=='fallback')
            parse_counts[parsing['status']]+=1;graph_counts[graph['status']]+=1;routes[expected_route]+=1
            cap_hits+=bool(generated['cap_hit'])
            label=parsing['status']+'/'+graph['status']
            examples.setdefault(label,{'method_id':method['method_id'],'question':payload['question'],
                'chain_excerpt':generated['text'][:600],'excerpt_may_be_truncated':len(generated['text'])>600,
                'parsing':parsing,'program':program,'graph':graph,'route':expected_route})
    assert replayed==gmello_ids and len(replayed)==1669
    report={'status':'PASS','matched_target_snapshot_correctness':snapshots,
        'current_four_history_output_agreement':dict(agreement_counts),'agreement_denominator_per_endpoint':901,
        'gmello_component_replay':{'distinct_main_requests':len(replayed),'parse_statuses':dict(parse_counts),
            'graph_statuses':dict(graph_counts),'routes':dict(routes),'chain_generation_cap_hits':cap_hits,
            'logical_route_outcomes':dict(gmello_logical_routes),
            'all_saved_parse_graph_and_routing_decisions_replayed':True,
            'neural_generation_or_embeddings_reexecuted':False,'retriever_choices_replayed_from_saved_normalization':True,
            'examples':examples},
        'historical_witness_cohorts':scientific['historical_witness_cohorts'],
        'no_post_outcome_runtime_change':True,'not_a_new_primary_analysis':True}
    write(evaluation/'BEHAVIOR_COMPONENT_AUDIT.json',report)
    text=['# Snapshot, invariance and component appendix','',
        'Pre and current columns below use the same 384 target questions across H0/H1/S0/S1 (1,536 reads each). Each pre input is that history\'s own initial snapshot; current is its independently constructed final snapshot. These are typed snapshot comparisons, not evidence of natural-language time resolution or a causal training ablation.','',
        '| Endpoint | Pre targets correct /1536 | Current targets correct /1536 | Exact output agreement across four current histories /901 |',
        '|---|---:|---:|---:|']
    for arm in ENDPOINTS:
        count=snapshots[arm]
        text.append(f'| {arm} | {count["pre_target_correct"]} | {count["current_target_correct"]} | {agreement_counts[arm]} |')
    text+=['','Agreement compares complete decoded outcomes, final-answer text and programs/selections, not just entity IDs. It is expected for these current-only interfaces and exact-request reuse after independent state construction. Uniform wrong answers/refusals are not leakage or repaired witnesses.','',
        '## GMeLLo component audit','',
        f'The local reimplementation used graph output for {routes["graph"]} of {len(replayed)} distinct main requests and neural fallback for {routes["fallback"]}. These requests are workload units, not independent statistical samples. Fallback dominance substantially limits interpreting its total score as evidence about an effective graph-answering method or the published full-benchmark system.','',
        f'Generated-chain parse statuses: {dict(parse_counts)}. Graph statuses: {dict(graph_counts)}. Chain cap hits: {cap_hits}.','',
        f'Graph-route answers were correct on {gmello_logical_routes["graph"]["correct"]}/{gmello_logical_routes["graph"]["reads"]} logical occurrences; repeated occurrences are not independent samples. Graph priority was retained even for wrong answers, without fallback rescue.','',
        'The paper\'s Appendix A.4 takes the first arrow-delimited segment as the entity and later non-variable segments as relations. The frozen parser uses that heuristic. Consequently, concrete intermediate entities in generated chains can be treated as relations; generated IDs also fail this implementation\'s label/alias-only entity linker. These interface weaknesses are retained, not repaired after outcomes. [Chen et al., Appendix A.4](https://aclanthology.org/2024.findings-emnlp.844.pdf).','',
        'Every saved main-request parse, formal graph result and graph-priority/fallback decision was replayed with the frozen code. Neural relation-normalization choices were held at their saved values; neither embeddings nor model responses were regenerated. This confirms mechanical consistency of the recorded pipeline, not optimal prompts, an author reproduction, or competence of its untrained chain generator. Representative failures and exact method IDs are retained in BEHAVIOR_COMPONENT_AUDIT.json; full outputs remain in raw/results.','',
        '## Preserved witness recovery','',
        '| Endpoint | Original94 preserved repairs | Later32 preserved repairs | Overlap15 preserved repairs |',
        '|---|---:|---:|---:|']
    cohorts=scientific['historical_witness_cohorts']
    for arm in ENDPOINTS:
        values=[cohorts[key]['endpoints'][arm]['repaired'] for key in ['original_94','later_32','overlap_15']]
        text.append(f'| {arm} | {values[0]} | {values[1]} | {values[2]} |')
    text+=['','Preserved repair requires all-history target correctness and no lost previously correct controls/reference guards under the original cohort definitions. These cohorts are not merged, and the overlap is not an additional independent sample. Exact losses are in WITNESS_RECOVERY.jsonl.','']
    (evaluation/'behavior_component_appendix.md').write_text('\n'.join(text),encoding='utf8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,default=Path(__file__).resolve().parent/'evaluation')
    args=parser.parse_args();report=report_behavior(args.raw,args.evaluation)
    component={k:v for k,v in report['gmello_component_replay'].items() if k!='examples'}
    print(json.dumps({'status':report['status'],'gmello_component_replay':component,
                      'current_four_history_output_agreement':report['current_four_history_output_agreement']},indent=2))
