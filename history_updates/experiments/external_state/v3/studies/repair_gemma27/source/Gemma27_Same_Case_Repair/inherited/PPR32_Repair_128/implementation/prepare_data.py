"""Rebuild the unchanged PPR interface; no neural inference or test-label exposure."""
from pathlib import Path
import argparse, collections, csv, gzip, hashlib, json, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'legacy_ppr'
sys.path[:0]=[str(LEGACY/'implementation'),str(LEGACY/'vendor')]
from prepare import prepare as legacy_prepare, sql_check
from ppr_requests import record_view, score_requests, answer_request
from execution import build_store, execute
from codec import decode, norm
from ranker_contract import key as candidate_key
HISTORIES=('H0','H1','S0','S1'); ALL=(*HISTORIES,'C','C+')

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def gzrows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def write_rows(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f,gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:
        for r in rows:z.write((canonical(r)+'\n').encode())

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        census=legacy_prepare(Path(t))
        for filename in ('RUNTIME_CASES.json','EVALUATOR_ONLY.json','CENSUS.json'):
            assert load(Path(t)/filename)==load(LEGACY/'prepared'/filename),('LEGACY_REBUILD_MISMATCH',filename)
    cases=load(LEGACY/'prepared/RUNTIME_CASES.json');ev=load(LEGACY/'prepared/EVALUATOR_ONLY.json')
    cohort=json.loads(gzip.decompress((LEGACY/'source/cohort.json.gz').read_bytes()))
    originals={c['case_id']:c for c in cohort['cases']};lookup={c['case_id']:c for c in cases}
    bank={};logical=[];state_receipts=[];versions=0
    def add(req,kind):
        r={'kind':kind,**req};h=digest(r)
        bank.setdefault(h,{'request_id':h,**r});return h
    for case in cases:
        cid=case['case_id'];visible=[]
        for history in ALL:
            h=case['histories'][history]
            assert h['source_text']==originals[cid]['contexts'][history]
            store=build_store(case,history);before=store.view(0)
            edges,payload=record_view(case,history,'post')
            assert edges==sql_check(h['initial'],h['edits'])
            assert sorted([s,p,o] for (s,p),o in store.view().items())==edges
            assert before=={(s,p):o for s,p,o in h['initial']}
            visible.append(digest(payload));versions+=1
            # Two independent source histories must actually build their own stores.
            state_receipts.append({'case_id':cid,'history':history,'source_sha256':digest(h['source_text']),
                'current_payload_sha256':digest(payload),'initial_state_sha256':digest(sorted([s,p,o] for (s,p),o in before.items())),
                'sql_equal':True,'initial_retained':True})
            for phase in ('post','pre'):
                if phase=='pre' and history not in HISTORIES:continue
                for q in case['queries']:
                    if phase=='pre' and not q['query_id'].startswith('J'):continue
                    paths,requests=score_requests(case,history,phase,q['question'])
                    ids=[add(r,'score') for r in requests]
                    for r in requests:
                        body=json.loads(r['messages'][1]['content'])
                        assert set(body)=={'question','relation_registry','current_records','candidate_program'}
                        assert set(body['candidate_program'])=={'start_id','entity','path','snapshot'}
                    row={'case_id':cid,'history':history,'phase':phase,'query_id':q['query_id'],
                         'question':q['question'],'candidates':paths,'score_request_ids':ids,
                         'CURRENT_TEXT':add(answer_request(case,history,phase,q['question'],'CURRENT_TEXT'),'generate'),
                         'CURRENT_JSON':add(answer_request(case,history,phase,q['question'],'CURRENT_JSON'),'generate')}
                    row['logical_id']=f"{cid}/{history}/{phase}/{q['query_id']}";logical.append(row)
        assert len(set(visible))==1,('CURRENT_INPUT_DIVERGENCE',cid)
    assert len(logical)==6942 and versions==768
    groups=collections.defaultdict(list)
    for r in logical:
        if r['phase']=='post':groups[r['case_id'],r['query_id']].append(r)
    for k,rs in groups.items():
        assert len(rs)==6
        assert len({canonical([r['score_request_ids'],r['candidates'],r['CURRENT_TEXT'],r['CURRENT_JSON']]) for r in rs})==1,k
    # Training data are the ORIGINAL frozen QSA1 draws, not data mined from PPR.
    train=gzrows(ROOT/'training/training_pairs.jsonl.gz');assert len(train)==12288
    tq={norm(json.loads(r['request']['messages'][1]['content'])['question']) for r in train}
    exposure=collections.Counter()
    for e in ev:
        if norm(e['question']) in tq:exposure[e['branch']]+=1
    assert exposure['J']==0 and exposure['K']==0 and sum(exposure.values())==33,exposure
    assert len({r['pair_id'] for r in train})==7507
    samples={r['sample_id'] for r in train};assert len(samples)==12288
    for seed in (0,1):
        order=load(ROOT/f'training/seed{seed}_ORDER.json')['sample_order'];assert len(order)==12288 and set(order)==samples
    # Crosswalk every saved native input and independently recode outcomes with the common PPR codec.
    raw=gzrows(ROOT/'baseline/native32_raw.jsonl.gz');nreq=gzrows(ROOT/'baseline/native32_requests.jsonl.gz')
    assert len(raw)==len(nreq)==5406
    rawmap={(r['case_id'],r['arm'],r['query_id']):r for r in raw};native=[]
    assert len(rawmap)==5406
    qs={(r['case_id'],r['query_id']):r for r in ev}
    for req in nreq:
        cid,h,qid=req['case_id'],req['arm'],req['query_id'];case=lookup[cid]
        q=next(q['question'] for q in case['queries'] if q['query_id']==qid)
        assert req['messages'][1]['content']==case['histories'][h]['source_text']+'\n\nQuestion: '+q+'\nAnswer:'
        r=rawmap[cid,h,qid]
        aliases={norm(v):k for k,v in case['labels'].items()}
        parsed=decode(r['text'],aliases,set(case['labels']),truncated=r['truncated'],invalid_control_tokens=not r['control_tokens_valid'])
        native.append({'case_id':cid,'history':h,'query_id':qid,'phase':'post','entity':parsed['entity'],
                       'status':parsed['status'],'correct':parsed['entity']==qs[cid,qid]['post']['entity'],
                       'raw_text':r['text'],'original_request_sha256':r['request_sha256']})
    with (ROOT/'baseline/SCALE_TARGETS.csv').open() as f:
        targets=list(csv.DictReader(f))
    # Inspect exact names rather than guessing which parser/model spelling was saved.
    print('target columns/model/parser', sorted({(r['model'],r['parser']) for r in targets}))
    ws=[{'case_id':int(r['case_id']),'query_id':r['query_id'],'K_required':r['long_chain_qualified'] in ('True','true','1')}
        for r in targets if r['model']=='large' and r['parser']=='primary' and r['qualified'] in ('True','true','1')]
    assert len(ws)==9,ws
    # CPU diagnostic executes supplied reference programs; NEVER a repair-method score.
    goldchecks=0
    eidx={(r['case_id'],r['query_id']):r for r in ev}
    for r in logical:
        e=eidx[r['case_id'],r['query_id']];g=e['post'] if r['phase']=='post' else e['pre'][r['history']]
        p=next(p for p in r['candidates'] if p['start_id']==g['start'] and p['path']==g['path'])
        assert execute(lookup[r['case_id']],r['history'],r['phase'],p)['entity']==g['entity'];goldchecks+=1
    nidx={(r['case_id'],r['history'],r['query_id']):r for r in native}
    complete=sum(all(nidx[c['case_id'],h,q['query_id']]['correct'] for h in HISTORIES for q in c['queries']) for c in cases)
    repeatcases=[172,755,1662,2061,2113,133,583,642,722,784,1617,1751]
    # All questions and all six contexts for these fixed twelve cases, without cache.
    repeatrows=[r['logical_id'] for r in logical if r['phase']=='post' and r['case_id'] in repeatcases]
    allj=[r for r in logical if r['phase']=='pre' and r['history']=='H0']
    selected=sorted(allj,key=lambda r:digest(r['logical_id']))[:16]
    returns=[{'case_id':r['case_id'],'history':'H0','query_id':r['query_id'],
              'sequence':[f"{r['case_id']}/H0/post/{r['query_id']}",r['logical_id'],f"{r['case_id']}/H0/post/{r['query_id']}"]} for r in selected]
    kinds=collections.Counter(r['kind'] for r in bank.values())
    report={'subjects':128,'post_questions':901,'targets':384,'controls':517,'post_logical_reads_per_endpoint':5406,
            'pre_logical_reads_per_endpoint':1536,'logical_reads_per_endpoint':6942,'new_endpoints':5,
            'new_logical_reads':34710,'unique_request_bank':len(bank),'unique_requests_by_kind':dict(kinds),
            'post_candidate_references_per_scorer':2033,'pre_candidate_references_per_scorer':3888,
            'independently_built_source_conditions':768,'six_way_identical_current_inputs':128,
            'equal_current_query_groups':901,'gold_executor_diagnostic_checks':goldchecks,
            'candidate_max':max(len(r['candidates']) for r in logical),'training_draws':12288,
            'training_distinct_pairs':7507,'exact_training_question_overlap':dict(exposure),
            'native_complete_subjects_common_whole_response_codec':complete,
            'native_historical_first_line_complete_subjects':89,'frozen_native_witnesses':9,
            'repeat_cases':repeatcases,'repeat_logical_reads_per_scorer':len(repeatrows),
            'return_programs_per_scorer':16,'neural_inference_performed':False}
    write_rows(out/'REQUESTS.jsonl.gz',[bank[k] for k in sorted(bank)])
    write_rows(out/'LOGICAL_ROWS.jsonl.gz',logical)
    write_rows(out/'NATIVE32_COMMON_CODEC.jsonl.gz',native)
    dump(out/'EVALUATOR_ONLY.json',ev);dump(out/'STATE_RECEIPTS.json',state_receipts)
    dump(out/'KNOWN_32B_WITNESSES.json',ws)
    dump(out/'CONTROLS.json',{'repeat_cases':repeatcases,'repeat_rows':repeatrows,'return_programs':returns})
    bylogical={r['logical_id']:r for r in logical}
    dump(out/'REPEAT_REQUEST_IDS.json',[i for lid in repeatrows for i in bylogical[lid]['score_request_ids']])
    dump(out/'RETURN_REQUEST_IDS.json',[i for x in returns for lid in x['sequence'] for i in bylogical[lid]['score_request_ids']])
    canaries=[]
    for r in train[:16]:
        req={'kind':'score',**r['request']};canaries.append({'request_id':digest(req),**req})
    write_rows(out/'TRAIN_CANARY_REQUESTS.jsonl.gz',canaries)
    dump(out/'PREPARATION.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.out),indent=2))
