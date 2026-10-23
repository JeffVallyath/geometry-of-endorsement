"""CPU contract tests. Artificial responses are temporary test fixtures, never model evidence."""
from pathlib import Path
import ast, collections, contextlib, gzip, hashlib, importlib.util, io, json, math, sys, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'implementation'))
from prepare_data import load,gzrows,canonical,digest,LEGACY,HISTORIES,ALL
from execution import build_store,execute
from ppr_requests import record_view,score_requests
from ranker_contract import choose,key
from codec import decode,norm
from metrics import paired_interval
from evaluate import evaluate,responses
from validate_controls import validate
from supplement import supplement

class FrozenInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=load(LEGACY/'prepared/RUNTIME_CASES.json');cls.case={c['case_id']:c for c in cls.cases}
        cls.rows=gzrows(ROOT/'prepared/LOGICAL_ROWS.jsonl.gz')
        cls.bank=gzrows(ROOT/'prepared/REQUESTS.jsonl.gz');cls.b={r['request_id']:r for r in cls.bank}
        cls.ev={(r['case_id'],r['query_id']):r for r in load(ROOT/'prepared/EVALUATOR_ONLY.json')}
    def test_01_full_population(self):
        self.assertEqual(len(self.cases),128)
        self.assertEqual(sum(len(c['queries']) for c in self.cases),901)
        self.assertEqual(len(self.rows),6942)
        self.assertEqual(sum(r['phase']=='post' for r in self.rows),5406)
        self.assertEqual(sum(r['phase']=='pre' for r in self.rows),1536)
    def test_02_request_inventory(self):
        self.assertEqual(len(self.bank),7315)
        self.assertEqual(collections.Counter(r['kind'] for r in self.bank),{'score':3977,'generate':3338})
        for r in self.bank:
            self.assertEqual(r['request_id'],digest({k:v for k,v in r.items() if k!='request_id'}))
    def test_03_no_evaluator_fields_in_neural_scores(self):
        for r in self.bank:
            self.assertEqual(set(r),{'request_id','kind','messages'} if r['kind']=='score' else {'request_id','kind','messages','max_new_tokens','do_sample'})
            if r['kind']=='score':
                payload=json.loads(r['messages'][1]['content'])
                self.assertEqual(set(payload),{'question','relation_registry','current_records','candidate_program'})
                self.assertEqual(set(payload['candidate_program']),{'start_id','entity','path','snapshot'})
    def test_04_every_reference_resolves(self):
        for r in self.rows:
            self.assertEqual(len(r['candidates']),len(r['score_request_ids']))
            self.assertTrue(all(i in self.b for i in r['score_request_ids']))
            self.assertEqual(self.b[r['CURRENT_TEXT']]['kind'],'generate')
            self.assertEqual(self.b[r['CURRENT_JSON']]['kind'],'generate')
    def test_05_independent_current_states(self):
        for c in self.cases:
            stores=[build_store(c,h) for h in ALL]
            self.assertEqual(len({id(s) for s in stores}),6)
            states=[canonical(sorted([s,p,o] for (s,p),o in x.view().items())) for x in stores]
            self.assertEqual(len(set(states)),1)
    def test_06_current_payload_and_candidates_equal(self):
        groups=collections.defaultdict(list)
        for r in self.rows:
            if r['phase']=='post':groups[r['case_id'],r['query_id']].append(r)
        self.assertEqual(len(groups),901)
        for rs in groups.values():
            self.assertEqual(len(rs),6)
            self.assertEqual(len({canonical([r['candidates'],r['score_request_ids'],r['CURRENT_TEXT'],r['CURRENT_JSON']]) for r in rs}),1)
    def test_07_history_is_retained_and_readable(self):
        c=self.case[6]
        for h in HISTORIES:
            s=build_store(c,h)
            before={(a,b):o for a,b,o in c['histories'][h]['initial']}
            self.assertEqual(s.view(0),before)
            changed=before.copy();changed.update({(a,b):o for a,b,o in c['histories'][h]['edits']})
            self.assertEqual(s.view(),changed)
            old=s.view(0);self.assertEqual(s.view(0),old)
            self.assertEqual(s.view(),changed)
    def test_08_every_reference_program_executes_cpu_only(self):
        # Gold programs ONLY exercise the software backend in this unit test.
        for r in self.rows:
            e=self.ev[r['case_id'],r['query_id']];g=e['post'] if r['phase']=='post' else e['pre'][r['history']]
            found=[p for p in r['candidates'] if p['start_id']==g['start'] and p['path']==g['path']]
            self.assertEqual(len(found),1)
            self.assertEqual(execute(self.case[r['case_id']],r['history'],r['phase'],found[0])['entity'],g['entity'])
    def test_09_known_nine_fixed(self):
        ws=load(ROOT/'prepared/KNOWN_32B_WITNESSES.json')
        self.assertEqual(len(ws),9)
        self.assertEqual(len({w['case_id'] for w in ws}),5)
        self.assertEqual(sum(w['K_required'] for w in ws),7)
    def test_10_all_nine_in_uncached_repeats(self):
        c=load(ROOT/'prepared/CONTROLS.json')
        self.assertEqual(len(c['repeat_cases']),12)
        self.assertEqual(len(c['repeat_rows']),492)
        for w in load(ROOT/'prepared/KNOWN_32B_WITNESSES.json'):
            for h in ALL:self.assertIn(f"{w['case_id']}/{h}/post/{w['query_id']}",c['repeat_rows'])
    def test_11_return_schedule(self):
        c=load(ROOT/'prepared/CONTROLS.json')
        self.assertEqual(len(c['return_programs']),16)
        for t in c['return_programs']:
            self.assertEqual(t['sequence'][0],t['sequence'][2]);self.assertIn('/pre/',t['sequence'][1])
    def test_12_training_counts_and_orders(self):
        tr=gzrows(ROOT/'training/training_pairs.jsonl.gz')
        self.assertEqual(len(tr),12288);self.assertEqual(len({r['pair_id'] for r in tr}),7507)
        for seed in (0,1):
            order=load(ROOT/f'training/seed{seed}_ORDER.json')['sample_order']
            self.assertEqual(len(order),12288);self.assertEqual(set(order),{r['sample_id'] for r in tr})
    def test_13_training_overlap_disclosed(self):
        tr=gzrows(ROOT/'training/training_pairs.jsonl.gz')
        tq={norm(json.loads(r['request']['messages'][1]['content'])['question']) for r in tr}
        counts=collections.Counter(e['branch'] for e in self.ev.values() if norm(e['question']) in tq)
        self.assertEqual(dict(counts),{'F':21,'Fdecoy':12})
    def test_14_baseline_complete89(self):
        rows=gzrows(ROOT/'prepared/NATIVE32_COMMON_CODEC.jsonl.gz')
        self.assertEqual(len(rows),5406)
        by={(r['case_id'],r['history'],r['query_id']):r for r in rows}
        result=sum(all(by[c['case_id'],h,q['query_id']]['correct'] for h in HISTORIES for q in c['queries']) for c in self.cases)
        self.assertEqual(result,89)
    def test_15_scope_full32_not7(self):
        c=load(ROOT/'REPAIR32_CONFIG.json')
        self.assertEqual(c['model']['repo'],'Qwen/Qwen2.5-32B-Instruct')
        self.assertEqual(c['training']['steps_each'],768);self.assertEqual(c['training']['seeds'],[0,1])
        self.assertFalse(c['model']['quantization']);self.assertFalse(c['model']['offload'])
    def test_16_sources_unmodified(self):
        report=load(ROOT/'audit/COPIED_PAYLOADS.json')
        # Support the actual manifest shape, without guessing byte identity from names.
        items=report['files'] if isinstance(report,dict) and 'files' in report else report
        if isinstance(items,dict):items=[dict(path=k,sha256=v) for k,v in items.items()]
        for x in items:
            self.assertEqual(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest(),x['sha256'])

class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.a={'start_id':'a','entity':'A','path':['r'],'endpoint':'b','edges':[['a','r','b']]}
        self.b={'start_id':'a','entity':'A','path':['s'],'endpoint':'c','edges':[['a','s','c']]}
    def test_17_refusal_boundary(self):
        self.assertIsNone(choose([self.a,self.b],[-1,0]));self.assertEqual(choose([self.a,self.b],[.1,-1]),self.a)
    def test_18_order_independent_tie(self):
        self.assertEqual(choose([self.a,self.b],[1,1]),choose([self.b,self.a],[1,1]))
    def test_19_missing_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):choose([self.a],[])
        with self.assertRaises(ValueError):choose([self.a],[float('nan')])
    def test_20_canonical_codec(self):
        self.assertEqual(decode('Answer: "Seoul".',{'Seoul':'Q8684'},{'Q8684'})['entity'],'Q8684')
        self.assertEqual(decode('Seoul\nExplanation',{'Seoul':'Q8684'},{'Q8684'})['status'],'MULTILINE_OUTPUT')
    def test_21_refusal_is_not_correct(self):
        self.assertEqual(decode('UNSUPPORTED',{},set())['status'],'REFUSAL')
    def test_22_native_format_flags_remain_invalid(self):
        self.assertIsNone(decode('Seoul',{'Seoul':'Q8684'},{'Q8684'},truncated=True)['entity'])
        self.assertIsNone(decode('Seoul',{'Seoul':'Q8684'},{'Q8684'},invalid_control_tokens=True)['entity'])
    def test_23_bootstrap_pairing(self):
        x=paired_interval([1,0,1],[1,0,1],draws=50)
        self.assertEqual(x['difference'],0);self.assertEqual(x['interval'],[0.,0.])
    def test_24_missing_output_not_failed_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('')
            with self.assertRaises(ValueError):responses(p,{'missing'})
    def test_25_worker_and_modules_syntax(self):
        for p in (ROOT/'implementation').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_26_worker_does_not_import_evaluator(self):
        tree=ast.parse((ROOT/'implementation/model_worker.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s and s.split('.')[0] in ('evaluate','prepare_data','execution','ppr_requests','codec') for s in imports))
    def test_27_launcher_under4000(self):
        self.assertLess(len((ROOT/'REPAIR32_LAUNCHER.txt').read_text()),4000)

class FullTemporaryFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.work=Path(cls.temp.name)
        cls.logical=gzrows(ROOT/'prepared/LOGICAL_ROWS.jsonl.gz')
        ev={(r['case_id'],r['query_id']):r for r in load(ROOT/'prepared/EVALUATOR_ONLY.json')}
        cases={r['case_id']:r for r in load(LEGACY/'prepared/RUNTIME_CASES.json')}
        sb={};gb={}
        for r in cls.logical:
            e=ev[r['case_id'],r['query_id']];g=e['post'] if r['phase']=='post' else e['pre'][r['history']]
            for p,i in zip(r['candidates'],r['score_request_ids']):
                m=1. if (p['start_id'],p['path'])==(g['start'],g['path']) else -1.
                rec={'request_id':i,'margin':m,'yes_logit':m,'no_logit':0.,'input_token_ids':[1,2],'full_logits_sha256':'TEMPORARY_UNIT_TEST_FIXTURE'}
                if i in sb:assert sb[i]==rec
                sb[i]=rec
            for a in ('CURRENT_TEXT','CURRENT_JSON'):
                i=r[a];rec={'request_id':i,'text':cases[r['case_id']]['labels'][g['entity']],'truncated':False,'invalid_control_tokens':False}
                if i in gb:assert gb[i]==rec
                gb[i]=rec
        cls.scores=sb
        def save(name,rows):
            p=cls.work/name
            with p.open('w',encoding='utf-8') as f:
                for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            return p
        cls.save=staticmethod(save)
        cls.base=save('base.jsonl',list(sb.values())+list(gb.values()));cls.seed=save('seed.jsonl',sb.values())
        cls.good=evaluate(cls.base,cls.seed,cls.seed,cls.work/'good')
        cls.extra=supplement(cls.work/'good')
        bad=save('refusal.jsonl',[dict(r,margin=-1.,yes_logit=-1.,no_logit=0.) for r in sb.values()])
        cls.refusal=evaluate(cls.base,bad,bad,cls.work/'refusal_out')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_28_full_valid_fixture_gives128_and9(self):
        for a in ('STATE_0','STATE_1'):
            self.assertEqual(self.good['endpoints'][a]['complete_faithful_subjects'],128)
            self.assertEqual(self.good['known_9_repairs'][a],9)
    def test_29_refusal_invariant_but_no_repair(self):
        for a in ('STATE_0','STATE_1'):
            self.assertEqual(self.refusal['endpoints'][a]['complete_subjects'],0)
            self.assertEqual(self.refusal['endpoints'][a]['current_output_differences_from_logical_replay'],0)
            self.assertEqual(self.refusal['known_9_repairs'][a],0)
    def test_30_full_outcome_inventory(self):
        self.assertEqual(len(gzrows(self.work/'good/ALL_OUTCOMES.jsonl.gz')),40116)
        self.assertEqual(len(gzrows(self.work/'good/FIXES_HARMS.jsonl.gz')),7208)
    def test_31_uncached_validator_requires_actual_schedule(self):
        repeat=self.save('repeats.jsonl',[self.scores[i] for i in load(ROOT/'prepared/REPEAT_REQUEST_IDS.json')])
        ret=self.save('returns.jsonl',[self.scores[i] for i in load(ROOT/'prepared/RETURN_REQUEST_IDS.json')])
        out=validate(self.seed,repeat,ret)
        self.assertTrue(out['selection_invariance_pass']);self.assertEqual(out['repeat']['logical_reads'],492)
        with repeat.open('a') as f:f.write(json.dumps(next(iter(self.scores.values())))+'\n')
        with self.assertRaises(ValueError):validate(self.seed,repeat,ret)
    def test_33_supplement_complete(self):
        self.assertEqual(self.extra['pre_post_target_transitions'],7680)
        self.assertEqual(self.extra['subject_fix_harm_rows'],1024)
        self.assertEqual(self.extra['native_pre'],'NOT_MEASURED')
        self.assertEqual(self.extra['first_line_sensitivity']['NATIVE32_SAVED']['correct']['complete_subjects'],89)
    def test_32_no_false_comparative_gain_vs_perfect_current(self):
        self.assertFalse(self.good['comparative_gain'])
        self.assertEqual(self.good['primary_paired_contrasts']['CURRENT_TEXT']['difference'],0.)

if __name__=='__main__':unittest.main()
