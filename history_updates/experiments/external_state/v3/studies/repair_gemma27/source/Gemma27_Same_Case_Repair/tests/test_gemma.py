"""Software-only regression tests. No pretrained tokenizer or model is simulated as evidence."""
import unittest,sys,json,gzip,copy,tempfile,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'implementation'))
from gemma_interface import gemma_messages,encode_train,label_ids,validate_config,EXPECTED
from gemma_prepare import INHERITED,LEGACY,load,gzrows,canonical,HISTORIES,ALL
from gemma_diagnostics import witness_rule
from gemma_evaluate import responses
from ranker_contract import choose

class FakeTokenizer:
    """Boundary fixture ONLY, not Gemma tokenization."""
    unk_token_id=99;eos_token_id=1;bos_token_id=2
    def encode(self,s,add_special_tokens=False):return {'Yes':[10],'No':[11]}.get(s,[98])
    def decode(self,ids,**kw):return {10:'Yes',11:'No',107:'<end_of_turn>'}.get(ids[0],'?')
    def convert_tokens_to_ids(self,s):return 107 if s=='<end_of_turn>' else 99
    def apply_chat_template(self,msgs,tokenize=True,add_generation_prompt=False):
        if add_generation_prompt:return [2,30,31]
        return [2,30,31,10 if msgs[-1]['content']=='Yes' else 11,107,90]

class InterfaceTests(unittest.TestCase):
    def setUp(self):self.messages=[{'role':'system','content':'Exact instruction.'},{'role':'user','content':'  Exact request\n'}]
    def test01_instruction_fold(self):self.assertEqual(gemma_messages(self.messages),[{'role':'user','content':'Exact instruction.\n\n  Exact request\n'}])
    def test02_input_not_mutated(self):
        old=copy.deepcopy(self.messages);gemma_messages(self.messages);self.assertEqual(old,self.messages)
    def test03_reject_missing_system(self):
        with self.assertRaises(ValueError):gemma_messages(self.messages[1:])
    def test04_reject_extra_turn(self):
        with self.assertRaises(ValueError):gemma_messages(self.messages+[{'role':'assistant','content':'x'}])
    def test05_eot_not_eos(self):
        t=FakeTokenizer();row={'request':{'messages':self.messages},'target':'Yes'};r=encode_train(t,row)
        self.assertEqual(r['labels'],[-100,-100,-100,10,107]);self.assertNotEqual(r['labels'][-1],t.eos_token_id)
    def test06_no_label(self):
        r=encode_train(FakeTokenizer(),{'request':{'messages':self.messages},'target':'No'});self.assertEqual(r['labels'][-2:],[11,107])
    def test07_wrong_label(self):
        with self.assertRaises(ValueError):encode_train(FakeTokenizer(),{'request':{'messages':self.messages},'target':'Maybe'})
    def test08_architecture(self):self.assertTrue(validate_config(EXPECTED))
    def test09_softcap_necessary(self):
        c=dict(EXPECTED,attn_logit_softcapping=None)
        with self.assertRaises(ValueError):validate_config(c)
    def test10_outputcap_necessary(self):
        with self.assertRaises(ValueError):validate_config(dict(EXPECTED,final_logit_softcapping=0))
    def test11_gemma_not_qwen(self):
        with self.assertRaises(ValueError):validate_config(dict(EXPECTED,model_type='qwen2'))
    def test12_rank_parameters(self):self.assertEqual(46*16*(8704+6656+6656+8704),22609920)

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=gzrows(ROOT/'prepared/LOGICAL_ROWS.jsonl.gz');cls.bank={r['request_id']:r for r in gzrows(ROOT/'prepared/REQUESTS.jsonl.gz')}
        cls.old={r['request_id']:r for r in gzrows(INHERITED/'prepared/REQUESTS.jsonl.gz')}
    def test13_total(self):self.assertEqual(len(self.rows),6942)
    def test14_post_native(self):self.assertEqual(sum('NATIVE_GEMMA'in r for r in self.rows),5406)
    def test15_old_requests_preserved(self):self.assertTrue(all(self.bank[k]==v for k,v in self.old.items()))
    def test16_no_gold_neural_input(self):
        for r in self.bank.values():
            self.assertEqual(set(r)-{'max_new_tokens','do_sample'},{'request_id','kind','messages'})
            if r['kind']=='score':self.assertEqual(set(json.loads(r['messages'][1]['content'])),{'question','relation_registry','current_records','candidate_program'})
    def test17_training_exposure(self):self.assertEqual(load(ROOT/'prepared/PREPARATION.json')['exact_training_question_overlap'],{'F':21,'Fdecoy':12})
    def test18_qwen_challenge_only(self):self.assertEqual(len(load(ROOT/'prepared/QWEN9_CHALLENGE.json')),9)
    def test19_whole_schedule(self):self.assertEqual(load(ROOT/'GEMMA_CONFIG.json')['evaluation']['fresh_logical_reads_total'],5*6942+5406)
    def test20_sixway_current(self):
        groups={}
        for r in self.rows:
            if r['phase']=='post':groups.setdefault((r['case_id'],r['query_id']),set()).add(canonical([r['score_request_ids'],r['candidates']]))
        self.assertEqual(len(groups),901);self.assertTrue(all(len(v)==1 for v in groups.values()))
    def test21_missing_outputs_raise(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.jsonl';p.write_text('')
            with self.assertRaises(ValueError):responses(p,{'a'})

class SemanticsTests(unittest.TestCase):
    def test22_witness(self):
        rr=[{'status':'VALID','entity':'new','correct':True},{'status':'VALID','entity':'old','correct':False}]
        self.assertTrue(witness_rule(rr,True,True))
    def test23_invalid_not_witness(self):
        rr=[{'status':'VALID','entity':'new','correct':True},{'status':'REFUSAL','entity':None,'correct':False}]
        self.assertFalse(witness_rule(rr,True,True))
    def test24_guard_failure_not_witness(self):
        rr=[{'status':'VALID','entity':'new','correct':True},{'status':'VALID','entity':'old','correct':False}]
        self.assertFalse(witness_rule(rr,False,True));self.assertFalse(witness_rule(rr,True,False))
    def test25_two_wrong_not_witness(self):
        self.assertFalse(witness_rule([{'status':'VALID','entity':v,'correct':False} for v in ['a','b']],True,True))
    def test26_negative_top_refuses(self):
        c={'start_id':'a','entity':'A','path':['r']};self.assertIsNone(choose([c],[-1.5]))
    def test27_zero_refuses(self):
        c={'start_id':'a','entity':'A','path':['r']};self.assertIsNone(choose([c],[0]))
    def test28_positive_executes_same_rule(self):
        c={'start_id':'a','entity':'A','path':['r']};self.assertEqual(choose([c],[1.5]),c)
if __name__=='__main__':unittest.main()
