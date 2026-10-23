import copy
import unittest
from baseline_core import parse_program,sql_chain,parse_chain,gmello,mello,drive_coroutines,input_text
import json

def fixture():
    reg={'P1':['parent'],'P2':['residence']}
    names={'Q1':'Ada','Q2':'Bryn','Q3':'City'}
    return {'question':'Where does the parent of Ada reside?','relation_registry':reg,
            'current_records':[{'subject':{'id':s,'labels':[names[s]]},'relation':{'id':p,'labels':reg[p]},'object':{'id':o,'labels':[names[o]]}} for s,p,o in [('Q1','P1','Q2'),('Q2','P2','Q3')]]}
def result(text):return {'text':text,'cap_hit':False}
def retrieve(q,facts,k):return [{'index':i,'score':1.0-i} for i in range(k)]

class TestBaselines(unittest.TestCase):
    def test_input_field_order_survives_canonical_storage(self):
        p=fixture();restored=json.loads(json.dumps(p,sort_keys=True))
        self.assertEqual(input_text(p),input_text(restored))
        self.assertEqual(list(json.loads(input_text(p))),['question','relation_registry','current_records'])
    def test_strict_program(self):
        p=fixture()
        self.assertEqual(parse_program('{"start_id":"Q1","path":["P1","P2"]}',p)[1],'VALID')
        for t in ['{"start_id":"Q9","path":["P1"]}','{"start_id":"Q1","path":[]}','{"start_id":"Q1","path":["P99"]}','```json\n{}\n```','{"answer":"City"}']:
            self.assertNotEqual(parse_program(t,p)[1],'VALID')
    def test_graph_priority_even_wrong_program(self):
        co=gmello(fixture(),[],retrieve);next(co)
        with self.assertRaises(StopIteration) as done:co.send(result('Relation Chain: Ada->parent->?x'))
        self.assertEqual(done.exception.value['entity'],'Q2')
        self.assertIsNone(done.exception.value['trace']['fallback'])
    def test_graph_fallback(self):
        co=gmello(fixture(),[],retrieve);next(co)
        request=co.send(result('Relation Chain: Ada->residence->?x'))
        self.assertEqual(request['max_new_tokens'],64)
        with self.assertRaises(StopIteration) as done:co.send(result('City'))
        self.assertEqual(done.exception.value['route'],'fallback')
    def test_invalid_chain_fallback(self):
        co=gmello(fixture(),[],retrieve);next(co)
        self.assertEqual(co.send(result('No chain'))['max_new_tokens'],64)
    def test_normalization(self):
        p,trace=parse_chain('Ada->parental relation->?x',fixture(),retrieve)
        self.assertEqual(p,{'start_id':'Q1','path':['P1']})
        self.assertEqual(trace['normalization'][0]['method'],'contriever-nearest')
    def test_sql_state_changes(self):
        p=fixture();original=copy.deepcopy(p)
        self.assertEqual(sql_chain(p,{'start_id':'Q1','path':['P1','P2']})['entity'],'Q3')
        p['current_records'][1]['object']={'id':'Q4','labels':['Elsewhere']}
        self.assertEqual(sql_chain(p,{'start_id':'Q1','path':['P1','P2']})['entity'],'Q4')
        self.assertEqual(sql_chain(original,{'start_id':'Q1','path':['P1','P2']})['entity'],'Q3')
    def test_mello_four_retrievals_and_final(self):
        co=mello(fixture(),'TRAIN DEMO',retrieve)
        req=next(co)
        for i in range(4):
            req=co.send(result('Subquestion: Who is Ada?\nGenerated answer: Ada'))
        self.assertEqual(req['max_new_tokens'],64)
        self.assertEqual(req['messages'][1]['content'].count('Retrieved fact:'),4)
        with self.assertRaises(StopIteration) as done:co.send(result('Final answer: City'))
        self.assertEqual(done.exception.value['answer_text'],'City')
        self.assertEqual(len(done.exception.value['trace']),5)
    def test_no_gold_substring_answer_search(self):
        co=mello(fixture(),'',retrieve);next(co)
        with self.assertRaises(StopIteration) as done:co.send(result('Subquestion: City\nGenerated answer: City\nRandom: City'))
        self.assertEqual(done.exception.value['status'],'INVALID_MELLO_FORMAT')
    def test_coroutine_batch(self):
        out=drive_coroutines({'a':mello(fixture(),'',retrieve),'b':mello(fixture(),'',retrieve)},lambda requests:[result('Final answer: City') for _ in requests])
        self.assertEqual(set(out),{'a','b'})
        self.assertTrue(all(x['answer_text']=='City' for x in out.values()))

if __name__=='__main__':unittest.main()
