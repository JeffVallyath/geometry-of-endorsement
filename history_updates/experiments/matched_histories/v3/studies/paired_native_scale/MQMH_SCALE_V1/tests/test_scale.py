from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'implementation'))
import study
from repro import mqmh

class ScaleChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cohort,cls.responses=study.inputs();cls.cases={x['case_id']:x for x in cls.cohort['cases']}
  cls.scored={m:mqmh.score(x['primary'],cls.cases) for m,x in cls.responses.items()}
 def test_cohort_and_inventory(self):
  self.assertEqual(len(self.cases),128);self.assertEqual(sum(len(x['queries']) for x in self.cases.values()),901)
  self.assertEqual(sum(x['hops']>2 for x in self.cases.values())*3,195)
 def test_original_reconstruction(self):
  got,_,_=mqmh.analyze(self.cohort,self.responses)
  expected=json.loads((mqmh.SOURCE/'RESULTS.json').read_text())
  mqmh.assert_equivalent(got,expected)
 def test_independent_all_old_endpoints(self):
  for m,sc in self.scored.items():
   for case in self.cases.values():
    for strict in (False,True):
     study.validate_independent(case,sc,mqmh.endpoints(case,sc,strict=strict),strict)
 def test_runtime_no_evaluator_fields(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'runtime';study.prepare(out)
   rows=[json.loads(l) for l in (out/'runtime_requests.jsonl').read_text().splitlines()]
   self.assertEqual(len(rows),5406)
   for r in rows:
    self.assertEqual(set(r),{'case_id','arm','query_id','messages'})
    case=self.cases[r['case_id']];q=next(q for q in case['queries'] if q['query_id']==r['query_id'])
    self.assertIn(case['contexts'][r['arm']],r['messages'][1]['content']);self.assertIn(q['question'],r['messages'][1]['content'])
 def test_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):study.prepare(Path(d))
 def test_invalid_parser(self):
  c=self.cases[1617];r=dict(text=c['canonical_labels'][c['current_terminal']],truncated=True,control_tokens_valid=True)
  self.assertEqual(mqmh.parse_response(r,c),(None,None))
  r.update(truncated=False,control_tokens_valid=False);self.assertEqual(mqmh.parse_response(r,c),(None,None))
 def test_exact_alias_ambiguity(self):
  c=dict(registry={'BE':'Q1','be':'Q2','Answer: BE':'Q3'},canonical_labels={'Q1':'BE','Q2':'be'},decoy_id='Q4',decoy_label='decoy')
  r=dict(text='BE',truncated=False,control_tokens_valid=True)
  self.assertEqual(mqmh.parse_response(r,c),('Q1','Q1'));self.assertEqual(mqmh.parse_response(dict(r,text='be'),c),('Q2','Q2'))
  self.assertEqual(mqmh.parse_response(dict(r,text='Answer: BE'),c),(None,None))
 def test_missing_read_rejected(self):
  sc=dict(self.scored['qwen']);sc.pop(next(k for k in sc if k[0]==1617))
  with self.assertRaises(ValueError):mqmh.endpoints(self.cases[1617],sc)
 def test_duplicate_rejected(self):
  rows=self.responses['qwen']['primary'];
  with self.assertRaises(ValueError):mqmh.score(rows+[rows[0]],self.cases)
 def test_changed_saved_classification_rejected(self):
  rows=copy.deepcopy(self.responses['qwen']['primary'][:1]);rows[0]['entity_id']='WRONG'
  with self.assertRaises(ValueError):study.scored_rows(rows,self.cases)
 def test_failed_reference_zero_not_filtered(self):
  c=self.cases[1617];sc=copy.deepcopy(self.scored['qwen']);sc[1617,'C','J2']['correct']=False
  e=mqmh.endpoints(c,sc);q=next(q for q in e['targets'] if q['query_id']=='J2')
  self.assertEqual(len(e['targets']),3);self.assertFalse(q['qualified']);self.assertEqual(q['relevance_difference'],0)
 def test_wrong_consensus_not_witness(self):
  c=self.cases[1617];sc=copy.deepcopy(self.scored['qwen'])
  for a in mqmh.ARMS:sc[1617,a,'J2'].update(entity_id=c['old_terminal'],correct=False)
  q=mqmh.endpoints(c,sc)['targets'][2];self.assertFalse(q['qualified'])
 def test_two_different_wrong_answers_not_witness(self):
  c=self.cases[1617];sc=copy.deepcopy(self.scored['qwen'])
  for i,a in enumerate(mqmh.ARMS):sc[1617,a,'J2'].update(entity_id=c['old_terminal'] if i%2 else c['old_bridge'],correct=False)
  self.assertFalse(mqmh.endpoints(c,sc)['targets'][2]['qualified'])
 def test_pair_beyond_h0h1(self):
  c=self.cases[1617];sc=copy.deepcopy(self.scored['qwen'])
  for a in mqmh.ARMS:sc[1617,a,'J2'].update(entity_id=c['current_terminal'],correct=True)
  sc[1617,'S1','J2'].update(entity_id=c['old_terminal'],correct=False)
  self.assertTrue(mqmh.endpoints(c,sc)['targets'][2]['qualified'])
 def test_k_only_secondary(self):
  c=self.cases[1617];sc=copy.deepcopy(self.scored['qwen']);sc[1617,'H0','K']['correct']=False
  q=mqmh.endpoints(c,sc)['targets'][2];self.assertTrue(q['qualified']);self.assertFalse(q['long_chain_qualified'])
 def test_fixed_config(self):
  c=json.loads((ROOT/'SCALE_CONFIG.json').read_text());self.assertEqual(c['inference']['batch_size'],1)
  self.assertFalse(c['inference']['training']);self.assertEqual(c['limits']['total_cloud_usd'],40)
  self.assertEqual(len(c['evaluation']['primary_family']),4);self.assertTrue(set(c['evaluation']['prespecified_repeat_cases'])<=set(self.cases))
 def test_canaries_separate_from_test(self):
  a=json.loads((ROOT/'TECHNICAL_CANARIES.json').read_text());self.assertEqual(len(a),12)
  self.assertTrue(all('case_id' not in r for r in a))
 def test_paired_helper_historical_self_comparison(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'rows.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in self.responses['qwen']['primary']))
   r=study.analyze(p,p,Path(d)/'out')
   self.assertEqual(r['models']['large']['primary']['qualified'],94)
   self.assertEqual(r['paired']['primary']['qualified']['large_minus_small'],0)
   self.assertEqual(r['historical_small_overlap']['intersection'],94)
   self.assertFalse(r['repeat_checks_included'])

if __name__=='__main__':unittest.main()
