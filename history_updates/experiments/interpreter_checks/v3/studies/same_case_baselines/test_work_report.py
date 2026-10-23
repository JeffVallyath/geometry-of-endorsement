import unittest
from report_work import call_ids,account_calls
class WorkReportTests(unittest.TestCase):
    def test_nested_refs_are_counted_once(self):
        obj=[{'call_ids':['score','score']},{'trace':[{'generation':{'call_id':'generation'}}]},{'call_id':'generation'}]
        self.assertEqual(call_ids(obj),{'score','generation'})
    def test_real_counts_not_caps_and_amortized_batch_time(self):
        calls={'s':{'kind':'score','started_after_scientific_freeze':False,'result':{'input_tokens':20,'amortized_seconds':.2}},
               'g':{'kind':'generate','started_after_scientific_freeze':True,'result':{'input_tokens':30,'output_tokens':4,'amortized_seconds':.3,'cap_hit':False}}}
        out=account_calls(['s','g','g'],calls,20)
        self.assertEqual(out['input_tokens'],50);self.assertEqual(out['generated_tokens'],4)
        self.assertEqual(out['unique_neural_calls'],2);self.assertAlmostEqual(out['amortized_neural_seconds'],.5)
        self.assertEqual(out['prefreeze_exact_request_reuse_calls'],1)
    def test_missing_call_is_not_zero_cost(self):
        with self.assertRaises(AssertionError):account_calls(['missing'],{},0)
if __name__=='__main__':unittest.main()
