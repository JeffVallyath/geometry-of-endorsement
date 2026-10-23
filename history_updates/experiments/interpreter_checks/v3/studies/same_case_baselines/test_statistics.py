import unittest
import numpy as np
from evaluate import bootstrap,ENDPOINTS
class StatisticsTests(unittest.TestCase):
    def test_linked_seed_mean_and_all_four_contrasts(self):
        ids=list(range(128));complete={a:{i:False for i in ids} for a in ENDPOINTS}
        complete['REPAIR_0']={i:True for i in ids}
        result,means=bootstrap(complete,ids)
        self.assertEqual(set(result),{'QA','PROGRAM','MELLO_CURRENT','GMELLO_LOCAL'})
        self.assertEqual(means['REPAIR'],[.5]*128)
        for x in result.values():self.assertEqual(x['difference'],.5);self.assertEqual(x['interval'],[.5,.5])
    def test_identical_systems_no_equivalence_claim(self):
        ids=list(range(128));complete={a:{i:i%2==0 for i in ids} for a in ENDPOINTS}
        result,_=bootstrap(complete,ids)
        for x in result.values():self.assertEqual(x['difference'],0);self.assertFalse(x['substantively_positive']);self.assertFalse(x['equivalence_claim'])
    def test_opposing_subjects_not_independent_seed_resamples(self):
        ids=list(range(128));complete={a:{i:False for i in ids} for a in ENDPOINTS}
        complete['REPAIR_0']={i:i%2==0 for i in ids};complete['REPAIR_1']={i:i%2==1 for i in ids}
        result,_=bootstrap(complete,ids);self.assertEqual(result['QA']['interval'],[.5,.5])
if __name__=='__main__':unittest.main()
