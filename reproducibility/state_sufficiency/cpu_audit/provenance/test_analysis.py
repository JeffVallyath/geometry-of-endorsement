"""Small adversarial regression suite for the new scientific joins/firewall."""
from common import *
from certify import gate_flags, RestrictedIndex, target_view
from join_outcomes import matched_outcomes, summarize
import copy
import unittest
from pipeline import exact_classification_check

class CertificationTests(unittest.TestCase):
    def test_roundoff_is_not_classification_mismatch(self):
        import math
        saved={'W':True,'predictions':[0,1],'correct_probabilities':[.3,.7]}
        row=copy.deepcopy(saved);row['correct_probabilities'][0]=math.nextafter(.3,1)
        exact_classification_check(row,saved,tuple(saved))
        row['W']=False
        with self.assertRaises(AssertionError):exact_classification_check(row,saved,tuple(saved))
        row['W']=True;row['correct_probabilities'][0]+=.000001
        with self.assertRaises(AssertionError):exact_classification_check(row,saved,tuple(saved))

    def test_nested_conjunctions(self):
        base={'operands':[True]*16,'native_ok':True,'noop_ok':True,'all_direct':[True]*96,'local_checks':[True]*32,'global_checks':[True]*256}
        self.assertEqual(gate_flags(base),dict(G0=True,G1=True,G2=True,G3=True))
        for field,expected in [('operands',[False]*4),('all_direct',[True,False,False,False]),('local_checks',[True,True,False,False]),('global_checks',[True,True,True,False])]:
            view=copy.deepcopy(base);view[field][-1]=False
            self.assertEqual(list(gate_flags(view).values()),expected)
        base['W']=True
        with self.assertRaises(AssertionError):gate_flags(base)

    def test_firewall_and_target_counterfactual_immunity(self):
        r=upstream();spec=load(HERE/'FROZEN/GATE_SPEC.json')
        rid=r.ROOTS[0];design=r.load(PACKAGE/f'inputs/designs/{rid}.json')
        qs=r.validate_design(design,rid);condition='EXISTING_CORRECTION';values=design['root']['terminal']['values']
        index={(context,qid):{'valid':True,'prediction':r.truth(values,q['spec'])} for context in ['NATIVE_FINAL',*[f'o{o}/{condition}' for o in range(8)]] for qid,q in qs.items()}
        rm=spec['mappings'][rid]
        for qid,m in rm['targets'].items():
            safe=RestrictedIndex(index,m['excluded_target_code_variants'],qid,condition)
            before=gate_flags(target_view(safe,condition,qid,m,rm['G1_direct_ids'],qs,values,r.truth))
            mutated=copy.deepcopy(index)
            for other in m['excluded_target_code_variants']:
                for o in range(1,8):mutated[f'o{o}/{condition}',other]={'valid':False,'prediction':1-r.truth(values,qs[other]['spec'])}
            after=gate_flags(target_view(RestrictedIndex(mutated,m['excluded_target_code_variants'],qid,condition),condition,qid,m,rm['G1_direct_ids'],qs,values,r.truth))
            self.assertEqual(before,after)
            with self.assertRaises(AssertionError):_ = safe[f'o1/{condition}',qid]

    def test_pairwise_validity_and_strong_extremes(self):
        pairs=[[0,1],[1,2],[2,3]];pred=[0,1,0,1];p=[.99,.49,.98,.01]
        disagreement,strong=matched_outcomes(pairs,pred,[True]*4,p)
        self.assertEqual(disagreement,pairs);self.assertEqual(strong,[[2,3]])
        self.assertEqual(matched_outcomes(pairs,pred,[True,False,True,True],p),([[2,3]],[[2,3]]))

    def test_mapping_excludes_code_duplicates(self):
        r=upstream();spec=load(HERE/'FROZEN/GATE_SPEC.json')
        for rid,rm in spec['mappings'].items():
            design=r.load(PACKAGE/f'inputs/designs/{rid}.json');qs={q['query_id']:q for q in design['questions']}
            for qid,m in rm['targets'].items():
                self.assertEqual(len(m['G2_checks']),4)
                self.assertEqual(len({semantic(qs[i]) for i in m['G2_checks']}),2)
                self.assertTrue(all(semantic(qs[i])[1:]==semantic(qs[qid])[1:] for i in m['G2_checks']))
                self.assertFalse(set(m['G2_checks'])&set(m['excluded_target_code_variants']))
                self.assertEqual(len(m['G3_checks']),32)

if __name__=='__main__':unittest.main()
