"""Model-free tests for the prospective LP1 convention, not GPU qualification."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from icmh.parse import build_registry,build_registry_union,extract,strict_correct

class TestLabelPrecedence(unittest.TestCase):
    def setUp(self):
        self.entities={
            'E1':{'labels':['Amber'],'aliases':['Amber city','A-town'],'types':['place']},
            'E2':{'labels':['Beryl'],'aliases':['Amber','Shared'],'types':['place']},
            'E3':{'labels':['Cedar'],'aliases':['Shared'],'types':['place']},
            'E4':{'labels':['Dale'],'aliases':['D-one'],'types':['place']},
            'E5':{'labels':['Dale'],'aliases':['D-two'],'types':['place']},
            'E6':{'labels':['Amber'],'aliases':[],'types':['person']},
        }
        self.r=build_registry(self.entities)
    def test_unique_label_beats_foreign_alias(self):
        self.assertEqual(extract('Amber',self.r,'place')['entity_id'],'E1')
        self.assertEqual(extract('Amber',build_registry_union(self.entities),'place')['status'],'ambiguous')
    def test_unique_alias_is_retained(self):
        self.assertEqual(extract('A-town',self.r,'place')['entity_id'],'E1')
    def test_two_primary_labels_abstain(self):
        self.assertEqual(extract('Dale',self.r,'place')['status'],'ambiguous')
    def test_ambiguous_aliases_abstain(self):
        self.assertEqual(extract('Shared',self.r,'place')['status'],'ambiguous')
    def test_type_scope_unchanged(self):
        self.assertEqual(extract('Amber',self.r,'person')['entity_id'],'E6')
    def test_normalization_same_for_labels_and_outputs(self):
        self.assertEqual(extract('  ＡＭＢＥＲ. ',self.r,'place')['entity_id'],'E1')
    def test_extra_prose_not_rescued(self):
        self.assertIsNone(extract('Amber is a city',self.r,'place')['entity_id'])
    def test_registry_does_not_mutate_entities(self):
        import copy
        before=copy.deepcopy(self.entities);build_registry(self.entities)
        self.assertEqual(self.entities,before)
    def test_no_gold_parameter_or_per_query_restriction(self):
        import inspect
        self.assertEqual(list(inspect.signature(extract).parameters),['text','registry','object_type'])
        # Its wrong-entity implication is intentional and must not become gold-aware.
        self.assertNotEqual(extract('Amber',self.r,'place')['entity_id'],'E2')
    def test_input_order_does_not_pick_ties(self):
        self.assertEqual(self.r,build_registry(dict(reversed(list(self.entities.items())))))
    def test_strict_correctness_unchanged(self):
        answers=[{'value':'Amber','aliases':['A-town']}]
        self.assertTrue(strict_correct('Amber.',answers))
        self.assertTrue(strict_correct('A-town',answers))
        self.assertFalse(strict_correct('amber',answers))
        self.assertFalse(strict_correct('The answer is Amber',answers))

if __name__=='__main__':unittest.main()
