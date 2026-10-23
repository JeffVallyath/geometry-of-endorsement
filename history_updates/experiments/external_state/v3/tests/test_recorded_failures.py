"""Real failure rows must survive the comparison used by production replay."""
import copy
import json
from pathlib import Path
import unittest

from repro.source_driver import scientific_equal


class RecordedFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=json.loads((Path(__file__).parent/'fixtures/recorded_failures.json').read_text(encoding='utf8'))
        cls.rows=fixture['failures']

    def test_required_refusals_and_wrong_programs_are_real(self):
        self.assertEqual(len(self.rows),4)
        self.assertTrue(all(not item['outcome']['correct'] for item in self.rows))
        self.assertEqual(sum(item['study']=='repair_qwen32' for item in self.rows),2)
        self.assertEqual(sum(item['study']=='repair_gemma27' for item in self.rows),2)

    def test_removing_any_failed_row_is_detected(self):
        for index in range(len(self.rows)):
            with self.subTest(row=index):
                changed=copy.deepcopy(self.rows);changed.pop(index)
                with self.assertRaisesRegex(ValueError,'SCIENTIFIC_REPLAY_MISMATCH'):
                    scientific_equal(changed,self.rows,'ALL_OUTCOMES.jsonl.gz')

    def test_turning_a_refusal_into_correctness_is_detected(self):
        for index,item in enumerate(self.rows):
            if item['study']!='repair_qwen32':continue
            changed=copy.deepcopy(self.rows);changed[index]['outcome']['correct']=True
            with self.assertRaisesRegex(ValueError,'SCIENTIFIC_REPLAY_MISMATCH'):
                scientific_equal(changed,self.rows,'ALL_OUTCOMES.jsonl.gz')


if __name__=='__main__':unittest.main()
