import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from repro.example import build, current_records, verify, verify_pair


class RecordedExampleTests(unittest.TestCase):
    def test_recorded_example_matches_included_sources(self):
        self.assertEqual('PASS', verify(ROOT)['status'])

    def fixture(self):
        data = json.loads((ROOT / 'data/examples/recorded_case_1617.json').read_bytes())
        case = {'case_id': data['case_id'], 'histories': data['history_inputs'], 'labels': data['labels']}
        rows = copy.deepcopy(data['repair_responses'])
        for r in rows:
            r.update(case_id=data['case_id'], query_id=data['query']['query_id'], arm=data['repair_arm'],
                     phase='post', question=data['query']['question'], candidates=[r['selected']], margins=[1.0],
                     expected_entity=data['query']['gold'], expected_program={'start': r['selected']['start_id'],
                     'path': r['selected']['path'], 'entity': data['query']['gold']})
        return case, data['query'], rows

    def test_unequal_independent_current_records_fail(self):
        case, query, rows = self.fixture()
        case['histories']['H1']['edits'].append(['Q433846', 'P140', 'SYNTHETIC_DIFFERENT_ENTITY'])
        with self.assertRaisesRegex(ValueError, 'current records differ'):
            verify_pair(case, query, rows, lambda c, m: c[0])

    def test_answer_query_seed_and_history_mutations_fail(self):
        for change in ('answer', 'query', 'seed', 'history'):
            case, query, rows = self.fixture()
            if change == 'answer':
                rows[0]['outcome']['answer'] = 'SYNTHETIC_WRONG_ANSWER'
            elif change == 'query':
                rows[0]['selected']['path'] = ['P140']
            elif change == 'seed':
                rows[0]['arm'] = 'STATE_1'
            else:
                rows[0]['history'] = 'H1'
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                verify_pair(case, query, rows, lambda c, m: c[0])

    def test_duplicate_starting_field_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            current_records({'initial': [['s', 'p', 'a'], ['s', 'p', 'b']], 'edits': []})
