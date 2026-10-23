"""Presentation and source-binding tests use small synthetic records."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repro.tables import (CONDITIONS, MODELS, NATIVE, SEEDS, current_error_counts,
                          document_block, harm_counts, ordered_repair_rows,
                          validate_paired_gain)


class ReviewerTableTests(unittest.TestCase):
    def endpoints(self):
        result = {'repair': []}
        for model in MODELS:
            for condition in reversed(CONDITIONS):
                if condition == 'REFLECT' and model != MODELS[0]:
                    continue
                arm = NATIVE[model] if condition == 'NATIVE' else condition
                result['repair'].append({'model': model, 'arm': arm,
                    'complete_subjects': 1, 'subjects': 2, 'correct_targets': 2,
                    'targets': 4, 'correct_controls': 3, 'controls': 6})
        return result

    def test_comparable_order_and_unmeasured_are_explicit(self):
        rows = ordered_repair_rows(self.endpoints())
        for model in MODELS:
            subset = [r for r in rows if r['model'] == model]
            self.assertEqual([NATIVE[model], *CONDITIONS[1:]], [r['arm'] for r in subset])
        absent = [r for r in rows if r['measurement_status'] == 'UNMEASURED']
        self.assertEqual(2, len(absent))
        self.assertTrue(all(r['arm'] == 'REFLECT' for r in absent))
        self.assertTrue(all('complete_subjects' not in r for r in absent))

    def test_missing_measured_condition_is_not_mislabeled_unmeasured(self):
        result = self.endpoints()
        result['repair'] = [r for r in result['repair'] if not (r['model'] == MODELS[1] and r['arm'] == 'STATE_1')]
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            ordered_repair_rows(result)

    def test_duplicate_endpoint_fails(self):
        result = self.endpoints()
        result['repair'].append(copy.deepcopy(result['repair'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ordered_repair_rows(result)

    def test_unrecognized_measured_condition_cannot_disappear(self):
        result = self.endpoints()
        extra = copy.deepcopy(result['repair'][0])
        extra['arm'] = 'SYNTHETIC_EXTRA_CONDITION'
        result['repair'].append(extra)
        with self.assertRaisesRegex(ValueError, 'omitted'):
            ordered_repair_rows(result)

    def outcomes(self):
        return [dict(arm=arm, history='H0', phase='post', case_id='SYNTHETIC_CASE',
                     query_id='Q' + str(i), status='REFUSAL' if arm == 'STATE_0' and i else 'VALID',
                     correct=i == 0) for arm in SEEDS for i in range(2)]

    def test_refusals_and_wrong_accepted_answers_remain_distinct(self):
        result = current_error_counts(self.outcomes(), 2)
        self.assertEqual({'queries': 2, 'refusals': 1, 'wrong_accepted': 0}, result['STATE_0'])
        self.assertEqual({'queries': 2, 'refusals': 0, 'wrong_accepted': 1}, result['STATE_1'])

    def test_qwen7_nested_outcome_schema_is_preserved(self):
        rows = self.outcomes()
        for row in rows:
            row['outcome'] = {'status': row.pop('status')}
        self.assertEqual(1, current_error_counts(rows, 2, qwen7=True)['STATE_0']['refusals'])

    def test_missing_or_duplicate_error_record_fails(self):
        for rows in (self.outcomes()[:-1], self.outcomes() + [self.outcomes()[0]]):
            with self.assertRaises(ValueError):
                current_error_counts(rows, 2)

    def harms(self):
        return [dict(candidate=arm, baseline=base, case_id='SYNTHETIC_CASE',
                     query_id='Q' + str(i), baseline_correct=True, candidate_correct=i == 0,
                     harmed=i == 1) for arm in SEEDS for base in ('A', 'B') for i in range(2)]

    def test_harms_are_comparator_specific_and_bound_to_correctness(self):
        self.assertEqual({'A': 1, 'B': 1}, harm_counts(self.harms(), ('A', 'B'), 2)['STATE_0'])
        mutated = self.harms()
        mutated[1]['harmed'] = False
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            harm_counts(mutated, ('A', 'B'), 2)

    def test_missing_harm_comparison_is_not_a_zero(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            harm_counts(self.harms()[:-1], ('A', 'B'), 2)

    def test_paired_gain_uses_both_seeds_and_original_denominator(self):
        validate_paired_gain({'difference': 0.5, 'subjects': 4}, [2, 4], 1, 4)
        for contrast, counts in (({'difference': 0.75, 'subjects': 4}, [2, 4]),
                                 ({'difference': 0.5, 'subjects': 3}, [2, 4]),
                                 ({'difference': 0.5, 'subjects': 4}, [4])):
            with self.assertRaises(ValueError):
                validate_paired_gain(contrast, counts, 1, 4)

    def test_document_binding_requires_unique_markers(self):
        from repro.tables import BEGIN, END
        self.assertEqual('table\n', document_block(BEGIN + '\ntable\n' + END))
        for text in ('unbound prose', BEGIN + BEGIN + END):
            with self.assertRaises(ValueError):
                document_block(text)


if __name__ == '__main__':
    unittest.main()
