"""CPU tests for ICMH1-RIPPLE-v2. No model, GPU, provider or network access.

Run: python -m icmh.tests  (or python -m unittest icmh.tests -v)
Structural tests run against the real built inputs when ICMH_INPUTS points at them.
"""
import os
from pathlib import Path
import tempfile
import unittest

from . import analyze as A
from . import build as B
from .common import Journal, IntegrityError, read_json, sha_bytes
from .parse import build_registry, extract, strict_correct, strict_line, unresolved

INPUTS = Path(os.environ.get('ICMH_INPUTS', 'D:/icmh1-ripple-v2/inputs/built'))


class TestStrictScorer(unittest.TestCase):
    def test_first_line_and_one_punctuation(self):
        self.assertEqual(strict_line(' Asia.'), 'Asia')
        self.assertEqual(strict_line('Asia\nEurope'), 'Asia')
        self.assertEqual(strict_line('Asia?!'), 'Asia?')
        self.assertEqual(strict_line('\n\nAsia'), 'Asia')
        self.assertEqual(strict_line('Washington, D.C.'), 'Washington, D.C')
        self.assertEqual(strict_line(''), '')
        self.assertIsNone(strict_line(None))

    def test_no_line_skipping_or_substring_rescue(self):
        answers = [{'value': 'Asia', 'aliases': ['Asian continent']}]
        self.assertFalse(strict_correct('I cannot say.\nAsia', answers))
        self.assertFalse(strict_correct('The answer is Asia', answers))
        self.assertFalse(strict_correct('asia', answers))
        self.assertTrue(strict_correct(' Asia.', answers))
        self.assertTrue(strict_correct('Asian continent', answers))

    def test_alias_processed_by_same_rule(self):
        self.assertTrue(strict_correct('Washington, D.C.', [{'value': 'Washington, D.C.', 'aliases': []}]))


class TestTypedExtractor(unittest.TestCase):
    def setUp(self):
        self.entities = {
            'Q48': {'labels': ['Asia'], 'aliases': ['Asian continent'], 'types': ['continent']},
            'Q3960': {'labels': ['Australia', 'Australian continent'], 'aliases': ['Sahul'], 'types': ['continent']},
            'Q408': {'labels': ['Australia'], 'aliases': [], 'types': ['country']},
            'Q6581097': {'labels': ['male'], 'aliases': ['man', '♂'], 'types': ['gender']},
            'Q97595519': {'labels': ['androgyne'], 'aliases': ['gynandre'], 'types': ['gender']},
        }
        self.registry = build_registry(self.entities)

    def test_unique_typed_match(self):
        result = extract(' Asia.', self.registry, 'continent')
        self.assertEqual((result['status'], result['entity_id']), ('unique', 'Q48'))

    def test_type_scoping_separates_same_name_ids(self):
        self.assertEqual(extract('Australia', self.registry, 'continent')['entity_id'], 'Q3960')
        self.assertEqual(extract('Australia', self.registry, 'country')['entity_id'], 'Q408')

    def test_ambiguous_same_name_inside_one_type(self):
        entities = dict(self.entities)
        entities['Q1316093'] = {'labels': ['Asia'], 'aliases': [], 'types': ['continent']}
        registry = build_registry(entities)
        result = extract('Asia', registry, 'continent')
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIsNone(result['entity_id'])
        self.assertTrue(unresolved(result['status']))

    def test_alias_and_unicode_normalization(self):
        self.assertEqual(extract('gynandre', self.registry, 'gender')['entity_id'], 'Q97595519')
        self.assertEqual(extract('MALE', self.registry, 'gender')['entity_id'], 'Q6581097')

    def test_wrong_extra_prose_empty_and_truncation_abstain(self):
        for text in ('Asia is the largest continent', '', 'Atlantis', 'The continent of Asi'):
            with self.subTest(text=text):
                result = extract(text, self.registry, 'continent')
                self.assertIsNone(result['entity_id'])
                self.assertTrue(unresolved(result['status']))

    def test_no_gold_dependence(self):
        first = extract('Asia', self.registry, 'continent')
        second = extract('Asia', self.registry, 'continent')
        self.assertEqual(first, second)


class TestJournal(unittest.TestCase):
    def test_chain_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'j.jsonl'
            journal = Journal(path)
            journal.append({'event': 'generation', 'unit_id': 'a', 'text': 'x'})
            journal.append({'event': 'generation', 'unit_id': 'b', 'text': 'y'})
            records, head, incomplete = journal.load()
            self.assertEqual([r['unit_id'] for r in records], ['a', 'b'])
            self.assertFalse(incomplete)
            self.assertEqual(journal.completed_units(), {'a', 'b'})
            lines = path.read_text(encoding='utf-8').splitlines()
            lines[0] = lines[0].replace('"x"', '"z"')
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            with self.assertRaises(IntegrityError):
                Journal(path).load()

    def test_incomplete_tail_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'j.jsonl'
            journal = Journal(path)
            journal.append({'event': 'generation', 'unit_id': 'a'})
            with path.open('a', encoding='utf-8') as stream:
                stream.write('{"payload": {"event"\n')
            _, _, incomplete = Journal(path).load()
            self.assertTrue(incomplete)


class TestFlagLogic(unittest.TestCase):
    """A/C/D/W semantics, including the formatting-only and two-wrong-answer exclusions."""

    def build(self, b_text, d_text, given='Asia', canonical='Asia', record_ok=True):
        root = {'root_id': 'r', 'cluster_id': 'c', 'changed_address': ['S', 'P', 't'],
                'changed_slot_ids': {'B': 'QB', 'D': 'QD', 'C': 'QC'},
                'current_table': [{'address': ['S', 'P', 't'], 'target_id': 'QC', 'target_label': 'Gold',
                                   'object_type': 'continent', 'template': 'x is', 'sentence': 'x is Gold',
                                   'record': {'target_aliases': []}}],
                'current_lines': ['x is Gold'],
                'questions': [{'question_id': 'q', 'category': 'Compositionality_I', 'prompt': 'p',
                               'answers': [{'value': 'Asia', 'aliases': []}], 'gold_ids': ['Q48'],
                               'final_hop_object_type': 'continent', 'final_hop_address': ['S', 'P', 't'],
                               'hop_count': 2, 'changed_address_hop': 1}]}
        plan = {'roots': ['r']}
        registry = build_registry({'Q48': {'labels': ['Asia'], 'aliases': [], 'types': ['continent']},
                                   'Q46': {'labels': ['Europe'], 'aliases': [], 'types': ['continent']},
                                   'QC': {'labels': ['Gold'], 'aliases': [], 'types': ['continent']}})
        rows = []

        def cell(kind, reference, context, text, object_type, gold):
            return {'unit_id': kind + context, 'root_id': 'r', 'kind': kind, 'context': context,
                    'reference': reference, 'measured': True, 'text': text,
                    'strict_correct': strict_correct(text, [{'value': gold, 'aliases': []}]),
                    'input_tokens': 10, 'output_tokens': 2, 'ended_eos': True, 'truncated': False,
                    'elapsed_seconds': 0.1, 'object_type': object_type,
                    **({'extracted_id': extract(text, registry, object_type)['entity_id'],
                        'extraction_status': extract(text, registry, object_type)['status'],
                        'extraction_line': strict_line(text), 'gold_ids': ['Q48'],
                        'extracted_is_gold': extract(text, registry, object_type)['entity_id'] == 'Q48'}
                       if object_type else {})}

        rows.append(cell('downstream', 'q', 'given_facts', given, 'continent', 'Asia'))
        rows.append(cell('downstream', 'q', 'canonical', canonical, 'continent', 'Asia'))
        rows.append(cell('downstream', 'q', 'chronological_B', b_text, 'continent', 'Asia'))
        rows.append(cell('downstream', 'q', 'chronological_D', d_text, 'continent', 'Asia'))
        rows.append(cell('downstream', 'q', 'none', 'Europe', 'continent', 'Asia'))
        for context in ('chronological_B', 'chronological_D', 'none'):
            text = 'Gold' if record_ok else 'Wrong'
            rows.append(cell('atomic', 'S|P|t', context, text, 'continent', 'Gold'))
        return A.flags(plan, A.index(rows), {'r': root})

    def test_witness_requires_a_c_and_one_correct_distinct_entity(self):
        per_root, per_question = self.build('Asia', 'Europe')
        self.assertTrue(per_root['r']['A'])
        question = per_question['q']
        self.assertTrue(question['C'] and question['D'] and question['W'])

    def test_formatting_only_difference_is_not_a_witness(self):
        _, per_question = self.build('Asia', 'Asia.')
        self.assertFalse(per_question['q']['D'])
        self.assertFalse(per_question['q']['W'])

    def test_two_different_wrong_answers_are_not_a_witness(self):
        _, per_question = self.build('Europe', 'Gold')
        self.assertFalse(per_question['q']['D'])
        self.assertFalse(per_question['q']['W'])

    def test_unmatched_answer_blocks_d(self):
        _, per_question = self.build('Asia', 'somewhere else entirely')
        self.assertFalse(per_question['q']['D'])

    def test_failed_records_block_a_and_w_but_keep_c(self):
        per_root, per_question = self.build('Asia', 'Europe', record_ok=False)
        self.assertFalse(per_root['r']['A'])
        self.assertTrue(per_question['q']['C'])
        self.assertFalse(per_question['q']['W'])

    def test_failed_reference_blocks_c(self):
        _, per_question = self.build('Asia', 'Europe', canonical='Europe')
        self.assertFalse(per_question['q']['C'])
        self.assertFalse(per_question['q']['W'])

    def test_missing_rows_are_missing_not_zero(self):
        plan = {'roots': ['r']}
        root = {'root_id': 'r', 'cluster_id': 'c', 'changed_address': ['S', 'P', 't'],
                'changed_slot_ids': {}, 'current_table': [], 'current_lines': [],
                'questions': [{'question_id': 'q', 'category': 'Compositionality_I', 'prompt': 'p',
                               'answers': [{'value': 'Asia', 'aliases': []}], 'gold_ids': ['Q48'],
                               'final_hop_object_type': 'continent', 'final_hop_address': ['S', 'P', 't'],
                               'hop_count': 2, 'changed_address_hop': 1}]}
        _, per_question = A.flags(plan, {}, {'r': root})
        self.assertIsNone(per_question['q']['W'])
        self.assertFalse(per_question['q']['C_measured'])


class TestGatesAndForecast(unittest.TestCase):
    def test_gate_thresholds_are_the_amendment_values(self):
        self.assertEqual(A.GATES['G1']['given_facts_min'], 13)
        self.assertEqual(A.GATES['G1']['canonical_min'], 13)
        self.assertEqual(A.GATES['G2']['records_correct_both_min'], 18)
        self.assertEqual(A.GATES['G4']['unresolved_max'], 4)
        self.assertEqual((A.GATES['G5']['questions_min'], A.GATES['G5']['roots_min']), (10, 3))

    def test_forecast_applies_allowance_and_reserve(self):
        measured = {'p95_seconds': 1.0, 'load_seconds_max': 60.0}
        fits = A.forecast(measured, 1000, 3, seconds_available=1000 + 180 + 1800 + 500)
        self.assertTrue(fits['fits'])
        self.assertAlmostEqual(fits['work_seconds_with_allowance'], (1000 + 180) * 1.25)
        tight = A.forecast(measured, 1000, 3, seconds_available=1800 + 100)
        self.assertFalse(tight['fits'])

    def test_forecast_without_measurement_fails_closed(self):
        self.assertFalse(A.forecast({}, 10, 1, 10000)['fits'])


class TestRepeatComparison(unittest.TestCase):
    def test_differences_are_preserved(self):
        main = [{'unit_id': 'u', 'measured': True, 'root_id': 'r', 'kind': 'downstream', 'context': 'none',
                 'reference': 'q', 'text': 'Asia', 'strict_correct': True, 'extracted_id': 'Q48'}]
        repeat = [dict(main[0], text='Europe', strict_correct=False, extracted_id='Q46')]
        result = A.repeat_comparison(main, repeat)
        self.assertFalse(result['identical'])
        self.assertEqual(len(result['differences']), 1)
        self.assertTrue(A.repeat_comparison(main, main)['identical'])


class TestBuiltInputs(unittest.TestCase):
    """Structural checks against the real built inputs (skipped when absent)."""

    @classmethod
    def setUpClass(cls):
        if not (INPUTS / 'STRUCTURE_CHECK.json').exists():
            raise unittest.SkipTest('built inputs not present: %s' % INPUTS)
        cls.structure = read_json(INPUTS / 'STRUCTURE_CHECK.json')
        cls.roots = read_json(INPUTS / 'ROOTS.json')['roots']
        cls.plans = {phase: read_json(INPUTS / ('PLAN_%s.json' % phase))
                     for phase in ('development', 'qualification', 'final', 'repeat_670', 'repeat_618')}

    def test_counts_match_scope(self):
        self.assertEqual(self.structure['checks']['roots'], 45)
        self.assertEqual(self.structure['checks']['distinct_clusters'], 45)
        self.assertEqual(sum(self.structure['category_counts'].values()), 128)

    def test_no_prompt_leakage(self):
        self.assertEqual(self.structure['checks']['prompt_leakage_findings'], 0)
        for phase, plan in self.plans.items():
            for unit in plan['units']:
                self.assertNotIn('[', unit['user_text'], phase)
                self.assertNotIn('benchmark_slot_assignment', unit['user_text'], phase)

    def test_streams_differ_only_in_the_obsolete_line(self):
        for root in self.roots:
            b = root['streams']['B']['lines']
            d = root['streams']['D']['lines']
            self.assertEqual(len(b), len(d))
            self.assertEqual(b[1:], d[1:], root['root_id'])
            self.assertNotEqual(b[0], d[0], root['root_id'])

    def test_given_facts_and_canonical_share_records_and_differ_only_in_header(self):
        for phase, plan in self.plans.items():
            given = {u['reference']: u['user_text'] for u in plan['units']
                     if u['kind'] == 'downstream' and u['context'] == 'given_facts'}
            canonical = {u['reference']: u['user_text'] for u in plan['units']
                         if u['kind'] == 'downstream' and u['context'] == 'canonical'}
            self.assertEqual(set(given), set(canonical), phase)
            for reference, text in given.items():
                self.assertEqual(text.split('\n', 1)[1], canonical[reference].split('\n', 1)[1])

    def test_chronological_contexts_contain_the_complete_current_table(self):
        for root in self.roots:
            for key in ('B', 'D'):
                lines = root['streams'][key]['lines']
                for sentence in root['current_lines']:
                    self.assertIn(sentence, lines, root['root_id'])
                self.assertEqual(len(lines), len(root['current_lines']) + 1)

    def test_bdc_distinct_and_changed_address_present(self):
        for root in self.roots:
            slot = root['changed_slot_ids']
            self.assertEqual(len({slot['B'], slot['D'], slot['C']}), 3, root['root_id'])
            addresses = [row['address'] for row in root['current_table']]
            self.assertIn(root['changed_address'], addresses, root['root_id'])

    def test_repeat_plans_match_their_final_root_units(self):
        final = self.plans['final']
        for phase, root_id in (('repeat_670', 'rippleedits:popular:670'), ('repeat_618', 'rippleedits:random:618')):
            expected = [u for u in final['units'] if u['root_id'] == root_id]
            actual = self.plans[phase]['units']
            self.assertEqual([u['unit_id'] for u in expected], [u['unit_id'] for u in actual], phase)
            self.assertEqual([u['user_text'] for u in expected], [u['user_text'] for u in actual], phase)

    def test_qualification_excludes_chronological_downstream(self):
        contexts = {u['context'] for u in self.plans['qualification']['units'] if u['kind'] == 'downstream'}
        self.assertEqual(contexts, {'given_facts', 'canonical', 'none'})
        self.assertFalse(any(u['kind'] == 'locality' for u in self.plans['qualification']['units']))

    def test_final_covers_all_five_contexts_and_locality(self):
        contexts = {u['context'] for u in self.plans['final']['units'] if u['kind'] == 'downstream'}
        self.assertEqual(contexts, {'none', 'given_facts', 'canonical', 'chronological_B', 'chronological_D'})
        self.assertEqual(sum(1 for u in self.plans['final']['units'] if u['kind'] == 'locality'), 34 * 8 * 2)

    def test_unit_ids_are_unique_within_each_plan(self):
        for phase, plan in self.plans.items():
            ids = [u['unit_id'] for u in plan['units']]
            self.assertEqual(len(ids), len(set(ids)), phase)


if __name__ == '__main__':
    unittest.main(verbosity=2)
