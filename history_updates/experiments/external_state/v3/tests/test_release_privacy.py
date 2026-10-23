"""Synthetic-only regressions for operational identifiers and container scope."""
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repro.privacy import find_operational_fields, inspect_text_container


class ReleasePrivacyTests(unittest.TestCase):
    RESOURCE = 'SYNTHETIC_TEST_RESOURCE_731'
    THREAD = 'SYNTHETIC_TEST_THREAD_912'

    def record(self):
        return {'owned_id': self.RESOURCE, 'original_contract': {'thread_id': self.THREAD},
                'subject_id': 'SCIENTIFIC_SUBJECT_1', 'model_id': 'MODEL_FOR_TEST',
                'score': 0.125, 'seed': 1}

    def test_top_level_resource_and_nested_thread_are_detected(self):
        findings = find_operational_fields(json.dumps(self.record()))
        self.assertEqual(['owned_id', 'thread_id'], [r['field'] for r in findings])
        serialized = json.dumps(findings)
        self.assertNotIn(self.RESOURCE, serialized)
        self.assertNotIn(self.THREAD, serialized)

    def test_anonymous_values_are_not_private_identifiers(self):
        value = self.record()
        value['owned_id'] = 'ANONYMIZED_RESOURCE'
        value['original_contract']['thread_id'] = 'ANONYMIZED_THREAD'
        self.assertEqual([], find_operational_fields(json.dumps(value)))

    def test_scientific_ids_names_models_scores_and_citations_are_untouched(self):
        value = {'id': self.RESOURCE, 'subject_id': self.THREAD,
                 'model_id': 'MODEL_FOR_TEST', 'citation': 'Required scientific citation',
                 'seed': 1, 'score': 0.125}
        raw = json.dumps(value)
        self.assertEqual([], find_operational_fields(raw))
        self.assertEqual(value, json.loads(raw))

    def test_json_embedded_in_json_is_detected(self):
        text = json.dumps({'message': json.dumps(self.record())})
        self.assertEqual(2, len(find_operational_fields(text)))

    def test_gzip_occurrences_are_not_skipped(self):
        raw = gzip.compress(json.dumps(self.record()).encode(), mtime=0)
        self.assertEqual(2, len(inspect_text_container(raw, 'record.json.gz')))

    def test_nested_zip_gzip_occurrences_are_not_skipped(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('records/', b'')
            archive.writestr('records/record.json.gz', gzip.compress(json.dumps(self.record()).encode(), mtime=0))
        self.assertEqual(2, len(inspect_text_container(buffer.getvalue(), 'records.zip')))

    def test_private_value_in_name_or_zip_comment_is_detected_without_disclosure(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.comment = self.THREAD.encode()
            archive.writestr(self.RESOURCE + '.txt', 'Synthetic content')
        findings = inspect_text_container(buffer.getvalue(), 'records.zip',
                                          forbidden_values=(self.RESOURCE, self.THREAD))
        self.assertTrue(any(r['kind'] == 'forbidden_private_value' for r in findings))
        self.assertNotIn(self.THREAD, json.dumps(findings))
        self.assertNotIn(self.RESOURCE, json.dumps(findings))

    def test_unsafe_archive_path_fails(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('../outside.json', '{}')
        with self.assertRaises(ValueError):
            inspect_text_container(buffer.getvalue(), 'records.zip')

    def test_truncated_gzip_fails_instead_of_passing(self):
        raw = gzip.compress(json.dumps(self.record()).encode(), mtime=0)
        with self.assertRaises((EOFError, OSError)):
            inspect_text_container(raw[:-6], 'record.json.gz')

    def test_pickle_payload_is_never_deserialized_or_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'typed release scanner'):
            inspect_text_container(b'not a pickle to execute', 'array.pkl')


if __name__ == '__main__':
    unittest.main()
