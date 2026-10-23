import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'tools'))
import review_examples
from review_support import digest, load, release_api, verify_bundle


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='history-review-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name) / 'readable results with spaces'
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = review_examples.run(ROOT / 'examples/records', cls.output)

    def test_real_synthetic_checks_and_probabilities(self):
        result = self.result['synthetic']
        self.assertEqual(result['case_id'], 'FHV1-FINAL-0017')
        self.assertEqual((result['checks_passed'], result['checks_total']), (160, 160))
        self.assertEqual(len(result['all_decisions']), 36)
        self.assertEqual(result['prefix_lengths'][5:7], [455, 455])
        self.assertEqual(result['changed_facts'][5:7], [2, 2])
        self.assertEqual(result['decision']['predictions'][5:7], [0, 1])
        self.assertAlmostEqual(result['decision']['correct_probabilities'][5], .7786703506616323)
        self.assertAlmostEqual(result['decision']['correct_probabilities'][6], .017623117607588112)
        self.assertTrue(result['decision']['both_matched'])
        self.assertFalse(result['contrast']['both_eligible'])

    def test_references_are_not_counted_as_160_checks(self):
        for name, response in self.result['synthetic']['references'].items():
            with self.subTest(reference=name):
                self.assertEqual(response['prediction'], 0)
                self.assertTrue(response['valid'])
        self.assertEqual(len(self.result['synthetic']['checks']), 160)

    def test_public_original_counts_and_qualified_simple_cases(self):
        for model, count in [('qwen', 94), ('gemma', 101)]:
            value = self.result['public'][model]
            self.assertEqual((value['qualified_questions'], value['scheduled_questions'], value['cases']),
                             (count, 384, 128))
            self.assertEqual(value['examples']['simple']['case']['hops'], 2)
            self.assertTrue(value['examples']['simple']['decision']['qualified'])
            self.assertTrue(value['examples']['contrast']['decision']['common_guard'])
            self.assertFalse(value['examples']['contrast']['decision']['qualified'])

    def test_recorded_paper_case_keeps_all_histories_and_checks(self):
        for model in ('qwen', 'gemma'):
            example = self.result['public'][model]['examples']['paper']
            self.assertEqual(example['case']['case_id'], 1617)
            self.assertEqual(len(example['responses']), 48)
            self.assertTrue(example['decision']['long_chain_qualified'])

    def test_readable_exports_match_committed_views_exactly(self):
        for source in (self.output / 'views').rglob('*'):
            if source.is_file():
                delivered = ROOT / 'examples' / source.relative_to(self.output / 'views')
                with self.subTest(file=str(delivered.relative_to(ROOT))):
                    self.assertEqual(source.read_bytes(), delivered.read_bytes())

    def clone_records(self):
        temporary = tempfile.TemporaryDirectory(prefix='review-input-fixture-')
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / 'records'
        shutil.copytree(ROOT / 'examples/records', destination)
        return destination

    def test_missing_input_rejected(self):
        records = self.clone_records()
        (records / 'synthetic/SCORES.json.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or changed'):
            verify_bundle(records)

    def test_tampered_observation_rejected(self):
        records = self.clone_records()
        with (records / 'synthetic/SCORES.json.gz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Missing or changed'):
            verify_bundle(records)

    def test_rehashing_tampered_export_does_not_change_submitted_binding(self):
        records = self.clone_records()
        path = records / 'synthetic/SCORES.json.gz'
        value = load(path)
        value['physical'][0]['prediction'] = 1 - value['physical'][0]['prediction']
        path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
        binding_path = records / 'BINDINGS.json'
        binding = load(binding_path)
        binding['records']['synthetic/SCORES.json.gz']['export_sha256'] = digest(path)
        binding_path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, 'exact source bytes'):
            verify_bundle(records)

    def test_undeclared_input_rejected(self):
        records = self.clone_records()
        (records / 'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Undeclared'):
            verify_bundle(records)

    def test_existing_output_preserved(self):
        with self.assertRaises(FileExistsError):
            review_examples.run(ROOT / 'examples/records', self.output)
        self.assertTrue((self.output / 'REPORT.md').is_file())

    def test_in_tree_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside protected'):
            review_examples.run(ROOT / 'examples/records', ROOT / 'never-create-this')
        self.assertFalse((ROOT / 'never-create-this').exists())

    def test_missing_original_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original input'):
            verify_bundle(ROOT / 'examples/records', Path(self.temp.name) / 'absent-original')

    def test_network_and_model_imports_blocked(self):
        for snippet in ('import torch', 'import socket;socket.socket()'):
            command = [sys.executable, '-B', '-X', 'utf8', '-c',
                       "import sys;sys.path.insert(0,'tools');from review_support import forbid_inference_and_network;"
                       'forbid_inference_and_network();' + snippet]
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                  timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.assertNotEqual(done.returncode, 0)

    def test_posix_style_parent_path_and_spaces_work_on_windows(self):
        # Regression for the inherited Path.absolute()/.. lexical containment issue.
        output = Path(self.temp.name) / 'unused' / '..' / 'another example output'
        with contextlib.redirect_stdout(io.StringIO()):
            review_examples.run(ROOT / 'examples/records', output)
        self.assertTrue((Path(self.temp.name) / 'another example output/REPORT.md').is_file())


if __name__ == '__main__':
    unittest.main()
