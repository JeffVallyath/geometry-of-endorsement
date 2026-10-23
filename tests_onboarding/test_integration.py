import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'tools'))
from review_support import RELEASE, digest, release_api


class IntegrationTests(unittest.TestCase):
    def test_onboarding_docs_avoid_personal_handoff_context(self):
        names = ('README.md', 'AGENTS.md', 'CLAUDE.md', 'HANDOFF.md', 'CONTRIBUTING.md',
                 'STATUS.md', 'docs/review_and_extensions.md', 'history_updates/README.md',
                 'history_updates/docs/reproduction.md', 'research_sources/readers/README.md')
        patterns = {
            'personal home path': r'(?i)(?:[A-Z]:[\\/]+Users[\\/]+|/(?:Users|home)/)',
            'personal message': r'(?m)^(?:Hi|Hello|Dear)\s+[A-Z]',
            'named contact': r'\b(?:Ask|Contact)\s+[A-Z][a-z]+\b',
            'conversation reference': r'(?i)\b(?:as you requested|the user (?:asked|requested|approved)|(?:current|this) task authorizes)\b',
            'chat handoff': r'(?i)\b(?:share|send)\b[^\n]{0,100}\b(?:Slack|Discord|WhatsApp)\b',
            'account-specific repository URL': r'https?://(?:www\.)?github\.com/[^/\s]+/',
        }
        for name in names:
            body = (ROOT / name).read_text(encoding='utf-8')
            for label, pattern in patterns.items():
                with self.subTest(document=name, category=label):
                    self.assertIsNone(re.search(pattern, body), f'{name}: {label}')

    def test_scientific_software_and_results_are_unchanged(self):
        manifest = RELEASE / 'metadata/files.sha256'
        self.assertEqual(digest(manifest), 'fdbe340a9aced0a21824fa55c56f59de2c5dcea51ee2aead719189a33ba6e34b')
        expected = dict(line.split('  ', 1)[::-1] for line in manifest.read_text().splitlines())
        docs = {'README.md', 'docs/methods.md', 'docs/paper_map.md', 'docs/design_notes.md', 'docs/reproduction.md'}
        for path in RELEASE.rglob('*'):
            if not path.is_file():
                continue
            name = path.relative_to(RELEASE).as_posix()
            if name in docs or name == 'metadata/files.sha256':
                continue
            with self.subTest(file=name):
                self.assertIn(name, expected)
                self.assertEqual(digest(path), expected[name])

    def test_local_links_and_anchors(self):
        docs = [ROOT / name for name in ('README.md', 'AGENTS.md', 'CLAUDE.md', 'HANDOFF.md', 'CONTRIBUTING.md',
                'PAPER_MAP.md', 'REPRODUCIBILITY.md', 'STATUS.md',
                'docs/experiment_overview.md', 'docs/review_and_extensions.md',
                'papers/README.md', 'history_updates/README.md', 'history_updates/docs/methods.md',
                'history_updates/docs/paper_map.md', 'history_updates/docs/design_notes.md',
                'history_updates/docs/reproduction.md')]
        docs += list((ROOT / 'examples').rglob('*.md')) + list((ROOT / 'research_sources').rglob('README.md'))
        for document in docs:
            body = re.sub(r'```.*?```', '', document.read_text(encoding='utf-8'), flags=re.S)
            for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', body):
                url = match.group(1).strip('<>')
                parts = urlsplit(url)
                if parts.scheme:
                    continue
                target = (document.parent / unquote(parts.path)).resolve() if parts.path else document
                with self.subTest(document=str(document.relative_to(ROOT)), link=url):
                    self.assertTrue(target.exists(), str(target))
                    if target.exists():
                        self.assertIn(target.name, [p.name for p in target.parent.iterdir()], 'Case-sensitive link')
                    if parts.fragment and target.is_file() and target.suffix == '.md':
                        target_body = target.read_text(encoding='utf-8')
                        headings = re.findall(r'^#+\s+(.+)$', target_body, flags=re.M)
                        slugs = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
                        slugs.update(re.findall(r'<a\b[^>]*\b(?:id|name)=["\']([^"\']+)["\']',
                                                target_body, flags=re.I))
                        self.assertIn(parts.fragment, slugs)

    def test_five_routes_and_all_are_retained(self):
        spec = importlib.util.spec_from_file_location('submitted_review_runner', RELEASE / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        api = release_api()
        self.assertEqual(api.SECTIONS, ('matched_histories', 'cache_interventions', 'external_state',
                                       'interpreter_checks', 'gmello_comparison'))
        for sections in [set(api.SECTIONS), *[{s} for s in api.SECTIONS]]:
            with self.subTest(sections=sections):
                tasks = module.commands(sections, Path('source'), Path('out'), 'core', 'historical', 'gmello')
                self.assertTrue(tasks)
        labels = [x[0] for x in module.commands(set(api.SECTIONS), Path('source'), Path('out'), 'core', 'historical', 'gmello')]
        for required in ('same_case_baselines', 'history_exposure', 'gmello-report_results'):
            self.assertIn(required, labels)

    def test_wrapper_verifies_then_forwards_and_resolves_output(self):
        spec = importlib.util.spec_from_file_location('collaborator_snapshot_wrapper', ROOT / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='snapshot-wrapper-') as folder:
            snapshot = Path(folder) / 'snapshot'
            snapshot.mkdir()
            for route in ['--all', *release_api().SECTIONS]:
                args = [route] if route == '--all' else ['--section', route]
                output = Path(folder) / 'intermediate' / '..' / 'result'
                with patch.object(module, 'digest', return_value='verified-manifest'), \
                     patch.object(release_api(), 'verify', return_value={'status': 'PASS'}) as verify:
                    command = module.command(snapshot, args + ['--output', str(output)])
                    verify.assert_called_once_with(snapshot)
                    self.assertEqual(command[-1], str(output.resolve()))
                    self.assertEqual(command[5:-2], args)

    def test_wrapper_help_exits_before_validation_or_execution(self):
        spec = importlib.util.spec_from_file_location('help_wrapper', ROOT / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for arguments in (['--help'], ['-h'], ['--snapshot', 'missing-snapshot', '--help']):
            with self.subTest(arguments=arguments), \
                 patch.object(module, 'digest', side_effect=AssertionError('Read an input')), \
                 patch.object(module, 'release_api', side_effect=AssertionError('Loaded the checker')), \
                 patch.object(module, 'command', side_effect=AssertionError('Started replay validation')), \
                 patch.object(module.subprocess, 'run', side_effect=AssertionError('Launched a child')):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                    module.main(arguments)
                self.assertEqual(stopped.exception.code, 0)
                for route in release_api().SECTIONS:
                    self.assertIn(route, output.getvalue())
                self.assertIn('tools/review_examples.py --all', output.getvalue())
                self.assertIn('--verify', output.getvalue())

    def test_wrapper_help_works_without_any_data_directory(self):
        with tempfile.TemporaryDirectory(prefix='review-help-') as folder:
            root = Path(folder)
            (root / 'tools').mkdir()
            shutil.copyfile(ROOT / 'reproduce.py', root / 'reproduce.py')
            shutil.copyfile(ROOT / 'tools/review_support.py', root / 'tools/review_support.py')
            result = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8',
                                     str(root / 'reproduce.py'), '--help'], cwd=root,
                                    capture_output=True, text=True, encoding='utf-8', timeout=30,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--snapshot', result.stdout)
            self.assertFalse((root / 'history_updates').exists())
            self.assertFalse((root / 'examples').exists())

    def test_wrapper_replay_still_requires_snapshot(self):
        spec = importlib.util.spec_from_file_location('required_snapshot_wrapper', ROOT / 'reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'command') as command, contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as stopped:
            module.main(['--all'])
        self.assertEqual(stopped.exception.code, 2)
        command.assert_not_called()

    def test_original_snapshot_checker_still_rejects_git_metadata(self):
        with tempfile.TemporaryDirectory(prefix='exact-snapshot-') as folder:
            root = Path(folder)
            (root / 'metadata').mkdir()
            (root / 'evidence').write_bytes(b'original')
            (root / 'metadata/files.sha256').write_text(hashlib.sha256(b'original').hexdigest() + '  evidence\n')
            self.assertEqual(release_api().verify(root)['status'], 'PASS')
            (root / '.git').mkdir()
            (root / '.git/HEAD').write_text('ref: refs/heads/review\n')
            with self.assertRaisesRegex(ValueError, 'Unexpected release file inventory'):
                release_api().verify(root)


if __name__ == '__main__':
    unittest.main()
