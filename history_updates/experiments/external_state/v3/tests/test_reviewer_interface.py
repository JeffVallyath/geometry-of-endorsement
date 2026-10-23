"""Contract checks for the visible-input reviewer interface."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from repro.file_manifest import InputStore,EvidenceError,MissingEvidence,inside
from repro import __main__ as cli


class VisibleInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.input=self.root/'studies/example/input.json'
        self.input.parent.mkdir(parents=True);self.input.write_bytes(b'{"result":false}\n')
        (self.root/'provenance').mkdir();self.bind()

    def bind(self):
        raw=self.input.read_bytes()
        manifest={'schema':'REVIEWER_VISIBLE_INPUTS_V1','studies':{'example':{'root':'studies/example',
            'files':{'input.json':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}}}
        path=self.root/'provenance/INPUTS.json';path.write_text(json.dumps(manifest),encoding='utf8')
        raw=path.read_bytes()
        (self.root/'provenance/INPUTS_BINDING.json').write_text(json.dumps({'file':'provenance/INPUTS.json',
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}),encoding='utf8')

    def test_visible_files_verified_without_restoration_or_mutation(self):
        before=self.input.read_bytes();source,result=InputStore(self.root).verify('example')
        self.assertEqual(source,self.input.parent);self.assertEqual(result['status'],'PASS')
        self.assertEqual(self.input.read_bytes(),before);self.assertFalse((self.root/'_inputs').exists())

    def test_missing_observation_fails(self):
        self.input.unlink()
        with self.assertRaises(MissingEvidence):InputStore(self.root).verify('example')

    def test_changed_observation_fails(self):
        self.input.write_bytes(b'{"result":true}\n')
        with self.assertRaises(EvidenceError):InputStore(self.root).verify('example')

    def test_extra_input_fails(self):
        (self.input.parent/'extra.json').write_bytes(b'{}')
        with self.assertRaises(EvidenceError):InputStore(self.root).verify('example')

    def test_changed_manifest_fails_before_input_use(self):
        (self.root/'provenance/INPUTS.json').write_bytes(b'{}')
        with self.assertRaises(EvidenceError):InputStore(self.root)

    def test_unknown_study_fails(self):
        with self.assertRaises(MissingEvidence):InputStore(self.root).verify('absent')

    def test_unsafe_and_ambiguous_paths_fail(self):
        for name in ('../escape','/absolute','a/../b','a\\b','a:stream','a//b','tail.','trail '):
            with self.subTest(name=name),self.assertRaises(EvidenceError):inside(self.root,name)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_existing_empty_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):cli.fresh_output(self.root)

    def test_output_cannot_target_inputs_or_ancestor(self):
        repo=self.root/'repository';repo.mkdir()
        with patch.object(cli,'ROOT',repo):
            with self.assertRaises(ValueError):cli.fresh_output(repo/'studies/new')
            with self.assertRaises(FileExistsError):cli.fresh_output(self.root)

    def test_unsupported_route_is_explicit_and_never_successful(self):
        with patch.object(cli.subprocess,'run') as process:
            result=cli.execute(('unavailable_history',),self.root/'new output')
        process.assert_not_called();self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['steps'][0]['status'],'NOT_RUN')

    def test_process_failure_wins_over_child_pass_receipt(self):
        output=self.root/'out';output.mkdir();child=output/'repair_qwen32';child.mkdir()
        (child/'RECONSTRUCTION_RECEIPT.json').write_text('{"status":"PASS"}')
        store=SimpleNamespace(verify=lambda identifier:(self.root,{'status':'PASS'}))
        with patch.object(cli.subprocess,'run',return_value=SimpleNamespace(returncode=9,stdout='failure')):
            result=cli.step('repair_qwen32',output,store)
        self.assertEqual(result['status'],'FAIL')

    def test_zero_exit_without_scientific_receipt_fails(self):
        output=self.root/'out';output.mkdir()
        store=SimpleNamespace(verify=lambda identifier:(self.root,{'status':'PASS'}))
        with patch.object(cli.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')):
            result=cli.step('repair_qwen32',output,store)
        self.assertEqual(result['status'],'FAIL');self.assertEqual(result['scientific_receipt_status'],'MISSING')

    def test_batch_stops_after_failure(self):
        with patch.object(cli,'InputStore',return_value=object()),patch.object(cli,'step',return_value={
                'study':'repair_qwen32','status':'FAIL'}) as step:
            result=cli.execute(('repair_qwen32','repair_gemma27'),self.root/'new')
        self.assertEqual(step.call_count,1);self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['steps'][-1]['status'],'NOT_RUN')

    def test_historical_environment_is_explicit(self):
        with self.assertRaises(MissingEvidence):cli.step('historical_paper',self.root,None)


if __name__=='__main__':unittest.main()
