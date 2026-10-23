import copy
import unittest
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from repro.update_tables import ENDPOINTS, LABELS, matched_snapshots, exposure_errors, check_baseline_coverage
from repro import __main__ as cli
from repro.privacy import inspect_text_container
from repro.portable_a10 import VerifiedDirectoryArchive, SOURCE_HASH


def snapshots():
    return [dict(arm=a, case_id='synthetic_subject', query_id='J0', history='H0',
                 phase=p, branch='J', question='Synthetic question?', correct=(p=='post'),
                 expected_entity='synthetic_old' if p=='pre' else 'synthetic_new',
                 outcome={'status':'VALID', 'entity':'synthetic_new'})
            for a in ENDPOINTS for p in ('pre','post')]


def exposure():
    return [dict(seed=s, arm=a, case_id='synthetic_subject', query_id='J0', history=h,
                 correct=a!='HIST', refused=a=='HIST', selected=None if a=='HIST' else {'program':'synthetic'},
                 outcome={'entity':None if a=='HIST' else 'synthetic_new'})
            for s in ('seed0','seed1') for a in ('ISO','HIST','CURRENT_EXTRA')
            for h in ('H0','H1','S0','S1','C','C+')]


class UpdateHelpers(unittest.TestCase):
    def test_matched_population_and_exploratory_join(self):
        result=matched_snapshots(snapshots(),expected_reads=1)
        self.assertEqual(set(result),set(ENDPOINTS))
        self.assertEqual(result['QA_0']['exploratory']['current_entity_on_original_input'],1)

    def test_program_cannot_be_omitted(self):
        with self.assertRaises(ValueError):
            matched_snapshots([r for r in snapshots() if not r['arm'].startswith('PROGRAM')],expected_reads=1)

    def test_missing_seed_fails(self):
        with self.assertRaises(ValueError):
            matched_snapshots([r for r in snapshots() if r['arm']!='QA_1'],expected_reads=1)

    def test_equal_counts_do_not_prove_a_match(self):
        data=snapshots();data[0]['query_id']='different'
        with self.assertRaises(ValueError):matched_snapshots(data,expected_reads=1)

    def test_changed_question_fails(self):
        data=snapshots();data[0]['question']='different'
        with self.assertRaises(ValueError):matched_snapshots(data,expected_reads=1)

    def test_duplicate_question_fails(self):
        data=snapshots();data.append(copy.deepcopy(data[0]))
        with self.assertRaises(ValueError):matched_snapshots(data,expected_reads=1)

    def test_refusal_is_not_a_wrong_entity(self):
        result=exposure_errors(exposure(),expected_queries=1)['seed0/HIST-minus-ISO']
        self.assertEqual(result['four_history_refusals'],4)
        self.assertEqual(result['four_history_wrong_entity'],0)
        self.assertEqual(result['robust_question_harms'],1)

    def test_wrong_entity_is_not_a_refusal(self):
        data=exposure()
        for row in data:
            if row['arm']=='HIST':
                row.update(refused=False,selected={'program':'synthetic_wrong'},outcome={'entity':'wrong'})
        result=exposure_errors(data,expected_queries=1)['seed0/HIST-minus-ISO']
        self.assertEqual(result['four_history_refusals'],0)
        self.assertEqual(result['four_history_wrong_entity'],4)

    def test_exposure_missing_seed_fails(self):
        with self.assertRaises(ValueError):
            exposure_errors([r for r in exposure() if r['seed']=='seed0'],expected_queries=1)

    def test_exposure_missing_reference_fails(self):
        with self.assertRaises(ValueError):exposure_errors(exposure()[:-1],expected_queries=1)

    def test_quick_scope_preserved(self):
        self.assertEqual(cli.QUICK,('original_public_history','paired_native_scale','repair_qwen7','repair_qwen32','repair_gemma27'))
        self.assertEqual(cli.UPDATES,('history_exposure','same_case_baselines'))

    def test_both_program_selection_routes_execute(self):
        self.assertIn('exact execution',LABELS['REPAIR'])
        self.assertIn('same exact executor',LABELS['PROGRAM'])
        self.assertIn('answer generation',LABELS['QA'])

    def test_new_operational_field_inside_compression(self):
        for field in ('task_id','threadId','pod_id','instance_id','contract_id','resource_id','ssh_host'):
            payload=json.dumps({field:'SYNTHETIC_PRIVATE_VALUE'}).encode()
            result=inspect_text_container(gzip.compress(payload,mtime=0),'synthetic.json.gz')
            self.assertTrue(result,field)

    def test_wrong_a10_source_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'A10_BINDING.json').write_text(json.dumps({'schema':'A10_EXTRACTED_DEPENDENCIES_V1',
                'source_archive_sha256':'wrong','files':{}}))
            with self.assertRaises(ValueError):VerifiedDirectoryArchive(root)

    def test_missing_a10_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'A10_BINDING.json').write_text(json.dumps({'schema':'A10_EXTRACTED_DEPENDENCIES_V1',
                'source_archive_sha256':SOURCE_HASH,'files':{}}))
            with self.assertRaises(ValueError):VerifiedDirectoryArchive(root)

    def test_primary_control_denominator_is_not_target_denominator(self):
        scientific={'endpoints':{arm:{'subjects':128,'targets':384,'controls':384} for arm in ENDPOINTS},
                    'primary_contrasts':{k:{} for k in ('QA','PROGRAM','MELLO_CURRENT','GMELLO_LOCAL')}}
        with self.assertRaises(ValueError):check_baseline_coverage(scientific)


if __name__=='__main__':unittest.main()
