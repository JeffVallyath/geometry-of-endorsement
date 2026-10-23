"""Small synthetic regressions for the paper-oriented release interface."""
import hashlib
import ast
import gzip
import json
import lzma
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments._shared import release
from experiments._shared import transport
from experiments._shared import record_counts
import reproduce

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'repo';self.root.mkdir()
        (self.root/'metadata').mkdir()
        self.input=self.root/'observation.json';self.input.write_bytes(b'{"correct":false}\n')
        self.bind()
    def bind(self):
        (self.root/'metadata/files.sha256').write_text(release.digest(self.input)+'  observation.json\n',encoding='utf-8')
    def test_unchanged_union_verifies(self):
        self.assertEqual(release.verify(self.root)['status'],'PASS')
    def test_tampered_observation_rejected(self):
        self.input.write_bytes(b'{"correct":true}\n')
        with self.assertRaises(ValueError):release.verify(self.root)
    def test_missing_observation_rejected(self):
        self.input.unlink()
        with self.assertRaises(ValueError):release.verify(self.root)
    def test_undeclared_observation_rejected(self):
        (self.root/'extra').write_text('extra')
        with self.assertRaises(ValueError):release.verify(self.root)
    def test_duplicate_manifest_member_rejected(self):
        p=self.root/'metadata/files.sha256';p.write_text(p.read_text()*2)
        with self.assertRaises(ValueError):release.verify(self.root)
    def test_unsafe_paths_rejected(self):
        for name in ('../escape','/absolute','a/../b','a\\b','a:stream','a//b','tail.','trail ','CON','sub/AUX.json','LPT1.txt'):
            with self.subTest(name=name),self.assertRaises(ValueError):release.safe(self.root,name)
    def test_existing_output_preserved(self):
        with self.assertRaises(FileExistsError):release.fresh_output(self.root,self.root)
        self.assertEqual(self.input.read_bytes(),b'{"correct":false}\n')
    def test_output_cannot_be_inside_inputs(self):
        with self.assertRaises(ValueError):release.fresh_output(self.root/'new',self.root)
    def test_deduplication_materializes_portable_equal_copies(self):
        row={'included':'observation.json','sha256':release.digest(self.input)}
        index={'mounts':{'one':{'a.json':row,'b.json':row}}}
        out=Path(self.tmp.name)/'runtime with spaces'
        release.mount(index,out,self.root)
        self.assertEqual((out/'one/a.json').read_bytes(),(out/'one/b.json').read_bytes())
        self.assertFalse((out/'one/a.json').is_symlink())
        (out/'one/a.json').write_bytes(b'changed output')
        self.assertEqual(self.input.read_bytes(),b'{"correct":false}\n')
    def test_resolver_rejects_changed_canonical_input(self):
        row={'included':'observation.json','sha256':'0'*64}
        with self.assertRaises(ValueError):release.mount({'mounts':{'one':{'a.json':row}}},Path(self.tmp.name)/'out',self.root)
    def test_all_includes_recent_studies_and_corrected_comparison(self):
        tasks=reproduce.commands(set(release.SECTIONS),Path('source'),Path('output'),'core','historical','gmello')
        labels=[t[0] for t in tasks]
        for required in ('history_exposure','same_case_baselines','gmello-replay','gmello-evaluate','gmello-report_results','refresh','primary'):
            self.assertIn(required,labels)
        for excluded in ('relation','consequences','textual-breadth','update-methods','ripple'):
            self.assertNotIn(excluded,labels)
    def test_separate_historical_environment_required(self):
        with self.assertRaises(ValueError):reproduce.commands({'matched_histories'},Path('source'),Path('output'),'core',None,None)
    def test_each_historical_package_uses_its_own_repro_namespace(self):
        legacy=self.root/'legacy';(legacy/'src/repro').mkdir(parents=True)
        (legacy/'src/repro/common.py').write_text('ROOT = None\n')
        inherited={'PYTHONPATH':os.pathsep.join(['offline-guard','wrapper-src'])}
        actual=release.child_environment(legacy,inherited)
        self.assertEqual(actual['PYTHONPATH'].split(os.pathsep),['offline-guard',str(legacy/'src')])
        self.assertEqual(inherited['PYTHONPATH'],os.pathsep.join(['offline-guard','wrapper-src']))
    def transport_fixture(self):
        payload=b'{"answer":"wrong","control":false,"tokens":[1,2,3]}\n'*200
        packed=self.root/'records.xz';packed.write_bytes(lzma.compress(payload))
        normalized=b''.join(transport.gzip_chunks([payload]))
        return payload,packed,{'included':'records.xz','sha256':release.digest(packed),
            'codec':'xz-to-gzip-v1','decoded_sha256':hashlib.sha256(payload).hexdigest(),
            'decoded_bytes':len(payload),'materialized_sha256':hashlib.sha256(normalized).hexdigest(),
            'materialized_bytes':len(normalized)}
    def test_transport_restores_all_record_bytes_and_anonymous_header(self):
        payload,packed,row=self.transport_fixture()
        output=Path(self.tmp.name)/'transport output'
        release.mount({'mounts':{'study':{'records.jsonl.gz':row}}},output,self.root)
        restored=output/'study/records.jsonl.gz'
        self.assertEqual(gzip.decompress(restored.read_bytes()),payload)
        self.assertEqual(restored.read_bytes()[:10],transport.HEADER)
    def test_transport_rejects_wrong_decoded_hash(self):
        payload,packed,row=self.transport_fixture();row['decoded_sha256']='0'*64
        with self.assertRaises(ValueError):transport.restore(packed,Path(self.tmp.name)/'bad.gz',row)
    def test_transport_rejects_wrong_materialized_hash(self):
        payload,packed,row=self.transport_fixture();row['materialized_sha256']='0'*64
        with self.assertRaises(ValueError):transport.restore(packed,Path(self.tmp.name)/'bad.gz',row)
    def test_transport_is_independent_of_read_chunk_boundaries(self):
        payload,packed,row=self.transport_fixture()
        one=b''.join(transport.gzip_chunks([payload]))
        many=b''.join(transport.gzip_chunks([payload[i:i+31] for i in range(0,len(payload),31)]))
        self.assertEqual(one,many)
    def test_transport_rejects_unknown_codec(self):
        payload,packed,row=self.transport_fixture();row['codec']='unrecognized'
        with self.assertRaises(ValueError):transport.restore(packed,Path(self.tmp.name)/'bad.gz',row)
    def test_timeout_is_recorded_and_deadline_propagated(self):
        output=Path(self.tmp.name)/'run';(output/'_runtime').mkdir(parents=True)
        env={'PYTHONPATH':'guard','REPRO_DEADLINE_UTC':'2099-01-01T00:00:00+00:00'}
        with patch.object(release.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',1)) as call:
            result=release.run_step('fixture','python',[],self.root,output,env,timeout=1)
        self.assertEqual(result['exit_code'],124)
        self.assertEqual(result['status'],'FAIL')
        self.assertTrue((output/result['diagnostics']).is_file())
        self.assertLess(call.call_args.kwargs['env']['REPRO_DEADLINE_UTC'],env['REPRO_DEADLINE_UTC'])
    def test_training_counts_distinguish_repeated_pairs_from_presentations(self):
        path=Path(self.tmp.name)/'training.jsonl.gz'
        rows=[dict(sample_id=str(i),pair_id='same',question_id='q',source_root='case') for i in range(2)]
        with gzip.open(path,'wt',encoding='utf-8') as output:
            for row in rows:output.write(json.dumps(row)+'\n')
        result=record_counts.training_counts(path)
        self.assertEqual(result,{'draws_per_fit':2,'unique_candidate_pairs':1,'unique_presented_questions':1,'source_cases':1})
        with gzip.open(path,'at',encoding='utf-8') as output:output.write(json.dumps(rows[0])+'\n')
        with self.assertRaises(ValueError):record_counts.training_counts(path)
    def test_token_comparison_rejects_missing_and_duplicate_arms(self):
        path=Path(self.tmp.name)/'tokens.jsonl.gz'
        rows=[dict(case_id=1,history='H0',query_id='q',candidate_sha256='candidate',arm=arm,n_prompt=n)
              for arm,n in [('ISO',10),('HIST',14),('CURRENT_EXTRA',14)]]
        def write(values):
            with gzip.open(path,'wt',encoding='utf-8') as output:
                for row in values:output.write(json.dumps(row)+'\n')
        write(rows);result=record_counts.token_matching(path)
        self.assertEqual(result['paired_candidate_contexts'],1)
        self.assertEqual(result['HIST_and_CURRENT_EXTRA_equal_token_lengths'],1)
        write(rows[:-1])
        with self.assertRaises(ValueError):record_counts.token_matching(path)
        write(rows+[rows[0]])
        with self.assertRaises(ValueError):record_counts.token_matching(path)

    def test_chronology_ordinals_preserve_ties_and_causal_order(self):
        source=release.ROOT/'experiments/external_state/v3/studies/repair_qwen7/controller/evaluate.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='event_order')
        scope={};exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),scope)
        event=scope['event_order']
        self.assertEqual(event({'completed_event_order':7},'completed'),7)
        self.assertLessEqual(event({'completed_event_order':7},'completed'),event({'started_event_order':7},'started'))
        self.assertLess(event({'completed_event_order':7},'completed'),event({'started_event_order':8},'started'))
        self.assertEqual(event({'completed_utc':'original-order-string'},'completed'),'original-order-string')

if __name__=='__main__':unittest.main()
