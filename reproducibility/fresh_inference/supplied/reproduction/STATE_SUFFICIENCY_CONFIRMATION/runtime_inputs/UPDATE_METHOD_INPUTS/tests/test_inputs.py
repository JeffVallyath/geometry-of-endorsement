import copy
from dataclasses import FrozenInstanceError, asdict
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import inputs as I
import generate_cases as G
from canonical import tasks, origins, readers, candidate
from canonical.config import MODELS
from canonical.scenes import scene_from_dict
from canonical.adapter import Backbone
from canonical.base_adapter import BoundaryError
import metrics as M

ROOT=Path(__file__).resolve().parents[1]
FIXED=ROOT/'cases/fixed'

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]

class CharTokenizer:
    """Deliberately synthetic tokenizer. Does NOT certify either real tokenizer."""
    pad_token_id=0
    eos_token_id=0
    def apply_chat_template(self,messages,**kwargs):
        return '<user>'+messages[0]['content']+'<assistant>'
    def encode(self,text,**kwargs):
        return list(text.encode('ascii'))
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])

class Cache:
    def __init__(self,length):
        self.layers=[SimpleNamespace(keys=torch.zeros(1,1,length,2),values=torch.ones(1,1,length,2))]
    def get_seq_length(self):return self.layers[0].keys.shape[-2]
    def batch_repeat_interleave(self,n):
        for layer in self.layers:
            layer.keys=layer.keys.repeat_interleave(n,0);layer.values=layer.values.repeat_interleave(n,0)

class TinyCall:
    def __init__(self):self.fail=False
    def __call__(self,input_ids,past_key_values,**kwargs):
        past_key_values.layers[0].keys.add_(2)
        if self.fail:raise RuntimeError('synthetic suffix exception')
        h=torch.zeros(*input_ids.shape,2);h[...,0]=1
        return SimpleNamespace(last_hidden_state=h)

def fake_backbone():
    bb=object.__new__(Backbone)
    bb.t=torch;bb.device=torch.device('cpu');bb.cfg=MODELS['gemma'];bb.tokenizer=CharTokenizer()
    bb.template_kwargs={};bb._prefix_cache={};bb.pad_id=0;bb.plan=None;bb.forward_count=0
    bb.cap=None;bb.W32=torch.zeros(128,2);bb.W32[89,0]=2;bb.model=SimpleNamespace(model=TinyCall())
    bb._reset()
    def compile_(ids,**kwargs):
        cache=Cache(len(ids))
        return dict(cache=cache,prefix_len=len(ids),prefix_ids=list(ids),batch=1,
                    hash=bb.cache_hash(cache),bytes=bb.cache_bytes(cache))
    bb.compile=compile_
    return bb

class InputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example=json.loads((FIXED/'example_input.json').read_text())
        cls.text=cls.example['source_text'];cls.commands=I.commands_from_json(cls.example['commands'])

    def test_every_selected_weight_hash_and_cpu_load(self):
        inv=I.inventory();self.assertEqual(sum(r['role']=='learned' for r in inv['selected']),8)
        self.assertEqual(sum(r['role']=='replay_only' for r in inv['selected']),2)
        for row in inv['selected']:
            with self.subTest(actor=row['actor'],recipe=row['recipe'],seed=row['seed']):
                actor=row['actor'];bb=SimpleNamespace(hidden=MODELS[actor]['hidden'],device=torch.device('cpu'))
                recipe=row['recipe'] if row['role']=='learned' else 'CANONICAL_REPLAY'
                bank,rec=I.load_editor(bb,actor,recipe,row['seed'])
                self.assertEqual(rec['setter_sha256'],I.file_hash(ROOT/rec['path']/'setter.npz'))
                with np.load(ROOT/rec['path']/'setter.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(bank.frozen_u.numpy(),z['basis_realized'])
                    self.assertEqual(z['head'].shape,(2,2*bb.hidden,16))
                self.assertTrue(all(not p.requires_grad for p in bank.parameters()))
                source=torch.linspace(-.2,.2,3*bb.hidden).reshape(3,bb.hidden);before=source.clone()
                for value in (0,1):
                    out=bank.fn(value,record=False)(source)
                    self.assertTrue(torch.isfinite(out).all())
                    # Independent arithmetic for BOTH setter kinds, using saved basis.
                    h=source.numpy().astype(np.float64)-bank.model.center.numpy()
                    u=bank.frozen_u.numpy().astype(np.float64);r=h-(h@u)@u.T
                    feature=r if bank.kind=='invariant' else h
                    f=np.concatenate([feature,np.broadcast_to(feature.mean(0),feature.shape)],axis=1)
                    expected=bank.model.center.numpy()+r+(f@bank.model.head[value].numpy()+bank.model.bias[value].numpy())@u.T
                    np.testing.assert_allclose(out.numpy(),expected,atol=2e-5,rtol=2e-5)
                self.assertTrue(torch.equal(source,before))

    def test_last_write_source_order_and_overlap(self):
        c=self.commands[0]
        commands=(I.AddressedUpdate(c.start,c.end,0),I.AddressedUpdate(c.start,c.end,1),I.AddressedUpdate(c.start,c.end,0))
        self.assertEqual(I.normalize(commands),(commands[-1],))
        self.assertEqual(I.normalize((I.AddressedUpdate(30,40,1),I.AddressedUpdate(10,20,0)))[0].start,10)
        with self.assertRaises(ValueError):I.normalize((I.AddressedUpdate(10,20,0),I.AddressedUpdate(19,25,1)))

    def test_noop_restoration_and_full_history(self):
        c=self.commands[0];original=int('in favor of' in self.text[c.start:c.end])
        flip=I.AddressedUpdate(c.start,c.end,1-original);restore=I.AddressedUpdate(c.start,c.end,original)
        for commands in ((restore,),(flip,restore),(flip,flip,flip,flip,restore)):
            rebuilt=I.prepare(self.text,commands,method='REBUILD')
            self.assertEqual(rebuilt.text,self.text)
            self.assertEqual(I.normalize(commands),(restore,))
            correction=I.prepare(self.text,commands,method='EXISTING_CORRECTION')
            normalized=I.prepare(self.text,commands,method='LATEST_SAME_WORDING')
            self.assertEqual(correction.text.count('Correction:'),len(commands))
            self.assertEqual(normalized.text.count('Correction:'),1)

    def test_source_master_request_and_output_schema(self):
        before=copy.deepcopy(self.example)
        context=I.prepare(self.text,self.commands,method='REBUILD')
        self.assertEqual(self.example,before)
        self.assertEqual(context.describe()['schema'],'UPDATED_CONTEXT_V1')
        self.assertEqual(context.describe()['backend'],'text_only_no_model')
        self.assertIsNone(context.artifact_sha256)
        with self.assertRaises(FrozenInstanceError):context.text='mutated'
        with self.assertRaises(TypeError):I.prepare(self.text,self.commands,method='SOURCE',gold=1)
        with self.assertRaises(ValueError):I.commands_from_json([dict(self.example['commands'][0],gold=1)])
        self.assertEqual(set(inspect.signature(I.prepare).parameters),{'source_text','commands','method','backbone','actor','seed','root'})

    def test_span_validation_and_token_rendering(self):
        bb=fake_backbone();c=self.commands[0]
        pos=bb.clause_positions(self.text,(c.start,c.end))
        self.assertEqual(len(pos),c.end-c.start)
        encoded=bb.encode(self.text,'Separate question?')
        self.assertEqual(encoded['full_ids'][:len(encoded['prefix_ids'])],encoded['prefix_ids'])
        self.assertTrue(encoded['suffix_ids'])
        with self.assertRaises(ValueError):I.prepare(self.text,(I.AddressedUpdate(c.start+1,c.end,c.value),),method='REBUILD')
        bb._prefix_cache={I.digest(self.text):[999]}
        with self.assertRaises(BoundaryError):bb.encode(self.text,'Separate question?')

    def test_cache_clone_ask_and_exception_master_immutability(self):
        bb=fake_backbone();context=I.prepare(self.text,self.commands,method='REBUILD',backbone=bb,actor='gemma')
        before=bb.cache_hash(context._artifact['cache'])
        question='Exactly unchanged question?'
        answer=I.answer(context,question,['N','Y'],backbone=bb)
        self.assertEqual(answer['question'],question);self.assertEqual(answer['schema'],'SCORED_ANSWER_V1')
        self.assertEqual(answer['prediction'],1);self.assertTrue(answer['master_unchanged'])
        self.assertEqual(before,bb.cache_hash(context._artifact['cache']))
        bb.model.model.fail=True
        with self.assertRaises(RuntimeError):I.answer(context,question,['N','Y'],backbone=bb)
        self.assertEqual(before,bb.cache_hash(context._artifact['cache']));self.assertIsNone(bb.plan)

    def test_forced_choice_not_vocabulary_argmax_validity(self):
        bb=fake_backbone();bb.W32[90,0]=5
        context=I.prepare(self.text,self.commands,method='SOURCE',backbone=bb,actor='gemma')
        result=I.answer(context,'unchanged?',['N','Y'],backbone=bb)
        self.assertFalse(result['argmax_in_labels']);self.assertTrue(result['valid'])

    def test_fixed_batch_chunk_and_same_context_alias(self):
        from serve_questions import serve_questions
        bb=fake_backbone();context=I.prepare(self.text,self.commands,method='SOURCE',backbone=bb,actor='gemma')
        qs=[dict(text=f'Question {i}?',labels=['N','Y']) for i in range(14)]
        qs.append(copy.deepcopy(qs[0]))
        rows=serve_questions(context,qs,backbone=bb)
        self.assertEqual(len(rows),15);self.assertEqual(bb.forward_count,2)
        self.assertEqual(rows[0]['batch_size'],12);self.assertEqual(rows[12]['batch_size'],2)
        self.assertEqual((rows[-1]['physical_batch'],rows[-1]['physical_slot']),(0,0))
        self.assertEqual(bb.cache_hash(context._artifact['cache']),context.artifact_sha256)

    def test_learned_full_sequence_and_replay_normalization(self):
        from canonical.writer import writer_plan
        from canonical.request import WriterInput
        from canonical.bank import canonical_program
        bb=SimpleNamespace(hidden=MODELS['gemma']['hidden'],device=torch.device('cpu'))
        bank,_=I.load_editor(bb,'gemma','INV_PAIR_NLL',0)
        request=WriterInput((1,2,3,4),((1,2),(1,2),(1,2)),(1,1,0))
        plan=writer_plan(bank,request)
        self.assertEqual(len(plan['maps']),3);self.assertEqual(plan['index'],[1,2])
        addresses,values=canonical_program(request.addresses,request.desired_values)
        self.assertEqual(values,[0]);self.assertEqual(len(addresses),1)
        with self.assertRaises(TypeError):writer_plan(bank,dict(prefix_ids=[1,2],gold=1))

    def test_missing_or_corrupt_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):I.inventory(temp)
            root=Path(temp);(root/'weights').mkdir()
            record=copy.deepcopy(I.inventory()['selected'][0]);record['path']='weights/one'
            (root/'weights/SELECTED.json').write_text(json.dumps({'selected':[record]}))
            bb=SimpleNamespace(hidden=MODELS[record['actor']]['hidden'],device=torch.device('cpu'))
            with self.assertRaises(FileNotFoundError):I.load_editor(bb,record['actor'],record['recipe'],record['seed'],root)
            (root/'weights/one').mkdir();(root/'weights/one/setter.npz').write_bytes(b'not the selected weights')
            with self.assertRaises(ValueError):I.load_editor(bb,record['actor'],record['recipe'],record['seed'],root)

    def test_external_and_learned_no_model_fail_explicitly(self):
        with self.assertRaises(NotImplementedError):I.prepare(self.text,self.commands,method='FIELD_PLUS_LATEST_ERRATUM')
        with self.assertRaises(ValueError):I.prepare(self.text,self.commands,method='INV_PAIR_NLL')

class PopulationTests(unittest.TestCase):
    def test_pre_evaluation_gate_json_roundtrip(self):
        from verify_package import verify_fixed_populations
        audit=verify_fixed_populations(ROOT)
        self.assertEqual(audit['remaining_collisions'],0)
        self.assertEqual(audit['fresh_unique_sources'],328)

    def test_fresh_complete_regeneration_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            result=G.export(Path(temp)/'fixed')
            saved=json.loads((FIXED/'MANIFEST.json').read_text())
            self.assertEqual(result,saved)
            for name,row in saved['files'].items():
                self.assertEqual(I.file_hash(FIXED/name),row['sha256'])
                self.assertEqual(I.file_hash(Path(temp)/'fixed'/name),row['sha256'])
        self.assertEqual(result['fresh_unique_sources'],328);self.assertEqual(result['remaining_collisions'],0)

    def test_balance_and_lowest_hash_program_subset(self):
        pops,audit=G.build(ROOT/'cases/PRIOR_SOURCE_HASHES.json')
        self.assertEqual([len(pops[k]) for k in ('main','origin','program','dev')],[64,64,32,8])
        for row in pops['program']:
            s=scene_from_dict(row)
            eligible=[scene_from_dict(r) for r in pops['main'] if G.cell(scene_from_dict(r))==G.cell(s)]
            self.assertEqual(candidate.semantic_hash(s),min(candidate.semantic_hash(v) for v in eligible))

    def test_all_source_terminal_facts_and_text_commands(self):
        pops,_=G.build(ROOT/'cases/PRIOR_SOURCE_HASHES.json')
        for rec in pops['main']:
            s=scene_from_dict(rec);source=asdict(s)
            for program,commands in tasks.programs(s).items():
                terminal=tasks.OLD.apply_program(s,commands)
                for actor,proc in [('gemma','C1'),('qwen','T4')]:
                    text,spans=tasks.prefix_text(s,proc,'early')
                    cmd=tuple(I.AddressedUpdate(*spans[(e.actor,e.project)],e.value) for e in commands)
                    self.assertEqual(I.prepare(text,cmd,method='REBUILD').text,tasks.prefix_text(s,proc,'early',world=terminal)[0])
                    self.assertEqual(I.prepare(text,cmd,method='EXISTING_CORRECTION').text,tasks.prefix_text(s,proc,'early',corrections=commands)[0])
                self.assertEqual(asdict(s),source)
                touched={(e.actor,e.project) for e in commands}
                self.assertTrue(all(s.values[a][p]==terminal.values[a][p] for a in range(len(s.actors)) for p in range(len(s.projects)) if (a,p) not in touched))

    def test_unchanged_questions_all_origins_and_independent_labels(self):
        for row in read_rows(FIXED/'origin.jsonl'):
            terminal=scene_from_dict(row['terminal']);shared=None
            for source in row['sources']:
                scene=scene_from_dict(source)
                self.assertEqual(tasks.OLD.apply_program(scene,[tasks.OLD.Edit(**e) for e in row['commands']]).values,terminal.values)
                qs=G.queries(scene,terminal,'SET_TERMINAL_ABC',True)
                strings=[(q['text'],q['labels']) for q in qs]
                self.assertEqual(len(qs),34)
                if shared is not None:self.assertEqual(shared,strings)
                shared=strings
                for q in qs:
                    self.assertEqual(q['gold'],origins.truth(terminal.values,q['spec']))
                    self.assertEqual(q['gold'],tasks.OLD.evaluate(terminal,q['spec']))
                    self.assertEqual(q['source_gold'],origins.truth(scene.values,q['spec']))

class MetricTests(unittest.TestCase):
    def test_origin_unit_34x4_and_disagreement(self):
        p=np.ones((4,34),dtype=int);y=np.ones(34,dtype=int)
        p[3,33]=0;m=origins.origin_metrics(p,y)
        self.assertEqual(m['all_origin_bundle_correct'],0)
        self.assertAlmostEqual(m['mean_accuracy'],135/136)
        self.assertAlmostEqual(m['all_origin_question_correct'],33/34)
        self.assertAlmostEqual(m['origin_disagreement'],1/34)
        with self.assertRaises(ValueError):origins.origin_metrics(p[:3],y)

    def test_invariant_harm_and_conditional_denominators(self):
        rows=[dict(gold=1,source_gold=1,source_prediction=s,prediction=p,valid=True,source_valid=True) for s,p in [(1,0),(1,1),(0,0),(0,1)]]
        m=M.origin_rates(rows)
        self.assertEqual(m['harm'],1/4);self.assertEqual(m['conditional_harm'],1/2)
        self.assertEqual(m['invariant_questions'],4);self.assertEqual(m['source_correct_invariant_questions'],2)
        self.assertIsNone(m['changed_accuracy'])

    def test_primary_seeds_within_64_roots_baseline_once(self):
        roots={a:{f'r{i}' for i in range(64)} for a in ('gemma','qwen')};vectors={}
        for a in roots:
            vectors[(a,M.BASELINE,None)]={r:0 for r in roots[a]}
            for recipe in I.RECIPES:
                for s in (0,1):vectors[(a,recipe,s)]={r:s for r in roots[a]}
        result=M.paired_primary(vectors,roots)
        self.assertEqual(len(result),4)
        for r in result:
            self.assertEqual(r['effect'],.5);self.assertEqual(r['roots'],64);self.assertEqual(r['level'],.9875)
            self.assertEqual(r['draws'],10000);self.assertEqual(r['per_seed']['0']['effect'],0)
        bad=copy.deepcopy(vectors);bad[('gemma',M.BASELINE,0)]=bad[('gemma',M.BASELINE,None)]
        with self.assertRaises(ValueError):M.paired_primary(bad,roots)
        bad=copy.deepcopy(vectors);bad[('gemma',I.RECIPES[0],0)].pop('r0')
        with self.assertRaises(ValueError):M.paired_primary(bad,roots)

    def test_scoring_gold_only_after_answer_and_source_question_identity(self):
        q={'text':'Q','gold':1};a=dict(question='Q',prediction=1,logps=[-2.,-1.],valid=True,answer_mass=.5,argmax_in_labels=True,tie=False)
        self.assertTrue(M.label_answer(a,q,a)['correct'])
        with self.assertRaises(ValueError):M.label_answer(dict(a,question='changed'),q,a)

    def test_all24_not_mean_accuracy_and_program_coverage(self):
        s=read_rows(FIXED/'main.jsonl')[0]
        qs=G.queries(scene_from_dict(s),None,'single')
        rs=[dict(q,actor='gemma',condition='TEST',scene_id=s['scene_id'],order='early',program='single',
                 prediction=q['gold'],source_prediction=q['source_gold'],valid=True,source_valid=True) for q in qs]
        rs[-1]['prediction']=1-rs[-1]['gold']
        m=M.scene_summary(rs,{s['scene_id']:s})[('gemma','TEST','single')]
        self.assertEqual(m['all24'],0);self.assertAlmostEqual(m['accuracy'],23/24);self.assertEqual(m['n_scenes'],1)

    def test_root_statistics_missing_question_and_duplicate_origin_fail(self):
        qs=[r for r in read_rows(FIXED/'evaluation.jsonl') if r['panel']=='ORIGIN' and r['actor']=='gemma' and r['root_id']=='UPDATE_COMPARISON-ORIGIN-0000']
        rows=[dict(q,condition='TEST_s0',seed=0,suffix_ids=[q['text']],prediction=q['gold'],
                   logps=[-1.,-1.5],valid=True,source_prediction=q['source_gold'],source_valid=True) for q in qs]
        result=M.root_statistics(rows)
        self.assertEqual(result['all34']['all_origin_bundle_correct'],1)
        self.assertEqual(result['all34']['questions'],34)
        with self.assertRaises(ValueError):M.root_statistics(rows[:-1])
        with self.assertRaises(ValueError):M.root_statistics(rows+[rows[0]])

    def test_score_case_joins_only_after_context_and_retains_source(self):
        example=json.loads((FIXED/'example_input.json').read_text())
        bb=fake_backbone();context=I.prepare(example['source_text'],I.commands_from_json(example['commands']),method='SOURCE',backbone=bb,actor='gemma')
        rows=[dict(input_id='test',query_id='q1',actor='gemma',panel='DEV',text='Question?',labels=['N','Y'],gold=1,source_gold=0)]
        result=M.score_case(context,rows,None,backbone=bb)
        self.assertEqual(result[0]['condition'],'SOURCE');self.assertIsNone(result[0]['seed'])
        self.assertTrue(result[0]['correct']);self.assertEqual(result[0]['source_prediction'],1)
        self.assertIsNotNone(result[0]['suffix_ids'])
        with self.assertRaises(ValueError):M.score_case(context,rows,None,backbone=bb,seed=0)

    def test_cpu_import_closure(self):
        for p in (ROOT/'canonical').glob('*.py'):
            importlib.import_module('canonical.'+p.stem)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
