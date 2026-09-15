"""Fresh score-based reconstruction with adversarial provenance/semantic joins."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reproducibility/state_sufficiency/independent_verification'
spec=importlib.util.spec_from_file_location('independent_v10',BASE/'reconstruct.py')
iv=importlib.util.module_from_spec(spec);spec.loader.exec_module(iv)


@pytest.fixture(scope='module')
def independent_result():
    original=iv.load
    def blind_load(path):
        assert 'expected' not in path.parts, 'Calculation opened reported results'
        return original(path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(iv,'load',blind_load)
        return iv.analyze(BASE)


def test_independent_calculation_reproduces_every_reported_classification(independent_result):
    report=iv.compare(independent_result,BASE)
    assert report['primary_cells']==8
    assert report['joint_query_rows']==16128
    assert report['root_rows']==896
    assert report['audit']['argmax_certified']==230656
    assert report['audit']['semantic_final_tables']==512
    assert report['audit']['aliases']==17408
    assert report['repeatability']=={'fresh_score_rows':8448,'pairwise_comparisons':12672,'max_logp_difference':0.0}


def test_committed_headlines_are_separate_and_match(independent_result):
    original=json.loads((BASE.parent/'v10/primary_results.json').read_text())
    for saved in original:
        result=next(r for r in independent_result['primary'] if (r['actor'],r['group'])==(saved['actor'],saved['group']))
        assert result['root_ids']==saved['root_ids']
        for key in ('mean','root_scores','corrected_interval','root_witness_prevalence'):
            assert np.allclose(saved[key],result[key],atol=1e-12,rtol=0)


@pytest.fixture
def panel():
    rid='SSC1-FINAL-0006'
    design=iv.load(BASE/f'inputs/designs/{rid}.json')
    data=iv.load(BASE/f'inputs/responses/qwen/{rid}/core.json.gz')
    qs=iv.validate_design(design,rid);protocol=iv.load(BASE/'inputs/protocol.json')
    return design,data,qs,protocol


@pytest.mark.parametrize('mutation',['duplicate_physical','duplicate_logical','missing','alias','tokens','question_id','score','prediction_flag','cache','suffix','method'])
def test_invalid_response_joins_fail_closed(panel,mutation):
    design,data,qs,protocol=panel
    if mutation=='duplicate_physical':data['physical'].append(copy.deepcopy(data['physical'][0]))
    elif mutation=='duplicate_logical':data['bindings'].append(copy.deepcopy(data['bindings'][0]))
    elif mutation=='missing':data['bindings'].pop()
    elif mutation=='alias':data['bindings'][-1]['physical_id']='NATIVE_FINAL|missing'
    elif mutation=='tokens':data['physical'][0]['tokens']['question_sha256']='0'*64
    elif mutation=='question_id':data['physical'][0]['query_id']='absent'
    elif mutation=='score':data['physical'][0]['logps']=[-100,-101]
    elif mutation=='prediction_flag':data['physical'][0]['recorded_checks']['prediction']=1-data['physical'][0]['recorded_checks']['prediction']
    elif mutation=='cache':data['contexts']['o0/SOURCE']['cache_hash']='0'*64
    elif mutation=='method':data['contexts']['o0/SOURCE']['method']='INV_PAIR_NLL'
    else:
        data['physical'][0]['tokens']['suffix_ids'][0]+=1
        data['physical'][0]['recorded_checks']['tokens_sha256']=iv.canonical(data['physical'][0]['tokens'])
    with pytest.raises((ValueError,KeyError)):iv.index_panel(data,design,qs,'core',protocol)


@pytest.mark.parametrize('mutation',['final','starting','mapping','wording','duplicate_question','source_text','command_span'])
def test_semantics_and_fixed_question_bindings_fail_closed(panel,mutation):
    design,_,_,_=panel
    if mutation=='final':design['root']['terminal']['values'][0][0]^=1
    elif mutation=='starting':design['root']['sources'][1]=copy.deepcopy(design['root']['sources'][0])
    elif mutation=='mapping':design['questions'][17]['labels'].reverse()
    elif mutation=='wording':design['questions'][2]['text']=design['questions'][0]['text']
    elif mutation=='duplicate_question':design['questions'][-1]=copy.deepcopy(design['questions'][0])
    elif mutation=='source_text':design['source_inputs'][0]['source_text']=design['source_inputs'][1]['source_text']+'\nAltered'
    elif mutation=='command_span':design['source_inputs'][0]['commands'][0]['start']+=1
    with pytest.raises(ValueError):iv.validate_design(design,'SSC1-FINAL-0006')


def test_hashes_are_checked_before_analysis(tmp_path):
    (tmp_path/'inputs').mkdir();target=tmp_path/'inputs/protocol.json';target.write_text('{}')
    row={'path':'inputs/protocol.json','role':'input','bytes':2,'sha256':'0'*64}
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'files':[row]}))
    with pytest.raises(ValueError,match='hash mismatch'):iv.check_manifest(tmp_path)


def test_missing_seed_or_root_cannot_change_denominator(independent_result):
    reduced=[r for r in independent_result['root_rows'] if not (r['actor']=='gemma' and r['condition']=='INV_PAIR_NLL_s1')]
    with pytest.raises(ValueError,match='root coverage'):iv.summarize(reduced,iv.load(BASE/'inputs/protocol.json'))


def test_inputs_contain_no_precomputed_primary_classifications():
    manifest=iv.check_manifest(BASE)
    assert len({r['path'] for r in manifest['files']})==len(manifest['files'])
    for row in manifest['files']:
        assert row['sources'] and row['row_counts'] and row['transformation']
        assert all(len(s['sha256'])==64 and s['member'] and s['archive'] for s in row['sources'])
        if '/responses/' in row['path']:
            panel=iv.load(BASE/row['path'])
            assert row['row_counts']=={'physical_responses':len(panel['physical']),'logical_responses':len(panel['bindings']),'contexts':len(panel['contexts'])}
            assert set(panel)=={'model','root_id','phase','contexts','aliases','physical','bindings'}
            for saved in panel['physical']:
                assert not {'W','A','C','D','strong','witness','gold','eligible'} & saved.keys()


def test_no_original_analysis_imports():
    tree=ast.parse((BASE/'reconstruct.py').read_text())
    imported=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imported.update(a.name.split('.')[0] for a in node.names)
        if isinstance(node,ast.ImportFrom):imported.add(node.module.split('.')[0])
    assert imported <= {'__future__','argparse','collections','csv','gzip','hashlib','itertools','json','math','pathlib','re','numpy'}


def test_examples_cover_witness_nonwitness_and_each_exclusion(independent_result):
    examples=iv.load(BASE/'examples/examples.json')
    assert {e['category'] for e in examples}=={'repeat_checked_positive','textual_positive','eligible_non_witness','excluded_direct_operand','excluded_native_reference','excluded_noop_reference'}
    index={(r['actor'],r['root_id'],r['condition'],r['query_id']):r for r in independent_result['query_rows']}
    for e in examples:
        r=index[e['model'],e['root_id'],e['method'],e['query_id']]
        assert e['calculated_flags']=={k:r[k] for k in e['calculated_flags']}
        assert len(e['histories'])==8
        for h in e['histories']:
            assert len(h['direct_operands'])==2 and h['joint']['question']==e['histories'][0]['joint']['question']
            for response in h['direct_operands']+[h['joint']]:
                assert response['logps'] and response['journal_line']>0 and response['physical_id']
    assert len(examples[0]['repeat_joint_rows'])==16


def test_example_derivative_hashes_and_source_bindings():
    manifest=iv.check_manifest(BASE,role='example')
    assert len([r for r in manifest['files'] if r['role']=='example'])==2


def test_example_generation_uses_checkout_stable_lf_newlines():
    # Both the existing artifacts and future generator writes must preserve LF.
    for name in ('examples.json','EXAMPLES.md'):
        assert b'\r' not in (BASE/'examples'/name).read_bytes()
    tree=ast.parse((BASE/'make_examples.py').read_text(encoding='utf8'))
    writes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text']
    assert len(writes)==2
    assert all(any(k.arg=='newline' and isinstance(k.value,ast.Constant) and k.value.value=='\n' for k in n.keywords) for n in writes)
