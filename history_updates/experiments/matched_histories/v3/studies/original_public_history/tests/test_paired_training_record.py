"""Static provenance and model-free checks of recovered paired-training records."""
import ast
import json
import random

from repro.common import ROOT, sha256

BASE = ROOT / 'reproducibility/learned_editors'
REFERENCE = BASE / 'training_reference'


def definitions(names, **bindings):
    tree = ast.parse((REFERENCE / 'source/rtc7_common.py.txt').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<recovered-pure-functions>', 'exec'), namespace)
    return namespace


def test_recovered_sources_are_inert_and_parse_without_importing_backends():
    sources = sorted((REFERENCE / 'source').iterdir())
    assert len(sources) == 5 and all(p.name.endswith('.py.txt') for p in sources)
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'))
    recipe = json.loads((REFERENCE / 'RECIPE.json').read_text(encoding='utf-8'))
    assert recipe['source_snapshots_executable_as_delivered'] is False
    assert recipe['no_grad_live_replay_tolerance'] == {'atol': 0.0, 'rtol': 0.0}
    assert recipe['explicit_optimizer']['betas'] == [.9, .999]
    assert recipe['explicit_optimizer']['eps'] == 1e-8
    assert recipe['explicit_optimizer']['weight_decay'] == 0


def test_executed_recipe_is_bound_to_all_ten_retained_weights():
    recipe = json.loads((REFERENCE / 'RECIPE.json').read_text(encoding='utf-8'))
    selected = json.loads((BASE / 'weights/SELECTED.json').read_text(encoding='utf-8'))['selected']
    assert len(recipe['selected_runs']) == len(selected) == 10
    for record in recipe['selected_runs']:
        item = next(r for r in selected if (r['actor'], r['recipe'], r['seed']) ==
                    (record['actor'], record['procedure'], record['seed']))
        assert record['selected_epoch'] == item['epoch']
        assert sha256(BASE / item['path'] / 'setter.npz') == record['selected_parameter_sha256']
        assert (record['attempted_epochs'], record['completed_steps']) == (1, 384)
    initial = [r for r in recipe['selected_runs'] if r['selected_epoch'] == 0]
    assert [(r['actor'], r['procedure'], r['seed']) for r in initial] == [('gemma', 'FREE_PAIR_CONSISTENCY', 0)]
    for split, size in [('fit', 64), ('cal', 32)]:
        values = [json.loads(line) for line in (REFERENCE / 'data' / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(values) == len({r['pair_id'] for r in values}) == size
        assert all(len(r['sources']) == 2 and len(r['commands']) == 1 for r in values)


def test_original_group_order_has_exact_aligned_slot_coverage():
    groups = definitions(['groups'], random=random, SEEDS=(0, 1))['groups']
    pairs = [{'pair_id': 'synthetic-' + str(i)} for i in range(64)]
    seed_orders = [groups(pairs, seed, 1) for seed in (0, 1)]
    for seed, order in enumerate(seed_orders):
        assert order == groups(pairs, seed, 1) and len(order) == 384
        assert all(len(indices) == len(set(indices)) == 4 for _, indices in order)
        for pair in range(64):
            assert sorted(i for p, indices in order if p == pair for i in indices) == list(range(24))
    assert seed_orders[0] != seed_orders[1]


def test_calibration_selection_retains_feasibility_and_earlier_checkpoint_tie():
    pure = definitions(['feasible', 'select_checkpoint'])
    good = dict(changed=.9, harm=.01, changed_by_direction={'0': .9, '1': .9},
                all24=.8, native_all24=1., noop_preservation=.96, pair_correct=.8)
    assert pure['feasible'](good)
    curve = [dict(epoch=epoch, cal=dict(good)) for epoch in (0, 1)]
    assert pure['select_checkpoint'](curve)['epoch'] == 0
    curve[1]['cal']['pair_correct'] = .9
    assert pure['select_checkpoint'](curve)['epoch'] == 1
    curve[1]['cal']['noop_preservation'] = .94
    assert pure['select_checkpoint'](curve)['epoch'] == 0
    curve[0]['cal']['noop_preservation'] = .94
    assert pure['select_checkpoint'](curve)['feasible'] is False
