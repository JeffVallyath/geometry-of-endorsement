"""Thin fresh-case adapter over the byte-preserved task generators; no model calls."""
from collections import Counter
from dataclasses import asdict, replace
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from bindings import ROOT, REPOSITORY, PUBLIC, OLD_SHARDS, HANDOFF, load, save, sha, digest
from canonical import candidate, origins, readers, tasks
from canonical.scenes import scene_from_dict
from canonical.batch_adapter import Backbone
from canonical.config import MODELS
import inputs as I
import plan_checks as P

SEED = 2609182101
MASKS = tuple(tuple((i >> j) & 1 for j in range(3)) for i in range(8))
CONDITIONS = ('INV_PAIR_NLL_s0', 'INV_PAIR_NLL_s1')

def truth(values, spec):
    x = values[spec['a']][spec['p']]
    if spec['kind'] == 'direct': return x
    if spec['kind'] == 'opposes': return 1-x
    return P.truth(spec['kind'], x, values[spec['b']][spec['p']])

def table_hash(scene):
    """Semantic table identity ignores IDs, layout and entity-index permutations."""
    return digest(sorted((a, p, int(scene.values[i][j]))
                         for i, a in enumerate(scene.actors) for j, p in enumerate(scene.projects)))

def scenes(value):
    if isinstance(value, dict):
        if {'actors', 'projects', 'values', 'target_actor', 'target_project', 'layout', 'scene_id', 'split'} <= value.keys():
            yield scene_from_dict(value)
        else:
            for child in value.values(): yield from scenes(child)
    elif isinstance(value, list):
        for child in value: yield from scenes(child)

def historical_inventory():
    semantic, textual = set(), set()
    inventory = []
    for path in sorted(REPOSITORY.rglob('*')):
        if path.suffix not in ('.json', '.jsonl') or not path.is_file(): continue
        raw = path.read_bytes()
        if b'"actors"' not in raw or b'"values"' not in raw:
            if path.name == 'PRIOR_SOURCE_HASHES.json':
                textual.update(json.loads(raw)['sha256'])
                inventory.append({'path': str(path.relative_to(REPOSITORY)), 'sha256': sha(path), 'prior_digest_count': len(json.loads(raw)['sha256'])})
            continue
        objects = (json.loads(x) for x in raw.splitlines() if x.strip()) if path.suffix == '.jsonl' else [json.loads(raw)]
        count = 0
        for obj in objects:
            for scene in scenes(obj):
                semantic.add(table_hash(scene))
                textual.add(candidate.semantic_hash(scene))
                for procedure in ('C1', 'T4'):
                    textual.add(I.digest(tasks.prefix_text(scene, procedure, 'early')[0]))
                count += 1
        if count: inventory.append({'path': str(path.relative_to(REPOSITORY)), 'sha256': sha(path), 'scene_records': count})
    if not semantic or not textual: raise ValueError('No old population inventory')
    return semantic, textual, inventory

def gate_mapping(root, questions):
    direct = [q for q in questions if q['spec']['kind'] == 'direct']
    joint = [q for q in questions if q['spec']['kind'] in P.FUNCTIONS]
    if (len(direct), len(joint), sum(q['spec']['kind'] == 'opposes' for q in questions)) != (12, 18, 4):
        raise ValueError('Inherited battery cardinality changed')
    def draw(q): return int(q['query_id'].split('-d')[1][0])
    def pair(q): return (q['spec']['p'], *sorted((q['spec']['a'], q['spec']['b'])))
    targets = {}
    for q in joint:
        same = [x['query_id'] for x in joint if draw(x) == draw(q) and pair(x) == pair(q) and x['spec']['kind'] != q['spec']['kind']]
        cross = [x['query_id'] for x in joint if draw(x) == draw(q) and pair(x) != pair(q)]
        operands = [x['query_id'] for x in direct if draw(x) == draw(q) and x['spec']['p'] == q['spec']['p'] and x['spec']['a'] in pair(q)[1:]]
        twins = [x['query_id'] for x in joint if pair(x) == pair(q) and x['spec']['kind'] == q['spec']['kind']]
        if tuple(map(len, (same, cross, operands, twins))) != (2, 6, 2, 2): raise ValueError('Gate mapping cardinality')
        targets[q['query_id']] = {'same_pair': sorted(same), 'cross_pair': sorted(cross), 'operands': sorted(operands),
                                    'twins': sorted(twins), 'draw': draw(q), 'pair': list(pair(q)), 'function': q['spec']['kind']}
    return {'direct': sorted(q['query_id'] for q in direct), 'targets': targets}

def verify_old_gate_mappings():
    specification = load(HANDOFF / 'source_audit/GATE_SPEC.json')['mappings']
    checked = 0
    for rid, original in specification.items():
        root = load(OLD_SHARDS / rid / 'root.json')
        questions = load(OLD_SHARDS / rid / 'questions.json')
        actual = gate_mapping(root, questions)
        assert actual['direct'] == sorted(original['G1_direct_ids'])
        for qid, mapping in actual['targets'].items():
            expected = original['targets'][qid]
            assert mapping['same_pair'] == sorted(expected['G2a_same_draw_checks'])
            assert mapping['cross_pair'] == sorted(expected['G2b_same_draw_checks'])
            assert mapping['operands'] == sorted(expected['operand_ids'])
            assert mapping['twins'] == sorted(expected['target_code_ids'])
            checked += 1
    return checked

def build_root(split, index, attempt=0):
    prior_seed, prior_masks = candidate.SEED, origins.ORIGIN_MASKS
    candidate.SEED, origins.ORIGIN_MASKS = SEED, MASKS
    try:
        terminal = replace(candidate.candidate(split, index + 1024*attempt),
                           scene_id=f'FHV1-{split}-{index:04d}', split=split)
        source = origins.origins(tasks.OLD, terminal)
    finally:
        candidate.SEED, origins.ORIGIN_MASKS = prior_seed, prior_masks
    commands = origins.commands(tasks.OLD, terminal)
    qs = [q for draw in (0, 1) for q in origins.questions(readers, terminal, terminal, 'R0', draw)]
    questions = [dict(query_id=q['query_id'], root_id=terminal.scene_id, split=split,
                      family='familiar', text=q['text'], labels=q['labels'], spec=q['spec']) for q in qs]
    inputs = []
    labels = []
    for origin, scene in enumerate(source):
        if tasks.OLD.apply_program(scene, commands).values != terminal.values: raise ValueError('Final-table nonconvergence')
        for actor, procedure in (('gemma', 'C1'), ('qwen', 'T4')):
            # Retain the inherited data schema; the execution plan selects only Gemma.
            text, spans = tasks.prefix_text(scene, procedure, 'early')
            addressed = tuple(I.AddressedUpdate(*spans[(c.actor, c.project)], c.value) for c in commands)
            I.validate(text, addressed)
            expected_text = tasks.prefix_text(terminal, procedure, 'early')[0]
            if I.rewrite(text, addressed, 'REBUILD') != expected_text: raise ValueError('Rewritten final text mismatch')
            inputs.append(dict(actor=actor, origin=origin, source_text=text, commands=[asdict(c) for c in addressed]))
        for q in questions:
            gold = truth(terminal.values, q['spec'])
            if gold != origins.truth(terminal.values, q['spec']): raise ValueError('Independent truth disagreement')
            labels.append(dict(root_id=terminal.scene_id, origin=origin, query_id=q['query_id'], gold=gold,
                               source_gold=truth(scene.values, q['spec'])))
    root = dict(root_id=terminal.scene_id, split=split, terminal=asdict(terminal),
                sources=[asdict(s) for s in source], commands=[asdict(c) for c in commands],
                source_hashes=[candidate.semantic_hash(s) for s in source])
    return root, questions, inputs, labels

def tokenizer_only_backbone(path):
    from transformers import AutoTokenizer
    bb = Backbone.__new__(Backbone)
    bb.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    bb.template_kwargs = dict(MODELS['gemma'].get('template_kwargs') or {})
    bb._prefix_cache = {}
    return bb

def main():
    out = ROOT / 'generated'
    if out.exists(): raise FileExistsError('Preserve the prior immutable case generation')
    from huggingface_hub import snapshot_download
    tokenizer_path = Path(snapshot_download(MODELS['gemma']['id'], revision=MODELS['gemma']['revision'],
        allow_patterns=['tokenizer*', 'special_tokens_map.json', 'config.json'], cache_dir=ROOT / 'model-cache'))
    bb = tokenizer_only_backbone(tokenizer_path)
    historical_semantic, historical_text, inventory = historical_inventory()
    old_semantic, old_text = len(historical_semantic), len(historical_text)
    old_gate_count = verify_old_gate_mappings()
    index_rows, rejected = [], []
    out.mkdir()
    for split, count in (('DEV', 8), ('FINAL', 64)):
        for index in range(count):
            for attempt in range(32):
                root, questions, inputs, labels = build_root(split, index, attempt)
                semantic = {table_hash(scene_from_dict(s)) for s in root['sources']}
                text = set(root['source_hashes']) | {I.digest(s['source_text']) for s in inputs}
                if semantic & historical_semantic or text & historical_text:
                    rejected.append(dict(split=split, index=index, attempt=attempt, reason='PREMODEL_COLLISION'))
                    continue
                if len(semantic) != 8: raise ValueError('History identity collision')
                historical_semantic.update(semantic); historical_text.update(text)
                break
            else: raise ValueError('Predeclared 32-candidate replacement supply exhausted')
            folder = out / root['root_id']
            for name, value in [('root', root), ('questions', questions), ('source_inputs', inputs), ('labels', labels)]:
                save(folder / (name+'.json'), value)
            sources = sorted((s for s in inputs if s['actor'] == 'gemma'), key=lambda s:s['origin'])
            counts = [sum(s['values'][c['actor']][c['project']] != c['value'] for c in root['commands']) for s in root['sources']]
            lengths = [len(bb.prefix_ids(s['source_text'])) for s in sources]
            eligible_pairs = [[a,b] for a,b in combinations(range(8), 2) if counts[a] == counts[b] and lengths[a] == lengths[b]]
            chosen = P.choose_pair(root['root_id'], counts, lengths)
            partition = {str(draw): P.query_partition(root['root_id'], index, questions, draw) for draw in (0,1)}
            mapping = gate_mapping(root, questions)
            save(folder / 'PLAN.json', dict(index=index, split=split, changed_counts=counts,
                prefix_lengths=lengths, structural_pairs=eligible_pairs, selected_anchor_pair=chosen,
                utility=partition, gates=mapping, conditions=list(CONDITIONS), contexts_per_case=17))
            with gzip.open(folder / 'PROMPTS.json.gz', 'xt', encoding='utf8') as handle:
                prompt_rows = []
                for origin, source in enumerate(sources):
                    for q in questions:
                        enc = bb.encode(source['source_text'], tasks.suffix_text(q['text']))
                        label_ids = bb.label_ids(enc['full_text'], q['labels'])
                        prompt_rows.append(dict(origin=origin, query_id=q['query_id'], label_token_ids=label_ids, **enc))
                json.dump(prompt_rows, handle, separators=(',', ':'), allow_nan=False)
            index_rows.append(dict(root_id=root['root_id'], split=split, index=index,
                cell=[len(root['terminal']['actors']), len(root['terminal']['projects']), sum(c['value'] << j for j,c in enumerate(root['commands']))],
                structural_pair_count=len(eligible_pairs), files={p.name:sha(p) for p in sorted(folder.iterdir())}))
            print(json.dumps({'prepared': root['root_id'], 'structural_pairs': len(eligible_pairs)}), flush=True)
    final_cells = Counter(tuple(r['cell']) for r in index_rows if r['split'] == 'FINAL')
    technical_cells = Counter((r['cell'][0], r['cell'][1], r['cell'][2] & 1) for r in index_rows if r['split'] == 'DEV')
    assert len(final_cells) == 32 and set(final_cells.values()) == {2}
    assert len(technical_cells) == 8 and set(technical_cells.values()) == {1}
    save(out / 'INDEX.json', index_rows)
    save(out / 'COLLISION_CHECK.json', dict(historical_semantic_tables=old_semantic, historical_textual_hashes=old_text,
        historical_inventory=inventory, fresh_semantic_tables=576, collisions_remaining=0, replacements=rejected,
        semantic_rule='named Boolean table sorted without case ID, layout or entity index order', seed=SEED))
    tokenizer_files = {p.name:sha(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file()}
    save(out / 'DATA_FREEZE.json', dict(data_seed=SEED, final_cases=64, technical_cases=8,
        final_balance='32 size/truth cells x 2', technical_balance='4 sizes x 2 focal directions',
        old_gate_mappings_exact=old_gate_count, tokenizer_revision=MODELS['gemma']['revision'],
        tokenizer_files=tokenizer_files, tokenizer_path=str(tokenizer_path),
        no_model_outputs_observed=True, files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({'status':'FRESH_DATA_FROZEN', 'cases':72, 'old_gate_mappings_verified':old_gate_count, 'collisions':0}))

if __name__ == '__main__': main()
