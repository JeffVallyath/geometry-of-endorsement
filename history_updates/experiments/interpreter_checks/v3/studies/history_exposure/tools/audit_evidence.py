"""Exact-ID bridge and independently derived selected-case effects; no neural calls."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NSR = ROOT / 'PRIVATE/inputs/nsr1/NSR1-Neurosymbolic-Repair-Closure'
FOLLOW = ROOT / 'PRIVATE/inputs/reviewer/reviewer_repository/studies/original_public_history/reproducibility/completed_followups'
OUT = ROOT / 'PUBLIC/evidence'


def rows(path):
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8') as f:
        yield from (json.loads(line) for line in f)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def a10():
    designs = {d['root']['root_id']: d for d in rows(NSR / 'inputs/a10_scores/designs.jsonl.gz')}
    visible = {d['root_id']: d for d in rows(NSR / 'implementation/source/data/visible/a10/roots.jsonl')}
    assert designs.keys() == visible.keys() and len(designs) == 256
    source_checks, question_checks, command_checks = 0, 0, 0
    for rid, d in designs.items():
        v, root = visible[rid], d['root']
        assert {q['question_id']: q['text'] for q in v['questions']} == {q['query_id']: q['text'] for q in d['questions']}
        question_checks += len(d['questions'])
        for h in v['histories']:
            origin = int(h['history_id'][1:])
            source = root['sources'][origin]
            text = '\n'.join(r['text'] for r in h['records']) + '\n'
            assert sha(text) == h['source_sha256'] == root['source_hashes'][origin]
            source_checks += 1
            expected = [{'writes': [{'entity': source['actors'][c['actor']], 'field': 'stance',
                        'topic': source['projects'][c['project']], 'value': {0: 'oppose', 1: 'favor'}[c['value']]}]}
                        for c in root['commands']]
            assert h['typed_commands'] == expected
            command_checks += len(expected)
    candidate = {}
    for r in rows(NSR / 'runs/pod_out/final/qwen/a10/rows.jsonl.gz'):
        if r['arm'] == 'A10_TEXT_SOURCE':
            key = (r['root_id'], r['history_id'], r['question_id'])
            assert key not in candidate
            candidate[key] = r
    assert len(candidate) == 69632
    used, seedcounts, families = Counter(), Counter(), Counter()
    with gzip.open(OUT / 'a10_exact_id_join.jsonl.gz', 'wt', encoding='utf-8') as dest:
        for p in sorted((NSR / 'inputs/a10_scores').glob('primary-seed*.jsonl.gz')):
            for panel in rows(p):
                rid, seed = panel['root_id'], panel['seed']
                qs = {q['query_id']: q for q in designs[rid]['questions']}
                physical = {(r['context'], r['query_id']): r for r in panel['physical']}
                for h in range(8):
                    for qid, q in qs.items():
                        native_key = (f'o{h}/INV_PAIR_NLL_s{seed}', qid)
                        assert native_key in physical
                        ck = (rid, f'o{h}', qid)
                        c = candidate[ck]
                        fam = 'joint' if q['spec']['kind'] in ('same', 'both', 'either') else q['spec']['kind']
                        used[ck] += 1
                        seedcounts[seed] += 1
                        families[fam] += 1
                        dest.write(json.dumps({'root_id': rid, 'seed': seed, 'history_id': f'o{h}', 'query_id': qid,
                            'family': fam, 'native_source': p.name, 'native_context': native_key[0],
                            'saved_direct_or_joint_prediction_record': physical[native_key],
                            'source_aware_answer': c['answer'], 'compile_key': c['compile_key'],
                            'source_aware_state_content': c['content']}, ensure_ascii=False) + '\n')
    assert set(used) == set(candidate) and set(used.values()) == {2}
    assert dict(families) == {'direct': 49152, 'opposes': 16384, 'joint': 73728}
    return dict(status='PASS', exact_original_source_hashes=source_checks, exact_original_question_texts=question_checks,
                exact_typed_commands=command_checks, unique_source_aware_reads=len(candidate),
                seed_expanded_comparison_rows=sum(seedcounts.values()), seed_counts={str(k): v for k, v in seedcounts.items()},
                comparison_families=dict(families), rows_reused_per_saved_editor_seed=2,
                explanation='256 designs x 8 histories x 34 questions = 69,632 actual source-aware reads. Each exact root/history/query is joined to BOTH saved editor seeds, yielding 139,264 comparison rows. This is reuse of one seed-independent neural compiler result, not two fresh neural replications.')


def followups():
    result = {}
    for study, name in [('carrier', 'ALL_SCORES_AND_EFFECTS.csv.gz'), ('mediation', 'ALL_UNIQUE_FORWARD_SCORES.csv.gz')]:
        with gzip.open(FOLLOW / study / name, 'rt', encoding='utf-8') as f:
            data = list(csv.DictReader(f))
        groups = defaultdict(list)
        for r in data:
            groups[(r.get('root', 'selected_pair'), r['stage'], r['condition'])].append(r)
        selected = {}
        for k, rs in groups.items():
            if (study == 'carrier' and (k[1:] in [('D', 'BLOCK'), ('D', 'INJECT'), ('A', 'P1')] or k[1:] == ('arms', 'C'))) or (study == 'mediation' and k[1:] in [('readers', 'TOP8_BLOCK'), ('readers', 'TOP8_INJECT')]):
                joint = [r for r in rs if r['kind'] != 'direct']
                direct = [r for r in rs if r['kind'] == 'direct']
                for r in joint:
                    predicted = int(float(r['logp_yes']) > float(r['logp_no']))
                    assert predicted == int(r['prediction'])
                    if r['hard_transfer']:
                        assert (predicted != int(r['baseline_prediction'])) == (r['hard_transfer'] == 'True')
                for r in direct:
                    preserved = (r['valid'] == 'True' and r['correct'] == 'True' and float(r['p_correct']) >= .95
                                 and float(r['candidate_mass']) >= .90 and float(r['direct_recipient_probability_drift']) <= .02)
                    assert preserved == (r['direct_preserved'] == 'True')
                selected['/'.join(k)] = dict(joint_count=len(joint), hard_transfers=sum(r['hard_transfer'] == 'True' for r in joint),
                    direct_count=len(direct), direct_preserved=sum(r['direct_preserved'] == 'True' for r in direct))
        result[study] = dict(score_rows=len(data), selected_conditions=selected,
            new_joint_errors=sum(r['kind'] != 'direct' and r['baseline_correct'] == 'True' and r['correct'] == 'False' for r in data))
        if study == 'mediation':
            cells = []
            for b in data:
                if b['condition'] != 'TOP8_BLOCK' or b['kind'] == 'direct':
                    continue
                i = next(r for r in data if r['condition'] == 'TOP8_INJECT' and r['recipient'] == b['recipient'] and r['query_id'] == b['query_id'])
                native, carrier = float(b['baseline_margin']), float(b['carrier_margin'])
                gap = carrier - native
                block = (carrier - float(b['margin'])) / gap
                inject = (float(i['margin']) - native) / gap
                cells.append(dict(recipient=b['recipient'], format=b['format'], query_id=b['query_id'],
                                  block_fraction=block, inject_fraction=inject, mediation_score=min(block, inject)))
            training = next(c for c in cells if c['recipient'] == '3' and c['format'] == 'd0')
            audit = json.loads((FOLLOW / study / 'NUMERICAL_AUDIT.json').read_bytes())
            expected = audit['selected_reader']['training_outcomes']['TOP8']['mediation_score']
            assert math.isclose(training['mediation_score'], expected, abs_tol=1e-12)
            assert training['mediation_score'] < .8
            result[study].update(top8_cells_recomputed=cells, failed_training_threshold=.8,
                                 qualification='NO_QUALIFIED_READER_SET', tensor_replay_performed=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    joins = a10()
    save('a10_schedule_reconciliation.json', joins)
    effects = followups()
    save('carrier_mediation_recomputed.json', effects)
    print(json.dumps({'a10': joins, 'followups': effects}, ensure_ascii=False))


if __name__ == '__main__':
    main()
