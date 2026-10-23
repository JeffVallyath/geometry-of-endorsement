"""Frozen-protocol local evaluation; call only after full two-seed completion.

The GPU worker has no access to this module or its evaluation inputs.
"""
import argparse
import ast
import gzip
import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / 'runtime/payload'
INPUT = ROOT / 'PRIVATE/inputs/ppr1/PPR1_SAME_CASE_PUBLIC_REPAIR-RESULT/input/PPR1_Same_Case_Public_Repair'
PREPARED = ROOT / 'PRIVATE/prepared'
PROTOCOL = ROOT / 'PUBLIC/protocol/exposure_v1.json'
sys.path[:0] = [str(PAYLOAD / 'implementation'), str(PAYLOAD / 'vendor')]
from ranker_contract import choose
from execution import execute
from ppr_requests import record_view

HISTORIES = ['H0', 'H1', 'S0', 'S1']
CONTEXTS = HISTORIES + ['C', 'C+']
ARMS = ['ISO', 'HIST', 'CURRENT_EXTRA']


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def lines(path):
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def original_endpoints():
    path = INPUT / 'reference/mqmh.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or not n.level]
    module = types.ModuleType('original_mqmh')
    module.__dict__['ROOT'] = INPUT
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def paired_interval(a_num, a_den, b_num, b_den, draws):
    a_num, a_den, b_num, b_den = map(lambda x: np.asarray(x, dtype=float), (a_num, a_den, b_num, b_den))
    assert len(a_num) == len(a_den) == len(b_num) == len(b_den) == 128
    values = a_num[draws].sum(1) / a_den[draws].sum(1) - b_num[draws].sum(1) / b_den[draws].sum(1)
    return dict(difference=float(a_num.sum() / a_den.sum() - b_num.sum() / b_den.sum()),
        lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)),
        level=.95, draws=10000, unit='paired subject, all nested questions/histories/seeds retained')


def evaluate(raw, output):
    assert not output.exists(), 'Preserve prior evaluation outputs'
    complete = json.loads((raw / 'SCIENCE_COMPLETE.json').read_bytes())
    assert complete['status'] == 'COMPLETE' and complete['unique_calls'] == 36594
    expected_manifest = json.loads((PAYLOAD / 'PAYLOAD_MANIFEST.json').read_bytes())
    assert complete['manifest_sha256'] == sha(expected_manifest)
    schedule = lines(PREPARED / 'logical_schedule.jsonl.gz')
    requests = {r['request_id']: r for r in lines(PAYLOAD / 'requests.jsonl.gz')}
    calls_list = lines(raw / 'calls.jsonl')
    calls = {r['call_id']: r for r in calls_list}
    assert len(calls) == len(calls_list) == 2 * len(requests)
    assert set(calls) == {seed + '/' + k for seed in ['seed0', 'seed1'] for k in requests}
    for call in calls.values():
        request = requests[call['request_id']]
        assert call['request'] == request['request']
        assert sha(call['input_token_ids']) == request['token_sha256']
        assert call['result']['n_prompt'] == request['n_prompt']
        assert abs(call['result']['margin'] - (call['result']['yes_logit'] - call['result']['no_logit'])) < 1e-5
    cases = json.loads((INPUT / 'prepared/RUNTIME_CASES.json').read_bytes())
    by_case = {c['case_id']: c for c in cases}
    ids = [c['case_id'] for c in cases]
    gold = {(r['case_id'], r['query_id']): r for r in json.loads((INPUT / 'prepared/EVALUATOR_ONLY.json').read_bytes())}
    source = json.loads(gzip.decompress((INPUT / 'source/cohort.json.gz').read_bytes()))
    sourcecases = {c['case_id']: c for c in source['cases']}
    endpoints = original_endpoints()
    original_responses = json.loads(gzip.decompress((INPUT / 'source/qwen_responses.json.gz').read_bytes()))
    original_scored = endpoints.score(original_responses['primary'], sourcecases)
    old_witnesses = {(cid, t['query_id']) for cid in ids for t in endpoints.endpoints(sourcecases[cid], original_scored)['targets'] if t['qualified']}
    assert len(old_witnesses) == 94
    scored, lookup = [], {}
    for seed in ['seed0', 'seed1']:
        for row in schedule:
            cid, qid, history, arm = (row[k] for k in ['case_id', 'query_id', 'history', 'arm'])
            margins = [calls[seed + '/' + r]['result']['margin'] for r in row['request_ids']]
            selected = choose(row['candidates'], margins)
            outcome = execute(by_case[cid], history, 'post', selected)
            expected = gold[cid, qid]['post']
            edges, payload = record_view(by_case[cid], history, 'post')
            assert sha(edges) == row['state_sha256'] and sha(payload) == row['current_payload_sha256']
            faithful = selected is not None and selected['start_id'] == expected['start'] and selected['path'] == expected['path']
            correct = outcome['entity'] is not None and outcome['entity'] == expected['entity']
            s = dict(row, seed=seed, margins=margins, selected=selected, outcome=outcome, correct=correct,
                program_faithful=faithful, refused=selected is None, wrong_program=selected is not None and not faithful,
                wrong_program_correct_endpoint=selected is not None and not faithful and correct,
                expected_program=expected, branch=gold[cid, qid]['branch'])
            scored.append(s)
            key = (seed, arm, cid, history, qid)
            assert key not in lookup
            lookup[key] = s
    assert len(scored) == 32436
    output.mkdir(parents=True)
    with gzip.open(output / 'scored.jsonl.gz', 'wt', encoding='utf-8') as f:
        for r in scored:
            f.write(canonical(r) + '\n')
    statistics, clusters, target_ledgers = {}, {}, []
    for seed in ['seed0', 'seed1']:
        for arm in ARMS:
            key = seed + '/' + arm
            cluster = []
            for cid in ids:
                get = lambda h, q: lookup[seed, arm, cid, h, q]
                queries = [q['query_id'] for q in by_case[cid]['queries']]
                ep_rows = {(cid, h, q): {'entity_id': get(h, q)['outcome']['entity'], 'correct': get(h, q)['correct']} for h in CONTEXTS for q in queries}
                targets = endpoints.endpoints(sourcecases[cid], ep_rows)['targets']
                for t in targets:
                    target_ledgers.append(dict(t, seed=seed, arm=arm, original_witness=(cid, t['query_id']) in old_witnesses,
                        target_all_correct=all(get(h, t['query_id'])['correct'] for h in HISTORIES),
                        all_controls_and_references_correct=all(get(h, q)['correct'] for h in CONTEXTS for q in queries if not q.startswith('J'))
                            and all(get(h, t['query_id'])['correct'] for h in ['C', 'C+'])))
                robust = {q: all(get(h, q)['correct'] for h in HISTORIES) for q in queries}
                c = dict(subject=1, whole_subject=int(all(robust.values())),
                    whole_subject_all_six=int(all(get(h, q)['correct'] for h in CONTEXTS for q in queries)),
                    targets=3, controls=len(queries)-3, robust_target=sum(robust[q] for q in queries if q.startswith('J')),
                    robust_control=sum(robust[q] for q in queries if not q.startswith('J')),
                    guarded_targets=sum(t['common_guard'] for t in targets), guarded_witnesses=sum(t['qualified'] for t in targets),
                    questions=len(queries), answer_disagreement=0, valid_answer_disagreement=0, program_disagreement=0,
                    all_six_answer_disagreement=0, all_six_program_disagreement=0)
                for q in queries:
                    values = [get(h, q)['outcome']['entity'] for h in HISTORIES]
                    programs = [canonical(get(h, q)['selected']) for h in HISTORIES]
                    c['answer_disagreement'] += len(set(values)) > 1
                    c['valid_answer_disagreement'] += len(set(v for v in values if v is not None)) > 1
                    c['program_disagreement'] += len(set(programs)) > 1
                    c['all_six_answer_disagreement'] += len({get(h, q)['outcome']['entity'] for h in CONTEXTS}) > 1
                    c['all_six_program_disagreement'] += len({canonical(get(h, q)['selected']) for h in CONTEXTS}) > 1
                for group, qs in [('target', [q for q in queries if q.startswith('J')]), ('control', [q for q in queries if not q.startswith('J')]), ('all', queries)]:
                    rr = [get(h, q) for h in HISTORIES for q in qs]
                    c[group + '_occurrences'] = len(rr)
                    for field in ['correct', 'refused', 'wrong_program', 'wrong_program_correct_endpoint', 'program_faithful']:
                        c[group + '_' + field] = sum(r[field] for r in rr)
                cluster.append(c)
            clusters[key] = cluster
            totals = {field: sum(c[field] for c in cluster) for field in cluster[0]}
            tl = [r for r in target_ledgers if r['seed'] == seed and r['arm'] == arm and r['original_witness']]
            totals['original_94_preserved_repairs'] = sum(r['target_all_correct'] and r['all_controls_and_references_correct'] for r in tl)
            statistics[key] = totals
    rng = np.random.default_rng(2026101001)
    draws = rng.integers(0, 128, size=(10000, 128))
    pairs = [('whole_subject', 'subject'), ('whole_subject_all_six', 'subject'), ('robust_target', 'targets'),
             ('robust_control', 'controls'), ('guarded_witnesses', 'targets'), ('guarded_targets', 'targets'),
             ('answer_disagreement', 'questions'), ('program_disagreement', 'questions'),
             ('target_correct', 'target_occurrences'), ('control_correct', 'control_occurrences'),
             ('all_refused', 'all_occurrences'), ('all_wrong_program', 'all_occurrences')]
    comparisons, harms = {}, []
    target_lookup = {(r['seed'], r['arm'], r['case_id'], r['query_id']): r for r in target_ledgers}
    for seed in ['seed0', 'seed1', 'seed_mean']:
        def vector(arm, field):
            if seed != 'seed_mean':
                return np.asarray([c[field] for c in clusters[seed + '/' + arm]], float)
            return (np.asarray([c[field] for c in clusters['seed0/' + arm]], float)
                    + np.asarray([c[field] for c in clusters['seed1/' + arm]], float)) / 2
        for base in ['ISO', 'CURRENT_EXTRA']:
            label = seed + '/HIST-minus-' + base
            comparisons[label] = {num: paired_interval(vector('HIST', num), vector('HIST', den), vector(base, num), vector(base, den), draws) for num, den in pairs}
            if seed == 'seed_mean':
                continue
            for cid in ids:
                for q in by_case[cid]['queries']:
                    qid = q['query_id']
                    a = all(lookup[seed, 'HIST', cid, h, qid]['correct'] for h in HISTORIES)
                    b = all(lookup[seed, base, cid, h, qid]['correct'] for h in HISTORIES)
                    r = dict(seed=seed, baseline=base, candidate='HIST', case_id=cid, query_id=qid,
                        corrected=not b and a, introduced_harm=b and not a, baseline_correct=b, candidate_correct=a)
                    if qid.startswith('J'):
                        ta = target_lookup[seed, 'HIST', cid, qid]
                        tb = target_lookup[seed, base, cid, qid]
                        r.update(guard_lost=tb['common_guard'] and not ta['common_guard'],
                            guarded_witness_repaired=tb['qualified'] and a and ta['all_controls_and_references_correct'],
                            witness_lost_without_repair=tb['qualified'] and not ta['qualified'] and not (a and ta['all_controls_and_references_correct']))
                    harms.append(r)
    harm_intervals = {}
    for seed in ['seed0', 'seed1', 'seed_mean']:
        for base in ['ISO', 'CURRENT_EXTRA']:
            selected_seeds = ['seed0', 'seed1'] if seed == 'seed_mean' else [seed]
            harm_intervals[seed + '/HIST-minus-' + base] = {}
            for field in ['introduced_harm', 'corrected', 'guard_lost', 'witness_lost_without_repair']:
                num, den = [], []
                for cid in ids:
                    rr = [r for r in harms if r['case_id']==cid and r['seed'] in selected_seeds and r['baseline']==base
                          and (not field.startswith(('guard_', 'witness_')) or r['query_id'].startswith('J'))]
                    num.append(sum(r.get(field, False) for r in rr) / len(selected_seeds))
                    den.append(len(rr) / len(selected_seeds))
                harm_intervals[seed + '/HIST-minus-' + base][field] = paired_interval(num, den, np.zeros(128), den, draws)
    summary = dict(status='COMPLETE_REANALYSIS', raw_calls=len(calls), logical_reads=len(scored),
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        metrics=statistics, paired_subject_intervals=comparisons, paired_harm_rate_intervals=harm_intervals,
        paired_harms={seed + '/HIST-minus-' + base: dict(introduced=sum(r['introduced_harm'] for r in harms if r['seed']==seed and r['baseline']==base),
            corrected=sum(r['corrected'] for r in harms if r['seed']==seed and r['baseline']==base),
            guard_losses=sum(r.get('guard_lost', False) for r in harms if r['seed']==seed and r['baseline']==base)) for seed in ['seed0', 'seed1'] for base in ['ISO', 'CURRENT_EXTRA']},
        interpretation_limits=['Same previously exposed MQuAKE cohort; not fresh population prevalence.',
            'Equal extra-arm harm is not obsolete-fact-specific evidence.', 'Refusals and guard loss remain failures; all 384 targets retained.',
            'No perfect semantic or token-length matching; 2 fixed trained seeds.', 'No neural answer rewrite or native tensor repair.'])
    save(output / 'METRICS.json', summary)
    save(output / 'SUBJECT_CLUSTERS.json', clusters)
    for name, data in [('TARGET_LEDGER', target_ledgers), ('PAIRED_HARMS', harms)]:
        with gzip.open(output / (name + '.jsonl.gz'), 'wt', encoding='utf-8') as f:
            for r in data:
                f.write(canonical(r) + '\n')
    print(json.dumps({'status': summary['status'], 'metrics': statistics, 'paired_harms': summary['paired_harms']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('raw', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    evaluate(a.raw, a.output)
