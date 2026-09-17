"""Independent saved-score replay of the completed Gemma crossover, without inference."""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from .common import ROOT, load_json, new_output, safe_path, sha256

BASE = ROOT / 'reproducibility/cache_crossover'
STATES = ('AA', 'BB', 'AB', 'BA')
LABELS = ('LATER-FOLLOWING', 'MIXED/INCONCLUSIVE', 'LOW-DISCRIMINATION', 'EARLY-FOLLOWING')
DISPLAY_LABELS = ('Clear later-state effect', 'Mixed', 'Insufficient original separation', 'Clear earlier-state effect')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(a, b, abs_tol=1e-12, rel_tol=0), f'Saved value mismatch: {a} != {b}')


def classify(joint, preserved):
    p = {s: r['p_yes'] for s, r in joint.items()}
    gap = abs(p['AA'] - p['BB'])
    early = (abs(p['BA'] - p['BB']) + abs(p['AB'] - p['AA'])) / 2
    later = (abs(p['BA'] - p['AA']) + abs(p['AB'] - p['BB'])) / 2
    label = 'MIXED/INCONCLUSIVE'
    if gap < .10:
        label = 'LOW-DISCRIMINATION'
    elif preserved and all(joint[s]['valid'] and joint[s]['mass'] >= .90 for s in ('AB', 'BA')):
        if early <= .05 and later >= .10 and all(joint[a]['prediction'] == joint[b]['prediction'] for a, b in [('AB', 'AA'), ('BA', 'BB')]):
            label = 'EARLY-FOLLOWING'
        elif later <= .05 and early >= .10 and all(joint[a]['prediction'] == joint[b]['prediction'] for a, b in [('AB', 'BB'), ('BA', 'AA')]):
            label = 'LATER-FOLLOWING'
    return dict(label=label, probability_gap=gap, error_early=early, error_later=later, probabilities=p)


def analyze(plan, worlds, rows):
    """Uses scores and semantic inputs, never the saved classification outputs."""
    cases = {r['root_id']: r for r in plan['records']}
    require(len(cases) == len(plan['records']) == 12, 'Expected twelve unique cases')
    require(Counter(c['role'] for c in cases.values()) == {'witness': 6, 'control': 6}, 'Case roles changed')
    require(plan['early_cache_layers_inclusive'] == [0, 15] and plan['later_cache_layers_inclusive'] == [16, 41], 'Layer cut changed')
    queries = {(rid, q['query_id']): q for rid, c in cases.items() for q in c['questions']}
    require(len(queries) == 72, 'Question inventory changed')

    def gold(rid, qid):
        q = queries[rid, qid]['spec']
        values = worlds[rid]['values']
        a, b = bool(values[q['a']][q['p']]), bool(values[q['b']][q['p']])
        return int({'direct': a, 'both': a and b, 'either': a or b, 'same': a == b}[q['kind']])

    def metrics(row, line):
        saved = row['metrics']
        lp = row.get('result', {}).get('logps', saved['raw_logps_no_yes'])
        require(len(lp) == 2 and all(math.isfinite(x) for x in lp), 'Nonfinite scores')
        weights = [math.exp(x - max(lp)) for x in lp]
        p = weights[1] / sum(weights)
        mass = sum(math.exp(x) for x in lp)
        # This sufficient bound checks vocabulary argmax membership without the full logits.
        require(max(math.exp(x) for x in lp) > 1 - mass, 'Cannot certify label-set argmax')
        valid = lp[0] != lp[1]
        pred = int(lp[1] > lp[0])
        correct = valid and pred == gold(row['root_id'], row['query_id'])
        for actual, expected in [(p, saved['p_yes']), (mass, saved['mass']),
                                 (lp[1] - lp[0], saved['semantic_log_odds'])]:
            close(actual, expected)
        require(saved['valid'] == valid and saved['prediction'] == pred and saved['correct'] == correct,
                'Saved discrete score mismatch')
        require(saved['finite'] and not saved['tie'] and saved['argmax_in_labels'], 'Invalid saved flags')
        if 'result' in row:
            r = row['result']
            require(r['master_unchanged'] and r['master_hash_before'] == r['master_hash_after'], 'Recorded state mutation')
            require(r['prediction'] == pred and r['finite'] and not r['tie'] and r['argmax_in_labels'], 'Result flag mismatch')
            close(mass, r['answer_mass'])
        return dict(p_yes=p, mass=mass, prediction=pred, valid=valid, correct=correct,
                    raw_logps_no_yes=lp, raw_line=line)

    cells, baselines = {}, {}
    stage_counts = Counter(r['stage'] for r in rows)
    require(stage_counts == dict(baseline=1296, baseline_comparison=1296, baseline_eligibility=24,
                                source_early_identity=24, cell=576, control_comparison=288, pair_complete=5),
            'Journal inventory changed')
    baseline_deltas = []
    for line, row in enumerate(rows, 1):
        stage = row['stage']
        if stage in ('baseline', 'cell'):
            rid, qid = row['root_id'], row['query_id']
            require((rid, qid) in queries, 'Unplanned query')
            target = cells if stage == 'cell' else baselines
            key = (rid, row['state'], qid) if stage == 'cell' else (rid, row['repeat'], row['context'], qid)
            require(key not in target, 'Duplicate score')
            target[key] = metrics(row, line)
            if stage == 'baseline':
                require(row['tokens']['label_token_ids'] == queries[rid, qid]['answer_label_token_ids'], 'Semantic label order changed')
            if 'provenance' in row:
                p = row['provenance']
                require(p['cut'] == 15 and p['layer_donors'] == [p['early']] * 16 + [p['later']] * 26,
                        'Recorded layer assignment changed')
        elif stage == 'baseline_comparison':
            for key in ('saved_differences', 'repeat_differences'):
                d = row[key]
                if d is not None:
                    require(max(d['logp']) <= .05 and d['probability'] <= .005 and d['mass'] <= .005, 'Recorded baseline tolerance failed')
                    baseline_deltas.append(d)
        elif stage == 'baseline_eligibility':
            require(all(r['saved'] == r['fresh'] and not r['changed'] for r in row['codes'].values() if r['primary']),
                    'Recorded primary eligibility changed')
        elif stage == 'source_early_identity':
            require(len(row['per_layer_equal']) == 42 and all(row['per_layer_equal'][:16]), 'Recorded early identity failed')

    expected_cells = {(rid, s, qid) for rid, qid in queries for s in (*STATES, 'sham_A', 'sham_B', 'copy_A', 'copy_B')}
    require(set(cells) == expected_cells, 'Missing or unplanned cells')
    for rid, c in cases.items():
        contexts = {'native'} | {f'o{i}/{c["condition"]}' for i in range(8)}
        # The native reference key is read from the journal; enforce all nine contexts below.
        actual = {k[2] for k in baselines if k[0] == rid}
        require(len(actual) == 9 and contexts - {'native'} <= actual, 'Baseline contexts changed')
        expected = {(rid, repeat, context, q['query_id']) for repeat in (0, 1) for context in actual for q in c['questions']}
        require({k for k in baselines if k[0] == rid} == expected, 'Missing baseline scores')
    for (rid, repeat, context, qid), r in baselines.items():
        if repeat == 1:
            ref = baselines[rid, 0, context, qid]
            require(r['prediction'] == ref['prediction'] and abs(r['p_yes'] - ref['p_yes']) <= .005
                    and abs(r['mass'] - ref['mass']) <= .005
                    and max(abs(a-b) for a, b in zip(r['raw_logps_no_yes'], ref['raw_logps_no_yes'])) <= .05,
                    'Fresh baseline repeat failed')
    max_control_delta = 0
    for (rid, state, qid), r in cells.items():
        if state in ('AA', 'BB'):
            ref = baselines[rid, 1, cases[rid]['context_' + state[0]], qid]
        elif state.startswith(('sham_', 'copy_')):
            ref = cells[rid, 'AA' if state.endswith('A') else 'BB', qid]
            max_control_delta = max(max_control_delta, abs(r['p_yes'] - ref['p_yes']))
        else:
            continue
        require({k: v for k, v in r.items() if k != 'raw_line'} == {k: v for k, v in ref.items() if k != 'raw_line'},
                'Original reuse or sham/full-copy mismatch')

    table, example = [], None
    for rid, case in cases.items():
        for q in [q for q in case['questions'] if q['is_joint']]:
            qid = q['query_id']
            primary = qid == case['primary_query_id']
            require(primary == q['is_primary'], 'Primary format binding changed')
            direct = [d for d in case['questions'] if not d['is_joint']
                      and set(d['answer_label_token_ids']) == set(q['answer_label_token_ids'])
                      and d['spec']['a'] in (q['spec']['a'], q['spec']['b']) and d['spec']['p'] == q['spec']['p']]
            require(len(direct) == 2, 'Direct operand binding failed')
            checks = []
            for state, ref_state in [('AB', 'BB'), ('BA', 'AA')]:
                for d in direct:
                    dqid = d['query_id']
                    r, ref = cells[rid, state, dqid], cells[rid, ref_state, dqid]
                    eligible = all(cells[rid, s, dqid]['correct'] for s in ('AA', 'BB'))
                    answer = gold(rid, dqid)
                    pcorrect = r['p_yes'] if answer else 1 - r['p_yes']
                    pref = ref['p_yes'] if answer else 1 - ref['p_yes']
                    delta = abs(pcorrect - pref)
                    passed = r['correct'] and (pcorrect >= .95 and r['mass'] >= .90 and delta <= .02 if primary else eligible)
                    checks.append(dict(state=state, query_id=dqid, p_correct=pcorrect,
                                       recipient_probability_change=delta, baseline_both_correct=eligible, passed=passed))
            preserved = all(d['passed'] for d in checks)
            joint = {s: cells[rid, s, qid] for s in STATES}
            table.append(dict(root_id=rid, pair_id=case['pair_id'], role=case['role'], query_id=qid,
                              primary=primary, format='main' if primary else 'alternate',
                              **classify(joint, preserved), direct_preservation=preserved, direct_checks=checks))
            if rid == 'SSC1-FINAL-0026' and primary:
                world = worlds[rid]
                example = dict(root_id=rid, query_id=qid, format='main', semantic_spec=q['spec'],
                    correct_semantic_answer=gold(rid, qid),
                    direct_facts=[dict(query_id=d['query_id'], actor=world['actors'][d['spec']['a']],
                                       project=world['projects'][d['spec']['p']], correct_semantic_answer=gold(rid, d['query_id'])) for d in direct],
                    states=[dict(state=s, joint=joint[s], direct=[cells[rid, s, d['query_id']] for d in direct]) for s in STATES])
    require(len(table) == 24 and example is not None, 'Missing classification or example')
    counts, preservation = {}, {}
    for role in ('witness', 'control'):
        counts[role], preservation[role] = {}, {}
        for fmt in ('main', 'alternate'):
            group = [r for r in table if r['role'] == role and r['format'] == fmt]
            require(len(group) == 6, 'Format coverage changed')
            n = Counter(r['label'] for r in group)
            counts[role][fmt] = dict(zip(('clear_later', 'mixed', 'low_separation', 'clear_early'), [n[k] for k in LABELS]), total=len(group))
            checks = [d for r in group for d in r['direct_checks']]
            preservation[role][fmt] = dict(minimum_correct_probability=min(d['p_correct'] for d in checks),
                maximum_probability_change=max(d['recipient_probability_change'] for d in checks),
                all_preserved=all(r['direct_preservation'] for r in group), checks=len(checks))
    controls = dict(sham_checks=144, full_copy_checks=144, total_checks=288,
                    maximum_probability_change=max_control_delta, baseline_records=len(baselines),
                    baseline_comparisons=stage_counts['baseline_comparison'],
                    maximum_recorded_baseline_logp_change=max(max(d['logp']) for d in baseline_deltas),
                    maximum_recorded_baseline_probability_change=max(d['probability'] for d in baseline_deltas))
    return dict(layers=dict(early=16, later=26), counts=counts, preservation=preservation,
                controls=controls, classifications=table, example=example)


def replay(base: Path = BASE):
    provenance = load_json(base / 'provenance.json')
    for row in provenance['files']:
        path = safe_path(row['path'], base)
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], 'Evidence hash mismatch: ' + row['path'])
    raw = gzip.decompress((base / 'raw.jsonl.gz').read_bytes())
    import hashlib
    source = next(r for r in provenance['files'] if r['path'] == 'raw.jsonl.gz')['sources'][0]
    require(hashlib.sha256(raw).hexdigest() == source['sha256'], 'Decompressed journal hash mismatch')
    result = analyze(load_json(base / 'case_plan.json'), load_json(base / 'terminal_worlds.json'),
                     [json.loads(line) for line in raw.splitlines()])
    # Expectations are opened only after independent reconstruction.
    saved = load_json(base / 'saved_summaries.json')
    expected = {(r['root_id'], r['query_id']): r for r in saved}
    require(len(saved) == len(expected) == len(result['classifications']), 'Saved summary inventory changed')
    for row in result['classifications']:
        old = expected[row['root_id'], row['query_id']]
        for key in ('label', 'primary', 'role', 'pair_id', 'direct_preservation'):
            require(row[key] == old[key], 'Classification mismatch: ' + key)
        for key in ('probability_gap', 'error_early', 'error_later'):
            close(row[key], old[key])
        for state, p in row['probabilities'].items():
            close(p, old['probability_effects']['cells'][state])
        for d in row['direct_checks']:
            saved_check = next(v for v in old['direct_checks'] if (v['state'], v['query_id']) == (d['state'], d['query_id']))
            for key in ('p_correct', 'recipient_probability_change'):
                close(d[key], saved_check[key])
            require(d['passed'] == saved_check['passed'], 'Direct preservation mismatch')
    return result


def export(output, base=BASE):
    result = replay(base)
    destination = new_output(output / 'cache_crossover')
    with (destination / 'results.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    fields = ['root_id', 'main', 'alternate']
    with (destination / 'six_cases.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result['classifications']:
            if row['role'] == 'witness' and row['primary']:
                alt = next(r for r in result['classifications'] if r['root_id'] == row['root_id'] and not r['primary'])
                writer.writerow(dict(root_id=row['root_id'], main=DISPLAY_LABELS[LABELS.index(row['label'])],
                                     alternate=DISPLAY_LABELS[LABELS.index(alt['label'])]))
    return dict(status='saved_crossover_replayed', classifications=24, controls=288, output=str(destination))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
