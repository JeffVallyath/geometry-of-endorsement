"""Independent arithmetic for the one-case extension; no experiment-code imports."""
import csv
import gzip
import io
import json
import math
from collections import Counter
from pathlib import Path

from repro.cache_crossover import classify, close, require
from repro.common import ROOT, load_json, new_output, safe_path, sha256

BASE = ROOT / 'reproducibility/cache_crossover/extension'
ARTIFACTS = ROOT / 'artifacts/cache_crossover'
DESIGN = 'reproducibility/state_sufficiency/independent_verification/inputs/designs/SSC1-FINAL-0035.json'
CORE = 'reproducibility/state_sufficiency/independent_verification/inputs/responses/gemma/SSC1-FINAL-0035/core.json.gz'


def analyze(design, core, candidate, rows, terminal):
    qids = candidate['six_question_ids']
    questions = {q['query_id']: q for q in design['questions'] if q['query_id'] in qids}
    require(len(questions) == len(qids) == 6, 'Six exact questions required')
    require(candidate['root_id'] == 'SSC1-FINAL-0035' and candidate['origin_A'] == 3 and candidate['origin_B'] == 5,
            'Selected history identity changed')
    world = design['root']['terminal']

    def gold(qid):
        q = questions[qid]['spec']
        a, b = [bool(world['values'][q[k]][q['p']]) for k in ('a', 'b')]
        return int({'direct': a, 'either': a or b}[q['kind']])

    histories = []
    for h in candidate['histories']:
        context = f'o{h["origin"]}/{candidate["condition"]}'
        prefix = core['contexts'][context]['prefix_ids']
        start = design['root']['sources'][h['origin']]['values']
        changed = sum(a != b for aa, bb in zip(start, world['values']) for a, b in zip(aa, bb))
        require(changed == h['actual_changed_facts'] == 2, 'Changed-fact matching failed')
        require(len(prefix) == h['prefix_tokens'] == 517, 'Prefix length matching failed')
        require(prefix[-8:] == h['last_eight_ids'], 'Prefix suffix binding failed')
        histories.append(dict(origin=h['origin'], changed_facts=changed, prefix_tokens=len(prefix)))
    require([h['origin'] for h in histories] == [3, 5], 'History inventory changed')
    expected_stages = dict(baseline=108, baseline_comparison=108, baseline_eligibility=2,
                           strict_dual_code_replay=2, source_early_identity=2, cell=48, control_comparison=24)
    require(Counter(r['stage'] for r in rows) == expected_stages, 'Scientific journal coverage changed')
    saved = {(r['context'], r['query_id']): r for r in core['physical'] if r['query_id'] in qids}
    contexts = {f'o{i}/{candidate["condition"]}' for i in range(8)} | {'NATIVE_FINAL'}
    cells, baselines = {}, {}
    for line, row in enumerate(rows, 1):
        if row['stage'] not in ('cell', 'baseline'):
            if row['stage'] == 'source_early_identity':
                require(len(row['per_layer_equal']) == 42 and all(row['per_layer_equal'][:16]), 'Recorded early identity failed')
            elif row['stage'] in ('baseline_comparison', 'control_comparison'):
                keys = ('saved_differences', 'repeat_differences') if row['stage'] == 'baseline_comparison' else ('differences',)
                for key in keys:
                    if row[key] is not None:
                        require(row[key] == dict(logp=[0.0, 0.0], probability=0.0, mass=0.0), 'Recorded exact replay failed')
            elif row['stage'] == 'strict_dual_code_replay':
                require(row['passed'], 'Recorded dual-format replay failed')
            elif row['stage'] == 'baseline_eligibility':
                require(all(v['saved'] == v['fresh'] and not v['changed'] for v in row['codes'].values()), 'Recorded eligibility changed')
            continue
        qid = row['query_id']
        require(row['root_id'] == candidate['root_id'] and qid in questions, 'Unexpected scientific row')
        m = row['metrics']
        lp = row.get('result', {}).get('logps', m['raw_logps_no_yes'])
        require(len(lp) == 2 and all(math.isfinite(x) for x in lp), 'Nonfinite log scores')
        # Alternative normalization to the frozen max-subtraction implementation.
        p = 1 / (1 + math.exp(lp[0] - lp[1]))
        mass = sum(math.exp(x) for x in lp)
        pred = int(lp[1] > lp[0])
        require(lp[0] != lp[1] and max(math.exp(x) for x in lp) > 1 - mass, 'Cannot certify valid global label argmax')
        close(p, m['p_yes']); close(mass, m['mass']); close(lp[1] - lp[0], m['semantic_log_odds'])
        require(lp == m['raw_logps_no_yes'] and pred == m['prediction'] and (pred == gold(qid)) == m['correct'], 'Saved semantic score mismatch')
        require(m['valid'] and m['finite'] and m['argmax_in_labels'] and not m['tie'], 'Saved flag mismatch')
        if 'result' in row:
            r = row['result']
            require(r['master_unchanged'] and r['master_hash_before'] == r['master_hash_after'], 'Recorded master mutation')
            require(r['finite'] and not r['tie'] and r['argmax_in_labels'] and r['prediction'] == pred, 'Result flag mismatch')
            close(mass, r['answer_mass'])
        value = dict(p_yes=p, mass=mass, prediction=pred, valid=True, correct=pred == gold(qid),
                     raw_logps_no_yes=lp, raw_line=line)
        if row['stage'] == 'baseline':
            key = row['repeat'], row['context'], qid
            require(key not in baselines, 'Duplicate baseline')
            reference = saved[row['context'], qid]
            require(lp == reference['logps'] and row['tokens'] == reference['tokens'], 'Older-score/token replay mismatch')
            for k in ('prediction', 'finite', 'tie', 'argmax_in_labels', 'valid'):
                require(m[k] == reference['recorded_checks'][k], 'Older-score flag mismatch')
            close(mass, reference['recorded_checks']['answer_mass'])
            baselines[key] = value
        else:
            key = row['state'], qid
            require(key not in cells, 'Duplicate cell')
            cells[key] = value
            if 'provenance' in row:
                prov = row['provenance']
                require(prov['cut'] == 15 and prov['layer_donors'] == [prov['early']] * 16 + [prov['later']] * 26,
                        'Recorded layer assignment changed')
    require(set(baselines) == {(rep, ctx, q) for rep in (0, 1) for ctx in contexts for q in qids}, 'Baseline coverage changed')
    states = ('AA', 'AB', 'BA', 'BB', 'sham_A', 'sham_B', 'copy_A', 'copy_B')
    require(set(cells) == {(s, q) for s in states for q in qids}, 'Cell coverage changed')
    for (state, qid), value in cells.items():
        if state in ('AA', 'BB'):
            ref = baselines[1, f'o{3 if state == "AA" else 5}/{candidate["condition"]}', qid]
        elif state.startswith(('sham_', 'copy_')):
            ref = cells['AA' if state.endswith('A') else 'BB', qid]
        else:
            continue
        require({k: v for k, v in value.items() if k != 'raw_line'} == {k: v for k, v in ref.items() if k != 'raw_line'},
                'Original reuse or sham/full-copy mismatch')
    results = []
    for code, qid in [('literal_No_Yes', candidate['primary_query_id']), ('coded_A_B', candidate['alternate_query_id'])]:
        q = questions[qid]
        direct = [d for d in questions.values() if d['spec']['kind'] == 'direct'
                  and set(d['labels']) == set(q['labels']) and d['spec']['a'] in (q['spec']['a'], q['spec']['b'])
                  and d['spec']['p'] == q['spec']['p']]
        require(len(direct) == 2, 'Direct semantic binding failed')
        joint = {s: cells[s, qid] for s in states[:4]}
        require(joint['AA']['p_yes'] >= .95 and joint['BB']['p_yes'] <= .05, 'Strong original disagreement failed')
        for rep in (0, 1):
            require(baselines[rep, 'NATIVE_FINAL', qid]['correct'], 'Native reference failed')
        checks = []
        for state, ref in [('AB', 'BB'), ('BA', 'AA')]:
            for d in direct:
                dqid = d['query_id']
                r, b = cells[state, dqid], cells[ref, dqid]
                require(all(cells[s, dqid]['correct'] for s in ('AA', 'BB')), 'Original direct correctness failed')
                answer = gold(dqid)
                pc = r['p_yes'] if answer else 1 - r['p_yes']
                pb = b['p_yes'] if answer else 1 - b['p_yes']
                delta = abs(pc - pb)
                passed = r['valid'] and r['correct'] and pc >= .95 and r['mass'] >= .90 and delta <= .02
                checks.append(dict(state=state, recipient=ref, query_id=dqid, p_correct=pc, mass=r['mass'],
                                   recipient_probability_change=delta, passed=passed))
        result = classify(joint, all(c['passed'] for c in checks))
        results.append(dict(code=code, query_id=qid, **result, strict_direct_checks=checks))
    require(terminal['status'] == 'SCIENTIFIC_EXECUTED' and terminal['error'] is None, 'Scientific terminal invalid')
    prefix = len({k[:2] for k in baselines}) + 4
    suffix = len(baselines) + sum(r['stage'] == 'cell' and 'result' in r for r in rows)
    require(terminal['calls'] == dict(prefix=prefix, suffix=suffix) and terminal['real_forwards'] == prefix + suffix == 166,
            'Forward accounting mismatch')
    checks = [c for r in results for c in r['strict_direct_checks']]
    return dict(root_id=candidate['root_id'], cases=1, formats=2, histories=histories, layers=dict(early=16, later=26),
                results=results, strict_direct_checks=len(checks), all_direct_passed=all(c['passed'] for c in checks),
                minimum_direct_probability=min(c['p_correct'] for c in checks),
                maximum_direct_change=max(c['recipient_probability_change'] for c in checks),
                minimum_direct_mass=min(c['mass'] for c in checks), replay_rows=len(baselines),
                sham_checks=12, full_copy_checks=12, scored_rows=len(cells)+len(baselines), recorded_forwards=prefix+suffix,
                neural_forwards_this_replay=0)


def check_exports(result, rows, queries, outcome, joint_csv, direct_csv):
    """Compare independently reconstructed scores to supplied immutable tables."""
    require(len(outcome['codes']) == 2 and outcome['retrospective_selection'] and outcome['stop_after_this_case'], 'Outcome scope changed')
    require(outcome['strict_joint_direct_preservation'] and not outcome['population_or_general_mechanism_claim'], 'Outcome boundary changed')
    for r, old in zip(result['results'], outcome['codes']):
        require(r['query_id'] == old['query_id'] and r['label'] == old['strict_label'] == 'LATER-FOLLOWING', 'Classification mismatch')
        for key in ('probability_gap', 'error_early', 'error_later'):
            close(r[key], old[key])
        for state, p in r['probabilities'].items():
            close(p, old['probability_effects']['cells'][state])
        for c, saved in zip(r['strict_direct_checks'], old['strict_direct_checks']):
            for key in ('state', 'recipient', 'query_id', 'passed'):
                require(c[key] == saved[key], 'Direct check binding mismatch')
            for key in ('p_correct', 'mass', 'recipient_probability_change'):
                close(c[key], saved[key])
    qmap = {q['query_id']: q for q in queries}
    cells = {(r['state'], r['query_id']): r for r in rows if r['stage'] == 'cell'}
    for blob, direct in [(joint_csv, False), (direct_csv, True)]:
        table = list(csv.DictReader(io.StringIO(blob.decode('utf8'))))
        require(len(table) == 8 and len({(r['state'], r['query_id']) for r in table}) == 8, 'Export inventory changed')
        for r in table:
            raw = rows[int(r['line']) - 1]
            require(raw is cells[r['state'], r['query_id']], 'Raw line selector mismatch')
            require(r['archive_sha256'] == '307a4f2344e81564300671d616af5cec21ac5bffe28b9553aedd5c4b253ef16a'
                    and r['member'] == 'remote/results/raw.jsonl', 'Source identity mismatch')
            require([r['label_no'], r['label_yes']] == qmap[r['query_id']]['labels'], 'Semantic label mapping changed')
            expected_code = 'literal_No_Yes' if set(qmap[r['query_id']]['labels']) == {'No', 'Yes'} else 'coded_A_B'
            require(r['code'] == expected_code, 'Answer-format binding changed')
            lp = raw['metrics']['raw_logps_no_yes']
            require([float(r[k]) for k in ('logp_no', 'logp_yes')] == lp, 'Export raw scores changed')
            p = 1 / (1 + math.exp(lp[0] - lp[1]))
            close(float(r['mass' if direct else 'candidate_mass']), sum(math.exp(v) for v in lp))
            if direct:
                require(r['recipient'] == {'AB': 'BB', 'BA': 'AA'}[r['state']], 'Original-state comparison binding changed')
                # Both queried facts in this fixed case are No; semantic order is never token order.
                close(float(r['p_correct']), 1 - p)
                base = cells[r['recipient'], r['query_id']]['metrics']['p_yes']
                close(float(r['baseline_p_correct']), 1 - base)
                close(float(r['recipient_probability_change']), abs(p-base))
                require(all(r[k] == 'True' for k in ('valid', 'correct', 'passed')), 'Export direct flag mismatch')
            else:
                require(int(r['early_origin']) == (3 if r['state'][0] == 'A' else 5)
                        and int(r['later_origin']) == (3 if r['state'][1] == 'A' else 5), 'Figure history identity changed')
                close(float(r['p_yes']), p)
                require(r['strict_label'] == 'LATER-FOLLOWING', 'Export classification mismatch')


def replay(root=ROOT):
    base = root / 'reproducibility/cache_crossover/extension'
    artifacts = root / 'artifacts/cache_crossover'
    provenance = load_json(base / 'provenance.json')
    for row in provenance['files']:
        path = safe_path(row['path'], root)
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], 'Extension evidence hash mismatch: ' + row['path'])
    candidate = load_json(base / 'candidate.json')
    require(sha256(root / DESIGN) == candidate['source_bindings']['design_sha256'], 'Design lineage mismatch')
    require(sha256(root / CORE) == candidate['source_bindings']['scores_sha256'], 'Original score lineage mismatch')
    raw_bytes = gzip.decompress((base / 'raw.jsonl.gz').read_bytes())
    import hashlib
    require(hashlib.sha256(raw_bytes).hexdigest() == provenance['raw_member_sha256'], 'Lossless journal binding mismatch')
    raw = [json.loads(line) for line in raw_bytes.splitlines()]
    result = analyze(load_json(root / DESIGN), json.loads(gzip.decompress((root / CORE).read_bytes())),
                     candidate, raw, load_json(base / 'terminal.json'))
    check_exports(result, raw, load_json(artifacts / 'query_mappings.json'),
                  load_json(artifacts / 'extension_outcome.json'),
                  (artifacts / 'extension_joint.csv').read_bytes(), (artifacts / 'extension_direct.csv').read_bytes())
    return result


def export(output, root=ROOT):
    """Regenerate extension and original-panel reporting tables in a fresh directory."""
    result = replay(root)
    destination = new_output(output / 'cache_crossover_extension')
    require(not any(destination.iterdir()), 'Choose a fresh reporting output directory')
    (destination / 'replay.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8', newline='\n')
    # Validated CSVs retain exact source arithmetic and line selectors, without roundoff drift.
    for name in ('extension_joint.csv', 'extension_direct.csv', 'original_panel.csv'):
        with (root / 'artifacts/cache_crossover' / name).open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if name == 'original_panel.csv':
            saved = load_json(root / 'reproducibility/cache_crossover/saved_summaries.json')
            require(len(saved) == len(rows) == 24, 'Original panel coverage changed')
            for row, old in zip(rows, saved):
                require(row['root_id'] == old['root_id'] and row['query_id'] == old['query_id']
                        and row['label'] == old['label'], 'Original panel classification mismatch')
                for state in ('AA', 'AB', 'BA', 'BB'):
                    require(float(row[state]) == old['probability_effects']['cells'][state], 'Original panel score mismatch')
        with (destination / name).open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return dict(status='extension_saved_scores_verified', cases=result['cases'], formats=result['formats'],
                direct_checks=result['strict_direct_checks'], output=str(destination), neural_forwards=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
