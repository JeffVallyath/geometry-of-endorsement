"""Reconstruct public multi-hop matched-history results from saved responses."""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path
import unicodedata

import numpy as np

from .common import ROOT, new_output
from .release_inputs import payload, verify

SOURCE = ROOT / 'reproducibility/mqmh'
ARMS = ('H0', 'H1', 'S0', 'S1')
ALL_ARMS = (*ARMS, 'C', 'C+')
MODELS = ('qwen', 'gemma')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def parse_response(row, case):
    """Frozen first-line exact-alias rule; never use the gold answer to parse."""
    if row['truncated'] or not row['control_tokens_valid']:
        return None, None
    lines = row['text'].strip().splitlines()
    if not lines:
        return None, None
    line = lines[0].strip()
    forms = {normalize(line)}
    prefixes = list(forms)
    if normalize(line).startswith('Answer:'):
        prefixes.append(normalize(line)[len('Answer:'):].strip())
    for text in prefixes:
        forms.add(text)
        unperiod = text[:-1].rstrip() if text.endswith('.') else text
        forms.add(unperiod)
        for left, right in [('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019')]:
            for candidate in (text, unperiod):
                if len(candidate) >= 2 and candidate.startswith(left) and candidate.endswith(right):
                    unquoted = candidate[1:-1].strip()
                    forms.add(unquoted)
                    if unquoted.endswith('.'):
                        forms.add(unquoted[:-1].rstrip())
    entities = {case['registry'][normalize(s)] for s in forms
                if normalize(s) and normalize(s) in case['registry']}
    canonical = dict(case['canonical_labels'], **{case['decoy_id']: case['decoy_label']})
    strict = {entity for entity, label in canonical.items() if label == line}
    return (next(iter(entities)) if len(entities) == 1 else None,
            next(iter(strict)) if len(strict) == 1 else None)


def score(rows, cases):
    result = {}
    for row in rows:
        key = row['case_id'], row['arm'], row['query_id']
        if key in result:
            raise ValueError('Duplicate saved response')
        case = cases[row['case_id']]
        query = next(q for q in case['queries'] if q['query_id'] == row['query_id'])
        entity, strict = parse_response(row, case)
        if (entity != row['entity_id'] or (entity is not None) != row['valid']
                or strict != row['strict_entity_id'] or (strict is not None) != row['strict_valid']):
            raise ValueError('Saved parser result differs from reconstructed response')
        if row['arm'] not in ALL_ARMS:
            raise ValueError('Unknown history arm')
        result[key] = dict(row, correct=entity is not None and entity == query['gold'],
                           strict_correct=strict is not None and strict == query['gold'])
    return result


def endpoints(case, scored, *, strict=False):
    cid = case['case_id']
    required = {(cid, arm, q['query_id']) for arm in ALL_ARMS for q in case['queries']}
    observed = {key for key in scored if key[0] == cid}
    if observed != required:
        raise ValueError('Incomplete or extra case responses cannot change the denominator')
    correct_key = 'strict_correct' if strict else 'correct'
    entity_key = 'strict_entity_id' if strict else 'entity_id'
    correct = lambda arm, query: scored[cid, arm, query][correct_key]
    entity = lambda arm, query: scored[cid, arm, query][entity_key]
    facts = all(correct(arm, f'F{i}') for arm in ARMS for i in range(case['hops']))
    suffix = all(correct(arm, 'K') for arm in ARMS) if case['hops'] > 2 else None
    decoy = all(correct(arm, 'Fdecoy') for arm in ARMS)
    rows, patterns = [], {}
    for query in ('J0', 'J1', 'J2'):
        references = correct('C', query) and correct('C+', query)
        guard = facts and references
        answers = {arm: entity(arm, query) for arm in ARMS}
        losses = {arm: int(not correct(arm, query)) for arm in ARMS}
        invalid = {arm: int(answers[arm] is None) for arm in ARMS}
        pairs = [(a, b) for a, b in combinations(ARMS, 2)
                 if answers[a] is not None and answers[b] is not None
                 and answers[a] != answers[b] and losses[a] != losses[b]]
        witness = bool(guard and pairs)
        delta = int(guard) * (losses['H0'] + losses['H1'] - losses['S0'] - losses['S1']) / 2
        delta_invalid = int(guard) * (invalid['H0'] + invalid['H1'] - invalid['S0'] - invalid['S1']) / 2
        rows.append(dict(case_id=cid, subject_id=case['subject_id'], hops=case['hops'],
                         query_id=query, current_facts=facts, references=references,
                         common_guard=guard, qualified=witness, relevance_difference=delta,
                         invalid_difference=delta_invalid,
                         long_chain_qualified=bool(witness and suffix) if suffix is not None else None,
                         decoy_qualified=bool(witness and decoy)))
        if guard:
            for a, b in pairs:
                key = a, b, answers[a], answers[b]
                patterns[key] = patterns.get(key, 0) + 1
    return dict(case_id=cid, targets=rows, suffix=suffix,
                cross_phrasing=any(count >= 2 for count in patterns.values()))


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260923)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    alpha = (1 - .9875) / 2
    return [float(x) for x in np.quantile(draws, [alpha, 1-alpha])]


def summarize(roots):
    rows = [q for root in roots for q in root['targets']]
    return dict(subjects=len(roots), target_phrasings=len(rows),
                common_guard=sum(q['common_guard'] for q in rows),
                qualified=sum(q['qualified'] for q in rows),
                qualified_rate=sum(q['qualified'] for q in rows) / len(rows),
                qualified_interval=interval([sum(q['qualified'] for q in r['targets'])/3 for r in roots]),
                relevance_difference=sum(q['relevance_difference'] for q in rows)/len(rows),
                relevance_interval=interval([sum(q['relevance_difference'] for q in r['targets'])/3 for r in roots]),
                invalid_contribution=sum(q['invalid_difference'] for q in rows)/len(rows),
                long_chain_qualified=sum(bool(q['long_chain_qualified']) for q in rows),
                long_chain_phrasings=sum(q['hops'] > 2 for q in rows),
                cross_phrasing_long_chain_subjects=sum(r['cross_phrasing'] and r['suffix'] is True for r in roots),
                decoy_qualified=sum(q['decoy_qualified'] for q in rows))


def assert_equivalent(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Summary field mismatch')
        for key in expected:
            assert_equivalent(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Summary length mismatch')
        for a, b in zip(actual, expected):
            assert_equivalent(a, b)
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError('Reconstructed statistic differs from saved output')
    elif actual != expected:
        raise ValueError('Reconstructed count or classification differs from saved output')


def analyze(cohort, responses):
    ids = cohort['case_ids']
    cases = {c['case_id']: c for c in cohort['cases']}
    if (len(ids) != 128 or len(set(ids)) != 128 or set(ids) != set(cases)
            or len({c['subject_id'] for c in cases.values()}) != 128 or set(responses) != set(MODELS)):
        raise ValueError('The two models must retain the same frozen 128-subject cohort')
    all_targets, summaries, examples = [], {}, {}
    for model in MODELS:
        rows = responses[model]
        primary = score(rows['primary'], cases)
        if {key[0] for key in primary} != set(ids):
            raise ValueError('Missing or extra cohort member')
        roots = [endpoints(cases[cid], primary) for cid in ids]
        strict = [endpoints(cases[cid], primary, strict=True) for cid in ids]
        repeated = score(rows['repeats'], cases)
        fixed, selected = rows['fixed_repeat_case_ids'], rows['selected_positive_repeat_case_ids']
        repeat_ids = set(fixed) | set(selected)
        if (len(fixed) != 8 or len(selected) != 4 or len(repeat_ids) != 12
                or {key[0] for key in repeated} != repeat_ids or not repeat_ids <= set(ids)):
            raise ValueError('Repeat selection or complete-case inventory differs')
        stable = {}
        for cid in sorted(repeat_ids):
            endpoints(cases[cid], repeated)
            keys = [key for key in repeated if key[0] == cid]
            stable[str(cid)] = all((repeated[k]['valid'], repeated[k]['entity_id'], repeated[k]['correct']) ==
                                   (primary[k]['valid'], primary[k]['entity_id'], primary[k]['correct']) for k in keys)
        for cid in selected:
            if not any(q['qualified'] for r in roots if r['case_id'] == cid for q in r['targets']):
                raise ValueError('Selected positive repeat was not positive')
        summaries[model] = dict(primary=summarize(roots), strict_canonical=summarize(strict),
                                repeats=dict(cases=len(stable), stable=sum(stable.values()),
                                             stable_by_case=stable, new_subjects=0))
        all_targets.extend(dict(model=model, **q) for r in roots for q in r['targets'])
        examples[model] = dict(target=next(q for r in roots for q in r['targets']
                                           if q['case_id'] == 1617 and q['query_id'] == 'J2'),
                               responses=[primary[k] for k in primary if k[0] == 1617],
                               repeat_stable=stable['1617'])
        if not examples[model]['target']['long_chain_qualified'] or not examples[model]['repeat_stable']:
            raise ValueError('Worked example does not pass the stated controls')
    return summaries, all_targets, dict(case=cases[1617], models=examples)


def run(output):
    if Path(output).exists():
        raise ValueError('Use a new output directory; existing results are preserved')
    binding = verify(SOURCE, 'mqmh')
    inputs = {r['member']: json.loads(payload(SOURCE, r)) for r in binding['files']}
    summaries, targets, example = analyze(inputs['cohort.json'],
                                         {m: inputs[m + '_responses.json'] for m in MODELS})
    expected = next(r for r in binding['support_files'] if r['path'] == 'RESULTS.json')
    assert_equivalent(summaries, json.loads(payload(SOURCE, expected)))
    destination = new_output(Path(output))
    for name, data in [('summary.json', summaries), ('case_1617.json', example)]:
        (destination / name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (destination / 'target_counts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    (destination / 'CHECKS.json').write_text(json.dumps(dict(status='PASS', models=list(MODELS),
        saved_response_reconstruction=True, pretrained_inference=False, intervals='128-subject clustered bootstrap; 10000 draws; 98.75%'), indent=2)+'\n', encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summaries = run(parser.parse_args().output)
    print(json.dumps({m: {k: s['primary'][k] for k in ['subjects','target_phrasings','common_guard',
                     'qualified','relevance_difference','long_chain_qualified','long_chain_phrasings']}
                      for m, s in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
