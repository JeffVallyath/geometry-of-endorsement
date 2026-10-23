"""Recount the measured examples on CPU. Never run a language model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from review_support import (CASE, REPO, digest, forbid_inference_and_network, load,
                            mount_code, release_api, save, scoring_modules, verify_bundle)


def synthetic(records, modules):
    analysis, panels, replay, _ = modules
    folder = records / 'synthetic'
    root, questions, plan = (load(folder / name) for name in ('root.json', 'questions.json', 'PLAN.json'))
    panel, index = replay.check_case(folder, (root, questions, plan))
    rows = analysis.primary(root, questions, plan, index)
    prompts = load(folder / 'PROMPTS.json.gz')
    prompt_index = {(p['origin'], p['query_id']): p for p in prompts}
    if len(prompt_index) != 8 * len(questions):
        raise ValueError('Missing or duplicate frozen prompts')
    for context, saved in panel['contexts'].items():
        origin = 0 if context == 'NATIVE_FINAL' else int(context.split('/')[0][1:])
        for q in questions:
            p = prompt_index[origin, q['query_id']]
            t = panel['tokens'][context][q['query_id']]
            if (p['prefix_ids'] != saved['cache_identity']['prefix_ids']
                    or p['suffix_ids'] != t['suffix_ids']
                    or p['label_token_ids'] != t['label_token_ids']
                    or p['full_ids'] != p['prefix_ids'] + p['suffix_ids']
                    or hashlib.sha256(p['full_text'].encode()).hexdigest() != t['full_prompt_sha256']):
                raise ValueError('Saved prompt/token binding differs')
    changed = [sum(s['values'][cmd['actor']][cmd['project']] != cmd['value']
                   for cmd in root['commands']) for s in root['sources']]
    lengths = [len(prompt_index[o, questions[0]['query_id']]['prefix_ids']) for o in range(8)]
    pairs = [[a, b] for a in range(8) for b in range(a + 1, 8)
             if changed[a] == changed[b] and lengths[a] == lengths[b]]
    if changed != plan['changed_counts'] or lengths != plan['prefix_lengths'] or pairs != plan['structural_pairs']:
        raise ValueError('Changed-fact/token-length matching differs')
    qid = CASE + '-d0-08'
    target = next(q for q in questions if q['query_id'] == qid)
    decision = next(r for r in rows if r['query_id'] == qid and r['seed'] == 0)
    controls = plan['gates']['direct'] + plan['gates']['targets'][qid]['same_pair'] + plan['gates']['targets'][qid]['cross_pair']
    qs = {q['query_id']: q for q in questions}
    checks = []
    for o in range(8):
        context = f'o{o}/INV_PAIR_NLL_s0'
        for key in controls:
            gold = analysis.truth(root['terminal']['values'], qs[key]['spec'])
            response = index[context, key]
            checks.append({'history': f'H{o}', 'query_id': key, 'question': qs[key]['text'],
                           'labels': qs[key]['labels'], 'gold': gold, **response,
                           'correct': response['valid'] and response['prediction'] == gold})
    # Show a real exclusion, not a fabricated negative example.
    contrast = next(r for r in rows if not r['both_eligible'])
    return {'case_id': root['root_id'], 'model': panel['actor'], 'seed': 0,
            'question': target, 'decision': decision, 'all_decisions': rows,
            'changed_facts': changed, 'prefix_lengths': lengths,
            'checks_passed': sum(c['correct'] for c in checks), 'checks_total': len(checks),
            'checks': checks, 'references': {'clean': index['NATIVE_FINAL', qid],
                                          'no_op': index['o0/INV_PAIR_NLL_s0', qid]},
            'derived_old_answers': [analysis.truth(root['sources'][o]['values'], target['spec']) for o in (5, 6)],
            'contrast': contrast}


def public(records, modules):
    mqmh = modules[3]
    cohort = load(records / 'public/cohort.json.gz')['cases']
    cases = {c['case_id']: c for c in cohort}
    result = {}
    native = load(records / 'public/case_1617.json')
    if cases[1617] != native['case']:
        raise ValueError('Paper example differs from original cohort')
    for model in ('qwen', 'gemma'):
        responses = load(records / f'public/{model}_responses.json.gz')['primary']
        scored = mqmh.score(responses, cases)
        prompts = load(records / f'public/{model}_native_prompts.json')
        pindex = {(p['case_id'], p['arm'], p['query_id']): p for p in prompts}
        selected_ids = {p['case_id'] for p in prompts}
        required = {(cid, arm, q['query_id']) for cid in selected_ids for arm in mqmh.ALL_ARMS
                    for q in cases[cid]['queries']}
        if len(pindex) != len(prompts) or set(pindex) != required:
            raise ValueError('Missing or duplicated native prompts')
        for key, prompt in pindex.items():
            case = cases[key[0]]
            query = next(q for q in case['queries'] if q['query_id'] == key[2])
            body = case['contexts'][key[1]] + '\n\nQuestion: ' + query['question']
            if not prompt['messages'][-1]['content'].endswith(body):
                raise ValueError('Native prompt/question identity differs')
        endpoints = [mqmh.endpoints(case, scored) for case in cohort]
        targets = [t for case in endpoints for t in case['targets']]
        paper = next(x for x in endpoints if x['case_id'] == 1617)
        target = next(x for x in paper['targets'] if x['query_id'] == 'J2')
        if target != native['models'][model]['target']:
            raise ValueError('Paper example qualification differs')
        for row in native['models'][model]['responses']:
            if scored[row['case_id'], row['arm'], row['query_id']] != row:
                raise ValueError('Recorded paper response differs from cohort')
        simple = next((t for t in targets if t['hops'] == 2 and t['qualified']), None)
        contrast = next((t for t in targets if t['common_guard'] and not t['qualified']), None)
        selected = {'paper': target, 'simple': simple, 'contrast': contrast}
        examples = {}
        for label, item in selected.items():
            if item is None:
                continue
            cid = item['case_id']
            examples[label] = {'case': cases[cid], 'decision': item,
                               'responses': [r for key, r in scored.items() if key[0] == cid]}
        result[model] = {'qualified_questions': sum(t['qualified'] for t in targets),
                         'scheduled_questions': len(targets), 'cases': len(cases),
                         'examples': examples}
    return result


def report_text(result):
    s = result['synthetic']
    lines = ['# Saved-output example replay', '',
             'Recomputed from recorded scores and responses. No model was loaded.', '',
             '## Synthetic paper example', '', s['question']['text'].splitlines()[0], '',
             '| History | Changed facts | Tokens before question | Recorded answer |',
             '|---|---:|---:|---|']
    for o, prediction in enumerate(s['decision']['predictions']):
        lines.append(f'| H{o} | {s["changed_facts"][o]} | {s["prefix_lengths"][o]} | {s["question"]["labels"][prediction]} |')
    lines += ['', f'{s["checks_passed"]}/{s["checks_total"]} checks pass across all eight histories.',
              'Clean and no-op references are reported separately in results.json.',
              'Old answers are derived from old facts, not measured prior model answers.', '',
              '## Public MQuAKE recount', '']
    for model, value in result['public'].items():
        lines.append(f'{model}: {value["qualified_questions"]}/{value["scheduled_questions"]} qualified history-dependent questions across {value["cases"]} cases.')
        example = value['examples']['paper']
        query = example['decision']['query_id']
        lines += ['', '| History/reference | Saved answer |', '|---|---|']
        lines += [f'| {r["arm"]} | {r["text"].strip()} |' for r in example['responses'] if r['query_id'] == query]
        lines.append('')
    lines += ['See results.json for every check, both references, a simpler two-hop case,',
              'and eligible non-disagreements. These are saved observations, not new inference.', '']
    return '\n'.join(lines)


def run(records, output, source=None, native_source=None):
    records = Path(records).resolve()
    binding = verify_bundle(records, source, native_source)
    requested = Path(output).absolute()
    for parent in (requested, *requested.parents):
        if parent.is_symlink() or getattr(parent, 'is_junction', lambda: False)():
            raise ValueError('Linked output path')
    output = requested.resolve()
    protected = [REPO.resolve(), records]
    if source is not None:
        protected.append(Path(source).resolve())
    if native_source is not None:
        protected.append(Path(native_source).resolve())
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be outside protected inputs')
    output = release_api().fresh_output(output, REPO)
    code = mount_code(output / '_runtime')
    with scoring_modules(code) as modules:
        result = {'saved_output_replay': True, 'new_inference': False,
                  'source_binding_sha256': digest(records / 'BINDINGS.json'),
                  'verified_export_files': len(binding['records']),
                  'synthetic': synthetic(records, modules), 'public': public(records, modules)}
    save(output / 'results.json', result)
    with (output / 'REPORT.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(report_text(result))
    from review_views import export_views
    export_views(records, result, output / 'views')
    print('PASS: saved scores, prompts, checks and public counts reconstructed; no inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=Path, default=REPO / 'examples/records')
    parser.add_argument('--verify-source', type=Path, help='Original full history_updates directory')
    parser.add_argument('--native-source', type=Path, help='Optional original MQMH plans for projection verification')
    args = parser.parse_args()
    forbid_inference_and_network()
    run(args.records, args.output, args.verify_source, args.native_source)


if __name__ == '__main__':
    main()
