"""Readable views of saved inputs and outputs; no invented model responses."""
from pathlib import Path

from review_support import CASE, load, save


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def export_views(records, result, output):
    output = Path(output)
    synthetic = result['synthetic']
    root = load(records / 'synthetic/root.json')
    sources = load(records / 'synthetic/source_inputs.json')
    panel = load(records / 'synthetic/SCORES.json.gz')
    prompts = load(records / 'synthetic/PROMPTS.json.gz')
    spans = load(records / 'synthetic/token_spans.json')
    hooks = []
    for origin in range(8):
        p = next(p for p in prompts if p['origin'] == origin and p['query_id'] == CASE + '-d0-08')
        write(output / f'synthetic/H{origin}.txt', p['full_text'])
        context = panel['contexts'][f'o{origin}/INV_PAIR_NLL_s0']
        source = next(s for s in sources if s['actor'] == 'gemma' and s['origin'] == origin)
        addresses = next(r for r in spans['rows'] if r['origin'] == origin)['commands']
        for address, command, measured in zip(addresses, source['commands'], context['realized']['map_records'], strict=True):
            if ({k: address[k] for k in command} != command
                    or address['sentence'] != source['source_text'][command['start']:command['end']]
                    or len(address['token_positions']) != measured['positions']
                    or any(p >= len(context['cache_identity']['prefix_ids']) for p in address['token_positions'])):
                raise ValueError('Hook addresses differ from source or recorded intervention')
        hooks.append({'history': f'H{origin}', 'seed': 0, 'method': context['method'],
                      'selected_weight_sha256': context['selected_weight_sha256'],
                      'zero_based_post_block_site': context['realized']['site'],
                      'workspace': context['realized']['workspace'], 'rank': 16,
                      'address_derivation': spans['provenance'],
                      'tokenizer_sha256': spans['tokenizer_sha256'], 'commands': addresses})
    save(output / 'synthetic/activation-inputs.json', hooks)
    questions = load(records / 'synthetic/questions.json')
    text = ['# Exact question suffixes', '', 'The full target prompts are H0.txt through H7.txt.',
            'Each question is served independently from a clone of its saved context cache.',
            'The suffix is followed by the same first-line reply instruction and model-turn boundary shown in those files.', '']
    for q in questions:
        text += ['## ' + q['query_id'], '', '```text', q['text'], '```', '']
    write(output / 'synthetic/questions.md', '\n'.join(text))
    text = ['# All facts in the teaching pair', '',
            '| Person | Proposal | H5 starts | H6 starts | Final |', '|---|---|---|---|---|']
    label = ['Opposed', 'In favor']
    final = root['terminal']
    for a, person in enumerate(final['actors']):
        for p, proposal in enumerate(final['projects']):
            values = [root['sources'][o]['values'][a][p] for o in (5, 6)] + [final['values'][a][p]]
            text.append('| ' + ' | '.join([person, proposal, *[label[v] for v in values]]) + ' |')
    write(output / 'synthetic/facts.md', '\n'.join(text) + '\n')
    text = ['# The 160 stricter checks', '',
            'Each row is one direct or other-combined check. Columns are the eight histories.',
            'The downstream target and two references are separate. Labels follow each question\'s own format.', '',
            '| Question and format | Expected | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for check in [c for c in synthetic['checks'] if c['history'] == 'H0']:
        rows = [c for c in synthetic['checks'] if c['query_id'] == check['query_id']]
        text.append('| ' + ' | '.join([check['question'].replace('\n', ' '), check['labels'][check['gold']],
                                      *[check['labels'][r['prediction']] if r['valid'] else 'INVALID' for r in rows]]) + ' |')
    text += ['', f'{synthetic["checks_passed"]}/{synthetic["checks_total"]} pass, recounted from full-vocabulary log probabilities.',
             'This is 12 direct checks plus the other 8 combined checks per history.', '']
    write(output / 'synthetic/checks.md', '\n'.join(text))
    for model, value in result['public'].items():
        all_prompts = load(records / f'public/{model}_native_prompts.json')
        for name, example in value['examples'].items():
            cid = example['case']['case_id']
            prompts_for_case = [p for p in all_prompts if p['case_id'] == cid]
            save(output / f'public/{model}-{name}-prompts.json', prompts_for_case)
            text = [f'# {model}: {name} example, case {cid}', '',
                    'Counterfactual benchmark facts, not claims about the people or works named.', '',
                    '| Question | Expected entity | H0 | H1 | S0 | S1 | Clean C | No-op C+ |',
                    '|---|---|---|---|---|---|---|---|']
            rows = {(r['arm'], r['query_id']): r for r in example['responses']}
            for q in example['case']['queries']:
                answers = [rows[arm, q['query_id']]['text'].strip().replace('|', '\\|')
                           for arm in ('H0', 'H1', 'S0', 'S1', 'C', 'C+')]
                text.append('| ' + ' | '.join([q['question'], q['gold'], *answers]) + ' |')
            text += ['', '## Full starting records and corrections', '']
            for arm, context in example['case']['contexts'].items():
                text += ['### ' + arm, '', '```text', context, '```', '']
            write(output / f'public/{model}-{name}.md', '\n'.join(text))
    return output
