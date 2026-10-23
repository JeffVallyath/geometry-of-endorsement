"""Export a compact, verified selection from an existing complete release."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from review_support import CASE, FRESH, LEGACY, RELEASE, digest, load, release_api, save


def export(source, output, native_source=None):
    source = Path(source).resolve()
    api = release_api()
    index = load(source / 'metadata/study_index.json')
    if digest(source / 'metadata/study_index.json') != digest(RELEASE / 'metadata/study_index.json'):
        raise ValueError('Source is not the included release resolver')
    # The complete snapshot remains subject to its exact checker. Do not rebuild it.
    api.verify(source)
    output = api.fresh_output(output, source)
    records = {}
    selections = {f'synthetic/{name}': FRESH + f'generated/{CASE}/{name}' for name in
                  ('root.json', 'questions.json', 'PLAN.json', 'labels.json',
                   'source_inputs.json', 'PROMPTS.json.gz')}
    selections.update({f'synthetic/{name}': FRESH + f'scientific/final/{CASE}/{name}'
                       for name in ('SCORES.json.gz', 'DONE.json')})
    selections.update({f'public/{name}': LEGACY + 'reproducibility/mqmh/' + name for name in
                       ('case_1617.json', 'cohort.json.gz', 'qwen_responses.json.gz',
                        'gemma_responses.json.gz', 'SOURCE_HASHES.json')})
    selections['public/recorded_case_1617.json'] = 'data/examples/recorded_case_1617.json'
    selections.update(MODEL_RECORDS)
    for name, logical in selections.items():
        row = index['mounts']['v3'][logical]
        original = api.safe(source, row['included'])
        if digest(original) != row['sha256']:
            raise ValueError('Source changed: ' + logical)
        target = api.safe(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if row.get('codec'):
            from experiments._shared.transport import restore
            restore(original, target, row)
        else:
            shutil.copyfile(original, target)
        records[name] = {'logical_source': 'v3/' + logical, 'source': row,
                         'export_sha256': digest(target)}
    save(output / 'BINDINGS.json', {'schema': 'COLLABORATOR_EXAMPLES_V1',
         'study_index_sha256': digest(source / 'metadata/study_index.json'),
         'selection': 'One full fresh synthetic case; original public response cohort and recorded paper case.',
         'records': records})
    if native_source is not None:
        enrich_native(native_source, output)
    print(f'Exported {len(records)} source-bound records. No inference.')
    return output


MODEL_RECORDS = {
    'synthetic/model-pins.json': LEGACY + 'reproducibility/update_method_comparison/inputs/models.json',
    'public/repair-qwen7-model.json': 'studies/repair_qwen7/input/PPR1_Same_Case_Public_Repair/reference/MODEL_VERIFY.json',
    'public/repair-qwen32-config.json': 'studies/repair_qwen32/PPR32_Repair_128/REPAIR32_CONFIG.json',
    'public/repair-gemma27-config.json': 'studies/repair_gemma27/source/Gemma27_Same_Case_Repair/GEMMA_CONFIG.json',
    'public/repair-gemma27-lock.json': 'studies/repair_gemma27/remote-evidence/MODEL_LOCK.json',
}


def enrich_native(source, output):
    """Select saved native message payloads; keep operational freeze files private."""
    source, output = Path(source).resolve(), Path(output).resolve()
    freeze = load(source / 'EXECUTION_FREEZE.json')
    extension = load(source / 'gemma_extension/EXECUTION_FREEZE.json')
    cohort = {c['case_id']: c for c in load(output / 'public/cohort.json.gz')['cases']}
    selected = [149, 531, 963, 1617]
    binding = load(output / 'BINDINGS.json')
    for model in ('qwen', 'gemma'):
        name = f'prepared/{model}_FINAL.json'
        expected = freeze['files'][name]
        if digest(source / name) != expected or extension['files'][name] != expected:
            raise ValueError('Native prompt plan differs from frozen source')
        plan = load(source / name)
        units = [u for u in plan['units'] if u['case_id'] in selected]
        keys = {(u['case_id'], u['arm'], u['query_id']) for u in units}
        required = {(cid, arm, q['query_id']) for cid in selected
                    for arm in ('H0', 'H1', 'S0', 'S1', 'C', 'C+') for q in cohort[cid]['queries']}
        if keys != required or len(keys) != len(units):
            raise ValueError('Incomplete native prompts')
        for unit in units:
            case = cohort[unit['case_id']]
            query = next(q for q in case['queries'] if q['query_id'] == unit['query_id'])
            body = case['contexts'][unit['arm']] + '\n\nQuestion: ' + query['question']
            if not unit['messages'][-1]['content'].endswith(body):
                raise ValueError('Native prompt/cohort identity differs')
        target_name = f'public/{model}_native_prompts.json'
        save(output / target_name, units)
        binding['records'][target_name] = {
            'source_kind': 'native', 'logical_source': name,
            'source': {'included': name, 'sha256': expected},
            'selection': {'field': 'units', 'case_ids': selected},
            'freeze_sha256': digest(source / 'EXECUTION_FREEZE.json'),
            'extension_freeze_sha256': digest(source / 'gemma_extension/EXECUTION_FREEZE.json'),
            'export_sha256': digest(output / target_name)}
    # This file is a generated export index, not a submitted scientific input.
    import json
    (output / 'BINDINGS.json').write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')


def enrich_token_spans(tokenizer_path, output):
    """Derive addresses with the frozen tokenizer; never load model weights."""
    from tokenizers import Tokenizer
    import json
    output = Path(output)
    expected = '3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922'
    if digest(tokenizer_path) != expected:
        raise ValueError('Tokenizer differs from the original DATA_FREEZE')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompts = load(output / 'synthetic/PROMPTS.json.gz')
    sources = load(output / 'synthetic/source_inputs.json')
    rows = []
    for origin in range(8):
        prompt = next(p for p in prompts if p['origin'] == origin and p['query_id'] == CASE + '-d0-08')
        source = next(s for s in sources if s['origin'] == origin and s['actor'] == 'gemma')
        encoded = tokenizer.encode(prompt['full_text'], add_special_tokens=False)
        if encoded.ids != prompt['full_ids']:
            raise ValueError('Derived tokenization differs from saved full token IDs')
        commands = []
        for command in source['commands']:
            start = prompt['header_offset'] + command['start']
            end = prompt['header_offset'] + command['end']
            positions = [i for i, (a, b) in enumerate(encoded.offsets) if b > start and a < end]
            commands.append(dict(command, sentence=source['source_text'][command['start']:command['end']],
                                 token_positions=positions))
        rows.append({'origin': origin, 'commands': commands})
    name = 'synthetic/token_spans.json'
    save(output / name, {'provenance': 'Derived from frozen character spans using the pinned tokenizer; full token IDs equal the saved PROMPTS records.',
                        'tokenizer_sha256': expected, 'rows': rows})
    binding = load(output / 'BINDINGS.json')
    binding['derived'] = {name: {'sha256': digest(output / name),
                                'inputs': ['synthetic/PROMPTS.json.gz', 'synthetic/source_inputs.json'],
                                'tokenizer_sha256': expected}}
    (output / 'BINDINGS.json').write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, help='Existing original MQMH source with frozen prepared plans')
    parser.add_argument('--tokenizer', type=Path, help='Existing pinned Gemma tokenizer.json; no download')
    args = parser.parse_args()
    export(args.source, args.output, args.native_source)
    if args.tokenizer:
        enrich_token_spans(args.tokenizer, args.output)
