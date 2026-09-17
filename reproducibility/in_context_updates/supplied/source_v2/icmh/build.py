"""Build every ICMH1-RIPPLE-v2 input from the immutable v7 source manifests.

Standard library only; no model, no provider, no network. Outputs:
  INPUT_RECEIPT.json   verified source member hashes and counts
  STRUCTURE_CHECK.json pre-output structural assertions (SCOPE/amendment section 2)
  ROOTS.json           per-root current table S, rendered lines, changed slots
  REGISTRY.json        typed entity alias registry used by the frozen parser
  PLAN_<phase>.json    ordered readout units with rendered user text (NO gold)
  SCORING_<phase>.json gold answers/targets, kept out of the worker's input
"""
import argparse
from pathlib import Path

from .common import (CONTEXTS, HEADER_CANONICAL, HEADER_CHRONOLOGICAL, HEADER_GIVEN_FACTS, PHASES,
                     STUDY_ID, digest, need, read_json, sha_file, write_json)
from .parse import build_registry, extract, registry_key

SOURCE_MEMBERS = ('manifests/splits.json', 'manifests/questions.json', 'manifests/histories.json',
                  'manifests/entities.json', 'manifests/dependencies.json', 'manifests/summary.json',
                  'evcpu/schedule.py', 'SEMANTIC_POLICY.md')
LOCALITY_MEMBER = 'manifests/counterfact_fidelity.json'
STRONG = 'strong_benchmark_scoped_chain'
REPEAT_PHASES = {'repeat_670': 'rippleedits:popular:670', 'repeat_618': 'rippleedits:random:618'}
QUALIFICATION_DOWNSTREAM = ('given_facts', 'canonical', 'none')
QUALIFICATION_ATOMIC = ('chronological_B', 'chronological_D', 'none')
FULL_DOWNSTREAM = CONTEXTS
FULL_ATOMIC = ('chronological_B', 'chronological_D', 'none')
LOCALITY_CONTEXTS = ('none', 'chronological_B')


def address(record):
    return (record['subject_id'], record['relation_id'], record['temporal_qualifier'])


def template(record):
    """The record's own public cloze: edit prompt when present, else its direct question."""
    if record.get('edit_prompt'):
        return record['edit_prompt'].format(record['subject_label'])
    return record['direct_question']


def sentence(record, target_label=None):
    """Plain record sentence; no bracketed metadata (amendment section 2)."""
    return template(record) + ' ' + (target_label if target_label is not None else record['target_label'])


def render(prompt, header, lines):
    if header is None:
        return prompt
    return header + '\n' + '\n'.join(lines) + '\n\n' + prompt


def ledger(records):
    """Deterministic last-write ledger over scalar benchmark slot assignments."""
    table = {}
    for record in records:
        need(record['semantics'] == 'benchmark_slot_assignment',
             'Only scalar benchmark slot assignments are in scope: %s' % record.get('source'))
        table[address(record)] = record['target_id']
    return table


def build_root(meta, questions, history):
    root_id = meta['root_id']
    selected = []
    for question_id in meta['question_ids']:
        question = questions[question_id]
        need(question['interpretation'] == STRONG, 'Non-strong question selected: %s' % question_id)
        need(question['root_id'] == root_id, 'Question/root mismatch: %s' % question_id)
        selected.append(question)

    candidates = {}
    for record in history['final_edits']:
        candidates.setdefault(address(record), []).append(record)
    for question in selected:
        for record in question['rule']['record_path']:
            candidates.setdefault(address(record), []).append(record)

    order = [tuple(item) for item in meta['current_record_order']]
    need(len(order) == len(set(order)), 'Duplicate address in declared record order: %s' % root_id)
    need(set(order) == set(candidates),
         'Declared record order does not cover exactly the required records: %s' % root_id)
    need(len(order) == meta['current_record_count'], 'Declared record count mismatch: %s' % root_id)

    table = []
    for item in order:
        rows = candidates[item]
        targets = {row['target_id'] for row in rows}
        need(len(targets) == 1, 'Conflicting current targets at %s in %s' % (item, root_id))
        chosen = sorted(rows, key=lambda row: (0 if row.get('edit_prompt') else 1, row['source']))[0]
        templates = {template(row) for row in rows}
        table.append({'address': list(item), 'record': chosen, 'template': template(chosen),
                      'sentence': sentence(chosen), 'target_id': chosen['target_id'],
                      'target_label': chosen['target_label'], 'object_type': chosen['object_type'],
                      'alternate_templates': sorted(templates - {template(chosen)}),
                      'sources': sorted(row['source'] for row in rows)})

    changed = tuple(meta['changed_address'])
    need(changed in set(order), 'Changed address missing from the current table: %s' % root_id)
    slot = meta['changed_slot_ids']
    paths = history['paths']
    need(set(paths) == {'B_then_C', 'C_only', 'C_then_C', 'D_then_C'}, 'Unexpected history paths: %s' % root_id)
    obsolete = {}
    for key, path_name in (('B', 'B_then_C'), ('D', 'D_then_C')):
        stages = paths[path_name]
        need(len(stages) == 2 and len(stages[0]) == 1, 'Expected one obsolete assignment then the final batch: %s' % root_id)
        record = stages[0][0]
        need(address(record) == changed, 'Obsolete assignment is not at the changed address: %s' % root_id)
        need(record['target_id'] == slot[key], 'Obsolete %s target differs from SCOPE: %s' % (key, root_id))
        final_stage = [(address(row), row['target_id']) for row in stages[1]]
        expected_final = [(address(row), row['target_id']) for row in history['final_edits']]
        need(final_stage == expected_final, 'Final batch differs between histories: %s' % root_id)
        obsolete[key] = record

    current_at_changed = next(row for row in table if tuple(row['address']) == changed)
    need(current_at_changed['target_id'] == slot['C'], 'Current value at the changed address differs from SCOPE: %s' % root_id)
    need(len({slot['B'], slot['D'], slot['C']}) == 3, 'B, D and C must be pairwise distinct: %s' % root_id)
    for key in ('B', 'D'):
        need(template(obsolete[key]) == current_at_changed['template'],
             'Obsolete %s uses a different template from the current record: %s' % (key, root_id))

    current_lines = [row['sentence'] for row in table]
    streams = {}
    for key in ('B', 'D'):
        obsolete_line = sentence(current_at_changed['record'], obsolete[key]['target_label'])
        streams[key] = {'obsolete_line': obsolete_line, 'lines': [obsolete_line] + current_lines,
                        'obsolete_target_id': obsolete[key]['target_id'],
                        'obsolete_target_label': obsolete[key]['target_label']}
        resolved = ledger([obsolete[key]] + [row['record'] for row in table])
        need(resolved == {tuple(row['address']): row['target_id'] for row in table},
             'Chronological %s stream does not resolve to the declared current table: %s' % (key, root_id))
    need(streams['B']['lines'][1:] == streams['D']['lines'][1:], 'B and D streams differ outside the obsolete line: %s' % root_id)
    need(streams['B']['obsolete_line'] != streams['D']['obsolete_line'], 'B and D obsolete lines are identical: %s' % root_id)

    questions_out = []
    for question in selected:
        path = question['rule']['record_path']
        final_hop = path[-1]
        questions_out.append({
            'question_id': question['question_id'], 'category': question['category'],
            'prompt': question['prompt'], 'answers': question['answers'],
            'gold_ids': list(question['target_ids']), 'test_condition': question.get('test_condition'),
            'final_hop_object_type': final_hop['object_type'],
            'final_hop_address': list(address(final_hop)),
            'hop_count': len(path),
            'changed_address_hop': [list(address(row)) for row in path].index(list(changed)) + 1
            if list(changed) in [list(address(row)) for row in path] else None,
        })
    need(all(q['changed_address_hop'] is not None for q in questions_out),
         'Changed address is not on the chain of every selected question: %s' % root_id)

    return {'root_id': root_id, 'phase': meta['phase'], 'cluster_id': meta['cluster_id'],
            'changed_address': list(changed), 'changed_slot_ids': slot,
            'current_table': table, 'current_lines': current_lines, 'streams': streams,
            'questions': questions_out}


def locality_items(manifest):
    items = []
    for case in manifest['selected_roots']:
        rewrite = case['requested_rewrite']
        items.append({'case_id': case['case_id'],
                      'prompt': rewrite['prompt'].format(rewrite['subject']),
                      'answers': [{'value': rewrite['target_true']['str'], 'aliases': []}]})
    return items


def contexts_for(root, context):
    if context == 'none':
        return None, []
    if context == 'given_facts':
        return HEADER_GIVEN_FACTS, root['current_lines']
    if context == 'canonical':
        return HEADER_CANONICAL, root['current_lines']
    if context == 'chronological_B':
        return HEADER_CHRONOLOGICAL, root['streams']['B']['lines']
    if context == 'chronological_D':
        return HEADER_CHRONOLOGICAL, root['streams']['D']['lines']
    raise IntegrityErrorContext(context)


class IntegrityErrorContext(Exception):
    pass


def unit_id(root_id, kind, context, reference):
    return digest([STUDY_ID, root_id, kind, context, reference])


def plan_for(roots, phase, locality):
    downstream_contexts = QUALIFICATION_DOWNSTREAM if phase == 'qualification' else FULL_DOWNSTREAM
    atomic_contexts = QUALIFICATION_ATOMIC if phase == 'qualification' else FULL_ATOMIC
    use_locality = phase != 'qualification'
    units, scoring = [], []
    for root in roots:
        for question in root['questions']:
            for context in downstream_contexts:
                header, lines = contexts_for(root, context)
                identifier = unit_id(root['root_id'], 'downstream', context, question['question_id'])
                units.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'downstream',
                              'context': context, 'reference': question['question_id'],
                              'user_text': render(question['prompt'], header, lines)})
                scoring.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'downstream',
                                'context': context, 'reference': question['question_id'],
                                'answers': question['answers'], 'gold_ids': question['gold_ids'],
                                'object_type': question['final_hop_object_type']})
        for row in root['current_table']:
            for context in atomic_contexts:
                header, lines = contexts_for(root, context)
                reference = '|'.join(row['address'])
                identifier = unit_id(root['root_id'], 'atomic', context, reference)
                units.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'atomic',
                              'context': context, 'reference': reference,
                              'user_text': render(row['template'], header, lines)})
                scoring.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'atomic',
                                'context': context, 'reference': reference,
                                'answers': [{'value': row['target_label'],
                                             'aliases': list(row['record'].get('target_aliases') or [])}],
                                'gold_ids': [row['target_id']], 'object_type': row['object_type'],
                                'is_changed_address': row['address'] == root['changed_address']})
        if use_locality:
            for item in locality:
                for context in LOCALITY_CONTEXTS:
                    header, lines = contexts_for(root, context)
                    reference = 'counterfact:%s' % item['case_id']
                    identifier = unit_id(root['root_id'], 'locality', context, reference)
                    units.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'locality',
                                  'context': context, 'reference': reference,
                                  'user_text': render(item['prompt'], header, lines)})
                    scoring.append({'unit_id': identifier, 'root_id': root['root_id'], 'kind': 'locality',
                                    'context': context, 'reference': reference,
                                    'answers': item['answers'], 'gold_ids': [], 'object_type': None})
    need(len({unit['unit_id'] for unit in units}) == len(units), 'Duplicate unit id in plan %s' % phase)
    return units, scoring


def leakage_check(units):
    """No gold answer lists, entity IDs, B/D labels, eligibility flags or bracketed metadata."""
    findings = []
    for unit in units:
        text = unit['user_text']
        for pattern in ('benchmark_slot_assignment', '[', ']', 'gold', 'alias', 'correct', 'eligib',
                        'chronological_B', 'chronological_D', 'obsolete'):
            if pattern in text:
                findings.append({'unit_id': unit['unit_id'], 'pattern': pattern})
        for token in text.replace('\n', ' ').split():
            stripped = token.strip('.,;:!?()')
            if len(stripped) > 1 and stripped[0] == 'Q' and stripped[1:].isdigit():
                findings.append({'unit_id': unit['unit_id'], 'pattern': 'entity_id_literal'})
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True, help='extracted nested v7 source directory')
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args(argv)
    source, out = arguments.source_root, arguments.out
    scope = read_json(arguments.scope)
    need(scope['study'] == 'ICMH1-RIPPLE-v2', 'Unexpected scope schema')

    package = read_json(source / 'PACKAGE_MANIFEST.json')['files']
    receipt = {'source_root': str(source), 'members': {}}
    for member in SOURCE_MEMBERS + (LOCALITY_MEMBER,):
        actual = sha_file(source / member)
        receipt['members'][member] = actual
        need(package.get(member) == actual, 'Source member differs from the nested package manifest: %s' % member)
        expected = scope['source_member_sha256'].get(member)
        if expected is not None:
            need(expected == actual, 'Source member differs from SCOPE: %s' % member)
    receipt['scope_sha256'] = sha_file(arguments.scope)

    splits = {row['root_id']: row for row in read_json(source / 'manifests/splits.json')}
    questions = {row['question_id']: row for row in read_json(source / 'manifests/questions.json')}
    histories = {row['root_id']: row for row in read_json(source / 'manifests/histories.json')}
    entities = read_json(source / 'manifests/entities.json')
    locality = locality_items(read_json(source / LOCALITY_MEMBER))
    need(len(locality) == 8, 'Expected the eight prepared locality items')

    roots, by_phase = {}, {'development': [], 'qualification': [], 'final': []}
    for meta in scope['root_metadata']:
        root_id = meta['root_id']
        need(root_id.startswith('rippleedits:'), 'No MQuAKE inference is in scope: %s' % root_id)
        split = splits[root_id]
        need(split['benchmark'] == 'rippleedits' and split['regime'] == 'replacement', 'Unexpected split: %s' % root_id)
        need(split['split'] == meta['phase'], 'Split/phase mismatch: %s' % root_id)
        need(split['cluster_id'] == meta['cluster_id'], 'Cluster mismatch: %s' % root_id)
        root = build_root(meta, questions, histories[root_id])
        roots[root_id] = root
        by_phase[meta['phase']].append(root)
    for phase in ('development', 'qualification', 'final'):
        expected_ids = list(scope[phase])
        need([root['root_id'] for root in by_phase[phase]] == expected_ids, 'Root order differs from SCOPE: %s' % phase)
        counts = scope['counts'][phase]
        need(len(by_phase[phase]) == counts['roots'], 'Root count mismatch: %s' % phase)
        need(sum(len(root['questions']) for root in by_phase[phase]) == counts['questions'],
             'Question count mismatch: %s' % phase)
        need(sum(len(root['current_table']) for root in by_phase[phase]) == counts['current_records'],
             'Current record count mismatch: %s' % phase)
    need(len({root['cluster_id'] for root in roots.values()}) == len(roots), 'Edited-subject clusters are not distinct')

    object_types = sorted({row['object_type'] for root in roots.values() for row in root['current_table']} |
                          {question['final_hop_object_type'] for root in roots.values() for question in root['questions']})
    registry = build_registry(entities, object_types)

    gold_self = []
    for root in roots.values():
        for row in root['current_table']:
            result = extract(row['target_label'], registry, row['object_type'])
            if result['status'] != 'unique' or result['entity_id'] != row['target_id']:
                gold_self.append({'root_id': root['root_id'], 'address': row['address'],
                                  'target_id': row['target_id'], 'target_label': row['target_label'],
                                  'object_type': row['object_type'], 'status': result['status'],
                                  'extracted': result['entity_id']})
        for question in root['questions']:
            keys = {registry_key(form) for form in [answer['value'] for answer in question['answers']]}
            need(all(keys), 'Empty public answer for %s' % question['question_id'])

    plans = {}
    for phase in PHASES:
        if phase in REPEAT_PHASES:
            selected = [roots[REPEAT_PHASES[phase]]]
        else:
            selected = by_phase[phase]
        units, scoring = plan_for(selected, 'qualification' if phase == 'qualification' else phase, locality)
        findings = leakage_check(units)
        need(not findings, 'Prompt leakage check failed: %s' % findings[:3])
        plan = {'study': STUDY_ID, 'phase': phase, 'system': None, 'roots': [root['root_id'] for root in selected],
                'unit_count': len(units), 'units': units,
                'note': 'Rendered user text only; no gold answers, aliases, entity ids, B/D labels or eligibility flags.'}
        plans[phase] = {'plan_sha256': write_json(out / ('PLAN_%s.json' % phase), plan),
                        'scoring_sha256': write_json(out / ('SCORING_%s.json' % phase),
                                                     {'study': STUDY_ID, 'phase': phase, 'rows': scoring}),
                        'units': len(units),
                        'by_kind': {kind: sum(1 for unit in units if unit['kind'] == kind)
                                    for kind in ('downstream', 'atomic', 'locality')},
                        'max_user_chars': max(len(unit['user_text']) for unit in units)}

    structure = {
        'study': STUDY_ID,
        'checks': {
            'source_members_match_package_manifest_and_scope': True,
            'roots': len(roots), 'distinct_clusters': len({root['cluster_id'] for root in roots.values()}),
            'all_rippleedits_replacement': True,
            'declared_record_order_covers_final_edits_and_all_question_supports': True,
            'scalar_semantics_only': True,
            'bdc_pairwise_distinct_at_changed_address': True,
            'identical_final_batch_in_both_histories': True,
            'identical_templates_and_current_table_across_streams': True,
            'streams_differ_only_in_the_obsolete_line': True,
            'chronological_ledger_resolves_to_declared_current_table': True,
            'changed_address_on_every_selected_chain': True,
            'prompt_leakage_findings': 0,
        },
        'changed_address_hop_distribution': {
            str(hop): sum(1 for root in roots.values() for question in root['questions']
                          if question['changed_address_hop'] == hop)
            for hop in (1, 2)},
        'category_counts': {category: sum(1 for root in roots.values() for question in root['questions']
                                          if question['category'] == category)
                            for category in sorted({question['category'] for root in roots.values()
                                                    for question in root['questions']})},
        'gold_label_self_extraction_exceptions': gold_self,
        'stream_token_proxy_chars': {root['root_id']: {
            'chronological_B': len('\n'.join(root['streams']['B']['lines'])),
            'chronological_D': len('\n'.join(root['streams']['D']['lines'])),
            'equal_line_counts': len(root['streams']['B']['lines']) == len(root['streams']['D']['lines']),
        } for root in roots.values()},
        'plans': plans,
        'registry_object_types': {object_type: len(registry.get(object_type, {})) for object_type in object_types},
        'note': ('Structural, pre-output checks only. Equal record counts do not imply equal token counts; '
                 'per-stream character sizes are reported and measured token lengths are journaled at run time.'),
    }
    receipt['input_counts'] = {phase: plans[phase]['units'] for phase in PHASES}
    write_json(out / 'ROOTS.json', {'study': STUDY_ID, 'roots': [roots[key] for key in
                                                                 [root['root_id'] for phase in ('development', 'qualification', 'final')
                                                                  for root in by_phase[phase]]]})
    write_json(out / 'REGISTRY.json', {'study': STUDY_ID, 'object_types': object_types, 'registry': registry})
    structure_sha = write_json(out / 'STRUCTURE_CHECK.json', structure)
    receipt['structure_check_sha256'] = structure_sha
    write_json(out / 'INPUT_RECEIPT.json', receipt)
    return {'roots': len(roots), 'plans': {phase: plans[phase]['units'] for phase in PHASES},
            'gold_self_exceptions': len(gold_self)}


if __name__ == '__main__':
    import json as _json
    print(_json.dumps(main(), indent=1))
