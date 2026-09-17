"""Frozen analysis for ICMH1-RIPPLE-v2: flags, qualification gates, results, repeats.

Standard library only; a pure function of the plan, the scoring map, the registry and
the journals. Never consults a model. Missing rows stay missing, never zero.
"""
import argparse
from pathlib import Path

from .common import Journal, need, read_json, write_json
from .parse import extract, strict_correct, unresolved

GATES = {
    'G1': {'given_facts_min': 13, 'canonical_min': 13, 'questions': 16},
    'G2': {'records_correct_both_min': 18, 'records': 22},
    'G4': {'unresolved_max': 4, 'reads': 44},
    'G5': {'questions_min': 10, 'questions': 16, 'roots_min': 3, 'roots': 6},
}


def load_rows(inputs, phase, journals):
    plan = read_json(Path(inputs) / ('PLAN_%s.json' % phase))
    scoring = {row['unit_id']: row for row in read_json(Path(inputs) / ('SCORING_%s.json' % phase))['rows']}
    registry = read_json(Path(inputs) / 'REGISTRY.json')['registry']
    generations, starts, completions = {}, [], []
    for path in journals:
        records, _, incomplete = Journal(path).load()
        need(not incomplete, 'Journal has an incomplete tail: %s' % path)
        for record in records:
            if record['event'] == 'generation':
                need(record['unit_id'] not in generations, 'Duplicate generation for %s' % record['unit_id'])
                generations[record['unit_id']] = record
            elif record['event'] == 'worker_start':
                starts.append(record)
            elif record['event'] == 'worker_complete':
                completions.append(record)
    return plan, scoring, registry, generations, starts, completions


def score_unit(unit, scoring, registry, generations):
    row = scoring[unit['unit_id']]
    record = generations.get(unit['unit_id'])
    if record is None:
        return {'unit_id': unit['unit_id'], 'root_id': unit['root_id'], 'kind': unit['kind'],
                'context': unit['context'], 'reference': unit['reference'], 'measured': False}
    strict = strict_correct(record['text'], row['answers'])
    result = {'unit_id': unit['unit_id'], 'root_id': unit['root_id'], 'kind': unit['kind'],
              'context': unit['context'], 'reference': unit['reference'], 'measured': True,
              'text': record['text'], 'strict_correct': strict, 'input_tokens': record['input_tokens'],
              'output_tokens': record['output_tokens'], 'ended_eos': record['ended_eos'],
              'truncated': record['truncated'], 'elapsed_seconds': record['elapsed_seconds']}
    if row.get('object_type'):
        extraction = extract(record['text'], registry, row['object_type'])
        result.update({'extracted_id': extraction['entity_id'], 'extraction_status': extraction['status'],
                       'extraction_line': extraction['line'], 'object_type': row['object_type'],
                       'gold_ids': row['gold_ids'],
                       'extracted_is_gold': extraction['entity_id'] in row['gold_ids'] if extraction['entity_id'] else False})
    if row.get('is_changed_address') is not None:
        result['is_changed_address'] = row['is_changed_address']
    return result


def index(scored):
    out = {}
    for row in scored:
        out[(row['root_id'], row['kind'], row['reference'], row['context'])] = row
    return out
def flags(plan, scored_index, roots_info):
    """A(root), C(q), D(q), W(q) exactly as defined in AMENDMENT.md section 4."""
    per_root, per_question = {}, {}
    for root_id in plan['roots']:
        info = roots_info[root_id]
        records = [('|'.join(row['address']), row['target_id']) for row in info['current_table']]
        reads = []
        for reference, target in records:
            entry = {'reference': reference, 'target_id': target}
            for context in ('chronological_B', 'chronological_D', 'none'):
                row = scored_index.get((root_id, 'atomic', reference, context))
                entry[context] = None if row is None or not row['measured'] else {
                    'text': row['text'], 'extracted_id': row.get('extracted_id'),
                    'status': row.get('extraction_status'), 'correct': row.get('extracted_id') == target,
                    'strict_correct': row['strict_correct']}
            entry['correct_both_histories'] = bool(entry['chronological_B'] and entry['chronological_D']
                                                   and entry['chronological_B']['correct']
                                                   and entry['chronological_D']['correct'])
            entry['measured_both_histories'] = bool(entry['chronological_B'] and entry['chronological_D'])
            reads.append(entry)
        measured = all(entry['measured_both_histories'] for entry in reads)
        per_root[root_id] = {'root_id': root_id, 'cluster_id': info['cluster_id'],
                             'changed_address': info['changed_address'], 'changed_slot_ids': info['changed_slot_ids'],
                             'record_reads': reads, 'records': len(reads),
                             'records_correct_both': sum(1 for entry in reads if entry['correct_both_histories']),
                             'A': bool(measured and all(entry['correct_both_histories'] for entry in reads)),
                             'A_measured': measured}
        for question in info['questions']:
            reference = question['question_id']
            cells = {context: scored_index.get((root_id, 'downstream', reference, context))
                     for context in ('none', 'given_facts', 'canonical', 'chronological_B', 'chronological_D')}
            measured_ref = all(cells[context] is not None and cells[context]['measured']
                               for context in ('given_facts', 'canonical'))
            c_flag = bool(measured_ref and cells['given_facts']['strict_correct'] and cells['canonical']['strict_correct'])
            b_cell, d_cell = cells['chronological_B'], cells['chronological_D']
            measured_hist = bool(b_cell and b_cell['measured'] and d_cell and d_cell['measured'])
            d_flag = None
            distinct_entities = None
            if measured_hist:
                b_id, d_id = b_cell.get('extracted_id'), d_cell.get('extracted_id')
                valid = b_id is not None and d_id is not None
                distinct_entities = bool(valid and b_id != d_id)
                exactly_one = (b_cell['strict_correct'] + d_cell['strict_correct']) == 1
                d_flag = bool(distinct_entities and exactly_one)
            per_question[reference] = {
                'question_id': reference, 'root_id': root_id, 'category': question['category'],
                'changed_address_hop': question['changed_address_hop'],
                'gold': [answer['value'] for answer in question['answers']],
                'answers_by_context': {context: (None if cells[context] is None or not cells[context]['measured'] else {
                    'text': cells[context]['text'], 'strict_correct': cells[context]['strict_correct'],
                    'extracted_id': cells[context].get('extracted_id'),
                    'extraction_status': cells[context].get('extraction_status')}) for context in cells},
                'C': c_flag, 'C_measured': measured_ref,
                'D': d_flag, 'D_measured': measured_hist, 'D_distinct_entities': distinct_entities,
                'A': per_root[root_id]['A'],
                'W': bool(per_root[root_id]['A'] and c_flag and d_flag) if (measured_ref and measured_hist and per_root[root_id]['A_measured']) else None,
                'A_and_C': bool(per_root[root_id]['A'] and c_flag) if (per_root[root_id]['A_measured'] and measured_ref) else None,
            }
    return per_root, per_question


def throughput(scored, starts, completions):
    times = sorted(row['elapsed_seconds'] for row in scored if row['measured'])
    if not times:
        return {'generations': 0}
    position = max(0, int(round(0.95 * (len(times) - 1))))
    return {'generations': len(times), 'mean_seconds': sum(times) / len(times), 'p95_seconds': times[position],
            'max_seconds': times[-1], 'total_generation_seconds': sum(times),
            'load_seconds_max': max([record['load_seconds'] for record in starts], default=None),
            'peak_allocated_bytes': max([record['peak_allocated_bytes'] for record in completions], default=None)}


def forecast(measured, remaining_units, remaining_processes, seconds_available, allowance=0.25,
             reserve_seconds=1800):
    per_unit = measured.get('p95_seconds')
    load = measured.get('load_seconds_max') or 0
    if per_unit is None:
        return {'fits': False, 'reason': 'no measured throughput'}
    work = remaining_units * per_unit + remaining_processes * load
    needed = work * (1 + allowance)
    return {'per_unit_p95_seconds': per_unit, 'load_seconds': load, 'remaining_units': remaining_units,
            'remaining_processes': remaining_processes, 'work_seconds_point': work,
            'work_seconds_with_allowance': needed, 'allowance': allowance,
            'seconds_available_before_reserve': seconds_available - reserve_seconds,
            'reserve_seconds': reserve_seconds,
            'fits': needed <= (seconds_available - reserve_seconds)}


def qualification_gates(plan, scored, per_root, per_question, measured, resource):
    downstream = [row for row in scored if row['kind'] == 'downstream']
    given = [row for row in downstream if row['context'] == 'given_facts']
    canonical = [row for row in downstream if row['context'] == 'canonical']
    atomic_hist = [row for row in scored if row['kind'] == 'atomic'
                   and row['context'] in ('chronological_B', 'chronological_D')]
    g1_given = sum(1 for row in given if row['measured'] and row['strict_correct'])
    g1_canonical = sum(1 for row in canonical if row['measured'] and row['strict_correct'])
    g2 = sum(entry['correct_both_histories'] for root in per_root.values() for entry in root['record_reads'])
    g4 = sum(1 for row in atomic_hist if row['measured'] and unresolved(row.get('extraction_status')))
    ac_questions = [question for question in per_question.values() if question['A_and_C']]
    g5_roots = sorted({question['root_id'] for question in ac_questions})
    complete = all(row['measured'] for row in scored)
    gates = {
        'G1': {'given_facts_correct': g1_given, 'canonical_correct': g1_canonical,
               'questions': len(given), 'threshold': GATES['G1'],
               'pass': g1_given >= GATES['G1']['given_facts_min'] and g1_canonical >= GATES['G1']['canonical_min']},
        'G2': {'records_correct_both': g2, 'records': sum(root['records'] for root in per_root.values()),
               'threshold': GATES['G2'], 'pass': g2 >= GATES['G2']['records_correct_both_min']},
        'G4': {'unresolved_chronological_atomic': g4, 'reads': len(atomic_hist), 'threshold': GATES['G4'],
               'pass': g4 <= GATES['G4']['unresolved_max']},
        'G5': {'questions_with_A_and_C': len(ac_questions), 'roots_with_any': len(g5_roots), 'roots': g5_roots,
               'threshold': GATES['G5'],
               'pass': len(ac_questions) >= GATES['G5']['questions_min'] and len(g5_roots) >= GATES['G5']['roots_min']},
        'technical': {'all_units_measured': complete, 'planned': plan['unit_count'],
                      'measured': sum(1 for row in scored if row['measured']),
                      'empty_outputs': sum(1 for row in scored if row['measured'] and not row['text'].strip()),
                      'truncated_outputs': sum(1 for row in scored if row['measured'] and row['truncated']),
                      'pass': complete},
        'resource': dict(resource, **{'pass': bool(resource.get('fits'))}),
    }
    gates['all_pass'] = all(gates[name]['pass'] for name in ('G1', 'G2', 'G4', 'G5', 'technical', 'resource'))
    gates['decision'] = 'OPEN_FINAL' if gates['all_pass'] else 'NO_GO_NO_FINAL'
    gates['measured_throughput'] = measured
    return gates


def final_results(per_root, per_question, scored):
    questions = list(per_question.values())
    w_true = [question for question in questions if question['W'] is True]
    ac = [question for question in questions if question['A_and_C'] is True]
    disagreements = [q for q in questions if q['D_measured'] and q['D_distinct_entities'] and not q['W']]
    locality = [row for row in scored if row['kind'] == 'locality']
    by_context = {}
    for context in ('none', 'chronological_B'):
        rows = [row for row in locality if row['context'] == context and row['measured']]
        by_context[context] = {'measured': len(rows), 'correct': sum(1 for row in rows if row['strict_correct'])}
    return {
        'denominators': {'roots': len(per_root), 'questions': len(questions)},
        'counts': {
            'A_roots': sum(1 for root in per_root.values() if root['A']),
            'A_roots_measured': sum(1 for root in per_root.values() if root['A_measured']),
            'C_questions': sum(1 for question in questions if question['C']),
            'D_questions': sum(1 for question in questions if question['D'] is True),
            'W_questions': len(w_true),
            'W_unmeasured': sum(1 for question in questions if question['W'] is None),
            'A_and_C_questions': len(ac),
            'roots_with_any_W': len({question['root_id'] for question in w_true}),
        },
        'coverage': {'A_and_C_fraction': (len(ac) / len(questions)) if questions else None,
                     'coverage_limited': (len(ac) / len(questions) < 0.6) if questions else None},
        'witness_rate_among_eligible_secondary': (len(w_true) / len(ac)) if ac else None,
        'by_category': {category: {
            'questions': sum(1 for question in questions if question['category'] == category),
            'A_and_C': sum(1 for question in questions if question['category'] == category and question['A_and_C']),
            'W': sum(1 for question in questions if question['category'] == category and question['W'] is True)}
            for category in sorted({question['category'] for question in questions})},
        'by_changed_hop': {str(hop): {
            'questions': sum(1 for question in questions if question['changed_address_hop'] == hop),
            'W': sum(1 for question in questions if question['changed_address_hop'] == hop and question['W'] is True)}
            for hop in (1, 2)},
        'per_root': {root_id: {'A': root['A'], 'records_correct_both': root['records_correct_both'],
                               'records': root['records'],
                               'questions': sum(1 for question in questions if question['root_id'] == root_id),
                               'C': sum(1 for question in questions if question['root_id'] == root_id and question['C']),
                               'D': sum(1 for question in questions if question['root_id'] == root_id and question['D'] is True),
                               'W': sum(1 for question in questions if question['root_id'] == root_id and question['W'] is True)}
                     for root_id, root in per_root.items()},
        'answer_disagreement_diagnostics': len(disagreements),
        'strict_versus_extracted_differences': sum(
            1 for row in scored if row['measured'] and row.get('extracted_id') is not None
            and row['strict_correct'] != row['extracted_is_gold']),
        'locality_panel_diagnostic': by_context,
        'witnesses': [{'question_id': question['question_id'], 'root_id': question['root_id'],
                       'category': question['category'], 'gold': question['gold'],
                       'chronological_B': question['answers_by_context']['chronological_B'],
                       'chronological_D': question['answers_by_context']['chronological_D']} for question in w_true],
        'interpretation': (
            'completed_with_witnesses' if w_true else 'completed_zero_witnesses'),
        'claim_limits': [
            'Nonparametric in-context supersession only; not a MEMIT/AlphaEdit replication or parametric claim.',
            'No p-values and no population bound: 98 questions are clustered inside 34 restricted roots.',
            'One witness is not a null; zero observed witnesses is not evidence of universal invariance.',
            'Repeats do not increase N and do not certify un-repeated roots.',
        ],
    }


def repeat_comparison(main_rows, repeat_rows):
    main = {row['unit_id']: row for row in main_rows if row['measured']}
    repeat = {row['unit_id']: row for row in repeat_rows if row['measured']}
    shared = sorted(set(main) & set(repeat))
    differences = []
    for unit_id in shared:
        a, b = main[unit_id], repeat[unit_id]
        if a['text'] != b['text'] or a.get('extracted_id') != b.get('extracted_id') or a['strict_correct'] != b['strict_correct']:
            differences.append({'unit_id': unit_id, 'root_id': a['root_id'], 'kind': a['kind'],
                                'context': a['context'], 'reference': a['reference'],
                                'main': {'text': a['text'], 'extracted_id': a.get('extracted_id'),
                                         'strict_correct': a['strict_correct']},
                                'repeat': {'text': b['text'], 'extracted_id': b.get('extracted_id'),
                                           'strict_correct': b['strict_correct']}})
    return {'compared_units': len(shared), 'main_only': sorted(set(main) - set(repeat)),
            'repeat_only': sorted(set(repeat) - set(main)), 'differences': differences,
            'identical': not differences,
            'note': 'Greedy decoding is not a guarantee of hardware-independent reproducibility.'}


def analyze(inputs, phase, journals, out, seconds_available=None, remaining_units=0, remaining_processes=0):
    plan, scoring, registry, generations, starts, completions = load_rows(inputs, phase, journals)
    roots_info = {root['root_id']: root for root in read_json(Path(inputs) / 'ROOTS.json')['roots']}
    scored = [score_unit(unit, scoring, registry, generations) for unit in plan['units']]
    scored_index = index(scored)
    per_root, per_question = flags(plan, scored_index, roots_info)
    measured = throughput(scored, starts, completions)
    payload = {'study': plan['study'], 'phase': phase, 'planned': plan['unit_count'],
               'measured': sum(1 for row in scored if row['measured']),
               'missing_unit_ids': [row['unit_id'] for row in scored if not row['measured']],
               'throughput': measured, 'per_root': per_root, 'per_question': per_question,
               'units': scored, 'runtime_receipts': starts, 'completions': completions}
    if phase == 'qualification':
        resource = forecast(measured, remaining_units, remaining_processes, seconds_available or 0) \
            if seconds_available is not None else {'fits': False, 'reason': 'no remaining time supplied'}
        payload['gates'] = qualification_gates(plan, scored, per_root, per_question, measured, resource)
    if phase in ('final', 'repeat_670', 'repeat_618'):
        payload['results'] = final_results(per_root, per_question, scored)
    if phase == 'development':
        payload['technical'] = {
            'all_units_measured': all(row['measured'] for row in scored),
            'empty_outputs': sum(1 for row in scored if row['measured'] and not row['text'].strip()),
            'truncated_outputs': sum(1 for row in scored if row['measured'] and row['truncated']),
            'ended_eos': sum(1 for row in scored if row['measured'] and row['ended_eos']),
            'extraction_status_counts': {status: sum(1 for row in scored if row.get('extraction_status') == status)
                                         for status in ('unique', 'unmatched', 'ambiguous', 'empty')},
            'max_input_tokens': max([row['input_tokens'] for row in scored if row['measured']], default=None),
            'note': 'Development is a technical/format/throughput check only; no adaptive scientific change is permitted.',
        }
    write_json(out, payload)
    summary = {'phase': phase, 'planned': plan['unit_count'], 'measured': payload['measured'],
               'throughput': measured}
    if 'gates' in payload:
        summary['gates'] = {name: payload['gates'][name].get('pass') for name in ('G1', 'G2', 'G4', 'G5', 'technical', 'resource')}
        summary['decision'] = payload['gates']['decision']
        summary['gate_values'] = {name: {k: v for k, v in payload['gates'][name].items() if k != 'threshold'}
                                  for name in ('G1', 'G2', 'G4', 'G5')}
    if 'results' in payload:
        summary['results'] = {'counts': payload['results']['counts'], 'coverage': payload['results']['coverage'],
                              'interpretation': payload['results']['interpretation']}
    if 'technical' in payload:
        summary['technical'] = payload['technical']
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--journal', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds-available', type=float)
    parser.add_argument('--remaining-units', type=int, default=0)
    parser.add_argument('--remaining-processes', type=int, default=0)
    arguments = parser.parse_args(argv)
    return analyze(arguments.inputs, arguments.phase, arguments.journal, arguments.out,
                   arguments.seconds_available, arguments.remaining_units, arguments.remaining_processes)


if __name__ == '__main__':
    import json as _json
    print(_json.dumps(main(), indent=1, default=str))
