"""Reconstruct one recorded example without inference or pooled histories."""
import hashlib
import importlib.util
import json
from pathlib import Path

CASE = 1617
QUERY = 'J2'
ARM = 'STATE_0'
BASE = 'studies/repair_qwen7/input/PPR1_Same_Case_Public_Repair/'
SOURCES = {
    'original_case': 'studies/original_public_history/reproducibility/mqmh/case_1617.json',
    'runtime_cases': BASE + 'prepared/RUNTIME_CASES.json',
    'selector': BASE + 'vendor/ranker_contract.py',
    'model': BASE + 'reference/MODEL_VERIFY.json',
    'outcomes': 'studies/repair_qwen7/evaluation/SCORED_LOGICAL.jsonl',
}
PRODUCT = 'data/examples/recorded_case_1617.json'


def current_records(history):
    initial = history['initial']
    state = {(s, p): o for s, p, o in initial}
    if len(state) != len(initial):
        raise ValueError('Duplicate starting field')
    for subject, relation, entity in history['edits']:
        state[subject, relation] = entity
    return [[s, p, o] for (s, p), o in sorted(state.items())]


def verify_pair(case, query, rows, choose):
    if len(rows) != 2 or {r['history'] for r in rows} != {'H0', 'H1'}:
        raise ValueError('Expected exactly one record per history')
    states = {h: current_records(case['histories'][h]) for h in ('H0', 'H1')}
    if states['H0'] != states['H1']:
        raise ValueError('Independently built current records differ')
    result = []
    for row in sorted(rows, key=lambda r: r['history']):
        if (row['case_id'], row['query_id'], row['arm'], row['phase'], row['question']) != (
                case['case_id'], query['query_id'], ARM, 'post', query['question']):
            raise ValueError('Case/query/seed/phase pairing differs')
        selected = choose(row['candidates'], row['margins'])
        if selected is None or selected != row['selected']:
            raise ValueError('Recorded selection differs from frozen ranker')
        state = {(s, p): o for s, p, o in states[row['history']]}
        entity = selected['start_id']
        edges = []
        for relation in selected['path']:
            endpoint = state[entity, relation]
            edges.append([entity, relation, endpoint])
            entity = endpoint
        outcome = row['outcome']
        expected = row['expected_program']
        if not (selected['endpoint'] == outcome['entity'] == outcome['trace']['endpoint'] == entity == query['gold']
                and outcome['answer'] == case['labels'][entity]
                and outcome['status'] == 'VALID'
                and edges == selected['edges'] == outcome['trace']['dependencies']
                and expected == {'start': selected['start_id'], 'path': selected['path'], 'entity': entity}
                and row['expected_entity'] == entity
                and row['correct'] is True and row['program_faithful'] is True
                and row['wrong_program_right_answer'] is False):
            raise ValueError('Recorded answer, query or execution checks differ')
        result.append({k: row[k] for k in ('history', 'logical_id', 'call_ids', 'selected', 'outcome',
                       'correct', 'program_faithful', 'wrong_program_right_answer',
                       'request_payload_sha256', 'state_sha256')})
    if result[0]['request_payload_sha256'] != result[1]['request_payload_sha256']:
        raise ValueError('Equal-current request payloads differ')
    return states, result


def build(repository, outcome_path=None):
    root = Path(repository)
    native = json.loads((root / SOURCES['original_case']).read_bytes())
    runtime = json.loads((root / SOURCES['runtime_cases']).read_bytes())
    cases = [c for c in runtime if c['case_id'] == CASE]
    if len(cases) != 1 or native['case']['case_id'] != CASE:
        raise ValueError('Worked-example case is missing or duplicated')
    case = cases[0]
    query = next(q for q in native['case']['queries'] if q['query_id'] == QUERY)
    if next(q for q in case['queries'] if q['query_id'] == QUERY)['question'] != query['question']:
        raise ValueError('Original question wording differs')
    for history in ('H0', 'H1'):
        if native['case']['contexts'][history] != case['histories'][history]['source_text']:
            raise ValueError('Original and repair history text differs')
    rows = []
    with Path(outcome_path or root / SOURCES['outcomes']).open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if (row['case_id'], row['query_id'], row['arm'], row['phase']) == (CASE, QUERY, ARM, 'post') and row['history'] in ('H0', 'H1'):
                rows.append(row)
    spec = importlib.util.spec_from_file_location('recorded_example_frozen_ranker', root / SOURCES['selector'])
    ranker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranker)
    states, repaired = verify_pair(case, query, rows, ranker.choose)
    target = native['models']['qwen']['target']
    if target['case_id'] != CASE or target['query_id'] != QUERY or not all(
            target[k] is True for k in ('current_facts', 'references', 'common_guard', 'qualified')):
        raise ValueError('Original diagnostic qualification differs')
    native_rows = [r for r in native['models']['qwen']['responses'] if r['arm'] in ('H0', 'H1') and r['query_id'] in (QUERY, 'F0', 'F1', 'F2')]
    if len(native_rows) != 8 or len({(r['arm'], r['query_id']) for r in native_rows}) != 8:
        raise ValueError('Native answer/current-fact evidence incomplete')
    gold = {q['query_id']: q['gold'] for q in native['case']['queries']}
    for row in native_rows:
        correct = row['entity_id'] == gold[row['query_id']]
        if row['correct'] != correct or (row['query_id'] != QUERY and not correct):
            raise ValueError('Native current-fact checks differ')
    return {'schema': 'RECORDED_WORKED_EXAMPLE_V1', 'case_id': CASE, 'query': query,
        'repair_arm': ARM, 'repair_model_evidence': json.loads((root / SOURCES['model']).read_bytes()),
        'history_inputs': {h: case['histories'][h] for h in ('H0', 'H1')},
        'independently_built_current_records': states, 'labels': case['labels'],
        'native_diagnostic_checks': target, 'native_responses': native_rows, 'repair_responses': repaired,
        'shared_recorded_call_ids': repaired[0]['call_ids'] == repaired[1]['call_ids'],
        'wrapper_relationship': 'Original native wrapper versus later PPR repair wrapper; not a same-wrapper causal intervention.',
        'sources': {key: {'file': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()} for key, name in SOURCES.items()}}


def verify(repository, output=None):
    root = Path(repository)
    result = build(root, Path(output) / 'repair_qwen7/analysis/SCORED_LOGICAL.jsonl' if output else None)
    if result != json.loads((root / PRODUCT).read_bytes()):
        raise ValueError('Recorded worked example differs from its full source evidence')
    if output:
        with (Path(output) / 'tables/recorded_case_1617.json').open('x', encoding='utf8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    return {'status': 'PASS', 'case_id': CASE, 'query_id': QUERY, 'arm': ARM,
            'separate_history_stores': True, 'shared_recorded_call_ids': result['shared_recorded_call_ids'],
            'same_wrapper_causal_comparison': False, 'fresh_neural_calls': 0}
