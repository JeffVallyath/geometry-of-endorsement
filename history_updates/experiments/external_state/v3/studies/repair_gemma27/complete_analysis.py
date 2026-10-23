"""Local-only real-store lifecycle, traces, exposure and reporting joins.

Never stage this evaluator-side module to the inference Pod.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import sys
import time

RUN = Path(__file__).resolve().parent
SOURCE = RUN / 'source/Gemma27_Same_Case_Repair'
sys.path.insert(0, str(SOURCE / 'implementation'))
from gemma_prepare import load, gzrows, LEGACY, HISTORIES, canonical
from execution import build_store
from prepare import candidates
from state_grounding import current_payload
from ranker_contract import request, choose, key
from codec import norm


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write_rows(path, values):
    with path.open('wb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed:
        for value in values:
            compressed.write((canonical(value) + '\n').encode())


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def triples(store, snapshot=None):
    return [[s, p, o] for (s, p), o in sorted(store.view(snapshot).items())]


def main():
    evidence = RUN / 'remote-evidence'
    out = RUN / 'analysis'
    all_outcomes = gzrows(out / 'ALL_OUTCOMES.jsonl.gz')
    assert len(all_outcomes) == 40116
    cases = {r['case_id']: r for r in load(LEGACY / 'prepared/RUNTIME_CASES.json')}
    ev = {(r['case_id'], r['query_id']): r for r in load(SOURCE / 'prepared/EVALUATOR_ONLY.json')}
    logical = {r['logical_id']: r for r in gzrows(SOURCE / 'prepared/LOGICAL_ROWS.jsonl.gz')}
    bank = {r['request_id']: r for r in gzrows(SOURCE / 'prepared/REQUESTS.jsonl.gz')}
    registry = load(LEGACY / 'mapping/RELATION_REGISTRY.json')
    control = load(SOURCE / 'prepared/CONTROLS.json')
    idx = {(r['arm'], r['case_id'], r['history'], r['phase'], r['query_id']): r for r in all_outcomes}
    traces = []
    casebook = {cid: {'case_id': cid, 'source': cases[cid], 'reads': []} for cid in control['repeat_cases'][:8]}
    physical = {arm: {r['request_id']: r for r in rows(evidence / folder / 'responses.jsonl')}
                for arm, folder in [('STATE_BASE', 'base'), ('STATE_0', 'seed0'), ('STATE_1', 'seed1')]}
    for r in all_outcomes:
        if not r['arm'].startswith('STATE'):
            continue
        case = cases[r['case_id']]
        t = time.perf_counter()
        store = build_store(case, r['history'])
        selected = r['selected']
        result = None if selected is None else store.query(selected['start_id'], selected['path'], 0 if r['phase'] == 'pre' else None)
        elapsed = time.perf_counter() - t
        assert (None if result is None else result['endpoint']) == r['entity']
        record = {k: r[k] for k in ['arm', 'logical_id', 'case_id', 'query_id', 'history', 'phase', 'selected', 'margins', 'status', 'correct', 'faithful']}
        record.update(execution_trace=result, cpu_store_and_query_seconds=elapsed,
                      request_ids=logical[r['logical_id']]['score_request_ids'])
        traces.append(record)
        if r['case_id'] in casebook:
            example = dict(record)
            example['question'] = logical[r['logical_id']]['question']
            example['actual_physical_scores'] = [physical[r['arm']][rid] for rid in record['request_ids']]
            example['physical_inference_seconds'] = sum(s['seconds'] for s in example['actual_physical_scores'])
            example['timing_note'] = 'Identical main request identities may be deduplicated across logical reads; do not sum these reused times as independent physical work.'
            casebook[r['case_id']]['reads'].append(example)
    write_rows(out / 'ALL_STATE_EXECUTION_TRACES.jsonl.gz', traces)
    write_rows(out / 'EIGHT_FIXED_CASEBOOK.jsonl.gz', list(casebook.values()))

    lifecycle = []
    for arm, folder in [('STATE_BASE', 'base'), ('STATE_0', 'seed0'), ('STATE_1', 'seed1')]:
        actual = rows(evidence / (folder + '_return') / 'responses.jsonl')
        offset = 0
        for program in control['return_programs']:
            first = logical[program['sequence'][0]]
            case = cases[first['case_id']]
            store = build_store(case, 'H0')
            current_before = triples(store)
            original_before = triples(store, 0)
            reads = []
            for lid in program['sequence']:
                r = logical[lid]
                snapshot = 0 if r['phase'] == 'pre' else None
                state = triples(store, snapshot)
                labels = {k: [v] for k, v in case['labels'].items()}
                options = candidates(state, labels, r['question'])
                assert options == r['candidates'], 'ACTUAL_STORE_CANDIDATE_MISMATCH'
                payload = current_payload(state, [], labels, registry)
                generated = [{'kind': 'score', **request(r['question'], payload, p)} for p in options]
                assert [digest(q) for q in generated] == r['score_request_ids'], 'ACTUAL_STORE_INPUT_MISMATCH'
                count = len(options)
                scores = actual[offset:offset + count]
                offset += count
                assert [s['request_id'] for s in scores] == r['score_request_ids']
                selected = choose(options, [s['margin'] for s in scores])
                t = time.perf_counter()
                result = None if selected is None else store.query(selected['start_id'], selected['path'], snapshot)
                elapsed = time.perf_counter() - t
                target = ev[r['case_id'], r['query_id']]
                gold = target['pre'][r['history']] if snapshot == 0 else target['post']
                reads.append({'logical_id': lid, 'phase': r['phase'], 'state_sha256': digest(state),
                    'selected': selected, 'trace': result, 'cpu_query_seconds': elapsed,
                    'correct': result is not None and result['endpoint'] == gold['entity'],
                    'faithful': selected is not None and selected['start_id'] == gold['start'] and selected['path'] == gold['path'],
                    'request_ids': r['score_request_ids'], 'input_tokens_sha256': [digest(s['input_token_ids']) for s in scores],
                    'full_logits_sha256': [s['full_logits_sha256'] for s in scores], 'margins': [s['margin'] for s in scores]})
            assert triples(store) == current_before and triples(store, 0) == original_before
            assert reads[0]['state_sha256'] == reads[2]['state_sha256']
            assert reads[0]['input_tokens_sha256'] == reads[2]['input_tokens_sha256']
            lifecycle.append({'arm': arm, 'case_id': first['case_id'], 'query_id': first['query_id'], 'reads': reads,
                'retained_store_nonmutation': True, 'historical_answer_correct': reads[1]['correct'],
                'current_selections_equal': reads[0]['selected'] == reads[2]['selected'],
                'current_full_logits_equal': reads[0]['full_logits_sha256'] == reads[2]['full_logits_sha256']})
        assert offset == len(actual)
    assert len(lifecycle) == 48
    write_rows(out / 'ACTUAL_RETAINED_STORE_LIFECYCLE.jsonl.gz', lifecycle)
    dump(out / 'ACTUAL_RETAINED_STORE_SUMMARY.json', {arm: {
        'programs': sum(r['arm'] == arm for r in lifecycle),
        'historical_correct': sum(r['arm'] == arm and r['historical_answer_correct'] for r in lifecycle),
        'selection_mismatches': sum(r['arm'] == arm and not r['current_selections_equal'] for r in lifecycle),
        'logit_hash_mismatches': sum(r['arm'] == arm and not r['current_full_logits_equal'] for r in lifecycle)
    } for arm in ['STATE_BASE', 'STATE_0', 'STATE_1']})

    training = gzrows(SOURCE / 'inherited/PPR32_Repair_128/training/training_pairs.jsonl.gz')
    training_questions = {norm(json.loads(r['request']['messages'][1]['content'])['question']) for r in training}
    exposure = [{'case_id': cid, 'query_id': qid, 'question': e['question'], 'role': e['branch'],
                 'exact_normalized_training_overlap': norm(e['question']) in training_questions} for (cid, qid), e in ev.items()]
    overlap = collections.Counter(r['role'] for r in exposure if r['exact_normalized_training_overlap'])
    assert dict(overlap) == {'F': 21, 'Fdecoy': 12}
    write_rows(out / 'SOURCE_EXPOSURE_ALL_901.jsonl.gz', exposure)
    native = gzrows(out / 'NATIVE_GEMMA_ALL_384_TARGETS.jsonl.gz')
    for r in native:
        r['included_in_fixed_repeat_panel'] = r['case_id'] in control['repeat_cases']
    write_rows(out / 'NATIVE_GEMMA_ALL_384_WITH_REPEAT_STATUS.jsonl.gz', native)
    qwen = load(out / 'QWEN9_CHALLENGE_LEDGER.json')
    for r in qwen:
        cid, qid = r['case_id'], r['query_id']
        r['own_gemma_witness'] = next(n['witness'] for n in native if n['case_id'] == cid and n['query_id'] == qid)
        r['native_gemma_already_correct_four_histories'] = all(idx['NATIVE_GEMMA', cid, h, 'post', qid]['correct'] for h in HISTORIES)
        r['interpretation'] = 'fixed cross-family challenge; already-correct native cases are preservation, not new repairs'
    dump(out / 'QWEN9_PRESERVATION_JOIN.json', qwen)
    subject = []
    transitions = []
    for arm in ['STATE_0', 'STATE_1']:
        for baseline in ['NATIVE_GEMMA', 'CURRENT_TEXT', 'CURRENT_JSON', 'STATE_BASE']:
            for cid, case in cases.items():
                complete = lambda a: all(idx[a, cid, h, 'post', q['query_id']]['correct'] for h in HISTORIES for q in case['queries'])
                old, new = complete(baseline), complete(arm)
                subject.append({'candidate': arm, 'baseline': baseline, 'case_id': cid,
                    'baseline_complete': old, 'candidate_complete': new, 'fixed': not old and new, 'harmed': old and not new})
    for r in all_outcomes:
        if r['phase'] != 'pre':
            continue
        post = idx[r['arm'], r['case_id'], r['history'], 'post', r['query_id']]
        transitions.append({k: r[k] for k in ['arm', 'case_id', 'query_id', 'history']} |
                           {'pre_correct': r['correct'], 'post_correct': post['correct'],
                            'pre_entity': r['entity'], 'post_entity': post['entity'],
                            'pre_status': r['status'], 'post_status': post['status']})
    write_rows(out / 'SUBJECT_FIXES_HARMS.jsonl.gz', subject)
    write_rows(out / 'PRE_POST_TRANSITIONS.jsonl.gz', transitions)
    dump(out / 'SUPPLEMENTARY_RECEIPT.json', {'main_logical_outcomes': len(all_outcomes),
         'state_execution_traces': len(traces), 'real_retained_store_programs': len(lifecycle),
         'exposure_queries': len(exposure), 'overlap': dict(overlap), 'fixed_casebook_subjects': list(casebook),
         'subject_comparisons': len(subject), 'pre_post_transitions': len(transitions),
         'native_pre': 'NOT_MEASURED', 'scientific_success_inferred_from_software_checks': False})
    print('Supplementary real-store and complete-ledger checks passed')


if __name__ == '__main__':
    main()
