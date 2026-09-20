"""CPU-only source-inherited primary analysis and frozen semantic conjunction."""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path

from text_scope.common import read, save, sha, METHODS, independent_evaluator
from text_scope.metrics import decode_score, root_metrics, summarize
from text_scope.sensitivities import evaluate as inherited_sensitivities, summarize as summarize_sensitivities


def load_raw(folder):
    done = read(folder/'DONE.json')
    for name, hashed in done['files'].items():
        assert sha(folder/name) == hashed
    raw = {}
    with (folder/'RAW_SCORES.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['context'], row['query_id']
            assert key not in raw
            raw[key] = row
    assert len(raw) == done['logical_scores']
    return raw


def joint_semantics(design, index, method, mapping, lengths, primary, evaluator):
    questions = {q['query_id']:q for q in design['questions']}
    values = design['root']['terminal']['values']
    changed = [sum(s['values'][c['actor']][c['project']] != c['value'] for c in design['root']['commands'])
               for s in design['root']['sources']]
    allowed = [list(p) for p in combinations(range(8),2)
               if changed[p[0]] == changed[p[1]] and lengths[f'o{p[0]}/{method}'] == lengths[f'o{p[1]}/{method}']]
    primary = {d['query_id']:d for d in primary}
    rows = []
    def r(origin, qid): return index[f'o{origin}/{method}', qid]
    def correct(score, qid):
        return score['valid'] and score['prediction'] == evaluator.truth(values, questions[qid]['spec'])
    for qid,m in mapping.items():
        if m['draw'] != '0': continue
        twin = m['twin']
        targets = [qid, twin]
        direct = m['direct']
        logical = m['same']+m['cross']+mapping[twin]['same']+mapping[twin]['cross']
        operands = m['operands']+mapping[twin]['operands']
        direct_ok = all(correct(r(o,q),q) for o in range(8) for q in direct)
        reference_ok = all(correct(index[k,q],q) for k in ('NATIVE_FINAL',f'o0/{method}') for q in targets)
        logical_ok = all(correct(r(o,q),q) for o in range(8) for q in logical)
        base_ok = all(primary[q]['AC'] for q in targets)
        def pair_valid(pair): return all(r(o,q)['valid'] for o in pair for q in targets)
        def disagree(pair):
            a,b = pair
            return pair_valid(pair) and all(r(a,q)['prediction'] != r(b,q)['prediction'] for q in targets) \
                and all(r(o,qid)['prediction'] == r(o,twin)['prediction'] for o in pair)
        pair_witnesses = [pair for pair in allowed if disagree(pair)]
        strongest = pair_witnesses if direct_ok and reference_ok and logical_ok else []
        # Each high-margin pair is a SUBSET of the exact already-qualified pairs;
        # no pair rematching, direction change or cross-format substitution.
        high = []
        for pair in strongest:
            margins = []
            for q in targets:
                gold = evaluator.truth(values, questions[q]['spec'])
                p = [r(o,q)['probabilities'][gold] for o in pair]
                margins.append(min(p) <= .05 and max(p) >= .95 and all(r(o,q)['mass'] >= .90 for o in pair))
            if all(margins) and all(primary[q]['strong'] for q in targets):
                high.append(pair)
        required = [(f'o{o}/{method}',q) for o in range(8) for q in direct+logical]
        required += [(k,q) for k in ('NATIVE_FINAL',f'o0/{method}') for q in targets]
        target_valid_pair = any(pair_valid(pair) for pair in (allowed or list(combinations(range(8),2))))
        failures = dict(invalid_answer=any(not index[k]['valid'] for k in required) or not target_valid_pair,
            failed_direct_readout=not direct_ok, failed_native_noop_target=not reference_ok,
            failed_other_logical_question=not logical_ok, unavailable_matched_pair=not allowed,
            no_target_disagreement=not pair_witnesses)
        reason = next((name for name,failed in failures.items() if failed), 'survivor')
        assert (reason == 'survivor') == bool(strongest)
        rows.append(dict(root_id=design['root']['root_id'], method=method, semantic=m['semantic'],
            target_ids=targets, direct_ids=direct, logical_ids=logical, operand_ids=operands,
            negated_ids=m['negated'], changed_fact_counts=changed,
            prefix_token_lengths=[lengths[f'o{o}/{method}'] for o in range(8)], allowed_pairs=allowed,
            all_direct=direct_ok, reference=reference_ok, all_logical=logical_ok,
            both_format_G0=base_ok, matched_crossformat_G0_pairs=pair_witnesses if base_ok else [],
            strongest_pairs=strongest, high_margin_pairs=high, attrition_reason=reason, failures=failures,
            all_history_targets_correct=all(correct(r(o,q),q) for o in range(8) for q in targets),
            invalid_target_responses=sum(not r(o,q)['valid'] for o in range(8) for q in targets),
            negated_correct=sum(correct(r(o,q),q) for o in range(8) for q in m['negated']), negated_denominator=32))
    assert len(rows) == 9
    return rows


def analyze_case(runtime, rid, phase='final'):
    evaluator = independent_evaluator(runtime)
    design = read(runtime/'cpu/designs'/(rid+'.json'))
    frozen = read(runtime/'ANALYSIS_FREEZE.json')['mappings'][rid]
    raw = load_raw(runtime/'results/qwen32_t4'/phase/rid)
    assert len(raw) == 578
    index = {k:decode_score(v['score']) for k,v in raw.items()}
    lengths = {k:v['prefix_length'] for k,v in frozen['contexts'].items()}
    result = {}
    for method in METHODS:
        metrics, primary = root_metrics(evaluator, design, index, method)
        semantic = joint_semantics(design, index, method, frozen['mapping'], lengths, primary, evaluator)
        coded = inherited_sensitivities(runtime,evaluator,design,index,method,lengths,primary)
        result[method] = dict(metrics=metrics, primary=primary, semantic=semantic, inherited_coded=coded)
    return result


def choose_cases(order, levels):
    selected = []
    for level in levels:
        for rid in order:
            if rid in level and rid not in selected and len(selected) < 4: selected.append(rid)
    seed = 'QWEN32-REPEAT-FILL-20260919-v1'
    remaining = sorted((r for r in order if r not in selected),
                       key=lambda rid:hashlib.sha256((seed+'\0'+rid).encode()).hexdigest())
    return selected + remaining[:4-len(selected)]


def run(runtime, output):
    freeze = read(runtime/'ANALYSIS_FREEZE.json')
    order = freeze['case_order']
    cases = {rid:analyze_case(runtime, rid) for rid in order}
    summary = {}
    levels = [set() for _ in range(4)]
    for rid,case in cases.items():
        save(output/'cases'/(rid+'.json'), case)
        for method in METHODS:
            sem = case[method]['semantic']
            for level, key in enumerate(('high_margin_pairs','strongest_pairs','matched_crossformat_G0_pairs')):
                if any(r[key] for r in sem): levels[level].add(rid)
            if case[method]['metrics']['W']: levels[3].add(rid)
    for method in METHODS:
        metrics = [cases[r][method]['metrics'] for r in order]
        semantic = [row for r in order for row in cases[r][method]['semantic']]
        primary = [row for r in order for row in cases[r][method]['primary']]
        summary[method] = dict(primary=summarize(metrics), semantic_opportunities=576,
            inherited_coded_controls=summarize_sensitivities([row for r in order for row in cases[r][method]['inherited_coded']]),
            semantic_all_history_correct=sum(r['all_history_targets_correct'] for r in semantic),
            semantic_counts={key:sum(bool(r[key]) for r in semantic) for key in
                ('both_format_G0','all_direct','reference','all_logical','allowed_pairs','matched_crossformat_G0_pairs','strongest_pairs','high_margin_pairs')},
            designs_with_survivor={key:sum(any(row[key] for row in cases[r][method]['semantic']) for r in order)
                for key in ('matched_crossformat_G0_pairs','strongest_pairs','high_margin_pairs')},
            attrition=dict(Counter(r['attrition_reason'] for r in semantic)),
            overlapping_failures={key:sum(r['failures'][key] for r in semantic) for key in semantic[0]['failures']},
            evaluation_decision_change=dict(coded_currentfact_reference_pass=sum(r['AC'] for r in primary),
                coded_pass_to_history_failure=sum(r['W'] for r in primary),
                denominator=1152, claim='inspected questions only, not unseen-error prediction'),
            negated_control_accuracy=dict(correct=sum(cases[r][method]['semantic'][0]['negated_correct'] for r in order), denominator=64*32))
        remaining=576;flow=[]
        for gate in freeze['attrition_order']:
            removed=sum(r['attrition_reason']==gate for r in semantic)
            flow.append(dict(gate=gate,entering=remaining,count=removed,remaining=remaining-removed))
            remaining-=removed
        assert remaining==0
        summary[method]['ordered_attrition_flow']=flow
        from text_scope.metrics import interval
        summary[method]['semantic_rates']={key:dict(
            estimate=sum(bool(row[key]) for row in semantic)/576,
            interval95=interval([sum(bool(row[key]) for row in cases[r][method]['semantic'])/9 for r in order],.95))
            for key in ('matched_crossformat_G0_pairs','strongest_pairs','high_margin_pairs')}
        zero = {}
        for key in ('strongest_pairs','high_margin_pairs'):
            for population in ('all_designs','any_both_format_G0','any_full_logical_certification'):
                subset = order if population == 'all_designs' else [rid for rid in order if any(
                    row['both_format_G0'] if population == 'any_both_format_G0' else
                    row['all_direct'] and row['reference'] and row['all_logical']
                    for row in cases[rid][method]['semantic'])]
                events = sum(any(row[key] for row in cases[rid][method]['semantic']) for rid in subset)
                if events == 0:
                    zero[key+'/'+population] = dict(n_designs=len(subset), events=0,
                        one_sided95=1-.05**(1/len(subset)) if subset else None,
                        one_sided99375=1-.00625**(1/len(subset)) if subset else None,
                        assumption=freeze['case_independence_bound_assumption'])
        summary[method]['zero_event_bounds'] = zero
    selection = dict(roots=choose_cases(order,levels), levels=[sorted(s) for s in levels],
        analysis_freeze_sha256=sha(runtime/'ANALYSIS_FREEZE.json'),
        selection_for_reproducibility_not_population_estimation=True)
    save(runtime/'REPEAT_SELECTION.json',selection)
    save(output/'SUMMARY.json',summary)
    save(output/'ANALYSIS_BINDING.json',dict(analysis_freeze_sha256=sha(runtime/'ANALYSIS_FREEZE.json'),
        source_scores={rid:sha(runtime/'results/qwen32_t4/final'/rid/'RAW_SCORES.jsonl') for rid in order}))
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.runtime,a.output)
