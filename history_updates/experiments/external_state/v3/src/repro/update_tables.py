"""Observation-bound presentation helpers for the two completed studies."""
import collections
import gzip
import json
from pathlib import Path

ENDPOINTS = ('REPAIR_0', 'REPAIR_1', 'QA_0', 'QA_1', 'PROGRAM_0', 'PROGRAM_1',
             'MELLO_CURRENT', 'GMELLO_LOCAL')
HISTORIES = ('H0', 'H1', 'S0', 'S1')
LABELS = {
    'REPAIR': 'Candidate scoring plus exact execution',
    'QA': 'Trained direct answer generation',
    'PROGRAM': 'Trained program generation plus the same exact executor',
    'MELLO_CURRENT': 'Supplied-record MeLLo adaptation',
    'GMELLO_LOCAL': 'Local-graph GMeLLo reimplementation',
}


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def matched_snapshots(records, *, expected_reads=1536):
    """Join actual question identities, not just equal aggregate denominators."""
    snapshots = {arm: {'pre': {}, 'post': {}} for arm in ENDPOINTS}
    for row in records:
        if row['arm'] not in snapshots:
            raise ValueError('Unknown endpoint would be omitted')
        if row['branch'] != 'J' or row['history'] not in HISTORIES:
            continue
        phase = row['phase']
        if phase not in ('pre', 'post') or type(row['correct']) is not bool:
            raise ValueError('Unsupported snapshot row')
        key = (row['case_id'], row['query_id'], row['history'])
        target = snapshots[row['arm']][phase]
        if key in target:
            raise ValueError('Duplicate snapshot question')
        target[key] = row
    result = {}
    population = None
    for arm, phases in snapshots.items():
        before, after = phases['pre'], phases['post']
        if len(before) != expected_reads or set(before) != set(after):
            raise ValueError('Missing endpoint, seed, or matched snapshot questions')
        if population is None:
            population = set(before)
        if set(before) != population:
            raise ValueError('Different target questions across endpoints')
        changed = wrong_original = updated_on_original = 0
        transitions = collections.Counter()
        for key in sorted(before):
            pre, post = before[key], after[key]
            if pre['question'] != post['question']:
                raise ValueError('Question text changed between snapshots')
            transitions[str(int(pre['correct'])) + '->' + str(int(post['correct']))] += 1
            differs = pre['expected_entity'] != post['expected_entity']
            changed += differs
            wrong_original += differs and not pre['correct']
            updated_on_original += (differs and pre['outcome']['status'] == 'VALID'
                                    and pre['outcome']['entity'] == post['expected_entity'])
        result[arm] = {
            'original_correct': sum(r['correct'] for r in before.values()),
            'current_correct': sum(r['correct'] for r in after.values()),
            'matched_reads': len(before), 'transitions': dict(sorted(transitions.items())),
            'exploratory': {'changed_answer_reads': changed,
                'incorrect_original_changed_answer_reads': wrong_original,
                'current_entity_on_original_input': updated_on_original,
                'unit': 'Matched target read; histories remain nested within subject',
                'status': 'Descriptive exact-row join, not a registered primary contrast'},
        }
    return result


def exposure_errors(records, *, expected_queries=901):
    """Retain read-level harms separately from robust-question harms."""
    lookup = {}
    for row in records:
        key = (row['seed'], row['arm'], row['case_id'], row['history'], row['query_id'])
        if key in lookup:
            raise ValueError('Duplicate exposure read')
        if row['seed'] not in ('seed0', 'seed1') or row['arm'] not in ('ISO', 'HIST', 'CURRENT_EXTRA'):
            raise ValueError('Unexpected exposure endpoint')
        if type(row['correct']) is not bool or type(row['refused']) is not bool:
            raise ValueError('Malformed exposure correctness')
        if row['refused'] != (row['selected'] is None):
            raise ValueError('Refusal differs from frozen selection')
        lookup[key] = row
    populations = {}
    for seed in ('seed0', 'seed1'):
        for arm in ('ISO', 'HIST', 'CURRENT_EXTRA'):
            keys = {(cid, h, q) for s, a, cid, h, q in lookup if (s, a) == (seed, arm)}
            if len(keys) != expected_queries * 6:
                raise ValueError('Missing exposure seed, control, or context')
            populations[seed, arm] = keys
    common = populations['seed0', 'ISO']
    if any(p != common for p in populations.values()):
        raise ValueError('Exposure arms have different question populations')
    result = {}
    for seed in ('seed0', 'seed1'):
        for baseline in ('ISO', 'CURRENT_EXTRA'):
            counts = collections.Counter()
            harmed_queries = set()
            for cid, history, qid in sorted(common):
                old = lookup[seed, baseline, cid, history, qid]
                new = lookup[seed, 'HIST', cid, history, qid]
                if old['correct'] and not new['correct']:
                    group = 'four_history' if history in HISTORIES else 'canonical_reference'
                    counts[group + '_new_errors'] += 1
                    category = 'refusals' if new['refused'] else 'wrong_entity'
                    counts[group + '_' + category] += 1
                    if history in HISTORIES:
                        harmed_queries.add((cid, qid))
            robust_harms = 0
            for cid, qid in harmed_queries:
                robust_harms += all(lookup[seed, baseline, cid, h, qid]['correct'] for h in HISTORIES)
            result[seed + '/HIST-minus-' + baseline] = {
                **{name: counts[name] for group in ('four_history', 'canonical_reference')
                   for name in (group + '_new_errors', group + '_refusals', group + '_wrong_entity')},
                'robust_question_harms': robust_harms,
                'read_errors_are_not_subject_counts': True,
            }
    return result


def check_baseline_coverage(scientific):
    endpoints = scientific['endpoints']
    if set(endpoints) != set(ENDPOINTS):
        raise ValueError('All eight endpoints, including PROGRAM and both seeds, are required')
    if set(scientific['primary_contrasts']) != {'QA', 'PROGRAM', 'MELLO_CURRENT', 'GMELLO_LOCAL'}:
        raise ValueError('Registered contrasts were omitted')
    for row in endpoints.values():
        if (row['subjects'], row['targets'], row['controls']) != (128, 384, 517):
            raise ValueError('Subject, target, or control denominator confusion')
    for key, size in [('original_94', 94), ('later_32', 32), ('overlap_15', 15)]:
        if scientific['historical_witness_cohorts'][key]['denominator'] != size:
            raise ValueError('Historical populations were pooled or changed')


def training_exposure(source, historical):
    training=list(rows(source/'training_join.jsonl.gz'))
    cases=json.loads((source/'evaluator_data/RUNTIME_CASES.json').read_bytes())
    normalize=lambda text:' '.join(text.lower().split())
    texts=collections.Counter(normalize(r['current_payload']['question']) for r in training)
    orders=json.loads((source/'sample_orders.json').read_bytes())
    if set(orders)!={'0','1'}:
        raise ValueError('Both recorded training orders are required')
    presented=collections.Counter(r['sample_id'] for r in training)
    if any(collections.Counter(row['sample_order'])!=presented for row in orders.values()):
        raise ValueError('Training order drops or changes presentations')
    questions=[(c['case_id'],q['query_id'],normalize(q['question'])) for c in cases for q in c['queries']]
    overlap={(cid,qid,text) for cid,qid,text in questions if text in texts}
    old=json.loads(historical.read_bytes())
    expected={(r['case_id'],r['query_id'],normalize(r['question'])) for r in old['overlaps']}
    if overlap!=expected or len(overlap)!=old['overlap_occurrences']:
        raise ValueError('Training overlap does not match the historical case/question identities')
    controls={text for _,qid,text in questions if not qid.startswith('J')}
    draws=sum(count for text,count in texts.items() if text in controls)
    if (len(training),len(texts),len({r['question_id'] for r in training}),draws,len(overlap))!=(12288,4148,4273,71,33):
        raise ValueError('Training exposure units differ from their source records')
    if any(qid.startswith('J') for _,qid,_ in overlap):
        raise ValueError('Unexpected target text overlap')
    return {'status':'PASS','presentations_per_fit':len(training),
        'unique_question_identities':4273,'unique_normalized_question_texts':4148,
        'overlapping_training_presentations':draws,'overlapping_cohort_control_occurrences':len(overlap),
        'shared_normalized_texts':len({text for _,_,text in overlap}), 'target_text_overlaps':0,
        'both_seed_order_multisets_match_all_presentations':True,
        'original_order_sha256':{key:orders[key]['original_order_sha256'] for key in ('0','1')},
        'matching_rule':'lowercase and whitespace normalization; exact case/query join to the historical exposure ledger',
        'overlap_mapping':[{'text':text,'training_presentations':texts[text],
            'cohort_controls':[{'case_id':cid,'query_id':qid} for cid,qid,t in sorted(overlap) if t==text]}
            for text in sorted({text for _,_,text in overlap})],
        'no_target_text_overlap_is_not_no_exposure':True}


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '|' + '|'.join('---' for _ in headers) + '|',
                       *['| ' + ' | '.join(map(str, record)) + ' |' for record in records]])


def presentations(data):
    scientific = data['same_case_baselines']['scientific']
    check_baseline_coverage(scientific)
    matched = data['same_case_baselines']['matched_snapshots']
    baselines = []
    snapshots = []
    for arm in ENDPOINTS:
        result = scientific['endpoints'][arm]
        baselines.append([arm, result['complete_subjects'], result['robust_targets'], result['robust_controls']])
        snapshot = matched[arm]
        snapshots.append([arm, snapshot['original_correct'], snapshot['current_correct'], snapshot['matched_reads']])
    exposure = data['history_exposure']['metrics']['metrics']
    return {
        'baseline_complete': (['Endpoint', 'Complete subjects /128', 'Robust targets /384', 'Robust controls /517'], baselines),
        'baseline_snapshots': (['Endpoint', 'Original targets correct', 'Current targets correct', 'Matched reads in each column'], snapshots),
        'history_exposure': (['Seed', 'Context', 'Complete subjects /128', 'Robust targets /384', 'Robust controls /517', 'Guarded witnesses /384'],
            [[seed, arm, exposure[seed+'/'+arm]['whole_subject'], exposure[seed+'/'+arm]['robust_target'],
              exposure[seed+'/'+arm]['robust_control'], exposure[seed+'/'+arm]['guarded_witnesses']]
             for seed in ('seed0', 'seed1') for arm in ('ISO', 'HIST', 'CURRENT_EXTRA')]),
        'baseline_contrasts': (['Comparator', 'Scorer minus comparator (pp)', '98.75% paired interval (pp)', 'Registered advantage criterion passed'],
            [[arm, f"{100*r['difference']:.2f}", f"[{100*r['interval'][0]:.2f}, {100*r['interval'][1]:.2f}]", str(r['substantively_positive'])]
             for arm, r in scientific['primary_contrasts'].items()]),
    }


def render(data):
    products = {'updates_tables.json': json.dumps(data, ensure_ascii=False, indent=2) + '\n'}
    tables = presentations(data)
    products['updates_tables.md'] = '\n\n'.join('## ' + name.replace('_', ' ').capitalize() + '\n\n' + table(*value)
                                               for name, value in tables.items()) + '\n'
    def tex(value):
        return ''.join({'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#'}.get(c, c) for c in str(value))
    for name, (headers, records) in tables.items():
        lines = ['% Generated from complete saved-observation replays.', r'\begin{tabular}{' + 'l'*len(headers) + '}',
                 r'\hline', ' & '.join(map(tex, headers)) + r' \\', r'\hline']
        lines += [' & '.join(map(tex, row)) + r' \\' for row in records]
        products[name + '.tex'] = '\n'.join(lines + [r'\hline', r'\end{tabular}']) + '\n'
    return products


def collect(baseline, exposure, bridge):
    def load(path):return json.loads(Path(path).read_bytes())
    scientific=load(baseline/'SCIENTIFIC_RESULTS.json')
    check_baseline_coverage(scientific)
    return {'schema':'TWO_STUDY_REPLAY_TABLES_V1',
        'history_exposure':{'metrics':load(exposure/'METRICS.json'),
                            'error_categories':load(exposure/'ERROR_CATEGORIES.json')},
        'same_case_baselines':{'scientific':scientific,
            'matched_snapshots':load(baseline/'MATCHED_SNAPSHOTS.json'),
            'training_exposure_join':load(baseline/'TRAINING_EXPOSURE_JOIN.json'),
            'component_audit':load(baseline/'BEHAVIOR_COMPONENT_AUDIT.json'),
            'workload':load(baseline/'COST_TIMING_WORKLOAD.json')},
        'recovered_bridge':{'recomposition':load(bridge/'a10/A10_RECOMPOSITION.json'),
            'neural_compiler':load(bridge/'NSR1_ANALYSIS.json'),
            'schedule_join':load(bridge/'a10_schedule_reconciliation.json'),
            'native_cache_effects':load(bridge/'carrier_mediation_recomputed.json')},
        'sampling':'Original subject/design clusters and frozen procedures; reused observations are not new replications'}


def reconstruct(output, repository):
    import hashlib
    output, repository=Path(output),Path(repository)
    data=collect(output/'same_case_baselines/analysis',output/'history_exposure/analysis',
                 output/'history_exposure/bridge')
    products=render(data)
    for name,text in products.items():
        expected=repository/('paper/tables' if name.endswith('.tex') else 'data/results')/name
        if expected.read_text(encoding='utf8')!=text:
            raise ValueError('Reconstructed update table differs: '+name)
    page=(repository/'docs/results/new_readout_results.md').read_text(encoding='utf8')
    begin,end='<!-- BEGIN GENERATED UPDATE TABLES -->','<!-- END GENERATED UPDATE TABLES -->'
    if page.count(begin)!=1 or page.count(end)!=1 or page.split(begin)[1].split(end)[0].strip()+'\n'!=products['updates_tables.md']:
        raise ValueError('Update result page differs from regenerated tables')
    target=output/'tables';target.mkdir()
    for name,text in products.items():
        with (target/name).open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
    return {'status':'PASS','baseline_endpoints':8,'exposure_arm_seed_outcomes':6,
            'generated_files':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in products.items()},
            'all_original_numerical_fields_retained':True}
