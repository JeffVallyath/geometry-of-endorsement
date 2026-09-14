"""Explicit claim-to-analysis mapping for the curated Results ledger.

No reported effect is hard-coded here. Values are selected from independently
replayed saved evidence. Scientific identity, not numerical similarity, selects
the record that supports each sentence/table cell.
"""
from __future__ import annotations


def from_replay(result):
    studies = result['representation_studies']
    source = result['source_independence']
    output = {'representation': result['representation'], 'fragility': {},
              'fingerprint': {}, 'full_state': {}, 'specificity': {}, 'counterfactual': {},
              'editing': {}, 'source_best_bundle_percent':
              100 * max(r['all34']['all_origin_bundle_correct'] for r in source['source_independence'])}
    for actor in ('llama', 'gemma'):
        fragility = studies['fragility'][actor]['increment_ci']
        output['fragility'][actor] = dict(zip(('gain', 'lower', 'upper'), fragility))
        fingerprint = studies['fingerprint'][actor]
        output['fingerprint'][actor] = {'correlation': fingerprint['correlation'],
            'similarity': fingerprint['median_per_direction_fingerprint_similarity']}
        output['full_state'][actor] = {'situations': studies['full_state'][actor + '/primary_replication']['point'],
            'checkerboards': studies['full_state'][actor + '/secondary_board_population']['point']}
        output['specificity'][actor] = {
            corpus: {comparator: studies['specificity'][actor + '/' + corpus_name + '/' + source_name]
                     for comparator, source_name in (('truth', 'factual_true_false'), ('sentiment', 'sentiment_valence'))}
            for corpus, corpus_name in (('valueprism', 'ValuePrism'), ('arguments', 'AMPERE++'))}
    for name, condition in (('rank_one', 'ORIGINAL_FROZEN_DIM@0.75'),
                            ('interpolation', 'NATURAL_FULL_STATE_INTERPOLATION@0.75'),
                            ('replacement', 'FULL_STATE_SOURCE_PATCH')):
        csr = studies['counterfactual']['primary/' + condition]['csr']
        output['counterfactual'][name] = {'recovery': csr['mean'], 'lower': csr['ci_low'], 'upper': csr['ci_high']}
    output['counterfactual']['direct_margin_hit_percent'] = 100 * studies['counterfactual']['hits']['ORIGINAL_FROZEN_DIM']['within_tolerance']
    for record in source['functional']:
        if record['architecture'] != 'INV_COMPLETE_SINGLE' or record['reader'] != 'R0':
            continue
        actor = record['actor']
        output['editing'][actor] = {}
        for seed, statistics in record['seeds'].items():
            output['editing'][actor]['seed' + seed] = {
                metric: 100 * statistics[metric]
                for metric in ('single_changed', 'max_repeat_disagreement', 'min_restoration', 'ABC_harm')}
    output['training_seeds'] = sorted(int(k) for k in source['functional'][0]['seeds'])
    return output
