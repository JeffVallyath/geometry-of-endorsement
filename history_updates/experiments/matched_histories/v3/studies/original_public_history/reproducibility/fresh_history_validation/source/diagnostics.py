"""Explicitly post-hoc checks, kept separate from the frozen primary/utility code."""
import argparse
from collections import Counter
from pathlib import Path
import json

import analyze as A
from bindings import load, save
from replay import metadata, check_case
from panels import load_panel, compare, current_rows


def reconstruct(source):
    source = Path(source)
    run = source / 'scientific'
    decisions = {(r['root_id'], r['seed'], r['draw'], p): r
                 for p in ('A', 'B') for r in load(run / ('DECISIONS_' + p + '.json'))}
    witnesses = []
    example = None
    for design in metadata(source):
        root, questions, plan = design
        rid = root['root_id']
        panel, index = check_case(run / 'final' / rid, design)
        qs = {q['query_id']: q for q in questions}
        for row in A.primary(*design, index):
            if not row['both_matched']:
                continue
            condition = A.CONDITIONS[row['seed']]
            token_pairs = []
            for left, right in row['matched_pairs']:
                prefixes = [panel['contexts'][f'o{o}/{condition}']['cache_identity']['prefix_ids']
                            for o in (left, right)]
                if Counter(prefixes[0]) == Counter(prefixes[1]):
                    token_pairs.append([left, right])
            pair = plan['selected_anchor_pair']
            qid = row['query_id']
            chosen = [index[f'o{o}/{condition}', qid] for o in pair]
            item = dict(root_id=rid, seed=row['seed'], draw=row['draw'], query_id=qid,
                        target_is_withheld=qid in plan['utility'][str(row['draw'])]['withheld_queries'],
                        validator_A_accept=decisions[rid, row['seed'], row['draw'], 'A']['accept'],
                        validator_B_accept=decisions[rid, row['seed'], row['draw'], 'B']['accept'],
                        selected_pair=pair,
                        selected_pair_target_error=any(not r['valid'] or r['prediction'] != row['gold'] for r in chosen),
                        selected_pair_valid_disagreement=all(r['valid'] for r in chosen) and chosen[0]['prediction'] != chosen[1]['prediction'],
                        matched_witness_pairs=row['matched_pairs'], exact_token_inventory_pairs=token_pairs)
            witnesses.append(item)
            if rid == 'FHV1-FINAL-0017' and example is None:
                repeated = load_panel(run / 'repeat' / rid)
                comparison = compare(panel, repeated)
                raw = current_rows(panel)
                raw_repeat = current_rows(repeated)
                histories = []
                controls = plan['gates']['direct'] + plan['gates']['targets'][qid]['same_pair'] + plan['gates']['targets'][qid]['cross_pair']
                for origin in range(8):
                    context = f'o{origin}/{condition}'
                    control_records = [dict(query_id=k, question=qs[k]['text'], gold=A.truth(root['terminal']['values'], qs[k]['spec']),
                                            prediction=index[context, k]['prediction'], valid=index[context, k]['valid']) for k in controls]
                    histories.append(dict(origin=origin, changed_facts=plan['changed_counts'][origin],
                        prefix_length=len(panel['contexts'][context]['cache_identity']['prefix_ids']),
                        target=raw[context, qid], repeated_target=raw_repeat[context, qid], controls=control_records))
                example = dict(root_id=rid, post_hoc_illustration=True, root=root, question=qs[qid],
                    witness=row, diagnostic=item, histories=histories,
                    native_target=raw['NATIVE_FINAL', qid], repeated_native_target=raw_repeat['NATIVE_FINAL', qid],
                    repeat_comparison=comparison, all_questions=questions, original_panel='scientific/final/'+rid+'/SCORES.json.gz',
                    repeated_panel='scientific/repeat/'+rid+'/SCORES.json.gz')
    withheld = [r for r in witnesses if r['target_is_withheld']]
    token_matched = [r for r in witnesses if r['exact_token_inventory_pairs']]
    return dict(classification='POST_HOC_DIAGNOSTICS_NOT_FROZEN_ENDPOINTS',
                primary_observations=len(witnesses), primary_cases=len({r['root_id'] for r in witnesses}),
                withheld_target_observations=len(withheld), withheld_target_cases=len({r['root_id'] for r in withheld}),
                withheld_target_both_accept=sum(r['validator_A_accept'] and r['validator_B_accept'] for r in withheld),
                withheld_target_selected_pair_errors=sum(r['selected_pair_target_error'] for r in withheld),
                withheld_target_selected_pair_disagreements=sum(r['selected_pair_valid_disagreement'] for r in withheld),
                exact_token_inventory_observations=len(token_matched), exact_token_inventory_cases=len({r['root_id'] for r in token_matched}),
                definition='Exact token inventory means equal token-ID multisets, including multiplicity, within an already count/length-matched witness pair. Token order and relation binding remain different.',
                witnesses=witnesses), example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    diagnostics, example = reconstruct(args.source)
    save(args.output / 'DIAGNOSTICS.json', diagnostics)
    save(args.output / 'EXAMPLE_0017.json', example)
    print(json.dumps({k:v for k,v in diagnostics.items() if k != 'witnesses'}))


if __name__ == '__main__':
    main()
