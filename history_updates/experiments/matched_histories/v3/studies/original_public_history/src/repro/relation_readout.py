"""Reconcile relation-readout aggregates, stored null draws and factual controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, load_json, new_output
from .release_inputs import verify


def tail_check(record, p_key):
    values = np.asarray(record['values'], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) != record['B']:
        raise ValueError('Incomplete or nonfinite permutation sample')
    count = int(np.sum(values >= record['observed']))
    p = (count + 1) / (len(values) + 1)
    if count != record['k_ge_observed'] or p != record[p_key]:
        raise ValueError('Saved permutation tail accounting differs')
    return {'draws': len(values), 'exceedances': count, 'p': p}


def check(root=ROOT):
    root = Path(root)
    verify(root / 'reproducibility/relation_readout', 'relation_readout', root)
    figure = load_json(root / 'artifacts/figures/figure_data.json')
    reference = load_json(root / 'artifacts/m1/development_reference.json')
    rows, nulls = [], {}
    for name, model in figure['models'].items():
        if model['sample_counts'] != dict(pilot_train=1500, pilot_select=300, pilot_eval=500, pilot_eval_boards=125):
            raise ValueError('Development sample accounting changed')
        if model['pilot_eval_used_for_selection'] or model['selected_layer'] != {'llama': 19, 'gemma': 27}[name]:
            raise ValueError('Frozen layer-selection boundary changed')
        nulls[name] = {probe: tail_check(value, 'p') for probe, value in model['permutation_null'].items()}
        for probe, value in model['evaluations'].items():
            rows.append(dict(model=name, probe=probe, layer=model['selected_layer'],
                             **{key: value[key] for key in ('auroc', 'I_b', 'ci_low', 'ci_high')}))
    for probe in ('difference_in_means', 'logistic'):
        value = figure['models']['llama']['evaluations'][probe]
        if value['I_b'] != reference['metrics'][probe + '_I_b']:
            raise ValueError('Development reference and figure disagree')
    large = load_json(root / 'artifacts/figures/permutation_null_llama_B10000.json')
    nulls['llama_10000'] = {probe: tail_check(value, 'p_empirical') for probe, value in large['probes'].items()}
    from relation_study.m1.runner import _truth_control_summary
    truth = _truth_control_summary(root / 'artifacts/truth/v2_results.json', reference['model']['id'])
    if truth['terminal_disposition'] != 'PASS':
        raise ValueError('Factual positive-control checks failed')
    truth = {key: truth[key] for key in ('terminal_disposition', 'checks', 'selected_layer',
                                       'primary_test_T', 'exact_permutation_p', 'directional_consensus_C')}
    return dict(status='PASS', scope='Published aggregates and saved null-tail reconciliation; no probe refit or model execution',
                relation=rows, permutation=nulls, llama_truth=truth,
                development_only=True, audited_confirmatory_result_available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    result = check()
    destination = new_output(args.output)
    (destination / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (destination / 'relation_table.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['relation'][0]))
        writer.writeheader()
        writer.writerows(result['relation'])
    print(json.dumps({key: result[key] for key in ('status', 'scope', 'permutation')}, indent=2))


if __name__ == '__main__':
    main()
