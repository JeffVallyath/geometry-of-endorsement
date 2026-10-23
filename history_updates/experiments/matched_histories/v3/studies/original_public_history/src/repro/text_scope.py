"""Reconstruct completed textual-breadth rows from saved scores on CPU."""
import argparse
import json
from pathlib import Path

from .common import ROOT, load_json, new_output
from .release_inputs import verify, restore
from .fresh_history import equivalent, run_offline

SOURCE = ROOT / 'reproducibility/text_scope'


def export(output, source=SOURCE):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a fresh output outside scientific inputs')
    binding = verify(source, 'text_scope')
    new_output(output)
    restored = output / 'source'
    restore(source, binding, restored)
    run_offline(source / 'tools/replay.py', [restored, output / 'analysis'], output)
    comparisons = []
    expected = restored / 'runtime/analysis_final'
    for path in sorted(expected.rglob('*.json')):
        name = path.relative_to(expected).as_posix()
        if not equivalent(load_json(output / 'analysis' / name), load_json(path)):
            raise ValueError('Textual breadth reconstruction differs: ' + name)
        comparisons.append(name)
    table = load_json(output / 'analysis/MAIN_TABLE.json')
    contrasts = load_json(output / 'analysis/PLANNED_COMPARISONS.json')
    if len(table) != 12 or not all(c['status'] == 'NOT_ASSESSABLE' for c in contrasts):
        raise ValueError('Row inventory or failed bridge disposition changed')
    absent = [r for r in table if r['row_id'] == 'qwen32_t4']
    if len(absent) != 2 or not all(r['status'] == 'NOT_MEASURED' and r['witness_rate'] is None for r in absent):
        raise ValueError('A cached-path technical failure became a measured result')
    rows = [dict(row_id=r['row_id'], method=r['method'], W=r['W'], denominator=r['opportunities'])
            for r in table if r['status'] == 'COMPLETE_FINAL_ROW']
    report = dict(status='PASS', compact_members=len(binding['files']), saved_outputs_compared=comparisons,
                  newly_measured_designs_per_row=64, new_design_sampling=False,
                  new_final_score_records=110976, measured_method_rows=rows,
                  historical_rows_reconstructed=4, failed_planned_contrasts=len(contrasts),
                  cached_qwen32_not_measured=True, scaling_inference=False,
                  integer_classification_equality='exact', finite_float_atol=1e-12,
                  model_imports_and_network_blocked=True, inference=False)
    (output / 'VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))


if __name__ == '__main__':
    main()
