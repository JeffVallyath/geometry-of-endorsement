"""Complete-earlier-panel analysis with exact raw/physical custody validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import rfr5_analyze as V5A
import rfr5_metrics as A
import rsi6_common as S
import rsi6_metrics as M
import rsi6_verify as V
import rsi6_continuation as CONT


def text_file(path, text):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        f.write(text)


def diagnostics(rows):
    """Saved-direct logic remains diagnostic; invalid operands remain missing."""
    result = A.recomposition(rows)
    lookup = {(r['actor'], r['scene_id'], r['condition'], r['reader'], r['program'], r['order'], r['query_id']): r for r in rows}
    for row in result:
        key = tuple(row[k] for k in ('actor', 'scene_id', 'condition', 'reader', 'program', 'order'))
        operands = [lookup[key + (qid,)] for qid in row['direct_query_ids']]
        if not all(M.valid(r) and M.valid(r, True) for r in operands):
            row.update(operand_coverage=False, recomposed_prediction=None, recomposed_source_prediction=None,
                       agreement_with_direct=None)
    return result


def produce(panel_rows, output, *, retained=None):
    """Called only after raw/custody verification in run(); also used by CPU fixtures."""
    output = Path(output)
    primary = {}
    panels = list(panel_rows)
    if retained is not None:
        if panels not in ([], ['C']):
            raise ValueError('Retained A/B reporting may add only C')
        for name, digest in retained['manifest']['files'].items():
            if name.startswith(('A/', 'B/')):
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError('Never overwrite a retained report member')
                shutil.copyfile(retained['root'] / name, target)
                if S.C.sha(target) != digest:
                    raise ValueError('Retained A/B report copy changed')
        panels = ['A', 'B'] + panels
        primary.update(A=retained['claims']['G1_primary_effects'],
                       B=retained['claims']['G3_primary_effects'])
    for panel, rows in panel_rows.items():
        dest = output / panel
        dest.mkdir()
        diagnostic = diagnostics(rows)
        S.C.write_rows(dest / 'RECOMPOSITION_DIAGNOSTIC.jsonl', diagnostic)
        S.C.write_rows(dest / 'ERROR_AND_INVALID_ROWS.jsonl', [r for r in rows if not M.correct(r)])
        S.C.dump(dest / 'GROUPED_FAMILY_METRICS.json', M.grouped_rates(rows,
            ('actor', 'condition', 'reader', 'program', 'order', 'family'),
            unit='root_id' if panel == 'B' else 'scene_id'))
        if panel == 'B':
            result = M.source_independence(rows)
            primary[panel] = result['primary']
            S.C.dump(dest / 'SOURCE_INDEPENDENCE.json', result)
            S.C.dump(dest / 'NATIVE_AGREEMENT_VS_TRUTH.json', M.native_agreement(rows))
            S.C.dump(dest / 'PER_ORIGIN_METRICS.json', M.grouped_rates(rows,
                ('actor', 'condition', 'reader', 'origin'), unit='root_id'))
            S.C.dump(dest / 'JOINT_SOURCE_FINAL_TRUTH_CELLS.json', M.grouped_rates(
                [r for r in rows if r['family'] in ('same', 'both', 'either')],
                ('actor', 'condition', 'reader', 'family', 'truth_cell'), unit='root_id'))
            S.C.dump(dest / 'DIAGNOSTIC_COVERAGE.json', {
                'queries': len(diagnostic), 'missing_operand_queries': sum(not r['operand_coverage'] for r in diagnostic),
                'direct_queries_used': len({(r['actor'], r['scene_id'], r['condition'], r['reader'], q)
                    for r in diagnostic for q in r['direct_query_ids']}),
                'root_grouping_required': True, 'never_primary_model_output_replacement': True})
        else:
            summaries = A.summaries(rows)
            S.C.dump(dest / 'FULL_FAMILY_METRICS.json', summaries)
            S.C.dump(dest / 'EQUALITY_TRUTH_TABLE.json', A.truth_cells(rows))
            S.C.dump(dest / 'RECOMPOSITION_SUMMARY.json', A.recomposition_summaries(diagnostic))
            if panel == 'A':
                primary[panel] = A.primary_contrasts(rows)
                S.C.dump(dest / 'PRIMARY_CONTRASTS.json', primary[panel])
                S.C.dump(dest / 'SECONDARY_CONTRASTS.json', A.secondary_contrasts(rows))
                S.C.dump(dest / 'FULL_FAMILY_CONTRASTS.json', A.full_family_contrasts(summaries))
                S.C.dump(dest / 'SOURCE_READER_EFFECTS.json', V5A.source_reader_changes(rows))
            else:
                S.C.dump(dest / 'REPEAT_RESTORATION_PER_SCENE.json', M.program_retention(rows))
    claims = {'historical_V5': 'PREPARED_NO_EFFICACY; zero historical fresh outcomes',
              'completed_panels': panels,
              'missing_panels': {p: 'NOT_MEASURED' for p in S.PANELS if p not in panels},
              'G1_primary_effects': primary.get('A'), 'G3_primary_effects': primary.get('B'),
              'functional_progress': M.functional_progress(panel_rows.get('A', []), panel_rows.get('C', []),
                  prior_a=retained['claims']['functional_progress'] if retained is not None else None),
              'R2_is_instruction_assisted': True, 'historical_V4_relabelled': False,
              'old_comparative_F2_rerun': False, 'fresh_F4_rerun': False,
              'native_is_not_truth_or_accuracy_ceiling': True,
              'no_equivalence_from_zero_crossing': True, 'no_practical_superiority_claim': True}
    S.C.dump(output / 'CLAIMS_AND_PROGRESS.json', claims)
    lines = ['# RESULTS — RELATIONAL_SOURCE_INDEPENDENCE_V6', '',
             'Completed panels: ' + (', '.join(panels) or 'none') + '.',
             'Missing panels: ' + (', '.join(claims['missing_panels']) or 'none') + '.', '',
             'All results are from frozen writers. R2 is instruction-assisted. Historical V4 and V5 dispositions remain unchanged.', '',
             '| Panel | Actor | Construction | Reader contrast | Effect (pp) | Corrected interval (pp) |',
             '|---|---|---|---|---:|---:|']
    for panel, tests in primary.items():
        for test in tests:
            lines.append(f"| {panel} | {test['actor']} | {test['architecture']} | {test['reader']}-R0 | "
                f"{100*test['effect']:.2f} | [{100*test['lower']:.2f}, {100*test['upper']:.2f}] |")
    lines += ['', 'Panel A: eight separate primary comparisons,99.375% paired whole-scene intervals. '
              'Panel B: four primary comparisons,98.75% paired root intervals. Both use10,000 draws '
              'and both seeds carried together within scene/root; seed-specific values are retained.', '',
              'Full original24 and complete program/34-question bundle correctness are separate. '
              'Same-reader SOURCE denominators, all conditions, unfavorable cells, invalids, '
              'and native/text/replay references remain visible in the tables.', '',
              'CLAIMS_AND_PROGRESS.json gives each seed’s prospective conjunctions. Missing Panel C '
              'does not earn new repeat/restoration or full functional replication. No fresh old '
              'comparative F2 or F4 was run. Exact saved-direct recomposition is diagnostic only.', '',
              'This scientific report is distinct from independent physical-artifact verification.']
    text_file(output / 'RESULTS.md', '\n'.join(lines) + '\n')
    text_file(output / 'DELTA_FROM_V4_AND_PREPARED_V5.md',
        '# Delta from V4 and prepared V5\n\nV5 previously supplied zero fresh efficacy outcomes. '
        'This continuation measures only the complete panels listed in RESULTS. All24 small '
        'setters retain original identities; there was no fitting, strength/layer/rank/reader '
        'search or new checkpoint selection. Panel B tests four overwritten origins reaching '
        'one terminal world; equality of wrong answers is not success. Panel C, when complete, '
        'adds prospective reader-qualified program measurements on the fixed smaller subset. '
        'No result rewrites V4, proves hidden-state erasure, or establishes practical superiority.\n')
    text_file(output / 'CLAIMS_AND_PROGRESS.md',
        '# Claims and progress\n\nMachine-readable seed-specific decisions and measured values '
        'are in CLAIMS_AND_PROGRESS.json; source-independence absolute operating points are in '
        'B/SOURCE_INDEPENDENCE.json when B is measured. A positive corrected reader contrast '
        'is not by itself the90% correctness/5% disagreement/70% all34 operating-point conjunction. '
        'All criteria are finite-task progression decisions, not population guarantees. '
        'No new reader retroactively changes V4 and R2-only gains remain instruction-assisted.\n')
    return claims


def run(input_dir, output, prepared=S.ROOT / 'prepared', terminal_path=None, prior_reports=None):
    root, output = Path(input_dir), Path(output)
    if output.exists():
        raise FileExistsError('Analysis is immutable; choose a new revision')
    terminal = S.C.load(terminal_path or root / 'TERMINAL.json')
    plan = S.C.load(root / 'RUN_PLAN.json')
    panels = terminal['complete_panels']
    retained = None
    if prior_reports is not None:
        if (plan.get('schema') != CONT.SCHEMA or plan.get('planned_complete_panels') != ['C']
                or plan.get('prior_complete_panels') != ['A', 'B'] or panels not in ([], ['C'])):
            raise ValueError('C-only continuation plan violated')
        retained = CONT.retained_reports(prior_reports, plan['prior_reports_manifest_sha256'])
        if retained['provenance']['run_plan_sha256'] != plan['prior_run_plan_sha256']:
            raise ValueError('Retained reports belong to another A/B plan')
    elif panels != list(S.PANELS[:len(panels)]) or any(p not in plan['planned_complete_panels'] for p in panels):
        raise ValueError('Panel priority or frozen run plan violated')
    rows, checks = {}, {}
    for panel in panels:
        marker = S.C.load(root / f'{panel}_COMPLETE.json')
        if marker['run_plan_sha256'] != S.C.sha(root / 'RUN_PLAN.json'):
            raise ValueError('Panel completion is bound to another run plan')
        rows[panel], checks[panel] = V.load_panel(root, panel, prepared)
    output.mkdir(parents=True)
    S.C.dump(output / 'INDEPENDENT_ARTIFACT_CHECKS.json', dict(retained['checks'], **checks) if retained else checks)
    claims = produce(rows, output, retained=retained)
    S.C.dump(output / 'INPUT_PROVENANCE.json', {'terminal_scientific_status': terminal.get('status'),
        'terminal_sha256': S.C.sha(terminal_path or root / 'TERMINAL.json'),
        'run_plan_sha256': S.C.sha(root / 'RUN_PLAN.json'),
        'retained_AB': {'manifest_sha256': retained['manifest_sha256'],
            'A_B_report_members_copied_byte_identically': True,
            'A_B_model_measurements_repeated': False} if retained else None})
    S.C.dump(output / 'MANIFEST.json', {'files': {p.relative_to(output).as_posix(): S.C.sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    return {'completed_panels': claims['completed_panels'], 'missing_panels': claims['missing_panels'],
            'manifest_sha256': S.C.sha(output / 'MANIFEST.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prepared', default=str(S.ROOT / 'prepared'))
    p.add_argument('--terminal')
    p.add_argument('--prior-reports')
    a = p.parse_args()
    print(json.dumps(run(a.input, a.output, a.prepared, a.terminal, a.prior_reports), indent=2))
