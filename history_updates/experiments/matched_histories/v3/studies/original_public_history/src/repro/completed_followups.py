"""Check selected-case carrier and mediation evidence without tensor/model execution."""
import argparse
import csv
import io
import json
from pathlib import Path

from .common import ROOT, new_output
from .release_inputs import payload, verify

SOURCE = ROOT / 'reproducibility/completed_followups'
STUDIES = ('carrier', 'mediation')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value):
    require(value in ('True', 'False'), 'Invalid Boolean cell')
    return value == 'True'


def direct_preservation(row):
    """The fixed recipient-preservation rule, evaluated on retained score fields."""
    return (boolean(row['valid']) and boolean(row['correct'])
            and float(row['p_correct']) >= .95 and float(row['candidate_mass']) >= .90
            and float(row['direct_recipient_probability_drift']) <= .02)


def read_records(source, study):
    folder = Path(source) / study
    binding = verify(folder, study)
    records = {record['member']: record for record in binding['files']}
    def saved(name):
        raw = payload(folder, records[name])
        if name.endswith('.csv'):
            return list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
        return json.loads(raw)
    return binding, saved


def check(source=SOURCE):
    result = {}
    for study, count in (('carrier', 2172), ('mediation', 1422)):
        binding, saved = read_records(source, study)
        table = saved('ALL_SCORES_AND_EFFECTS.csv' if study == 'carrier' else 'ALL_UNIQUE_FORWARD_SCORES.csv')
        require(len(table) == count, 'Missing measured score rows')
        identity_fields = ('stage', 'condition', 'recipient', 'query_id')
        keys = [(row.get('root'), *(row[key] for key in identity_fields)) for row in table]
        require(len(keys) == len(set(keys)), 'Duplicate measured score identity')
        direct = [row for row in table if row['kind'] == 'direct' and row['direct_preserved'] != '']
        for row in direct:
            require(direct_preservation(row) == boolean(row['direct_preserved']), 'Direct-preservation classification drift')
        new_errors = [row for row in table if row['kind'] != 'direct'
                      and boolean(row['baseline_correct']) and not boolean(row['correct'])]
        adverse = saved('ADVERSE_AND_OMISSIONS.json')
        require(len(new_errors) == len(adverse['new_joint_errors']), 'Adverse joint changes do not reconcile')
        require(all(boolean(row['direct_preserved']) for row in direct), 'Changed direct-preservation result')
        reported_direct = ([row for row in direct if row['stage'] not in ('baseline', 'native_reference')]
                           if study == 'carrier' else direct)
        require(len(reported_direct) == (1416 if study == 'carrier' else 948), 'Changed direct-check denominator')
        require(not adverse['direct_failures'], 'Source direct failure disposition changed')
        if study == 'carrier':
            require(len(new_errors) == 179, 'Carrier adverse effect count changed')
        else:
            audit = saved('NUMERICAL_AUDIT.json')
            require(audit['selected_writer']['name'] == 'L16_ATTN_ONLY', 'Selected writer changed')
            reader = audit['selected_reader']
            require(reader['selected'] == 'TOP8' and reader['status'] == 'NO_QUALIFIED_READER_SET', 'Failed reader qualification changed')
            require(reader['training_outcomes']['TOP8']['mediation_score'] < .8
                    and not reader['training_outcomes']['TOP8']['qualified_training'], 'Reader threshold or qualification drift')
        result[study] = dict(files_checked=len(binding['files']), measured_rows=len(table),
                             all_direct_rows_checked=len(direct), reported_direct_preservation_rows=len(reported_direct),
                             baseline_reference_rows_outside_reported_denominator=len(direct) - len(reported_direct),
                             new_joint_errors=len(new_errors),
                             direct_failures=0, invalid_rows=sum(not boolean(row['valid']) for row in table))
    return dict(status='PASS', studies=result,
                scope='Public integrity and exact field-bound score accounting; no tensor replay',
                active_outcomes_read=False, inference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output is not None:
        require(not args.output.exists(), 'Use a fresh output directory')
        output = new_output(args.output)
        (output / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
