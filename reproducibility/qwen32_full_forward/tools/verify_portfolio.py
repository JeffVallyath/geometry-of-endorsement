"""Verify the earlier completed checkpoint rows; never treats missing outcomes as zero."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

ARCHIVE_SHA256 = 'eb5bee9d247bdfbea59ad894053a4b2054a0d605a898e008b889517c614b7441'
METHODS = ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING')
SHARDS = 'runtime/source/geometry-of-endorsement/reproducibility/fresh_inference/supplied/reproduction/reports/state_sufficiency_confirmation/shards/'


def truth(spec, values):
    a, b = values[spec['a']][spec['p']], values[spec['b']][spec['p']]
    return {'direct': a, 'opposes': 1-a, 'same': int(a == b), 'both': a & b, 'either': a | b}[spec['kind']]


def decode(score):
    left, right = score['logps']
    finite = math.isfinite(left) and math.isfinite(right)
    prediction = int(right > left)
    member = score['full_vocab_argmax_token_id'] in score['label_token_ids']
    valid = finite and left != right and member
    if (prediction != score['prediction'] or finite != score['finite']
            or member != score['argmax_in_labels'] or valid != score['valid']):
        raise ValueError('Historical score decoding changed')
    return prediction, valid


def check(archive, output):
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Wrong historical archive digest')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('MANIFEST.json'))
        if len(z.namelist()) != len(manifest['files']) + 1 or set(z.namelist()) != set(manifest['files']) | {'MANIFEST.json'}:
            raise ValueError('Historical archive inventory mismatch')
        for name, record in manifest['files'].items():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or p.as_posix() != name:
                raise ValueError('Unsafe historical member')
            with z.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if z.getinfo(name).file_size != record['bytes'] or digest != record['sha256']:
                raise ValueError('Historical member mismatch: ' + name)
        load = lambda name: json.loads(z.read(name))
        table = load('runtime/analysis_final/MAIN_TABLE.json')
        roster = {r['id']: r for r in load('runtime/MODEL_ROSTER_FROZEN.json')['rows']}
        rows = []
        for row_id in ('gemma9_t4', 'llama8_t4', 'qwen4_t4'):
            counts = dict.fromkeys(METHODS, 0)
            designs = {method: set() for method in METHODS}
            for i in range(64):
                rid = f'SSC1-FINAL-{i:04d}'
                world = load(SHARDS + rid + '/root.json')
                questions = load(SHARDS + rid + '/questions.json')
                scores = load(f'runtime/results/{row_id}/final/{rid}/SCORES.json')
                index = {(c, q): decode(s) for c, values in scores.items() for q, s in values.items()}
                if len(index) != 578:
                    raise ValueError('Incomplete historical final case')
                direct = [q for q in questions if q['spec']['kind'] == 'direct']
                joints = [q for q in questions if q['spec']['kind'] in ('same', 'both', 'either')]
                values = world['terminal']['values']
                def correct(context, q):
                    prediction, valid = index[context, q['query_id']]
                    return valid and prediction == truth(q['spec'], values)
                for method in METHODS:
                    for q in joints:
                        spec = q['spec']
                        fmt = q['query_id'].split('-d')[1][0]
                        operands = [d for d in direct if d['spec']['a'] in (spec['a'], spec['b'])
                                    and d['spec']['p'] == spec['p'] and d['query_id'].split('-d')[1][0] == fmt]
                        if len(operands) != 2:
                            raise ValueError('Unexpected inherited operand schema')
                        a = all(correct(f'o{o}/{method}', d) for o in range(8) for d in operands)
                        c = correct('NATIVE_FINAL', q) and correct(f'o0/{method}', q)
                        answers = {index[f'o{o}/{method}', q['query_id']][0] for o in range(8)
                                   if index[f'o{o}/{method}', q['query_id']][1]}
                        if a and c and len(answers) > 1:
                            counts[method] += 1
                            designs[method].add(rid)
            for method in METHODS:
                saved = next(r for r in table if r['row_id'] == row_id and r['method'] == method)
                if saved['status'] != 'COMPLETE_FINAL_ROW' or counts[method] != saved['W']:
                    raise ValueError('Historical primary reconstruction differs')
                if len(designs[method]) != saved['roots_with_witness'] or saved['opportunities'] != 1152:
                    raise ValueError('Historical design count/denominator differs')
                rows.append(dict(row_id=row_id, model=roster[row_id]['model'], revision=roster[row_id]['revision'],
                                 method=method, witnesses=counts[method], denominator=1152, designs=len(designs[method])))
        old_32 = [r for r in table if r['row_id'] == 'qwen32_t4']
        if len(old_32) != 2 or any(r['status'] != 'NOT_MEASURED' or r['witness_rate'] is not None for r in old_32):
            raise ValueError('Historical missing-result disposition changed')
        selected = ('runtime/analysis_final/MAIN_TABLE.json', 'runtime/analysis_final/PLANNED_COMPARISONS.json',
                    'runtime/analysis_final/ROW_DISPOSITIONS.json', 'runtime/MODEL_ROSTER_FROZEN.json')
        output.mkdir(parents=True)
        for name in selected:
            (output / Path(name).name).write_bytes(z.read(name))
        result = dict(status='PASS', source_archive_sha256=ARCHIVE_SHA256,
                      source_archive_bytes=archive.stat().st_size, members_verified=len(manifest['files']),
                      raw_score_records_decoded=3 * 64 * 578, primary_rows=rows,
                      historical_qwen32_not_measured_preserved=True, numerical_bridges_repaired=False,
                      original_member_bindings={name: manifest['files'][name] for name in selected},
                      source_is_completed_archive_not_live_or_sealed_results=True, inference=False)
    (output / 'PORTFOLIO_CHECK.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.archive, args.output), indent=2))
