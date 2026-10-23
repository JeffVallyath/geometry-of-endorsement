"""Exposure-only schedule adapter around the unchanged PPR1 scorer and journal.

No evaluator, training, selection sweep, state update, executor, or answer rewrite.
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from borrowed_worker import Engine, Journal, canonical, sha, file_sha, publish, utc, deadline, verified_payload

ROOT = Path(__file__).resolve().parent


def workload():
    with gzip.open(ROOT / 'requests.jsonl.gz', 'rt', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    assert len(records) == len({r['request_id'] for r in records})
    for r in records:
        assert set(r) == {'request_id', 'request', 'n_prompt', 'token_sha256'}
        assert sha(r['request']) == r['request_id']
        assert set(r['request']) == {'messages'}
    return records


def canary(engine, journal, stage):
    expected = json.loads((ROOT / 'canaries' / (engine.checkpoint + '.json')).read_bytes())
    values, tokens = engine.calculate(expected['requests'], 'score')
    error = max(abs(a[k] - b[k]) for a, b in zip(values, expected['scores']) for k in ('yes_logit', 'no_logit', 'margin'))
    receipt = dict(status='PASS' if error <= expected['tolerance'] else 'FAILED', max_absolute_error=error,
                   tolerance=expected['tolerance'], original_canary_sha256=file_sha(ROOT / 'canaries' / (engine.checkpoint + '.json')),
                   scores=values, tokens=tokens, runtime=engine.identity, utc=utc())
    publish(journal.out / (engine.checkpoint + '_' + stage + '_CANARY.json'), receipt)
    journal.export(force=True)
    assert receipt['status'] == 'PASS', ('ORIGINAL_SCORER_CANARY_FAILED', engine.checkpoint, error)


def preflight(model_path, manifest_hash, journal, records):
    synthetic_times = []
    for seed in ('seed0', 'seed1'):
        engine = Engine(seed, model_path, manifest_hash)
        canary(engine, journal, 'PREFLIGHT')
        if seed == 'seed0':
            for r in records:
                ids = engine.tokens(r['request'])
                assert len(ids) == r['n_prompt'] and sha(ids) == r['token_sha256'], 'CPU_REMOTE_TOKEN_MISMATCH'
            synthetic = {'messages': [{'role': 'system', 'content': 'Use only these synthetic records.'},
                {'role': 'user', 'content': 'Record: Alpha is in Beta. Candidate: Alpha is in Beta. Answer Yes or No.'}]}
            base = engine.tokens(synthetic)
            length = max(r['n_prompt'] for r in records)
            filler = engine.tok.encode(' record', add_special_tokens=False)[0]
            shaped = base[:-1] + [filler] * (length - len(base)) + base[-1:]
            assert len(shaped) == length
            for _ in range(3):
                values, _ = engine.calculate([synthetic] * 8, 'score', token_override=[shaped] * 8)
                synthetic_times.append(values[0]['batch_seconds'])
        engine.close()
    rental = json.loads((ROOT / 'RENTAL.json').read_bytes())
    elapsed = time.time() - rental['allocated_epoch']
    prior = rental.get('prior_billed_gpu_hours', 0) * 3600
    # Charge both complete seed schedules; no past test-result cache discount.
    compute = len(records) * 2 * max(synthetic_times) / 8
    remaining_upper = (compute + 20 * 60) * 1.25
    closeout = 45 * 60
    fits = (remaining_upper <= deadline() - time.time()
            and prior + elapsed + remaining_upper + closeout <= 7 * 3600
            and rental.get('prior_cloud_usd_upper_bound', 0)
                + (elapsed + remaining_upper + closeout) / 3600 * rental['all_in_hourly_upper_bound'] <= 40)
    result = dict(status='PASS' if fits else 'FULL_WORKLOAD_INFEASIBLE', total_physical_requests=len(records) * 2,
        max_input_tokens=max(r['n_prompt'] for r in records), synthetic_batch_seconds=synthetic_times,
        complete_remaining_seconds_with_25pct_contingency=remaining_upper, closeout_reserve_seconds=closeout,
        current_attempt_elapsed_seconds=elapsed, prior_gpu_seconds=prior, time_utc=utc(), manifest_sha256=manifest_hash,
        test_outcomes_observed=False, original_canaries_passed_both=True)
    publish(journal.out / 'FORECAST.json', result)
    journal.export(force=True)
    print(canonical({k: v for k, v in result.items() if k != 'synthetic_batch_seconds'}), flush=True)
    assert fits, 'NO_TRIMMING_ALLOWED_FULL_WORKLOAD_INFEASIBLE'


def run(model_path, manifest_hash, journal, records):
    forecast = json.loads((journal.out / 'FORECAST.json').read_bytes())
    assert forecast['status'] == 'PASS' and forecast['manifest_sha256'] == manifest_hash
    assert forecast['complete_remaining_seconds_with_25pct_contingency'] <= deadline() - time.time()
    for seed in ('seed0', 'seed1'):
        engine = Engine(seed, model_path, manifest_hash)
        canary(engine, journal, 'RUN')
        publish(journal.out / (seed + '_RUNTIME.json'), engine.identity)
        # On technical resume, preserve original full-batch boundaries: completed
        # batches are reused; a partial batch is forbidden rather than silently rebatched.
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            keys = [seed + '/' + r['request_id'] for r in batch]
            have = [key in journal.calls for key in keys]
            if all(have):
                for key, r in zip(keys, batch):
                    assert journal.calls[key]['request'] == r['request']
                    assert journal.calls[key]['runtime_sha256'] == sha(engine.identity)
                continue
            assert not any(have), 'PARTIAL_BATCH_REQUIRES_EXACT_RECOVERY_NOT_REBATCHING'
            began = utc()
            values, tokens = engine.calculate([r['request'] for r in batch], 'score')
            for key, r, value, ids in zip(keys, batch, values, tokens):
                assert len(ids) == r['n_prompt'] and sha(ids) == r['token_sha256']
                journal.add('calls', dict(call_id=key, seed=seed, request_id=r['request_id'], request=r['request'],
                    result=value, input_token_ids=ids, runtime_sha256=sha(engine.identity),
                    batch_start=start, started_utc=began, completed_utc=utc()), 'call_id')
            if start % 256 == 0:
                journal.status('SCORING', seed=seed, completed_unique_requests=len(journal.calls), total_unique_requests=2 * len(records))
        engine.close()
        journal.export(force=True)
    expected = {seed + '/' + r['request_id'] for seed in ('seed0', 'seed1') for r in records}
    assert set(journal.calls) == expected
    publish(journal.out / 'SCIENCE_COMPLETE.json', dict(status='COMPLETE', unique_calls=len(expected),
        logical_reads=32436, seeds=['seed0', 'seed1'], manifest_sha256=manifest_hash,
        inference_only=True, test_outcomes_not_used_for_adaptation=True, time_utc=utc()))
    journal.status('COMPLETE', completed_unique_requests=len(expected))
    journal.export(force=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'run'], required=True)
    parser.add_argument('--model-path', default='/workspace/models/qwen')
    args = parser.parse_args()
    manifest_hash = verified_payload()
    journal = Journal()
    records = workload()
    (preflight if args.stage == 'preflight' else run)(args.model_path, manifest_hash, journal, records)
