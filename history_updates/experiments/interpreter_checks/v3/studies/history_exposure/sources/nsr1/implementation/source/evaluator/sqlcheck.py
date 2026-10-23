"""SQL_SHARED_PARSE: execute ISOLATED_SYMBOLIC's exact parsed state and programs in ordinary SQLite (the independently
written reference) and compare values, missingness, versions and read sets with the custom executor's rows.
Also returns the candidate's parsed state per history for intake-error localization."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sqlref'))
import sqlref as S  # noqa: E402

S.SQLStore.evaluate_all = False


def load_intake(out, panel, arm):
    with gzip.open(Path(out) / panel / 'intake.json.gz', 'rt', encoding='utf-8') as f:
        return json.load(f)[f'{panel}:{arm}']


def units(root):
    qs = root['questions']
    key = 'checkpoint_events' if any('checkpoint_events' in q for q in qs) else ('after_updates' if any('after_updates' in q for q in qs) else None)
    return key


def check(data, out, panel, rows, arm='ISOLATED_SYMBOLIC', public_relations=None):
    """rows: candidate rows of this arm. Returns (summary, parsed_states)."""
    intake = load_intake(out, panel, arm)
    vis = [json.loads(l) for l in open(Path(data) / 'visible' / panel / 'roots.jsonl', encoding='utf-8')]
    by = {}
    for r in rows:
        by.setdefault((r['root_id'], r['history_id']), {})[r['question_id']] = r
    summ = dict(rows=0, executed=0, agree_value=0, agree_error=0, agree_read_set=0, agree_version=0, disagreements=[],
                state_rejected_both=0, not_executed=0)
    parsed = {}
    structured = arm == 'A10_STRUCTURED_SOURCE'
    for root in vis:
        key = units(root)
        for h in root['histories']:
            if (root['root_id'], h['history_id']) not in by:  # fixed subsets (secondary backend) run fewer roots / neural-arm histories
                continue
            writes, state_err = [], None
            if structured:
                writes = h['typed_initial']
            else:
                for rec in h['records']:
                    pr = intake['records'][rec['text']]
                    if pr['error'] is None:
                        writes += pr['value']
            events = h.get('typed_commands') or []
            if not events:
                for u in h['updates']:
                    pu = intake['updates'][u['text']]
                    events.append(pu['value'] if pu['error'] is None else None)
            parsed[(root['root_id'], h['history_id'])] = dict(initial=writes, events=events)
            try:
                st = S.SQLStore(writes, public_relations=public_relations) if public_relations else S.SQLStore(writes)
            except S.SQLRefError as exc:
                st, state_err = None, exc.code
            done, failed = 0, False
            ks = sorted({q[key] for q in root['questions']}) if key else [len(events)]
            for k in ks:
                while st is not None and done < k:
                    ev = events[done]
                    if ev is None or failed or st.apply(ev)['status'] != 'ACCEPTED':
                        failed = True
                    done += 1
                for q in root['questions']:
                    if key and q[key] != k:
                        continue
                    r = by[(root['root_id'], h['history_id'])][q['question_id']]
                    summ['rows'] += 1
                    if st is None:
                        summ['state_rejected_both'] += int((r['error'] or '').startswith('STATE_') and r['error'][6:] == state_err)
                        summ['not_executed'] += 1
                        continue
                    if r['program'] is None or r['exec_status'] is None:
                        summ['not_executed'] += 1
                        continue
                    res = st.execute(r['program'])
                    summ['executed'] += 1
                    okv = res['status'] == r['exec_status'] and (res.get('value') == r.get('value') if res['status'] == 'OK'
                                                               else (res.get('error') == r.get('exec_error') or r.get('exec_error') == 'CONSTANT_PROGRAM'))
                    okr = [list(x) for x in res['read_set']] == [list(x) for x in (r['read_set'] or [])] or r.get('exec_error') == 'CONSTANT_PROGRAM'
                    okver = st.current_index() == r['snapshot_index']
                    summ['agree_value' if res['status'] == 'OK' else 'agree_error'] += int(okv)
                    summ['agree_read_set'] += int(okr)
                    summ['agree_version'] += int(okver)
                    if not (okv and okr and okver) and len(summ['disagreements']) < 20:
                        summ['disagreements'].append(dict(root=root['root_id'], history=h['history_id'], question=q['question_id'],
                                                          sql=res, candidate={k_: r.get(k_) for k_ in ('exec_status', 'value', 'exec_error', 'read_set', 'snapshot_index')}))
    summ['all_agree'] = (summ['agree_read_set'] == summ['executed'] == summ['agree_version'] and
                         summ['agree_value'] + summ['agree_error'] == summ['executed'])
    return summ, parsed
