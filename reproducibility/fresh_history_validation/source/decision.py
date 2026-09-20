"""Standalone validator: one allowlisted JSON packet on stdin, decisions on stdout.

The neural host invokes this with -I -S as nobody; the private run/capsule tree
must be unreadable by that account. No filesystem/network input is allowed after
stdlib initialization. This module has no primary analyzer or withheld outcome API.
"""
import json
import hashlib
import math
import sys

def schema_digest(packet):
    """Only pre-model metadata and permitted gold labels, never response values."""
    cases=[]
    for case in packet['cases']:
        view={k:case[k] for k in ('root_id','seed','draw','tie_breaker','applicable','expected_keys')}
        for key in ('common','validation'):
            view[key]=[{k:r[k] for k in ('context','query_id','gold')} for r in case[key]]
        cases.append(view)
    return hashlib.sha256(json.dumps(dict(schema=packet['schema'],procedure=packet['procedure'],cases=cases),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def decide(packet,expected_schema=None):
    if expected_schema is not None and schema_digest(packet)!=expected_schema:
        raise ValueError('Decision allowlist differs from the pre-model frozen schema')
    if set(packet) != {'schema', 'procedure', 'cases'} or packet['schema'] != 'FHV1_DECISION_INPUT_V1':
        raise ValueError('Unexpected decision input schema')
    if packet['procedure'] not in ('A', 'B'): raise ValueError('Unknown procedure')
    result = []
    for case in packet['cases']:
        if set(case) != {'root_id', 'seed', 'draw', 'tie_breaker', 'applicable', 'common', 'validation', 'expected_keys'}:
            raise ValueError('Unexpected decision case field')
        if len(case['validation']) != (6 if case['applicable'] else 0): raise ValueError('Unequal validation read budget')
        rows = case['common'] + case['validation']
        keys = [(r['context'], r['query_id']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != {tuple(x) for x in case['expected_keys']}:
            raise ValueError('Decision input does not match the exact permitted response IDs')
        for row in rows:
            if set(row) != {'context', 'query_id', 'prediction', 'valid', 'gold'}:
                raise ValueError('Forbidden decision row field')
            if type(row['valid']) is not bool or row['gold'] not in (0, 1) or row['prediction'] not in (0, 1):
                raise ValueError('Malformed decision row')
        def wrong(row): return not row['valid'] or row['prediction'] != row['gold']
        common = case['applicable'] and not any(wrong(r) for r in case['common'])
        errors = sum(wrong(r) for r in case['validation'])
        disagreements = 0
        if packet['procedure'] == 'B':
            by_query = {}
            for row in case['validation']: by_query.setdefault(row['query_id'], []).append(row)
            if case['applicable'] and (len(by_query) != 3 or any(len(v) != 2 for v in by_query.values())):
                raise ValueError('History-diversity query budget differs')
            disagreements = sum(all(x['valid'] for x in pair) and pair[0]['prediction'] != pair[1]['prediction'] for pair in by_query.values())
        result.append({k: case[k] for k in ('root_id', 'seed', 'draw', 'tie_breaker')} | dict(
            procedure=packet['procedure'], common_eligible=bool(common), wrong_validation=errors,
            accept=bool(common and errors == 0), history_disagreements=disagreements))
    return result

def main():
    def deny_io(event, args):
        if event in ('open', 'os.listdir', 'os.scandir', 'os.system', 'subprocess.Popen') or event.startswith('socket.'):
            raise PermissionError('Decision process has no filesystem or network input')
    sys.addaudithook(deny_io)
    packet = json.load(sys.stdin)
    if len(sys.argv)!=2:raise ValueError('Require the pre-model frozen schema digest')
    print(json.dumps(decide(packet,sys.argv[1]), sort_keys=True, allow_nan=False))

if __name__ == '__main__': main()
