"""Language-path version of the five predeclared negative contract cases that carry a question text
(data/evaluator/negative/cases.json, frozen), plus three supported controls on the same records.

The typed state of those cases is rendered with the ordinary record wordings; expected behaviour for the five
negative questions is rejection (no answer); the three controls must be answered. Analysed separately from in-scope
efficacy. Built after the freeze from frozen cases; no outcome existed when it was added (records/DEV_RECORD.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

study = Path(sys.argv[1])
cases = json.loads((study / 'data' / 'evaluator' / 'negative' / 'cases.json').read_text())
records = [
    'Record 01: Abel Drummond on the Kestrel proposal: in favor; status approved; shipment day Monday.',
    'Record 02: Brisa Holloway on the Kestrel proposal: opposed.',
    'Record 03: Brisa Holloway is on the Falcon team.',
    'Record 04: The Falcon team is based at the Delta hub.',
    'Record 05: The Delta hub is in the North region.',
    'Record 06: Corvin Whitlock is on the Heron team.',
]
neg = [c for c in cases if c.get('question_text')]
assert len(neg) == 5
controls = [('Is Abel Drummond currently in favor of the Kestrel proposal?\nAnswer with exactly one of No, Yes.', 'Yes'),
            ("Which hub is Brisa Holloway's team based at?", 'Delta'),
            ("What is the current status of Abel Drummond's Kestrel record?", 'approved')]
qs = [dict(question_id=c['case_id'], text=c['question_text']) for c in neg] + \
     [dict(question_id=f'NSR1-NEGCTL-{i}', text=t) for i, (t, _) in enumerate(controls)]
vis = dict(root_id='NSR1-NEGNL-0000', histories=[dict(history_id='n0', records=[dict(record_id=f'{i + 1:02d}', text=t) for i, t in enumerate(records)],
                                                     updates=[])], questions=qs)
exp = [dict(question_id=c['case_id'], expect='REJECT', category=c['category']) for c in neg] + \
      [dict(question_id=f'NSR1-NEGCTL-{i}', expect='ANSWER', gold=g) for i, (_, g) in enumerate(controls)]
(study / 'data' / 'visible' / 'negative_nl').mkdir(parents=True, exist_ok=True)
(study / 'data' / 'evaluator' / 'negative_nl').mkdir(parents=True, exist_ok=True)
(study / 'data' / 'visible' / 'negative_nl' / 'roots.jsonl').write_text(json.dumps(vis, sort_keys=True) + '\n', encoding='utf-8')
(study / 'data' / 'evaluator' / 'negative_nl' / 'expected.json').write_text(json.dumps(exp, indent=1), encoding='utf-8')
print(len(neg), 'negative questions,', len(controls), 'controls')
