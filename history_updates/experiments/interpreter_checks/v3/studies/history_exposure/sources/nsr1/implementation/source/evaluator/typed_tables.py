"""Evaluator-side: canonical CURRENT/ORIGINAL record renderings built from the TYPED reference state.

Used only by the labelled diagnostic arm CURRENT_NEURAL_TYPED_REFERENCE (it separates intake errors from
neural reading errors). These tables contain evaluator truth about the state (never answers or programs) and
are read by no main arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nsr import core, pipeline as P, prompts  # noqa: E402


def build(data, panel):
    out = {}
    if 'public' in panel:
        vocab = json.loads((Path(data) / 'visible' / 'PUBLIC_SCHEMA.json').read_text(encoding='utf-8'))
        core.register_public_relations([r['relation'] for r in vocab])
        prompts.use_public_schema(vocab)
    else:
        prompts.use_fixed_schema()
    for line in open(Path(data) / 'evaluator' / panel / 'roots.jsonl', encoding='utf-8'):
        er = json.loads(line)
        for th in er['typed']:
            if 'factorial' in panel and th['N'] not in (8, 32):
                continue
            st = core.replay(th['initial'], th['events'])
            out[f'{er["root_id"]}\t{th["history_id"]}\t{len(th["events"])}'] = [P.table(st.current.values), P.table(st.snapshots[0].values)]
    dest = Path(data) / 'diagnostic' / 'typed_tables'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f'{panel}.json').write_text(json.dumps(out), encoding='utf-8')
    prompts.use_fixed_schema()
    return len(out)


if __name__ == '__main__':
    for panel in sys.argv[2].split(','):
        print(panel, build(sys.argv[1], panel))
