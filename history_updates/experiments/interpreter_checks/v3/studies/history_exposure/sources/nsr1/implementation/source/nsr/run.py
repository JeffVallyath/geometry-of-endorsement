"""NSR1 candidate runner: panels x arms with any Gen backend (vLLM on the pod, grammar reference on CPU).

Reads data/visible (and, for the labelled typed-reference diagnostic only, data/diagnostic tables).
Writes rows, intake parses, raw generations and stats per panel. Never reads data/evaluator.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from . import core, pipeline as P, prompts

NEURAL_ARMS = {
    'core': ('NATIVE', 'CURRENT', 'FLAT'), 'factorial': ('NATIVE', 'CURRENT', 'FLAT'),
    'temporal': ('NATIVE', 'FLAT'), 'graph': ('NATIVE', 'CURRENT', 'FLAT'),
    'public': ('NATIVE', 'CURRENT', 'FLAT'), 'boundary': ('EXPOSED', 'RENDERER'), 'a10': (),
    'workload': (),
}


def base_panel(panel):
    return panel[4:] if panel.startswith('dev_') else panel


def load_roots(data, panel, limit=None):
    roots = [P.sanitize_root(json.loads(l)) for l in open(Path(data) / 'visible' / panel / 'roots.jsonl', encoding='utf-8')]
    return roots[:limit] if limit else roots


class Logger:
    """Raw generation log: every output's tokens/text; prompts are reproducible from the visible inputs and the
    frozen prompt code, so only the prompt hash/length is stored (full prompt text for the first 12 per tag)."""

    def __init__(self, path):
        self.f = gzip.open(path, 'at', encoding='utf-8')
        self.seen = {}

    def write(self, line):
        o = json.loads(line)
        n = self.seen.get(o['tag'], 0)
        self.seen[o['tag']] = n + 1
        if n >= 12:
            o.pop('messages', None)
        self.f.write(json.dumps(o) + '\n')

    def close(self):
        self.f.close()


def run_panel(backend, data, out_dir, panel, limit=None, model_tag='model', typed_tables=True):
    out_dir = Path(out_dir) / panel
    if (out_dir / 'DONE').exists():
        return json.loads((out_dir / 'stats.json').read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = load_roots(data, panel, limit)
    base = base_panel(panel)
    log = Logger(out_dir / 'generations.jsonl.gz')
    gen = P.CachedGen(backend, log=log)
    out = dict(rows=[])
    t0 = time.time()
    stage = {}

    def lap(name, s0, t_start):
        stage[name] = dict(seconds=round(time.time() - t_start, 3),
                           **{k: gen.stats[k] - s0[k] for k in ('calls', 'hits', 'generated', 'prompt_tokens', 'output_tokens')})

    if base == 'public':
        vocab = json.loads((Path(data) / 'visible' / 'PUBLIC_SCHEMA.json').read_text(encoding='utf-8'))
        core.register_public_relations([r['relation'] for r in vocab])
        prompts.use_public_schema(vocab)
    else:
        prompts.use_fixed_schema()
    neural = roots
    if base == 'factorial':  # neural arms: N8 and N32 variants only (protocol 7B); symbolic runs every N
        neural = [dict(r, histories=[h for h in r['histories'] if h['history_id'].split('-')[0] in ('N8', 'N32')]) for r in roots]

    s0, ts = dict(gen.stats), time.time()
    if base == 'a10':
        tc = {(r['root_id'], h['history_id']): h['typed_commands'] for r in roots for h in r['histories']}
        st, cmp_, tables = P.run_isolated(gen, panel, roots, out, typed_commands=tc, arm='A10_TEXT_SOURCE')
        lap('A10_TEXT_SOURCE', s0, ts)
        s0, ts = dict(gen.stats), time.time()
        ti = {(r['root_id'], h['history_id']): h['typed_initial'] for r in roots for h in r['histories']}
        P.run_isolated(gen, panel, roots, out, typed_commands=tc, arm='A10_STRUCTURED_SOURCE', structured=ti)
        lap('A10_STRUCTURED_SOURCE', s0, ts)
    else:
        st, cmp_, tables = P.run_isolated(gen, panel, roots, out)
        lap('ISOLATED_SYMBOLIC', s0, ts)
    arms = NEURAL_ARMS[base]
    iso_rows = [r for r in out['rows'] if r['arm'] == 'ISOLATED_SYMBOLIC']
    for arm in arms:
        s0, ts = dict(gen.stats), time.time()
        if arm == 'NATIVE':
            P.run_native(gen, panel, neural, out)
        elif arm == 'CURRENT':
            P.run_current(gen, panel, neural, out, tables)
            tpath = Path(data) / 'diagnostic' / 'typed_tables' / f'{panel}.json'
            if typed_tables and tpath.exists():  # labelled typed-reference diagnostic (separates intake errors)
                lap('CURRENT_NEURAL', s0, ts)
                s0, ts = dict(gen.stats), time.time()
                tt = {tuple(k.split('\t')[:2]) + (int(k.split('\t')[2]),): tuple(v) for k, v in json.loads(tpath.read_text(encoding='utf-8')).items()}
                P.run_current(gen, panel, neural, out, tt, arm='CURRENT_NEURAL_TYPED_REFERENCE')
                arm = 'CURRENT_TYPED'
        elif arm == 'FLAT':
            P.run_flat(gen, panel, neural, out)
        elif arm == 'EXPOSED':
            P.run_exposed_compiler(gen, panel, roots, out, st)
        elif arm == 'RENDERER':
            P.run_neural_renderer(gen, panel, roots, out, iso_rows)
        lap({'NATIVE': 'NATIVE_NEURAL', 'CURRENT': 'CURRENT_NEURAL', 'FLAT': 'FLAT_SYMBOLIC', 'EXPOSED': 'HISTORY_EXPOSED_QUERY_COMPILER',
             'RENDERER': 'NEURAL_RENDERER_OVERRIDE', 'CURRENT_TYPED': 'CURRENT_NEURAL_TYPED_REFERENCE'}[arm], s0, ts)
    log.close()
    with gzip.open(out_dir / 'rows.jsonl.gz', 'wt', encoding='utf-8') as f:
        for r in out['rows']:
            f.write(json.dumps(r) + '\n')
    with gzip.open(out_dir / 'intake.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(out.get('intake', {}), f)
    stats = dict(panel=panel, model=model_tag, backend=getattr(backend, 'model_id', '?'), prompt_set=prompts.PROMPT_SET_ID,
                 executor=core.EXECUTOR_REVISION, roots=len(roots), rows=len(out['rows']), seconds=round(time.time() - t0, 2),
                 gen=gen.stats, stages=stage, recheck=out.get('recheck', {}))
    (out_dir / 'stats.json').write_text(json.dumps(stats, indent=1))
    (out_dir / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['vllm', 'hf', 'grammar'], required=True)
    ap.add_argument('--model-path')
    ap.add_argument('--model-id', default='qwen')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--panels', required=True)
    ap.add_argument('--limit', type=int, action='append', default=None, help='per-panel root limits, same order as --panels')
    ap.add_argument('--max-model-len', type=int, default=16384)
    a = ap.parse_args()
    if a.backend == 'vllm':
        from .llm_vllm import VLLMGen
        backend = VLLMGen(a.model_path, a.model_id, max_model_len=a.max_model_len)
    elif a.backend == 'hf':
        from .llm_hf import HFGen
        backend = HFGen(a.model_path, a.model_id, max_model_len=min(a.max_model_len, 8192))
    else:
        from .grammar_ref import GrammarGen
        backend = GrammarGen()
    panels = a.panels.split(',')
    limits = a.limit or [None] * len(panels)
    for panel, lim in zip(panels, limits):
        s = run_panel(backend, a.data, a.out, panel, lim or None, model_tag=a.model_id)
        print(json.dumps(dict(panel=panel, rows=s['rows'], seconds=s['seconds'], gen=s['gen'])), flush=True)


if __name__ == '__main__':
    main()
