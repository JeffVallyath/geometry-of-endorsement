"""Full fixed comparison driver. Resumable, sharded, outcome-independent.

Scale (from the supplied fixed cases):
    1,616 plan rows x 12 declared arms = 19,392 context builds
    44,288 planned questions x 12 arms = ~531k scored answers

That does not fit one Colab session, so work is sharded by
(actor, panel, method, seed) and every shard writes an append-only JSONL with a
completion sentinel. Re-running skips finished shards. A killed session loses at
most one shard.

Ordering is fixed in advance and never depends on results. A compute shortfall
yields IDENTIFIED MISSING CELLS, not a selected favorable subset: `--report-gaps`
prints exactly which shards are absent.

SOURCE runs first for every (actor, panel) because metrics.score_case requires
the original same-interface SOURCE answers for every question in every other arm.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import methods as M


# ------------------------------------------------------------------ io helpers

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def shard_name(actor, panel, method, seed):
    return f"{actor}__{panel}__{M.condition_name(method, seed)}"


def sentinel(raw: Path, name):
    return raw / f"{name}.done.json"


def planned_shards(pkg: Path, actors, panels):
    out = []
    for actor in actors:
        for panel in panels:
            for method, seed in M.arms():
                out.append((actor, panel, method, seed))
    # SOURCE first within each (actor, panel): every other arm depends on it.
    out.sort(key=lambda x: (x[0], x[1], x[2] != "SOURCE"))
    return out


# ------------------------------------------------------------------ source cache

def source_answers_path(raw: Path, actor, panel):
    return raw / f"{actor}__{panel}__SOURCE.answers.json"


def load_source_answers(raw: Path, actor, panel):
    p = source_answers_path(raw, actor, panel)
    if not p.is_file():
        raise FileNotFoundError(
            f"SOURCE answers missing for {actor}/{panel}. Run the SOURCE shard first; "
            "every other arm is scored against the original same-interface SOURCE.")
    blob = json.loads(p.read_text())
    return {k: {qid: a for qid, a in v.items()} for k, v in blob.items()}


# ------------------------------------------------------------------ one shard

def run_shard(pkg: Path, raw: Path, backbone, actor, panel, method, seed,
              plan_rows, inputs_by_id, eval_by_input, limit=None):
    import inputs as INP
    from metrics import score_case

    name = shard_name(actor, panel, method, seed)
    out_path = raw / f"{name}.jsonl"
    tel_path = raw / f"{name}.telemetry.jsonl"
    done = sentinel(raw, name)
    if done.is_file():
        return json.loads(done.read_text())

    rows = [r for r in plan_rows if r["actor"] == actor and r["panel"] == panel]
    if limit:
        rows = rows[:limit]

    is_source = method == "SOURCE"
    source_answers = None if is_source else load_source_answers(raw, actor, panel)
    collected_source = {}

    # append-only; resume inside a shard by skipping already-written input_ids
    seen = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["input_id"])

    started = time.time()
    failures = []
    fout = out_path.open("a", encoding="utf8")
    tout = tel_path.open("a", encoding="utf8")

    try:
        for i, row in enumerate(rows):
            if row["input_id"] in seen:
                continue
            src = inputs_by_id[row["input_id"]]
            records = eval_by_input[row["input_id"]]
            commands = INP.commands_from_json(src["commands"])

            try:
                context, telemetry = M.build_context(
                    src["source_text"], commands,
                    method=method, seed=seed, backbone=backbone, actor=actor)

                scored = score_case(
                    context, records,
                    None if is_source else {r["query_id"]: source_answers[row["input_id"]][r["query_id"]]
                                            for r in records},
                    backbone=backbone, seed=seed)

                if is_source:
                    collected_source[row["input_id"]] = {
                        r["query_id"]: dict(question=r["text"], prediction=s["prediction"],
                                            valid=s["valid"], logps=s["logps"])
                        for r, s in zip(records, scored)
                    }

                for s in scored:
                    s["input_id"] = row["input_id"]
                    s["root_id"] = row["root_id"]
                    s["origin"] = row["origin"]
                    s["program"] = row["program"]
                    s["order"] = row["order"]
                    # grouping keys analysis.py depends on. score_case already
                    # supplies actor/condition, but they are set explicitly so a
                    # change upstream cannot silently empty the primary endpoint.
                    s["input_panel"] = panel
                    s["scene_id"] = row.get("scene_id")
                    s["actor"] = actor
                    s["condition"] = M.condition_name(method, seed)
                    fout.write(json.dumps(s) + "\n")

                telemetry.update(input_id=row["input_id"], root_id=row["root_id"],
                                 panel=panel, actor=actor, condition=M.condition_name(method, seed),
                                 questions=len(records),
                                 preprocess_seconds=context.preprocess_seconds,
                                 compile_seconds=context.compile_seconds,
                                 artifact_sha256=context.artifact_sha256,
                                 retained_state=M.retained_state(method))
                tout.write(json.dumps(telemetry) + "\n")

            except Exception as exc:  # a failing cell is recorded, never hidden
                failures.append(dict(input_id=row["input_id"], error=repr(exc),
                                     traceback=traceback.format_exc()[-2000:]))

            if (i + 1) % 25 == 0:
                fout.flush(); tout.flush()
                print(f"  {name}: {i+1}/{len(rows)}", flush=True)
    finally:
        fout.close(); tout.close()

    if is_source:
        merged = {}
        p = source_answers_path(raw, actor, panel)
        if p.is_file():
            merged = json.loads(p.read_text())
        merged.update(collected_source)
        p.write_text(json.dumps(merged))

    receipt = dict(shard=name, actor=actor, panel=panel, method=method, seed=seed,
                   rows=len(rows), failures=failures,
                   status="PASS" if not failures else "PARTIAL",
                   seconds=time.time() - started)
    done.write_text(json.dumps(receipt, indent=2))
    return receipt


# ------------------------------------------------------------------ entrypoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True)
    p.add_argument("--raw", required=True, help="output dir for raw score rows")
    p.add_argument("--cache", help="local HF snapshot cache dir")
    p.add_argument("--actor", choices=("gemma", "qwen"), action="append")
    p.add_argument("--panel", choices=("DEV", "MAIN", "ORIGIN", "PROGRAM"), action="append")
    p.add_argument("--method", action="append", help="restrict to these methods")
    p.add_argument("--limit", type=int, help="cases per shard (smoke only)")
    p.add_argument("--max-seconds", type=int, default=0,
                   help="stop cleanly before a session limit; remaining shards resume later")
    p.add_argument("--report-gaps", action="store_true",
                   help="list missing shards and exit without running anything")
    args = p.parse_args()

    pkg = Path(args.package).resolve()
    raw = Path(args.raw).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    os.sys.path.insert(0, str(pkg))

    actors = args.actor or ["gemma", "qwen"]
    panels = args.panel or ["MAIN", "ORIGIN", "PROGRAM"]
    shards = planned_shards(pkg, actors, panels)
    if args.method:
        shards = [s for s in shards if s[2] in args.method or s[2] == "SOURCE"]

    if args.report_gaps:
        missing = [shard_name(*s) for s in shards if not sentinel(raw, shard_name(*s)).is_file()]
        print(json.dumps(dict(planned=len(shards), complete=len(shards) - len(missing),
                              missing=missing), indent=2))
        return 0

    plan_rows = load_jsonl(pkg / "cases/fixed/plan.jsonl")
    inputs_by_id = {r["input_id"]: r for r in load_jsonl(pkg / "cases/fixed/inputs.jsonl")}
    eval_by_input = {}
    for r in load_jsonl(pkg / "cases/fixed/evaluation.jsonl"):
        eval_by_input.setdefault(r["input_id"], []).append(r)

    from canonical import config as Q
    from canonical.adapter import Backbone

    began = time.time()
    current_actor, backbone = None, None
    receipts = []

    for actor, panel, method, seed in shards:
        name = shard_name(actor, panel, method, seed)
        if sentinel(raw, name).is_file():
            continue
        if args.max_seconds and time.time() - began > args.max_seconds:
            print(f"time budget reached; stopping before {name}. Re-run to resume.")
            break

        if actor != current_actor:
            if backbone is not None:
                backbone.finish(); del backbone
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            print(f"loading backbone: {actor}", flush=True)
            backbone = Backbone(cache=args.cache, cfg=Q.MODELS[actor])
            current_actor = actor

        print(f"shard {name}", flush=True)
        receipts.append(run_shard(pkg, raw, backbone, actor, panel, method, seed,
                                  plan_rows, inputs_by_id, eval_by_input, args.limit))

    if backbone is not None:
        backbone.finish()

    print(json.dumps(dict(completed=len(receipts),
                          partial=[r["shard"] for r in receipts if r["status"] != "PASS"]),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
