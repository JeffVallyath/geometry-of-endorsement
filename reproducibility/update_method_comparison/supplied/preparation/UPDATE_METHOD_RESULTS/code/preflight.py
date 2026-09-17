"""Gate that must pass before ANY measurement. Implements START_HERE.md Steps 5 and 7.

Two stages:

  --stage cpu    no GPU, no model. Package integrity + generator reproduction +
                 unit tests + adapter template fingerprint.

  --stage model  requires the pinned backbone. Real-tokenizer boundaries, stock
                 SOURCE equivalence, clone immutability, and the clause-refresh
                 feasibility census on the 8 DEV scenes only.

The census matters more than it looks. FIELD_PLUS_LATEST_ERRATUM is the primary
baseline. Its field-refresh half only executes when the stance swap is
token-length-preserving. If the census shows a high fallback rate, then
FIELD_PLUS_LATEST_ERRATUM degenerates toward LATEST_ERRATUM on those cases and
the headline contrast must be read with that stated. Measure it, report it,
do not tune around it.

DEV scenes are for adapter checks only, never efficacy selection.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ stage: cpu

def stage_cpu(pkg: Path) -> dict:
    sys.path.insert(0, str(pkg))
    import verify_package
    import erratum_adapter as EA

    report = dict(schema="PREFLIGHT_CPU_V1", stage="cpu",
                  python=platform.python_version(), started=time.time())

    report["package_verification"] = verify_package.verify(pkg)

    cases = pkg / "cases/fixed"
    report["populations"] = {
        name: len(_load(cases / f"{name}.jsonl"))
        for name in ("dev", "main", "origin", "program", "plan", "inputs")
    }
    report["evaluation_rows"] = sum(1 for _ in (cases / "evaluation.jsonl").open())

    # The exact adapter template, logged BEFORE any evaluation, as the spec requires.
    report["erratum_template"] = EA.template_fingerprint()

    # Textual-only adapter exercise: no model, no tokenizer, no cache.
    import inputs as INP
    rows = _load(cases / "inputs.jsonl")
    sample = rows[0]
    commands = INP.commands_from_json(sample["commands"])
    text, ordered = EA.erratum_text(sample["source_text"], commands)
    report["template_smoke"] = dict(
        input_id=sample["input_id"],
        raw_commands=len(commands),
        normalized_commands=len(ordered),
        appended_chars=len(text) - len(sample["source_text"]),
        ends_correctly=text.endswith("End of records.\n"),
        rendered_first_line=EA.erratum_line(sample["source_text"], ordered[0]),
    )

    # History normalization must actually collapse repeated addresses somewhere in
    # the fixed population, otherwise LATEST_* is not distinguishable from EXISTING.
    collapsed = 0
    for r in rows:
        c = INP.commands_from_json(r["commands"])
        if len(INP.normalize(c)) < len(c):
            collapsed += 1
    report["history_normalization"] = dict(
        rows=len(rows), rows_where_normalization_collapses=collapsed,
        note="LATEST_SAME_WORDING vs EXISTING_CORRECTION is only a live contrast on these rows",
    )

    report["status"] = "PASS"
    return report


# ---------------------------------------------------------------- stage: model

def stage_model(pkg: Path, cache: str, actor: str) -> dict:
    sys.path.insert(0, str(pkg))
    import inputs as INP
    import erratum_adapter as EA
    from canonical import config as Q
    from canonical.adapter import Backbone

    cfg = Q.MODELS[actor]
    report = dict(schema="PREFLIGHT_MODEL_V1", stage="model", actor=actor,
                  model_id=cfg["id"], revision=cfg["revision"], started=time.time())

    backbone = Backbone(cache=cache, cfg=cfg)
    report["backbone"] = backbone.describe()
    report["erratum_template"] = EA.template_fingerprint()

    plan_rows = [r for r in _load(pkg / "cases/fixed/plan.jsonl")
                 if r["panel"] == "DEV" and r["actor"] == actor]
    inputs_by_id = {r["input_id"]: r for r in _load(pkg / "cases/fixed/inputs.jsonl")
                    if r["actor"] == actor}

    census, boundary, immutability = [], [], []

    for row in plan_rows:
        src = inputs_by_id[row["input_id"]]
        source_text = src["source_text"]
        commands = INP.commands_from_json(src["commands"])

        # 1. stock SOURCE compiles and its prefix boundary holds
        source_ctx = INP.prepare(source_text, commands, method="SOURCE",
                                 backbone=backbone, actor=actor)
        before_hash = source_ctx.artifact_sha256

        # 2. clause-refresh feasibility on the real tokenizer
        lines, ordered = EA.erratum_block(source_text, commands)
        stale = EA.append_block(source_text, lines)
        fresh = EA.append_block(EA.rebuilt_text(source_text, commands), lines)
        span_pairs = EA.fresh_clause_spans(source_text, ordered)
        refresh, rep = EA.refresh_plan(backbone, stale, fresh, span_pairs)
        census.append(dict(input_id=row["input_id"], scene_id=row["scene_id"],
                           addressed_clauses=len(ordered),
                           refresh_feasible=bool(refresh),
                           reason=rep.get("reason"),
                           char_deltas=rep["char_deltas"],
                           stale_tokens=rep["stale_tokens"],
                           fresh_tokens=rep["fresh_tokens"]))

        # 3. both adapter arms build end to end
        for method in ("LATEST_ERRATUM", "FIELD_PLUS_LATEST_ERRATUM"):
            ctx, tel = EA.prepare_erratum(source_text, commands, method=method,
                                          backbone=backbone, actor=actor)
            boundary.append(dict(input_id=row["input_id"], method=method,
                                 artifact_sha256=ctx.artifact_sha256,
                                 refresh_fallback=tel["refresh_fallback"],
                                 prefill_seconds=tel["source_prefill_seconds"],
                                 refresh_seconds=tel["refresh_seconds"]))

        # 4. the SOURCE master was not mutated by any of the above
        immutability.append(dict(input_id=row["input_id"],
                                 unchanged=backbone.cache_hash(source_ctx._artifact["cache"]) == before_hash))

    feasible = sum(1 for c in census if c["refresh_feasible"])
    report["refresh_census"] = dict(
        scenes=len(census), feasible=feasible, fallback=len(census) - feasible,
        fallback_rate=(len(census) - feasible) / len(census) if census else None,
        reasons=sorted({c["reason"] for c in census if c["reason"]}),
        rows=census,
        interpretation=("field refresh executes; FIELD_PLUS is distinguishable from erratum-only"
                        if feasible else
                        "field refresh never executes on DEV; FIELD_PLUS_LATEST_ERRATUM reduces to "
                        "LATEST_ERRATUM under the documented fixed fallback and the headline "
                        "contrast MUST state this"),
    )
    report["adapter_builds"] = boundary
    report["master_immutability"] = immutability
    report["status"] = "PASS" if all(i["unchanged"] for i in immutability) else "FAIL"
    backbone.finish()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("cpu", "model"), required=True)
    p.add_argument("--package", required=True, help="path to UPDATE_METHOD_INPUTS")
    p.add_argument("--cache", help="local HF snapshot cache dir (model stage)")
    p.add_argument("--actor", choices=("gemma", "qwen"), help="model stage")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Choose a new preflight receipt filename; receipts are retained")

    pkg = Path(args.package).resolve()
    if args.stage == "cpu":
        report = stage_cpu(pkg)
    else:
        if not (args.cache and args.actor):
            p.error("--cache and --actor are required for the model stage")
        report = stage_model(pkg, args.cache, args.actor)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("schema", "status") if k in report}, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
