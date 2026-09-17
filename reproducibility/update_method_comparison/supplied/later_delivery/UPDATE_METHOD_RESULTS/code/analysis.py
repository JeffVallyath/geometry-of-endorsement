"""Turn raw score rows into the required tables.

The primary endpoint and its bootstrap are NOT reimplemented here. The supplied
package already fixes them in metrics.paired_primary:

    endpoint  complete 34-question bundle correct from all four origins
    unit      terminal root
    seeds     carried together inside each root, baseline subtracted ONCE
    bootstrap paired root, 10,000 draws, seed 260913902, 98.75% intervals
    contrasts exactly four (2 recipes x 2 models) vs FIELD_PLUS_LATEST_ERRATUM

This module's only job is to build the `vectors` argument correctly and to emit
the secondary tables. metrics.paired_primary fails closed on missing roots,
duplicate baseline seeds, or incomplete contrasts, which is the behaviour we want.

Secondary outputs (all explicitly secondary per the spec, and all kept visible
whether or not they favour the learned arms):
    all-origin question accuracy, disagreement, required-change accuracy,
    invariant error, direct-operand checks, all-24/main correctness, and the
    inherited single/repeat/restore/AB/ABC criteria.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(raw: Path):
    rows = []
    for p in sorted(raw.glob("*.jsonl")):
        if p.name.endswith(".telemetry.jsonl"):
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_telemetry(raw: Path):
    out = []
    for p in sorted(raw.glob("*.telemetry.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------- primary endpoint

def bundle_vectors(rows):
    """{(actor, recipe_or_baseline, seed): {root_id: 0/1}} for metrics.paired_primary.

    A root scores 1 only when every question of the 34-question bundle is correct
    from all four origins. Any invalid answer fails the root. Nothing is imputed.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    tagged = sum(1 for r in rows if "input_panel" in r)
    if tagged == 0:
        raise SystemExit(
            "No score row carries 'input_panel'. These rows predate the grouping-key "
            "fix in run_comparison.py, so the primary endpoint cannot be assembled. "
            "Re-tag them from plan.jsonl by input_id, or re-run the affected shards. "
            "Failing loudly here rather than silently reporting zero contrasts.")
    for r in rows:
        if r.get("input_panel") != "ORIGIN":
            continue
        grouped[(r["actor"], r["condition"])][r["root_id"]].append(r)

    vectors, coverage = {}, {}
    for (actor, condition), roots in grouped.items():
        if "_s" in condition:
            method, seed = condition.rsplit("_s", 1)
            seed = int(seed)
        else:
            method, seed = condition, None
        vec, cov = {}, {}
        for root, rr in roots.items():
            origins = {x["origin"] for x in rr}
            complete = len(origins) == 4 and all(x["correct"] for x in rr)
            vec[root] = 1 if complete else 0
            cov[root] = dict(rows=len(rr), origins=sorted(o for o in origins if o is not None))
        vectors[(actor, method, seed)] = vec
        coverage[(actor, condition)] = cov
    return vectors, coverage


def primary_table(pkg: Path, rows):
    sys.path.insert(0, str(pkg))
    from metrics import paired_primary
    from inputs import RECIPES

    vectors, _ = bundle_vectors(rows)
    expected_roots = defaultdict(set)
    for r in rows:
        if r.get("input_panel") == "ORIGIN":
            expected_roots[r["actor"]].add(r["root_id"])

    wanted = {(a, rc, s) for a in ("gemma", "qwen") for rc in RECIPES for s in (0, 1)}
    wanted |= {(a, "FIELD_PLUS_LATEST_ERRATUM", None) for a in ("gemma", "qwen")}
    have = set(vectors) & wanted
    missing = sorted(map(str, wanted - have))
    if missing:
        return None, dict(status="INCOMPLETE", missing_contrast_cells=missing,
                          note="paired_primary fails closed; these cells are identified, not dropped")

    return paired_primary({k: v for k, v in vectors.items() if k in wanted},
                          {a: sorted(v) for a, v in expected_roots.items()}), dict(status="COMPLETE")


# ------------------------------------------------------------ secondary tables

def per_scene(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["actor"], r["condition"], r.get("scene_id"), r.get("input_panel"))].append(r)
    out = []
    for (actor, cond, scene, panel), rr in sorted(g.items(), key=lambda x: str(x[0])):
        valid = [x for x in rr if x["valid"]]
        out.append(dict(actor=actor, condition=cond, scene_id=scene, panel=panel,
                        questions=len(rr), valid=len(valid),
                        correct=sum(1 for x in rr if x["correct"]),
                        accuracy=(sum(1 for x in rr if x["correct"]) / len(rr)) if rr else None,
                        all_correct=int(bool(rr) and all(x["correct"] for x in rr)),
                        disagreement=sum(1 for x in rr if x["valid"] and x["source_valid"]
                                         and x["prediction"] != x["source_prediction"])))
    return out


def per_root(rows):
    g = defaultdict(list)
    for r in rows:
        if r.get("input_panel") != "ORIGIN":
            continue
        g[(r["actor"], r["condition"], r["root_id"])].append(r)
    out = []
    for (actor, cond, root), rr in sorted(g.items(), key=lambda x: str(x[0])):
        origins = sorted({x["origin"] for x in rr if x["origin"] is not None})
        out.append(dict(actor=actor, condition=cond, root_id=root,
                        origins_present=len(origins), questions=len(rr),
                        correct=sum(1 for x in rr if x["correct"]),
                        bundle_complete=int(len(origins) == 4 and all(x["correct"] for x in rr)),
                        mean_question_accuracy=(sum(1 for x in rr if x["correct"]) / len(rr)) if rr else None))
    return out


def headline(rows, primary):
    g = defaultdict(list)
    for r in rows:
        g[(r["actor"], r["condition"])].append(r)
    eff = {}
    for entry in (primary or []):
        eff[(entry["actor"], entry["recipe"])] = entry

    out = []
    for (actor, cond), rr in sorted(g.items(), key=lambda x: str(x[0])):
        base = cond.rsplit("_s", 1)[0] if "_s" in cond else cond
        e = eff.get((actor, base))
        origin_rows = [x for x in rr if x.get("input_panel") == "ORIGIN"]
        main_rows = [x for x in rr if x.get("input_panel") == "MAIN"]
        out.append(dict(
            actor=actor, condition=cond, method=base,
            questions=len(rr),
            all_origin_question_accuracy=(sum(1 for x in origin_rows if x["correct"]) / len(origin_rows)) if origin_rows else None,
            main_question_accuracy=(sum(1 for x in main_rows if x["correct"]) / len(main_rows)) if main_rows else None,
            invalid=sum(1 for x in rr if not x["valid"]),
            disagreement_vs_source=sum(1 for x in rr if x["valid"] and x["source_valid"]
                                       and x["prediction"] != x["source_prediction"]),
            primary_effect_vs_baseline=e["effect"] if e else None,
            primary_ci_low=e["lower"] if e else None,
            primary_ci_high=e["upper"] if e else None,
        ))
    return out


def timing_memory(telemetry):
    g = defaultdict(list)
    for t in telemetry:
        g[(t["actor"], t["condition"])].append(t)
    out = []
    for (actor, cond), tt in sorted(g.items(), key=lambda x: str(x[0])):
        def mean(key):
            vals = [x[key] for x in tt if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        attempted = sum(1 for x in tt if x.get("refresh_attempted"))
        fallback = sum(1 for x in tt if x.get("refresh_fallback"))
        out.append(dict(actor=actor, condition=cond, cases=len(tt),
                        mean_preprocess_seconds=mean("preprocess_seconds"),
                        mean_source_prefill_seconds=mean("source_prefill_seconds"),
                        mean_update_refresh_seconds=mean("refresh_seconds"),
                        mean_compile_seconds=mean("compile_seconds"),
                        refresh_attempted=attempted, refresh_fallback=fallback,
                        refresh_fallback_rate=(fallback / attempted) if attempted else None))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--tables", required=True)
    args = p.parse_args()

    pkg, raw, tables = Path(args.package).resolve(), Path(args.raw), Path(args.tables)
    rows, telemetry = load_rows(raw), load_telemetry(raw)
    if not rows:
        raise SystemExit("No raw score rows found. Run run_comparison.py first.")

    primary, status = primary_table(pkg, rows)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "primary_contrasts.json").write_text(json.dumps(
        dict(status=status, contrasts=primary), indent=2, default=str))

    write_csv(tables / "headline_results.csv", headline(rows, primary),
              ["actor", "condition", "method", "questions", "all_origin_question_accuracy",
               "main_question_accuracy", "invalid", "disagreement_vs_source",
               "primary_effect_vs_baseline", "primary_ci_low", "primary_ci_high"])
    write_csv(tables / "per_scene_results.csv", per_scene(rows),
              ["actor", "condition", "scene_id", "panel", "questions", "valid",
               "correct", "accuracy", "all_correct", "disagreement"])
    write_csv(tables / "per_root_results.csv", per_root(rows),
              ["actor", "condition", "root_id", "origins_present", "questions",
               "correct", "bundle_complete", "mean_question_accuracy"])
    write_csv(tables / "timing_memory.csv", timing_memory(telemetry),
              ["actor", "condition", "cases", "mean_preprocess_seconds",
               "mean_source_prefill_seconds", "mean_update_refresh_seconds",
               "mean_compile_seconds", "refresh_attempted", "refresh_fallback",
               "refresh_fallback_rate"])

    print(json.dumps(dict(rows=len(rows), telemetry=len(telemetry), primary=status), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
