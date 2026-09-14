#!/usr/bin/env python3
"""Prospectively frozen internal replication (addendum replication_pilot_v1) of the exploratory
Arm C finding: does the selected-layer activation improve out-of-fold prediction of the reviewed
Supports/Opposes label beyond text, answer mapping, and the model's own native report margin?

  analyze --actor A --gpu-root <extraction of POPULATION.jsonl>  -> <out>/<actor>/REPLICATION_<actor>.json
  assemble                                                        -> REPLICATION_RESULTS.{md,json,csv}, packet, manifest

Everything analytic is inherited unchanged from rrrd_v1_q_report_v2 (NestedProbe, text features,
logistic grid); only the population, the unit, the baseline definition and the decision rule differ,
exactly as frozen in replication_pilot_v1/freeze/REPLICATION_CONTRACT.json.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as C  # noqa: E402
from rrrd_v1_no_inference import WorldBootstrap, fmt  # noqa: E402
from rrrd_v1_q_report_v2 import NestedProbe, auroc, ba, logloss, text_features  # noqa: E402

REPL = C.STUDY_ROOT / "replication_pilot_v1"
FREEZE = REPL / "freeze"


def rule(delta: dict[str, Any]) -> str:
    if delta["ci_low"] > 0:
        return "POSITIVE"
    if delta["ci_high"] < 0:
        return "NEGATIVE"
    return "UNDETERMINED"


def analyze(actor: str, gpu_root: Path, out_dir: Path, replicates: int, cache_dir: Path) -> dict[str, Any]:
    man = C.read_json(gpu_root / "artifact_manifest.json")
    bad = [f["path"] for f in man["files"] if C.sha256_file(gpu_root / f["path"]) != f["sha256"]]
    if bad:
        raise RuntimeError(f"extraction hash mismatch: {bad[:5]}")
    pop = {r["row_id"]: r for r in C.read_jsonl(FREEZE / "POPULATION.jsonl")}
    if man.get("cells_file_sha256") != C.sha256_file(FREEZE / "POPULATION.jsonl"):
        raise RuntimeError("extraction was not run on the frozen POPULATION.jsonl")
    folds = C.read_json(FREEZE / "FOLD_ASSIGNMENTS.json")["assignment"]
    scores = C.read_jsonl(gpu_root / "naturalistic_scores.jsonl")
    cfg = C.ACTORS[actor]
    with np.load(gpu_root / "naturalistic_activations.npz", allow_pickle=False) as z:
        ids = [str(v) for v in z["prompt_ids"]]
        x_all = np.asarray(z[f"layer_{cfg['primary_layer']}"], dtype=np.float64)
    if ids != [r["row_id"] for r in scores]:
        raise RuntimeError("activation rows do not align with scores")
    bundle = np.load(C.TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz", allow_pickle=False)
    token_dir = np.asarray(bundle["answer_token"], dtype=np.float64)
    res: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION", "actor": actor, "evidence_status": "PROSPECTIVELY_FROZEN_INTERNAL_REPLICATION", "layer": cfg["primary_layer"], "populations": {}}
    for popname in ("primary_replication", "secondary_board_population"):
        sel = [i for i, r in enumerate(scores) if pop[r["row_id"]]["population"] == popname]
        rows = [pop[scores[i]["row_id"]] for i in sel]
        x = x_all[sel]
        y_label = np.array([r["label"] for r in rows])
        margin = np.array([scores[i]["semantic_margin"] for i in sel])
        y_report = (margin > 0).astype(int)
        units = np.array([r["unit_id"] for r in rows])
        tf = text_features(rows, cache_dir)
        feats = {"text": tf["text"], "covariates_nopos": tf["covariates_nopos"], "native_margin": margin[:, None], "label": y_label[:, None].astype(float), "activation": x, "activation_tok": x - np.outer(x @ token_dir / (token_dir @ token_dir), token_dir)}
        probe = NestedProbe(units, folds[popname])
        boot = WorldBootstrap(sorted(set(units)), replicates, C.BOOTSTRAP_SEED)

        def unit_mean(v: np.ndarray) -> dict[str, float]:
            d: dict[str, list[float]] = collections.defaultdict(list)
            for u, val in zip(units, v):
                d[u].append(float(val))
            return {u: float(np.mean(vals)) for u, vals in d.items()}

        def ladder(y: np.ndarray, base: tuple[str, ...], act: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
            f = dict(feats)
            f["activation"] = feats[act]
            m0, m1 = probe.run(f, y, base), probe.run(f, y, base + ("activation",))
            imp = m0["ll"] - m1["ll"]
            d = boot.summary(unit_mean(imp))
            return {"baseline": {"chosen_C": m0["chosen_C"], "logloss": float(m0["ll"].mean()), "auroc": auroc(y, m0["prob"]), "balanced_accuracy": ba(y, (m0["prob"] > 0.5).astype(int))}, "augmented": {"chosen_C": m1["chosen_C"], "logloss": float(m1["ll"].mean()), "auroc": auroc(y, m1["prob"]), "balanced_accuracy": ba(y, (m1["prob"] > 0.5).astype(int))}, "delta_logloss": d, "chance_logloss": float(logloss(y, np.full(len(y), y.mean())).mean())}, m0["prob"], m1["prob"]

        L, p0, p1 = ladder(y_label, ("text", "covariates_nopos", "native_margin"), "activation")
        L_tok, _, _ = ladder(y_label, ("text", "covariates_nopos", "native_margin"), "activation_tok")
        R, _, _ = ladder(y_report, ("text", "covariates_nopos", "label"), "activation")
        lab = probe.run(feats, y_label, ("activation",))
        rep = probe.run(feats, y_report, ("activation",))
        cos = [float(lab["coef"][k] @ rep["coef"][k] / (np.linalg.norm(lab["coef"][k]) * np.linalg.norm(rep["coef"][k]) + 1e-12)) for k in lab["coef"]]
        dis = y_label != y_report
        oof = [{"row_id": rows[j]["row_id"], "unit_id": units[j], "fold": folds[popname][units[j]], "label": int(y_label[j]), "report_sign": int(y_report[j]), "native_margin": float(margin[j]), "L0_prob": float(p0[j]), "L1_prob": float(p1[j]), "label_probe_prob": float(lab["prob"][j]), "report_probe_prob": float(rep["prob"][j])} for j in range(len(rows))]
        C.write_create_only(out_dir / f"OOF_PREDICTIONS_{popname}.csv", C.csv_bytes(oof, list(oof[0].keys())))
        res["populations"][popname] = {
            "rows": len(rows), "units": len(set(units)), "report_equals_label": float(np.mean(y_report == y_label)), "disagreement_rows": int(dis.sum()),
            "primary": {"L": L, "rule": rule(L["delta_logloss"])},
            "robustness_token_removed": {"L": L_tok, "rule": rule(L_tok["delta_logloss"])},
            "descriptive": {"R": R, "label_probe": {"auroc": auroc(y_label, lab["prob"]), "balanced_accuracy": ba(y_label, (lab["prob"] > 0.5).astype(int))}, "report_probe": {"auroc": auroc(y_report, rep["prob"]), "balanced_accuracy": ba(y_report, (rep["prob"] > 0.5).astype(int))}, "coefficient_cosine_mean": float(np.mean(cos)), "disagreement_rows": {"count": int(dis.sum()), "label_probe_sides_with_label": float(np.mean((lab["prob"][dis] > 0.5) == y_label[dis])) if dis.any() else None, "L1_sides_with_label": float(np.mean((p1[dis] > 0.5) == y_label[dis])) if dis.any() else None, "L0_sides_with_label": float(np.mean((p0[dis] > 0.5) == y_label[dis])) if dis.any() else None, "report_probe_sides_with_report": float(np.mean((rep["prob"][dis] > 0.5) == y_report[dis])) if dis.any() else None}},
        }
    res["classification"] = res["populations"]["primary_replication"]["primary"]["rule"]
    res["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return res


def assemble(output_root: Path, gpu_roots: dict[str, Path]) -> None:
    per = {a: C.read_json(output_root / a / f"REPLICATION_{a}.json") for a in C.ACTORS if (output_root / a / f"REPLICATION_{a}.json").is_file()}
    actors = list(per)
    for a in actors:
        dst = output_root / a / "extraction"
        if not dst.exists():
            dst.mkdir(parents=True)
            for name in ("naturalistic_scores.jsonl", "artifact_manifest.json", "RUN_RECEIPT.json", "environment.json", "PROMOTION_RECEIPT.json"):
                if (gpu_roots[a] / name).is_file():
                    shutil.copy2(gpu_roots[a] / name, dst / name)
    classes = {a: per[a]["classification"] for a in actors}
    md = ["# Replication of the exploratory Arm C finding (replication_pilot_v1)", "", "Evidence status: PROSPECTIVELY_FROZEN_INTERNAL_REPLICATION (plan committed before extraction; populations disjoint from the 500-cell discovery set; rows fed M1 development and the DIM fit, so no DIM statistic is computed).", "", "Primary: Delta = logloss(L0) - logloss(L1) on the reviewed label, L0 = text + mapping + native margin, L1 = L0 + activation; unit bootstrap 10,000; rule POSITIVE / UNDETERMINED / NEGATIVE by the interval's position relative to zero.", ""]
    csv_rows = []
    for a in actors:
        r = per[a]
        md += [f"## {a}: primary population **{r['classification']}**", "", "| population | rows | units | report=label | Delta log-loss [95% CI] | rule | token-removed Delta | L0 AUROC → L1 AUROC | L0 BA → L1 BA | R Delta (descriptive) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for popname, p in r["populations"].items():
            L, Lt, R = p["primary"]["L"], p["robustness_token_removed"]["L"], p["descriptive"]["R"]
            md.append(f"| {popname} | {p['rows']} | {p['units']} | {p['report_equals_label']:.3f} | {fmt(L['delta_logloss'])} | {p['primary']['rule']} | {fmt(Lt['delta_logloss'])} ({p['robustness_token_removed']['rule']}) | {L['baseline']['auroc']:.3f} → {L['augmented']['auroc']:.3f} | {L['baseline']['balanced_accuracy']:.3f} → {L['augmented']['balanced_accuracy']:.3f} | {fmt(R['delta_logloss'])} |")
            csv_rows.append({"actor": a, "population": popname, "rows": p["rows"], "units": p["units"], "delta_point": L["delta_logloss"]["point"], "delta_ci_low": L["delta_logloss"]["ci_low"], "delta_ci_high": L["delta_logloss"]["ci_high"], "rule": p["primary"]["rule"], "token_removed_delta_point": Lt["delta_logloss"]["point"], "token_removed_ci_low": Lt["delta_logloss"]["ci_low"], "token_removed_ci_high": Lt["delta_logloss"]["ci_high"], "L0_auroc": L["baseline"]["auroc"], "L1_auroc": L["augmented"]["auroc"], "L0_ba": L["baseline"]["balanced_accuracy"], "L1_ba": L["augmented"]["balanced_accuracy"], "R_delta_point": R["delta_logloss"]["point"]})
        for popname, p in r["populations"].items():
            d = p["descriptive"]
            md += ["", f"{popname} descriptives: label probe AUROC {d['label_probe']['auroc']:.3f} / BA {d['label_probe']['balanced_accuracy']:.3f}; report probe AUROC {d['report_probe']['auroc']:.3f}; coefficient cosine {d['coefficient_cosine_mean']:.3f}; disagreement rows {d['disagreement_rows']}"]
        md.append("")
    C.write_create_only(output_root / "REPLICATION_RESULTS.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_RESULTS", "classification": classes, "actors": per}))
    C.write_create_only(output_root / "REPLICATION_RESULTS.md", ("\n".join(md) + "\n").encode("utf-8"))
    C.write_create_only(output_root / "REPLICATION_RESULTS.csv", C.csv_bytes(csv_rows, list(csv_rows[0].keys())))
    contract = C.read_json(FREEZE / "REPLICATION_CONTRACT.json")
    packet = {"schema_version": "EXTERNAL_REVIEW_PACKET_V1", "unit": f"{C.STUDY_ID} / replication_pilot_v1", "evidence_status": contract["evidence_status"], "research_question": {"question": contract["hypothesis"], "target_variable": contract["primary_statistic"], "decision_rule": contract["decision_rule"], "discovery": contract["discovery"]}, "capability_delta": {"new_runnable_behavior": "scripts/rrrd_v1_replication_pilot.py on a fresh batch-size-one extraction of the 1,800 pilot rows (scripts/rrrd_v1_gpu_run.py rerun-naturalistic --cells-file)"}, "evidence_topology": {a: {p: {"rows": per[a]["populations"][p]["rows"], "units": per[a]["populations"][p]["units"], "unit": "situation" if p == "primary_replication" else "board"} for p in per[a]["populations"]} for a in actors}, "exact_empirical_result": {"classification": classes, "per_actor": {a: {p: {"primary": per[a]["populations"][p]["primary"], "token_removed": per[a]["populations"][p]["robustness_token_removed"], "descriptive": per[a]["populations"][p]["descriptive"]} for p in per[a]["populations"]} for a in actors}}, "negative_and_surprising_evidence": [f"{a} {p}: {per[a]['populations'][p]['primary']['rule']} ({fmt(per[a]['populations'][p]['primary']['L']['delta_logloss'])})" for a in actors for p in per[a]["populations"] if per[a]["populations"][p]["primary"]["rule"] != "POSITIVE"], "what_was_held_fixed": ["everything in freeze/REPLICATION_CONTRACT.json"], "deviations": [], "claim_boundary": {"licensed_if_positive_in_llama": contract["licensed_wording_if_positive_in_llama"], "prohibited": contract["prohibited_wording"]}, "artifact_pickup": {"results": "REPLICATION_RESULTS.json", "oof": {a: f"{a}/OOF_PREDICTIONS_*.csv" for a in actors}, "extraction": "NFS outputs/rrrd_v1/replication_pilot_v1/<actor>"}}
    C.write_create_only(output_root / "external_review_packet_v1.json", C.canonical_json(packet))
    C.write_create_only(output_root / "external_review_packet_v1.md", ("\n".join(["# External review packet — replication_pilot_v1", "", f"Classification (primary population): {classes}", "", "## Non-positive rows", ""] + [f"- {n}" for n in packet["negative_and_surprising_evidence"]] + ["", "See REPLICATION_RESULTS.md.", ""])).encode("utf-8"))
    C.write_create_only(output_root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_MANIFEST", "classification": classes, "files": C.manifest_for(output_root)}))
    print(C.json.dumps(classes, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    a.add_argument("--gpu-root", type=Path, required=True)
    a.add_argument("--output-root", type=Path, required=True)
    a.add_argument("--cache-dir", type=Path, required=True)
    a.add_argument("--replicates", type=int, default=C.BOOTSTRAP_REPLICATES)
    s = sub.add_parser("assemble")
    s.add_argument("--output-root", type=Path, required=True)
    s.add_argument("--gpu-root-llama", type=Path, required=True)
    s.add_argument("--gpu-root-gemma", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        out_dir = args.output_root / args.actor
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"output not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        res = analyze(args.actor, args.gpu_root, out_dir, args.replicates, args.cache_dir)
        res["source"] = {"extraction_manifest_sha256": C.sha256_file(args.gpu_root / "artifact_manifest.json"), "script_sha256": C.sha256_file(Path(__file__).resolve()), "replicates": args.replicates}
        C.write_create_only(out_dir / f"REPLICATION_{args.actor}.json", C.canonical_json(res))
        C.write_create_only(out_dir / "artifact_manifest.json", C.canonical_json({"files": C.manifest_for(out_dir)}))
        print(C.json.dumps({"classification": res["classification"]}, indent=2))
    else:
        assemble(args.output_root, {"llama": args.gpu_root_llama, "gemma": args.gpu_root_gemma})
    return 0


if __name__ == "__main__":
    sys.exit(main())
