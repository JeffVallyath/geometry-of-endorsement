#!/usr/bin/env python3
"""CPU fit and calibration stages for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2.

Modes (all read only unsteered fit/selection-world outputs; none reads a final-world outcome):
  select-protocol   Llama: the five natural reference gates for each predeclared protocol on the 32 selection
                    worlds, frozen lexicographic choice; Gemma: P1 gates recorded (protocol fixed by the goal).
  fit-directions    regime-matched relation DIM and answer-token DIM on the 64 fit worlds (selected protocol),
                    world-grouped 8-fold qualification; writes DIRECTIONS_<actor>.npz/.json (every unit direction
                    used by the runner) and DIRECTION_FIT_<actor>.json.
  fit-witness       template-matched witness (witness layer, relation directions projected out) on the fit worlds.
  calibrate         scales from the fit-world natural signatures, alpha-grid reachability and norm diagnostics per
                    arm and tau from the fit-world calibrate stage; DEV_CALIBRATION_<actor>.json.
  final-freeze      FINAL_ANALYSIS_FREEZE.json (protocol, scales, direction/witness hashes, thresholds) from the
                    per-actor calibrations; must precede every final-world forward.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stcrf_v2_common as C  # noqa: E402

FIT_QUERIES = ("Q1", "Q2", "Q3", "Q4")


# ---------------------------------------------------------------- shared extraction
def act_key(fmt: str, real: str, state: str, q: str, intervention: str = "NO_INTERVENTION", tau: float | None = None) -> str:
    return f"{fmt}|{real}|{state}|{q}|{intervention}|{'' if tau is None else tau}"


def load_world_acts(gpu_root: Path, wid: str, layer: str) -> dict[str, np.ndarray]:
    with np.load(gpu_root / "worlds" / f"{wid}.npz", allow_pickle=False) as z:
        keys, arr = z["keys"], z[layer]
    return {str(k): arr[i] for i, k in enumerate(keys)}


def natural_signature(summary: dict[str, Any], fmt: str, real: str) -> dict[str, Any]:
    rows = summary["rows"]
    nat = {(r["state"], r["query"]): r for r in rows if r["format"] == fmt and r["realization"] == real and r["intervention"] == "NO_INTERVENTION"}
    base = {q: nat[("base", q)]["semantic_margin"] for q in C.QUERIES}
    cf = {q: nat[("counterfactual", q)]["semantic_margin"] for q in C.QUERIES}
    N = np.array([cf[q] - base[q] for q in C.CONSEQUENCE_QUERIES])
    native = {(s, q): {"margin": nat[(s, q)]["semantic_margin"], "label": nat[(s, q)]["label"], "native": "ENTAILED" if nat[(s, q)]["semantic_margin"] > 0 else "NOT_ENTAILED"} for s in ("base", "counterfactual") for q in C.QUERIES}
    return {"N": N, "N_Q1": cf["Q1"] - base["Q1"], "base": base, "cf": cf, "native": native}


def natural_gates(summaries: list[dict[str, Any]], scales: np.ndarray | None, idx: np.ndarray, patch_csr: dict[str, Any] | None = None) -> dict[str, Any]:
    """The natural reference gates (five natural-only gates; the full-state patch gate when patch_csr is given)."""
    sA = [natural_signature(s, "F2", "A") for s in summaries]
    sB = [natural_signature(s, "F2", "B") for s in summaries]
    f1 = [natural_signature(s, "F1", "A") for s in summaries]
    directions = [s["direction"] for s in summaries]
    pred, truth = [], []
    for sig in sA + sB:
        for v in sig["native"].values():
            pred.append(int(v["native"] == "ENTAILED"))
            truth.append(int(v["label"] == "ENTAILED"))
    ba = C.balanced_accuracy(pred, truth)
    agree = float(np.mean([int(a["native"][k]["native"] == b["native"][k]["native"]) for a, b in zip(f1, sA) for k in b["native"]]))
    exp = [int(np.sign(sig["N"][C.CONSEQUENCE_QUERIES.index(q)]) == C.expected_sign(d, q)) for sig, d in zip(sA, directions) for q in C.FOCAL_QUERIES]
    focal_dir = float(np.mean(exp))
    NA = np.stack([s["N"] for s in sA])
    NB = np.stack([s["N"] for s in sB])
    fi = [C.CONSEQUENCE_QUERIES.index(q) for q in C.FOCAL_QUERIES]
    ii = [C.CONSEQUENCE_QUERIES.index(q) for q in C.INVARIANCE_QUERIES]
    ratio = float(np.median(np.abs(NA[:, ii])) / np.median(np.abs(NA[:, fi])))
    if scales is None:
        scales = C.robust_scales(NA)
    d0 = C.squared_distance(np.zeros_like(NA), NA, scales)
    ctnc_w = 1.0 - C.squared_distance(NB, NA, scales) / d0
    ctnc = C.bootstrap_mean(ctnc_w, idx)
    g = C.REFERENCE_GATES
    checks = {
        "primary_format_balanced_accuracy": {"value": ba, "min": g["primary_format_balanced_accuracy_min"], "pass": bool(ba >= g["primary_format_balanced_accuracy_min"])},
        "mapped_format_semantic_agreement": {"value": agree, "min": g["mapped_format_semantic_agreement_min"], "pass": bool(agree >= g["mapped_format_semantic_agreement_min"])},
        "focal_consequence_expected_direction": {"value": focal_dir, "min": g["focal_consequence_expected_direction_min"], "pass": bool(focal_dir >= g["focal_consequence_expected_direction_min"])},
        "control_stability_ratio": {"value": ratio, "max": g["control_stability_ratio_max"], "pass": bool(ratio <= g["control_stability_ratio_max"])},
        "cross_template_consistency": {"value": {k: ctnc[k] for k in ("mean", "ci_low", "ci_high")}, "mean_min": g["cross_template_consistency_mean_min"], "ci_low_min": g["cross_template_consistency_ci_low_min"], "pass": bool(ctnc["mean"] >= g["cross_template_consistency_mean_min"] and ctnc["ci_low"] >= g["cross_template_consistency_ci_low_min"])},
    }
    if patch_csr is not None:
        checks["full_state_patch_csr"] = {"value": patch_csr, "min": g["full_state_patch_csr_min"], "pass": bool(patch_csr["mean"] >= g["full_state_patch_csr_min"])}
    natural_pass_count = sum(int(checks[k]["pass"]) for k in C.NATURAL_ONLY_GATES)
    return {"checks": checks, "natural_gates_passed": natural_pass_count, "all_pass": bool(all(v["pass"] for v in checks.values())), "scales_used": scales.tolist(), "ctnc_per_world": ctnc_w.tolist(), "natural_signature_means_A": {q: float(NA[:, i].mean()) for i, q in enumerate(C.CONSEQUENCE_QUERIES)}, "natural_q1_change_A": {"mean": float(np.mean([s["N_Q1"] for s in sA])), "median": float(np.median([s["N_Q1"] for s in sA])), "min": float(np.min([s["N_Q1"] for s in sA]))}, "worlds": len(summaries)}


# ---------------------------------------------------------------- protocol selection
def select_protocol(actor: str, roots: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per: dict[str, Any] = {}
    for proto, root in roots.items():
        summaries = [s for s in C.read_jsonl(root / "world_summaries.jsonl") if s["split"] == "selection" and s["protocol"] == proto]
        idx = C.bootstrap_indices(len(summaries))
        per[proto] = natural_gates(summaries, None, idx) | {"gpu_root": str(root), "gpu_manifest_sha256": C.sha256_file(root / "artifact_manifest.json")}
    if actor == "gemma":
        chosen, rule = C.GEMMA_PROTOCOL, "fixed by the goal (V1 validated F2 protocol)"
        qualified = True
    else:
        order = sorted(per, key=lambda p: (-per[p]["natural_gates_passed"], -per[p]["checks"]["cross_template_consistency"]["value"]["mean"], -per[p]["checks"]["primary_format_balanced_accuracy"]["value"], p))
        chosen = order[0]
        qualified = per[chosen]["natural_gates_passed"] == len(C.NATURAL_ONLY_GATES)
        rule = C.PROTOCOL_SELECTION_RULE
    rec = {"schema_version": f"{C.SCHEMA_PREFIX}_PROTOCOL_SELECTION", "actor": actor, "candidates": {p: {"natural_gates_passed": v["natural_gates_passed"], "checks": v["checks"], "natural_q1_change_A": v["natural_q1_change_A"], "gpu_root": v["gpu_root"], "gpu_manifest_sha256": v["gpu_manifest_sha256"], "worlds": v["worlds"]} for p, v in per.items()}, "rule": rule, "selected_protocol": chosen, "assay_qualified": bool(qualified), "status": "PROTOCOL_FROZEN" if qualified else "ASSAY_NOT_QUALIFIED", "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(), "note": "selection uses unsteered natural base/counterfactual behavior on the 32 selection worlds only; the choice is frozen before any fit or final inference and is never revisited"}
    C.write_create_only(output_dir / f"PROTOCOL_SELECTION_{actor}.json", C.canonical_json(rec))
    print(json.dumps({"actor": actor, "selected": chosen, "qualified": qualified, "gates": {p: v["natural_gates_passed"] for p, v in per.items()}, "ctnc": {p: round(v["checks"]["cross_template_consistency"]["value"]["mean"], 3) for p, v in per.items()}, "ba": {p: round(v["checks"]["primary_format_balanced_accuracy"]["value"], 3) for p, v in per.items()}}, indent=2))
    return rec


# ---------------------------------------------------------------- directions
def fit_population(gpu_root: Path, summaries: list[dict[str, Any]], worlds: dict[str, dict[str, Any]], layer: str = "primary") -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        acts = load_world_acts(gpu_root, s["world_id"], layer)
        w = worlds[s["world_id"]]
        for fmt in ("F2", "F1"):
            for real in (C.REALIZATIONS if fmt == "F2" else ("A",)):
                for state in ("base", "counterfactual"):
                    for q in FIT_QUERIES:
                        spec = w["queries"][q]
                        rows.append({"world_id": s["world_id"], "world_index": w["world_index"], "format": fmt, "realization": real, "state": state, "query": q, "stance": spec["queried_stance"][state], "label": spec[f"label_{state}"], "x": acts[act_key(fmt, real, state, q)].astype(np.float64)})
    return rows


def dim(rows: list[dict[str, Any]], key: str, pos: str) -> np.ndarray:
    X = np.stack([r["x"] for r in rows])
    y = np.array([r[key] == pos for r in rows])
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def threshold(rows: list[dict[str, Any]], d: np.ndarray, key: str, pos: str) -> float:
    X = np.stack([r["x"] for r in rows]) @ d
    y = np.array([r[key] == pos for r in rows])
    return float(0.5 * (X[y].mean() + X[~y].mean()))


def qualify_direction(rows: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    f2 = [r for r in rows if r["format"] == "F2"]
    f1 = {(r["world_id"], r["state"], r["query"]): r for r in rows if r["format"] == "F1"}
    pred, truth, agree, xr_pred, xr_truth = [], [], [], [], []
    for k in range(folds):
        train = [r for r in f2 if r["world_index"] % folds != k]
        test = [r for r in f2 if r["world_index"] % folds == k]
        d = dim(train, "stance", "SUPPORT")
        thr = threshold(train, d, "stance", "SUPPORT")
        for r in test:
            p = int(r["x"] @ d > thr)
            pred.append(p)
            truth.append(int(r["stance"] == "SUPPORT"))
            if r["realization"] == "A":
                m = f1[(r["world_id"], r["state"], r["query"])]
                agree.append(int(int(m["x"] @ d > thr) == p))
        for src, dst in (("A", "B"), ("B", "A")):
            tr = [r for r in train if r["realization"] == src]
            d2 = dim(tr, "stance", "SUPPORT")
            thr2 = threshold(tr, d2, "stance", "SUPPORT")
            for r in test:
                if r["realization"] == dst:
                    xr_pred.append(int(r["x"] @ d2 > thr2))
                    xr_truth.append(int(r["stance"] == "SUPPORT"))
    g = C.DIRECTION_GATES
    ba, ag, xba = C.balanced_accuracy(pred, truth), float(np.mean(agree)), C.balanced_accuracy(xr_pred, xr_truth)
    checks = {"cv_relation_sign_balanced_accuracy": {"value": ba, "min": g["cv_relation_sign_balanced_accuracy_min"], "pass": bool(ba >= g["cv_relation_sign_balanced_accuracy_min"]), "n": len(pred)}, "mapping_agreement": {"value": ag, "min": g["mapping_agreement_min"], "pass": bool(ag >= g["mapping_agreement_min"]), "n": len(agree)}, "cross_realization_balanced_accuracy": {"value": xba, "min": g["cross_realization_balanced_accuracy_min"], "pass": bool(xba >= g["cross_realization_balanced_accuracy_min"]), "n": len(xr_pred)}}
    return {"checks": checks, "qualified": bool(all(v["pass"] for v in checks.values())), "folds": folds}


def fit_directions(actor: str, gpu_root: Path, freeze_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [s for s in C.read_jsonl(gpu_root / "world_summaries.jsonl") if s["split"] == "fit"]
    if len(summaries) != C.SPLIT_SIZES["fit"]:
        raise RuntimeError(f"expected {C.SPLIT_SIZES['fit']} fit worlds, found {len(summaries)}")
    protocol = {s["protocol"] for s in summaries}
    if len(protocol) != 1:
        raise RuntimeError("mixed protocols in fit outputs")
    worlds = {w["world_id"]: w for w in C.read_jsonl(freeze_root / "CORPUS.jsonl")}
    rows = fit_population(gpu_root, summaries, worlds)
    f2 = [r for r in rows if r["format"] == "F2"]
    balance = {"stance": {k: sum(r["stance"] == k for r in f2) for k in ("SUPPORT", "OPPOSE")}, "label": {k: sum(r["label"] == k for r in f2) for k in ("ENTAILED", "NOT_ENTAILED")}, "stance_x_label": {f"{a}|{b}": sum(r["stance"] == a and r["label"] == b for r in f2) for a in ("SUPPORT", "OPPOSE") for b in ("ENTAILED", "NOT_ENTAILED")}, "query": {q: sum(r["query"] == q for r in f2) for q in FIT_QUERIES}, "realization": {x: sum(r["realization"] == x for r in f2) for x in C.REALIZATIONS}, "worlds": len(summaries)}
    relation = dim(f2, "stance", "SUPPORT")
    answer = dim(f2, "label", "ENTAILED")
    a_hat = C.unit(answer)
    residual = relation - float(relation @ a_hat) * a_hat
    qual = qualify_direction(rows, C.DIRECTION_GATES["folds"])
    track_a = C.load_track_a_bundle(actor)
    dirs = {"ORIGINAL_FROZEN_DIM": C.unit(track_a["raw_relation_dim"]), "REGIME_MATCHED_DIM": C.unit(relation), "REGIME_ANSWER_RESIDUAL_DIM": C.unit(residual), "REGIME_ANSWER_TOKEN_DIM": a_hat, "MATCHED_SENTIMENT_DIRECTION": C.unit(track_a["sentiment"]), "MATCHED_ANSWER_TOKEN_DIRECTION": C.unit(track_a["answer_token"]), "MATCHED_TRUTH_DIRECTION": C.unit(track_a["factual"])}
    arrays = {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in dirs.items()}
    npz = output_dir / f"DIRECTIONS_{actor}.npz"
    np.savez(npz, **arrays)
    sha = {k: C.sha256_bytes(v.tobytes()) for k, v in arrays.items()}
    cos = {f"{a}|{b}": float(dirs[a] @ dirs[b]) for a in dirs for b in dirs if a < b}
    meta = {"schema_version": f"{C.SCHEMA_PREFIX}_DIRECTIONS", "actor": actor, "protocol": next(iter(protocol)), "file": npz.name, "file_sha256": C.sha256_file(npz), "sha256": sha, "fit_population": balance, "relation_dim_raw_l2": float(np.linalg.norm(relation)), "answer_dim_raw_l2": float(np.linalg.norm(answer)), "relation_answer_cosine": float(C.unit(relation) @ a_hat), "cosines": cos, "qualification": qual, "status": "DIRECTION_QUALIFIED" if qual["qualified"] else "DIRECTION_NOT_QUALIFIED", "gpu_root": str(gpu_root), "gpu_manifest_sha256": C.sha256_file(gpu_root / "artifact_manifest.json"), "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rule": "fit only on the 64 fit worlds' unsteered F2 prompts (Q1-Q4, base and counterfactual, both realizations) under the selected protocol; never on final worlds; frozen before any final-world forward"}
    C.write_create_only(output_dir / f"DIRECTIONS_{actor}.json", C.canonical_json(meta))
    C.write_create_only(output_dir / f"DIRECTION_FIT_{actor}.json", C.canonical_json(meta))
    print(json.dumps({"actor": actor, "status": meta["status"], "checks": {k: round(v["value"], 3) for k, v in qual["checks"].items()}, "relation_answer_cosine": round(meta["relation_answer_cosine"], 3), "cos_regime_original": round(cos.get("ORIGINAL_FROZEN_DIM|REGIME_MATCHED_DIM", float("nan")), 3)}, indent=2))
    return meta


# ---------------------------------------------------------------- witness
def project_out(X: np.ndarray, dirs: list[np.ndarray]) -> np.ndarray:
    Q, _ = np.linalg.qr(np.stack(dirs, axis=1))
    return X - (X @ Q) @ Q.T


def fit_witness(actor: str, gpu_root: Path, freeze_root: Path, directions: Path, output_dir: Path) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [s for s in C.read_jsonl(gpu_root / "world_summaries.jsonl") if s["split"] == "fit"]
    worlds = {w["world_id"]: w for w in C.read_jsonl(freeze_root / "CORPUS.jsonl")}
    meta = C.read_json(directions.with_suffix(".json"))
    d = C.load_direction_file(directions, meta["sha256"])
    proj = [d["REGIME_MATCHED_DIM"], d["ORIGINAL_FROZEN_DIM"]]
    rows = [r for r in fit_population(gpu_root, summaries, worlds, layer="witness") if r["format"] == "F2" and r["realization"] == "A"]
    X = project_out(np.stack([r["x"] for r in rows]), proj)
    y = np.array([1 if r["stance"] == "SUPPORT" else 0 for r in rows])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C.WITNESS_RIDGE_C, max_iter=5000)
    clf.fit((X - mu) / sd, y)
    train_ba = C.balanced_accuracy(list(clf.predict((X - mu) / sd)), list(y))
    Q, _ = np.linalg.qr(np.stack(proj, axis=1))
    wpath = output_dir / f"WITNESS_{actor}.npz"
    np.savez(wpath, mu=mu.astype(np.float32), sd=sd.astype(np.float32), coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]), projector=Q.astype(np.float32))
    rec = {"schema_version": f"{C.SCHEMA_PREFIX}_WITNESS", "actor": actor, "witness_layer": C.WITNESS_LAYER[actor], "n_train": int(len(rows)), "worlds": len(summaries), "train_balanced_accuracy": train_ba, "projected_out": ["REGIME_MATCHED_DIM", "ORIGINAL_FROZEN_DIM"], "directions_sha256": meta["file_sha256"], "weights_path": wpath.name, "weights_sha256": C.sha256_file(wpath), "ridge_C": C.WITNESS_RIDGE_C, "rule": "fit only on the 64 fit worlds' unsteered realization-A F2 prompts (Q1-Q4, base and counterfactual) at the witness layer with both relation directions projected out; gates evaluated on final unsteered natural states", "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    C.write_create_only(output_dir / f"WITNESS_{actor}.json", C.canonical_json(rec))
    print(json.dumps({"actor": actor, "train_ba": round(train_ba, 3), "n": rec["n_train"]}, indent=2))
    return rec


def witness_logit(w: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Q = w["projector"].astype(np.float64)
    Xp = X - (X @ Q) @ Q.T
    return ((Xp - w["mu"]) / w["sd"]) @ w["coef"].astype(np.float64) + float(w["intercept"])


def load_witness(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {"mu": z["mu"].astype(np.float64), "sd": z["sd"].astype(np.float64), "coef": z["coef"], "intercept": float(z["intercept"]), "projector": z["projector"]}


# ---------------------------------------------------------------- calibration
def reachability(summaries: list[dict[str, Any]], fmt: str = "F2") -> dict[str, Any]:
    out: dict[str, Any] = {}
    arms = sorted({a for s in summaries for a in s["arms"].get(fmt, {})})
    for arm in arms:
        recs = [s["arms"][fmt][arm] for s in summaries if arm in s["arms"].get(fmt, {})]
        if arm == C.LOGIT_CONTROL:
            continue
        per_tau = {}
        taus = sorted({t for r in recs for t in r.get("roots", {})}, key=float)
        for t in taus:
            roots = [r["roots"][t] for r in recs]
            reached = [x for x in roots if x["status"] == "REACHED"]
            mult = np.array([x["multiple"] for x in reached]) if reached else np.array([])
            per_tau[t] = {"n": len(roots), "fraction_reached": float(len(reached) / len(roots)), "fraction_saturated": float(np.mean([bool(x.get("saturated")) for x in reached])) if reached else None, "multiple_median": float(np.median(mult)) if mult.size else None, "multiple_p90": float(np.percentile(mult, 90)) if mult.size else None, "multiple_max": float(mult.max()) if mult.size else None, "statuses": {k: sum(x["status"] == k for x in roots) for k in ("REACHED", "UNREACHABLE_IN_GRID", "NUMERICALLY_INVALID")}, "fraction_sign_agrees_with_natural": float(np.mean([bool(x.get("sign_agrees_with_natural")) for x in reached])) if reached and "sign" in reached[0] else None, "fraction_positive_ray": float(np.mean([x.get("sign", 0) > 0 for x in reached])) if reached and "sign" in reached[0] else None}
        out[arm] = {"per_tau": per_tau}
    return out


def calibrate(actor: str, gpu_root: Path, freeze_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = sorted([s for s in C.read_jsonl(gpu_root / "world_summaries.jsonl") if s["split"] == "fit"], key=lambda s: s["world_id"])
    if len(summaries) != C.SPLIT_SIZES["fit"]:
        raise RuntimeError("fit calibrate outputs incomplete")
    sA = [natural_signature(s, "F2", "A") for s in summaries]
    NA = np.stack([s["N"] for s in sA])
    scales = C.robust_scales(NA)
    idx = C.bootstrap_indices(len(summaries))
    # development full-state patch CSR (reference-gate diagnostic on fit worlds)
    I_patch = []
    for s, sig in zip(summaries, sA):
        got = {r["query"]: r for r in s["rows"] if r["format"] == "F2" and r["realization"] == "A" and r["intervention"] == C.FULL_PATCH}
        I_patch.append([got[q]["semantic_margin"] - sig["base"][q] for q in C.CONSEQUENCE_QUERIES])
    patch = C.bootstrap_mean(C.csr(np.array(I_patch), NA, scales)["csr"], idx)
    gates = natural_gates(summaries, scales, idx, patch)
    reach = reachability(summaries, "F2")
    rep = {"schema_version": f"{C.SCHEMA_PREFIX}_DEV_CALIBRATION", "actor": actor, "protocol": summaries[0]["protocol"], "worlds": len(summaries), "scales_Q2_to_Q6": scales.tolist(), "raw_mad_scales": (np.median(np.abs(NA - np.median(NA, axis=0)), axis=0) * 1.4826).tolist(), "development_reference_gates": gates, "development_patch_csr": patch, "reachability": reach, "natural_full_state_l2_q1": {"median": float(np.median([s["natural"]["F2"]["natural_full_state_l2_q1_A"] for s in summaries])), "min": float(np.min([s["natural"]["F2"]["natural_full_state_l2_q1_A"] for s in summaries])), "max": float(np.max([s["natural"]["F2"]["natural_full_state_l2_q1_A"] for s in summaries]))}, "alpha_grid": list(C.ALPHA_GRID), "interpolation_grid": list(C.INTERPOLATION_GRID), "grid_rule": "the frozen grid and root rule are applied unchanged to the final worlds; no extension after outcomes", "note": "fit-world numbers are calibration diagnostics, not the result", "gpu_root": str(gpu_root), "gpu_manifest_sha256": C.sha256_file(gpu_root / "artifact_manifest.json")}
    C.write_create_only(output_dir / f"DEV_CALIBRATION_{actor}.json", C.canonical_json(rep))
    print(json.dumps({"actor": actor, "scales": [round(x, 2) for x in scales.tolist()], "dev_gates_all_pass": gates["all_pass"], "patch_csr": round(patch["mean"], 3), "reach": {a: {t: (round(v["fraction_reached"], 2), v["multiple_median"] and round(v["multiple_median"], 2), v["multiple_p90"] and round(v["multiple_p90"], 2)) for t, v in r["per_tau"].items()} for a, r in reach.items()}}, indent=2))
    return rep


def final_freeze(dev_root: Path, output_path: Path, actors: list[str]) -> dict[str, Any]:
    fz: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_FINAL_ANALYSIS_FREEZE", "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(), "actors": {}, "rule": "everything below was fixed from selection/fit worlds before any final-world forward; final analysis applies it unchanged"}
    for actor in actors:
        sel = C.read_json(dev_root / f"PROTOCOL_SELECTION_{actor}.json")
        entry: dict[str, Any] = {"protocol": sel["selected_protocol"], "assay_qualified": sel["assay_qualified"], "protocol_selection_sha256": C.sha256_file(dev_root / f"PROTOCOL_SELECTION_{actor}.json")}
        if sel["assay_qualified"]:
            dmeta = C.read_json(dev_root / f"DIRECTIONS_{actor}.json")
            wmeta = C.read_json(dev_root / f"WITNESS_{actor}.json")
            cal = C.read_json(dev_root / f"DEV_CALIBRATION_{actor}.json")
            entry |= {"scales_Q2_to_Q6": cal["scales_Q2_to_Q6"], "directions_file": dmeta["file"], "directions_file_sha256": dmeta["file_sha256"], "direction_sha256": dmeta["sha256"], "direction_status": dmeta["status"], "direction_qualification": dmeta["qualification"], "witness_weights": wmeta["weights_path"], "witness_weights_sha256": wmeta["weights_sha256"], "witness_layer": wmeta["witness_layer"], "dev_calibration_sha256": C.sha256_file(dev_root / f"DEV_CALIBRATION_{actor}.json"), "development_reachability": cal["reachability"]}
        fz["actors"][actor] = entry
    fz["thresholds"] = {"reference": C.REFERENCE_GATES, "direction": C.DIRECTION_GATES, "witness": C.WITNESS_GATES, "dose_validity": C.DOSE_VALIDITY, "truth_coverage_min": C.COVERAGE_MIN_FOR_TRUTH, "comparator_coverage_min": C.COMPARATOR_COVERAGE_MIN, "bootstrap": {"replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED}, "holm_alpha": C.HOLM_ALPHA, "taus": list(C.TAUS), "primary_tau": C.PRIMARY_TAU, "alpha_grid": list(C.ALPHA_GRID), "interpolation_grid": list(C.INTERPOLATION_GRID)}
    C.write_create_only(output_path, C.canonical_json(fz))
    print(json.dumps({a: {k: v for k, v in e.items() if k in ("protocol", "assay_qualified", "direction_status", "scales_Q2_to_Q6")} for a, e in fz["actors"].items()}, indent=2))
    return fz


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select-protocol", "fit-directions", "fit-witness", "calibrate", "final-freeze"])
    ap.add_argument("--actor", choices=sorted(C.ACTORS))
    ap.add_argument("--natural-roots", nargs="*", help="select-protocol: P1=<root> P2=<root> ...")
    ap.add_argument("--gpu-root", type=Path)
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    ap.add_argument("--directions", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--output-path", type=Path)
    ap.add_argument("--dev-root", type=Path)
    ap.add_argument("--actors", default="llama,gemma")
    args = ap.parse_args(argv)
    if args.mode == "select-protocol":
        roots = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in args.natural_roots}
        select_protocol(args.actor, roots, args.output_dir)
    elif args.mode == "fit-directions":
        fit_directions(args.actor, args.gpu_root, args.freeze_root, args.output_dir)
    elif args.mode == "fit-witness":
        fit_witness(args.actor, args.gpu_root, args.freeze_root, args.directions, args.output_dir)
    elif args.mode == "calibrate":
        calibrate(args.actor, args.gpu_root, args.freeze_root, args.output_dir)
    else:
        final_freeze(args.dev_root, args.output_path, args.actors.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
