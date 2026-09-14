#!/usr/bin/env python3
"""Arm C of RELATION_READOUT_REPORT_DISCRIMINATION_V1: naturalistic ValuePrism Q-REPORT V2.

Per actor, on the promoted GPU extraction of the open 125-board / 500-cell
population (batch size 1, band activations):

  1. extraction verification against the frozen batch-size-one baseline and the
     Unit 2 cell table (prompt-id hash, margin reproduction, DIM reproduction);
     failure -> NATURALISTIC_EXTRACTION_INVALID and nothing else is fitted;
  2. frozen board-grouped five-fold nested CV; label probe and report probe at
     the primary layer (activation only), and the conditional-information ladders
     L0->L1 (label beyond report) and R0->R1 (report beyond label) with the pinned
     G4 comparator text features; token-direction-removed and consensus-clear
     robustness variants;
  3. probe relationship on out-of-fold predictions; layer-band trajectories.

  analyze --actor A  -> <out>/<actor>/Q_REPORT_V2_<actor>.json (+ OOF csv, probe npz, receipt)
  assemble           -> Q_REPORT_V2_RESULTS.{md,json,csv}, LAYERWISE_LABEL_REPORT_TRAJECTORY.csv,
                        Q_REPORT_V2_INTERPRETATION.md, external_review_packet_v1.{md,json},
                        layer-band activation manifest, artifact_manifest.json
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

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
INNER_FOLDS = 4
PCA_COMPONENTS = 32
MAX_ITER = 5000
MARGIN_TOL = 0.002
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DOMAINS = ("Value", "Right", "Duty")
POSITIONS = ("s1_A", "s1_B", "s2_A", "s2_B")
EPS = 1e-6


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def ba(y: np.ndarray, p: np.ndarray) -> float:
    r = [float(np.mean(p[y == c] == c)) for c in np.unique(y)]
    return float(np.mean(r)) if r else float("nan")


# ---------------------------------------------------------------- features
def text_features(cells: list[dict[str, Any]], cache_dir: Path) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER_ID, revision=ENCODER_REVISION, device="cpu", cache_folder=str(cache_dir))
    sits = sorted({c["situation"] for c in cells})
    cons = sorted({c["consideration"] for c in cells})
    emb = lambda texts: np.asarray(model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)  # noqa: E731
    s_emb, c_emb = dict(zip(sits, emb(sits))), dict(zip(cons, emb(cons)))
    text = np.stack([np.concatenate([s_emb[c["situation"]], c_emb[c["consideration"]]]) for c in cells])
    base = [np.log1p([len(c["situation"]) for c in cells]), np.log1p([len(c["consideration"]) for c in cells])] + [[1.0 if c["consideration_type"] == d else 0.0 for c in cells] for d in DOMAINS]
    mapping = [[1.0 if c["mapping_name"].endswith("reversed") else 0.0 for c in cells]]
    position = [[1.0 if c["cell_position"] == p else 0.0 for c in cells] for p in POSITIONS]
    cov = np.column_stack(base + position + mapping)
    cov_nopos = np.column_stack(base + mapping)
    return {"text": text, "covariates": cov, "covariates_nopos": cov_nopos}


class NestedProbe:
    """Nested grouped CV logistic model with training-fold-only preprocessing."""

    def __init__(self, boards: np.ndarray, folds: dict[str, int]):
        self.boards = boards
        self.fold = np.array([folds[b] for b in boards])

    def _blocks(self, feats: dict[str, np.ndarray], train: np.ndarray, test: np.ndarray, use: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        tr, te = [], []
        for name in use:
            a = feats[name]
            if name == "text":
                pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full").fit(a[train])
                a_tr, a_te = pca.transform(a[train]), pca.transform(a[test])
            else:
                a_tr, a_te = a[train], a[test]
            sc = StandardScaler().fit(a_tr)
            tr.append(sc.transform(a_tr))
            te.append(sc.transform(a_te))
        return np.column_stack(tr), np.column_stack(te)

    def _fit(self, x: np.ndarray, y: np.ndarray, c: float):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=c, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=MAX_ITER).fit(x, y)

    def _inner_select(self, feats: dict[str, np.ndarray], y: np.ndarray, train: np.ndarray, use: tuple[str, ...]) -> float:
        tb = sorted(set(self.boards[train]))
        inner = {b: i % INNER_FOLDS for i, b in enumerate(tb)}
        scores = {c: [] for c in C_GRID}
        for k in range(INNER_FOLDS):
            itr = train[np.array([inner[b] != k for b in self.boards[train]])]
            ite = train[np.array([inner[b] == k for b in self.boards[train]])]
            if len(np.unique(y[itr])) < 2 or len(ite) == 0:
                continue
            xtr, xte = self._blocks(feats, itr, ite, use)
            for c in C_GRID:
                m = self._fit(xtr, y[itr], c)
                scores[c].append(float(np.mean(logloss(y[ite], m.predict_proba(xte)[:, 1]))))
        mean = {c: float(np.mean(v)) if v else float("inf") for c, v in scores.items()}
        best = min(mean.values())
        return min(c for c in C_GRID if mean[c] <= best + 1e-12)

    def run(self, feats: dict[str, np.ndarray], y: np.ndarray, use: tuple[str, ...], project_out: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        n = len(y)
        prob, chosen, coefs = np.full(n, np.nan), {}, {}
        for k in sorted(set(self.fold)):
            train, test = np.where(self.fold != k)[0], np.where(self.fold == k)[0]
            c = self._inner_select(feats, y, train, use)
            xtr, xte = self._blocks(feats, train, test, use)
            if project_out is not None and "activation" in use:
                # remove the training-fold direction supplied for this fold from the (standardized) activation block
                d = project_out[str(k)]
                off = xtr.shape[1] - d.shape[0]
                u = d / (np.linalg.norm(d) + 1e-12)
                xtr[:, off:] -= np.outer(xtr[:, off:] @ u, u)
                xte[:, off:] -= np.outer(xte[:, off:] @ u, u)
            m = self._fit(xtr, y[train], c)
            prob[test] = m.predict_proba(xte)[:, 1]
            chosen[str(k)] = c
            coefs[str(k)] = m.coef_.reshape(-1)
        return {"prob": prob, "chosen_C": chosen, "coef": coefs, "ll": logloss(y, prob)}


# ---------------------------------------------------------------- analysis
def verify_extraction(actor: str, scores: list[dict[str, Any]], proj_raw: np.ndarray, midpoint: float) -> dict[str, Any]:
    hash_ok = sum(1 for r in scores if r["prompt_ids_match_frozen_bs1"])
    new = np.array([r["semantic_margin"] for r in scores])
    ref = np.array([r["reference"]["bs1_semantic_margin"] for r in scores])
    u2 = np.array([r["reference"]["unit2_native_answer_margin"] for r in scores])
    diff = np.abs(new - ref)
    within = float(np.mean(diff <= MARGIN_TOL))
    nz = np.abs(ref) > MARGIN_TOL
    hard = float(np.mean((new[nz] > 0) == (ref[nz] > 0))) if nz.any() else float("nan")
    dim_new = proj_raw - midpoint
    dim_ref = np.array([r["reference"]["unit2_difference_in_means"] for r in scores])
    dim_sign = float(np.mean((dim_new > 0) == (dim_ref > 0)))
    dim_r = float(np.corrcoef(dim_new, dim_ref)[0, 1])
    rec = {"rows": len(scores), "prompt_hash_matches": hash_ok, "margin_vs_bs1": {"tolerance": MARGIN_TOL, "fraction_within_tolerance": within, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "hard_verdict_agreement_nonzero": hard, "nonzero_rows": int(nz.sum())}, "margin_vs_unit2_table": {"origin": scores[0]["reference"]["unit2_margin_origin"], "sign_agreement": float(np.mean((new > 0) == (u2 > 0))), "max_abs_diff": float(np.max(np.abs(new - u2))), "mean_abs_diff": float(np.mean(np.abs(new - u2)))}, "dim_vs_unit2_table": {"sign_agreement": dim_sign, "pearson_r": dim_r, "max_abs_diff": float(np.max(np.abs(dim_new - dim_ref))), "note": "Unit 2 Llama DIM values came from float16 activations"}}
    rec["gates"] = {"prompt_identity": hash_ok == len(scores), "margin_reproduction": within >= 0.99 and (np.isnan(hard) or hard == 1.0), "dim_reproduction": dim_sign >= 0.99 and dim_r >= 0.99}
    rec["status"] = "VALID" if all(rec["gates"].values()) else "NATURALISTIC_EXTRACTION_INVALID"
    return rec


def analyze(actor: str, gpu_root: Path, freeze_root: Path, out_dir: Path, replicates: int, cache_dir: Path) -> dict[str, Any]:
    man = C.read_json(gpu_root / "artifact_manifest.json")
    bad = [f["path"] for f in man["files"] if C.sha256_file(gpu_root / f["path"]) != f["sha256"]]
    if bad:
        raise RuntimeError(f"GPU output hash mismatch: {bad[:5]}")
    scores = C.read_jsonl(gpu_root / "naturalistic_scores.jsonl")
    cells = {c["row_id"]: c for c in C.read_jsonl(freeze_root / "NATURALISTIC_CELL_INPUTS.jsonl")}
    folds = C.read_json(freeze_root / "NATURALISTIC_FOLD_ASSIGNMENTS.json")["assignment"]
    inputs = C.read_json(freeze_root / "INPUT_MANIFEST.json")
    midpoint = float(inputs["source_calibration"]["raw_relation_dim"][actor]["difference_in_means_midpoint"])
    cfg = C.ACTORS[actor]
    with np.load(gpu_root / "naturalistic_activations.npz", allow_pickle=False) as z:
        ids = [str(v) for v in z["prompt_ids"]]
        acts = {L: np.asarray(z[f"layer_{L}"], dtype=np.float64) for L in cfg["band"]}
    if ids != [r["row_id"] for r in scores]:
        raise RuntimeError("activation rows do not align with scores")
    bundle = np.load(C.TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz", allow_pickle=False)
    raw_dir = np.asarray(bundle["raw_relation_dim"], dtype=np.float64)
    token_dir = np.asarray(bundle["answer_token"], dtype=np.float64)
    x = acts[cfg["primary_layer"]]
    proj_raw = x @ raw_dir
    res: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_Q_REPORT_V2", "actor": actor, "label": "PROSPECTIVE_FOLLOW_UP_EVIDENCE (extraction) + frozen probe analysis", "primary_layer": cfg["primary_layer"], "band": cfg["band"], "source_midpoint": midpoint}
    ver = verify_extraction(actor, scores, proj_raw, midpoint)
    res["extraction_verification"] = ver
    C.write_create_only(out_dir / "EXTRACTION_VERIFICATION_RECEIPT.json", C.canonical_json({"actor": actor, **ver}))
    if ver["status"] != "VALID":
        res["classification"] = "NATURALISTIC_EXTRACTION_INVALID"
        return res
    # ---- targets and features
    rows = [cells[r["row_id"]] for r in scores]
    y_label = np.array([r["label"] for r in scores])
    margin = np.array([r["semantic_margin"] for r in scores])
    y_report = (margin > 0).astype(int)
    boards = np.array([r["board_id"] for r in scores])
    clear = np.array([r["consensus_clear"] for r in scores], dtype=bool)
    tf = text_features(rows, cache_dir)
    native = np.column_stack([margin, y_report.astype(float)])
    x_tok = x - np.outer(x @ token_dir / (token_dir @ token_dir), token_dir)
    feats_base = {"text": tf["text"], "covariates": tf["covariates"], "covariates_nopos": tf["covariates_nopos"], "native": native, "label": y_label[:, None].astype(float), "activation": x, "activation_tok": x_tok}
    probe = NestedProbe(boards, folds)
    boot = WorldBootstrap(sorted(set(boards)), replicates, C.BOOTSTRAP_SEED)

    def board_mean(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
        d: dict[str, list[float]] = collections.defaultdict(list)
        for b, v, keep in zip(boards, values, (mask if mask is not None else np.ones(len(values), bool))):
            if keep:
                d[b].append(float(v))
        return {b: float(np.mean(v)) for b, v in d.items()}

    def ladder(y: np.ndarray, base_use: tuple[str, ...], act_key: str, tag: str) -> dict[str, Any]:
        f = dict(feats_base)
        f["activation"] = feats_base[act_key]
        m0 = probe.run(f, y, base_use)
        m1 = probe.run(f, y, base_use + ("activation",))
        imp = m0["ll"] - m1["ll"]
        out = {"baseline": {"chosen_C": m0["chosen_C"], "logloss": float(m0["ll"].mean()), "auroc": auroc(y, m0["prob"]), "balanced_accuracy": ba(y, (m0["prob"] > 0.5).astype(int))}, "augmented": {"chosen_C": m1["chosen_C"], "logloss": float(m1["ll"].mean()), "auroc": auroc(y, m1["prob"]), "balanced_accuracy": ba(y, (m1["prob"] > 0.5).astype(int))}, "logloss_improvement": boot.summary(board_mean(imp)), "consensus_clear_subset": boot.summary(board_mean(imp, clear)), "auroc_gain": auroc(y, m1["prob"]) - auroc(y, m0["prob"]), "chance_logloss": float(logloss(y, np.full(len(y), y.mean())).mean())}
        out["_prob0"], out["_prob1"] = m0["prob"], m1["prob"]
        return out

    ladders = {}
    for tag, act_key in (("primary", "activation"), ("token_removed", "activation_tok")):
        ladders[tag] = {"L": ladder(y_label, ("text", "covariates", "native"), act_key, tag), "R": ladder(y_report, ("text", "covariates", "label"), act_key, tag)}
    # Post hoc exploratory variant (NOT frozen): the frozen G4-comparator covariates include the
    # checkerboard cell position, which determines the reviewed label by construction (s1_A/s2_B
    # Supports, s1_B/s2_A Opposes), so the frozen L0 is degenerate. This variant drops the position
    # block; it is reported as POST_HOC_EXPLORATORY_NOT_FROZEN and never replaces the frozen rule.
    exploratory = {}
    for tag, act_key in (("primary", "activation"), ("token_removed", "activation_tok")):
        exploratory[tag] = {"L": ladder(y_label, ("text", "covariates_nopos", "native"), act_key, tag), "R": ladder(y_report, ("text", "covariates_nopos", "label"), act_key, tag)}
    position_determines_label = all(len({(r["cell_position"], r["label"]) for r in scores if r["cell_position"] == pos}) == 1 for pos in POSITIONS)
    frozen_l0_degenerate = ladders["primary"]["L"]["baseline"]["auroc"] >= 0.999
    # ---- activation-only probes and their relationship
    lab = probe.run(feats_base, y_label, ("activation",))
    rep = probe.run(feats_base, y_report, ("activation",))
    cos = [float(lab["coef"][k] @ rep["coef"][k] / (np.linalg.norm(lab["coef"][k]) * np.linalg.norm(rep["coef"][k]) + 1e-12)) for k in lab["coef"]]
    dis = y_label != y_report
    lab_after_rep = probe.run(feats_base, y_label, ("activation",), project_out=rep["coef"])
    rep_after_lab = probe.run(feats_base, y_report, ("activation",), project_out=lab["coef"])
    mapping_rev = np.array([r["mapping_name"].endswith("reversed") for r in scores])
    l_imp_rows = logloss(y_label, ladders["primary"]["L"]["_prob0"]) - logloss(y_label, ladders["primary"]["L"]["_prob1"])
    band_abs = np.abs(margin)
    bands = {"lt_0.5": band_abs < 0.5, "0.5_to_2": (band_abs >= 0.5) & (band_abs < 2), "ge_2": band_abs >= 2}
    rel = {
        "label_probe": {"auroc": auroc(y_label, lab["prob"]), "balanced_accuracy": ba(y_label, (lab["prob"] > 0.5).astype(int)), "chosen_C": lab["chosen_C"]},
        "report_probe": {"auroc": auroc(y_report, rep["prob"]), "balanced_accuracy": ba(y_report, (rep["prob"] > 0.5).astype(int)), "chosen_C": rep["chosen_C"]},
        "direction_cosine_per_fold": cos, "direction_cosine_mean": float(np.mean(cos)),
        "disagreement_cells": {"count": int(dis.sum()), "boards": int(len(set(boards[dis]))), "label_probe_sides_with_label": float(np.mean((lab["prob"][dis] > 0.5) == y_label[dis])) if dis.any() else float("nan"), "report_probe_sides_with_report": float(np.mean((rep["prob"][dis] > 0.5) == y_report[dis])) if dis.any() else float("nan"), "label_probe_sides_with_report": float(np.mean((lab["prob"][dis] > 0.5) == y_report[dis])) if dis.any() else float("nan"), "frozen_dim_sides_with_label": float(np.mean(((proj_raw[dis] - midpoint) > 0) == y_label[dis].astype(bool))) if dis.any() else float("nan")},
        "label_after_report_direction_removed": {"auroc": auroc(y_label, lab_after_rep["prob"]), "balanced_accuracy": ba(y_label, (lab_after_rep["prob"] > 0.5).astype(int))},
        "report_after_label_direction_removed": {"auroc": auroc(y_report, rep_after_lab["prob"]), "balanced_accuracy": ba(y_report, (rep_after_lab["prob"] > 0.5).astype(int))},
        "mapping_strata": {m: {"n": int(sel.sum()), "label_probe_auroc": auroc(y_label[sel], lab["prob"][sel]), "report_probe_auroc": auroc(y_report[sel], rep["prob"][sel]), "L_improvement_board_mean": float(np.mean(list(board_mean(l_imp_rows, sel).values())))} for m, sel in (("standard", ~mapping_rev), ("reversed", mapping_rev))},
        "native_margin_bands": {k: {"n": int(sel.sum()), "label_probe_accuracy": float(np.mean((lab["prob"][sel] > 0.5) == y_label[sel])) if sel.any() else float("nan"), "report_probe_accuracy": float(np.mean((rep["prob"][sel] > 0.5) == y_report[sel])) if sel.any() else float("nan"), "report_equals_label": float(np.mean(y_report[sel] == y_label[sel])) if sel.any() else float("nan")} for k, sel in bands.items()},
        "unit2_fixed_comparators": {"dim_classification": "READOUT_IDENTICAL_TO_REPORT (frozen)", "P_dim_sides_with_label_given_disagreement_unit2": {"llama": 0.082, "gemma": 0.049}[actor], "P_logistic_sides_with_label_unit2": {"llama": 0.203, "gemma": 0.225}[actor]},
    }
    # ---- layer band trajectories (no selection)
    traj = []
    for L in cfg["band"]:
        f = dict(feats_base)
        f["activation"] = acts[L]
        lb = probe.run(f, y_label, ("activation",))
        rb = probe.run(f, y_report, ("activation",))
        traj.append({"actor": actor, "layer": L, "primary": L == cfg["primary_layer"], "label_probe_auroc": auroc(y_label, lb["prob"]), "label_probe_logloss": float(lb["ll"].mean()), "report_probe_auroc": auroc(y_report, rb["prob"]), "report_probe_logloss": float(rb["ll"].mean())})
    # ---- classification
    def cls(lad: dict[str, Any]) -> str:
        L, R = lad["L"]["logloss_improvement"], lad["R"]["logloss_improvement"]
        lpos, rpos = L["ci_low"] > 0, R["ci_low"] > 0
        ladv, radv = L["ci_high"] < 0, R["ci_high"] < 0
        if lpos and rpos:
            return "BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY"
        if ladv or radv:
            return "ACTIVATION_INCREMENT_ADVERSE"
        if lpos:
            return "LABEL_INFORMATION_BEYOND_REPORT"
        if rpos:
            return "REPORT_INFORMATION_BEYOND_LABEL"
        return "NO_DETECTED_INCREMENT"

    primary_cls, tok_cls = cls(ladders["primary"]), cls(ladders["token_removed"])
    frozen_rule_classification = primary_cls if primary_cls == tok_cls else "MIXED_OR_UNDERDETERMINED"
    validity = {"position_covariate_determines_label": bool(position_determines_label), "frozen_L0_baseline_auroc": ladders["primary"]["L"]["baseline"]["auroc"], "frozen_L0_degenerate": bool(frozen_l0_degenerate), "verdict": "FROZEN_L_LADDER_INVALID_BY_DESIGN" if (position_determines_label and frozen_l0_degenerate) else "FROZEN_PLAN_VALID", "reason": "the frozen G4-comparator covariates carry the checkerboard cell position, which fixes the reviewed label by construction; the L0 baseline therefore predicts the label perfectly and the L0->L1 increment cannot measure label information beyond report" if (position_determines_label and frozen_l0_degenerate) else None}
    if validity["verdict"] == "FROZEN_L_LADDER_INVALID_BY_DESIGN":
        classification = "MIXED_OR_UNDERDETERMINED"
        classification_reason = "frozen L direction invalid by design (degenerate L0); frozen R direction reported; no Case 5/6 claim licensed from the frozen plan"
    else:
        classification, classification_reason = frozen_rule_classification, "frozen rule"
    expl_primary, expl_tok = cls(exploratory["primary"]), cls(exploratory["token_removed"])
    exploratory_classification = expl_primary if expl_primary == expl_tok else "MIXED_OR_UNDERDETERMINED"
    oof_rows = [{"row_id": r["row_id"], "board_id": r["board_id"], "fold": folds[r["board_id"]], "label": int(y_label[i]), "report_sign": int(y_report[i]), "native_margin": float(margin[i]), "consensus_clear": bool(clear[i]), "mapping": r["mapping_name"], "L0_prob": ladders["primary"]["L"]["_prob0"][i], "L1_prob": ladders["primary"]["L"]["_prob1"][i], "R0_prob": ladders["primary"]["R"]["_prob0"][i], "R1_prob": ladders["primary"]["R"]["_prob1"][i], "label_probe_prob": lab["prob"][i], "report_probe_prob": rep["prob"][i], "frozen_dim_strict_score": float(proj_raw[i] - midpoint)} for i, r in enumerate(scores)]
    C.write_create_only(out_dir / "FROZEN_FOLD_PREDICTIONS.csv", C.csv_bytes(oof_rows, list(oof_rows[0].keys())))
    probe_arrays = {f"label_probe_coef_fold{k}": np.asarray(v, dtype=np.float32) for k, v in lab["coef"].items()} | {f"report_probe_coef_fold{k}": np.asarray(v, dtype=np.float32) for k, v in rep["coef"].items()}
    np.savez(out_dir / "PROBE_DIRECTIONS.npz", **probe_arrays)
    for group in (ladders, exploratory):
        for tag in group:
            for k in ("L", "R"):
                group[tag][k].pop("_prob0"), group[tag][k].pop("_prob1")
    res.update({"population": {"cells": len(scores), "boards": len(set(boards)), "disagreement_cells": int(dis.sum()), "consensus_clear_cells": int(clear.sum())}, "ladders": ladders, "probe_relationship": rel, "layer_trajectory": traj, "classification": classification, "classification_reason": classification_reason, "classification_frozen_rule": frozen_rule_classification, "classification_primary_variant": primary_cls, "classification_token_removed_variant": tok_cls, "frozen_plan_validity": validity, "exploratory_no_position_covariate": {"status": "POST_HOC_EXPLORATORY_NOT_FROZEN", "ladders": exploratory, "rule_classification": exploratory_classification, "variants": {"primary": expl_primary, "token_removed": expl_tok}}, "probe_directions_sha256": C.sha256_file(out_dir / "PROBE_DIRECTIONS.npz"), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    return res


# ---------------------------------------------------------------- assemble
def assemble(output_root: Path, gpu_roots: dict[str, Path]) -> None:
    per = {a: C.read_json(output_root / a / f"Q_REPORT_V2_{a}.json") for a in C.ACTORS if (output_root / a / f"Q_REPORT_V2_{a}.json").is_file()}
    actors = list(per)
    band_manifest = {}
    for a in actors:
        man = C.read_json(gpu_roots[a] / "artifact_manifest.json")
        band_manifest[a] = {"nfs_root": str(gpu_roots[a]), "files": [f for f in man["files"] if f["path"].startswith("naturalistic")], "band": C.ACTORS[a]["band"]}
        dst = output_root / a / "gpu_scores"
        if not dst.exists():
            dst.mkdir(parents=True)
            for name in ("naturalistic_scores.jsonl", "artifact_manifest.json", "RUN_RECEIPT.json", "PROMOTION_RECEIPT.json"):
                shutil.copy2(gpu_roots[a] / name, dst / name)
    C.write_create_only(output_root / "LAYER_BAND_ACTIVATION_MANIFEST.json", C.canonical_json(band_manifest))
    classes = {a: per[a]["classification"] for a in actors}
    md = ["# Naturalistic Q-REPORT V2 results (Arm C)", "", "Population: the open human-reviewed 125-board / 500-cell ValuePrism set of Unit 2 (sealed 60-board set untouched). Unit 2's frozen DIM classification is retained as a fixed comparator.", ""]
    csv_rows, traj_rows = [], []
    for a in actors:
        r = per[a]
        v = r["extraction_verification"]
        md += [f"## {a}: **{r['classification']}**", "", f"Extraction verification: {v['status']} — prompt hash matches {v['prompt_hash_matches']}/{v['rows']}; margin within {MARGIN_TOL} of the batch-size-one baseline on {v['margin_vs_bs1']['fraction_within_tolerance']:.3f} of rows (max abs diff {v['margin_vs_bs1']['max_abs_diff']:.4f}, hard-verdict agreement {v['margin_vs_bs1']['hard_verdict_agreement_nonzero']:.3f}); Unit 2 table margin sign agreement {v['margin_vs_unit2_table']['sign_agreement']:.3f} (origin: {v['margin_vs_unit2_table']['origin']}); frozen DIM vs Unit 2 table sign agreement {v['dim_vs_unit2_table']['sign_agreement']:.3f}, r = {v['dim_vs_unit2_table']['pearson_r']:.4f}.", ""]
        if r["classification"] == "NATURALISTIC_EXTRACTION_INVALID":
            continue
        md += [f"Population: {r['population']}", "", "| variant | ladder | baseline log-loss | augmented log-loss | improvement (board mean) [95% CI] | consensus-clear subset | AUROC gain |", "|---|---|---|---|---|---|---|"]
        for tag, lad in r["ladders"].items():
            for k in ("L", "R"):
                d = lad[k]
                md.append(f"| {tag} | {k}0→{k}1 | {d['baseline']['logloss']:.4f} | {d['augmented']['logloss']:.4f} | {fmt(d['logloss_improvement'])} | {fmt(d['consensus_clear_subset'])} | {d['auroc_gain']:+.4f} |")
                csv_rows.append({"actor": a, "variant": tag, "ladder": k, "baseline_logloss": d["baseline"]["logloss"], "augmented_logloss": d["augmented"]["logloss"], "improvement_point": d["logloss_improvement"]["point"], "improvement_ci_low": d["logloss_improvement"]["ci_low"], "improvement_ci_high": d["logloss_improvement"]["ci_high"], "baseline_auroc": d["baseline"]["auroc"], "augmented_auroc": d["augmented"]["auroc"], "baseline_ba": d["baseline"]["balanced_accuracy"], "augmented_ba": d["augmented"]["balanced_accuracy"]})
        rel = r["probe_relationship"]
        md += ["", f"Frozen-rule output: {r['classification_frozen_rule']} (primary {r['classification_primary_variant']}, token-removed {r['classification_token_removed_variant']}). Frozen-plan validity: **{r['frozen_plan_validity']['verdict']}** — {r['frozen_plan_validity']['reason']}. Reported classification: {r['classification']} ({r['classification_reason']}).", "", "Post hoc exploratory variant (NOT frozen; position covariate removed from L0/R0):", "", "| variant | ladder | baseline log-loss | augmented log-loss | improvement [95% CI] | consensus-clear subset | AUROC |", "|---|---|---|---|---|---|---|"]
        ex = r["exploratory_no_position_covariate"]
        for tag, lad in ex["ladders"].items():
            for k in ("L", "R"):
                d = lad[k]
                md.append(f"| {tag} | {k}0→{k}1 | {d['baseline']['logloss']:.4f} | {d['augmented']['logloss']:.4f} | {fmt(d['logloss_improvement'])} | {fmt(d['consensus_clear_subset'])} | {d['baseline']['auroc']:.3f}→{d['augmented']['auroc']:.3f} |")
                csv_rows.append({"actor": a, "variant": f"exploratory_nopos_{tag}", "ladder": k, "baseline_logloss": d["baseline"]["logloss"], "augmented_logloss": d["augmented"]["logloss"], "improvement_point": d["logloss_improvement"]["point"], "improvement_ci_low": d["logloss_improvement"]["ci_low"], "improvement_ci_high": d["logloss_improvement"]["ci_high"], "baseline_auroc": d["baseline"]["auroc"], "augmented_auroc": d["augmented"]["auroc"], "baseline_ba": d["baseline"]["balanced_accuracy"], "augmented_ba": d["augmented"]["balanced_accuracy"]})
        md += ["", f"Exploratory rule classification (post hoc, not licensed as frozen evidence): {ex['rule_classification']} (primary {ex['variants']['primary']}, token-removed {ex['variants']['token_removed']}).", "", "Probe relationship (activation-only probes, out-of-fold):", "", f"- label probe AUROC {rel['label_probe']['auroc']:.3f} / BA {rel['label_probe']['balanced_accuracy']:.3f}; report probe AUROC {rel['report_probe']['auroc']:.3f} / BA {rel['report_probe']['balanced_accuracy']:.3f}; coefficient cosine {rel['direction_cosine_mean']:.3f}", f"- disagreement cells: {rel['disagreement_cells']}", f"- label after report-direction removal: AUROC {rel['label_after_report_direction_removed']['auroc']:.3f}; report after label-direction removal: AUROC {rel['report_after_label_direction_removed']['auroc']:.3f}", f"- mapping strata: {rel['mapping_strata']}", f"- native-margin bands: {rel['native_margin_bands']}", f"- Unit 2 fixed comparators: {rel['unit2_fixed_comparators']}", ""]
        traj_rows += r["layer_trajectory"]
    C.write_create_only(output_root / "Q_REPORT_V2_RESULTS.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_Q_REPORT_V2_RESULTS", "classification": classes, "actors": per}))
    C.write_create_only(output_root / "Q_REPORT_V2_RESULTS.md", ("\n".join(md) + "\n").encode("utf-8"))
    C.write_create_only(output_root / "Q_REPORT_V2_RESULTS.csv", C.csv_bytes(csv_rows, ["actor", "variant", "ladder", "baseline_logloss", "augmented_logloss", "improvement_point", "improvement_ci_low", "improvement_ci_high", "baseline_auroc", "augmented_auroc", "baseline_ba", "augmented_ba"]))
    C.write_create_only(output_root / "LAYERWISE_LABEL_REPORT_TRAJECTORY.csv", C.csv_bytes(traj_rows, ["actor", "layer", "primary", "label_probe_auroc", "label_probe_logloss", "report_probe_auroc", "report_probe_logloss"]))
    interp = ["# Q-REPORT V2 — interpretation (Arm C)", "", "Unit 2's frozen DIM classification (READOUT_IDENTICAL_TO_REPORT) is not overwritten; this arm asks whether the *full* activation carries reviewed-label information beyond the report (and vice versa) under a frozen nested board-grouped procedure.", ""]
    cases = {"BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY": "Case 5 (and its converse): the broader activation state contains separable label-related and report-related information; the old DIM scalar is report-aligned but not the whole story.", "LABEL_INFORMATION_BEYOND_REPORT": "Case 5: the old DIM scalar is report-aligned, but the broader activation state contains separable label-related information.", "REPORT_INFORMATION_BEYOND_LABEL": "The activation predicts the report beyond text+label while adding no label information beyond the report: a report-dominant account at the tested layer.", "NO_DETECTED_INCREMENT": "Case 6: the open population supports a report-dominant account at the tested layer with power limits retained; no claim of global absence.", "ACTIVATION_INCREMENT_ADVERSE": "An augmented model is reliably worse under the frozen nested procedure; the activation increment is not usable at this population size.", "MIXED_OR_UNDERDETERMINED": "The classification depends on the token-direction control; no single-sentence claim is licensed.", "NATURALISTIC_EXTRACTION_INVALID": "Extraction did not reproduce the frozen cell table; no probe was fitted and no Q-REPORT V2 claim is made."}
    for a in actors:
        interp += [f"## {a}: {per[a]['classification']}", "", cases[per[a]["classification"]], ""]
    interp += ["No layer was selected from the band; the trajectory CSV is descriptive and no layerwise significance claim is made.", ""]
    C.write_create_only(output_root / "Q_REPORT_V2_INTERPRETATION.md", "\n".join(interp).encode("utf-8"))
    packet = {"schema_version": "EXTERNAL_REVIEW_PACKET_V1", "unit": f"{C.STUDY_ID} / Arm C naturalistic Q-REPORT V2", "evidence_status": "prospective extraction; frozen probe analysis", "research_question": {"question": "Do full naturalistic activations contain reviewed-label information beyond the model report (and report information beyond the label)?", "target_variable": "held-out log-loss improvement (board mean) L0->L1 and R0->R1", "expected_positive_result": "board-bootstrap CI low > 0", "expected_falsifying_result": "CI includes zero (no detected increment) or CI high < 0 (adverse)", "frozen_practical_threshold": 0.0}, "capability_delta": {"new_runnable_behavior": "scripts/rrrd_v1_q_report_v2.py: nested board-grouped conditional-information ladders with the pinned G4 text comparator, token-direction control, probe-relationship diagnostics, layer-band trajectories"}, "evidence_topology": {"independent_unit": "board_id", "boards": 125, "cells": 500, "folds": 5, "nested_inner_folds": INNER_FOLDS, "disagreement_cells": {a: per[a].get("population", {}).get("disagreement_cells") for a in actors}, "clustering": "four cells per board; bootstrap resamples boards"}, "exact_empirical_result": {"classification": classes, "frozen_rule_classification": {a: per[a].get("classification_frozen_rule") for a in actors}, "frozen_plan_validity": {a: per[a].get("frozen_plan_validity") for a in actors}, "variants": {a: {"primary": per[a].get("classification_primary_variant"), "token_removed": per[a].get("classification_token_removed_variant")} for a in actors}, "ladders": {a: per[a].get("ladders") for a in actors}, "exploratory_no_position_covariate": {a: per[a].get("exploratory_no_position_covariate") for a in actors}, "probe_relationship": {a: per[a].get("probe_relationship") for a in actors}, "extraction_verification": {a: per[a]["extraction_verification"] for a in actors}}, "negative_and_surprising_evidence": [f"{a}: {k}0->{k}1 {tag} improvement {fmt(per[a]['ladders'][tag][k]['logloss_improvement'])}" for a in actors if "ladders" in per[a] for tag in per[a]["ladders"] for k in ("L", "R") if per[a]["ladders"][tag][k]["logloss_improvement"]["ci_low"] <= 0], "what_was_held_fixed": ["frozen folds", "C grid and inner CV", "PCA(32) text features fit in training folds", "frozen answer_token direction for the control", "10,000 board bootstraps"], "deviations": ["the frozen L0 covariate set (copied from the G4 comparator) contains the checkerboard cell position, which determines the reviewed label by construction; the frozen L ladder is therefore degenerate and its rule output is recorded but not interpreted; a post hoc variant without the position block is reported as exploratory only", "the first naturalistic extraction failed the frozen 0.002-nat margin gate because the Track A candidate-sequence scorer forwards each candidate alone while the frozen baseline forwards both candidates as one batch; a naturalistic-only re-extraction on the exact baseline path (technical recovery, first extraction preserved, no probe fitted on it) reproduced the baseline on 500/500 rows for both actors"], "claim_boundary": "see freeze/CLAIM_BOUNDARY.md; Unit 2's DIM verdict is unchanged", "artifact_pickup": {"results": "Q_REPORT_V2_RESULTS.json", "fold_predictions": {a: f"{a}/FROZEN_FOLD_PREDICTIONS.csv" for a in actors}, "probe_directions": {a: f"{a}/PROBE_DIRECTIONS.npz" for a in actors}, "activations": "LAYER_BAND_ACTIVATION_MANIFEST.json (NFS)"}}
    C.write_create_only(output_root / "external_review_packet_v1.json", C.canonical_json(packet))
    C.write_create_only(output_root / "external_review_packet_v1.md", ("\n".join(["# External review packet — Arm C (Q-REPORT V2)", "", f"Classification: {classes}", "", "## Negative and surprising evidence", ""] + [f"- {n}" for n in packet["negative_and_surprising_evidence"]] + ["", "See Q_REPORT_V2_RESULTS.md and Q_REPORT_V2_INTERPRETATION.md.", ""])).encode("utf-8"))
    C.write_create_only(output_root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_ARM_C_MANIFEST", "classification": classes, "files": C.manifest_for(output_root)}))
    print(C.json.dumps(classes, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    a.add_argument("--gpu-root", type=Path, required=True)
    a.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
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
        res = analyze(args.actor, args.gpu_root, args.freeze_root, out_dir, args.replicates, args.cache_dir)
        res["source"] = {"gpu_manifest_sha256": C.sha256_file(args.gpu_root / "artifact_manifest.json"), "script_sha256": C.sha256_file(Path(__file__).resolve()), "replicates": args.replicates, "encoder": {"id": ENCODER_ID, "revision": ENCODER_REVISION}}
        C.write_create_only(out_dir / f"Q_REPORT_V2_{args.actor}.json", C.canonical_json(res))
        C.write_create_only(out_dir / "artifact_manifest.json", C.canonical_json({"files": C.manifest_for(out_dir)}))
        print(C.json.dumps({"classification": res["classification"], "extraction": res["extraction_verification"]["status"]}, indent=2))
    else:
        assemble(args.output_root, {"llama": args.gpu_root_llama, "gemma": args.gpu_root_gemma})
    return 0


if __name__ == "__main__":
    sys.exit(main())
