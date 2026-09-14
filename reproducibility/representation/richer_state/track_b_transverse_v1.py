#!/usr/bin/env python3
"""Track B activation addendum + full-vector transverse test (Unit 8) — answers G5.

Extends the frozen scalar-v001 Track B analysis (67 endpoint-eligible
originals, frozen outer folds, Ridge alpha 1.0, normalized-RMSE endpoint
``rms_semantic_margin_movement``) with transverse features computed from the
selected-layer activations that the v2 addendum extraction produced for all 112
originals in each actor.

Preregistered feature families (small, fixed; all fold-local — every fit uses
only training-fold originals plus the 45 endpoint-free reference originals):

* readout_family_disagreement: |z(dim projection) − z(native margin)|, sign
  disagreement indicator, and for the own actor |z(dim) − z(logistic margin)|
* orthogonal_component_norm: ‖relation residual‖ and its ratio to ‖activation‖
* local_density: mean distance to the k=5 nearest same-predicted-class training
  references in a rank-10 whitened PCA space, plus Mahalanobis distance to the
  class centroid in that space
* low_rank_subspace_distance: reconstruction residual norm from a rank-5 PCA
  fitted on same-predicted-class training references
* cross_layer_trajectory: NOT AVAILABLE — the addendum stores the selected layer
  only; recorded as a boundary, not imputed.

Ladder (frozen): baseline_text_native → +own_llama_scalar → +transverse(own
actor); cross-actor transverse (Gemma activations) reported as a secondary rung.
Gate G5: normalized-RMSE increment of the transverse rung over +own_llama_scalar
≥ 0.05 with grouped bootstrap CI lower bound > 0, original-level outcome
permutation p ≤ 0.05, and ≥ 4 of 5 outer folds improved — interpretable only if
the augmented rung's normalized RMSE < 1.0; otherwise ENDPOINT_AT_FLOOR.
Null band: 128 matched-norm random directions replace the DIM direction in every
direction-dependent feature; the observed increment is compared with the band.

``prepare`` pins inputs (private frozen Track B bindings, addendum activations,
direction bundles, fold column) and writes the freeze; ``run`` requires the
committed freeze and writes one create-only result package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_endorsement.relation_transverse_fragility.analysis import (  # noqa: E402
    AnalysisResult, FeatureBlock, FoldAssignment, _metrics, _pipeline, fold_improvement_count, paired_increment_bootstrap,
)
from geometry_endorsement.relation_transverse_fragility.prepare import (  # noqa: E402
    BASELINE_NATIVE, BASELINE_STRATUM, BASELINE_TEXT, OWN_MODEL_SCALAR, PRIMARY_ENDPOINT, assemble_scalar_analysis_table,
)

SCHEMA = "TRACK_B_TRANSVERSE_V1"
FREEZE_FILE = "TRACK_B_TRANSVERSE_FREEZE.json"
PREOUTCOME_FILE = "TRACK_B_TRANSVERSE_PREOUTCOME_MANIFEST.json"
TRACK_B_DIR = ROOT / "reports" / "relation_operator_transverse_fragility_v1" / "track_b"
FREEZE_V2 = ROOT / "reports" / "relation_operator_transverse_fragility_v1" / "track_a_v2" / "freeze"
ACTORS = ("llama", "gemma")
ALPHA = 1.0
K_NEIGHBOURS = 5
WHITEN_RANK = 10
SUBSPACE_RANK = 5
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 1_000
RANDOM_DIRECTIONS = 128
DEFAULT_BAND_WORKERS = max(1, min(16, (os.cpu_count() or 1) - 2))
SEED = 152
GATE = {"normalized_rmse_increment_minimum": 0.05, "grouped_bootstrap_ci_lower_strictly_greater_than": 0.0, "grouped_permutation_p_maximum": 0.05, "outer_folds_improved_minimum": 4, "floor_augmented_nrmse_max": 1.0}
TRANSVERSE_COLUMNS = ("rf_dim_native_abs_z_gap", "rf_dim_native_sign_disagree", "rf_dim_logistic_abs_z_gap", "oc_residual_norm", "oc_residual_ratio", "ld_knn_mean_distance", "ld_mahalanobis_centroid", "lr_subspace_residual_norm")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canonical_json(v: Any) -> bytes:
    return (json.dumps(v, indent=2, sort_keys=True, default=float) + "\n").encode("utf-8")


def write_create_only(p: Path, b: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(b)


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ inputs
def load_bindings() -> dict[str, Path]:
    raw = json.loads((TRACK_B_DIR / "INPUT_BINDINGS.json").read_text(encoding="utf-8"))["inputs"]
    out = {}
    for name, rec in raw.items():
        p = Path(rec["path"])
        if sha256_file(p) != rec["sha256"]:
            raise RuntimeError(f"frozen Track B input hash mismatch: {name}")
        out[name] = p
    return out


def read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def load_folds() -> tuple[tuple[FoldAssignment, ...], str]:
    p = TRACK_B_DIR / "TRACK_B_ORIGINAL_LEVEL_DATA.csv"
    frame = pd.read_csv(p, dtype={"base_item_id": str})
    if len(frame) != 67 or set(frame["fold"].astype(int)) != set(range(5)):
        raise RuntimeError("frozen original-level data does not carry the 67-original five-fold topology")
    return tuple(FoldAssignment(str(r.base_item_id), int(r.fold)) for r in frame.itertuples(index=False)), sha256_file(p)


def load_addendum(actor: str, addendum_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    recs = [json.loads(l) for l in (addendum_root / "track_b_completed_addendum.jsonl").open(encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["actor"] == actor]
    if len(recs) != 112:
        raise RuntimeError(f"{actor}: expected 112 addendum originals, found {len(recs)}")
    acts, resids, ids, rows = [], [], [], []
    for r in recs:
        if r["technical_receipt"]["state"] != "COMPLETE":
            raise RuntimeError(f"incomplete addendum item {r['base_item_id']}")
        p = addendum_root / r["full_selected_layer_activation"]["path"]
        if sha256_file(p) != r["full_selected_layer_activation"]["artifact_sha256"]:
            raise RuntimeError(f"addendum activation hash mismatch: {r['base_item_id']}")
        with np.load(p) as z:
            acts.append(np.asarray(z["selected_layer_activation"], dtype=np.float64).reshape(-1))
            resids.append(np.asarray(z["relation_residual"], dtype=np.float64).reshape(-1))
        ids.append(str(r["base_item_id"]))
        rows.append({"base_item_id": str(r["base_item_id"]), "frozen_dim_projection": float(r["frozen_dim_projection"]), "artifact_sha256": r["full_selected_layer_activation"]["artifact_sha256"]})
    return pd.DataFrame(rows), np.stack(acts), np.stack(resids), ids


def load_direction(actor: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    z = np.load(FREEZE_V2 / "runtime" / actor / "competitor_directions_v2.npz")
    d = np.asarray(z["raw_relation_dim"], dtype=np.float64)
    rand = np.asarray(z["matched_norm_random__raw_relation_dim"], dtype=np.float64)
    hashes = {"raw_relation_dim": hashlib.sha256(np.ascontiguousarray(z["raw_relation_dim"]).tobytes()).hexdigest(), "random_matrix": hashlib.sha256(np.ascontiguousarray(z["matched_norm_random__raw_relation_dim"]).tobytes()).hexdigest()}
    return d, rand, hashes


# ---------------------------------------------------------------- features
def transverse_features(acts: np.ndarray, direction: np.ndarray, native: np.ndarray, logistic: np.ndarray | None, train_mask: np.ndarray, resid_override: np.ndarray | None = None, basis_mask: np.ndarray | None = None) -> np.ndarray:
    """Fold-local transverse features for every item; statistics fit on train_mask items only.

    ``direction`` defines the projection (and, unless resid_override is given,
    the residual act − proj·d/‖d‖²). ``native`` is the native margin, ``logistic``
    the own-actor logistic margin (or None)."""
    dn = direction / np.linalg.norm(direction)
    proj = acts @ dn
    resid = acts - np.outer(proj, dn) if resid_override is None else resid_override
    tr = train_mask
    z = lambda v: (v - v[tr].mean()) / (v[tr].std(ddof=0) + 1e-12)  # noqa: E731
    zp, zn = z(proj), z(native)
    f = np.zeros((len(acts), len(TRANSVERSE_COLUMNS)))
    f[:, 0] = np.abs(zp - zn)
    f[:, 1] = (np.sign(proj) != np.sign(native)).astype(float)
    f[:, 2] = np.abs(zp - z(logistic)) if logistic is not None else 0.0
    rn = np.linalg.norm(resid, axis=1)
    f[:, 3] = rn
    f[:, 4] = rn / (np.linalg.norm(acts, axis=1) + 1e-12)
    # whitened PCA space from training references
    # Whitening basis from the endpoint-free reference originals (basis_mask) so every
    # scored item, training or test, is out-of-sample with respect to it; neighbour
    # and subspace statistics use training references excluding the item itself,
    # which is symmetric between training and test rows.
    bm = train_mask if basis_mask is None else basis_mask
    mu = acts[bm].mean(axis=0)
    xc = acts - mu
    u, s, vt = np.linalg.svd(xc[bm], full_matrices=False)
    rank = min(WHITEN_RANK, len(s))
    comps, scale = vt[:rank], s[:rank] / np.sqrt(max(bm.sum() - 1, 1)) + 1e-12
    w = (xc @ comps.T) / scale
    pred_class = np.sign(proj)
    for i in range(len(acts)):
        same = tr & (pred_class == pred_class[i])
        same[i] = False
        if same.sum() < K_NEIGHBOURS + 1:
            same = tr.copy(); same[i] = False
        dists = np.linalg.norm(w[same] - w[i], axis=1)
        f[i, 5] = float(np.sort(dists)[:K_NEIGHBOURS].mean())
        centroid = w[same].mean(axis=0)
        cov = np.cov(w[same].T) + 1e-6 * np.eye(rank)
        diff = w[i] - centroid
        f[i, 6] = float(np.sqrt(diff @ np.linalg.solve(cov, diff)))
        xs = xc[same]
        _, _, vts = np.linalg.svd(xs - xs.mean(axis=0), full_matrices=False)
        basis = vts[: min(SUBSPACE_RANK, len(vts))]
        local = xc[i] - xs.mean(axis=0)
        f[i, 7] = float(np.linalg.norm(local - basis.T @ (basis @ local)))
    return f


# --------------------------------------------------------------- cross-fit
def cross_fit_with_fold_local_features(table: pd.DataFrame, endpoint: str, blocks: tuple[FeatureBlock, ...], folds: tuple[FoldAssignment, ...], rung: str, feature_maker=None) -> AnalysisResult:
    """Ridge cross-fit mirroring the frozen module; transverse columns are recomputed per fold from training references."""
    fold_map = {f.original_id: f.fold for f in folds}
    frame = table.copy()
    frame["__fold"] = frame["base_item_id"].map(fold_map)
    preds, fold_rows = [], []
    for k in sorted(set(fold_map.values())):
        fk = frame.copy()
        test = fk["__fold"] == k
        if feature_maker is not None:
            feats = feature_maker(k)
            for j, col in enumerate(TRANSVERSE_COLUMNS):
                fk[col] = feats[:, j]
        model = _pipeline(blocks, alpha=ALPHA)
        model.fit(fk.loc[~test], fk.loc[~test, endpoint].astype(float))
        p = model.predict(fk.loc[test])
        y = fk.loc[test, endpoint].astype(float).to_numpy()
        preds.append(pd.DataFrame({"base_item_id": fk.loc[test, "base_item_id"].to_numpy(), "fold": k, "observed": y, "predicted": p, "rung": rung}))
        fold_rows.append({"fold": k, "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "n": int(test.sum())})
    predictions = pd.concat(preds, ignore_index=True).sort_values("base_item_id", kind="stable").reset_index(drop=True)
    metrics = _metrics(predictions["observed"].to_numpy(), predictions["predicted"].to_numpy())
    return AnalysisResult(rung=rung, predictions=predictions, metrics=metrics, fold_metrics=pd.DataFrame(fold_rows))


def increment(baseline: AnalysisResult, expanded: AnalysisResult, replicates: int) -> dict[str, Any]:
    boot = paired_increment_bootstrap(baseline, expanded, replicates=replicates, seed=SEED)
    folds = fold_improvement_count(baseline, expanded)
    inc = float(boot["estimate"])
    return {"normalized_rmse_increment": inc, "grouped_bootstrap": {k: float(v) for k, v in boot.items()}, "outer_folds_improved": folds, "passes_magnitude": inc >= GATE["normalized_rmse_increment_minimum"], "passes_bootstrap_ci": float(boot["ci_lower"]) > GATE["grouped_bootstrap_ci_lower_strictly_greater_than"], "passes_fold_consistency": folds >= GATE["outer_folds_improved_minimum"], "augmented_normalized_rmse": float(expanded.metrics["normalized_rmse"]), "baseline_normalized_rmse": float(baseline.metrics["normalized_rmse"])}


# ------------------------------------------------------------------ driver
def build_inputs(addendum_roots: dict[str, Path]) -> dict[str, Any]:
    paths = load_bindings()
    table = assemble_scalar_analysis_table(candidates=read_table(paths["candidate_outcomes"]), base_items=read_table(paths["base_items"]), own_model_geometry=read_table(paths["own_model_geometry"]), cross_model_geometry=read_table(paths["cross_model_geometry"]))
    if len(table) != 67:
        raise RuntimeError(f"expected 67 endpoint-eligible originals, got {len(table)}")
    base_items = read_table(paths["base_items"]).astype({"base_item_id": str})
    folds, folds_sha = load_folds()
    if set(f.original_id for f in folds) != set(table["base_item_id"].astype(str)):
        raise RuntimeError("frozen folds and endpoint population differ")
    actors = {}
    for actor, root in addendum_roots.items():
        meta, acts, resids, ids = load_addendum(actor, root)
        direction, rand, hashes = load_direction(actor)
        native = base_items.set_index("base_item_id").loc[ids, "native_margin"].astype(float).to_numpy()
        logistic = base_items.set_index("base_item_id").loc[ids, "raw_logistic_margin"].astype(float).to_numpy() if actor == "llama" else None
        actors[actor] = {"meta": meta, "acts": acts, "resids": resids, "ids": ids, "direction": direction, "random": rand, "hashes": hashes, "native": native, "logistic": logistic, "root": root}
    return {"table": table.astype({"base_item_id": str}), "folds": folds, "folds_sha256": folds_sha, "bindings": {k: str(v) for k, v in paths.items()}, "actors": actors}


def feature_maker_for(actor_data: dict[str, Any], table_ids: list[str], folds: tuple[FoldAssignment, ...], direction: np.ndarray, use_stored_residual: bool):
    ids = actor_data["ids"]
    idx = {i: n for n, i in enumerate(ids)}
    fold_of = {f.original_id: f.fold for f in folds}
    order = [idx[i] for i in table_ids]

    basis_mask = np.array([i not in fold_of for i in ids])  # endpoint-free originals: whitening basis, always references
    cache: dict[int, np.ndarray] = {}

    def make(k: int) -> np.ndarray:
        if k not in cache:  # features never depend on the endpoint, so per-fold caching is exact
            train_mask = np.array([fold_of.get(i, -1) != k for i in ids])
            feats = transverse_features(actor_data["acts"], direction, actor_data["native"], actor_data["logistic"], train_mask, resid_override=actor_data["resids"] if use_stored_residual else None, basis_mask=basis_mask)
            cache[k] = feats[order]
        return cache[k]

    return make


# ------------------------------------------------ random-direction band
# The 128 matched-norm directions are pre-drawn from the frozen matrix and the
# band loop consumes no random numbers, so its members are independent and are
# evaluated in worker processes. Every member runs single-threaded BLAS in both
# the serial and the parallel path so the two produce identical floats.
_BAND_CTX: dict[str, Any] = {}


def _band_init(ctx: dict[str, Any]) -> None:
    global _BAND_CTX
    _BAND_CTX = ctx


def _band_member(j: int) -> tuple[int, float]:
    c = _BAND_CTX
    with threadpool_limits(limits=1):
        mk = feature_maker_for(c["data"], c["table_ids"], c["folds"], c["data"]["random"][j], use_stored_residual=False)
        e_ = cross_fit_with_fold_local_features(c["table"], c["endpoint"], c["blocks"], c["folds"], "r", mk)
    return j, float(e_.metrics["normalized_rmse"])


def random_direction_band(data: dict[str, Any], table: pd.DataFrame, table_ids: list[str], folds: tuple[FoldAssignment, ...], endpoint: str, blocks: tuple[FeatureBlock, ...], rel_nrmse: float, random_directions: int, workers: int) -> np.ndarray:
    ctx = {"data": data, "table": table, "table_ids": table_ids, "folds": folds, "endpoint": endpoint, "blocks": blocks}
    band = np.empty(random_directions)
    if workers <= 1:
        _band_init(ctx)
        results = map(_band_member, range(random_directions))
    else:
        ex = ProcessPoolExecutor(max_workers=min(workers, random_directions), initializer=_band_init, initargs=(ctx,))
        results = ex.map(_band_member, range(random_directions))
    for j, v in results:
        band[j] = rel_nrmse - v
    if workers > 1:
        ex.shutdown()
    return band


def run_analysis(inputs: dict[str, Any], *, bootstrap: int = BOOTSTRAP_REPLICATES, permutations: int = PERMUTATION_REPLICATES, random_directions: int = RANDOM_DIRECTIONS, workers: int = 1) -> dict[str, Any]:
    table, folds = inputs["table"], inputs["folds"]
    table_ids = table["base_item_id"].astype(str).tolist()
    baseline_blocks = (BASELINE_TEXT, BASELINE_NATIVE, BASELINE_STRATUM)
    relation_blocks = (*baseline_blocks, OWN_MODEL_SCALAR)
    transverse_block = FeatureBlock("transverse", numeric=TRANSVERSE_COLUMNS)
    expanded_blocks = (*relation_blocks, transverse_block)
    for col in TRANSVERSE_COLUMNS:
        table[col] = np.nan
    endpoint = PRIMARY_ENDPOINT
    base = cross_fit_with_fold_local_features(table, endpoint, baseline_blocks, folds, "baseline_text_native")
    rel = cross_fit_with_fold_local_features(table, endpoint, relation_blocks, folds, "baseline_plus_own_llama_scalar")
    out: dict[str, Any] = {"schema_version": SCHEMA, "endpoint": endpoint, "originals": int(len(table)), "rungs": {"baseline_text_native": base.metrics, "baseline_plus_own_llama_scalar": rel.metrics}, "actors": {}}
    rng = np.random.default_rng(SEED)
    for actor, data in inputs["actors"].items():
        own = actor == "llama"
        maker = feature_maker_for(data, table_ids, folds, data["direction"], use_stored_residual=True)
        rung = f"baseline_plus_relation_plus_transverse_{actor}"
        exp = cross_fit_with_fold_local_features(table, endpoint, expanded_blocks, folds, rung, maker)
        inc = increment(rel, exp, bootstrap)
        # original-level outcome permutation: refit relation and transverse rungs under shuffled endpoints
        null_inc = np.empty(permutations)
        for i in range(permutations):
            perm = table.copy()
            perm[endpoint] = rng.permutation(perm[endpoint].to_numpy())
            b_ = cross_fit_with_fold_local_features(perm, endpoint, relation_blocks, folds, "b")
            e_ = cross_fit_with_fold_local_features(perm, endpoint, expanded_blocks, folds, "e", maker)
            null_inc[i] = b_.metrics["normalized_rmse"] - e_.metrics["normalized_rmse"]
        p_perm = float((1 + np.sum(null_inc >= inc["normalized_rmse_increment"])) / (permutations + 1))
        # matched-norm random-direction band
        band = random_direction_band(data, table, table_ids, folds, endpoint, expanded_blocks, float(rel.metrics["normalized_rmse"]), random_directions, workers)
        floor_ok = inc["augmented_normalized_rmse"] < GATE["floor_augmented_nrmse_max"]
        passes = inc["passes_magnitude"] and inc["passes_bootstrap_ci"] and inc["passes_fold_consistency"] and p_perm <= GATE["grouped_permutation_p_maximum"]
        if not floor_ok:
            classification = "ENDPOINT_AT_FLOOR"
        elif passes:
            classification = "TRANSVERSE_INCREMENT_SURVIVES"
        else:
            classification = "TRANSVERSE_INCREMENT_ABSENT"
        out["actors"][actor] = {"role": "own_actor_primary" if own else "cross_actor_secondary", "rung": rung, "metrics": exp.metrics, "fold_metrics": exp.fold_metrics.to_dict(orient="records"), "increment_over_relation_rung": inc, "permutation": {"p_value": p_perm, "replicates": permutations, "null_mean": float(null_inc.mean()), "null_q95": float(np.quantile(null_inc, 0.95))}, "random_direction_band": {"count": random_directions, "q025": float(np.quantile(band, 0.025)), "q975": float(np.quantile(band, 0.975)), "max": float(band.max()), "observed_exceeds_q975": bool(inc["normalized_rmse_increment"] > np.quantile(band, 0.975))}, "floor_guard": {"augmented_normalized_rmse": inc["augmented_normalized_rmse"], "interpretable": floor_ok}, "classification": classification, "direction_hashes": data["hashes"], "predictions": exp.predictions.to_dict(orient="records")}
    out["G5"] = {"primary_actor": "llama", "classification": out["actors"]["llama"]["classification"], "cross_actor_classification": out["actors"].get("gemma", {}).get("classification"), "cross_layer_trajectory": "NOT_AVAILABLE_IN_ADDENDUM (selected layer only)"}
    return out


def freeze_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "TRACK_B_TRANSVERSE_FREEZE_V1", "unit": "RELATION_REPRESENTATION_PROGRAM_V2 / Unit 8",
        "question": "Do full-vector transverse features of the frozen selected-layer activation predict rewrite fragility (rms_semantic_margin_movement) beyond text, native confidence, and the relation scalar?",
        "hypothesis": "transverse features add a normalized-RMSE increment ≥ 0.05 over the relation-scalar rung", "endpoint": PRIMARY_ENDPOINT,
        "expected_positive_result": "TRANSVERSE_INCREMENT_SURVIVES with augmented nRMSE < 1.0", "expected_falsifying_result": "TRANSVERSE_INCREMENT_ABSENT, or ENDPOINT_AT_FLOOR when augmented nRMSE ≥ 1.0 (population insufficient; itself a valid verdict)",
        "gate": GATE, "feature_families": list(TRANSVERSE_COLUMNS), "feature_constants": {"k_neighbours": K_NEIGHBOURS, "whiten_rank": WHITEN_RANK, "subspace_rank": SUBSPACE_RANK}, "cross_layer_trajectory": "NOT_AVAILABLE_IN_ADDENDUM",
        "ladder": ["baseline_text_native", "baseline_plus_own_llama_scalar", "baseline_plus_relation_plus_transverse_llama (primary)", "baseline_plus_relation_plus_transverse_gemma (cross-actor secondary)"],
        "model": {"ridge_alpha": ALPHA, "preprocess": "median impute + standardize numeric, one-hot stratum (frozen module _pipeline)"},
        "folds": {"source": "TRACK_B_ORIGINAL_LEVEL_DATA.csv fold column", "sha256": inputs["folds_sha256"], "originals": 67, "folds": 5, "reference_items": "45 endpoint-free addendum originals are always training references, never scored"},
        "uncertainty": {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "permutation_replicates": PERMUTATION_REPLICATES, "random_directions": RANDOM_DIRECTIONS, "seed": SEED},
        "inputs": {"track_b_bindings": inputs["bindings"], "addendum": {a: {"root": str(d["root"]), "items": len(d["ids"]), "activation_sha256_all": hashlib.sha256("".join(sorted(d["meta"]["artifact_sha256"])).encode()).hexdigest(), "direction_hashes": d["hashes"]} for a, d in inputs["actors"].items()}},
        "outcome_values_parsed_or_computed": False, "note": "assemble_scalar_analysis_table reads the endpoint column to build the table; no model was fit and no metric computed at freeze time",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run"):
        s = sub.add_parser(name)
        s.add_argument("--llama-addendum-root", type=Path, required=True)
        s.add_argument("--gemma-addendum-root", type=Path, required=True)
        s.add_argument("--freeze-root", type=Path, required=True)
        if name == "run":
            s.add_argument("--output-root", type=Path, required=True)
            s.add_argument("--repository", type=Path, default=ROOT)
            s.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
            s.add_argument("--permutations", type=int, default=PERMUTATION_REPLICATES)
            s.add_argument("--random-directions", type=int, default=RANDOM_DIRECTIONS)
            s.add_argument("--band-workers", type=int, default=DEFAULT_BAND_WORKERS, help="worker processes for the random-direction band (results are independent of this value)")
    a = ap.parse_args(argv)
    inputs = build_inputs({"llama": a.llama_addendum_root, "gemma": a.gemma_addendum_root})
    if a.cmd == "prepare":
        if a.freeze_root.exists() and any(a.freeze_root.iterdir()):
            raise SystemExit("freeze root not empty")
        write_create_only(a.freeze_root / FREEZE_FILE, canonical_json(freeze_payload(inputs)))
        write_create_only(a.freeze_root / PREOUTCOME_FILE, canonical_json({"schema_version": "TRACK_B_TRANSVERSE_PREOUTCOME_MANIFEST_V1", "status": "FROZEN_PRE_OUTCOME", "files": [{"path": FREEZE_FILE, "sha256": sha256_file(a.freeze_root / FREEZE_FILE)}]}))
        print(json.dumps({"freeze_sha256": sha256_file(a.freeze_root / FREEZE_FILE)}))
        return 0
    if a.output_root.exists() and any(a.output_root.iterdir()):
        raise SystemExit("output root not empty")
    rel = a.freeze_root.resolve().relative_to(a.repository.resolve()).as_posix()
    if not git(a.repository, "ls-files", "--", rel):
        raise SystemExit("freeze is not committed")
    manifest = json.loads((a.freeze_root / PREOUTCOME_FILE).read_text())
    if sha256_file(a.freeze_root / FREEZE_FILE) != manifest["files"][0]["sha256"]:
        raise SystemExit("freeze changed since manifest")
    frozen = json.loads((a.freeze_root / FREEZE_FILE).read_text())
    live = freeze_payload(inputs)
    if frozen["inputs"] != live["inputs"] or frozen["folds"] != live["folds"]:
        raise SystemExit("inputs drifted from the frozen bindings")
    res = run_analysis(inputs, bootstrap=a.bootstrap, permutations=a.permutations, random_directions=a.random_directions, workers=a.band_workers)
    res["source"] = {"commit": git(a.repository, "rev-parse", "HEAD"), "freeze_sha256": manifest["files"][0]["sha256"], "script_sha256": sha256_file(Path(__file__).resolve()), "band_workers": int(a.band_workers)}
    packet = {"schema_version": "EXTERNAL_REVIEW_PACKET_V1", "unit": "RELATION_REPRESENTATION_PROGRAM_V2 / Unit 8 / Track B transverse v1", "research_question": {k: frozen[k] for k in ("question", "hypothesis", "endpoint", "expected_positive_result", "expected_falsifying_result")}, "evidence_topology": {"independent_unit": "base_item_id", "endpoint_eligible_originals": 67, "reference_originals": 45, "folds": 5, "clustering": "rewrite rows nest in originals; one row per original in the analysis; endpoint actor is Llama; Gemma features are cross-actor"}, "exact_empirical_result": {"G5": res["G5"], "rungs": res["rungs"], "actors": {a_: {k: v for k, v in d.items() if k != "predictions"} for a_, d in res["actors"].items()}}, "negative_and_surprising_evidence": [f"{a_}: {d['classification']} (increment {d['increment_over_relation_rung']['normalized_rmse_increment']:.4f}, CI low {d['increment_over_relation_rung']['grouped_bootstrap']['ci_lower']:.4f}, permutation p {d['permutation']['p_value']:.3f}, folds improved {d['increment_over_relation_rung']['outer_folds_improved']}/5, augmented nRMSE {d['floor_guard']['augmented_normalized_rmse']:.3f})" for a_, d in res["actors"].items()] + ["cross-layer trajectory feature family not available: the addendum stores the selected layer only"], "claim_boundary": "67 endpoint-eligible originals from one behavioural actor; a null result here is not evidence that transverse geometry carries no fragility information, only that these frozen families do not predict it at this population size.", "artifact_pickup": res["source"]}
    A = a.output_root
    write_create_only(A / "TRACK_B_TRANSVERSE_RESULTS.json", canonical_json(res))
    write_create_only(A / "external_review_packet_v1.json", canonical_json(packet))
    lines = [f"# Track B transverse v1 — G5", "", f"Classification (own actor Llama): **{res['G5']['classification']}**; cross-actor Gemma: {res['G5']['cross_actor_classification']}", "", "| rung | nRMSE | Spearman |", "|---|---|---|"]
    for r, m in res["rungs"].items():
        lines.append(f"| {r} | {m['normalized_rmse']:.3f} | {m['spearman']:.3f} |")
    for a_, d in res["actors"].items():
        m, i = d["metrics"], d["increment_over_relation_rung"]
        lines.append(f"| {d['rung']} | {m['normalized_rmse']:.3f} | {m['spearman']:.3f} |")
        lines += ["", f"- {a_}: increment {i['normalized_rmse_increment']:.4f} [{i['grouped_bootstrap']['ci_lower']:.4f}, {i['grouped_bootstrap']['ci_upper']:.4f}], folds improved {i['outer_folds_improved']}/5, permutation p {d['permutation']['p_value']:.3f}, random-direction band q97.5 {d['random_direction_band']['q975']:.4f}, augmented nRMSE {d['floor_guard']['augmented_normalized_rmse']:.3f} → {d['classification']}"]
    write_create_only(A / "TRACK_B_TRANSVERSE_REPORT.md", ("\n".join(lines) + "\n").encode())
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(A.iterdir())]
    write_create_only(A / "artifact_manifest.json", canonical_json({"schema_version": "TRACK_B_TRANSVERSE_RESULT_MANIFEST_V1", "classification": res["G5"], "files": files}))
    print(json.dumps(res["G5"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
