from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .input_audit import canonical_json_bytes, load_config, sha256_bytes, sha256_file


READOUTS = ("difference_in_means", "logistic", "native_answer_margin", "text_only")
RELATION_STRATA = (
    "BOTH_CLEAR",
    "ONE_CLEAR",
    "NEITHER_CLEAR",
    "REVIEWER2_CLEAR",
    "REVIEWER2_AMBIGUOUS",
    "REVIEWER2_INVALID",
)


def build_human_strata(
    consensus_cells: pd.DataFrame,
    frozen_scores: pd.DataFrame,
    consensus_pairs: pd.DataFrame,
    type_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if len(consensus_cells) != 500 or consensus_cells["row_id"].duplicated().any():
        raise RuntimeError("Claim 1 consensus cells must contain 500 unique row IDs.")
    if len(frozen_scores) != 500 or frozen_scores["row_id"].duplicated().any():
        raise RuntimeError("Frozen Claim 1 scores must contain 500 unique row IDs.")
    score_columns = [
        "row_id",
        "board_id",
        "cell_position",
        "difference_in_means",
        "logistic",
        "native_answer_margin",
    ]
    missing = set(score_columns) - set(frozen_scores.columns)
    if missing:
        raise RuntimeError(f"Frozen score columns are missing: {sorted(missing)}")
    merged = consensus_cells.merge(
        frozen_scores[score_columns],
        on=["row_id", "board_id", "cell_position"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != 500 or not merged["_merge"].eq("both").all():
        raise RuntimeError("Claim 1 score/consensus stable-ID join is not exact.")
    merged = merged.drop(columns="_merge")

    reviewer_1_clear = merged["reviewer_1_clarity"].eq("CLEAR")
    reviewer_2_clear = merged["reviewer_2_clarity"].eq("CLEAR")
    merged["ordered_clarity_stratum"] = np.select(
        [reviewer_1_clear & reviewer_2_clear, reviewer_1_clear ^ reviewer_2_clear],
        ["BOTH_CLEAR", "ONE_CLEAR"],
        default="NEITHER_CLEAR",
    )
    merged["reviewer2_stratum"] = "REVIEWER2_" + merged["reviewer_2_clarity"].astype(str)

    merged["consideration_slot"] = merged["cell_position"].astype(str).str[-1]
    pair_columns = [
        "board_id",
        "consideration_slot",
        "reviewer_1_sense",
        "reviewer_2_sense",
        "sense_disputed",
    ]
    pairs = consensus_pairs[pair_columns].copy()
    if len(pairs) != 250 or pairs.duplicated(["board_id", "consideration_slot"]).any():
        raise RuntimeError("Repeated-consideration rows are not unique and complete.")
    merged = merged.merge(
        pairs, on=["board_id", "consideration_slot"], how="left", validate="many_to_one"
    )
    if merged["reviewer_1_sense"].isna().any():
        raise RuntimeError("Repeated-consideration join is incomplete.")
    stable_1 = merged["reviewer_1_sense"].eq("STABLE")
    stable_2 = merged["reviewer_2_sense"].eq("STABLE")
    merged["repeated_consideration_stratum"] = np.select(
        [stable_1 & stable_2, stable_1 ^ stable_2],
        ["BOTH_STABLE", "ONE_STABLE"],
        default="NEITHER_STABLE_OR_DISPUTED",
    )

    merged["label"] = merged["stored_relation"].map({"Supports": 1, "Opposes": 0})
    if merged["label"].isna().any():
        raise RuntimeError("Stored relation must be Supports or Opposes.")
    merged["relation_sign"] = merged["label"].map({1: 1.0, 0: -1.0})
    for readout in READOUTS[:-1]:
        merged[readout] = pd.to_numeric(merged[readout], errors="raise")
        merged[f"aligned_{readout}"] = merged["relation_sign"] * merged[readout]
    merged["text_only"] = np.nan
    merged["aligned_text_only"] = np.nan

    if type_map is not None:
        required = {"row_id", "consideration_type"}
        if not required <= set(type_map.columns) or type_map["row_id"].duplicated().any():
            raise RuntimeError("Authoritative type map is malformed.")
        merged = merged.merge(
            type_map[["row_id", "consideration_type"]],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        if merged["consideration_type"].isna().any():
            raise RuntimeError("Authoritative type map is incomplete.")
    return merged.sort_values(["board_id", "cell_position"]).reset_index(drop=True)


def _separation(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(positive) - 1) * np.var(positive, ddof=1) + (len(negative) - 1) * np.var(negative, ddof=1))
        / (len(positive) + len(negative) - 2)
    )
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else float("nan")


def _metrics(frame: pd.DataFrame, readout: str) -> dict[str, float]:
    clean = frame.dropna(subset=[readout]).copy()
    if clean.empty:
        return {
            "auroc": float("nan"),
            "accuracy": float("nan"),
            "mean_aligned_score": float("nan"),
            "standardized_class_separation": float("nan"),
        }
    labels = clean["label"].to_numpy(dtype=int)
    scores = clean[readout].to_numpy(dtype=float)
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "auroc": auroc,
        "accuracy": float(np.mean((scores >= 0.0).astype(int) == labels)),
        "mean_aligned_score": float(clean[f"aligned_{readout}"].mean()),
        "standardized_class_separation": _separation(labels, scores),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    readout: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    clean = frame.dropna(subset=[readout]).copy()
    boards, board_inverse = np.unique(clean["board_id"].astype(str), return_inverse=True)
    if not len(boards) or clean.empty:
        return {
            name: {"low": None, "high": None, "valid_replicates": 0}
            for name in _metrics(frame, readout)
        }
    labels = clean["label"].to_numpy(dtype=int)
    scores = clean[readout].to_numpy(dtype=float)
    aligned = clean[f"aligned_{readout}"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in _metrics(frame, readout)}
    for _ in range(replicates):
        counts = np.bincount(
            rng.choice(len(boards), size=len(boards), replace=True), minlength=len(boards)
        ).astype(float)
        weights = counts[board_inverse]
        for name, value in _weighted_metrics(labels, scores, aligned, weights).items():
            if np.isfinite(value):
                draws[name].append(float(value))
    result: dict[str, dict[str, float | int | None]] = {}
    for name, values in draws.items():
        if values:
            low, high = np.quantile(values, [0.025, 0.975])
            result[name] = {
                "low": float(low),
                "high": float(high),
                "valid_replicates": len(values),
            }
        else:
            result[name] = {"low": None, "high": None, "valid_replicates": 0}
    return result


def _weighted_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    aligned: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    active = weights > 0
    labels = labels[active]
    scores = scores[active]
    aligned = aligned[active]
    weights = weights[active]
    if not len(labels) or weights.sum() <= 0:
        return {name: float("nan") for name in ("auroc", "accuracy", "mean_aligned_score", "standardized_class_separation")}
    auroc = (
        float(roc_auc_score(labels, scores, sample_weight=weights))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    accuracy = float(np.average((scores >= 0).astype(int) == labels, weights=weights))
    mean_aligned = float(np.average(aligned, weights=weights))
    class_stats = []
    for label in (1, 0):
        mask = labels == label
        count = float(weights[mask].sum())
        if count <= 1:
            class_stats.append((count, float("nan"), float("nan")))
            continue
        mean = float(np.average(scores[mask], weights=weights[mask]))
        variance = float(np.sum(weights[mask] * (scores[mask] - mean) ** 2) / (count - 1))
        class_stats.append((count, mean, variance))
    (n_pos, mean_pos, var_pos), (n_neg, mean_neg, var_neg) = class_stats
    if n_pos > 1 and n_neg > 1:
        pooled = np.sqrt(((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2))
        separation = float((mean_pos - mean_neg) / pooled) if pooled > 0 else float("nan")
    else:
        separation = float("nan")
    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "mean_aligned_score": mean_aligned,
        "standardized_class_separation": separation,
    }


def relation_stratum_metrics(
    cells: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    definitions: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("BOTH_CLEAR", lambda value: value["ordered_clarity_stratum"].eq("BOTH_CLEAR")),
        ("ONE_CLEAR", lambda value: value["ordered_clarity_stratum"].eq("ONE_CLEAR")),
        ("NEITHER_CLEAR", lambda value: value["ordered_clarity_stratum"].eq("NEITHER_CLEAR")),
        ("REVIEWER2_CLEAR", lambda value: value["reviewer_2_clarity"].eq("CLEAR")),
        ("REVIEWER2_AMBIGUOUS", lambda value: value["reviewer_2_clarity"].eq("AMBIGUOUS")),
        ("REVIEWER2_INVALID", lambda value: value["reviewer_2_clarity"].eq("INVALID")),
    ]
    rows: list[dict[str, Any]] = []
    for stratum_index, (stratum, definition) in enumerate(definitions):
        group = cells.loc[definition(cells)].copy()
        for readout_index, readout in enumerate(READOUTS):
            point = _metrics(group, readout)
            intervals = _cluster_bootstrap(
                group,
                readout,
                replicates=replicates,
                seed=seed + 1000 * stratum_index + readout_index,
            )
            row: dict[str, Any] = {
                "stratum": stratum,
                "readout": readout,
                "available": bool(group[readout].notna().any()),
                "cells": int(len(group)),
                "boards": int(group["board_id"].nunique()),
                "supports": int(group["label"].sum()),
                "opposes": int((1 - group["label"]).sum()),
                **point,
            }
            for metric, interval in intervals.items():
                row[f"{metric}_ci_low"] = interval["low"]
                row[f"{metric}_ci_high"] = interval["high"]
                row[f"{metric}_bootstrap_valid"] = interval["valid_replicates"]
            rows.append(row)
    return pd.DataFrame(rows)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.var(x) == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def _weighted_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    variance = float(np.average((x - mean_x) ** 2, weights=weights))
    if variance <= 0:
        return float("nan")
    covariance = float(np.average((x - mean_x) * (y - mean_y), weights=weights))
    return covariance / variance


def ordered_clarity_gradient(
    cells: pd.DataFrame,
    readout: str,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> dict[str, Any]:
    clean = cells.dropna(subset=[readout]).copy()
    if clean.empty:
        return {"readout": readout, "available": False, "reason": "row-level scores unavailable"}
    order = {"NEITHER_CLEAR": 0.0, "ONE_CLEAR": 1.0, "BOTH_CLEAR": 2.0}
    clean["clarity_order"] = clean["ordered_clarity_stratum"].map(order)
    observed = _slope(
        clean["clarity_order"].to_numpy(float),
        clean[f"aligned_{readout}"].to_numpy(float),
    )
    boards, board_inverse = np.unique(clean["board_id"].astype(str), return_inverse=True)
    x_values = clean["clarity_order"].to_numpy(float)
    y_values = clean[f"aligned_{readout}"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(bootstrap_replicates):
        counts = np.bincount(
            rng.choice(len(boards), size=len(boards), replace=True), minlength=len(boards)
        ).astype(float)
        value = _weighted_slope(x_values, y_values, counts[board_inverse])
        if np.isfinite(value):
            boot.append(value)

    pivot_score = clean.pivot(index="board_id", columns="cell_position", values=f"aligned_{readout}")
    pivot_order = clean.pivot(index="board_id", columns="cell_position", values="clarity_order")
    common = pivot_score.dropna().index.intersection(pivot_order.dropna().index)
    score_matrix = pivot_score.loc[common].to_numpy(float)
    order_matrix = pivot_order.loc[common].to_numpy(float)
    null: list[float] = []
    for _ in range(permutation_replicates):
        shuffled = score_matrix[rng.permutation(len(score_matrix))]
        value = _slope(order_matrix.reshape(-1), shuffled.reshape(-1))
        if np.isfinite(value):
            null.append(value)
    low, high = (np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan))
    p_greater = (
        float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))) if null else float("nan")
    )
    means = clean.groupby("ordered_clarity_stratum")[f"aligned_{readout}"].mean().to_dict()
    return {
        "readout": readout,
        "available": True,
        "ordered_means": {name: float(means.get(name, np.nan)) for name in order},
        "slope_per_clarity_step": observed,
        "slope_ci_low": float(low),
        "slope_ci_high": float(high),
        "bootstrap_valid": len(boot),
        "board_permutation_p_greater": p_greater,
        "permutation_valid": len(null),
        "board_permutation_preserved_cell_position": True,
    }


def checkerboard_table(cells: pd.DataFrame, boards: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    board_flags = boards.set_index("board_id")
    for board_id, group in cells.groupby("board_id", sort=True):
        if len(group) != 4:
            raise RuntimeError(f"Checkerboard {board_id} does not contain four cells.")
        record: dict[str, Any] = {
            "board_id": str(board_id),
            "all_four_consensus_clear": str(
                board_flags.loc[str(board_id), "all_four_consensus_clear"]
            ).lower()
            == "true",
            "strict_first_stage_consensus_board": str(
                board_flags.loc[str(board_id), "strict_first_stage_consensus_board"]
            ).lower()
            == "true",
            "disputed_board": str(board_flags.loc[str(board_id), "disputed_board"]).lower()
            == "true",
        }
        for readout in READOUTS:
            record[f"mean_aligned_{readout}"] = (
                float(group[f"aligned_{readout}"].mean())
                if group[f"aligned_{readout}"].notna().any()
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def run_claim1_human_strata(repo_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = repo / config["paths"]["output_root"]
    if not (output / "INPUT_MANIFEST.json").is_file() or not (output / "STUDY_CONTRACT.json").is_file():
        raise RuntimeError("AUDIT_AND_FREEZE must complete before Claim 1 analysis.")
    human = repo / config["paths"]["human_derived_root"]
    cells = pd.read_csv(human / "claim1_consensus_cells.csv", dtype=str).fillna("")
    boards = pd.read_csv(human / "claim1_consensus_boards.csv", dtype=str).fillna("")
    pairs = pd.read_csv(human / "claim1_consensus_reason_pairs.csv", dtype=str).fillna("")
    scores = pd.read_csv(repo / config["paths"]["llama_cell_scores"], dtype=str).fillna("")
    type_map_path = output / "frozen_valueprism_type_map.csv"
    type_map = pd.read_csv(type_map_path, dtype=str).fillna("") if type_map_path.is_file() else None
    stratified = build_human_strata(cells, scores, pairs, type_map)
    statistics = config["statistics"]
    metrics = relation_stratum_metrics(
        stratified,
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["seed"]),
    )
    gradients = [
        ordered_clarity_gradient(
            stratified,
            readout,
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            permutation_replicates=int(statistics["permutation_replicates"]),
            seed=int(statistics["seed"]) + 10000 * index,
        )
        for index, readout in enumerate(READOUTS)
    ]
    board_table = checkerboard_table(stratified, boards)

    cell_payload = stratified.to_csv(index=False, lineterminator="\n").encode("utf-8")
    board_payload = board_table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metrics_payload = metrics.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_once(output / "claim1_human_strata_cells.csv", cell_payload)
    _write_once(output / "claim1_human_strata_boards.csv", board_payload)
    _write_once(output / "claim1_human_strata_metrics.csv", metrics_payload)

    report = _report(metrics, gradients, board_table)
    report_sha = _write_once(output / "CLAIM1_HUMAN_STRATA_REPORT.md", report.encode("utf-8"))
    return {
        "status": "CLAIM1_HUMAN_STRATA_COMPLETE",
        "phase_label": "POST-HOC DEVELOPMENT SEMANTIC STRATIFICATION",
        "models": {"llama": "available", "gemma": "row-level frozen evidence unavailable"},
        "cells": len(stratified),
        "strict_consensus_boards": int(board_table["strict_first_stage_consensus_board"].sum()),
        "report_sha256": report_sha,
        "gradients": gradients,
    }


def run_gemma_human_strata_supplement(
    repo_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    """Use the outcome-blind extracted Gemma cache to fill the missing row-level lane.

    This is explicitly supplemental: the cache was not available at the original
    Phase 1 freeze, and the text-only comparator remains unavailable.
    """
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = repo / config["paths"]["output_root"]
    cache_path = output / "checkpoints" / "gemma_claim1_activations.npz"
    causal_path = output / "checkpoints" / "causal_claim1_gemma.parquet"
    if not cache_path.is_file() or not causal_path.is_file():
        raise RuntimeError("Gemma supplemental strata require the frozen activation and causal baselines.")
    probe_path = Path(config["paths"]["gemma_probe_bundle"])
    if not probe_path.is_absolute():
        probe_path = repo / probe_path
    if sha256_file(probe_path) != config["expected_inputs"]["gemma_probe_bundle_sha256"]:
        raise RuntimeError("Gemma frozen probe bundle hash changed.")
    with zipfile.ZipFile(probe_path) as archive:
        arrays = np.load(io.BytesIO(archive.read("m1_probe_parameters.npz")), allow_pickle=False)
    direction = np.asarray(arrays["difference_in_means_direction"], dtype=np.float32)
    midpoint = float(np.asarray(arrays["difference_in_means_midpoint"]).item())
    logistic = np.asarray(arrays["logistic_coef"], dtype=np.float32).reshape(-1)
    intercept = float(np.asarray(arrays["logistic_intercept"]).reshape(-1)[0])
    if (
        int(np.asarray(arrays["selected_layer"]).item()) != 27
        or str(np.asarray(arrays["model_revision"]).item())
        != config["models"]["gemma"]["revision"]
    ):
        raise RuntimeError("Gemma probe layer or revision changed.")

    cache = np.load(cache_path, allow_pickle=False)
    cache_frame = pd.DataFrame(
        {
            "row_id": np.asarray(cache["row_id"]).astype(str),
            "split": np.asarray(cache["split"]).astype(str),
            "activation": list(np.asarray(cache["activation"], dtype=np.float32)),
        }
    )
    cache_frame = cache_frame.loc[cache_frame["split"].eq("pilot_eval")].copy()
    if len(cache_frame) != 500 or cache_frame["row_id"].duplicated().any():
        raise RuntimeError("Gemma held-out activation topology changed.")
    activation = np.stack(cache_frame["activation"].to_numpy()).astype(np.float32)
    cache_frame["difference_in_means"] = activation @ direction - np.float32(midpoint)
    cache_frame["logistic"] = activation @ logistic + np.float32(intercept)

    causal = pd.read_parquet(causal_path)
    baseline = causal.loc[causal["family"].eq("no_intervention")].copy()
    if len(baseline) != 500 or baseline["row_id"].duplicated().any():
        raise RuntimeError("Gemma unsteered Claim 1 baseline topology changed.")
    cache_frame = cache_frame.merge(
        baseline[["row_id", "semantic_margin"]].rename(
            columns={"semantic_margin": "native_answer_margin"}
        ),
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    human = repo / config["paths"]["human_derived_root"]
    cells = pd.read_csv(human / "claim1_consensus_cells.csv", dtype=str).fillna("")
    boards = pd.read_csv(human / "claim1_consensus_boards.csv", dtype=str).fillna("")
    pairs = pd.read_csv(human / "claim1_consensus_reason_pairs.csv", dtype=str).fillna("")
    scores = cells[["row_id", "board_id", "cell_position"]].merge(
        cache_frame[
            ["row_id", "difference_in_means", "logistic", "native_answer_margin"]
        ],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    type_map = pd.read_csv(output / "frozen_valueprism_type_map.csv", dtype=str).fillna("")
    stratified = build_human_strata(cells, scores, pairs, type_map)
    statistics = config["statistics"]
    metrics = relation_stratum_metrics(
        stratified,
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["seed"]) + 200000,
    )
    gradients = [
        ordered_clarity_gradient(
            stratified,
            readout,
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            permutation_replicates=int(statistics["permutation_replicates"]),
            seed=int(statistics["seed"]) + 210000 + 10000 * index,
        )
        for index, readout in enumerate(READOUTS)
    ]
    board_table = checkerboard_table(stratified, boards)
    _write_once(
        output / "claim1_human_strata_gemma_cells.csv",
        stratified.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    _write_once(
        output / "claim1_human_strata_gemma_metrics.csv",
        metrics.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    report = _gemma_supplement_report(metrics, gradients, board_table)
    report_path = output / "CLAIM1_HUMAN_STRATA_GEMMA_SUPPLEMENT.md"
    report_sha = _write_once(report_path, report.encode("utf-8"))
    result = {
        "status": "GEMMA_HUMAN_STRATA_SUPPLEMENT_COMPLETE",
        "phase_label": "POST-HOC DEVELOPMENT SEMANTIC STRATIFICATION — SUPPLEMENTAL ROW EXTRACTION",
        "report_sha256": report_sha,
        "cache_sha256": sha256_file(cache_path),
        "gradients": gradients,
    }
    result_path = output / "CLAIM1_HUMAN_STRATA_GEMMA_RESULT.json"
    _write_once(result_path, canonical_json_bytes(result))
    return {**result, "result_sha256": sha256_file(result_path)}


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen analysis artifact differs: {path}")
    else:
        path.write_bytes(payload)
    return sha256_bytes(payload)


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.3f}" if isinstance(value, (float, np.floating)) else str(value)


def _report(metrics: pd.DataFrame, gradients: list[dict[str, Any]], boards: pd.DataFrame) -> str:
    rows = [
        "# Claim 1 human-consensus semantic-stratification report",
        "",
        "Status: **POST-HOC DEVELOPMENT SEMANTIC STRATIFICATION**. This is not sealed confirmation.",
        "",
        "## Evidence availability",
        "",
        "- Llama: all 500 frozen row-level Claim 1 scores are available and joined by exact stable IDs.",
        "- Gemma: the frozen direction and aggregate development result exist, but no row-level Claim 1 score/cache payload is retained. Gemma stratum metrics are therefore unavailable rather than reconstructed.",
        "- Text-only comparator: no exact row-level frozen score is retained; the lane is unavailable rather than refit.",
        "- Reviewer 2 used an owner-accepted session-log waiver; the independent judgments were not overwritten, but strict protocol-compliance is not claimed.",
        "",
        "## Llama relation strata",
        "",
        "| Stratum | Readout | n | AUROC | Accuracy | Mean aligned | Separation | 95% CI, aligned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in metrics.to_dict("records"):
        if not record["available"]:
            continue
        rows.append(
            "| {stratum} | {readout} | {cells} | {auroc} | {accuracy} | {mean} | {sep} | [{low}, {high}] |".format(
                stratum=record["stratum"],
                readout=record["readout"],
                cells=record["cells"],
                auroc=_format(record["auroc"]),
                accuracy=_format(record["accuracy"]),
                mean=_format(record["mean_aligned_score"]),
                sep=_format(record["standardized_class_separation"]),
                low=_format(record["mean_aligned_score_ci_low"]),
                high=_format(record["mean_aligned_score_ci_high"]),
            )
        )
    rows.extend(["", "## Ordered human-clarity gradient", ""])
    for gradient in gradients:
        if not gradient.get("available"):
            rows.append(f"- {gradient['readout']}: unavailable ({gradient['reason']}).")
            continue
        means = gradient["ordered_means"]
        rows.append(
            f"- {gradient['readout']}: BOTH_CLEAR={_format(means['BOTH_CLEAR'])}, "
            f"ONE_CLEAR={_format(means['ONE_CLEAR'])}, NEITHER_CLEAR={_format(means['NEITHER_CLEAR'])}; "
            f"slope={_format(gradient['slope_per_clarity_step'])} "
            f"95% CI [{_format(gradient['slope_ci_low'])}, {_format(gradient['slope_ci_high'])}], "
            f"board-permutation one-sided p={_format(gradient['board_permutation_p_greater'])}."
        )
    rows.extend(
        [
            "",
            "## Strict checkerboards",
            "",
            f"Strict two-reviewer first-stage checkerboards: {int(boards['strict_first_stage_consensus_board'].sum())} of {len(boards)}. These results are descriptive because the surviving board count is small.",
            "",
            "## Claim boundary",
            "",
            "This analysis can show whether frozen readouts are cleaner where two existing reviewers agree the relation is clear. It cannot establish moral truth, sealed generalization, causal control, or cross-model replication when Gemma row-level evidence is absent.",
            "",
        ]
    )
    return "\n".join(rows)


def _gemma_supplement_report(
    metrics: pd.DataFrame, gradients: list[dict[str, Any]], boards: pd.DataFrame
) -> str:
    rows = [
        "# Claim 1 human-consensus stratification: Gemma supplement",
        "",
        "Status: **POST-HOC DEVELOPMENT SEMANTIC STRATIFICATION — SUPPLEMENTAL ROW EXTRACTION**. This is not sealed confirmation.",
        "",
        "The original audit found no retained row-level Gemma cache. These rows were obtained later by the frozen, outcome-blind batch-size-one extraction required for the causal/type-transfer arms. The DIM and logistic vectors, layer 27, revision, prompts, human strata, and statistical rules were not refit or selected from these results.",
        "",
        "The text-only comparator remains unavailable. Native answer margins are the unsteered causal baselines and had to reproduce any previously retained reason scores before causal interpretation.",
        "",
        "| Stratum | Readout | n | AUROC | Accuracy | Mean aligned | Separation | 95% CI, aligned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in metrics.to_dict("records"):
        if not record["available"]:
            continue
        rows.append(
            "| {stratum} | {readout} | {cells} | {auroc} | {accuracy} | {mean} | {sep} | [{low}, {high}] |".format(
                stratum=record["stratum"],
                readout=record["readout"],
                cells=record["cells"],
                auroc=_format(record["auroc"]),
                accuracy=_format(record["accuracy"]),
                mean=_format(record["mean_aligned_score"]),
                sep=_format(record["standardized_class_separation"]),
                low=_format(record["mean_aligned_score_ci_low"]),
                high=_format(record["mean_aligned_score_ci_high"]),
            )
        )
    rows.extend(["", "## Ordered human-clarity gradient", ""])
    for gradient in gradients:
        if not gradient.get("available"):
            rows.append(f"- {gradient['readout']}: unavailable ({gradient['reason']}).")
            continue
        means = gradient["ordered_means"]
        rows.append(
            f"- {gradient['readout']}: BOTH_CLEAR={_format(means['BOTH_CLEAR'])}, "
            f"ONE_CLEAR={_format(means['ONE_CLEAR'])}, NEITHER_CLEAR={_format(means['NEITHER_CLEAR'])}; "
            f"slope={_format(gradient['slope_per_clarity_step'])} "
            f"95% CI [{_format(gradient['slope_ci_low'])}, {_format(gradient['slope_ci_high'])}], "
            f"board-permutation one-sided p={_format(gradient['board_permutation_p_greater'])}."
        )
    rows.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"Strict checkerboards: {int(boards['strict_first_stage_consensus_board'].sum())}; descriptive only.",
            "This supplement can test cross-model replication of the human-clarity gradient. Without the frozen text-only lane it cannot establish activation specificity, and it does not convert this post-hoc development analysis into sealed confirmation.",
            "",
        ]
    )
    return "\n".join(rows)
