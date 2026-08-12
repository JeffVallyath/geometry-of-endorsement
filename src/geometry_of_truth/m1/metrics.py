from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from .support.checkerboard_eval import evaluate_boards
from .support.relation_purity import paired_accuracy
from .support.split_stress_test import CON, SIT, SUPPORTS, VAL
from .support.dependence_power import dyadic_se
from .support.metrics import (
    BOTH_WAYS_ADDITIVE,
    BOTH_WAYS_CHANCE,
    Scaler,
    both_ways,
    interaction_contrast,
    signed_board_score,
)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2
        else float("nan")
    )


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, np.asarray(scores) >= 0))


def macro_pairwise(
    frame: pd.DataFrame, scores: np.ndarray, group_column: str
) -> tuple[float, np.ndarray, np.ndarray]:
    scored = frame.copy()
    scored["__score"] = np.asarray(scores)
    values: list[float] = []
    groups: list[str] = []
    for group, rows in scored.groupby(group_column, sort=False):
        supports = rows.loc[rows[VAL] == SUPPORTS, "__score"].to_numpy()
        opposes = rows.loc[rows[VAL] != SUPPORTS, "__score"].to_numpy()
        if not len(supports) or not len(opposes):
            continue
        delta = supports[:, None] - opposes[None, :]
        value = ((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size
        values.append(float(value))
        groups.append(str(group))
    array = np.asarray(values, dtype=float)
    return (float(array.mean()) if len(array) else float("nan")), array, np.asarray(groups)


def mirrored_pairwise(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    consideration_group = (
        "consideration_cluster_id"
        if "consideration_cluster_id" in frame.columns
        else CON
    )
    within_situation, per_situation, situation_groups = macro_pairwise(
        frame, scores, SIT
    )
    within_consideration, per_consideration, consideration_groups = macro_pairwise(
        frame, scores, consideration_group
    )
    exact_situation, n_situation_pairs, n_situation_groups = paired_accuracy(
        frame, np.asarray(scores), SIT, CON
    )
    exact_consideration, n_consideration_pairs, n_consideration_groups = (
        paired_accuracy(
            frame,
            np.asarray(scores),
            consideration_group,
            SIT,
        )
    )
    return {
        "within_situation_macro": within_situation,
        "within_consideration_macro": within_consideration,
        "mean_mirrored_pairwise": float(
            np.nanmean([within_situation, within_consideration])
        ),
        "within_situation_exact_m0": float(exact_situation),
        "within_consideration_exact_m0": float(exact_consideration),
        "n_situation_pairs": int(n_situation_pairs),
        "n_consideration_pairs": int(n_consideration_pairs),
        "n_situation_groups": int(n_situation_groups),
        "n_consideration_groups": int(n_consideration_groups),
        "_per_situation": per_situation,
        "_situation_groups": situation_groups,
        "_per_consideration": per_consideration,
        "_consideration_groups": consideration_groups,
    }


def boards_from_manifest(
    board_manifest: pd.DataFrame, evaluation_frame: pd.DataFrame
) -> list[dict[str, Any]]:
    position = {
        str(row_id): int(index)
        for index, row_id in enumerate(evaluation_frame["row_id"].astype(str))
    }
    boards: list[dict[str, Any]] = []
    for _, record in board_manifest.iterrows():
        try:
            indices = {
                "i_s1_c1": position[str(record["row_id_s1_A"])],
                "i_s1_c2": position[str(record["row_id_s1_B"])],
                "i_s2_c1": position[str(record["row_id_s2_A"])],
                "i_s2_c2": position[str(record["row_id_s2_B"])],
            }
            cluster_column = (
                "consideration_cluster_id"
                if "consideration_cluster_id" in evaluation_frame.columns
                else CON
            )
            node_a = str(evaluation_frame.iloc[indices["i_s1_c1"]][cluster_column])
            node_b = str(evaluation_frame.iloc[indices["i_s1_c2"]][cluster_column])
            if (
                str(evaluation_frame.iloc[indices["i_s2_c1"]][cluster_column]) != node_a
                or str(evaluation_frame.iloc[indices["i_s2_c2"]][cluster_column]) != node_b
            ):
                raise RuntimeError(
                    "A checkerboard's consideration endpoints are not stable across situations."
                )
            boards.append(
                {
                    "board_id": str(record["board_id"]),
                    **indices,
                    "node_a": node_a,
                    "node_b": node_b,
                }
            )
        except KeyError as exc:
            raise RuntimeError("A board cell is absent from its evaluation manifest.") from exc
    return boards


def board_metrics(
    boards: list[dict[str, Any]], raw_scores: np.ndarray, scaler: Scaler
) -> dict[str, Any]:
    raw_scores = np.asarray(raw_scores, dtype=float)
    scores = scaler(raw_scores)
    result = evaluate_boards(boards, dict(enumerate(raw_scores)))
    per_board: list[dict[str, Any]] = []
    for board in boards:
        d1 = float(raw_scores[board["i_s1_c1"]] - raw_scores[board["i_s1_c2"]])
        d2 = float(raw_scores[board["i_s2_c2"]] - raw_scores[board["i_s2_c1"]])
        per_board.append(
            {
                "board_id": board["board_id"],
                "node_a": board["node_a"],
                "node_b": board["node_b"],
                "gap_1": d1,
                "gap_2": d2,
                "pairwise": (
                    (1.0 if d1 > 0 else 0.5 if d1 == 0 else 0.0)
                    + (1.0 if d2 > 0 else 0.5 if d2 == 0 else 0.0)
                )
                / 2.0,
                "interaction_contrast": interaction_contrast(board, scores),
                "signed_board_score": float(signed_board_score(board, raw_scores)),
                "both_ways": float(both_ways(board, raw_scores)),
            }
        )
    interactions = np.asarray(
        [row["interaction_contrast"] for row in per_board], dtype=float
    )
    signed = np.asarray([row["signed_board_score"] for row in per_board], dtype=float)
    return {
        "interaction_contrast_mean": float(interactions.mean()),
        "interaction_contrast_sd": (
            float(interactions.std(ddof=1)) if len(interactions) > 1 else 0.0
        ),
        "signed_board_mean": float(signed.mean()),
        "pairwise": float(result["pairwise"]),
        "both_ways": float(result["both"]),
        "both_ways_chance": BOTH_WAYS_CHANCE,
        "both_ways_additive": BOTH_WAYS_ADDITIVE,
        "both_ways_descriptive_only": True,
        "n_boards": int(result["n_boards"]),
        "scaler": {
            "mu": scaler.mu,
            "sigma": scaler.sigma,
            "fitted_on": scaler.fitted_on,
        },
        "_per_board": per_board,
    }


def grouped_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"low": float("nan"), "high": float("nan"), "replicates": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    sampled = array[draws].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "low": float(np.quantile(sampled, alpha)),
        "high": float(np.quantile(sampled, 1.0 - alpha)),
        "replicates": int(replicates),
    }


def board_ci(
    metric: dict[str, Any],
    *,
    field: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    values = np.asarray([row[field] for row in metric["_per_board"]], dtype=float)
    return grouped_mean_ci(
        values,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )


def paired_board_delta_ci(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    left_by_id = {row["board_id"]: row["both_ways"] for row in left["_per_board"]}
    right_by_id = {row["board_id"]: row["both_ways"] for row in right["_per_board"]}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("Paired board comparison has different board IDs.")
    values = np.asarray(
        [left_by_id[key] - right_by_id[key] for key in sorted(left_by_id)],
        dtype=float,
    )
    result = grouped_mean_ci(
        values,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    result["point"] = float(values.mean())
    return result


def dyadic_mean_ci(
    values: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float | str]:
    """Normal CI for a board mean using the frozen dyadic-robust estimator."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "point": float("nan"), "standard_error": float("nan"),
            "low": float("nan"), "high": float("nan"),
            "confidence_level": float(confidence_level),
            "estimator": "dyadic_robust",
        }
    standard_error = dyadic_se(values, node_a, node_b)
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    point = float(values.mean())
    return {
        "point": point,
        "standard_error": float(standard_error),
        "low": float(point - z * standard_error),
        "high": float(point + z * standard_error),
        "confidence_level": float(confidence_level),
        "estimator": "dyadic_robust",
    }


def paired_interaction_delta_ci(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    confidence_level: float,
) -> dict[str, float | str]:
    """Dyadic-robust CI for paired Delta I_b on identical boards."""
    left_by_id = {row["board_id"]: row for row in left["_per_board"]}
    right_by_id = {row["board_id"]: row for row in right["_per_board"]}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("Paired board comparison has different board IDs.")
    ordered = sorted(left_by_id)
    values = np.asarray([
        left_by_id[key]["interaction_contrast"]
        - right_by_id[key]["interaction_contrast"]
        for key in ordered
    ])
    node_a = np.asarray([left_by_id[key]["node_a"] for key in ordered])
    node_b = np.asarray([left_by_id[key]["node_b"] for key in ordered])
    if any(
        right_by_id[key]["node_a"] != left_by_id[key]["node_a"]
        or right_by_id[key]["node_b"] != left_by_id[key]["node_b"]
        for key in ordered
    ):
        raise RuntimeError("Paired board comparison has different dyadic endpoints.")
    return dyadic_mean_ci(
        values, node_a, node_b, confidence_level=confidence_level
    )


def calibration(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    y = labels.astype(np.int8)
    probability = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if not mask.any():
            continue
        confidence = float(probability[mask].mean())
        frequency = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - frequency)
        rows.append(
            {
                "bin": index,
                "n": int(mask.sum()),
                "mean_probability": confidence,
                "positive_frequency": frequency,
            }
        )
    return {
        "brier": float(brier_score_loss(y, probability)),
        "ece_10_bin": float(ece),
        "bins": rows,
    }


def strip_private(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if not key.startswith("_")}
