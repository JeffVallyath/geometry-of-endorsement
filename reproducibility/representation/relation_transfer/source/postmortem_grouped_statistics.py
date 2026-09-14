"""Grouped uncertainty primitives for the frozen postmortem contract."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def semantic_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    binary = (np.asarray(labels) == 1).astype(np.int8)
    if len(np.unique(binary)) != 2:
        raise ValueError("AUROC requires both semantic classes")
    return float(roc_auc_score(binary, values))


def orientation_free_auroc(auroc: float) -> float:
    return float(max(auroc, 1.0 - auroc))


def standardized_separation(
    scores: np.ndarray,
    labels: np.ndarray,
    reference_scores: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    reference = np.asarray(reference_scores, dtype=np.float64)
    scale = float(reference.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("frozen reference scores have zero variance")
    return float((scores[labels == 1].mean() - scores[labels == -1].mean()) / scale)


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 1.0 - alpha])
    return float(low), float(high)


def grouped_bootstrap_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[group] for group in chosen])
        draws[index] = metric(scores[take], labels[take])
    low, high = _interval(draws, confidence)
    return {"low": low, "high": high, "replicates": int(replicates)}


def grouped_label_sign_permutation(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    observed = semantic_auroc(scores, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=len(unique))
        null[index] = semantic_auroc(scores, labels * signs[inverse])
    distance = abs(observed - 0.5)
    p = (1 + int(np.count_nonzero(np.abs(null - 0.5) >= distance))) / (replicates + 1)
    low, high = _interval(null, 0.95)
    return {
        "observed": float(observed),
        "p_two_sided": float(p),
        "null_low": low,
        "null_high": high,
        "replicates": int(replicates),
    }


def paired_group_bootstrap_mean(
    differences: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(groups)
    if len(values) != len(groups) or not len(values):
        raise ValueError("paired differences and groups must be non-empty and aligned")
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[group] for group in chosen])
        draws[index] = float(values[take].mean())
    low, high = _interval(draws, confidence)
    return {
        "point": float(values.mean()),
        "low": low,
        "high": high,
        "rows": int(len(values)),
        "groups": int(len(unique)),
        "replicates": int(replicates),
    }
