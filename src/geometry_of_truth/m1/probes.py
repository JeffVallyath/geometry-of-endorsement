from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class DifferenceInMeansProbe:
    direction: np.ndarray
    midpoint: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return features.astype(np.float32) @ self.direction - self.midpoint


@dataclass(frozen=True)
class LogisticProbe:
    model: Any
    c: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.model.decision_function(features.astype(np.float32))

    def probability(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features.astype(np.float32))[:, 1]


def fit_difference_in_means(
    features: np.ndarray, labels: np.ndarray
) -> DifferenceInMeansProbe:
    x = features.astype(np.float32)
    y = labels.astype(np.int8)
    positive = x[y == 1]
    negative = x[y == 0]
    if not len(positive) or not len(negative):
        raise RuntimeError("Difference-in-means requires both labels.")
    mean_positive = positive.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_negative = negative.mean(axis=0, dtype=np.float64).astype(np.float32)
    direction = mean_positive - mean_negative
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Difference-in-means direction has invalid norm.")
    direction = direction / norm
    midpoint = float(0.5 * (mean_positive + mean_negative) @ direction)
    return DifferenceInMeansProbe(direction, midpoint)


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c: float,
    max_iter: int,
    seed: int,
) -> LogisticProbe:
    model = LogisticRegression(
        C=float(c),
        solver="liblinear",
        dual=True,
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    model.fit(features.astype(np.float32), labels.astype(np.int8))
    return LogisticProbe(model, float(c))


def grouped_label_flip(
    labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Null that preserves rows and within-situation grouping.

    Every situation independently keeps or reverses all of its binary labels.
    This destroys the learned semantic orientation without fragmenting a
    situation across incompatible null assignments.
    """
    result = labels.astype(np.int8).copy()
    for group in np.unique(groups):
        if bool(rng.integers(0, 2)):
            mask = groups == group
            result[mask] = 1 - result[mask]
    return result
