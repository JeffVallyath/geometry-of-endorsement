from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .output_channel import difference_direction


def fit_difference_in_means(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    return difference_direction(train_x, train_y)


def scores(direction: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return np.asarray(activations, dtype=np.float64) @ np.asarray(direction, dtype=np.float64)


def auroc(direction: np.ndarray, activations: np.ndarray, labels: np.ndarray) -> float:
    binary = (np.asarray(labels) == 1).astype(int)
    return float(roc_auc_score(binary, scores(direction, activations)))


def standardized_separation(direction: np.ndarray, activations: np.ndarray,
                            labels: np.ndarray) -> float:
    values = scores(direction, activations); labels = np.asarray(labels)
    pooled = float(values.std(ddof=1))
    return float((values[labels == 1].mean() - values[labels == -1].mean()) / pooled) if pooled else 0.0
