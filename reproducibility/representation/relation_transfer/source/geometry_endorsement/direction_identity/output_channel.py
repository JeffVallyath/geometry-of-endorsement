from __future__ import annotations

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm == 0: raise ValueError("Zero vector")
    return value / norm


def difference_direction(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = np.asarray(activations, dtype=np.float64); y = np.asarray(labels)
    if set(np.unique(y)) != {-1, 1}: raise ValueError("Labels must be -1/+1")
    return unit(x[y == 1].mean(axis=0) - x[y == -1].mean(axis=0))


def semantic_and_token_directions(activations: np.ndarray, semantic: np.ndarray,
                                  physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return difference_direction(activations, semantic), difference_direction(activations, physical)


def orthonormal_span(vectors: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1: matrix = matrix[None, :]
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return right[singular > tolerance]


def residualize(direction: np.ndarray, channel_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(direction, dtype=np.float64)
    basis = orthonormal_span(channel_vectors)
    component = basis.T @ (basis @ original) if len(basis) else np.zeros_like(original)
    residual = original - component
    return component, residual


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))
