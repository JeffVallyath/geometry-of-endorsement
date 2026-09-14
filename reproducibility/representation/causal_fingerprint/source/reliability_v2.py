from __future__ import annotations

import numpy as np


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fixed_direction_split_half(effects: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(effects, dtype=np.float64)
    group_ids = np.asarray(groups).astype(str)
    unique = sorted(set(group_ids.tolist()))
    left_groups = set(unique[::2])
    left = values[np.asarray([group in left_groups for group in group_ids])].mean(axis=0)
    right = values[np.asarray([group not in left_groups for group in group_ids])].mean(axis=0)
    return left, right, pearson(left, right)


def per_direction_fingerprint_similarity(half_one: np.ndarray, half_two: np.ndarray) -> np.ndarray:
    one, two = np.asarray(half_one, dtype=np.float64), np.asarray(half_two, dtype=np.float64)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("fingerprints must share [directions, environments] shape")
    return np.asarray([pearson(one[index], two[index]) for index in range(one.shape[0])], dtype=np.float64)


def causal_gram_reliability(half_one: np.ndarray, half_two: np.ndarray) -> float:
    one, two = np.asarray(half_one, dtype=np.float64), np.asarray(half_two, dtype=np.float64)
    if one.shape != two.shape or one.ndim != 2:
        raise ValueError("fingerprints must share [directions, environments] shape")
    one_norm = one / np.maximum(np.linalg.norm(one, axis=1, keepdims=True), 1e-12)
    two_norm = two / np.maximum(np.linalg.norm(two, axis=1, keepdims=True), 1e-12)
    gram_one, gram_two = one_norm @ one_norm.T, two_norm @ two_norm.T
    indices = np.triu_indices(one.shape[0], 1)
    return pearson(gram_one[indices], gram_two[indices])


def stable_gradient_random_direction_fixture(seed: int = 2026082501) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    gradient = rng.normal(size=128)
    directions = rng.normal(size=(32, 128))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    half_one = gradient @ directions.T + rng.normal(scale=0.01, size=32)
    half_two = gradient @ directions.T + rng.normal(scale=0.01, size=32)
    return {"random_direction_split_half_reliability": pearson(half_one, half_two), "semantic_structure_planted": 0.0}
