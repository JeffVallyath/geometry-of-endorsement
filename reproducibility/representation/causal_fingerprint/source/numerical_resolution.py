from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Diagnostics:
    epsilon_fraction: float
    correlation: float
    sign_agreement: float
    median_absolute_relative_error: float
    saturation_fraction: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def diagnostics(jvp: np.ndarray, finite_difference: np.ndarray, saturation: np.ndarray, fraction: float) -> Diagnostics:
    left = np.asarray(jvp, dtype=np.float64)
    right = np.asarray(finite_difference, dtype=np.float64)
    if left.shape != right.shape or not left.size:
        raise ValueError("JVP/finite-difference shapes must match and be non-empty")
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 0.0
    sign = float(np.mean(np.sign(left) == np.sign(right)))
    relative = float(np.median(np.abs(left - right) / np.maximum(np.abs(left), 1e-8)))
    saturation_fraction = float(np.mean(np.asarray(saturation, dtype=bool)))
    passed = correlation >= 0.90 and sign >= 0.90 and relative <= 0.25 and saturation_fraction < 0.05
    return Diagnostics(float(fraction), correlation, sign, relative, saturation_fraction, passed)


def symmetric_margin_change(finite_difference: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    return np.asarray(finite_difference, dtype=np.float64) * (2.0 * np.asarray(epsilon, dtype=np.float64))


def estimate_resolution_floor(changes: np.ndarray) -> float:
    """Estimate a robust lattice step from saved symmetric margin changes.

    Very small reconstruction residue is removed relative to the 90th
    percentile. Candidate steps are lower-tail nonzero magnitudes; the step
    minimizing median distance to an integer lattice wins.
    """
    values = np.abs(np.asarray(changes, dtype=np.float64))
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 0.0
    scale = float(np.quantile(values, 0.90))
    meaningful = values[values > max(scale * 1e-6, 1e-10)]
    if not meaningful.size:
        return 0.0
    candidates = np.unique(np.round(meaningful, 10))
    candidates = candidates[candidates > 0]
    candidates = candidates[: min(64, len(candidates))]
    best_step, best_score = float(candidates[0]), float("inf")
    for step in candidates:
        ratios = meaningful / step
        residual = np.abs(ratios - np.rint(ratios))
        score = float(np.median(residual) + 0.01 * step / max(scale, 1e-12))
        if score < best_score:
            best_step, best_score = float(step), score
    return best_step


def resolution_bins(changes: np.ndarray, q: float) -> dict[str, np.ndarray]:
    values = np.abs(np.asarray(changes, dtype=np.float64))
    if q <= 0:
        return {
            "exact_zero": values == 0,
            "within_one_step": np.zeros(values.shape, dtype=bool),
            "within_two_steps": np.zeros(values.shape, dtype=bool),
            "above_four_steps": values > 0,
        }
    tolerance = max(q * 1e-6, 1e-12)
    return {
        "exact_zero": values <= tolerance,
        "within_one_step": (values > tolerance) & (values <= q + tolerance),
        "within_two_steps": (values > q + tolerance) & (values <= 2.0 * q + tolerance),
        "above_four_steps": values > 4.0 * q + tolerance,
    }


def subset_diagnostics(jvp: np.ndarray, finite_difference: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    keep = np.asarray(mask, dtype=bool)
    if int(keep.sum()) < 3:
        return {"n": int(keep.sum()), "correlation": None, "sign_agreement": None, "median_absolute_relative_error": None}
    d = diagnostics(np.asarray(jvp)[keep], np.asarray(finite_difference)[keep], np.zeros(int(keep.sum()), dtype=bool), 0.0)
    return {"n": int(keep.sum()), "correlation": d.correlation, "sign_agreement": d.sign_agreement, "median_absolute_relative_error": d.median_absolute_relative_error}


def most_common_nonzero_increments(changes: np.ndarray, limit: int = 12) -> list[dict[str, float | int]]:
    values = np.abs(np.asarray(changes, dtype=np.float64))
    values = values[values > 1e-10]
    if not values.size:
        return []
    rounded = np.round(values, decimals=8)
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:limit]
    return [{"increment": float(unique[index]), "count": int(counts[index])} for index in order]
