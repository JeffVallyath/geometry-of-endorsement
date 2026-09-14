from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm


PREDICTOR_COLUMNS = (
    "native_confidence",
    "primary_jury_scalar",
    "frozen_dim_geometry",
)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    positive = vectors @ np.diag(np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(positive))
    result = positive / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)
    return result


def observed_predictor_correlation(frame: pd.DataFrame) -> np.ndarray:
    missing = set(PREDICTOR_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Power simulation is missing observed predictor columns: {sorted(missing)}"
        )
    numeric = frame.loc[:, PREDICTOR_COLUMNS].apply(pd.to_numeric, errors="raise")
    if len(numeric) < 20 or numeric.isna().any().any():
        raise RuntimeError("Observed predictor distribution is incomplete or too small.")
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    if standardized.isna().any().any():
        raise RuntimeError("An observed power-simulation predictor has zero variance.")
    return _nearest_correlation(standardized.corr().to_numpy(dtype=float))


def _scenario_rng(seed: int, values: tuple[Any, ...]) -> np.random.Generator:
    payload = "|".join([str(seed), *map(str, values)]).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(derived)


def _mapping_consistency(config: dict, scenario: str) -> float:
    power = config["power"]
    if scenario == "current":
        observed = power["observed_input_bindings"]
        value = (
            int(observed["repaired_gate2_mapping_consistent_wordings"])
            / int(observed["repaired_gate2_total_wordings"])
        )
    elif scenario == "improved":
        value = float(power["improved_mapping_consistency"])
    else:
        raise RuntimeError(f"Unknown mapping-consistency scenario: {scenario}")
    if not 0.0 < value <= 1.0:
        raise RuntimeError("Mapping consistency must lie in (0, 1].")
    return float(value)


def _event_intercept(
    *, event_rate: float, beta: float, rewrites: int, consistency: float
) -> float:
    quantiles = norm.ppf((np.arange(20001, dtype=float) + 0.5) / 20001.0)

    def objective(intercept: float) -> float:
        per_rewrite = expit(intercept + beta * quantiles)
        ever = 1.0 - np.power(1.0 - consistency * per_rewrite, rewrites)
        return float(ever.mean() - event_rate)

    return float(brentq(objective, -30.0, 30.0))


def _blend_for_target_gain(
    baseline: np.ndarray, oracle: np.ndarray, target_gain: float
) -> tuple[float, float, float, bool]:
    baseline = np.clip(baseline, 1e-8, 1 - 1e-8)
    oracle = np.clip(oracle, 1e-8, 1 - 1e-8)
    baseline_loss = float(
        np.mean(-(oracle * np.log(baseline) + (1 - oracle) * np.log1p(-baseline)))
    )

    def gain(blend: float) -> float:
        prediction = np.clip(baseline + blend * (oracle - baseline), 1e-8, 1 - 1e-8)
        loss = np.mean(
            -(oracle * np.log(prediction) + (1 - oracle) * np.log1p(-prediction))
        )
        return float(baseline_loss - loss)

    maximum = gain(1.0)
    if target_gain >= maximum:
        return 1.0, maximum, maximum, bool(np.isclose(target_gain, maximum))
    blend = float(brentq(lambda value: gain(value) - target_gain, 0.0, 1.0))
    achieved = gain(blend)
    return blend, achieved, maximum, True


def _cross_fitted_predictions(
    *,
    baseline_probability: np.ndarray,
    target_probability: np.ndarray,
    residual_predictor: np.ndarray,
    expected_information: float,
    folds: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulations, items = baseline_probability.shape
    order = rng.permutation(items)
    base_fold = np.empty(items, dtype=int)
    base_fold[order] = np.arange(items) % folds
    common_sd = 1.0 / np.sqrt(max(expected_information, 1.0))
    residual_variance = max(float(np.var(residual_predictor)), 1e-4)
    incremental_sd = 1.0 / np.sqrt(
        max(expected_information * residual_variance, 1.0)
    )
    baseline_logit = logit(np.clip(baseline_probability, 1e-8, 1 - 1e-8))
    target_logit = logit(np.clip(target_probability, 1e-8, 1 - 1e-8))
    baseline_sum = np.zeros_like(baseline_probability)
    target_sum = np.zeros_like(target_probability)
    fold_assignments = np.empty((repeats, items), dtype=int)
    for repeat in range(repeats):
        assignment = (base_fold + repeat) % folds
        fold_assignments[repeat] = assignment
        common_error = rng.normal(0.0, common_sd, size=(simulations, folds))
        incremental_error = rng.normal(
            0.0, incremental_sd, size=(simulations, folds)
        )
        selected_common = common_error[:, assignment]
        selected_incremental = incremental_error[:, assignment]
        baseline_sum += expit(baseline_logit + selected_common)
        target_sum += expit(
            target_logit
            + selected_common
            + selected_incremental * residual_predictor
        )
    return baseline_sum / repeats, target_sum / repeats, fold_assignments


def _loss_and_uncertainty(
    k_i: np.ndarray,
    n_i: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.clip(baseline, 1e-8, 1 - 1e-8)
    target = np.clip(target, 1e-8, 1 - 1e-8)
    baseline_loss = -(k_i * np.log(baseline) + (n_i - k_i) * np.log1p(-baseline))
    target_loss = -(k_i * np.log(target) + (n_i - k_i) * np.log1p(-target))
    contribution = baseline_loss - target_loss
    total_trials = np.maximum(n_i.sum(axis=1), 1)
    delta = contribution.sum(axis=1) / total_trials
    centered = contribution - delta[:, None] * n_i
    standard_error = np.sqrt(np.square(centered).sum(axis=1)) / total_trials
    return delta, delta - 1.96 * standard_error, delta + 1.96 * standard_error


def _calibration_instability(
    k_i: np.ndarray,
    n_i: np.ndarray,
    prediction: np.ndarray,
    fold_assignment: np.ndarray,
) -> np.ndarray:
    errors = []
    for fold in np.unique(fold_assignment):
        mask = fold_assignment == fold
        trials = n_i[:, mask].sum(axis=1)
        observed = k_i[:, mask].sum(axis=1) / np.maximum(trials, 1)
        expected = (n_i[:, mask] * prediction[:, mask]).sum(axis=1) / np.maximum(
            trials, 1
        )
        errors.append(observed - expected)
    return np.std(np.column_stack(errors), axis=1, ddof=0)


def simulate_grid(
    config: dict,
    *,
    event_rates: list[float],
    observed_predictors: pd.DataFrame,
    replicates: int | None = None,
) -> pd.DataFrame:
    """Run the frozen effect-size grid with original-item Monte Carlo inference.

    This is a prospective design diagnostic, not post-hoc power based on the
    observed null. Synthetic covariates retain the representative predictor
    correlation matrix. Requested held-out gains are interpolated between the
    baseline and the odds-ratio-defined oracle when feasible. Cross-fitting
    noise is approximated from training-fold Fisher information; uncertainty,
    folds, and calibration summaries remain clustered at original-item level.
    """
    power = config["power"]
    simulations = int(replicates or power["simulation_replicates"])
    if simulations < 2:
        raise RuntimeError("Power simulation requires at least two replicates.")
    correlation = observed_predictor_correlation(observed_predictors)
    baseline_correlation = correlation[:2, :2]
    cross_correlation = correlation[:2, 2]
    projection = np.linalg.solve(baseline_correlation, cross_correlation)
    residual_variance = max(
        1.0 - float(cross_correlation @ projection), 1e-6
    )
    method = power["simulation_method"]
    folds = int(method["grouped_outer_folds"])
    repeats = int(method["grouped_outer_repeats"])
    seed = int(config["inference"]["random_seed"])
    observed = power["observed_input_bindings"]
    eligible_yield = (
        int(observed["representative_eligible_yield_numerator"])
        / int(observed["representative_eligible_yield_denominator"])
    )
    rows: list[dict[str, Any]] = []
    base_combinations = itertools.product(
        power["eligible_originals"],
        power["accepted_rewrites_per_original"],
        event_rates,
        power["standardized_odds_ratio"],
        power["mapping_consistency"],
    )
    for n_items, rewrites, event_rate, odds_ratio, mapping in base_combinations:
        n_items = int(n_items)
        rewrites = int(rewrites)
        event_rate = float(event_rate)
        odds_ratio = float(odds_ratio)
        consistency = _mapping_consistency(config, str(mapping))
        rng = _scenario_rng(
            seed, (n_items, rewrites, event_rate, odds_ratio, mapping)
        )
        latent = rng.multivariate_normal(
            np.zeros(3), correlation, size=(simulations, n_items)
        )
        baseline_latent = latent[:, :, :2] @ projection
        full_latent = latent[:, :, 2]
        residual = full_latent - baseline_latent
        beta = float(np.log(odds_ratio))
        intercept = _event_intercept(
            event_rate=event_rate,
            beta=beta,
            rewrites=rewrites,
            consistency=consistency,
        )
        oracle_probability = expit(intercept + beta * full_latent)
        baseline_probability = expit(intercept + beta * baseline_latent)
        n_i = rng.binomial(rewrites, consistency, size=(simulations, n_items))
        k_i = rng.binomial(n_i, oracle_probability)
        expected_information = (
            0.8
            * n_items
            * rewrites
            * consistency
            * float(np.mean(oracle_probability * (1 - oracle_probability)))
        )
        for requested_gain in power["incremental_log_loss_gain"]:
            requested_gain = float(requested_gain)
            blend, expected_gain, maximum_gain, feasible = _blend_for_target_gain(
                baseline_probability, oracle_probability, requested_gain
            )
            target_probability = baseline_probability + blend * (
                oracle_probability - baseline_probability
            )
            baseline_cv, target_cv, fold_assignments = _cross_fitted_predictions(
                baseline_probability=baseline_probability,
                target_probability=target_probability,
                residual_predictor=residual,
                expected_information=expected_information,
                folds=folds,
                repeats=repeats,
                rng=rng,
            )
            delta, interval_low, interval_high = _loss_and_uncertainty(
                k_i, n_i, baseline_cv, target_cv
            )
            fold_assignment = fold_assignments[0]
            fold_positive = np.column_stack(
                [
                    k_i[:, fold_assignment == fold].sum(axis=1) > 0
                    for fold in range(folds)
                ]
            )
            calibration_instability = _calibration_instability(
                k_i, n_i, target_cv, fold_assignment
            )
            sorted_prediction = np.sort(target_cv, axis=1)
            rows.append(
                {
                    "eligible_originals": n_items,
                    "accepted_rewrites_per_original": rewrites,
                    "event_rate": event_rate,
                    "incremental_log_loss_gain": requested_gain,
                    "standardized_odds_ratio": odds_ratio,
                    "mapping_consistency_scenario": str(mapping),
                    "mapping_consistency_rate": consistency,
                    "simulation_replicates": simulations,
                    "expected_gain_before_crossfit_noise": expected_gain,
                    "maximum_gain_supported_by_odds_ratio": maximum_gain,
                    "requested_gain_feasible": feasible,
                    "mean_realized_cross_fitted_gain": float(np.mean(delta)),
                    "probability_interval_excludes_zero": float(
                        np.mean((interval_low > 0) | (interval_high < 0))
                    ),
                    "probability_interval_excludes_zero_favorable": float(
                        np.mean(interval_low > 0)
                    ),
                    "expected_flipping_originals": float(
                        np.mean((k_i > 0).sum(axis=1))
                    ),
                    "expected_mapping_consistent_rewrites": float(
                        np.mean(n_i.sum(axis=1))
                    ),
                    "no_positive_fold_probability": float(
                        np.mean(~fold_positive.all(axis=1))
                    ),
                    "calibration_instability": float(
                        np.mean(calibration_instability)
                    ),
                    "prediction_variance": float(
                        np.mean(np.var(sorted_prediction, axis=0, ddof=1))
                    ),
                    "total_candidate_originals_after_attrition": int(
                        np.ceil(n_items / eligible_yield)
                    ),
                    "eligible_yield": eligible_yield,
                    "geometry_native_confidence_correlation": float(
                        correlation[2, 0]
                    ),
                    "geometry_jury_correlation": float(correlation[2, 1]),
                    "incremental_predictor_residual_variance": residual_variance,
                    "inference_unit": "original_item",
                    "cross_validation_method": "5x5_grouped_crossfit_fisher_information_approximation",
                    "evidence_level": "prospective_design_simulation_not_observed_effect",
                }
            )
    result = pd.DataFrame(rows)
    expected_rows = (
        len(power["eligible_originals"])
        * len(power["accepted_rewrites_per_original"])
        * len(event_rates)
        * len(power["incremental_log_loss_gain"])
        * len(power["standardized_odds_ratio"])
        * len(power["mapping_consistency"])
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Frozen power grid expected {expected_rows} rows, observed {len(result)}."
        )
    return result
