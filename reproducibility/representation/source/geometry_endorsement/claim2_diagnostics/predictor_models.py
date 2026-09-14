from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GroupedFold:
    repeat: int
    fold: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def validate_grouped_splits(assignments: pd.DataFrame) -> None:
    required = {"base_item_id", "group_id", "repeat", "outer_fold"}
    if required - set(assignments.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(assignments.columns))}")
    if assignments.duplicated(["base_item_id", "repeat"]).any():
        raise RuntimeError("An item has multiple outer folds in one repeat.")
    crossed = assignments.groupby(["repeat", "group_id"])["outer_fold"].nunique().gt(1)
    if crossed.any():
        raise RuntimeError("A grouping unit crossed outer folds.")


def frozen_assignment_splits(items: pd.DataFrame, assignments: pd.DataFrame) -> Iterable[GroupedFold]:
    validate_grouped_splits(assignments)
    ids = set(items["base_item_id"].astype(str))
    for (repeat, fold), test in assignments.groupby(["repeat", "outer_fold"], sort=True):
        test_ids = tuple(sorted(set(test["base_item_id"].astype(str)) & ids))
        train_ids = tuple(sorted(ids - set(test_ids)))
        if not test_ids or not train_ids:
            raise RuntimeError("Frozen fold has an empty train or test partition.")
        yield GroupedFold(int(repeat), int(fold), train_ids, test_ids)


def make_grouped_splits(items: pd.DataFrame, folds: int, repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    y = items["k_i"].gt(0).astype(int).to_numpy()
    groups = items["group_id"].astype(str).to_numpy()
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (_, test) in enumerate(splitter.split(np.zeros(len(items)), y, groups)):
            for index in test:
                rows.append({"base_item_id": str(items.iloc[index]["base_item_id"]), "group_id": groups[index], "repeat": repeat, "outer_fold": fold})
    result = pd.DataFrame(rows)
    validate_grouped_splits(result)
    return result


def fit_low_dimensional_fold(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], c_value: float) -> np.ndarray:
    successes = train["k_i"].to_numpy(dtype=int)
    trials = train["n_i"].to_numpy(dtype=int)
    if not columns:
        return np.full(len(test), (successes.sum() + 0.5) / (trials.sum() + 1.0), dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].to_numpy(dtype=float))
    x_test = scaler.transform(test[columns].to_numpy(dtype=float))
    expanded_x = np.repeat(x_train, trials, axis=0)
    expanded_y = np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for k, n in zip(successes, trials, strict=True)])
    if len(np.unique(expanded_y)) < 2:
        return np.full(len(test), (successes.sum() + 0.5) / (trials.sum() + 1.0), dtype=float)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=4000, random_state=0)
    model.fit(expanded_x, expanded_y)
    return model.predict_proba(x_test)[:, 1]


def item_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    k = frame["k_i"].to_numpy(dtype=int)
    n = frame["n_i"].to_numpy(dtype=int)
    p = np.clip(frame["predicted_probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    observed = k / n
    ever = (k > 0).astype(int)
    return {
        "items": len(frame),
        "accepted_rewrites": int(n.sum()),
        "binomial_log_loss_per_rewrite": float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum()),
        "weighted_brier": float(np.average((observed - p) ** 2, weights=n)),
        "auroc_ever_flip": float(roc_auc_score(ever, p)) if len(np.unique(ever)) > 1 else None,
        "auprc_ever_flip": float(average_precision_score(ever, p)) if len(np.unique(ever)) > 1 else None,
        "spearman_susceptibility": float(spearmanr(observed, p).statistic) if len(np.unique(observed)) > 1 else None,
    }


def item_bootstrap_metric_delta(left: pd.DataFrame, right: pd.DataFrame, replicates: int, seed: int) -> np.ndarray:
    merged = left.merge(right, on="base_item_id", suffixes=("_left", "_right"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = merged.iloc[rng.integers(0, len(merged), len(merged))]
        k = sample["k_i_left"].to_numpy(dtype=int)
        n = sample["n_i_left"].to_numpy(dtype=int)
        losses = []
        for suffix in ("left", "right"):
            p = np.clip(sample[f"predicted_probability_{suffix}"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
            losses.append(float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum()))
        values[index] = losses[0] - losses[1]
    return values


def _expand_trials(x: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(x, n, axis=0), np.concatenate(
        [np.r_[np.ones(int(successes)), np.zeros(int(trials - successes))] for successes, trials in zip(k, n, strict=True)]
    ).astype(int)


def _transform_design(
    train_text: np.ndarray,
    test_text: np.ndarray,
    train_scalar: np.ndarray,
    test_scalar: np.ndarray,
    pca_dimensions: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    if train_text.shape[1]:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_text)
        test_scaled = scaler.transform(test_text)
        maximum = min(train_scaled.shape[0] - 1, train_scaled.shape[1])
        components = min(int(pca_dimensions or maximum), maximum)
        if components < 1:
            raise RuntimeError("Training fold is too small for text PCA.")
        pca = PCA(n_components=components, random_state=0)
        train_parts.append(pca.fit_transform(train_scaled))
        test_parts.append(pca.transform(test_scaled))
    if train_scalar.shape[1]:
        scaler = StandardScaler()
        train_parts.append(scaler.fit_transform(train_scalar))
        test_parts.append(scaler.transform(test_scalar))
    if not train_parts:
        return np.empty((len(train_text), 0)), np.empty((len(test_text), 0))
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def _predict_binomial(train_x: np.ndarray, train: pd.DataFrame, test_x: np.ndarray, c_value: float) -> np.ndarray:
    k = train["k_i"].to_numpy(dtype=int)
    n = train["n_i"].to_numpy(dtype=int)
    if train_x.shape[1] == 0:
        return np.full(len(test_x), (k.sum() + 0.5) / (n.sum() + 1.0), dtype=float)
    expanded_x, expanded_y = _expand_trials(train_x, k, n)
    if len(np.unique(expanded_y)) < 2:
        return np.full(len(test_x), (k.sum() + 0.5) / (n.sum() + 1.0), dtype=float)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=4000, random_state=0)
    model.fit(expanded_x, expanded_y)
    return model.predict_proba(test_x)[:, 1]


def _binomial_loss(frame: pd.DataFrame, probability: np.ndarray) -> float:
    k = frame["k_i"].to_numpy(dtype=int)
    n = frame["n_i"].to_numpy(dtype=int)
    p = np.clip(probability, 1e-8, 1 - 1e-8)
    return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum())


def _inner_splits(train: pd.DataFrame, folds: int, seed: int):
    y = train["k_i"].gt(0).astype(int).to_numpy()
    groups = train["consideration_cluster_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise RuntimeError("Too few consideration groups for frozen inner folds.")
    splitter = (
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        if len(np.unique(y)) > 1
        else GroupKFold(n_splits=folds)
    )
    return list(splitter.split(np.zeros(len(train)), y, groups))


def run_frozen_grouped_ladder(
    items: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    model_contract: dict[str, list[str]],
    *,
    c_grid: list[float],
    pca_grid: list[int],
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the low-dimensional ladder on exact stored outer assignments."""
    ids = items["base_item_id"].astype(str).tolist()
    if len(set(ids)) != len(ids) or len(embeddings) != len(items):
        raise RuntimeError("Item IDs/embeddings are not unique and aligned.")
    id_to_index = {value: index for index, value in enumerate(ids)}
    prediction_rows: list[dict] = []
    choice_rows: list[dict] = []
    for (repeat, fold), fold_rows in assignments.groupby(["repeat", "outer_fold"], sort=True):
        test_ids = set(fold_rows["base_item_id"].astype(str))
        test_index = np.asarray([id_to_index[value] for value in ids if value in test_ids], dtype=int)
        train_index = np.asarray([id_to_index[value] for value in ids if value not in test_ids], dtype=int)
        train = items.iloc[train_index].reset_index(drop=True)
        test = items.iloc[test_index].reset_index(drop=True)
        seed_values = fold_rows["seed"].astype(int).unique()
        if len(seed_values) != 1:
            raise RuntimeError("Frozen outer fold has inconsistent seeds.")
        seed = int(seed_values[0])
        inner = _inner_splits(train, inner_folds, seed + int(fold) + 1000)
        for model_name, predictors in model_contract.items():
            uses_text = "original_text" in predictors
            scalar_columns = [value for value in predictors if value != "original_text"]
            candidates = [(1.0, None)] if not predictors else [
                (float(c), int(pca) if uses_text else None)
                for c in c_grid
                for pca in (pca_grid if uses_text else [None])
            ]
            trials = []
            for c_value, pca_value in candidates:
                total_nll = 0.0
                total_trials = 0
                for inner_train, inner_valid in inner:
                    train_text = embeddings[train_index][inner_train] if uses_text else np.empty((len(inner_train), 0))
                    valid_text = embeddings[train_index][inner_valid] if uses_text else np.empty((len(inner_valid), 0))
                    train_scalar = train.iloc[inner_train][scalar_columns].to_numpy(dtype=float) if scalar_columns else np.empty((len(inner_train), 0))
                    valid_scalar = train.iloc[inner_valid][scalar_columns].to_numpy(dtype=float) if scalar_columns else np.empty((len(inner_valid), 0))
                    x_train, x_valid = _transform_design(train_text, valid_text, train_scalar, valid_scalar, pca_value)
                    probability = _predict_binomial(x_train, train.iloc[inner_train], x_valid, c_value)
                    valid_frame = train.iloc[inner_valid]
                    trials_in_fold = int(valid_frame["n_i"].sum())
                    total_nll += _binomial_loss(valid_frame, probability) * trials_in_fold
                    total_trials += trials_in_fold
                trials.append({"regularization_c": c_value, "pca_dimensions": pca_value, "inner_log_loss": float(total_nll / total_trials)})
            best = min(trials, key=lambda row: (row["inner_log_loss"], -1 if row["pca_dimensions"] is None else row["pca_dimensions"], row["regularization_c"]))
            train_text = embeddings[train_index] if uses_text else np.empty((len(train_index), 0))
            test_text = embeddings[test_index] if uses_text else np.empty((len(test_index), 0))
            train_scalar = train[scalar_columns].to_numpy(dtype=float) if scalar_columns else np.empty((len(train), 0))
            test_scalar = test[scalar_columns].to_numpy(dtype=float) if scalar_columns else np.empty((len(test), 0))
            x_train, x_test = _transform_design(train_text, test_text, train_scalar, test_scalar, best["pca_dimensions"])
            probability = _predict_binomial(x_train, train, x_test, best["regularization_c"])
            choice_rows.append({"repeat": int(repeat), "outer_fold": int(fold), "seed": seed, "model": model_name, **best})
            for position, row in test.iterrows():
                prediction_rows.append({
                    "repeat": int(repeat), "outer_fold": int(fold), "seed": seed, "model": model_name,
                    "base_item_id": str(row["base_item_id"]), "k_i": int(row["k_i"]), "n_i": int(row["n_i"]),
                    "predicted_probability": float(probability[position]),
                })
    predictions = pd.DataFrame(prediction_rows)
    expected = len(items) * assignments["repeat"].nunique() * len(model_contract)
    if len(predictions) != expected:
        raise RuntimeError(f"Expected {expected} held-out predictions, observed {len(predictions)}.")
    return predictions, pd.DataFrame(choice_rows)


def summarize_ladder(predictions: pd.DataFrame, *, bootstrap_replicates: int, permutation_replicates: int, seed: int) -> pd.DataFrame:
    from claim2_pilot.analysis import _model_metrics

    averaged = predictions.groupby(["base_item_id", "model"], as_index=False).agg(
        k_i=("k_i", "first"), n_i=("n_i", "first"),
        predicted_probability=("predicted_probability", "mean"),
        prediction_sd_across_repeats=("predicted_probability", "std"),
    )
    baseline_m0 = averaged[averaged["model"].eq("M0")]
    baseline_m2 = averaged[averaged["model"].eq("M2")]
    baseline_m0_loss = _model_metrics(baseline_m0)["binomial_log_loss_per_rephrasing"]
    baseline_m2_loss = _model_metrics(baseline_m2)["binomial_log_loss_per_rephrasing"]
    rng = np.random.default_rng(seed)
    rows = []
    for model_name, frame in averaged.groupby("model", sort=False):
        metrics = _model_metrics(frame)
        boot_m0 = item_bootstrap_metric_delta(baseline_m0, frame, bootstrap_replicates, seed + sum(map(ord, model_name)))
        boot_m2 = item_bootstrap_metric_delta(baseline_m2, frame, bootstrap_replicates, seed + 1000 + sum(map(ord, model_name)))
        observed = frame["k_i"].to_numpy(dtype=float) / frame["n_i"].to_numpy(dtype=float)
        predicted = frame["predicted_probability"].to_numpy(dtype=float)
        statistic = float(spearmanr(observed, predicted).statistic) if len(np.unique(observed)) > 1 else 0.0
        null = np.asarray([spearmanr(rng.permutation(observed), predicted).statistic for _ in range(permutation_replicates)], dtype=float)
        failures = predictions[predictions["model"].eq(model_name)].groupby(["repeat", "outer_fold"])["k_i"].sum()
        rows.append({
            "model": model_name, **metrics,
            "delta_log_loss_vs_M0": float(baseline_m0_loss - metrics["binomial_log_loss_per_rephrasing"]),
            "delta_vs_M0_ci_low": float(np.quantile(boot_m0, .025)),
            "delta_vs_M0_ci_high": float(np.quantile(boot_m0, .975)),
            "delta_log_loss_vs_M2": float(baseline_m2_loss - metrics["binomial_log_loss_per_rephrasing"]),
            "delta_vs_M2_ci_low": float(np.quantile(boot_m2, .025)),
            "delta_vs_M2_ci_high": float(np.quantile(boot_m2, .975)),
            "item_permutation_p_two_sided": float((1 + (np.abs(null) >= abs(statistic)).sum()) / (permutation_replicates + 1)),
            "mean_item_prediction_sd": float(frame["prediction_sd_across_repeats"].mean()),
            "maximum_item_prediction_sd": float(frame["prediction_sd_across_repeats"].max()),
            "no_positive_test_fold_count": int(failures.eq(0).sum()),
            "fold_count": int(len(failures)),
        })
    return pd.DataFrame(rows)
