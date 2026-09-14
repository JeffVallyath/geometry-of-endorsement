from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .config import Claim2Config
from .util import (
    canonical_json_bytes,
    immutable_json,
    immutable_write,
    sha256_bytes,
    sha256_file,
)


EPS = 1e-7
PRIMARY_MODELS = ("M0", "M1", "M2", "M3", "M4")


@dataclass(frozen=True)
class FoldTransform:
    text_scaler: StandardScaler | None
    text_pca: PCA | None
    scalar_scaler: StandardScaler | None


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def semantic_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"Situation and action: {situation}\nNamed consideration: {consideration}"
        for situation, consideration in zip(
            frame["situation_action_text"].astype(str),
            frame["consideration_text"].astype(str),
            strict=True,
        )
    ]


def encode_original_text(
    config: Claim2Config,
    items: pd.DataFrame,
    output_path: str | Path,
) -> dict:
    """Encode original semantic fields once with the frozen encoder."""
    from sentence_transformers import SentenceTransformer

    prediction = config.section("prediction")
    model = SentenceTransformer(
        prediction["text_encoder"],
        revision=prediction["text_encoder_revision"],
    )
    texts = semantic_text(items)
    embedding = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    target = Path(output_path)
    if target.exists():
        raise RuntimeError(f"Text embedding artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            base_item_id=np.asarray(
                items["base_item_id"].astype(str).tolist(), dtype=str
            ),
            embedding=embedding,
        )
    metadata = {
        "schema_version": 1,
        "status": "ORIGINAL_TEXT_EMBEDDINGS_FROZEN",
        "config_sha256": config.digest,
        "semantic_input_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "base_item_id": items["base_item_id"].astype(str).tolist(),
                    "semantic_text": texts,
                }
            )
        ),
        "encoder": prediction["text_encoder"],
        "revision": prediction["text_encoder_revision"],
        "input": "original situation/action plus named consideration",
        "rows": len(items),
        "dimensions": int(embedding.shape[1]),
        "artifact_sha256": sha256_file(target),
        "rephrased_text_used": False,
        "outcome_used": False,
    }
    immutable_json(target.with_suffix(".metadata.json"), metadata)
    return metadata


def load_text_embeddings(
    path: str | Path,
    items: pd.DataFrame,
    config: Claim2Config,
) -> np.ndarray:
    target = Path(path)
    metadata = json.loads(
        target.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    if sha256_file(target) != metadata["artifact_sha256"]:
        raise RuntimeError("Frozen text embedding hash mismatch.")
    expected_input_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "base_item_id": items["base_item_id"].astype(str).tolist(),
                "semantic_text": semantic_text(items),
            }
        )
    )
    prediction = config.section("prediction")
    expected_metadata = {
        "config_sha256": config.digest,
        "semantic_input_sha256": expected_input_sha256,
        "encoder": prediction["text_encoder"],
        "revision": prediction["text_encoder_revision"],
        "rows": len(items),
        "rephrased_text_used": False,
        "outcome_used": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Frozen text embedding metadata mismatch: {mismatches}"
        )
    with np.load(target, allow_pickle=False) as z:
        ids = list(map(str, z["base_item_id"]))
        embedding = z["embedding"].astype(np.float32)
    index = {value: i for i, value in enumerate(ids)}
    requested = items["base_item_id"].astype(str).tolist()
    if len(index) != len(ids) or set(requested) - set(index):
        raise RuntimeError("Text embeddings do not cover unique base items.")
    return embedding[[index[value] for value in requested]]

def _splitter(
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(groups)) < folds:
        raise RuntimeError("Too few consideration groups for grouped CV.")
    if len(np.unique(y)) > 1:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        return splitter.split(np.zeros(len(y)), y, groups)
    splitter = GroupKFold(n_splits=folds)
    return splitter.split(np.zeros(len(y)), y, groups)


def _expand_binomial(
    x: np.ndarray, k: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for features, successes, trials in zip(x, k, n, strict=True):
        successes = int(successes)
        trials = int(trials)
        rows.extend([features] * trials)
        labels.extend([1] * successes)
        labels.extend([0] * (trials - successes))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=int)


def _binomial_nll(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum())


def _scalar_columns(model_name: str) -> list[str]:
    if model_name == "M0":
        return []
    if model_name == "M1":
        return ["native_confidence"]
    if model_name == "M2":
        return ["native_confidence"]
    if model_name == "M3":
        return ["native_confidence", "dim_weakness"]
    if model_name == "M4":
        return [
            "native_confidence",
            "dim_weakness",
            "standardized_signed_dim_margin",
            "standardized_logistic_margin",
        ]
    raise ValueError(f"Unknown prediction model: {model_name}")


def _uses_text(model_name: str) -> bool:
    return model_name in {"M2", "M3", "M4"}


def _fit_transform(
    train_text: np.ndarray,
    test_text: np.ndarray,
    train_scalar: np.ndarray,
    test_scalar: np.ndarray,
    pca_dimensions: int | None,
) -> tuple[np.ndarray, np.ndarray, FoldTransform]:
    pieces_train: list[np.ndarray] = []
    pieces_test: list[np.ndarray] = []
    text_scaler = None
    text_pca = None
    scalar_scaler = None
    if train_text.shape[1]:
        text_scaler = StandardScaler()
        train_scaled = text_scaler.fit_transform(train_text)
        test_scaled = text_scaler.transform(test_text)
        maximum = min(train_scaled.shape[0] - 1, train_scaled.shape[1])
        components = min(int(pca_dimensions or maximum), maximum)
        if components < 1:
            raise RuntimeError("Training fold is too small for text PCA.")
        text_pca = PCA(n_components=components, random_state=0)
        pieces_train.append(text_pca.fit_transform(train_scaled))
        pieces_test.append(text_pca.transform(test_scaled))
    if train_scalar.shape[1]:
        scalar_scaler = StandardScaler()
        pieces_train.append(scalar_scaler.fit_transform(train_scalar))
        pieces_test.append(scalar_scaler.transform(test_scalar))
    return (
        np.concatenate(pieces_train, axis=1),
        np.concatenate(pieces_test, axis=1),
        FoldTransform(text_scaler, text_pca, scalar_scaler),
    )


def _fit_predict(
    x_train: np.ndarray,
    k_train: np.ndarray,
    n_train: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
) -> np.ndarray:
    expanded_x, expanded_y = _expand_binomial(x_train, k_train, n_train)
    if len(np.unique(expanded_y)) < 2:
        mean = (k_train.sum() + 0.5) / (n_train.sum() + 1.0)
        return np.full(len(x_test), mean, dtype=float)
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=4000,
        random_state=0,
    )
    model.fit(expanded_x, expanded_y)
    return model.predict_proba(x_test)[:, 1]


def _design(
    items: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    text = embeddings if _uses_text(model_name) else np.empty((len(items), 0))
    columns = _scalar_columns(model_name)
    scalar = (
        items[columns].astype(float).to_numpy()
        if columns
        else np.empty((len(items), 0))
    )
    return text, scalar


def _select_hyperparameters(
    config: Claim2Config,
    items: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
    groups: np.ndarray,
    seed: int,
) -> tuple[float, int | None, list[dict]]:
    if model_name == "M0":
        return 1.0, None, []
    prediction = config.section("prediction")
    c_grid = [float(value) for value in prediction["regularization_c"]]
    pca_grid = (
        [int(value) for value in prediction["pca_dimensions"]]
        if _uses_text(model_name)
        else [None]
    )
    y = items["k_i"].gt(0).astype(int).to_numpy()
    splits = list(
        _splitter(y, groups, int(prediction["inner_folds"]), seed)
    )
    text, scalar = _design(items, embeddings, model_name)
    trials: list[dict] = []
    for pca_dim in pca_grid:
        for c_value in c_grid:
            nll = 0.0
            denominator = 0
            for train, valid in splits:
                train_x, valid_x, _ = _fit_transform(
                    text[train], text[valid], scalar[train], scalar[valid], pca_dim
                )
                probability = _fit_predict(
                    train_x,
                    items.iloc[train]["k_i"].to_numpy(),
                    items.iloc[train]["n_i"].to_numpy(),
                    valid_x,
                    c_value,
                )
                nll += _binomial_nll(
                    items.iloc[valid]["k_i"].to_numpy(),
                    items.iloc[valid]["n_i"].to_numpy(),
                    probability,
                )
                denominator += int(items.iloc[valid]["n_i"].sum())
            trials.append(
                {
                    "pca_dimensions": pca_dim,
                    "regularization_c": c_value,
                    "inner_log_loss": nll / denominator,
                }
            )
    best = min(
        trials,
        key=lambda row: (
            row["inner_log_loss"],
            -1 if row["pca_dimensions"] is None else row["pca_dimensions"],
            row["regularization_c"],
        ),
    )
    return (
        float(best["regularization_c"]),
        best["pca_dimensions"],
        trials,
    )


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    predictor = logit(np.clip(probability, EPS, 1 - EPS)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000)
    model.fit(predictor, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= left) & (
            (probability <= right) if right == 1 else (probability < right)
        )
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(total)


def _model_metrics(frame: pd.DataFrame) -> dict:
    k = frame["k_i"].to_numpy(dtype=int)
    n = frame["n_i"].to_numpy(dtype=int)
    p = frame["predicted_probability"].to_numpy(dtype=float)
    observed = k / n
    ever = (k > 0).astype(int)
    calibration_intercept, calibration_slope = _calibration(ever, p)
    saturated = _binomial_nll(k, n, np.clip(observed, EPS, 1 - EPS))
    nll = _binomial_nll(k, n, p)
    return {
        "items": len(frame),
        "accepted_rephrasings": int(n.sum()),
        "binomial_log_loss_per_rephrasing": nll / n.sum(),
        "held_out_deviance": 2 * (nll - saturated),
        "weighted_brier": float(np.average((observed - p) ** 2, weights=n)),
        "calibration_intercept_ever_flip": calibration_intercept,
        "calibration_slope_ever_flip": calibration_slope,
        "ece_ever_flip_5_bins": _ece(ever, p),
        "auroc_ever_flip": (
            float(roc_auc_score(ever, p)) if len(np.unique(ever)) > 1 else None
        ),
        "auprc_ever_flip": (
            float(average_precision_score(ever, p))
            if len(np.unique(ever)) > 1
            else None
        ),
        "spearman_susceptibility": (
            float(spearmanr(observed, p).statistic)
            if len(np.unique(observed)) > 1
            else None
        ),
        "ece_caveat": "Five-bin item-level diagnostic; unstable in a small pilot.",
    }


def _paired_bootstrap(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    averaged = predictions.groupby(["base_item_id", "model"], as_index=False).agg(
        k_i=("k_i", "first"),
        n_i=("n_i", "first"),
        predicted_probability=("predicted_probability", "mean"),
    )
    wide = averaged.pivot(index="base_item_id", columns="model")
    required = {"M2", "M3"}
    if not required <= set(wide["predicted_probability"].columns):
        raise RuntimeError("Paired bootstrap requires M2 and M3 predictions.")
    ids = wide.index.to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        sample = wide.loc[sampled]
        k = sample["k_i"]["M2"].to_numpy(dtype=int)
        n = sample["n_i"]["M2"].to_numpy(dtype=int)
        loss_m2 = _binomial_nll(
            k, n, sample["predicted_probability"]["M2"].to_numpy()
        ) / n.sum()
        loss_m3 = _binomial_nll(
            k, n, sample["predicted_probability"]["M3"].to_numpy()
        ) / n.sum()
        deltas[index] = loss_m2 - loss_m3
    point_m2 = averaged[averaged["model"] == "M2"]
    point_m3 = averaged[averaged["model"] == "M3"]
    merged = point_m2.merge(point_m3, on="base_item_id", suffixes=("_m2", "_m3"))
    point = (
        _binomial_nll(merged["k_i_m2"].to_numpy(), merged["n_i_m2"].to_numpy(), merged["predicted_probability_m2"].to_numpy())
        - _binomial_nll(merged["k_i_m3"].to_numpy(), merged["n_i_m3"].to_numpy(), merged["predicted_probability_m3"].to_numpy())
    ) / merged["n_i_m2"].sum()
    return deltas, {
        "point_estimate": float(point),
        "interval_95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "replicates": replicates,
        "resampling_unit": "base_item",
        "positive_means_geometry_improves": True,
    }


def _calibration_bins_frame(averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, 6)
    for model_name, frame in averaged.groupby("model"):
        probability = frame["predicted_probability"].to_numpy()
        ever = frame["k_i"].gt(0).astype(int).to_numpy()
        for bin_index, (left, right) in enumerate(
            zip(edges[:-1], edges[1:], strict=True)
        ):
            mask = (probability >= left) & (
                (probability <= right)
                if right == 1
                else (probability < right)
            )
            rows.append(
                {
                    "model": model_name,
                    "bin": bin_index,
                    "left": left,
                    "right": right,
                    "n": int(mask.sum()),
                    "mean_predicted_probability": (
                        float(probability[mask].mean())
                        if mask.any()
                        else None
                    ),
                    "observed_ever_flip_rate": (
                        float(ever[mask].mean()) if mask.any() else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _calibration_plot_bytes(bins: pd.DataFrame) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    for model_name, frame in bins.groupby("model"):
        valid = frame.dropna(
            subset=[
                "mean_predicted_probability",
                "observed_ever_flip_rate",
            ]
        )
        axis.plot(
            valid["mean_predicted_probability"],
            valid["observed_ever_flip_rate"],
            marker="o",
            label=model_name,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="Observed ever-flip rate",
        title="Development calibration diagnostic",
    )
    axis.legend()
    figure.tight_layout()
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=160)
    plt.close(figure)
    return stream.getvalue()


def run_nested_prediction(
    config: Claim2Config,
    item_susceptibility: pd.DataFrame,
    text_embeddings: np.ndarray,
    output_dir: str | Path,
) -> dict:
    prediction = config.section("prediction")
    items = item_susceptibility[
        item_susceptibility["stratum"].eq(prediction["primary_stratum"])
        & item_susceptibility["n_i"].gt(0)
    ].reset_index(drop=True)
    if len(items) < 10:
        raise RuntimeError("Too few representative items with accepted rephrasings.")
    all_ids = item_susceptibility["base_item_id"].astype(str).tolist()
    embedding_index = {value: i for i, value in enumerate(all_ids)}
    embeddings = text_embeddings[[embedding_index[value] for value in items["base_item_id"].astype(str)]]
    groups = items["consideration_cluster_id"].astype(str).to_numpy()
    y = items["k_i"].gt(0).astype(int).to_numpy()
    predictions: list[dict] = []
    choices: list[dict] = []
    assignments: list[dict] = []
    for repeat, seed in enumerate(prediction["seeds"]):
        outer = list(_splitter(y, groups, int(prediction["outer_folds"]), int(seed)))
        for fold, (train, test) in enumerate(outer):
            train_items = items.iloc[train].reset_index(drop=True)
            train_embeddings = embeddings[train]
            for item_index in test:
                assignments.append({
                    "repeat": repeat,
                    "seed": int(seed),
                    "outer_fold": fold,
                    "base_item_id": str(items.iloc[item_index]["base_item_id"]),
                    "consideration_cluster_id": str(items.iloc[item_index]["consideration_cluster_id"]),
                })
            for model_name in PRIMARY_MODELS:
                if model_name == "M0":
                    probability = np.full(
                        len(test),
                        (train_items["k_i"].sum() + 0.5) / (train_items["n_i"].sum() + 1.0),
                    )
                    best_c, best_pca, trials = 1.0, None, []
                    preprocessing = "intercept_only"
                else:
                    best_c, best_pca, trials = _select_hyperparameters(
                        config,
                        train_items,
                        train_embeddings,
                        model_name,
                        train_items["consideration_cluster_id"].astype(str).to_numpy(),
                        int(seed) + fold + 1000,
                    )
                    train_text, train_scalar = _design(train_items, train_embeddings, model_name)
                    test_text, test_scalar = _design(items.iloc[test], embeddings[test], model_name)
                    train_x, test_x, transform = _fit_transform(
                        train_text, test_text, train_scalar, test_scalar, best_pca
                    )
                    probability = _fit_predict(
                        train_x,
                        train_items["k_i"].to_numpy(),
                        train_items["n_i"].to_numpy(),
                        test_x,
                        best_c,
                    )
                    preprocessing = {
                        "text_scaler_fit_rows": len(train) if transform.text_scaler else 0,
                        "text_pca_fit_rows": len(train) if transform.text_pca else 0,
                        "scalar_scaler_fit_rows": len(train) if transform.scalar_scaler else 0,
                        "test_rows_seen_during_fit": 0,
                    }
                choices.append({
                    "repeat": repeat,
                    "outer_fold": fold,
                    "model": model_name,
                    "regularization_c": best_c,
                    "pca_dimensions": best_pca,
                    "inner_trials": trials,
                    "preprocessing": preprocessing,
                })
                for position, item_index in enumerate(test):
                    row = items.iloc[item_index]
                    predictions.append({
                        "repeat": repeat,
                        "seed": int(seed),
                        "outer_fold": fold,
                        "model": model_name,
                        "base_item_id": str(row["base_item_id"]),
                        "board_id": str(row["board_id"]),
                        "consideration_cluster_id": str(row["consideration_cluster_id"]),
                        "k_i": int(row["k_i"]),
                        "n_i": int(row["n_i"]),
                        "observed_susceptibility": float(row["k_i"] / row["n_i"]),
                        "predicted_probability": float(probability[position]),
                    })
    pred = pd.DataFrame(predictions)
    assignment_frame = pd.DataFrame(assignments)
    leakage = assignment_frame.groupby(["repeat", "consideration_cluster_id"])["outer_fold"].nunique()
    if (leakage > 1).any():
        raise RuntimeError("A consideration identity crossed outer folds.")
    averaged = pred.groupby(["base_item_id", "model"], as_index=False).agg(
        k_i=("k_i", "first"),
        n_i=("n_i", "first"),
        predicted_probability=("predicted_probability", "mean"),
    )
    metrics = {
        model_name: _model_metrics(averaged[averaged["model"] == model_name])
        for model_name in PRIMARY_MODELS
    }
    calibration_bins = _calibration_bins_frame(averaged)
    deltas, delta_summary = _paired_bootstrap(
        pred,
        int(prediction["bootstrap_replicates"]),
        int(config.section("run")["seed"]) + 9000,
    )
    target = Path(output_dir)
    calibration_bins_hash = immutable_write(
        target / "calibration_bins.csv",
        _csv_bytes(calibration_bins),
    )
    calibration_plot_hash = immutable_write(
        target / "calibration_plot.png",
        _calibration_plot_bytes(calibration_bins),
    )
    prediction_hash = immutable_write(target / "nested_predictions_private.csv", _csv_bytes(pred))
    assignment_hash = immutable_write(target / "outer_fold_assignments.csv", _csv_bytes(assignment_frame))
    choice_hash = immutable_json(target / "nested_hyperparameters.json", choices)
    bootstrap_hash = immutable_write(
        target / "m2_m3_base_item_bootstrap.npy",
        _npy_bytes(deltas),
    )
    report = {
        "schema_version": 1,
        "status": "PRIMARY_PREDICTION_COMPLETE",
        "primary_stratum": prediction["primary_stratum"],
        "models": metrics,
        "delta_log_loss_m2_minus_m3": delta_summary,
        "nested_predictions_sha256": prediction_hash,
        "calibration_bins_sha256": calibration_bins_hash,
        "calibration_plot_sha256": calibration_plot_hash,
        "fold_assignments_sha256": assignment_hash,
        "hyperparameters_sha256": choice_hash,
        "bootstrap_sha256": bootstrap_hash,
        "all_paraphrases_from_item_share_fold": True,
        "consideration_clusters_grouped": True,
        "preprocessing_fit_on_training_only": True,
        "rephrased_text_used_as_predictor": False,
        "confirmatory_claim": False,
    }
    immutable_json(target / "prediction_summary.json", report)
    return report


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()
