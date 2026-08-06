from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


LOGGER = logging.getLogger("truth_control_v2")


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _macro_dataset_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    datasets: np.ndarray,
) -> tuple[float, dict[str, float]]:
    per_dataset = {
        str(name): _auroc(labels[datasets == name], scores[datasets == name])
        for name in sorted(np.unique(datasets))
    }
    finite = [value for value in per_dataset.values() if not math.isnan(value)]
    return (float(np.mean(finite)) if finite else float("nan")), per_dataset


def _score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    datasets: np.ndarray,
) -> dict[str, Any]:
    macro, per_dataset = _macro_dataset_auroc(labels, scores, datasets)
    return {
        "macro_dataset_auroc": macro,
        "overall_auroc": _auroc(labels, scores),
        "per_dataset_auroc": per_dataset,
        "accuracy_at_zero": float(accuracy_score(labels, scores >= 0)),
        "mean_score_true": float(np.mean(scores[labels == 1])),
        "mean_score_false": float(np.mean(scores[labels == 0])),
    }


def native_answer_controls(
    examples: pd.DataFrame,
    truth_scores: np.ndarray,
) -> dict[str, Any]:
    labels = examples["label"].to_numpy(dtype=np.int8)
    datasets = examples["dataset"].to_numpy()
    result: dict[str, Any] = {}
    for scheme in ("primary", "transfer"):
        scheme_result: dict[str, Any] = {}
        for split in ("dev", "test"):
            split_result: dict[str, Any] = {}
            base = (examples["scheme"].to_numpy() == scheme) & (
                examples["split"].to_numpy() == split
            )
            split_result["overall"] = _score_metrics(
                labels[base], truth_scores[base], datasets[base]
            )
            for mapping in ("standard", "reversed"):
                mask = base & (examples["mapping"].to_numpy() == mapping)
                split_result[mapping] = _score_metrics(
                    labels[mask], truth_scores[mask], datasets[mask]
                )
            scheme_result[split] = split_result
        result[scheme] = scheme_result
    return result


def assign_training_partitions(
    examples: pd.DataFrame,
    partitions: int,
    seed: int,
) -> dict[str, int]:
    mask = (examples["scheme"] == "primary") & (examples["split"] == "train")
    groups = sorted(examples.loc[mask, "group_id"].astype(str).unique())
    if len(groups) < partitions:
        raise RuntimeError("Fewer training groups than frozen training partitions.")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:truth-v2-partition:{group}".encode()
        ).hexdigest(),
    )
    assignment = {group: index % partitions for index, group in enumerate(ordered)}
    for partition in range(partitions):
        frame = examples.loc[
            mask & (examples["group_id"].astype(str).map(assignment) == partition)
        ]
        if set(frame["label"]) != {0, 1}:
            raise RuntimeError(f"Training partition {partition} lacks one semantic label.")
        if set(frame["mapping"]) != {"standard", "reversed"}:
            raise RuntimeError(f"Training partition {partition} lacks one A/B mapping orientation.")
    return assignment


def _dim_fit_with_scale(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    positive = features[labels == 1]
    negative = features[labels == 0]
    if not len(positive) or not len(negative):
        raise RuntimeError("DIM requires both labels.")
    mean_positive = positive.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_negative = negative.mean(axis=0, dtype=np.float64).astype(np.float32)
    direction = mean_positive - mean_negative
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("DIM direction has zero or non-finite norm.")
    direction = direction / norm
    midpoint = float(0.5 * (mean_positive + mean_negative) @ direction)
    scores = features @ direction - midpoint
    scale = float(np.std(scores, ddof=0))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("Training projection scale is zero or non-finite.")
    return direction.astype(np.float32), midpoint, scale


def _primary_train_rows(examples: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(
        (examples["scheme"].to_numpy() == "primary")
        & (examples["split"].to_numpy() == "train")
    )


def fit_partition_directions(
    layer_features: np.ndarray,
    examples: pd.DataFrame,
    train_rows: np.ndarray,
    train_labels: np.ndarray,
    assignment: dict[str, int],
    partitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = examples.iloc[train_rows]["group_id"].astype(str).to_numpy()
    partition_ids = np.asarray([assignment[group] for group in groups], dtype=np.int16)
    directions: list[np.ndarray] = []
    midpoints: list[float] = []
    scales: list[float] = []
    train_x = layer_features[train_rows].astype(np.float32)
    for partition in range(partitions):
        mask = partition_ids == partition
        direction, midpoint, scale = _dim_fit_with_scale(
            train_x[mask], train_labels[mask]
        )
        directions.append(direction)
        midpoints.append(midpoint)
        scales.append(scale)
    return (
        np.stack(directions),
        np.asarray(midpoints, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def standardized_scores(
    features: np.ndarray,
    directions: np.ndarray,
    midpoints: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raw = features.astype(np.float32) @ directions.T - midpoints[None, :]
    return raw / scales[None, :]


def signed_separation(
    standardized: np.ndarray,
    labels: np.ndarray,
    datasets: np.ndarray,
) -> dict[str, Any]:
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Signed separation requires both semantic labels.")
    effects = (
        standardized[labels == 1].mean(axis=0)
        - standardized[labels == 0].mean(axis=0)
    )
    ensemble = standardized.mean(axis=1)
    macro, per_dataset = _macro_dataset_auroc(labels, ensemble, datasets)
    return {
        "T": float(np.mean(effects)),
        "partition_effects": [float(value) for value in effects],
        "ensemble_macro_dataset_auroc": macro,
        "ensemble_overall_auroc": _auroc(labels, ensemble),
        "ensemble_per_dataset_auroc": per_dataset,
    }


def _evaluate_mask(
    layer_features: np.ndarray,
    examples: pd.DataFrame,
    mask: np.ndarray,
    directions: np.ndarray,
    midpoints: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    standardized = standardized_scores(
        layer_features[rows], directions, midpoints, scales
    )
    return signed_separation(
        standardized,
        examples.iloc[rows]["label"].to_numpy(dtype=np.int8),
        examples.iloc[rows]["dataset"].to_numpy(),
    )


def _layer_development_result(
    layer: int,
    activations: np.ndarray,
    examples: pd.DataFrame,
    train_rows: np.ndarray,
    train_labels: np.ndarray,
    assignment: dict[str, int],
    partitions: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    layer_features = activations[:, layer].astype(np.float32)
    directions, midpoints, scales = fit_partition_directions(
        layer_features,
        examples,
        train_rows,
        train_labels,
        assignment,
        partitions,
    )
    base = (examples["scheme"].to_numpy() == "primary") & (
        examples["split"].to_numpy() == "dev"
    )
    by_mapping = {
        mapping: _evaluate_mask(
            layer_features,
            examples,
            base & (examples["mapping"].to_numpy() == mapping),
            directions,
            midpoints,
            scales,
        )
        for mapping in ("standard", "reversed")
    }
    result = {
        "layer": layer,
        "selection_score": float(min(by_mapping[name]["T"] for name in by_mapping)),
        "primary_dev_by_mapping": by_mapping,
        "directional_consensus_C": float(np.linalg.norm(directions.mean(axis=0))),
    }
    return result, (directions, midpoints, scales)


def select_layer(
    activations: np.ndarray,
    examples: pd.DataFrame,
    train_labels: np.ndarray,
    assignment: dict[str, int],
    partitions: int,
) -> tuple[int, list[dict[str, Any]]]:
    train_rows = _primary_train_rows(examples)
    rows: list[dict[str, Any]] = []
    for layer in range(activations.shape[1]):
        result, _ = _layer_development_result(
            layer,
            activations,
            examples,
            train_rows,
            train_labels,
            assignment,
            partitions,
        )
        rows.append(result)
    selected = max(rows, key=lambda row: (row["selection_score"], -row["layer"]))
    return int(selected["layer"]), rows


def _test_cells(
    layer_features: np.ndarray,
    examples: pd.DataFrame,
    directions: np.ndarray,
    midpoints: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    split_test = examples["split"].to_numpy() == "test"
    cells: dict[str, Any] = {}
    for scheme in ("primary", "transfer"):
        scheme_mask = split_test & (examples["scheme"].to_numpy() == scheme)
        cells[scheme] = {
            "overall": _evaluate_mask(
                layer_features,
                examples,
                scheme_mask,
                directions,
                midpoints,
                scales,
            )
        }
        for mapping in ("standard", "reversed"):
            cells[scheme][mapping] = _evaluate_mask(
                layer_features,
                examples,
                scheme_mask & (examples["mapping"].to_numpy() == mapping),
                directions,
                midpoints,
                scales,
            )
    return cells


def _bootstrap_scheme_t(
    layer_features: np.ndarray,
    examples: pd.DataFrame,
    scheme: str,
    directions: np.ndarray,
    midpoints: np.ndarray,
    scales: np.ndarray,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    mask = (examples["split"].to_numpy() == "test") & (
        examples["scheme"].to_numpy() == scheme
    )
    rows = np.flatnonzero(mask)
    frame = examples.iloc[rows].reset_index(drop=True)
    standardized = standardized_scores(
        layer_features[rows], directions, midpoints, scales
    )
    groups = sorted(frame["group_id"].astype(str).unique())
    group_rows = {
        group: np.flatnonzero(frame["group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    labels = frame["label"].to_numpy(dtype=np.int8)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sampled_rows = np.concatenate([group_rows[str(group)] for group in sampled])
        effect = signed_separation(
            standardized[sampled_rows],
            labels[sampled_rows],
            frame.iloc[sampled_rows]["dataset"].to_numpy(),
        )["T"]
        draws.append(float(effect))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
        "replicates": replicates,
        "confidence_level": confidence_level,
    }


def evaluate_test_layer(
    layer: int,
    activations: np.ndarray,
    examples: pd.DataFrame,
    true_train_labels: np.ndarray,
    assignment: dict[str, int],
    partitions: int,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    train_rows = _primary_train_rows(examples)
    layer_features = activations[:, layer].astype(np.float32)
    directions, midpoints, scales = fit_partition_directions(
        layer_features,
        examples,
        train_rows,
        true_train_labels,
        assignment,
        partitions,
    )
    cells = _test_cells(
        layer_features, examples, directions, midpoints, scales
    )
    intervals = {
        scheme: _bootstrap_scheme_t(
            layer_features,
            examples,
            scheme,
            directions,
            midpoints,
            scales,
            bootstrap_replicates,
            confidence_level,
            seed + 1000 * layer + offset,
        )
        for offset, scheme in enumerate(("primary", "transfer"), start=1)
    }
    transfer_cells_positive = all(
        cells[scheme][mapping]["T"] > 0.0
        for scheme in ("primary", "transfer")
        for mapping in ("standard", "reversed")
    )
    clear = bool(
        intervals["primary"]["low"] > 0.0
        and intervals["transfer"]["low"] > 0.0
        and transfer_cells_positive
    )
    return {
        "layer": layer,
        "directional_consensus_C": float(np.linalg.norm(directions.mean(axis=0))),
        "test": cells,
        "group_bootstrap_ci": intervals,
        "clear_adjacent_effect": clear,
    }


def group_sign_flip_labels(
    examples: pd.DataFrame,
    train_rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = examples.iloc[train_rows]
    groups = sorted(frame["group_id"].astype(str).unique())
    flips = {group: int(value) for group, value in zip(groups, rng.integers(0, 2, len(groups)), strict=True)}
    labels = frame["label"].to_numpy(dtype=np.int8)
    flipped = np.asarray(
        [label ^ flips[group] for label, group in zip(labels, frame["group_id"].astype(str), strict=True)],
        dtype=np.int8,
    )
    for _, rows in frame.assign(permuted=flipped).groupby("source_row", sort=False):
        if len(rows) != 2 or int(rows["permuted"].sum()) != 1:
            raise RuntimeError("Group sign flip broke an affirmative/negated training pair.")
    return flipped


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_permutation_checkpoint(
    local_path: Path | None,
    persistent_path: Path | None,
    signature: str | None,
    permutations: int,
    seed: int,
) -> tuple[list[float], list[int], str | None]:
    if local_path is None and persistent_path is None:
        return [], [], None
    if not signature:
        raise RuntimeError("Permutation checkpoints require a cache-bound signature.")

    for candidate in (local_path, persistent_path):
        if candidate is None or not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable permutation checkpoint: {candidate}") from exc
        expected = {
            "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
            "checkpoint_signature": signature,
            "permutations": permutations,
            "permutation_seed": seed,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Permutation checkpoint {candidate} has mismatched {key}."
                )
        values = [float(value) for value in payload.get("values", [])]
        layers = [int(layer) for layer in payload.get("selected_layers", [])]
        if len(values) != len(layers) or len(values) > permutations:
            raise RuntimeError(f"Malformed permutation checkpoint: {candidate}")
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite value in permutation checkpoint: {candidate}")
        if (
            candidate == persistent_path
            and local_path is not None
            and local_path != persistent_path
        ):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, local_path)
            if _file_sha256(candidate) != _file_sha256(local_path):
                raise RuntimeError("Restored permutation checkpoint failed SHA-256 verification.")
        return values, layers, str(candidate)
    return [], [], None


def _write_permutation_checkpoint(
    local_path: Path | None,
    persistent_path: Path | None,
    signature: str | None,
    permutations: int,
    seed: int,
    values: list[float],
    selected_layers: list[int],
) -> None:
    if local_path is None and persistent_path is None:
        return
    if local_path is None or not signature:
        raise RuntimeError("A local path and cache-bound signature are required for checkpoints.")
    payload = {
        "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
        "checkpoint_signature": signature,
        "permutations": permutations,
        "permutation_seed": seed,
        "completed": len(values),
        "values": values,
        "selected_layers": selected_layers,
    }
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_suffix(local_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, local_path)
    if persistent_path is not None and persistent_path != local_path:
        persistent_path.parent.mkdir(parents=True, exist_ok=True)
        persistent_temporary = persistent_path.with_suffix(
            persistent_path.suffix + ".tmp"
        )
        shutil.copy2(local_path, persistent_temporary)
        if _file_sha256(local_path) != _file_sha256(persistent_temporary):
            raise RuntimeError("Persistent permutation checkpoint failed SHA-256 verification.")
        os.replace(persistent_temporary, persistent_path)


def complete_permutation_null(
    activations: np.ndarray,
    examples: pd.DataFrame,
    assignment: dict[str, int],
    partitions: int,
    permutations: int,
    observed: float,
    seed: int,
    checkpoint_every: int = 25,
    checkpoint_path: Path | None = None,
    persistent_checkpoint_path: Path | None = None,
    checkpoint_signature: str | None = None,
) -> dict[str, Any]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive.")
    train_rows = _primary_train_rows(examples)
    values, selected_layers, restored_from = _load_permutation_checkpoint(
        checkpoint_path, persistent_checkpoint_path, checkpoint_signature,
        permutations, seed,
    )
    resumed_from = len(values)
    if resumed_from:
        LOGGER.info("Restored %d/%d v2 null permutations from %s", resumed_from, permutations, restored_from)
    for index in range(resumed_from, permutations):
        rng = np.random.default_rng(seed + index)
        permuted = group_sign_flip_labels(examples, train_rows, rng)
        selected, _ = select_layer(
            activations, examples, permuted, assignment, partitions
        )
        layer_features = activations[:, selected].astype(np.float32)
        directions, midpoints, scales = fit_partition_directions(
            layer_features,
            examples,
            train_rows,
            permuted,
            assignment,
            partitions,
        )
        primary_test = (
            (examples["scheme"].to_numpy() == "primary")
            & (examples["split"].to_numpy() == "test")
        )
        value = _evaluate_mask(
            layer_features,
            examples,
            primary_test,
            directions,
            midpoints,
            scales,
        )["T"]
        values.append(float(value))
        selected_layers.append(selected)
        if (index + 1) % checkpoint_every == 0 or index + 1 == permutations:
            _write_permutation_checkpoint(
                checkpoint_path,
                persistent_checkpoint_path,
                checkpoint_signature,
                permutations,
                seed,
                values,
                selected_layers,
            )
            LOGGER.info("Completed v2 null permutation %d/%d", index + 1, permutations)
    array = np.asarray(values, dtype=np.float64)
    exceedances = int(np.sum(array >= observed))
    return {
        "definition": (
            "Within-training-group semantic sign flips; complete all-layer dev selection "
            "and one untouched primary test evaluation per null run."
        ),
        "permutations": permutations,
        "observed_T": observed,
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "quantiles": {
            "0.00": float(np.quantile(array, 0.00)),
            "0.05": float(np.quantile(array, 0.05)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.95": float(np.quantile(array, 0.95)),
            "1.00": float(np.quantile(array, 1.00)),
        },
        "count_greater_equal_observed": exceedances,
        "p_greater_equal": float((1 + exceedances) / (permutations + 1)),
        "selected_layer_counts": {
            str(layer): int(count)
            for layer, count in sorted(Counter(selected_layers).items())
        },
        "values": [float(value) for value in array],
        "checkpoint": {
            "resumed_permutations": resumed_from,
            "restored_from": restored_from,
            "checkpoint_signature": checkpoint_signature,
        },
    }


def _native_gate(native: dict[str, Any], threshold: float) -> tuple[bool, dict[str, float]]:
    observed: dict[str, float] = {}
    for scheme in ("primary", "transfer"):
        for cell in ("overall", "standard", "reversed"):
            value = float(native[scheme]["test"][cell]["macro_dataset_auroc"])
            observed[f"{scheme}:{cell}"] = value
    return bool(all(value > threshold for value in observed.values())), observed


def analyze(
    activations: np.ndarray,
    truth_scores: np.ndarray,
    predicted_labels: np.ndarray,
    examples: pd.DataFrame,
    analysis_config: dict[str, Any],
    seed: int,
    mode: str,
    checkpoint_dir: Path | None = None,
    persistent_checkpoint_dir: Path | None = None,
    checkpoint_signature: str | None = None,
) -> dict[str, Any]:
    del predicted_labels
    if activations.ndim != 3 or len(activations) != len(examples):
        raise RuntimeError("Activation tensor shape differs from the manifest.")
    if not np.isfinite(activations).all() or not np.isfinite(truth_scores).all():
        raise RuntimeError("Non-finite activation or candidate score detected.")

    partitions = int(analysis_config["training_partitions"])
    assignment = assign_training_partitions(examples, partitions, seed)
    train_rows = _primary_train_rows(examples)
    true_train_labels = examples.iloc[train_rows]["label"].to_numpy(dtype=np.int8)
    selected_layer, development = select_layer(
        activations,
        examples,
        true_train_labels,
        assignment,
        partitions,
    )
    candidate_layers = sorted(
        {
            layer
            for layer in (selected_layer - 1, selected_layer, selected_layer + 1)
            if 0 <= layer < activations.shape[1]
        }
    )
    layer_results = {
        str(layer): evaluate_test_layer(
            layer,
            activations,
            examples,
            true_train_labels,
            assignment,
            partitions,
            int(analysis_config["bootstrap_replicates"]),
            float(analysis_config["confidence_level"]),
            seed,
        )
        for layer in candidate_layers
    }
    selected_result = layer_results[str(selected_layer)]
    observed_t = float(selected_result["test"]["primary"]["overall"]["T"])
    checkpoint_path = (
        checkpoint_dir / "complete_permutation_null.json" if checkpoint_dir else None
    )
    persistent_checkpoint_path = (
        persistent_checkpoint_dir / "complete_permutation_null.json"
        if persistent_checkpoint_dir else None
    )
    permutation = complete_permutation_null(
        activations,
        examples,
        assignment,
        partitions,
        int(analysis_config["permutations"]),
        observed_t,
        seed + 4000,
        int(analysis_config["checkpoint_every_permutations"]),
        checkpoint_path,
        persistent_checkpoint_path,
        checkpoint_signature,
    )
    native = native_answer_controls(examples, truth_scores)
    native_pass, native_values = _native_gate(
        native, float(analysis_config["native_auroc_gate"])
    )
    adjacent_clear = any(
        result["clear_adjacent_effect"]
        for layer, result in (
            (int(key), value) for key, value in layer_results.items()
        )
        if layer != selected_layer and abs(layer - selected_layer) == 1
    )
    cell_positive = all(
        selected_result["test"][scheme][mapping]["T"] > 0.0
        for scheme in ("primary", "transfer")
        for mapping in ("standard", "reversed")
    )
    checks = {
        "native_semantic_auroc_strictly_above_0.70": native_pass,
        "complete_group_permutation_p_below_0.05": permutation["p_greater_equal"]
        < float(analysis_config["permutation_p_gate"]),
        "selected_primary_ci_above_zero": selected_result["group_bootstrap_ci"]["primary"]["low"] > 0.0,
        "selected_transfer_ci_above_zero": selected_result["group_bootstrap_ci"]["transfer"]["low"] > 0.0,
        "all_four_mapping_transfer_T_positive": cell_positive,
        "selected_layer_clear": selected_result["clear_adjacent_effect"],
        "at_least_one_immediate_neighbor_clear": adjacent_clear,
        "layer_selected_from_primary_development_only": True,
    }
    if mode == "smoke":
        disposition = "TRUTH_CONTROL_V2_SMOKE_COMPLETE"
    else:
        disposition = (
            "TRUTH_CONTROL_V2_PASS"
            if all(checks.values())
            else "TRUTH_CONTROL_V2_STRICT_FAIL"
        )
    return {
        "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
        "terminal_disposition": disposition,
        "mode": mode,
        "selected_layer": selected_layer,
        "selection_rule": (
            "maximize the minimum primary A/B development T across standard and reversed "
            "mapping subsets; lower-layer exact tie break"
        ),
        "training_partitions": partitions,
        "development_layer_sweep": development,
        "confirmatory_layer_candidates": candidate_layers,
        "confirmatory_layer_results": layer_results,
        "primary_test_T": observed_t,
        "directional_consensus_C": selected_result["directional_consensus_C"],
        "permutation_null": permutation,
        "native_answer_controls": native,
        "native_gate_values": native_values,
        "checks": checks,
        "auroc_role": "descriptive_only_not_a_permutation_gate",
        "v1_disposition_changed": False,
    }
