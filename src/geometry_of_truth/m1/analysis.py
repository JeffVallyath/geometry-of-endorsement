from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .support.metrics import fit_scaler

from .metrics import (
    auroc,
    balanced_accuracy,
    board_ci,
    board_metrics,
    boards_from_manifest,
    calibration,
    grouped_mean_ci,
    mirrored_pairwise,
    dyadic_mean_ci,
    paired_interaction_delta_ci,
    strip_private,
)
from .probes import fit_difference_in_means, fit_logistic, grouped_label_flip
from .text_baseline import fit_sbert_interaction


CORE_METRICS = (
    "within_situation_macro",
    "within_consideration_macro",
    "checkerboard_interaction_mean",
    "checkerboard_signed_board_mean",
    "checkerboard_pairwise",
    "checkerboard_both_ways",
)


def _mask(frame: pd.DataFrame, split: str) -> np.ndarray:
    return frame["split"].astype(str).to_numpy() == split


def _average_by_identity(values: np.ndarray, identities: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    identities = np.asarray(identities).astype(str)
    result = np.empty(len(values), dtype=np.float64)
    for identity in np.unique(identities):
        mask = identities == identity
        result[mask] = float(values[mask].mean())
    return result


def _random_scores(item_ids: np.ndarray, seed: int) -> np.ndarray:
    import hashlib

    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:random-baseline:{item}".encode()).digest()[:8],
                "big",
            )
            / float(2**64)
            for item in item_ids.astype(str)
        ],
        dtype=np.float64,
    )


def _evaluate(
    frame: pd.DataFrame,
    raw_scores: np.ndarray,
    board_manifest: pd.DataFrame,
    scaler: Any,
    *,
    decision_threshold: float = 0.0,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    mirrored = mirrored_pairwise(frame, raw_scores)
    boards = boards_from_manifest(board_manifest, frame)
    checkerboard = board_metrics(boards, raw_scores, scaler)
    interaction_rows = np.asarray(
        [row["interaction_contrast"] for row in checkerboard["_per_board"]]
    )
    signed_rows = np.asarray(
        [row["signed_board_score"] for row in checkerboard["_per_board"]]
    )
    node_a = np.asarray([row["node_a"] for row in checkerboard["_per_board"]])
    node_b = np.asarray([row["node_b"] for row in checkerboard["_per_board"]])
    result = {
        "auroc": auroc(labels, raw_scores),
        "balanced_accuracy": balanced_accuracy(
            labels, raw_scores - float(decision_threshold)
        ),
        **strip_private(mirrored),
        "checkerboard_interaction_mean": checkerboard["interaction_contrast_mean"],
        "checkerboard_interaction_sd": checkerboard["interaction_contrast_sd"],
        "checkerboard_signed_board_mean": checkerboard["signed_board_mean"],
        "checkerboard_pairwise": checkerboard["pairwise"],
        "checkerboard_both_ways": checkerboard["both_ways"],
        "checkerboard_both_ways_chance": checkerboard["both_ways_chance"],
        "checkerboard_both_ways_additive": checkerboard["both_ways_additive"],
        "checkerboard_both_ways_descriptive_only": True,
        "score_scaler": checkerboard["scaler"],
        "n_boards": checkerboard["n_boards"],
        "confidence_intervals": {
            "checkerboard_interaction_mean": dyadic_mean_ci(
                interaction_rows, node_a, node_b,
                confidence_level=confidence_level,
            ),
            "checkerboard_signed_board_mean": dyadic_mean_ci(
                signed_rows, node_a, node_b,
                confidence_level=confidence_level,
            ),
            "within_situation_macro": grouped_mean_ci(
                mirrored["_per_situation"],
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 1,
            ),
            "within_consideration_macro": grouped_mean_ci(
                mirrored["_per_consideration"],
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 2,
            ),
            "checkerboard_pairwise": board_ci(
                checkerboard,
                field="pairwise",
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 3,
            ),
            "checkerboard_both_ways": board_ci(
                checkerboard,
                field="both_ways",
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 4,
            ),
        },
        "_board_metric": checkerboard,
    }
    return result


def _public_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _selection_value(result: dict[str, Any]) -> float:
    return float(result["mean_mirrored_pairwise"])


def _select_layer(
    train_features: np.ndarray,
    select_features: np.ndarray,
    train_labels: np.ndarray,
    select_frame: pd.DataFrame,
    select_boards: pd.DataFrame,
    analysis_config: dict[str, Any],
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for layer in range(train_features.shape[1]):
        probe = fit_difference_in_means(train_features[:, layer], train_labels)
        select_scores = probe.score(select_features[:, layer])
        result = _evaluate(
            select_frame,
            select_scores,
            select_boards,
            fit_scaler(select_scores),
            bootstrap_replicates=max(1, min(50, int(analysis_config["bootstrap_replicates"]))),
            confidence_level=float(analysis_config["confidence_level"]),
            seed=seed + 1000 * layer,
        )
        rows.append(
            {
                "layer": layer,
                "within_situation_macro": result["within_situation_macro"],
                "within_consideration_macro": result["within_consideration_macro"],
                "selection_metric": _selection_value(result),
            }
        )
    selected = min(
        rows,
        key=lambda row: (-float(row["selection_metric"]), int(row["layer"])),
    )["layer"]
    return int(selected), rows


def _select_logistic_c(
    train_features: np.ndarray,
    select_features: np.ndarray,
    train_labels: np.ndarray,
    select_frame: pd.DataFrame,
    select_boards: pd.DataFrame,
    analysis_config: dict[str, Any],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for c in analysis_config["logistic_c_grid"]:
        probe = fit_logistic(
            train_features,
            train_labels,
            c=float(c),
            max_iter=int(analysis_config["logistic_max_iter"]),
            seed=seed,
        )
        select_scores = probe.score(select_features)
        result = _evaluate(
            select_frame,
            select_scores,
            select_boards,
            fit_scaler(select_scores),
            bootstrap_replicates=max(1, min(50, int(analysis_config["bootstrap_replicates"]))),
            confidence_level=float(analysis_config["confidence_level"]),
            seed=seed + int(round(float(c) * 10000)),
        )
        rows.append({"c": float(c), "selection_metric": _selection_value(result)})
    selected = min(rows, key=lambda row: (-row["selection_metric"], row["c"]))["c"]
    return float(selected), rows


def _permutation_summary(real: float, values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    observed = float(real)
    return {
        "observed": observed,
        "mean": float(array.mean()),
        "interval_95": [float(x) for x in np.quantile(array, [0.025, 0.975])],
        "percentile_95": float(np.quantile(array, 0.95)),
        "empirical_p_greater_equal": float(
            (1 + np.sum(array >= observed)) / (len(array) + 1)
        ),
        "rank_ascending": int(1 + np.sum(array < observed)),
        "n": int(len(array)),
        "values": array.tolist(),
    }


def _run_permutations(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    select_features: np.ndarray,
    eval_features: np.ndarray,
    eval_frame: pd.DataFrame,
    eval_boards: pd.DataFrame,
    selected_c: float,
    analysis_config: dict[str, Any],
    seed: int,
    observed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 404)
    values = {
        family: {metric: [] for metric in CORE_METRICS}
        for family in ("difference_in_means", "logistic")
    }
    for permutation in range(int(analysis_config["permutations"])):
        labels = grouped_label_flip(train_labels, train_groups, rng)
        if len(np.unique(labels)) != 2:
            raise RuntimeError("Grouped label null collapsed to one class.")
        dim = fit_difference_in_means(train_features, labels)
        logistic = fit_logistic(
            train_features,
            labels,
            c=selected_c,
            max_iter=int(analysis_config["logistic_max_iter"]),
            seed=seed + permutation,
        )
        for family, scores in (
            ("difference_in_means", dim.score(eval_features)),
            ("logistic", logistic.score(eval_features)),
        ):
            select_scores = (
                dim.score(select_features)
                if family == "difference_in_means"
                else logistic.score(select_features)
            )
            result = _evaluate(
                eval_frame,
                scores,
                eval_boards,
                fit_scaler(select_scores),
                bootstrap_replicates=1,
                confidence_level=float(analysis_config["confidence_level"]),
                seed=seed + permutation,
            )
            for metric in CORE_METRICS:
                values[family][metric].append(float(result[metric]))
    return {
        family: {
            metric: _permutation_summary(observed[family][metric], rows[metric])
            for metric in CORE_METRICS
        }
        for family, rows in values.items()
    }


def _board_family_analysis(
    frame: pd.DataFrame,
    board_manifest: pd.DataFrame,
    board_metric: dict[str, Any],
) -> list[dict[str, Any]]:
    by_row = frame.set_index("row_id", drop=False)
    manifest_by_board = board_manifest.set_index("board_id", drop=False)
    metric_by_board = {
        row["board_id"]: row for row in board_metric["_per_board"]
    }
    rows: list[dict[str, Any]] = []
    for board in boards_from_manifest(board_manifest, frame):
        record = manifest_by_board.loc[board["board_id"]]
        vrd = {
            str(by_row.loc[str(record[column]), "vrd"])
            for column in (
                "row_id_s1_A",
                "row_id_s1_B",
                "row_id_s2_A",
                "row_id_s2_B",
            )
        }
        metric = metric_by_board[board["board_id"]]
        rows.append(
            {
                "family": "+".join(sorted(vrd)),
                "interaction_contrast": metric["interaction_contrast"],
                "signed_board_score": metric["signed_board_score"],
                "both_ways": metric["both_ways"],
            }
        )
    return [
        {
            "family": family,
            "n_boards": int(len(group)),
            "interaction_contrast_mean": float(group["interaction_contrast"].mean()),
            "signed_board_mean": float(group["signed_board_score"].mean()),
            "both_ways_descriptive": float(group["both_ways"].mean()),
        }
        for family, group in pd.DataFrame(rows).groupby("family", sort=True)
    ]


def _error_analysis(board_metric: dict[str, Any]) -> dict[str, Any]:
    rows = list(board_metric["_per_board"])
    right = sorted(
        (row for row in rows if row["both_ways"] == 1.0),
        key=lambda row: min(row["gap_1"], row["gap_2"]),
        reverse=True,
    )[:20]
    wrong = sorted(
        (row for row in rows if row["both_ways"] == 0.0),
        key=lambda row: min(row["gap_1"], row["gap_2"]),
    )[:20]
    return {
        "both_right": right,
        "both_wrong_or_partial": wrong,
        "contains_source_text": False,
    }


def _mapping_control(
    frame: pd.DataFrame,
    scores: np.ndarray,
    mapping_names: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for suffix in ("standard", "reversed"):
        mask = np.asarray([str(value).endswith(suffix) for value in mapping_names])
        labels = frame.loc[mask, "label"].to_numpy(dtype=np.int8)
        subset = np.asarray(scores)[mask]
        result[suffix] = {
            "n": int(mask.sum()),
            "auroc": auroc(labels, subset) if mask.any() else float("nan"),
            "balanced_accuracy": (
                balanced_accuracy(labels, subset) if mask.any() else float("nan")
            ),
        }
    return result


def analyze(
    *,
    frame: pd.DataFrame,
    features: dict[str, np.ndarray],
    native_margins: dict[str, np.ndarray],
    mapping_names: dict[str, np.ndarray],
    select_board_manifest: pd.DataFrame,
    eval_board_manifest: pd.DataFrame,
    truth_control: dict[str, Any],
    pipeline_checks: dict[str, bool],
    config: Any,
    text_baseline_train: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    required_features = {
        "primary_joint",
        "primary_situation",
        "primary_consideration",
        "transfer_joint",
    }
    if set(features) != required_features:
        raise RuntimeError(f"Expected feature sets {sorted(required_features)}.")
    row_count = len(frame)
    if any(array.shape[0] != row_count for array in features.values()):
        raise RuntimeError("Feature arrays are not aligned to the pilot manifest.")
    if any(len(array) != row_count for array in native_margins.values()):
        raise RuntimeError("Native margins are not aligned to the pilot manifest.")

    analysis_config = config.section("analysis")
    seed = int(config.raw["run"]["seed"])
    train_mask = _mask(frame, "pilot_train")
    select_mask = _mask(frame, "pilot_select")
    eval_mask = _mask(frame, "pilot_eval")
    train = frame.loc[train_mask].reset_index(drop=True)
    selection = frame.loc[select_mask].reset_index(drop=True)
    evaluation = frame.loc[eval_mask].reset_index(drop=True)
    train_labels = train["label"].to_numpy(dtype=np.int8)

    primary = features["primary_joint"]
    selected_layer, selection_curve = _select_layer(
        primary[train_mask],
        primary[select_mask],
        train_labels,
        selection,
        select_board_manifest,
        analysis_config,
        seed,
    )
    selected_c, c_selection = _select_logistic_c(
        primary[train_mask, selected_layer],
        primary[select_mask, selected_layer],
        train_labels,
        selection,
        select_board_manifest,
        analysis_config,
        seed,
    )
    dim = fit_difference_in_means(primary[train_mask, selected_layer], train_labels)
    logistic = fit_logistic(
        primary[train_mask, selected_layer],
        train_labels,
        c=selected_c,
        max_iter=int(analysis_config["logistic_max_iter"]),
        seed=seed,
    )
    dim_select_scores = dim.score(primary[select_mask, selected_layer])
    logistic_select_scores = logistic.score(primary[select_mask, selected_layer])
    dim_eval_scores = dim.score(primary[eval_mask, selected_layer])
    logistic_eval_scores = logistic.score(primary[eval_mask, selected_layer])

    evaluations: dict[str, dict[str, Any]] = {}
    private_evaluations: dict[str, dict[str, Any]] = {}

    def record(
        name: str,
        select_scores: np.ndarray,
        eval_scores: np.ndarray,
        offset: int,
        decision_threshold: float = 0.0,
    ) -> None:
        scaler = fit_scaler(select_scores)
        full = _evaluate(
            evaluation,
            eval_scores,
            eval_board_manifest,
            scaler,
            decision_threshold=decision_threshold,
            bootstrap_replicates=int(analysis_config["bootstrap_replicates"]),
            confidence_level=float(analysis_config["confidence_level"]),
            seed=seed + offset,
        )
        private_evaluations[name] = full
        evaluations[name] = _public_evaluation(full)

    record("difference_in_means", dim_select_scores, dim_eval_scores, 10)
    record("logistic", logistic_select_scores, logistic_eval_scores, 20)

    situation_probe = fit_difference_in_means(
        features["primary_situation"][train_mask, selected_layer], train_labels
    )
    consideration_probe = fit_difference_in_means(
        features["primary_consideration"][train_mask, selected_layer], train_labels
    )
    raw_situation_select = situation_probe.score(
        features["primary_situation"][select_mask, selected_layer]
    )
    raw_situation_eval = situation_probe.score(
        features["primary_situation"][eval_mask, selected_layer]
    )
    raw_consideration_select = consideration_probe.score(
        features["primary_consideration"][select_mask, selected_layer]
    )
    raw_consideration_eval = consideration_probe.score(
        features["primary_consideration"][eval_mask, selected_layer]
    )
    situation_select = _average_by_identity(
        raw_situation_select, selection["situation_id"].to_numpy()
    )
    situation_eval = _average_by_identity(
        raw_situation_eval, evaluation["situation_id"].to_numpy()
    )
    consideration_select = _average_by_identity(
        raw_consideration_select,
        selection["consideration_cluster_id"].to_numpy(),
    )
    consideration_eval = _average_by_identity(
        raw_consideration_eval,
        evaluation["consideration_cluster_id"].to_numpy(),
    )
    additive_select = np.rint(situation_select * 1e6) + np.rint(
        consideration_select * 1e6
    )
    additive_eval = np.rint(situation_eval * 1e6) + np.rint(
        consideration_eval * 1e6
    )
    sbert_train = train if text_baseline_train is None else text_baseline_train
    (sbert_select, sbert_eval), sbert_metadata = fit_sbert_interaction(
        sbert_train, selection, evaluation, device="cpu"
    )
    sbert_metadata["training_scope"] = (
        "vertical_slice_pilot_train"
        if text_baseline_train is None
        else "manifest_train_common"
    )
    random_select = _random_scores(selection["item_id"].to_numpy(), seed)
    random_eval = _random_scores(evaluation["item_id"].to_numpy(), seed)
    primary_native_select = np.asarray(native_margins["primary"])[select_mask]
    primary_native = np.asarray(native_margins["primary"])[eval_mask]
    for offset, (name, select_scores, eval_scores) in enumerate(
        (
            ("situation_only_activation", situation_select, situation_eval),
            ("consideration_only_activation", consideration_select, consideration_eval),
            ("separate_encoding_additive", additive_select, additive_eval),
            ("sbert_interaction", sbert_select, sbert_eval),
            ("native_answer_margin", primary_native_select, primary_native),
            ("deterministic_random", random_select, random_eval),
        ),
        start=30,
    ):
        record(
            name,
            select_scores,
            eval_scores,
            offset,
            decision_threshold=(
                0.5 if name in {"sbert_interaction", "deterministic_random"} else 0.0
            ),
        )

    layerwise: list[dict[str, Any]] = []
    for layer in range(primary.shape[1]):
        layer_probe = fit_difference_in_means(primary[train_mask, layer], train_labels)
        layer_select_scores = layer_probe.score(primary[select_mask, layer])
        layer_result = _evaluate(
            evaluation,
            layer_probe.score(primary[eval_mask, layer]),
            eval_board_manifest,
            fit_scaler(layer_select_scores),
            bootstrap_replicates=1,
            confidence_level=float(analysis_config["confidence_level"]),
            seed=seed + 10000 + layer,
        )
        layerwise.append(
            {
                "layer": layer,
                "auroc": layer_result["auroc"],
                "balanced_accuracy": layer_result["balanced_accuracy"],
                "within_situation_macro": layer_result["within_situation_macro"],
                "within_consideration_macro": layer_result[
                    "within_consideration_macro"
                ],
                "checkerboard_interaction_mean": layer_result[
                    "checkerboard_interaction_mean"
                ],
                "checkerboard_signed_board_mean": layer_result[
                    "checkerboard_signed_board_mean"
                ],
                "checkerboard_pairwise": layer_result["checkerboard_pairwise"],
                "checkerboard_both_ways": layer_result[
                    "checkerboard_both_ways"
                ],
            }
        )

    transfer_select_features = features["transfer_joint"][select_mask, selected_layer]
    transfer_eval_features = features["transfer_joint"][eval_mask, selected_layer]
    transfer_dim_select = dim.score(transfer_select_features)
    transfer_logistic_select = logistic.score(transfer_select_features)
    transfer_dim_scores = dim.score(transfer_eval_features)
    transfer_logistic_scores = logistic.score(transfer_eval_features)
    transfer = {}
    private_transfer = {}
    for offset, (name, select_scores, scores) in enumerate(
        (
            ("difference_in_means", transfer_dim_select, transfer_dim_scores),
            ("logistic", transfer_logistic_select, transfer_logistic_scores),
        ),
        start=300,
    ):
        private_transfer[name] = _evaluate(
                evaluation,
                scores,
                eval_board_manifest,
                fit_scaler(select_scores),
                bootstrap_replicates=int(
                    analysis_config["bootstrap_replicates"]
                ),
                confidence_level=float(analysis_config["confidence_level"]),
                seed=seed + offset,
            )
        transfer[name] = _public_evaluation(private_transfer[name])

    observed = {
        family: evaluations[family]
        for family in ("difference_in_means", "logistic")
    }
    permutations = _run_permutations(
        train_features=primary[train_mask, selected_layer],
        train_labels=train_labels,
        train_groups=train["situation_id"].to_numpy(),
        select_features=primary[select_mask, selected_layer],
        eval_features=primary[eval_mask, selected_layer],
        eval_frame=evaluation,
        eval_boards=eval_board_manifest,
        selected_c=selected_c,
        analysis_config=analysis_config,
        seed=seed,
        observed=observed,
    )
    delta_ci = {
        family: paired_interaction_delta_ci(
            private_evaluations[family]["_board_metric"],
            private_evaluations["sbert_interaction"]["_board_metric"],
            confidence_level=float(analysis_config["confidence_level"]),
        )
        for index, family in enumerate(("difference_in_means", "logistic"))
    }
    transfer_delta_ci = {
        family: paired_interaction_delta_ci(
            private_transfer[family]["_board_metric"],
            private_evaluations["sbert_interaction"]["_board_metric"],
            confidence_level=float(analysis_config["confidence_level"]),
        )
        for family in ("difference_in_means", "logistic")
    }
    invariants = {
        "situation_only_within_situation_exactly_half": (
            evaluations["situation_only_activation"][
                "within_situation_exact_m0"
            ]
            == 0.5
        ),
        "consideration_only_within_consideration_exactly_half": (
            evaluations["consideration_only_activation"][
                "within_consideration_exact_m0"
            ]
            == 0.5
        ),
        "additive_checkerboard_both_ways_exactly_zero": (
            evaluations["separate_encoding_additive"][
                "checkerboard_both_ways"
            ]
            == 0.0
        ),
        "additive_interaction_contrast_exactly_zero": abs(
            evaluations["separate_encoding_additive"][
                "checkerboard_interaction_mean"
            ]
        ) < 1e-12,
        "additive_signed_board_score_exactly_zero": (
            evaluations["separate_encoding_additive"][
                "checkerboard_signed_board_mean"
            ]
            == 0.0
        ),
    }
    family_analysis = {
        family: _board_family_analysis(
            evaluation,
            eval_board_manifest,
            private_evaluations[family]["_board_metric"],
        )
        for family in ("difference_in_means", "logistic")
    }
    mapping_controls = {
        "primary_native_margin": _mapping_control(
            evaluation,
            primary_native,
            np.asarray(mapping_names["primary"])[eval_mask],
        ),
        "transfer_native_margin": _mapping_control(
            evaluation,
            np.asarray(native_margins["transfer"])[eval_mask],
            np.asarray(mapping_names["transfer"])[eval_mask],
        ),
    }
    logistic_probability = logistic.probability(
        primary[eval_mask, selected_layer]
    )
    native_probability = 1.0 / (1.0 + np.exp(-np.clip(primary_native, -80, 80)))
    calibration_results = {
        "logistic": calibration(
            evaluation["label"].to_numpy(dtype=np.int8), logistic_probability
        ),
        "native_answer": calibration(
            evaluation["label"].to_numpy(dtype=np.int8), native_probability
        ),
    }

    sbert_ib = evaluations["sbert_interaction"]["checkerboard_interaction_mean"]
    neighboring_layers = [
        row
        for row in layerwise
        if abs(int(row["layer"]) - selected_layer) == 1
    ]

    def family_count(name: str) -> int:
        return sum(
            row["n_boards"] > 0 and row["interaction_contrast_mean"] > 0.0
            for row in family_analysis[name]
        )

    concentration_checks = {
        "difference_in_means_adjacent_layer_signal": any(
            row["within_situation_macro"] > 0.5
            and row["within_consideration_macro"] > 0.5
            and row["checkerboard_interaction_mean"] > sbert_ib
            for row in neighboring_layers
        ),
        "logistic_not_applicable_to_layer_curve": True,
        "difference_in_means_multiple_board_families": family_count(
            "difference_in_means"
        )
        >= 2,
        "logistic_multiple_board_families": family_count("logistic") >= 2,
    }
    truth_pass = truth_control.get("terminal_disposition") == "PASS"
    validation = {
        "truth_positive_control_passed": truth_pass,
        "all_pipeline_checks_passed": bool(pipeline_checks)
        and all(bool(value) for value in pipeline_checks.values()),
        "all_exact_invariants_passed": all(invariants.values()),
        "both_answer_mappings_present_primary": all(
            mapping_controls["primary_native_margin"][name]["n"] > 0
            for name in ("standard", "reversed")
        ),
        "both_answer_mappings_present_transfer": all(
            mapping_controls["transfer_native_margin"][name]["n"] > 0
            for name in ("standard", "reversed")
        ),
    }
    pipeline_valid = all(validation.values())

    def clear_result(name: str, transfer_name: str) -> bool:
        metric = evaluations[name]
        null = permutations[name]
        transferred = transfer[transfer_name]
        return (
            metric["within_situation_macro"] > 0.5
            and metric["within_consideration_macro"] > 0.5
            and metric["checkerboard_interaction_mean"] > 0.0
            and metric["checkerboard_interaction_mean"]
            > null["checkerboard_interaction_mean"]["percentile_95"]
            and delta_ci[name]["low"] > 0.0
            and transferred["within_situation_macro"] > 0.5
            and transferred["within_consideration_macro"] > 0.5
            and transferred["checkerboard_interaction_mean"] > 0.0
            and transfer_delta_ci[transfer_name]["low"] > 0.0
        )

    dim_clear = clear_result("difference_in_means", "difference_in_means")
    logistic_clear = clear_result("logistic", "logistic")
    dim_supported = dim_clear
    logistic_supported = logistic_clear
    practical_lift = float(analysis_config["practical_effect_sd"])
    valid_null = all(
        delta_ci[name]["high"] < practical_lift
        for name in ("difference_in_means", "logistic")
    )

    if config.mode == "smoke":
        disposition = "M1_SMOKE_ONLY_NOT_EMPIRICAL"
        rationale = "Smoke mode validates mechanics but cannot assign a scientific disposition."
    elif not pipeline_valid:
        disposition = "M1_PIPELINE_NOT_VALIDATED"
        rationale = "At least one truth-control, provenance, mapping, leakage, or exact-invariant gate failed."
    elif evaluations["difference_in_means"]["n_boards"] < int(
        analysis_config["minimum_interpretable_boards"]
    ):
        disposition = "M1_VERTICAL_SLICE_INCONCLUSIVE"
        rationale = "Fewer than the frozen minimum of 75 usable evaluation checkerboards were available."
    elif dim_supported:
        disposition = "M1_VERTICAL_SLICE_SIGNAL_SUPPORTED"
        rationale = "The frozen DIM direction cleared generalization, I_b permutation, SBERT delta, and held-out-template gates."
    elif logistic_supported:
        disposition = "M1_VERTICAL_SLICE_LINEAR_DECODING_ONLY"
        rationale = "Logistic decoding cleared the gates but the transparent DIM direction did not."
    elif valid_null:
        disposition = "M1_VERTICAL_SLICE_VALID_NULL"
        rationale = "Both dyadic-robust upper bounds exclude a 0.30-SD advantage over the frozen SBERT baseline."
    else:
        disposition = "M1_VERTICAL_SLICE_INCONCLUSIVE"
        rationale = "The valid pipeline neither supports the frozen interaction signal nor rules out a 0.30-SD advantage."

    results = {
        "terminal_disposition": disposition,
        "disposition_rationale": rationale,
        "claim_boundary": (
            "This nonconfirmatory development vertical slice tests linear information "
            "in one joint pre-answer state. It does not open the audited confirmatory "
            "boards, implement the final 107,368-row activation-probe training run, "
            "or establish causality or moral understanding."
        ),
        "selection": {
            "selected_layer": selected_layer,
            "logistic_c": selected_c,
            "criterion": analysis_config["selection_metric"],
            "selection_curve": selection_curve,
            "logistic_c_selection": c_selection,
            "pilot_eval_used_for_selection": False,
            "transfer_used_for_selection": False,
        },
        "evaluations": evaluations,
        "layerwise_difference_in_means": layerwise,
        "held_out_template_transfer": transfer,
        "permutation_nulls": permutations,
        "dyadic_delta_ib_over_sbert": delta_ci,
        "held_out_template_delta_ib_over_sbert": transfer_delta_ci,
        "frozen_text_baseline": sbert_metadata,
        "endpoint_contract": {
            "primary": "standardized reciprocal interaction contrast I_b",
            "secondary": "signed exact-board resolution B_b",
            "descriptive_only": ["pairwise accuracy", "both-ways accuracy"],
            "both_ways_chance": 0.258,
            "practical_effect_sd": practical_lift,
            "inference": "dyadic_robust",
            "scaler_fit_split": "pilot_select",
        },
        "exact_invariants": invariants,
        "validation": validation,
        "pipeline_checks": pipeline_checks,
        "concentration_checks": concentration_checks,
        "board_family_analysis": family_analysis,
        "mapping_controls": mapping_controls,
        "calibration": calibration_results,
        "error_analysis": {
            family: _error_analysis(private_evaluations[family]["_board_metric"])
            for family in ("difference_in_means", "logistic")
        },
        "truth_positive_control": truth_control,
        "sample_counts": {
            "pilot_train": int(train_mask.sum()),
            "pilot_select": int(select_mask.sum()),
            "pilot_eval": int(eval_mask.sum()),
            "pilot_eval_boards": int(
                evaluations["difference_in_means"]["n_boards"]
            ),
        },
    }
    probes = {
        "difference_in_means_direction": dim.direction.astype(np.float32),
        "difference_in_means_midpoint": np.asarray(dim.midpoint, dtype=np.float32),
        "logistic_coef": logistic.model.coef_.astype(np.float32),
        "logistic_intercept": logistic.model.intercept_.astype(np.float32),
        "selected_layer": np.asarray(selected_layer, dtype=np.int16),
        "logistic_c": np.asarray(selected_c, dtype=np.float32),
        "dim_scaler_mu": np.asarray(
            evaluations["difference_in_means"]["score_scaler"]["mu"],
            dtype=np.float64,
        ),
        "dim_scaler_sigma": np.asarray(
            evaluations["difference_in_means"]["score_scaler"]["sigma"],
            dtype=np.float64,
        ),
        "logistic_scaler_mu": np.asarray(
            evaluations["logistic"]["score_scaler"]["mu"], dtype=np.float64
        ),
        "logistic_scaler_sigma": np.asarray(
            evaluations["logistic"]["score_scaler"]["sigma"], dtype=np.float64
        ),
    }
    return results, probes
