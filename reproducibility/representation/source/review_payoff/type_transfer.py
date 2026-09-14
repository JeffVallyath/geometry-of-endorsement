from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .input_audit import load_config, sha256_bytes, sha256_file


TYPE_ORDER = ("Value", "Right", "Duty")


def validate_type_transfer_splits(frame: pd.DataFrame) -> None:
    required = {"row_id", "split", "label", "board_id", "situation_id", "consideration_type", "activation"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Type-transfer frame lacks: {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise RuntimeError("Type-transfer rows are not unique.")
    train = frame.loc[frame["split"].eq("pilot_train")]
    test = frame.loc[frame["split"].eq("pilot_eval")]
    if train.empty or test.empty:
        raise RuntimeError("Type-transfer requires frozen pilot_train and pilot_eval rows.")
    if set(train["row_id"]) & set(test["row_id"]):
        raise RuntimeError("Type-transfer train/eval row IDs overlap.")
    if set(train["situation_id"]) & set(test["situation_id"]):
        raise RuntimeError("Type-transfer train/eval situations overlap.")
    if set(frame["consideration_type"]) != set(TYPE_ORDER):
        raise RuntimeError("Authoritative type mapping must contain Value, Right, and Duty.")
    for split_name, split in (("train", train), ("eval", test)):
        for kind in TYPE_ORDER:
            subset = split.loc[split["consideration_type"].eq(kind)]
            if set(subset["label"].astype(int)) != {0, 1}:
                raise RuntimeError(f"{split_name} {kind} lacks both relation classes.")


def fit_type_directions(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    validate_type_transfer_splits(frame)
    train = frame.loc[frame["split"].eq("pilot_train")]
    result: dict[str, dict[str, Any]] = {}
    for kind in (*TYPE_ORDER, "Global"):
        subset = train if kind == "Global" else train.loc[train["consideration_type"].eq(kind)]
        labels = subset["label"].astype(int).to_numpy()
        activations = np.stack(subset["activation"].to_numpy()).astype(np.float32)
        positive = activations[labels == 1]
        negative = activations[labels == 0]
        direction = positive.mean(axis=0) - negative.mean(axis=0)
        projections_positive = positive @ direction
        projections_negative = negative @ direction
        midpoint = float((projections_positive.mean() + projections_negative.mean()) / 2.0)
        result[kind] = {
            "direction": direction.astype(np.float32),
            "midpoint": midpoint,
            "train_rows": int(len(subset)),
            "train_boards": int(subset["board_id"].nunique()),
            "supports": int((labels == 1).sum()),
            "opposes": int((labels == 0).sum()),
        }
    return result


def bind_frozen_global_direction(
    directions: dict[str, dict[str, Any]],
    probe_bundle: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_revision: str,
    expected_layer: int,
) -> dict[str, Any]:
    path = Path(probe_bundle)
    if sha256_file(path) != expected_bundle_sha256:
        raise RuntimeError("Type-transfer global probe-bundle hash changed.")
    with zipfile.ZipFile(path) as archive:
        arrays = np.load(
            io.BytesIO(archive.read("m1_probe_parameters.npz")), allow_pickle=False
        )
    direction = np.asarray(arrays["difference_in_means_direction"], dtype=np.float32)
    midpoint = float(np.asarray(arrays["difference_in_means_midpoint"]).item())
    if (
        str(np.asarray(arrays["model_revision"]).item()) != expected_revision
        or int(np.asarray(arrays["selected_layer"]).item()) != int(expected_layer)
    ):
        raise RuntimeError("Type-transfer frozen global direction revision/layer changed.")
    fitted = np.asarray(directions["Global"]["direction"], dtype=np.float32)
    cosine = float(direction @ fitted / (np.linalg.norm(direction) * np.linalg.norm(fitted)))
    directions["Global"]["direction"] = direction
    directions["Global"]["midpoint"] = midpoint
    directions["Global"]["source"] = "exact_frozen_global_dim"
    directions["Global"]["cache_refit_cosine_diagnostic"] = cosine
    return {
        "source": "exact_frozen_global_dim",
        "bundle_sha256": expected_bundle_sha256,
        "cache_refit_cosine_diagnostic": cosine,
    }


def direction_cosines(directions: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for left in TYPE_ORDER:
        for right in TYPE_ORDER:
            a = np.asarray(directions[left]["direction"], dtype=np.float32)
            b = np.asarray(directions[right]["direction"], dtype=np.float32)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            rows.append(
                {
                    "train_type_a": left,
                    "train_type_b": right,
                    "cosine_similarity": float(a @ b / denominator) if denominator > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _separation(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(positive) - 1) * np.var(positive, ddof=1) + (len(negative) - 1) * np.var(negative, ddof=1))
        / (len(positive) + len(negative) - 2)
    )
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else np.nan


def _evaluate_once(subset: pd.DataFrame, direction: np.ndarray, midpoint: float) -> dict[str, float | int]:
    activations = np.stack(subset["activation"].to_numpy()).astype(np.float32)
    labels = subset["label"].astype(int).to_numpy()
    scores = activations @ direction - np.float32(midpoint)
    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan
    accuracy = float(np.mean((scores >= 0).astype(int) == labels))
    aligned = np.where(labels == 1, scores, -scores)
    within = subset.assign(prediction=(scores >= 0).astype(int))
    eligible_situations = within.groupby("situation_id")["label"].nunique()
    eligible = within.loc[
        within["situation_id"].isin(eligible_situations[eligible_situations.eq(2)].index)
    ]
    full_boards = within.groupby("board_id").filter(lambda value: len(value) == 4)
    return {
        "rows": int(len(subset)),
        "boards": int(subset["board_id"].nunique()),
        "supports": int(labels.sum()),
        "opposes": int((1 - labels).sum()),
        "auroc": auroc,
        "accuracy": accuracy,
        "aligned_class_separation": _separation(labels, scores),
        "mean_aligned_score": float(aligned.mean()),
        "within_situation_rows": int(len(eligible)),
        "within_situation_accuracy": (
            float(np.mean(eligible["prediction"].astype(int) == eligible["label"].astype(int)))
            if len(eligible)
            else np.nan
        ),
        "full_type_checkerboards": int(full_boards["board_id"].nunique()),
    }


def evaluate_transfer_matrix(
    frame: pd.DataFrame,
    directions: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
    human_clear_row_ids: set[str] | None = None,
) -> pd.DataFrame:
    validate_type_transfer_splits(frame)
    test = frame.loc[frame["split"].eq("pilot_eval")].copy()
    rows = []
    train_directions = (*TYPE_ORDER, "Global")
    for train_index, train_type in enumerate(train_directions):
        direction = np.asarray(directions[train_type]["direction"], dtype=np.float32)
        midpoint = float(directions[train_type]["midpoint"])
        for test_index, test_type in enumerate(TYPE_ORDER):
            subset = test.loc[test["consideration_type"].eq(test_type)].copy()
            point = _evaluate_once(subset, direction, midpoint)
            boards = subset["board_id"].astype(str).unique()
            activations = np.stack(subset["activation"].to_numpy()).astype(np.float32)
            labels = subset["label"].astype(int).to_numpy()
            scores = activations @ direction - np.float32(midpoint)
            _, board_inverse = np.unique(subset["board_id"].astype(str), return_inverse=True)
            rng = np.random.default_rng(seed + 1000 * train_index + test_index)
            draws = {"auroc": [], "aligned_class_separation": [], "accuracy": []}
            for _ in range(bootstrap_replicates):
                counts = np.bincount(
                    rng.choice(len(boards), len(boards), replace=True), minlength=len(boards)
                ).astype(float)
                metrics = _weighted_transfer_metrics(labels, scores, counts[board_inverse])
                for name in draws:
                    if np.isfinite(metrics[name]):
                        draws[name].append(float(metrics[name]))
            row: dict[str, Any] = {
                "train_type": train_type,
                "test_type": test_type,
                **point,
                "train_rows": int(directions[train_type]["train_rows"]),
            }
            for name, values in draws.items():
                low, high = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
                row[f"{name}_ci_low"] = float(low)
                row[f"{name}_ci_high"] = float(high)
                row[f"{name}_bootstrap_valid"] = len(values)
            if human_clear_row_ids is not None:
                clear = subset.loc[subset["row_id"].astype(str).isin(human_clear_row_ids)]
                clear_metrics = _evaluate_once(clear, direction, midpoint) if len(clear) else {}
                row["human_clear_rows"] = int(len(clear))
                row["human_clear_auroc"] = clear_metrics.get("auroc", np.nan)
                row["human_clear_accuracy"] = clear_metrics.get("accuracy", np.nan)
                row["human_clear_aligned_class_separation"] = clear_metrics.get(
                    "aligned_class_separation", np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _weighted_transfer_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    active = weights > 0
    labels = labels[active]
    scores = scores[active]
    weights = weights[active]
    auroc = (
        float(roc_auc_score(labels, scores, sample_weight=weights))
        if len(np.unique(labels)) == 2
        else np.nan
    )
    accuracy = float(np.average((scores >= 0).astype(int) == labels, weights=weights))
    stats = []
    for label in (1, 0):
        mask = labels == label
        count = float(weights[mask].sum())
        if count <= 1:
            stats.append((count, np.nan, np.nan))
            continue
        mean = float(np.average(scores[mask], weights=weights[mask]))
        variance = float(np.sum(weights[mask] * (scores[mask] - mean) ** 2) / (count - 1))
        stats.append((count, mean, variance))
    (n_pos, mean_pos, var_pos), (n_neg, mean_neg, var_neg) = stats
    if n_pos > 1 and n_neg > 1:
        pooled = np.sqrt(((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2))
        separation = float((mean_pos - mean_neg) / pooled) if pooled > 0 else np.nan
    else:
        separation = np.nan
    return {"auroc": auroc, "accuracy": accuracy, "aligned_class_separation": separation}


def load_llama_type_transfer_frame(
    compact_path: str | Path,
    source_bundle: str | Path,
    type_map_path: str | Path,
) -> pd.DataFrame:
    compact = np.load(compact_path, allow_pickle=False)
    compact_frame = pd.DataFrame(
        {
            "row_id": compact["row_id"].astype(str),
            "split": compact["split"].astype(str),
            "board_id": compact["board_id"].astype(str),
            "label": compact["reference_label"].astype(int),
            "activation": list(compact["activation"].astype(np.float32)),
        }
    )
    if len(compact_frame) != 2300 or compact_frame["row_id"].duplicated().any():
        raise RuntimeError("Frozen Llama compact cache topology changed.")
    with zipfile.ZipFile(source_bundle) as archive:
        manifests = []
        for split in ("pilot_train", "pilot_eval"):
            payload = archive.read(f"manifests/{split}.csv")
            manifests.append(pd.read_csv(io.BytesIO(payload), dtype=str).fillna(""))
    source = pd.concat(manifests, ignore_index=True)[
        ["row_id", "situation_id", "board_id", "split", "label"]
    ]
    compact_frame = compact_frame.loc[compact_frame["split"].isin(["pilot_train", "pilot_eval"])]
    merged = compact_frame.merge(
        source,
        on=["row_id", "board_id", "split"],
        suffixes=("", "_source"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or not merged["label"].astype(int).eq(
        merged["label_source"].astype(int)
    ).all():
        raise RuntimeError("Compact activation/source manifest join changed.")
    types = pd.read_csv(type_map_path, dtype=str).fillna("")
    merged = merged.merge(
        types[["row_id", "consideration_type"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["consideration_type"].isna().any():
        raise RuntimeError("Authoritative type metadata is incomplete.")
    result = merged[
        ["row_id", "split", "label", "board_id", "situation_id", "consideration_type", "activation"]
    ].copy()
    validate_type_transfer_splits(result)
    return result


def load_extracted_type_transfer_frame(
    activation_path: str | Path,
    source_bundle: str | Path,
    type_map_path: str | Path,
) -> pd.DataFrame:
    arrays = np.load(activation_path, allow_pickle=False)
    activation_frame = pd.DataFrame(
        {
            "row_id": np.asarray(arrays["row_id"]).astype(str),
            "split": np.asarray(arrays["split"]).astype(str),
            "activation": list(np.asarray(arrays["activation"], dtype=np.float32)),
        }
    )
    if len(activation_frame) != 2300 or activation_frame["row_id"].duplicated().any():
        raise RuntimeError("Extracted Claim 1 activation topology changed.")
    with zipfile.ZipFile(source_bundle) as archive:
        manifests = []
        for split in ("pilot_train", "pilot_eval"):
            manifests.append(
                pd.read_csv(io.BytesIO(archive.read(f"manifests/{split}.csv")), dtype=str).fillna("")
            )
    source = pd.concat(manifests, ignore_index=True)[
        ["row_id", "situation_id", "board_id", "split", "label"]
    ]
    frame = activation_frame.merge(
        source,
        on=["row_id", "split"],
        how="inner",
        validate="one_to_one",
    )
    types = pd.read_csv(type_map_path, dtype=str).fillna("")
    frame = frame.merge(
        types[["row_id", "consideration_type"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    result = frame[
        ["row_id", "split", "label", "board_id", "situation_id", "consideration_type", "activation"]
    ].copy()
    validate_type_transfer_splits(result)
    return result


def run_llama_type_transfer(repo_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = repo / config["paths"]["output_root"]
    contract_path = output / "STUDY_CONTRACT.json"
    amendment_path = output / "STUDY_CONTRACT_AMENDMENT_V1.json"
    if not contract_path.is_file() or not amendment_path.is_file():
        raise RuntimeError("The base study contract and outcome-blind amendment must be frozen first.")
    llama_source = Path(config["paths"]["llama_source_bundle"])
    if not llama_source.is_absolute():
        llama_source = repo / llama_source
    frame = load_llama_type_transfer_frame(
        repo / config["paths"]["llama_compact"],
        llama_source,
        output / "frozen_valueprism_type_map.csv",
    )
    human_cells = pd.read_csv(output / "claim1_human_strata_cells.csv", dtype=str).fillna("")
    both_clear = set(
        human_cells.loc[human_cells["ordered_clarity_stratum"].eq("BOTH_CLEAR"), "row_id"].astype(str)
    )
    directions = fit_type_directions(frame)
    llama_probe = Path(config["paths"]["llama_probe_bundle"])
    if not llama_probe.is_absolute():
        llama_probe = repo / llama_probe
    global_binding = bind_frozen_global_direction(
        directions,
        llama_probe,
        expected_bundle_sha256=config["expected_inputs"]["llama_probe_bundle_sha256"],
        expected_revision=config["models"]["llama"]["revision"],
        expected_layer=int(config["models"]["llama"]["selected_layer"]),
    )
    stats = config["statistics"]
    matrix = evaluate_transfer_matrix(
        frame,
        directions,
        bootstrap_replicates=int(stats["bootstrap_replicates"]),
        seed=int(stats["seed"]) + 50000,
        human_clear_row_ids=both_clear,
    )
    matrix.insert(0, "model", "llama")
    cosines = direction_cosines(directions)
    cosines.insert(0, "model", "llama")
    interpretation = _interpret_matrix(matrix, cosines, json.loads(amendment_path.read_text(encoding="utf-8"))["type_transfer"])
    matrix_path = output / "type_transfer_matrix_frozen_global_v2.csv"
    cosine_path = output / "type_direction_cosines_frozen_global_v2.csv"
    _write_once(
        matrix_path,
        matrix.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    _write_once(
        cosine_path,
        cosines.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    report = _type_report(matrix, cosines, interpretation)
    report_sha = _write_once(
        output / "TYPE_TRANSFER_REPORT_FROZEN_GLOBAL_V2.md", report.encode("utf-8")
    )
    return {
        "status": "LLAMA_TYPE_TRANSFER_FROZEN_GLOBAL_V2_COMPLETE",
        "interpretation": interpretation,
        "global_direction_binding": global_binding,
        "report_sha256": report_sha,
        "matrix_sha256": sha256_bytes(matrix_path.read_bytes()),
        "cosines_sha256": sha256_bytes(cosine_path.read_bytes()),
        "supersedes_refit_global_report": "TYPE_TRANSFER_REPORT.md",
        "gemma": "PENDING_MISSING_ACTIVATION_EXTRACTION",
    }


def run_gemma_type_transfer(repo_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Run the frozen Gemma replication from the batch-one Claim 1 cache."""
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = repo / config["paths"]["output_root"]
    amendment_path = output / "STUDY_CONTRACT_AMENDMENT_V1.json"
    activation_path = output / "checkpoints" / "gemma_claim1_activations.npz"
    if not activation_path.is_file():
        raise RuntimeError(
            "Gemma activation extraction is absent; run the frozen Gemma causal model phase first."
        )
    llama_source = Path(config["paths"]["llama_source_bundle"])
    if not llama_source.is_absolute():
        llama_source = repo / llama_source
    frame = load_extracted_type_transfer_frame(
        activation_path,
        llama_source,
        output / "frozen_valueprism_type_map.csv",
    )
    human_cells = pd.read_csv(output / "claim1_human_strata_cells.csv", dtype=str).fillna("")
    both_clear = set(
        human_cells.loc[
            human_cells["ordered_clarity_stratum"].eq("BOTH_CLEAR"), "row_id"
        ].astype(str)
    )
    directions = fit_type_directions(frame)
    gemma_probe = Path(config["paths"]["gemma_probe_bundle"])
    if not gemma_probe.is_absolute():
        gemma_probe = repo / gemma_probe
    global_binding = bind_frozen_global_direction(
        directions,
        gemma_probe,
        expected_bundle_sha256=config["expected_inputs"]["gemma_probe_bundle_sha256"],
        expected_revision=config["models"]["gemma"]["revision"],
        expected_layer=int(config["models"]["gemma"]["selected_layer"]),
    )
    stats = config["statistics"]
    matrix = evaluate_transfer_matrix(
        frame,
        directions,
        bootstrap_replicates=int(stats["bootstrap_replicates"]),
        seed=int(stats["seed"]) + 60000,
        human_clear_row_ids=both_clear,
    )
    matrix.insert(0, "model", "gemma")
    cosines = direction_cosines(directions)
    cosines.insert(0, "model", "gemma")
    interpretation = _interpret_matrix(
        matrix,
        cosines,
        json.loads(amendment_path.read_text(encoding="utf-8"))["type_transfer"],
    )
    matrix_path = output / "type_transfer_matrix_gemma.csv"
    cosine_path = output / "type_direction_cosines_gemma.csv"
    _write_once(
        matrix_path,
        matrix.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    _write_once(
        cosine_path,
        cosines.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    lines = [
        "# Value / Right / Duty cross-transfer: Gemma replication",
        "",
        "Status: development-only frozen-layer analysis. No Claim 2 behavioral outcome was used for fitting or evaluation.",
        "",
        f"Prespecified disposition: **{interpretation['disposition']}**.",
        "",
        f"Mean diagonal AUROC: {interpretation['mean_diagonal_auroc']:.3f}; mean off-diagonal AUROC: {interpretation['mean_off_diagonal_auroc']:.3f}; global-DIM mean AUROC: {interpretation['mean_global_auroc']:.3f}.",
        f"Minimum pairwise type-direction cosine: {interpretation['minimum_pairwise_cosine']:.3f}.",
        "",
        "| Train type | Test type | n | AUROC | 95% CI | Separation | Accuracy | Human-clear AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix.to_dict("records"):
        lines.append(
            f"| {row['train_type']} | {row['test_type']} | {row['rows']} | {row['auroc']:.3f} | "
            f"[{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] | "
            f"{row['aligned_class_separation']:.3f} | {row['accuracy']:.3f} | "
            f"{row['human_clear_auroc']:.3f} |"
        )
    report_path = output / "TYPE_TRANSFER_GEMMA_REPORT.md"
    report_sha = _write_once(report_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return {
        "status": "GEMMA_TYPE_TRANSFER_COMPLETE",
        "interpretation": interpretation,
        "global_direction_binding": global_binding,
        "matrix_sha256": sha256_bytes(matrix_path.read_bytes()),
        "cosines_sha256": sha256_bytes(cosine_path.read_bytes()),
        "report_sha256": report_sha,
    }


def _interpret_matrix(
    matrix: pd.DataFrame,
    cosines: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    typed = matrix.loc[matrix["train_type"].isin(TYPE_ORDER)].copy()
    diagonal = typed.loc[typed["train_type"].eq(typed["test_type"])]
    off_diagonal = typed.loc[typed["train_type"].ne(typed["test_type"])]
    global_rows = matrix.loc[matrix["train_type"].eq("Global")]
    unique_cosines = cosines.loc[
        cosines["train_type_a"].map(TYPE_ORDER.index)
        < cosines["train_type_b"].map(TYPE_ORDER.index)
    ]
    mean_diagonal = float(diagonal["auroc"].mean())
    mean_off = float(off_diagonal["auroc"].mean())
    mean_global = float(global_rows["auroc"].mean())
    minimum_cosine = float(unique_cosines["cosine_similarity"].min())
    shared = minimum_cosine >= 0.70 and mean_diagonal - mean_off <= 0.05
    multidimensional = (
        mean_diagonal - mean_off >= 0.10
        and int((diagonal["auroc_ci_low"] > 0.50).sum()) >= 2
    )
    data_limited = mean_global - float(typed["auroc"].mean()) >= 0.05
    weak = mean_global <= 0.55 and float(typed["auroc"].mean()) <= 0.55
    if shared:
        disposition = "SHARED_CONTEXTUAL_VALENCE_AXIS"
    elif multidimensional:
        disposition = "MULTIDIMENSIONAL_OR_TYPE_SPECIFIC_RELATION_STRUCTURE"
    elif data_limited:
        disposition = "GROUP_SPECIFIC_ESTIMATES_ARE_DATA_LIMITED"
    elif weak:
        disposition = "CLAIM1_GENERALITY_IS_WEAK"
    else:
        disposition = "TYPE_TRANSFER_INCONCLUSIVE"
    return {
        "disposition": disposition,
        "mean_diagonal_auroc": mean_diagonal,
        "mean_off_diagonal_auroc": mean_off,
        "mean_global_auroc": mean_global,
        "minimum_pairwise_cosine": minimum_cosine,
        "shared_rule_met": shared,
        "multidimensional_rule_met": multidimensional,
        "data_limited_rule_met": data_limited,
        "weak_rule_met": weak,
        "thresholds": dict(thresholds),
    }


def _type_report(matrix: pd.DataFrame, cosines: pd.DataFrame, result: Mapping[str, Any]) -> str:
    lines = [
        "# Value / Right / Duty cross-transfer report",
        "",
        "Status: development-only frozen-layer analysis. No Claim 2 behavioral outcome was loaded.",
        "",
        f"Prespecified disposition: **{result['disposition']}**.",
        "",
        f"Mean diagonal AUROC: {result['mean_diagonal_auroc']:.3f}; mean off-diagonal AUROC: {result['mean_off_diagonal_auroc']:.3f}; global-DIM mean AUROC: {result['mean_global_auroc']:.3f}.",
        f"Minimum pairwise type-direction cosine: {result['minimum_pairwise_cosine']:.3f}.",
        "",
        "## Held-out matrix (Llama, selected layer 19)",
        "",
        "| Train type | Test type | n | AUROC | 95% CI | Separation | Accuracy | Human-clear AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix.to_dict("records"):
        lines.append(
            f"| {row['train_type']} | {row['test_type']} | {row['rows']} | {row['auroc']:.3f} | [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] | {row['aligned_class_separation']:.3f} | {row['accuracy']:.3f} | {row['human_clear_auroc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Availability and claim boundary",
            "",
            "Gemma type transfer remains pending because its row-level frozen activation cache is absent. Any extraction must use the frozen Gemma prompt, revision, layer 27, and batch size one. The Llama matrix alone cannot establish cross-model generality.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen type-transfer artifact differs: {path}")
    else:
        path.write_bytes(payload)
    return sha256_bytes(payload)
