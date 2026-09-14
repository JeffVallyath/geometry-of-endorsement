"""Q3: convert retained intervention outcomes to equal-L2 displacement units."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .control_decoding import load_frozen_metadata
from .grouped_statistics import paired_group_bootstrap_mean
from .input_audit import sha256_file


RELATION_DIRECTION = "residual_dim"
CONTROL_DIRECTIONS = (
    "factual_true_false",
    "sentiment_valence",
    "physical_token",
    "random_orthogonal",
)


def validate_paired_ids(left_ids: np.ndarray, right_ids: np.ndarray) -> None:
    if not np.array_equal(np.asarray(left_ids).astype(str), np.asarray(right_ids).astype(str)):
        raise RuntimeError("INTERVENTION_ID_PAIRING_MISMATCH")


def unit_displacement_norm(vector: np.ndarray, training_scale: float) -> float:
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("intervention direction has invalid norm")
    return float(abs(training_scale) / norm)


def equal_norm_central_slope(
    margin_minus_one: np.ndarray,
    margin_plus_one: np.ndarray,
    displacement_at_unit_dose: float,
) -> np.ndarray:
    if displacement_at_unit_dose <= 0:
        raise ValueError("unit displacement must be positive")
    return (
        np.asarray(margin_plus_one, dtype=np.float64)
        - np.asarray(margin_minus_one, dtype=np.float64)
    ) / (2.0 * displacement_at_unit_dose)


def _load_bundle(runtime_root: Path, actor: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(runtime_root / actor / "directions.npz", allow_pickle=False) as archive:
        vectors = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in ("frozen_dim", "residual_dim", "physical_token", "amperepp_relation", "pooled_external")
        }
        scales = {key: float(value) for key, value in json.loads(str(archive["scales_json"])).items()}
    basis = np.stack(list(vectors.values()))
    rng = np.random.default_rng(20260822)
    random = rng.normal(size=vectors["frozen_dim"].shape[0])
    random -= basis.T @ np.linalg.pinv(basis @ basis.T) @ (basis @ random)
    random /= np.linalg.norm(random)
    vectors["random_orthogonal"] = random
    scales["random_orthogonal"] = scales["frozen_dim"]
    with np.load(
        runtime_root / actor / "controls/control_directions.npz", allow_pickle=False
    ) as archive:
        control_scales = json.loads(str(archive["scales_json"][0]))
        for key in ("factual_true_false", "sentiment_valence"):
            vectors[key] = np.asarray(archive[key], dtype=np.float64)
            scales[key] = float(control_scales[key])
    return vectors, scales


def _folder_for_direction(runtime_root: Path, actor: str, direction: str) -> Path:
    if direction in {"factual_true_false", "sentiment_valence"}:
        return runtime_root / actor / "controls/intervention"
    return runtime_root / actor / "intervention"


def _load_dose_file(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt.get("sha256") != sha256_file(path) or receipt.get("complete", True) is False:
        raise RuntimeError(f"INTERVENTION_RECEIPT_INVALID: {path}")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["row_ids"].astype(str)
        margins = np.asarray(archive["semantic_margin"], dtype=np.float64)
        dose = float(np.asarray(archive["dose"]).reshape(-1)[0])
    return ids, margins, dose


def _direction_matrices(
    runtime_root: Path,
    actor: str,
    dataset: str,
    direction: str,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = _folder_for_direction(runtime_root, actor, direction)
    paths = sorted(
        folder.glob(f"{dataset}__{mapping}__{direction}__*.npz"),
        key=lambda path: float(path.stem.split("__")[-1]),
    )
    if not paths:
        raise RuntimeError(f"INTERVENTION_DIRECTION_MISSING: {actor} {dataset} {direction}")
    ids = None
    doses = []
    margins = []
    for path in paths:
        current_ids, current_margins, dose = _load_dose_file(path)
        if ids is None:
            ids = current_ids
        elif not np.array_equal(ids, current_ids):
            raise RuntimeError(f"INTERVENTION_ROW_DRIFT: {path}")
        doses.append(dose)
        margins.append(current_margins)
    return np.asarray(ids), np.asarray(doses), np.stack(margins)


def _central_by_row(
    runtime_root: Path,
    actor: str,
    dataset: str,
    direction: str,
    unit_norm: float,
) -> dict[str, dict[str, float]]:
    by_row: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"per_sd": [], "equal_norm": []}
    )
    for mapping in ("MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED"):
        ids, doses, matrix = _direction_matrices(
            runtime_root, actor, dataset, direction, mapping
        )
        if set(doses.tolist()) < {-1.0, 0.0, 1.0}:
            raise RuntimeError(f"CENTRAL_DOSE_GRID_INCOMPLETE: {actor} {dataset} {direction}")
        minus = matrix[np.flatnonzero(doses == -1.0)[0]]
        plus = matrix[np.flatnonzero(doses == 1.0)[0]]
        per_sd = (plus - minus) / 2.0
        equal = equal_norm_central_slope(minus, plus, unit_norm)
        for rid, original, normalized in zip(ids, per_sd, equal, strict=True):
            by_row[str(rid)]["per_sd"].append(float(original))
            by_row[str(rid)]["equal_norm"].append(float(normalized))
    return {
        rid: {
            "per_sd": float(np.mean(values["per_sd"])),
            "equal_norm": float(np.mean(values["equal_norm"])),
        }
        for rid, values in by_row.items()
    }


def _linearity_diagnostic(
    runtime_root: Path, actor: str, dataset: str
) -> dict[str, Any]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for mapping in ("MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED"):
        ids, doses, matrix = _direction_matrices(
            runtime_root, actor, dataset, RELATION_DIRECTION, mapping
        )
        for column, rid in enumerate(ids):
            prediction = np.polyval(np.polyfit(doses, matrix[:, column], 1), doses)
            residual_ss = float(np.sum((matrix[:, column] - prediction) ** 2))
            total_ss = float(np.sum((matrix[:, column] - matrix[:, column].mean()) ** 2))
            r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else 1.0
            full_slope = float(np.polyfit(doses, matrix[:, column], 1)[0])
            rows[str(rid)].append((r2, full_slope))
    return {
        "dose_grid": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "mean_row_r_squared": float(
            np.mean([np.mean([value[0] for value in values]) for values in rows.values()])
        ),
        "median_row_r_squared": float(
            np.median([np.mean([value[0] for value in values]) for values in rows.values()])
        ),
        "mean_full_grid_per_sd_slope": float(
            np.mean([np.mean([value[1] for value in values]) for values in rows.values()])
        ),
        "rows": int(len(rows)),
    }


def run_equal_norm_audit(
    runtime_root: Path,
    study_root: Path,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    metadata = load_frozen_metadata(study_root)
    vp_groups = metadata["vp_eval"].set_index("row_id")["board_id"].astype(str).to_dict()
    amp_groups = metadata["datasets"]["AMPERE++"]["groups_by_id"]
    actors = {}
    for actor_index, actor in enumerate(("llama", "gemma")):
        vectors, scales = _load_bundle(runtime_root, actor)
        scale_audit = {}
        for direction in (RELATION_DIRECTION,) + CONTROL_DIRECTIONS:
            vector = vectors[direction]
            raw_norm = float(np.linalg.norm(vector))
            displacement = unit_displacement_norm(vector, scales[direction])
            scale_audit[direction] = {
                "raw_vector_norm": raw_norm,
                "stored_training_projection_sd": float(scales[direction]),
                "stored_vector_is_unit_norm": bool(np.isclose(raw_norm, 1.0, rtol=1e-5, atol=1e-6)),
                "actual_l2_delta_at_plus_one": displacement,
                "actual_l2_delta_at_minus_one": displacement,
                "normalization_formula": "delta = vector * scale * dose / dot(vector, vector)",
            }
        relation_unit = scale_audit[RELATION_DIRECTION]["actual_l2_delta_at_plus_one"]
        for direction in scale_audit:
            scale_audit[direction]["unit_displacement_ratio_to_relation"] = (
                scale_audit[direction]["actual_l2_delta_at_plus_one"] / relation_unit
            )
        datasets = {}
        for dataset_index, dataset in enumerate(("ValuePrism", "AMPERE++")):
            groups_by_id = vp_groups if dataset == "ValuePrism" else amp_groups
            slopes = {
                direction: _central_by_row(
                    runtime_root,
                    actor,
                    dataset,
                    direction,
                    scale_audit[direction]["actual_l2_delta_at_plus_one"],
                )
                for direction in (RELATION_DIRECTION,) + CONTROL_DIRECTIONS
            }
            common = set(slopes[RELATION_DIRECTION])
            for direction in CONTROL_DIRECTIONS:
                if set(slopes[direction]) != common:
                    raise RuntimeError(
                        f"INTERVENTION_ID_PAIRING_MISMATCH: {actor} {dataset} {direction}"
                    )
            ordered = np.asarray(sorted(common))
            groups = np.asarray([groups_by_id[str(rid)] for rid in ordered])
            summaries = {}
            relation_values = slopes[RELATION_DIRECTION]
            for control_index, control in enumerate(CONTROL_DIRECTIONS):
                original_delta = np.asarray(
                    [relation_values[rid]["per_sd"] - slopes[control][rid]["per_sd"] for rid in ordered]
                )
                normalized_delta = np.asarray(
                    [relation_values[rid]["equal_norm"] - slopes[control][rid]["equal_norm"] for rid in ordered]
                )
                summaries[control] = {
                    "relation_mean_per_sd_slope": float(
                        np.mean([relation_values[rid]["per_sd"] for rid in ordered])
                    ),
                    "control_mean_per_sd_slope": float(
                        np.mean([slopes[control][rid]["per_sd"] for rid in ordered])
                    ),
                    "original_per_sd_relation_minus_control": paired_group_bootstrap_mean(
                        original_delta,
                        groups,
                        replicates=bootstrap_replicates,
                        seed=seed + actor_index * 100 + dataset_index * 20 + control_index,
                    ),
                    "relation_mean_equal_norm_slope": float(
                        np.mean([relation_values[rid]["equal_norm"] for rid in ordered])
                    ),
                    "control_mean_equal_norm_slope": float(
                        np.mean([slopes[control][rid]["equal_norm"] for rid in ordered])
                    ),
                    "equal_norm_relation_minus_control": paired_group_bootstrap_mean(
                        normalized_delta,
                        groups,
                        replicates=bootstrap_replicates,
                        seed=seed + actor_index * 100 + dataset_index * 20 + control_index + 10,
                    ),
                    "paired_row_ids_sha256": __import__("hashlib").sha256(
                        "\n".join(ordered).encode("utf-8")
                    ).hexdigest(),
                }
            datasets[dataset] = {
                "controls": summaries,
                "linearity": _linearity_diagnostic(runtime_root, actor, dataset),
            }
        actors[actor] = {"scale_audit": scale_audit, "datasets": datasets}
    all_conditions = [
        actors[actor]["datasets"][dataset]["controls"][control][
            "equal_norm_relation_minus_control"
        ]["low"]
        for actor in actors
        for dataset in ("ValuePrism", "AMPERE++")
        for control in ("factual_true_false", "sentiment_valence")
    ]
    if all(value <= 0 for value in all_conditions):
        disposition = "CAUSAL_NONSPECIFICITY_SURVIVES_EQUAL_NORM"
    else:
        original_failed = all(
            actors[actor]["datasets"][dataset]["controls"][control][
                "original_per_sd_relation_minus_control"
            ]["low"]
            <= 0
            for actor in actors
            for dataset in ("ValuePrism", "AMPERE++")
            for control in ("factual_true_false", "sentiment_valence")
        )
        disposition = (
            "ORIGINAL_FAILURE_EXPLAINED_BY_SCALE"
            if original_failed and all(value > 0 for value in all_conditions)
            else "EQUAL_NORM_RESULT_MIXED"
        )
    return {
        "schema_version": 1,
        "question": "Q3_EQUAL_NORM_CAUSAL_COMPARISON",
        "actors": actors,
        "disposition": disposition,
        "frozen_headline_changed": False,
    }
