from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GEOMETRY_KEYS = ("claim1_board_id", "claim1_cell_field", "claim1_source_row_id")
STABLE_JOIN_COLUMNS = (
    "base_item_id",
    "board_id",
    "cell_role",
    "source_row_id",
    "claim1_review_id",
    "claim1_cell_field",
    "valueprism_relation",
    "stratum",
)


def exact_geometry_join(items: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    missing_left = set(GEOMETRY_KEYS) - set(items.columns)
    missing_right = set(GEOMETRY_KEYS) - set(geometry.columns)
    if missing_left or missing_right:
        raise ValueError(f"Missing geometry join keys: items={sorted(missing_left)}, geometry={sorted(missing_right)}")
    if geometry.duplicated(list(GEOMETRY_KEYS)).any():
        raise RuntimeError("A frozen source cell maps to more than one geometry row.")
    merged = items.merge(geometry, on=list(GEOMETRY_KEYS), how="left", validate="many_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), "base_item_id"].astype(str).tolist()
        raise RuntimeError(f"Stable-ID geometry join failed for {missing}")
    if merged.duplicated("base_item_id").any():
        raise RuntimeError("An original item maps to more than one geometry score.")
    return merged.drop(columns="_merge")


def mapping_invariance(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"base_item_id", "mapping", "geometry_score"}
    if required - set(scores.columns):
        raise ValueError(f"Missing mapping-invariance columns: {sorted(required - set(scores.columns))}")
    pivot = scores.pivot(index="base_item_id", columns="mapping", values="geometry_score")
    expected = {"ab_standard", "ab_reversed", "12_standard", "12_reversed"}
    if not expected <= set(pivot.columns):
        raise RuntimeError("All four frozen geometry mappings are required.")
    result = pivot.reset_index()
    values = pivot[list(sorted(expected))]
    result["mapping_average_geometry"] = values.mean(axis=1).to_numpy()
    result["geometry_mapping_variance"] = values.var(axis=1, ddof=0).to_numpy()
    result["geometry_sign_agreement"] = values.apply(lambda row: len(set(np.sign(row))) == 1, axis=1).to_numpy()
    result["geometry_absolute_score_range"] = (values.abs().max(axis=1) - values.abs().min(axis=1)).to_numpy()
    return result


def range_summary(frame: pd.DataFrame, feature: str, group: str) -> pd.DataFrame:
    rows = []
    for name, subset in frame.groupby(group, dropna=False, sort=False):
        values = subset[feature].dropna().astype(float)
        rows.append({
            group: name,
            "feature": feature,
            "n": len(values),
            "sd": float(values.std(ddof=0)),
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "fraction_near_boundary": float(values.abs().le(values.abs().quantile(0.25)).mean()),
        })
    return pd.DataFrame(rows)


def _row_vector_sha256(vector: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(vector)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def audit_frozen_claim1_mapping(
    stable_join: pd.DataFrame,
    owner_base_manifest: pd.DataFrame,
    compact_npz: str | Path,
    probe_npz: str | Path,
    *,
    model_revision: str,
    selected_layer: int,
    compact_artifact_sha256: str,
    probe_artifact_sha256: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify every Claim 2 original against its exact frozen Claim 1 source row.

    This function performs no text or fuzzy matching and fits no direction.  It
    only replays the already-frozen scalar calculation from retained activation
    and probe bytes.
    """
    if tuple(stable_join.columns) != STABLE_JOIN_COLUMNS:
        raise RuntimeError("The stable Claim 1-to-Claim 2 join schema changed.")
    if len(stable_join) != 112 or stable_join["base_item_id"].duplicated().any():
        raise RuntimeError("The stable join must contain 112 unique originals.")
    required_owner = {
        "base_item_id", "item_id", "row_id", "board_id", "stratum",
        "cell_role", "valueprism_relation", "raw_dim_margin",
        "standardized_signed_dim_margin", "dim_weakness",
        "raw_logistic_margin", "native_margin", "native_confidence",
        "rendered_prompt_sha256",
    }
    missing_owner = required_owner - set(owner_base_manifest.columns)
    if missing_owner:
        raise RuntimeError(f"Owner base manifest is missing {sorted(missing_owner)}")
    if len(owner_base_manifest) != 112 or owner_base_manifest["base_item_id"].duplicated().any():
        raise RuntimeError("Owner base manifest must contain 112 unique originals.")
    joined = stable_join.merge(
        owner_base_manifest,
        on="base_item_id",
        how="left",
        validate="one_to_one",
        suffixes=("_join", "_owner"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("Stable join and owner base manifest IDs differ.")
    exact_fields = {
        "board_id": joined["board_id_join"].astype(str).eq(joined["board_id_owner"].astype(str)),
        "source_row_id": joined["source_row_id"].astype(str).eq(joined["row_id"].astype(str)),
        "cell_role": joined["cell_role_join"].astype(str).eq(joined["cell_role_owner"].astype(str)),
        "stored_relation": joined["valueprism_relation_join"].astype(str).str.casefold().eq(
            joined["valueprism_relation_owner"].astype(str).str.casefold()
        ),
        "stratum": joined["stratum_join"].astype(str).eq(joined["stratum_owner"].astype(str)),
    }
    failures = {name: int((~mask).sum()) for name, mask in exact_fields.items()}
    if any(failures.values()):
        raise RuntimeError(f"Stable source-field mapping mismatch: {failures}")

    with np.load(compact_npz, allow_pickle=False) as compact, np.load(probe_npz, allow_pickle=False) as probe:
        required_arrays = {
            "item_id", "row_id", "split", "board_id", "mapping_name",
            "rendered_prompt_sha256", "activation", "prompt_token_index",
        }
        if required_arrays - set(compact.files):
            raise RuntimeError("Compact Claim 1 cache lacks required audit arrays.")
        if len(set(map(str, compact["row_id"]))) != len(compact["row_id"]):
            raise RuntimeError("Compact Claim 1 source-row IDs are not unique.")
        row_index = {str(value): index for index, value in enumerate(compact["row_id"])}
        missing_rows = sorted(set(joined["source_row_id"].astype(str)) - set(row_index))
        if missing_rows:
            raise RuntimeError(f"Stable source rows are absent from compact cache: {missing_rows}")
        direction = probe["difference_in_means_direction"].astype(np.float32)
        midpoint = float(probe["difference_in_means_midpoint"])
        dim_mu = float(probe["dim_scaler_mu"])
        dim_sigma = float(probe["dim_scaler_sigma"])
        logistic_coef = probe["logistic_coef"].astype(np.float32).reshape(-1)
        logistic_intercept = float(probe["logistic_intercept"].reshape(-1)[0])
        probe_layer = int(probe["selected_layer"])
        if probe_layer != int(selected_layer):
            raise RuntimeError("Configured selected layer differs from frozen probe layer.")
        compact_indices = [
            row_index[str(value)] for value in joined["source_row_id"]
        ]
        activation_stored_batch = compact["activation"][compact_indices]
        activation_batch = activation_stored_batch.astype(np.float32)
        # Replay the owner-manifest construction as one 112-row matrix
        # operation. Per-row dot products can select a different BLAS kernel
        # and differ at the micro-unit level even with identical float32 bytes.
        raw_dim_batch = activation_batch @ direction - midpoint
        signed_dim_batch = (raw_dim_batch - dim_mu) / dim_sigma
        logistic_batch = (
            activation_batch @ logistic_coef + logistic_intercept
        )
        rows: list[dict[str, object]] = []
        for position, record in enumerate(joined.to_dict("records")):
            index = compact_indices[position]
            activation_stored = activation_stored_batch[position]
            raw_dim = float(raw_dim_batch[position])
            signed_dim = float(signed_dim_batch[position])
            dim_weakness = float(-abs(signed_dim))
            logistic_margin = float(logistic_batch[position])
            audit_row = {
                "base_item_id": str(record["base_item_id"]),
                "claim1_board_id": str(record["board_id_join"]),
                "claim1_situation_position": str(record["cell_role_join"])[:2],
                "claim1_consideration_position": str(record["cell_role_join"])[3:4],
                "claim1_cell_role": str(record["cell_role_join"]),
                "claim1_cell_field": str(record["claim1_cell_field"]),
                "claim1_source_row_id": str(record["source_row_id"]),
                "stored_relation": str(record["valueprism_relation_join"]),
                "compact_item_id": str(compact["item_id"][index]),
                "compact_split": str(compact["split"][index]),
                "compact_board_id": str(compact["board_id"][index]),
                "model_revision": model_revision,
                "frozen_compact_artifact_sha256": compact_artifact_sha256,
                "frozen_probe_artifact_sha256": probe_artifact_sha256,
                "frozen_vector_sha256": _row_vector_sha256(activation_stored),
                "selected_layer": int(selected_layer),
                "claim1_prompt_contract": "frozen_m1_primary",
                "claim1_answer_mapping": str(compact["mapping_name"][index]),
                "claim1_rendered_prompt_sha256": str(compact["rendered_prompt_sha256"][index]),
                "claim1_prompt_token_index": int(compact["prompt_token_index"][index]),
                "activation_token_position": "final_nonpadding_prompt_token_before_assistant_answer",
                "score_normalization": "negative_absolute_pilot_select_standardized_DIM_margin",
                "recomputed_raw_dim_margin": raw_dim,
                "stored_raw_dim_margin": float(record["raw_dim_margin"]),
                "recomputed_signed_dim_margin": signed_dim,
                "stored_signed_dim_margin": float(record["standardized_signed_dim_margin"]),
                "recomputed_dim_weakness": dim_weakness,
                "stored_dim_weakness": float(record["dim_weakness"]),
                "recomputed_logistic_margin": logistic_margin,
                "stored_logistic_margin": float(record["raw_logistic_margin"]),
            }
            rows.append(audit_row)
    result = pd.DataFrame(rows).sort_values("base_item_id").reset_index(drop=True)
    if not result["compact_split"].eq("pilot_eval").all():
        raise RuntimeError("A Claim 2 original maps outside the frozen pilot-eval cache.")
    if not result["claim1_board_id"].eq(result["compact_board_id"]).all():
        raise RuntimeError("Claim 1 board identity differs between join and compact cache.")
    tolerances = {
        "raw_dim": float(np.max(np.abs(result["recomputed_raw_dim_margin"] - result["stored_raw_dim_margin"]))),
        "signed_dim": float(np.max(np.abs(result["recomputed_signed_dim_margin"] - result["stored_signed_dim_margin"]))),
        "dim_weakness": float(np.max(np.abs(result["recomputed_dim_weakness"] - result["stored_dim_weakness"]))),
        "logistic_margin": float(np.max(np.abs(result["recomputed_logistic_margin"] - result["stored_logistic_margin"]))),
    }
    # Stored public-facing CSV values are decimal serializations of float32
    # computations.  One micro-unit is a strict serialization tolerance, not a
    # fitted or outcome-selected scientific threshold.
    if any(value > 1e-6 for value in tolerances.values()):
        raise RuntimeError(f"Frozen geometry scalar replay differs from owner manifest: {tolerances}")
    report: dict[str, object] = {
        "status": "FROZEN_CLAIM1_TO_CLAIM2_GEOMETRY_MAPPING_VERIFIED",
        "rows": len(result),
        "unique_base_item_ids": int(result["base_item_id"].nunique()),
        "unique_source_rows": int(result["claim1_source_row_id"].nunique()),
        "exact_source_field_failures": failures,
        "maximum_serialization_differences": tolerances,
        "model_revision": model_revision,
        "selected_layer": int(selected_layer),
        "compact_artifact_sha256": compact_artifact_sha256,
        "probe_artifact_sha256": probe_artifact_sha256,
        "geometry_recomputed_or_refit": False,
        "new_direction_fit": False,
    }
    return result, report


def compare_claim1_claim2_prompts(
    mapping_audit: pd.DataFrame,
    claim2_geometry_scores: pd.DataFrame,
    *,
    claim1_prompt_specification: dict[str, object],
    claim2_prompt_specification: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Report exact prompt hashes and the complete contract-level differences."""
    required = {"base_item_id", "mapping", "rendered_prompt_sha256", "prompt_token_ids_sha256", "prompt_token_index"}
    missing = required - set(claim2_geometry_scores.columns)
    if missing:
        raise RuntimeError(f"Claim 2 geometry prompt audit lacks {sorted(missing)}")
    left = mapping_audit[["base_item_id", "claim1_rendered_prompt_sha256", "claim1_answer_mapping", "claim1_prompt_token_index"]]
    rows = claim2_geometry_scores.merge(left, on="base_item_id", how="left", validate="many_to_one")
    rows["rendered_prompt_hash_equal_to_claim1"] = rows["rendered_prompt_sha256"].eq(rows["claim1_rendered_prompt_sha256"])
    rows = rows.rename(columns={
        "mapping": "claim2_answer_mapping",
        "rendered_prompt_sha256": "claim2_rendered_prompt_sha256",
        "prompt_token_ids_sha256": "claim2_prompt_token_ids_sha256",
        "prompt_token_index": "claim2_prompt_token_index",
    })
    contract_differences = {
        "claim1_contract_sha256": hashlib.sha256(json.dumps(claim1_prompt_specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "claim2_contract_sha256": hashlib.sha256(json.dumps(claim2_prompt_specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "different_system_template": claim1_prompt_specification.get("system_template") != claim2_prompt_specification.get("system_template"),
        "different_user_template": claim1_prompt_specification.get("user_template") != claim2_prompt_specification.get("user_template"),
        "different_answer_mapping_scheme": claim1_prompt_specification.get("schemes") != claim2_prompt_specification.get("schemes"),
        "different_generation_boundary_rule": claim1_prompt_specification.get("add_generation_prompt") != claim2_prompt_specification.get("add_generation_prompt"),
        "shared_fields_exact": sorted(
            key for key in set(claim1_prompt_specification) & set(claim2_prompt_specification)
            if claim1_prompt_specification[key] == claim2_prompt_specification[key]
        ),
    }
    report: dict[str, object] = {
        "status": "CLAIM1_VERSUS_CLAIM2_PROMPT_CONTRACTS_COMPARED",
        "rows": len(rows),
        "claim2_mappings": sorted(rows["claim2_answer_mapping"].astype(str).unique()),
        "rendered_prompt_hash_matches": int(rows["rendered_prompt_hash_equal_to_claim1"].sum()),
        "rendered_prompt_hash_mismatches": int((~rows["rendered_prompt_hash_equal_to_claim1"]).sum()),
        "contract_differences": contract_differences,
        "interpretation": "Hash inequality is descriptive contract drift, not evidence that either prompt is invalid.",
    }
    return rows, report
