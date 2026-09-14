from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .input_audit import load_config, sha256_file
from .intervention import (
    build_conditions,
    extract_final_prompt_activation,
    load_frozen_dim,
    norm_matched_orthogonal,
    projection_sd_step,
    render_claim1_reason_prompt,
    render_explicit_reason_prompt,
    score_reason_with_intervention,
)


def local_dose_slope(frame: pd.DataFrame) -> float:
    local = frame.loc[frame["dose"].astype(float).isin([-0.5, 0.0, 0.5])]
    x = local["dose"].to_numpy(float)
    y = local["semantic_margin"].to_numpy(float)
    if len(x) < 3 or np.var(x) == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def monotonic_fraction(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("dose")
    changes = np.diff(ordered["semantic_margin"].to_numpy(float))
    return float(np.mean(changes >= 0)) if len(changes) else float("nan")


def summarize_causal_claim1(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "item_id", "board_id", "population", "family", "direction_id", "dose", "semantic_margin"}
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"Causal Claim 1 rows are missing: {sorted(missing)}")
    summaries = []
    group_keys = ["model", "population", "family", "direction_id"]
    for key, group in rows.groupby(group_keys, sort=True):
        per_item = group.groupby("item_id", group_keys=False).apply(
            lambda value: pd.Series(
                {
                    "local_slope": local_dose_slope(value),
                    "monotonic_fraction": monotonic_fraction(value),
                    "verdict_change_rate": float(
                        (np.sign(value["semantic_margin"]) != np.sign(value.loc[value["dose"].abs().idxmin(), "semantic_margin"])).mean()
                    ),
                }
            ),
            include_groups=False,
        )
        summaries.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "items": int(len(per_item)),
                "mean_local_slope": float(per_item["local_slope"].mean()),
                "mean_monotonic_fraction": float(per_item["monotonic_fraction"].mean()),
                "mean_verdict_change_rate": float(per_item["verdict_change_rate"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def signed_rescue_movement(
    original_margin: float,
    baseline_rewrite_margin: float,
    steered_rewrite_margin: float,
) -> float:
    target_sign = 1.0 if float(original_margin) >= 0 else -1.0
    return float(target_sign * (float(steered_rewrite_margin) - float(baseline_rewrite_margin)))


def summarize_robustness_rescue(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "base_item_id",
        "candidate_id",
        "population",
        "family",
        "original_margin",
        "baseline_rewrite_margin",
        "steered_rewrite_margin",
    }
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"Robustness-rescue rows are missing: {sorted(missing)}")
    frame = rows.copy()
    frame["signed_movement_toward_original"] = [
        signed_rescue_movement(a, b, c)
        for a, b, c in zip(
            frame["original_margin"],
            frame["baseline_rewrite_margin"],
            frame["steered_rewrite_margin"],
            strict=True,
        )
    ]
    frame["absolute_gap_reduction"] = (
        (frame["original_margin"] - frame["baseline_rewrite_margin"]).abs()
        - (frame["original_margin"] - frame["steered_rewrite_margin"]).abs()
    )
    frame["baseline_flip"] = np.sign(frame["original_margin"]) != np.sign(
        frame["baseline_rewrite_margin"]
    )
    frame["steered_flip"] = np.sign(frame["original_margin"]) != np.sign(
        frame["steered_rewrite_margin"]
    )
    frame["flip_rescued"] = frame["baseline_flip"] & ~frame["steered_flip"]
    frame["stable_disrupted"] = ~frame["baseline_flip"] & frame["steered_flip"]
    return frame.groupby(["model", "population", "family"], sort=True).agg(
        originals=("base_item_id", "nunique"),
        pairs=("candidate_id", "nunique"),
        mean_signed_movement_toward_original=("signed_movement_toward_original", "mean"),
        mean_absolute_gap_reduction=("absolute_gap_reduction", "mean"),
        flips_rescued=("flip_rescued", "sum"),
        stable_pairs_disrupted=("stable_disrupted", "sum"),
    ).reset_index()


def original_item_bootstrap_contrast(
    rows: pd.DataFrame,
    *,
    treatment_family: str = "dim",
    control_family: str = "random_orthogonal",
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    pivot = rows.pivot_table(
        index="base_item_id",
        columns="family",
        values="signed_movement_toward_original",
        aggfunc="mean",
    ).dropna(subset=[treatment_family, control_family])
    values = (pivot[treatment_family] - pivot[control_family]).to_numpy(float)
    if not len(values):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "originals": 0}
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(replicates)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "originals": int(len(values)),
    }


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Versioned causal artifact differs: {path}")
    else:
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_or_verify_parquet(path: Path, frame: pd.DataFrame) -> str:
    """Write a versioned checkpoint once, or prove an existing one is identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        try:
            pd.testing.assert_frame_equal(
                existing.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise RuntimeError(f"Versioned causal parquet differs: {path}") from exc
    else:
        frame.to_parquet(path, index=False)
    return sha256_file(path)


def _load_prompt_config(repo: Path) -> dict[str, Any]:
    from m1_vertical_slice.config import load_config as load_m1_config

    return load_m1_config(repo / "configs" / "m1_vertical_slice.yaml").section("prompt")


def _claim1_records(repo: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    compact_path = _resolve(repo, str(config["paths"]["llama_compact"]))
    arrays = np.load(compact_path, allow_pickle=False)
    frame = pd.DataFrame(
        {
            "item_id": np.asarray(arrays["item_id"]).astype(str),
            "row_id": np.asarray(arrays["row_id"]).astype(str),
            "split": np.asarray(arrays["split"]).astype(str),
            "board_id": np.asarray(arrays["board_id"]).astype(str),
            "reference_label": np.asarray(arrays["reference_label"]).astype(int),
        }
    )
    types = pd.read_csv(
        _resolve(repo, str(config["paths"]["output_root"])) / "frozen_valueprism_type_map.csv",
        dtype=str,
    ).fillna("")
    frame = frame.merge(types, on="row_id", how="left", validate="one_to_one")
    if frame[["situation", "consideration", "stored_relation"]].isna().any().any():
        raise RuntimeError("Claim 1 causal source join is incomplete.")
    cells = pd.read_csv(
        _resolve(repo, str(config["paths"]["human_derived_root"]))
        / "claim1_consensus_cells.csv",
        dtype=str,
    ).fillna("")
    cells = cells[["row_id", "consensus_clear", "clarity_disputed"]]
    frame = frame.merge(cells, on="row_id", how="left", validate="one_to_one")
    is_eval = frame["split"].eq("pilot_eval")
    if int(is_eval.sum()) != 500 or frame.loc[is_eval, "board_id"].eq("").any():
        raise RuntimeError("Claim 1 held-out evaluation topology changed.")
    frame["human_population"] = np.where(
        frame["consensus_clear"].eq("True"), "BOTH_CLEAR", "DISPUTED"
    )
    return frame


def _load_llama_activation_cache(repo: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve(repo, str(config["paths"]["llama_compact"]))
    arrays = np.load(path, allow_pickle=False)
    activation = np.asarray(arrays["activation"], dtype=np.float32)
    if activation.shape != (2300, int(config["models"]["llama"]["hidden_size"])):
        raise RuntimeError("Frozen Llama activation cache topology changed.")
    return pd.DataFrame(
        {
            "row_id": np.asarray(arrays["row_id"]).astype(str),
            "split": np.asarray(arrays["split"]).astype(str),
            "activation": list(activation),
        }
    )


def _extract_gemma_activation_cache(
    loaded: Any,
    *,
    records: pd.DataFrame,
    prompt_config: Mapping[str, Any],
    selected_layer: int,
    output: Path,
) -> Path:
    checkpoint = output / "checkpoints" / "gemma_claim1_activations.npz"
    if checkpoint.is_file():
        arrays = np.load(checkpoint, allow_pickle=False)
        if (
            len(arrays["row_id"]) != len(records)
            or list(np.asarray(arrays["row_id"]).astype(str)) != list(records["row_id"])
            or np.asarray(arrays["activation"]).shape != (len(records), 3584)
        ):
            raise RuntimeError("Gemma Claim 1 activation checkpoint changed.")
        return checkpoint
    chunks = output / "checkpoints" / "gemma_claim1_activation_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    width = int(loaded.model.config.hidden_size)
    if width != 3584:
        raise RuntimeError("Loaded Gemma hidden width changed.")
    all_ids: list[str] = []
    all_values: list[np.ndarray] = []
    for start in range(0, len(records), 25):
        part = records.iloc[start : start + 25]
        part_path = chunks / f"rows_{start:04d}_{start + len(part) - 1:04d}.npz"
        if part_path.is_file():
            saved = np.load(part_path, allow_pickle=False)
            ids = list(np.asarray(saved["row_id"]).astype(str))
            values = np.asarray(saved["activation"], dtype=np.float32)
            if ids != list(part["row_id"]) or values.shape != (len(part), width):
                raise RuntimeError("Gemma activation chunk does not match its frozen rows.")
        else:
            ids = []
            values_list: list[np.ndarray] = []
            for row in part.itertuples(index=False):
                prompt_ids, _, _, _ = render_claim1_reason_prompt(
                    loaded,
                    item_id=str(row.item_id),
                    situation=str(row.situation),
                    consideration=str(row.consideration),
                    prompt_config=prompt_config,
                )
                values_list.append(
                    extract_final_prompt_activation(
                        loaded,
                        prompt_ids=prompt_ids,
                        layer=selected_layer,
                        batch_size=1,
                    )
                )
                ids.append(str(row.row_id))
            values = np.stack(values_list).astype(np.float16)
            np.savez_compressed(part_path, row_id=np.asarray(ids), activation=values)
        all_ids.extend(ids)
        all_values.extend(np.asarray(values, dtype=np.float32))
    if all_ids != list(records["row_id"]):
        raise RuntimeError("Gemma activation extraction order changed.")
    np.savez_compressed(
        checkpoint,
        row_id=np.asarray(all_ids),
        split=records["split"].to_numpy(str),
        activation=np.stack(all_values).astype(np.float16),
    )
    return checkpoint


def _projection_training_sigma(
    activation: np.ndarray, split: np.ndarray, direction: np.ndarray
) -> float:
    projections = np.asarray(activation, dtype=np.float32) @ np.asarray(direction, dtype=np.float32)
    train = projections[np.asarray(split).astype(str) == "pilot_train"]
    if len(train) != 1500:
        raise RuntimeError("Claim 1 pilot_train calibration population changed.")
    sigma = float(np.std(train, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("Training projection SD is invalid.")
    return sigma


def _condition_step(
    condition: Any,
    *,
    dim_step: np.ndarray,
    direction: np.ndarray,
    orthogonal: Mapping[int, np.ndarray],
) -> np.ndarray:
    if condition.family == "random_orthogonal":
        return np.asarray(orthogonal[int(condition.seed)], dtype=np.float32)
    return np.asarray(dim_step, dtype=np.float32)


def _score_claim1_causal(
    loaded: Any,
    *,
    model_name: str,
    records: pd.DataFrame,
    prompt_config: Mapping[str, Any],
    direction: np.ndarray,
    projection_sigma: float,
    config: Mapping[str, Any],
    output: Path,
) -> pd.DataFrame:
    selected_layer = int(config["models"][model_name]["selected_layer"])
    wrong_layer = int(config["models"][model_name]["wrong_layer"])
    dim_step = projection_sd_step(direction, projection_sigma)
    seeds = [int(value) for value in config["intervention"]["random_orthogonal_seeds"]]
    orthogonal = {
        seed: norm_matched_orthogonal(dim_step, direction, seed) for seed in seeds
    }
    conditions = build_conditions(
        selected_layer=selected_layer,
        wrong_layer=wrong_layer,
        doses=config["intervention"]["doses_training_projection_sd"],
        orthogonal_seeds=seeds,
    )
    chunk_root = output / "checkpoints" / f"causal_claim1_{model_name}_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for start in range(0, len(records), 10):
        part = records.iloc[start : start + 10]
        chunk = chunk_root / f"rows_{start:04d}_{start + len(part) - 1:04d}.parquet"
        if chunk.is_file():
            saved = pd.read_parquet(chunk)
            if set(saved["row_id"].astype(str)) != set(part["row_id"].astype(str)):
                raise RuntimeError("Claim 1 causal checkpoint rows changed.")
            frames.append(saved)
            continue
        rows: list[dict[str, Any]] = []
        for source in part.itertuples(index=False):
            prompt_ids, supports, opposes, mapping_name = render_claim1_reason_prompt(
                loaded,
                item_id=str(source.item_id),
                situation=str(source.situation),
                consideration=str(source.consideration),
                prompt_config=prompt_config,
            )
            for condition in conditions:
                step = _condition_step(
                    condition,
                    dim_step=dim_step,
                    direction=direction,
                    orthogonal=orthogonal,
                )
                score = score_reason_with_intervention(
                    loaded,
                    prompt_ids=prompt_ids,
                    supports_candidate=supports,
                    opposes_candidate=opposes,
                    direction_step=step,
                    condition=condition,
                    batch_size=1,
                )
                rows.append(
                    {
                        "analysis_phase": "claim1_causal",
                        "model": model_name,
                        "item_id": str(source.item_id),
                        "row_id": str(source.row_id),
                        "board_id": str(source.board_id),
                        "population": str(source.human_population),
                        "stored_relation": str(source.stored_relation),
                        "mapping": mapping_name,
                        "family": condition.family,
                        "direction_id": condition.direction_id,
                        "seed": condition.seed,
                        **score.__dict__,
                    }
                )
        frame = pd.DataFrame(rows)
        if not frame["batch_size"].eq(1).all():
            raise RuntimeError("Claim 1 causal scoring violated batch_size=1.")
        frame.to_parquet(chunk, index=False)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(records) * len(conditions)
    if len(result) != expected:
        raise RuntimeError(f"Claim 1 causal result count changed: {len(result)} != {expected}")
    return result


def _rescue_population(repo: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    human = _resolve(repo, str(config["paths"]["human_derived_root"]))
    accepted = pd.read_csv(human / "claim2_strict_consensus_ordinary.csv", dtype=str).fillna("")
    first = pd.read_csv(
        human / "claim1_claim2_first_stage_pair_sensitivity.csv", dtype=str
    ).fillna("")
    first_ids = set(first["candidate_id"].astype(str))
    accepted["stronger_first_stage_sensitivity"] = accepted["candidate_id"].isin(first_ids)
    accepted["population"] = accepted["stratum"].map(
        {"representative": "representative_primary", "disagreement": "enriched_diagnostic"}
    )
    if len(accepted) != 398 or accepted["base_item_id"].nunique() != 112:
        raise RuntimeError("Claim 2 rescue population changed.")
    return accepted


def _verify_rescue_baseline_reproduction(
    repo: Path,
    config: Mapping[str, Any],
    *,
    model_name: str,
    rescue_rows: pd.DataFrame,
    tolerance: float = 0.002,
) -> dict[str, Any]:
    from .verdict_analysis import collapse_reason_scores

    if model_name == "llama":
        frozen_raw = pd.read_csv(
            repo
            / "outputs"
            / "claim2-behavioral-viability"
            / "v005"
            / "inference"
            / "wording_results_private.csv",
            dtype=str,
        ).fillna("")
    elif model_name == "gemma":
        root = _resolve(repo, str(config["paths"]["claim2_behavioral_root"]))
        frozen_raw = pd.concat(
            [
                pd.read_parquet(root / "checkpoints" / "gemma_candidate_behavior.parquet"),
                pd.read_parquet(root / "checkpoints" / "gemma_original_repeat_control.parquet"),
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        raise ValueError(model_name)
    frozen = collapse_reason_scores(frozen_raw, model_name)
    observed = rescue_rows.groupby(
        ["base_item_id", "candidate_id"], as_index=False, sort=True
    )[["original_margin", "baseline_rewrite_margin"]].mean()
    candidate = frozen.loc[frozen["wording_kind"].eq("candidate"), [
        "base_item_id",
        "candidate_id",
        "reason_margin",
    ]]
    candidate = candidate.loc[
        candidate["candidate_id"].astype(str).isin(set(observed["candidate_id"].astype(str)))
    ]
    candidate_join = observed.merge(
        candidate,
        on=["base_item_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    originals = frozen.loc[frozen["wording_kind"].eq("original"), [
        "base_item_id",
        "reason_margin",
    ]].drop_duplicates("base_item_id")
    original_join = observed[["base_item_id", "original_margin"]].drop_duplicates(
        "base_item_id"
    ).merge(originals, on="base_item_id", how="inner", validate="one_to_one")
    if len(candidate_join) != len(candidate) or len(candidate_join) != 234:
        raise RuntimeError("Not every retained frozen candidate baseline resolved by stable ID.")
    if len(original_join) != len(originals):
        raise RuntimeError("Not every retained frozen original baseline resolved by stable ID.")
    candidate_error = float(
        (
            candidate_join["baseline_rewrite_margin"].astype(float)
            - candidate_join["reason_margin"].astype(float)
        ).abs().max()
    )
    original_error = float(
        (
            original_join["original_margin"].astype(float)
            - original_join["reason_margin"].astype(float)
        ).abs().max()
    )
    maximum = max(candidate_error, original_error)
    if maximum > float(tolerance):
        raise RuntimeError(
            "Unsteered causal baseline does not reproduce the frozen batch-size-one reason "
            f"scores within {tolerance}: max_abs_error={maximum}."
        )
    candidate_hard_mismatches = int(
        (
            np.sign(candidate_join["baseline_rewrite_margin"].astype(float))
            != np.sign(candidate_join["reason_margin"].astype(float))
        ).sum()
    )
    original_hard_mismatches = int(
        (
            np.sign(original_join["original_margin"].astype(float))
            != np.sign(original_join["reason_margin"].astype(float))
        ).sum()
    )
    if candidate_hard_mismatches or original_hard_mismatches:
        raise RuntimeError(
            "Unsteered causal baseline changed one or more frozen hard reason verdicts."
        )
    return {
        "tolerance": float(tolerance),
        "candidate_rows_verified": int(len(candidate_join)),
        "original_rows_verified": int(len(original_join)),
        "candidate_max_abs_error": candidate_error,
        "original_max_abs_error": original_error,
        "maximum_abs_error": maximum,
        "hard_verdict_agreement": True,
        "candidate_hard_verdict_mismatches": candidate_hard_mismatches,
        "original_hard_verdict_mismatches": original_hard_mismatches,
        "unpreviously_scored_accepted_pairs": int(
            observed["candidate_id"].nunique() - len(candidate_join)
        ),
    }


def _score_rescue(
    loaded: Any,
    *,
    model_name: str,
    records: pd.DataFrame,
    prompt_config: Mapping[str, Any],
    direction: np.ndarray,
    projection_sigma: float,
    config: Mapping[str, Any],
    output: Path,
) -> pd.DataFrame:
    selected_layer = int(config["models"][model_name]["selected_layer"])
    wrong_layer = int(config["models"][model_name]["wrong_layer"])
    dim_step = projection_sd_step(direction, projection_sigma)
    seeds = [int(value) for value in config["intervention"]["random_orthogonal_seeds"]]
    orthogonal = {
        seed: norm_matched_orthogonal(dim_step, direction, seed) for seed in seeds
    }
    chunk_root = output / "checkpoints" / f"causal_rescue_{model_name}_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for start in range(0, len(records), 10):
        part = records.iloc[start : start + 10]
        chunk = chunk_root / f"rows_{start:04d}_{start + len(part) - 1:04d}.parquet"
        if chunk.is_file():
            saved = pd.read_parquet(chunk)
            if set(saved["candidate_id"].astype(str)) != set(part["candidate_id"].astype(str)):
                raise RuntimeError("Rescue checkpoint rows changed.")
            frames.append(saved)
            continue
        rows: list[dict[str, Any]] = []
        for source in part.itertuples(index=False):
            for mapping_name in ("standard", "reversed"):
                original_prompt, supports, opposes, mapping = render_explicit_reason_prompt(
                    loaded,
                    situation=str(source.original_situation_action),
                    consideration=str(source.unchanged_named_consideration),
                    mapping_name=mapping_name,
                    prompt_config=prompt_config,
                )
                rewrite_prompt, rewrite_supports, rewrite_opposes, rewrite_mapping = (
                    render_explicit_reason_prompt(
                        loaded,
                        situation=str(source.candidate_situation_action),
                        consideration=str(source.unchanged_named_consideration),
                        mapping_name=mapping_name,
                        prompt_config=prompt_config,
                    )
                )
                if (supports, opposes, mapping) != (
                    rewrite_supports,
                    rewrite_opposes,
                    rewrite_mapping,
                ):
                    raise RuntimeError("Original/rewrite answer mapping diverged.")
                baseline_condition = build_conditions(
                    selected_layer=selected_layer,
                    wrong_layer=wrong_layer,
                    doses=config["intervention"]["doses_training_projection_sd"],
                    orthogonal_seeds=seeds,
                )[0]
                original = score_reason_with_intervention(
                    loaded,
                    prompt_ids=original_prompt,
                    supports_candidate=supports,
                    opposes_candidate=opposes,
                    direction_step=dim_step,
                    condition=baseline_condition,
                    batch_size=1,
                )
                baseline = score_reason_with_intervention(
                    loaded,
                    prompt_ids=rewrite_prompt,
                    supports_candidate=supports,
                    opposes_candidate=opposes,
                    direction_step=dim_step,
                    condition=baseline_condition,
                    batch_size=1,
                )
                target_sign = 1.0 if original.semantic_margin >= 0 else -1.0
                intervention_specs = [
                    ("dim", "frozen_dim", selected_layer, None, dim_step),
                    ("wrong_layer", "frozen_dim", wrong_layer, None, dim_step),
                    *[
                        (
                            "random_orthogonal",
                            f"orthogonal_{seed}",
                            selected_layer,
                            seed,
                            orthogonal[seed],
                        )
                        for seed in seeds
                    ],
                ]
                for family, direction_id, layer, seed, step in intervention_specs:
                    from .intervention import InterventionCondition

                    condition = InterventionCondition(
                        family=family,
                        direction_id=direction_id,
                        layer=layer,
                        dose=target_sign,
                        seed=seed,
                    )
                    steered = score_reason_with_intervention(
                        loaded,
                        prompt_ids=rewrite_prompt,
                        supports_candidate=supports,
                        opposes_candidate=opposes,
                        direction_step=step,
                        condition=condition,
                        batch_size=1,
                    )
                    rows.append(
                        {
                            "analysis_phase": "rewrite_rescue",
                            "model": model_name,
                            "base_item_id": str(source.base_item_id),
                            "candidate_id": str(source.candidate_id),
                            "population": str(source.population),
                            "stronger_first_stage_sensitivity": bool(
                                source.stronger_first_stage_sensitivity
                            ),
                            "mapping": mapping_name,
                            "family": family,
                            "direction_id": direction_id,
                            "seed": seed,
                            "dose": target_sign,
                            "original_margin": original.semantic_margin,
                            "baseline_rewrite_margin": baseline.semantic_margin,
                            "steered_rewrite_margin": steered.semantic_margin,
                            "signed_movement_toward_original": signed_rescue_movement(
                                original.semantic_margin,
                                baseline.semantic_margin,
                                steered.semantic_margin,
                            ),
                            "batch_size": steered.batch_size,
                            "prompt_token_ids_sha256": steered.prompt_token_ids_sha256,
                            "target_delta_max_abs_error": steered.target_delta_max_abs_error,
                            "untargeted_max_abs_difference": steered.untargeted_max_abs_difference,
                        }
                    )
        frame = pd.DataFrame(rows)
        if not frame["batch_size"].eq(1).all():
            raise RuntimeError("Rescue scoring violated batch_size=1.")
        frame.to_parquet(chunk, index=False)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(records) * 2 * 5
    if len(result) != expected:
        raise RuntimeError(f"Rescue result count changed: {len(result)} != {expected}")
    return result


def run_causal_model(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    target_model: str,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run one frozen causal model phase, resumably and in Llama/Gemma order."""
    if target_model not in {"llama", "gemma"}:
        raise ValueError("TARGET_MODEL must be llama or gemma.")
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = _resolve(repo, str(config["paths"]["output_root"]))
    amendment = output / "STUDY_CONTRACT_AMENDMENT_V3.json"
    if not amendment.is_file():
        raise RuntimeError("Causal scoring requires frozen study-contract amendment V3.")
    verdict_report = output / "VERDICT_ARM_REPORT_V2.md"
    verdict_result = output / "VERDICT_ARM_RESULT_V2.json"
    if not verdict_report.is_file() or not verdict_result.is_file():
        raise RuntimeError(
            "The corrected complete-verdict report V2 must be written before any intervention outcome is opened."
        )
    verdict_receipt = json.loads(verdict_result.read_text(encoding="utf-8"))
    if (
        verdict_receipt.get("status") != "VERDICT_ARM_COMPLETE"
        or int(verdict_receipt.get("analysis_revision", -1)) != 2
        or verdict_receipt.get("report_sha256") != sha256_file(verdict_report)
    ):
        raise RuntimeError("The corrected complete-verdict prerequisite failed hash verification.")
    if target_model == "gemma":
        llama_report = output / "CAUSAL_INTERVENTION_LLAMA_REPORT.md"
        llama_import = output / "CAUSAL_LLAMA_IMPORT_RECEIPT.json"
        if not llama_report.is_file() or not llama_import.is_file():
            raise RuntimeError("The verified Llama causal report must be written before Gemma loads.")
        llama_receipt = json.loads(llama_import.read_text(encoding="utf-8"))
        if (
            llama_receipt.get("status") != "CAUSAL_RESULT_IMPORTED_AND_VERIFIED"
            or llama_receipt.get("model") != "llama"
            or llama_receipt.get("report_sha256") != sha256_file(llama_report)
        ):
            raise RuntimeError("The Llama causal prerequisite failed import-receipt verification.")
    metadata_path = output / "checkpoints" / f"causal_{target_model}.metadata.json"
    claim1_path = output / "checkpoints" / f"causal_claim1_{target_model}.parquet"
    rescue_path = output / "checkpoints" / f"causal_rescue_{target_model}.parquet"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") != "CAUSAL_MODEL_COMPLETE"
            or metadata.get("model_revision")
            != config["models"][target_model]["revision"]
            or metadata.get("claim1_sha256") != sha256_file(claim1_path)
            or metadata.get("rescue_sha256") != sha256_file(rescue_path)
        ):
            raise RuntimeError("Causal model checkpoint failed verification.")
        report_path = output / f"CAUSAL_INTERVENTION_{target_model.upper()}_REPORT.md"
        if not report_path.is_file():
            analyze_causal_results(repo, config_path, models=[target_model])
        if target_model == "gemma" and not (
            output / "CLAIM1_HUMAN_STRATA_GEMMA_RESULT.json"
        ).is_file():
            from .human_strata import run_gemma_human_strata_supplement

            run_gemma_human_strata_supplement(repo, config_path)
        return {**metadata, "resumed": True}
    from geometry_endorsement.claim2_diagnostics.jury import load_jury_model
    from geometry_endorsement.claim2_diagnostics.sequence_scoring import unload_model

    entry = {
        "id": str(config["models"][target_model]["id"]),
        "revision": str(config["models"][target_model]["revision"]),
        "family": target_model,
    }
    probe_key = f"{target_model}_probe_bundle"
    expected_key = f"{target_model}_probe_bundle_sha256"
    frozen = load_frozen_dim(
        _resolve(repo, str(config["paths"][probe_key])),
        expected_sha256=str(config["expected_inputs"][expected_key]),
        expected_revision=str(config["models"][target_model]["revision"]),
        expected_layer=int(config["models"][target_model]["selected_layer"]),
        expected_width=int(config["models"][target_model]["hidden_size"]),
    )
    claim1_records = _claim1_records(repo, config)
    eval_records = claim1_records.loc[claim1_records["split"].eq("pilot_eval")].reset_index(
        drop=True
    )
    prompt_config = _load_prompt_config(repo)
    started = time.monotonic()
    loaded = load_jury_model(entry, token=hf_token)
    try:
        if target_model == "llama":
            arrays = np.load(
                _resolve(repo, str(config["paths"]["llama_compact"])), allow_pickle=False
            )
            activation = np.asarray(arrays["activation"], dtype=np.float32)
            split = np.asarray(arrays["split"]).astype(str)
        else:
            cache = _extract_gemma_activation_cache(
                loaded,
                records=claim1_records,
                prompt_config=prompt_config,
                selected_layer=int(config["models"][target_model]["selected_layer"]),
                output=output,
            )
            arrays = np.load(cache, allow_pickle=False)
            activation = np.asarray(arrays["activation"], dtype=np.float32)
            split = np.asarray(arrays["split"]).astype(str)
        projection_sigma = _projection_training_sigma(
            activation, split, np.asarray(frozen["direction"], dtype=np.float32)
        )
        claim1 = _score_claim1_causal(
            loaded,
            model_name=target_model,
            records=eval_records,
            prompt_config=prompt_config,
            direction=np.asarray(frozen["direction"], dtype=np.float32),
            projection_sigma=projection_sigma,
            config=config,
            output=output,
        )
        rescue = _score_rescue(
            loaded,
            model_name=target_model,
            records=_rescue_population(repo, config),
            prompt_config=prompt_config,
            direction=np.asarray(frozen["direction"], dtype=np.float32),
            projection_sigma=projection_sigma,
            config=config,
            output=output,
        )
        baseline_reproduction = _verify_rescue_baseline_reproduction(
            repo,
            config,
            model_name=target_model,
            rescue_rows=rescue,
            tolerance=0.002,
        )
        _write_or_verify_parquet(claim1_path, claim1)
        _write_or_verify_parquet(rescue_path, rescue)
        import torch
        import platform
        import transformers

        metadata = {
            "schema_version": 1,
            "status": "CAUSAL_MODEL_COMPLETE",
            "target_model": target_model,
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_revision": loaded.tokenizer_revision,
            "dtype": loaded.dtype,
            "device_name": loaded.device_name,
            "gpu_memory_mib": loaded.gpu_memory_mib,
            "attention_implementation": loaded.attention_implementation,
            "batch_size": 1,
            "float32_margin_calculation": True,
            "selected_layer": int(config["models"][target_model]["selected_layer"]),
            "wrong_layer": int(config["models"][target_model]["wrong_layer"]),
            "direction_sha256": frozen["direction_sha256"],
            "training_projection_sigma": projection_sigma,
            "claim1_rows": len(claim1),
            "rescue_rows": len(rescue),
            "claim1_sha256": sha256_file(claim1_path),
            "rescue_sha256": sha256_file(rescue_path),
            "frozen_reason_baseline_reproduction": baseline_reproduction,
            "runtime_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "entry_point": f"run_causal_model(target_model={target_model!r})",
        }
        _write_once(
            metadata_path,
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
    finally:
        unload_model(loaded)
        gc.collect()
    if target_model == "gemma":
        from .human_strata import run_gemma_human_strata_supplement

        metadata["gemma_human_strata_supplement"] = run_gemma_human_strata_supplement(
            repo, config_path
        )
    analyze_causal_results(repo, config_path, models=[target_model])
    return metadata


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    value: str,
    cluster: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    per_cluster = frame.groupby(cluster, sort=True)[value].mean().dropna().to_numpy(float)
    if not len(per_cluster):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = rng.choice(per_cluster, len(per_cluster), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(per_cluster.mean()), "ci_low": float(low), "ci_high": float(high)}


def _cluster_bootstrap_paired_contrast(
    frame: pd.DataFrame,
    *,
    value: str,
    cluster: str,
    treatment: str,
    controls: tuple[str, ...],
    replicates: int,
    permutation_replicates: int,
    seed: int,
) -> dict[str, float]:
    collapsed = frame.groupby([cluster, "family"], sort=True)[value].mean().unstack()
    available = [name for name in controls if name in collapsed.columns]
    if treatment not in collapsed.columns or not available:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "permutation_p_greater": float("nan"),
            "independent_clusters": 0,
        }
    values = (collapsed[treatment] - collapsed[available].mean(axis=1)).dropna().to_numpy(float)
    if not len(values):
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "permutation_p_greater": float("nan"),
            "independent_clusters": 0,
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    observed = float(values.mean())
    null = np.empty(permutation_replicates, dtype=float)
    for index in range(permutation_replicates):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values), replace=True)
        null[index] = float(np.mean(values * signs))
    p_greater = float((1 + np.sum(null >= observed)) / (permutation_replicates + 1))
    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "permutation_p_greater": p_greater,
        "independent_clusters": int(len(values)),
    }


def _claim1_dose_response(
    claim: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active = claim.loc[claim["family"].ne("no_intervention")].copy()
    for model in sorted(active["model"].unique()):
        model_frame = active.loc[active["model"].eq(model)]
        for population in ("ALL", "BOTH_CLEAR", "DISPUTED"):
            subset = (
                model_frame
                if population == "ALL"
                else model_frame.loc[model_frame["population"].eq(population)]
            )
            for (family, direction_id), direction_frame in subset.groupby(
                ["family", "direction_id"], sort=True
            ):
                zero = (
                    direction_frame.loc[direction_frame["dose"].astype(float).eq(0.0)]
                    .set_index("item_id")["semantic_margin"]
                    .astype(float)
                )
                if zero.index.duplicated().any():
                    raise RuntimeError("Causal dose-zero rows are not unique per item/direction.")
                for dose, dose_frame in direction_frame.groupby("dose", sort=True):
                    observed = dose_frame.copy()
                    observed["zero_margin"] = observed["item_id"].map(zero)
                    if observed["zero_margin"].isna().any():
                        raise RuntimeError("Causal dose response lacks a matched zero-dose row.")
                    observed["delta_from_zero"] = (
                        observed["semantic_margin"].astype(float)
                        - observed["zero_margin"].astype(float)
                    )
                    observed["verdict_changed_from_zero"] = np.sign(
                        observed["semantic_margin"].astype(float)
                    ) != np.sign(observed["zero_margin"].astype(float))
                    interval = _cluster_bootstrap_mean(
                        observed,
                        value="delta_from_zero",
                        cluster="board_id",
                        replicates=bootstrap_replicates,
                        seed=seed + int(round((float(dose) + 3.0) * 1000)),
                    )
                    rows.append(
                        {
                            "model": model,
                            "population": population,
                            "family": family,
                            "direction_id": direction_id,
                            "dose": float(dose),
                            "items": int(observed["item_id"].nunique()),
                            "boards": int(observed["board_id"].nunique()),
                            "mean_semantic_margin": float(
                                observed["semantic_margin"].astype(float).mean()
                            ),
                            "mean_delta_from_zero": interval["estimate"],
                            "delta_ci_low": interval["ci_low"],
                            "delta_ci_high": interval["ci_high"],
                            "verdict_change_rate_from_zero": float(
                                observed["verdict_changed_from_zero"].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _write_causal_figure(
    dose_response: pd.DataFrame,
    *,
    output: Path,
    suffix: str,
) -> str:
    import io

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    selected = dose_response.loc[
        dose_response["population"].eq("ALL")
        & dose_response["family"].eq("dim")
        & dose_response["direction_id"].eq("frozen_dim")
    ]
    if selected.empty:
        raise RuntimeError("The frozen DIM dose-response figure has no rows.")
    for model, frame in selected.groupby("model", sort=True):
        ordered = frame.sort_values("dose")
        axis.plot(
            ordered["dose"],
            ordered["mean_delta_from_zero"],
            marker="o",
            label=str(model),
        )
        axis.fill_between(
            ordered["dose"].astype(float),
            ordered["delta_ci_low"].astype(float),
            ordered["delta_ci_high"].astype(float),
            alpha=0.18,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Frozen DIM dose (pilot-train projection SD)")
    axis.set_ylabel("Mean Supports−Opposes margin change from dose 0")
    axis.set_title("Held-out Claim 1 causal dose response")
    axis.legend(frameon=False)
    figure.tight_layout()
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)
    path = output / "figures" / f"causal_dose_response{suffix}.png"
    _write_once(path, stream.getvalue())
    return sha256_file(path)


def analyze_causal_results(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = _resolve(repo, str(config["paths"]["output_root"]))
    requested = models or ["llama", "gemma"]
    claim_frames = []
    rescue_frames = []
    for model in requested:
        claim_path = output / "checkpoints" / f"causal_claim1_{model}.parquet"
        rescue_path = output / "checkpoints" / f"causal_rescue_{model}.parquet"
        if not claim_path.is_file() or not rescue_path.is_file():
            raise RuntimeError(f"Causal checkpoint is incomplete for {model}.")
        claim_frames.append(pd.read_parquet(claim_path))
        rescue_frames.append(pd.read_parquet(rescue_path))
    claim = pd.concat(claim_frames, ignore_index=True)
    rescue_raw = pd.concat(rescue_frames, ignore_index=True)
    rescue = (
        rescue_raw.groupby(
            [
                "model",
                "base_item_id",
                "candidate_id",
                "population",
                "stronger_first_stage_sensitivity",
                "family",
                "direction_id",
                "seed",
            ],
            dropna=False,
            as_index=False,
        )[
            ["original_margin", "baseline_rewrite_margin", "steered_rewrite_margin"]
        ]
        .mean()
    )
    rescue["signed_movement_toward_original"] = [
        signed_rescue_movement(a, b, c)
        for a, b, c in zip(
            rescue["original_margin"],
            rescue["baseline_rewrite_margin"],
            rescue["steered_rewrite_margin"],
            strict=True,
        )
    ]
    rescue["absolute_gap_reduction"] = (
        (rescue["original_margin"] - rescue["baseline_rewrite_margin"]).abs()
        - (rescue["original_margin"] - rescue["steered_rewrite_margin"]).abs()
    )
    rescue["baseline_flip"] = np.sign(rescue["original_margin"]) != np.sign(
        rescue["baseline_rewrite_margin"]
    )
    rescue["steered_flip"] = np.sign(rescue["original_margin"]) != np.sign(
        rescue["steered_rewrite_margin"]
    )
    rescue["flip_rescued"] = rescue["baseline_flip"] & ~rescue["steered_flip"]
    rescue["stable_disrupted"] = ~rescue["baseline_flip"] & rescue["steered_flip"]
    combined = pd.concat([claim, rescue], ignore_index=True, sort=False)
    if models is None:
        _write_or_verify_parquet(output / "intervention_item_results.parquet", combined)
    statistics = config["statistics"]
    report_rows = []
    for model in requested:
        model_claim = claim.loc[claim["model"].eq(model)]
        model_rescue = rescue.loc[rescue["model"].eq(model)]
        for population in ("ALL", "BOTH_CLEAR", "DISPUTED"):
            subset = model_claim if population == "ALL" else model_claim.loc[
                model_claim["population"].eq(population)
            ]
            slopes = []
            for (item_id, board_id, family, direction_id), group in subset.groupby(
                ["item_id", "board_id", "family", "direction_id"], sort=True
            ):
                slopes.append(
                    {
                        "item_id": item_id,
                        "board_id": board_id,
                        "family": family,
                        "direction_id": direction_id,
                        "local_slope": local_dose_slope(group),
                        "monotonic_fraction": monotonic_fraction(group),
                    }
                )
            slope_frame = pd.DataFrame(slopes)
            for family in ("dim", "opposite_dim", "wrong_layer", "random_orthogonal"):
                family_frame = slope_frame.loc[slope_frame["family"].eq(family)]
                if family_frame.empty:
                    continue
                estimate = _cluster_bootstrap_mean(
                    family_frame,
                    value="local_slope",
                    cluster="board_id",
                    replicates=int(statistics["bootstrap_replicates"]),
                    seed=int(statistics["seed"]),
                )
                report_rows.append(
                    {
                        "model": model,
                        "endpoint": "claim1_local_slope",
                        "population": population,
                        "family": family,
                        **estimate,
                        "monotonicity": float(family_frame["monotonic_fraction"].mean()),
                        "items": int(family_frame["item_id"].nunique()),
                    }
                )
            contrast = _cluster_bootstrap_paired_contrast(
                slope_frame,
                value="local_slope",
                cluster="board_id",
                treatment="dim",
                controls=("random_orthogonal", "wrong_layer"),
                replicates=int(statistics["bootstrap_replicates"]),
                permutation_replicates=int(statistics["permutation_replicates"]),
                seed=int(statistics["seed"]),
            )
            dim_frame = slope_frame.loc[slope_frame["family"].eq("dim")]
            report_rows.append(
                {
                    "model": model,
                    "endpoint": "claim1_dim_minus_controls_local_slope",
                    "population": population,
                    "family": "dim_minus_controls",
                    **contrast,
                    "monotonicity": float(dim_frame["monotonic_fraction"].mean()),
                    "items": int(dim_frame["item_id"].nunique()),
                }
            )
        for population in (
            "representative_primary",
            "enriched_diagnostic",
            "stronger_first_stage_sensitivity",
        ):
            subset = (
                model_rescue.loc[model_rescue["stronger_first_stage_sensitivity"].astype(bool)]
                if population == "stronger_first_stage_sensitivity"
                else model_rescue.loc[model_rescue["population"].eq(population)]
            )
            for family in ("dim", "wrong_layer", "random_orthogonal"):
                family_frame = subset.loc[subset["family"].eq(family)]
                if family_frame.empty:
                    continue
                pair_level = family_frame.groupby(
                    ["base_item_id", "candidate_id"], sort=True, as_index=False
                ).agg(
                    baseline_flip=("baseline_flip", "mean"),
                    flip_rescued=("flip_rescued", "mean"),
                    stable_disrupted=("stable_disrupted", "mean"),
                )
                estimate = _cluster_bootstrap_mean(
                    family_frame,
                    value="signed_movement_toward_original",
                    cluster="base_item_id",
                    replicates=int(statistics["bootstrap_replicates"]),
                    seed=int(statistics["seed"]),
                )
                report_rows.append(
                    {
                        "model": model,
                        "endpoint": "rewrite_signed_movement",
                        "population": population,
                        "family": family,
                        **estimate,
                        "monotonicity": float("nan"),
                        "items": int(family_frame["base_item_id"].nunique()),
                        "pairs": int(family_frame["candidate_id"].nunique()),
                        "baseline_flips": float(pair_level["baseline_flip"].sum()),
                        "flips_rescued": float(pair_level["flip_rescued"].sum()),
                        "stable_pairs_disrupted": float(pair_level["stable_disrupted"].sum()),
                    }
                )
                gap = _cluster_bootstrap_mean(
                    family_frame,
                    value="absolute_gap_reduction",
                    cluster="base_item_id",
                    replicates=int(statistics["bootstrap_replicates"]),
                    seed=int(statistics["seed"]) + 1,
                )
                report_rows.append(
                    {
                        "model": model,
                        "endpoint": "rewrite_absolute_gap_reduction",
                        "population": population,
                        "family": family,
                        **gap,
                        "monotonicity": float("nan"),
                        "items": int(family_frame["base_item_id"].nunique()),
                        "pairs": int(family_frame["candidate_id"].nunique()),
                        "baseline_flips": float(pair_level["baseline_flip"].sum()),
                        "flips_rescued": float(pair_level["flip_rescued"].sum()),
                        "stable_pairs_disrupted": float(pair_level["stable_disrupted"].sum()),
                    }
                )
            contrast = _cluster_bootstrap_paired_contrast(
                subset,
                value="signed_movement_toward_original",
                cluster="base_item_id",
                treatment="dim",
                controls=("random_orthogonal", "wrong_layer"),
                replicates=int(statistics["bootstrap_replicates"]),
                permutation_replicates=int(statistics["permutation_replicates"]),
                seed=int(statistics["seed"]),
            )
            report_rows.append(
                {
                    "model": model,
                    "endpoint": "rewrite_dim_minus_controls_signed_movement",
                    "population": population,
                    "family": "dim_minus_controls",
                    **contrast,
                    "monotonicity": float("nan"),
                    "items": int(subset["base_item_id"].nunique()),
                    "pairs": int(subset["candidate_id"].nunique()),
                    "baseline_flips": float(
                        subset.loc[subset["family"].eq("dim"), "baseline_flip"].sum()
                    ),
                    "flips_rescued": float(
                        subset.loc[subset["family"].eq("dim"), "flip_rescued"].sum()
                    ),
                    "stable_pairs_disrupted": float(
                        subset.loc[subset["family"].eq("dim"), "stable_disrupted"].sum()
                    ),
                }
            )
    metrics = pd.DataFrame(report_rows)
    dose_response = _claim1_dose_response(
        claim,
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["seed"]) + 70000,
    )
    metrics_path = output / (
        "causal_intervention_metrics.csv"
        if models is None
        else f"causal_intervention_{requested[0]}_metrics.csv"
    )
    _write_once(
        metrics_path,
        metrics.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    dose_path = output / (
        "causal_dose_response.csv"
        if models is None
        else f"causal_dose_response_{requested[0]}.csv"
    )
    _write_once(
        dose_path,
        dose_response.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    figure_sha = _write_causal_figure(
        dose_response,
        output=output,
        suffix="" if models is None else f"_{requested[0]}",
    )
    title = (
        "# Frozen DIM causal intervention"
        if models is None
        else f"# Frozen DIM causal intervention: {requested[0].title()} checkpoint"
    )
    lines = [
        title,
        "",
        "This arm is prospective for intervention outcomes. Natural rewrite outcomes were already open. Every model call used one logical prompt, BF16 weights, and float32 answer-margin calculation.",
        "",
        "| Model | Endpoint | Population | Family | Estimate | 95% CI | Permutation p | Monotonicity | Items | Rescued / disrupted |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.endpoint} | {row.population} | {row.family} | "
            f"{row.estimate:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
            f"{getattr(row, 'permutation_p_greater', float('nan')):.4f} | "
            f"{row.monotonicity:.3f} | {row.items} | "
            f"{getattr(row, 'flips_rescued', float('nan')):.1f} / "
            f"{getattr(row, 'stable_pairs_disrupted', float('nan')):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen DIM dose response (all held-out Claim 1 cells)",
            "",
            "| Model | Dose | Mean margin change from zero | 95% CI | Verdict-change rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    dim_dose = dose_response.loc[
        dose_response["population"].eq("ALL")
        & dose_response["family"].eq("dim")
        & dose_response["direction_id"].eq("frozen_dim")
    ].sort_values(["model", "dose"])
    for row in dim_dose.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dose:+.1f} | {row.mean_delta_from_zero:.4f} | "
            f"[{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
            f"{row.verdict_change_rate_from_zero:.3f} |"
        )
    lines.extend(
        [
            "",
            "No intervention family is interpreted as selective unless the prespecified DIM-minus-control and monotonicity rules in STUDY_CONTRACT_AMENDMENT_V1 are met.",
        ]
    )
    report_name = (
        "CAUSAL_INTERVENTION_REPORT.md"
        if models is None
        else f"CAUSAL_INTERVENTION_{requested[0].upper()}_REPORT.md"
    )
    _write_once(output / report_name, ("\n".join(lines) + "\n").encode("utf-8"))
    result = {
        "status": "CAUSAL_ANALYSIS_COMPLETE",
        "models": requested,
        "metrics_sha256": sha256_file(metrics_path),
        "dose_response_sha256": sha256_file(dose_path),
        "dose_response_figure_sha256": figure_sha,
        "report_sha256": sha256_file(output / report_name),
    }
    if models is None:
        result_path = output / "CAUSAL_INTERVENTION_RESULT.json"
        _write_once(
            result_path,
            (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        result["result_sha256"] = sha256_file(result_path)
    return result
