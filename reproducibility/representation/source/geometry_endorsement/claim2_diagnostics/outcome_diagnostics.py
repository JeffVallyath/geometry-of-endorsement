from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_audit import FROZEN_SENTENCE, load_contract
from .power_simulation import simulate_grid
from .predictor_models import run_frozen_grouped_ladder, summarize_ladder
from .reporting import artifact_manifest, sha256_file, write_json_once, write_once, write_text_once


def aggregate_item_outcomes(rewrites: pd.DataFrame) -> pd.DataFrame:
    required = {"base_item_id", "flipped", "mapping_consistent_flip", "original_margin", "rewrite_margin"}
    missing = required - set(rewrites.columns)
    if missing:
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")
    frame = rewrites.copy()
    frame["absolute_movement"] = (frame["rewrite_margin"] - frame["original_margin"]).abs()
    frame["toward_boundary"] = frame["original_margin"].abs() - frame["rewrite_margin"].abs()
    rows = []
    for base_item_id, group in frame.groupby("base_item_id", sort=False):
        rows.append({
            "base_item_id": base_item_id,
            "n_i": len(group),
            "k_i": int(group["flipped"].astype(bool).sum()),
            "any_accepted_rewrite_flips": bool(group["flipped"].astype(bool).any()),
            "mapping_consistent_flips_only": int(group["mapping_consistent_flip"].astype(bool).sum()),
            "mean_absolute_semantic_margin_movement": float(group["absolute_movement"].mean()),
            "maximum_absolute_semantic_margin_movement": float(group["absolute_movement"].max()),
            "mean_movement_toward_boundary": float(group["toward_boundary"].mean()),
            "rewrite_margin_variance": float(group["rewrite_margin"].var(ddof=0)),
        })
    return pd.DataFrame(rows)


def paired_binary_table(first: pd.Series, second: pd.Series) -> dict[str, int | float]:
    a = first.astype(bool).to_numpy()
    b = second.astype(bool).to_numpy()
    if len(a) != len(b):
        raise ValueError("Paired outcomes differ in length.")
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    statistic = float((abs(b01 - b10) - 1) ** 2 / (b01 + b10)) if b01 + b10 else 0.0
    return {"first_only": b10, "second_only": b01, "discordant": b01 + b10, "mcnemar_continuity_corrected": statistic}


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _load_feature_freeze(output: Path) -> dict[str, Any]:
    path = output / "BLIND_FEATURE_FREEZE.json"
    if not path.is_file():
        raise RuntimeError("Blind feature hashes must be frozen before Claim 2 outcomes are loaded.")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "BLIND_FEATURES_FROZEN" or freeze.get("claim2_outcomes_loaded") is not False:
        raise RuntimeError("Blind feature freeze is invalid.")
    for name, expected in freeze["feature_artifact_sha256"].items():
        if sha256_file(output / name) != expected:
            raise RuntimeError(f"Frozen feature hash mismatch: {name}")
    auxiliary = {
        "PROMPT_CONTRACT_REPORT.md": freeze.get("prompt_contract_report_sha256"),
        "prompt_contract_competence.csv": freeze.get("prompt_contract_competence_sha256"),
    }
    for name, expected in auxiliary.items():
        if not expected or sha256_file(output / name) != expected:
            raise RuntimeError(f"Frozen auxiliary feature hash mismatch: {name}")
    return freeze


def _paths(config_path: str | Path) -> tuple[Path, Path, Any, Path]:
    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    manifest = json.loads((output / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    frozen_root = Path(os.environ.get("CLAIM2_FROZEN_GEOMETRY_ROOT", manifest["frozen_geometry"]["path"]))
    return repo, output, contract, frozen_root


def _aligned_embeddings(path: Path, ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        stored_ids = list(map(str, arrays["base_item_id"]))
        embedding = arrays["embedding"].astype(np.float32)
    index = {value: position for position, value in enumerate(stored_ids)}
    if len(index) != len(stored_ids) or set(ids) - set(index):
        raise RuntimeError("Frozen text embeddings do not align to population items.")
    return embedding[[index[value] for value in ids]]


def _prepare_population_items(
    population_items: pd.DataFrame,
    population: str,
    jury: pd.DataFrame,
    mechanical: pd.DataFrame,
) -> pd.DataFrame:
    items = population_items[population_items["population"].eq(population)].copy()
    items = items.merge(jury, on="base_item_id", how="left", validate="one_to_one")
    items = items.merge(mechanical, on="base_item_id", how="left", validate="one_to_one")
    items["frozen_dim_geometry"] = pd.to_numeric(items["dim_weakness"], errors="raise")
    items["native_confidence"] = pd.to_numeric(items["native_confidence"], errors="raise")
    items["primary_jury_scalar"] = pd.to_numeric(items["leave_target_out_vote_entropy"], errors="raise")
    items["primary_mechanical_scalar"] = pd.to_numeric(items["irrelevant_string_verdict_retention"], errors="raise")
    for column in ["k_i", "n_i"]:
        items[column] = pd.to_numeric(items[column], errors="raise").astype(int)
    required = ["frozen_dim_geometry", "native_confidence", "primary_jury_scalar", "primary_mechanical_scalar"]
    if items[required].isna().any().any():
        raise RuntimeError(f"Primary ladder features are missing in {population}.")
    return items.reset_index(drop=True)


def _format_outcomes(target_scores: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    formats = target_scores[target_scores["trial_type"].eq("format")].copy()
    originals = formats[formats["dataset"].eq("claim2_original")][["base_item_id", "prompt_contract", "semantic_margin"]].rename(columns={"semantic_margin": "original_margin"})
    rewrites = formats[formats["dataset"].eq("claim2_accepted_rewrite")].merge(originals, on=["base_item_id", "prompt_contract"], validate="many_to_one")
    rewrites["format_flip"] = np.sign(rewrites["semantic_margin"].astype(float)) != np.sign(rewrites["original_margin"].astype(float))
    wide = rewrites.pivot(index=["base_item_id", "candidate_id"], columns="prompt_contract", values="format_flip").reset_index()
    required = ["ab_standard", "ab_reversed", "12_standard", "12_reversed", "direct_semantic"]
    if not set(required) <= set(wide.columns):
        raise RuntimeError("All five prompt formats are required for robust flip outcomes.")
    wide["cross_format_robust_flip"] = wide[required].all(axis=1)
    wide["any_format_flip"] = wide[required].any(axis=1)
    wide["encoding_specific_flip"] = wide[required].sum(axis=1).between(1, len(required) - 1)
    aggregate = wide.groupby("base_item_id", as_index=False).agg(
        accepted_rewrites=("candidate_id", "size"),
        cross_format_robust_flips=("cross_format_robust_flip", "sum"),
        any_format_flips=("any_format_flip", "sum"),
        encoding_specific_flips=("encoding_specific_flip", "sum"),
    )
    official_item = official[["base_item_id", "n_i", "k_i", "strict_mapping_consistent_flips", "mean_absolute_margin_movement", "mean_signed_margin_movement"]].copy()
    return wide, aggregate.merge(official_item, on="base_item_id", how="left", validate="one_to_one")


def _rewrite_diagnostics(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    strength_manifest: pd.DataFrame,
) -> pd.DataFrame:
    required_strength = {
        "review_id", "base_item_id", "control_type", "stratum",
        "token_overlap_jaccard", "character_edit_similarity",
        "character_length_ratio", "sentence_embedding_cosine",
        "descriptive_text_change_class",
    }
    if required_strength - set(strength_manifest.columns):
        raise RuntimeError("Frozen rewrite-strength manifest schema changed.")
    if len(strength_manifest) != 481 or strength_manifest["review_id"].duplicated().any():
        raise RuntimeError("Frozen rewrite-strength manifest must contain 481 unique review IDs.")
    strength = strength_manifest.copy()
    numeric = [
        "token_overlap_jaccard", "character_edit_similarity",
        "character_length_ratio", "sentence_embedding_cosine",
    ]
    strength[numeric] = strength[numeric].apply(pd.to_numeric, errors="raise")
    if strength[numeric].isna().any().any() or strength["character_length_ratio"].le(0).any():
        raise RuntimeError("Frozen rewrite-strength measurements are invalid.")
    strength["token_change_distance"] = 1.0 - strength["token_overlap_jaccard"]
    strength["character_change_distance"] = 1.0 - strength["character_edit_similarity"]
    strength["absolute_log_character_length_ratio"] = np.abs(np.log(strength["character_length_ratio"]))
    strength["sentence_embedding_distance"] = 1.0 - strength["sentence_embedding_cosine"]
    distance_columns = [
        "token_change_distance", "character_change_distance",
        "absolute_log_character_length_ratio", "sentence_embedding_distance",
    ]
    ordinary_strength = strength["control_type"].eq("none")
    strength["outcome_blind_strength_percentile_mean"] = np.nan
    strength.loc[ordinary_strength, "outcome_blind_strength_percentile_mean"] = (
        strength.loc[ordinary_strength, distance_columns]
        .rank(method="average", pct=True)
        .mean(axis=1)
    )
    ordinary = candidates[candidates["control_type"].eq("none")].copy()
    ordinary = ordinary.drop(
        columns=[
            "descriptive_text_change_class", "token_change_distance",
            "character_change_distance", "absolute_log_character_length_ratio",
            "sentence_embedding_distance", "outcome_blind_strength_percentile_mean",
        ],
        errors="ignore",
    ).merge(
        strength,
        on="review_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_strength"),
    )
    for column in ("base_item_id", "control_type", "stratum"):
        if not ordinary[column].astype(str).equals(ordinary[f"{column}_strength"].astype(str)):
            raise RuntimeError(f"Frozen rewrite-strength join differs on {column}.")
    if ordinary[distance_columns + ["outcome_blind_strength_percentile_mean"]].isna().any().any():
        raise RuntimeError("An ordinary rewrite lacks authoritative strength features.")
    ordinary["accepted"] = ordinary["primary_analysis_eligible"].astype(str).str.lower().eq("true")
    flips = outcomes[["candidate_id", "primary_semantic_flip"]].copy()
    flips["primary_semantic_flip"] = flips["primary_semantic_flip"].astype(str).str.lower().eq("true")
    ordinary = ordinary.merge(flips, on="candidate_id", how="left", validate="one_to_one")
    ordinary["primary_semantic_flip"] = ordinary["primary_semantic_flip"].fillna(False)
    strength = pd.to_numeric(ordinary["outcome_blind_strength_percentile_mean"], errors="coerce")
    ordinary["strength_quartile"] = pd.qcut(strength.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for (accepted, quartile), frame in ordinary.groupby(["accepted", "strength_quartile"], observed=True):
        rows.append({
            "accepted": bool(accepted), "strength_quartile": str(quartile), "rewrites": len(frame),
            "mean_token_change_distance": pd.to_numeric(frame["token_change_distance"], errors="coerce").mean(),
            "mean_character_change_distance": pd.to_numeric(frame["character_change_distance"], errors="coerce").mean(),
            "mean_embedding_distance": pd.to_numeric(frame["sentence_embedding_distance"], errors="coerce").mean(),
            "flip_rate": float(frame["primary_semantic_flip"].mean()),
            "flips": int(frame["primary_semantic_flip"].sum()),
        })
    return pd.DataFrame(rows)


def _range_diagnostics(repo: Path, frozen_root: Path, geometry_features: pd.DataFrame, target_scores: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "population_base_items_private.csv")
    score = geometry_features[["base_item_id", "mapping_average_geometry"]].rename(columns={"mapping_average_geometry": "dim_weakness"})
    native = target_scores[target_scores["dataset"].eq("claim2_original") & target_scores["trial_type"].eq("format") & target_scores["prompt_contract"].isin(["ab_standard", "ab_reversed"])].groupby("base_item_id", as_index=False)["semantic_margin"].mean()
    native["native_confidence"] = native["semantic_margin"].abs()
    native = native[["base_item_id", "native_confidence"]]
    base = base.merge(score, on="base_item_id", how="left", validate="one_to_one").merge(native, on="base_item_id", how="left", validate="one_to_one")
    items = pd.read_csv(frozen_root / "population_items_private.csv")
    memberships = {
        "all_112": set(base["base_item_id"]),
        "representative_80": set(base.loc[base["stratum"].eq("representative"), "base_item_id"]),
        "enriched_32": set(base.loc[base["stratum"].eq("disagreement"), "base_item_id"]),
        "representative_final_49": set(items.loc[items["population"].eq("representative_primary"), "base_item_id"]),
        "enriched_final_18": set(items.loc[items["population"].eq("enriched_geometry_confidence_diagnostic"), "base_item_id"]),
    }
    rows = []
    for name, ids in memberships.items():
        frame = base[base["base_item_id"].isin(ids)]
        for feature in ["dim_weakness", "native_confidence"]:
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            rows.append({"population": name, "feature": feature, "n": len(values), "sd": values.std(ddof=0), "iqr": values.quantile(.75)-values.quantile(.25), "minimum": values.min(), "maximum": values.max(), "fraction_near_boundary": values.abs().le(values.abs().quantile(.25)).mean()})
    return pd.DataFrame(rows)


def _make_figures(
    output: Path,
    ladder: pd.DataFrame,
    range_frame: pd.DataFrame,
    rewrite: pd.DataFrame,
    power: pd.DataFrame,
    prompt_format_summary: pd.DataFrame,
    predictor_scatter: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = ladder[ladder["population"].eq("representative_primary")]
    plots = []
    colors = ["#4c78a8" if model in {"M2", "M3"} else "#f58518" for model in primary["model"]]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(primary["model"], primary["binomial_log_loss_per_rephrasing"], color=colors); ax.set_ylabel("Held-out log loss"); ax.set_title("Frozen M2/M3 and post-outcome exploratory predictor ladder"); plots.append((fig, "figure_1_predictor_ladder.png"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, column, title in zip(
        axes,
        ["leave_target_out_vote_entropy", "native_confidence", "dim_weakness"],
        ["Jury entropy", "Native confidence", "Frozen DIM weakness"],
        strict=True,
    ):
        axis.scatter(predictor_scatter[column], predictor_scatter["susceptibility"], alpha=.75)
        axis.set_xlabel(title)
        axis.set_title("Exploratory")
    axes[0].set_ylabel("Observed flips / accepted rewrites")
    fig.suptitle("Jury, confidence, and geometry versus Llama susceptibility")
    plots.append((fig, "figure_2_predictor_relationships.png"))
    format_rows = prompt_format_summary[prompt_format_summary["prompt_contract"].isin(["ab_standard", "ab_reversed", "12_standard", "12_reversed", "direct_semantic"])]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(format_rows["prompt_contract"], format_rows["flip_rate"], color="#72b7b2"); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Rewrite flip rate"); ax.set_title("Post-outcome prompt-format sensitivity (not a replacement endpoint)"); plots.append((fig, "figure_3_prompt_consistency.png"))
    dim = range_frame[range_frame["feature"].eq("dim_weakness")]; fig, ax = plt.subplots(figsize=(9,4)); ax.bar(dim["population"], dim["sd"]); ax.tick_params(axis="x",rotation=30); ax.set_title("Geometry range before and after filtering"); plots.append((fig,"figure_4_geometry_range.png"))
    fig, ax = plt.subplots(figsize=(7,4)); accepted = rewrite[rewrite["accepted"]]; ax.bar(accepted["strength_quartile"].astype(str), accepted["flip_rate"]); ax.set_title("Accepted rewrite strength and flip rate"); plots.append((fig,"figure_5_rewrite_strength.png"))
    subset = power[(power["accepted_rewrites_per_original"].eq(3)) & (power["standardized_odds_ratio"].eq(1.5)) & (power["mapping_consistency_scenario"].eq("improved")) & (power["incremental_log_loss_gain"].eq(.005))]; fig, ax = plt.subplots(figsize=(7,4)); [ax.plot(group["eligible_originals"], group["probability_interval_excludes_zero_favorable"], label=f"event={rate}") for rate,group in subset.groupby("event_rate")]; ax.legend(); ax.set_title("Prospective sample-size grid"); plots.append((fig,"figure_6_power_curves.png"))
    for fig, name in plots:
        fig.tight_layout(); fig.savefig(figure_dir / name, dpi=160); plt.close(fig)


def _decision_categories(ladder: pd.DataFrame, outcome: pd.DataFrame, range_frame: pd.DataFrame, rewrite: pd.DataFrame, power: pd.DataFrame, thresholds: dict[str, Any]) -> list[str]:
    primary = ladder[ladder["population"].eq("representative_primary")].set_index("model")
    favorable = lambda model: model in primary.index and primary.loc[model, "delta_log_loss_vs_M2"] >= float(thresholds["favorable_predictive_increment_minimum_log_loss"]) and primary.loc[model, "delta_vs_M2_ci_low"] > 0
    categories = []
    if favorable("M5") and not favorable("M3"):
        categories.append("PARAPHRASE_INSTABILITY_PREDICTABLE_NOT_BY_FROZEN_DIM")
    if not any(favorable(model) for model in ["M3", "M5", "M7", "M8", "M9"]):
        categories.append("CURRENT_STUDY_UNDERPOWERED")
    robust = int(outcome["cross_format_robust_flips"].sum())
    official = int(outcome["k_i"].sum())
    if official - robust >= int(thresholds["substantially_different_flip_count_absolute"]):
        categories.append("BINARY_ENDPOINT_TOO_NOISY_OR_COARSE")
    sd = range_frame.pivot(index="feature", columns="population", values="sd")
    if "dim_weakness" in sd.index and sd.loc["dim_weakness", "representative_final_49"] < float(thresholds["severe_range_sd_retention_ratio"]) * sd.loc["dim_weakness", "representative_80"]:
        categories.append("PRIMARY_SAMPLE_RANGE_RESTRICTED")
    accepted = rewrite[rewrite["accepted"]]
    if accepted["flips"].sum() and accepted.loc[accepted["strength_quartile"].eq("Q4"), "flips"].sum() / accepted["flips"].sum() > float(thresholds["stronger_rewrite_flip_concentration"]):
        categories.append("PERTURBATION_DISTRIBUTION_TOO_WEAK")
    return categories or ["CURRENT_STUDY_UNDERPOWERED"]


def analyze_existing_llama(config_path: str | Path) -> dict[str, Any]:
    repo, output, contract, frozen_root = _paths(config_path)
    freeze = _load_feature_freeze(output)
    population_items = pd.read_csv(frozen_root / "population_items_private.csv")
    jury = pd.read_parquet(output / "jury_item_features.parquet")
    mechanical = pd.read_parquet(output / "mechanical_features.parquet")
    geometry_diagnostic = pd.read_parquet(output / "geometry_diagnostic_features.parquet")
    with np.load(frozen_root / "original_text_embeddings.npz", allow_pickle=False) as arrays:
        all_embedding_ids = list(map(str, arrays["base_item_id"]))
        all_embeddings = arrays["embedding"].astype(np.float32)
    embedding_index = {value: index for index, value in enumerate(all_embedding_ids)}
    ladder_rows = []
    geometry_sensitivity_rows = []
    geometry_relationship_rows = []
    model_contract = contract.raw["predictor_ladder"]
    model_contract = {name: values for name, values in model_contract.items() if name.startswith("M")}
    for population in contract.raw["inference"]["populations"]:
        items = _prepare_population_items(population_items, population, jury, mechanical)
        embeddings = all_embeddings[[embedding_index[value] for value in items["base_item_id"].astype(str)]]
        assignments = pd.read_csv(frozen_root / "populations" / population / "outer_fold_assignments.csv")
        predictions, choices = run_frozen_grouped_ladder(
            items, embeddings, assignments, model_contract,
            c_grid=list(map(float, contract.raw["inference"]["regularization_c"])),
            pca_grid=list(map(int, contract.raw["inference"]["text_pca_dimensions"])),
            inner_folds=int(contract.raw["inference"]["frozen_inner_folds"]),
        )
        summary = summarize_ladder(predictions, bootstrap_replicates=int(contract.raw["inference"]["bootstrap_replicates"]), permutation_replicates=int(contract.raw["inference"]["permutation_replicates"]), seed=int(contract.raw["inference"]["random_seed"]))
        summary.insert(0, "population", population)
        ladder_rows.append(summary)
        write_once(output / "checkpoints" / "analysis" / f"{population}_predictions.parquet", _parquet_bytes(predictions))
        write_once(output / "checkpoints" / "analysis" / f"{population}_hyperparameters.parquet", _parquet_bytes(choices))
        geometry_items = items.merge(geometry_diagnostic, on="base_item_id", how="left", validate="one_to_one")
        geometry_items["absolute_dim_distance"] = geometry_items["signed_dim_score"].abs()
        geometry_items["mapping_averaged_dim"] = geometry_items["mapping_average_geometry"]
        geometry_items["dim_mapping_variance"] = geometry_items["geometry_mapping_variance"]
        geometry_items["dim_logistic_disagreement"] = (np.sign(geometry_items["signed_dim_score"]) != np.sign(geometry_items["logistic_probe_margin"])).astype(float)
        feature_names = ["absolute_dim_distance", "signed_dim_score", "logistic_probe_margin", "mapping_averaged_dim", "dim_mapping_variance", "dim_logistic_disagreement"]
        sensitivity_models = {"M0": [], "M2": ["original_text", "native_confidence"], "M5": ["original_text", "native_confidence", "primary_jury_scalar"]}
        for feature in feature_names:
            sensitivity_models[f"M2_plus_{feature}"] = ["original_text", "native_confidence", feature]
            sensitivity_models[f"M5_plus_{feature}"] = ["original_text", "native_confidence", "primary_jury_scalar", feature]
        sensitivity_predictions, _ = run_frozen_grouped_ladder(
            geometry_items, embeddings, assignments, sensitivity_models,
            c_grid=list(map(float, contract.raw["inference"]["regularization_c"])), pca_grid=list(map(int, contract.raw["inference"]["text_pca_dimensions"])), inner_folds=int(contract.raw["inference"]["frozen_inner_folds"]),
        )
        sensitivity_summary = summarize_ladder(sensitivity_predictions, bootstrap_replicates=int(contract.raw["inference"]["bootstrap_replicates"]), permutation_replicates=int(contract.raw["inference"]["permutation_replicates"]), seed=int(contract.raw["inference"]["random_seed"])+17)
        sensitivity_summary.insert(0, "population", population); geometry_sensitivity_rows.append(sensitivity_summary)
        susceptibility = geometry_items["k_i"] / geometry_items["n_i"]
        predictors = geometry_items[["native_confidence", "primary_jury_scalar"]].to_numpy(dtype=float)
        design = np.c_[np.ones(len(predictors)), predictors]
        geometry_residual = geometry_items["frozen_dim_geometry"].to_numpy(dtype=float) - design @ np.linalg.lstsq(design, geometry_items["frozen_dim_geometry"].to_numpy(dtype=float), rcond=None)[0]
        outcome_residual = susceptibility.to_numpy(dtype=float) - design @ np.linalg.lstsq(design, susceptibility.to_numpy(dtype=float), rcond=None)[0]
        r_squared = 1 - np.var(geometry_residual) / np.var(geometry_items["frozen_dim_geometry"].to_numpy(dtype=float)) if np.var(geometry_items["frozen_dim_geometry"].to_numpy(dtype=float)) else np.nan
        geometry_relationship_rows.append({
            "population": population,
            "geometry_confidence_spearman": geometry_items["frozen_dim_geometry"].corr(geometry_items["native_confidence"], method="spearman"),
            "geometry_jury_spearman": geometry_items["frozen_dim_geometry"].corr(geometry_items["primary_jury_scalar"], method="spearman"),
            "geometry_susceptibility_spearman": geometry_items["frozen_dim_geometry"].corr(susceptibility, method="spearman"),
            "partial_geometry_susceptibility_after_confidence_and_jury": float(np.corrcoef(geometry_residual, outcome_residual)[0,1]),
            "geometry_vif_after_confidence_and_jury": float(1/(1-r_squared)) if np.isfinite(r_squared) and r_squared < 1 else np.nan,
        })
    ladder = pd.concat(ladder_rows, ignore_index=True)
    write_once(output / "predictor_ladder.csv", _csv_bytes(ladder))
    write_once(output / "geometry_feature_sensitivity.csv", _csv_bytes(pd.concat(geometry_sensitivity_rows, ignore_index=True)))
    write_once(output / "geometry_relationships.csv", _csv_bytes(pd.DataFrame(geometry_relationship_rows)))
    write_json_once(output / "geometry_feature_availability.json", {
        "available": ["frozen_dim_primary", "absolute_dim_distance", "signed_dim_score", "logistic_probe_margin", "mapping_averaged_dim", "dim_mapping_variance", "dim_logistic_disagreement"],
        "unavailable_not_manufactured": ["independently_fitted_direction_variance", "neighboring_layer_persistence", "centroid_distance_ratio"],
        "reason": "No already-frozen row-level direction ensemble, neighboring-layer cache, or centroid artifact was available.",
        "outcome_selected_feature": False,
    })

    target_scores = pd.read_parquet(output / "checkpoints" / "target_llama_blind.parquet")
    official = pd.read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "gate2_per_original_outcomes_private.csv")
    format_rows, outcome = _format_outcomes(target_scores, official)
    write_once(output / "checkpoints" / "analysis" / "prompt_format_flip_rows.parquet", _parquet_bytes(format_rows))
    format_summary = pd.DataFrame([
        {"prompt_contract": column, "rewrite_flips": int(format_rows[column].sum()), "flip_rate": float(format_rows[column].mean()), "originals_with_flip": int(format_rows.loc[format_rows[column], "base_item_id"].nunique())}
        for column in ["ab_standard", "ab_reversed", "12_standard", "12_reversed", "direct_semantic"]
    ])
    rng = np.random.default_rng(int(contract.raw["inference"]["random_seed"]))
    comparison_rows = []
    for left, right in [("ab_standard","ab_reversed"),("12_standard","12_reversed"),("ab_standard","direct_semantic")]:
        paired = paired_binary_table(format_rows[left], format_rows[right])
        per_item = format_rows.groupby("base_item_id")[[left,right]].mean()
        deltas = []
        for _ in range(int(contract.raw["inference"]["bootstrap_replicates"])):
            sampled = per_item.iloc[rng.integers(0,len(per_item),len(per_item))]
            deltas.append(float((sampled[left]-sampled[right]).mean()))
        comparison_rows.append({"prompt_contract":f"{left}_vs_{right}","rewrite_flips":np.nan,"flip_rate":float((format_rows[left].astype(int)-format_rows[right].astype(int)).mean()),"originals_with_flip":np.nan,**paired,"paired_item_bootstrap_ci_low":float(np.quantile(deltas,.025)),"paired_item_bootstrap_ci_high":float(np.quantile(deltas,.975))})
    format_summary = pd.concat([format_summary, pd.DataFrame(comparison_rows)], ignore_index=True)
    write_once(output / "prompt_format_outcome_diagnostics.csv", _csv_bytes(format_summary))
    outcome_joined = outcome.merge(jury[["base_item_id", "leave_target_out_vote_entropy"]], on="base_item_id", how="left", validate="one_to_one").merge(
        pd.read_parquet(output / "geometry_diagnostic_features.parquet")[["base_item_id", "mapping_average_geometry"]], on="base_item_id", how="left", validate="one_to_one"
    )
    outcome_rows = []
    for name in ["k_i", "strict_mapping_consistent_flips", "mean_absolute_margin_movement", "mean_signed_margin_movement", "cross_format_robust_flips", "encoding_specific_flips"]:
        values = pd.to_numeric(outcome_joined[name], errors="coerce")
        outcome_rows.append({
            "outcome_definition": name, "items": int(values.notna().sum()), "mean": float(values.mean()), "sum": float(values.sum()),
            "spearman_with_frozen_geometry": float(values.corr(outcome_joined["mapping_average_geometry"], method="spearman")),
            "spearman_with_jury_entropy": float(values.corr(outcome_joined["leave_target_out_vote_entropy"], method="spearman")),
            "inference_unit": "base_item_id", "status": "exploratory_post_outcome",
        })
    write_once(output / "outcome_definition_sensitivity.csv", _csv_bytes(pd.DataFrame(outcome_rows)))
    candidates = pd.read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "population_candidates_private.csv")
    candidate_outcomes = pd.read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "gate2_candidate_outcomes_private.csv")
    strength_manifest = pd.read_csv(
        output / "frozen_inputs" / "rewrite_strength_manifest.csv"
    )
    rewrite = _rewrite_diagnostics(candidates, candidate_outcomes, strength_manifest)
    write_once(output / "rewrite_selection_diagnostics.csv", _csv_bytes(rewrite))
    geometry_features = pd.read_parquet(output / "geometry_diagnostic_features.parquet")
    range_frame = _range_diagnostics(repo, frozen_root, geometry_features, target_scores)
    write_once(output / "range_restriction_diagnostics.csv", _csv_bytes(range_frame))
    power_predictors = population_items[
        population_items["population"].eq("representative_primary")
    ][["base_item_id", "native_confidence", "dim_weakness"]].merge(
        jury[["base_item_id", "leave_target_out_vote_entropy"]],
        on="base_item_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={
        "leave_target_out_vote_entropy": "primary_jury_scalar",
        "dim_weakness": "frozen_dim_geometry",
    })
    power = simulate_grid(
        contract.raw,
        event_rates=list(map(float, contract.raw["power"]["plausible_event_rates"])),
        observed_predictors=power_predictors,
        replicates=int(contract.raw["power"]["simulation_replicates"]),
    )
    write_once(output / "power_simulation.csv", _csv_bytes(power))
    categories = _decision_categories(ladder, outcome, range_frame, rewrite, power, contract.raw["diagnostic_thresholds"])
    scatter = population_items[
        population_items["population"].eq("representative_primary")
    ][["base_item_id", "k_i", "n_i", "native_confidence", "dim_weakness"]].copy()
    scatter = scatter.merge(
        jury[["base_item_id", "leave_target_out_vote_entropy"]],
        on="base_item_id",
        how="left",
        validate="one_to_one",
    )
    scatter["susceptibility"] = (
        pd.to_numeric(scatter["k_i"], errors="raise")
        / pd.to_numeric(scatter["n_i"], errors="raise")
    )
    for column in ["native_confidence", "dim_weakness", "leave_target_out_vote_entropy"]:
        scatter[column] = pd.to_numeric(scatter[column], errors="raise")
    _make_figures(output, ladder, range_frame, rewrite, power, format_summary, scatter)
    decision = ["# Decision report", "", FROZEN_SENTENCE, "", "All Llama diagnostics below are exploratory and post-outcome.", "", "## Assigned categories", ""] + [f"- `{value}`" for value in categories] + ["", "The primary population contains 13 flipping originals; the mapping-consistent secondary contains only 4."]
    write_text_once(output / "LLAMA_EXPLORATORY_DECISION.md", "\n".join(decision))
    required_n = int(power.loc[power["probability_interval_excludes_zero_favorable"].ge(.8), "eligible_originals"].min()) if power["probability_interval_excludes_zero_favorable"].ge(.8).any() else 800
    prospective = f"# Prospective replication specification\n\nThis specification is generated deterministically from `{categories}`.\n\n- Required eligible originals: {required_n}.\n- Candidate originals after observed 49/80 eligibility: {int(np.ceil(required_n/(49/80)))}.\n- Accepted rewrites per original: 3.\n- Statistical unit: original item.\n- Primary outcome: frozen mapping-consistent semantic flip rule.\n- Baseline: original text plus native confidence.\n- Jury: frozen leave-target-out eligible panel.\n- Geometry: frozen Claim 1 feature only.\n- Split: grouped, with an untouched final test split.\n- No outcome-adaptive thresholds or feature selection.\n"
    write_text_once(output / "PROSPECTIVE_REPLICATION_SPEC_DRAFT.md", prospective)
    write_json_once(output / "LLAMA_EXPLORATORY_RESULT.json", {"schema_version": 1, "status": "EXPLORATORY_POST_OUTCOME", "categories": categories, "required_eligible_originals": required_n, "frozen_llama_result": FROZEN_SENTENCE})
    llama_manifest = artifact_manifest(output, exclude=("artifact_manifest.json", "LLAMA_ANALYSIS_MANIFEST.json"))
    llama_manifest["status"] = "EXPLORATORY_LLAMA_ANALYSIS_FROZEN"
    llama_manifest["gemma_claim2_outcomes_loaded"] = False
    llama_manifest_sha = write_json_once(output / "LLAMA_ANALYSIS_MANIFEST.json", llama_manifest)
    return {"status": "EXPLORATORY_LLAMA_ANALYSIS_COMPLETE", "categories": categories, "llama_analysis_manifest_sha256": llama_manifest_sha, "blind_feature_freeze_sha256": sha256_file(output / "BLIND_FEATURE_FREEZE.json")}


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    import io

    stream = io.BytesIO(); frame.to_parquet(stream, index=False, engine="pyarrow", compression="zstd"); return stream.getvalue()


def run_prospective_gemma(config_path: str | Path) -> dict[str, Any]:
    from .prospective_gemma import run_prospective_gemma as run

    return run(config_path)


def final_diagnostic_audit(config_path: str | Path) -> dict[str, Any]:
    repo, output, contract, _ = _paths(config_path)
    required = [
        "INPUT_MANIFEST.json", "DIAGNOSTIC_CONTRACT.json", "FROZEN_LLAMA_REPRODUCTION.md", "CODE_FREEZE.json", "PROMPT_CONTRACT_REPORT.md", "prompt_contract_competence.csv",
        "frozen_claim1_mapping_audit.parquet", "FROZEN_CLAIM1_MAPPING_AUDIT.json", "claim1_claim2_prompt_comparison.csv", "CLAIM1_CLAIM2_PROMPT_COMPARISON.json",
        "jury_model_scores.parquet", "jury_item_features.parquet", "mechanical_features.parquet", "geometry_diagnostic_features.parquet",
        "predictor_ladder.csv", "outcome_definition_sensitivity.csv", "rewrite_selection_diagnostics.csv", "power_simulation.csv",
        "BLIND_FEATURE_FREEZE.json", "TEST_VERIFICATION.json", "LLAMA_ANALYSIS_MANIFEST.json", "LLAMA_EXPLORATORY_RESULT.json",
        "GEMMA_CLAIM2_PROSPECTIVE_FREEZE.json", "GEMMA_BEHAVIORAL_VIABILITY.json", "GEMMA_PROSPECTIVE_REPORT.md",
        "DECISION_REPORT.md", "PROSPECTIVE_REPLICATION_SPEC.md", "artifact_manifest.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Diagnostic output is incomplete: {missing}")
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest_missing = [
        name for name in manifest["files"] if not (output / name).is_file()
    ]
    if manifest_missing:
        raise RuntimeError(f"Manifest-listed artifacts are missing: {manifest_missing}")
    mismatches = [name for name, digest in manifest["files"].items() if sha256_file(output / name) != digest]
    if mismatches:
        raise RuntimeError(f"Artifact hash mismatches: {mismatches}")
    feature_freeze = json.loads((output / "BLIND_FEATURE_FREEZE.json").read_text(encoding="utf-8"))
    for name, digest in feature_freeze["feature_artifact_sha256"].items():
        if sha256_file(output / name) != digest:
            raise RuntimeError(f"Blind feature freeze mismatch: {name}")
    if sha256_file(output / "prompt_contract_competence.csv") != feature_freeze.get("prompt_contract_competence_sha256"):
        raise RuntimeError("Blind prompt-competence freeze mismatch.")
    tests = json.loads((output / "TEST_VERIFICATION.json").read_text(encoding="utf-8"))
    if tests.get("status") != "NEW_AND_ADJACENT_TESTS_PASSED" or int(tests.get("failed", -1)) != 0:
        raise RuntimeError("Frozen test verification did not pass.")
    code_freeze = json.loads((output / "CODE_FREEZE.json").read_text(encoding="utf-8"))
    if code_freeze.get("status") != "PRE_GPU_CODE_FROZEN":
        raise RuntimeError("Pre-GPU code freeze is invalid.")
    code_mismatches = [
        name for name, digest in code_freeze["files"].items()
        if not (repo / name).is_file() or sha256_file(repo / name) != digest
    ]
    if code_mismatches:
        raise RuntimeError(f"Code changed after the pre-GPU freeze: {code_mismatches}")
    mapping_bindings = {
        "frozen_claim1_mapping_audit.parquet": feature_freeze.get("frozen_claim1_mapping_audit_sha256"),
        "FROZEN_CLAIM1_MAPPING_AUDIT.json": feature_freeze.get("frozen_claim1_mapping_report_sha256"),
        "claim1_claim2_prompt_comparison.csv": feature_freeze.get("claim1_claim2_prompt_rows_sha256"),
        "CLAIM1_CLAIM2_PROMPT_COMPARISON.json": feature_freeze.get("claim1_claim2_prompt_comparison_sha256"),
    }
    mapping_mismatches = [
        name for name, digest in mapping_bindings.items()
        if not digest or sha256_file(output / name) != digest
    ]
    if mapping_mismatches:
        raise RuntimeError(f"Geometry audit binding mismatch: {mapping_mismatches}")
    llama = json.loads((output / "LLAMA_EXPLORATORY_RESULT.json").read_text(encoding="utf-8"))
    gemma_freeze = json.loads((output / "GEMMA_CLAIM2_PROSPECTIVE_FREEZE.json").read_text(encoding="utf-8"))
    gemma = json.loads((output / "GEMMA_BEHAVIORAL_VIABILITY.json").read_text(encoding="utf-8"))
    runtime_files = sorted((output / "checkpoints").rglob("*.metadata.json"))
    runtime = {
        path.relative_to(output).as_posix(): json.loads(path.read_text(encoding="utf-8"))
        for path in runtime_files
    }
    model_revisions = {
        "target_llama": contract.raw["models"]["target_llama"]["revision"],
        "prospective_gemma": contract.raw["models"]["prospective_gemma"]["revision"],
        "jury": {
            value["id"]: value["revision"] for value in contract.raw["models"]["jury"]
        },
    }
    try:
        import subprocess

        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        repository_commit = f"UNAVAILABLE:{type(exc).__name__}"
    result_hashes = {
        name: sha256_file(output / name)
        for name in [
            "predictor_ladder.csv",
            "outcome_definition_sensitivity.csv",
            "rewrite_selection_diagnostics.csv",
            "power_simulation.csv",
            "GEMMA_BEHAVIORAL_VIABILITY.json",
            "GEMMA_PROSPECTIVE_REPORT.md",
            "DECISION_REPORT.md",
            "PROSPECTIVE_REPLICATION_SPEC.md",
        ]
    }
    return {
        "status": "FINAL_DIAGNOSTIC_AUDIT_PASSED",
        "repository_commit": repository_commit,
        "contract_id": contract.raw["contract_id"],
        "diagnostic_contract_sha256": sha256_file(output / "DIAGNOSTIC_CONTRACT.json"),
        "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "code_freeze_sha256": sha256_file(output / "CODE_FREEZE.json"),
        "model_revisions": model_revisions,
        "feature_artifact_sha256": feature_freeze["feature_artifact_sha256"],
        "result_artifact_sha256": result_hashes,
        "tests": {"passed": int(tests["passed"]), "failed": int(tests["failed"]), "files": tests["test_files"]},
        "gpu": feature_freeze.get("gpu"),
        "runtime_per_model_and_phase": runtime,
        "gemma_genuinely_prospective": (
            gemma_freeze.get("prior_gemma_claim2_outcomes_inspected") is False
            and gemma_freeze.get("geometry_access_before_behavioral_viability") is False
            and gemma.get("prospective_freeze_sha256") == sha256_file(output / "GEMMA_CLAIM2_PROSPECTIVE_FREEZE.json")
        ),
        "final_diagnostic_categories": llama["categories"],
        "artifact_count": len(manifest["files"]),
        "artifact_manifest_sha256": sha256_file(output / "artifact_manifest.json"),
        "missing": [],
        "hash_mismatches": [],
    }
