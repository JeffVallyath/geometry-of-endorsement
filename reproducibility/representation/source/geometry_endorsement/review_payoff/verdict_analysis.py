from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .input_audit import load_config, sha256_bytes, sha256_file


PAIR_CLASSES = (
    "REASON_STABLE_VERDICT_STABLE",
    "REASON_CHANGED_VERDICT_STABLE",
    "REASON_STABLE_VERDICT_CHANGED",
    "REASON_CHANGED_VERDICT_CHANGED",
)


def build_verdict_population(
    primary_intersection: pd.DataFrame,
    stronger_sensitivity: pd.DataFrame,
    independent_judgments: pd.DataFrame,
    base_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen complete-verdict population with exact IDs only."""
    primary = primary_intersection.copy()
    if len(primary) != 234 or primary["candidate_id"].duplicated().any():
        raise RuntimeError("Primary human-clear Claim 2 population changed.")
    if not primary["both_preserved"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Primary complete-verdict population includes a non-preserved rewrite.")
    if not primary["claim1_source_consensus_clear"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Primary complete-verdict population includes a non-human-clear source.")
    stronger_ids = set(stronger_sensitivity["candidate_id"].astype(str))
    if not stronger_ids.issubset(set(primary["candidate_id"].astype(str))):
        raise RuntimeError("Stronger sensitivity population is not nested in the primary population.")

    controls = independent_judgments.loc[
        independent_judgments["control_type"].astype(str).ne("none")
    ].copy()
    if len(controls) != 33 or controls["candidate_id"].duplicated().any():
        raise RuntimeError("Hidden-control population changed.")
    candidates = pd.concat([primary, controls], ignore_index=True, sort=False)
    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("Candidate population contains duplicate stable IDs.")
    candidates["wording_kind"] = "candidate"
    candidates["action_text"] = candidates["candidate_situation_action"].astype(str)
    candidates["is_control"] = candidates["control_type"].astype(str).ne("none")
    candidates["representative_primary"] = (
        ~candidates["is_control"] & candidates["stratum"].astype(str).eq("representative")
    )
    candidates["enriched_diagnostic"] = (
        ~candidates["is_control"]
        & candidates["stratum"].astype(str).isin(["enriched", "disagreement"])
    )
    candidates["stronger_first_stage_sensitivity"] = candidates["candidate_id"].astype(str).isin(
        stronger_ids
    )
    candidates["controls"] = candidates["is_control"]

    bases = base_manifest.copy()
    if len(bases) != 112 or bases["base_item_id"].duplicated().any():
        raise RuntimeError("Frozen base-item manifest changed.")
    required_base_ids = set(candidates["base_item_id"].astype(str))
    originals = bases.loc[bases["base_item_id"].astype(str).isin(required_base_ids)].copy()
    if set(originals["base_item_id"].astype(str)) != required_base_ids:
        raise RuntimeError("Candidate population does not resolve to base items exactly.")
    membership = candidates.groupby("base_item_id").agg(
        representative_primary=("representative_primary", "max"),
        enriched_diagnostic=("enriched_diagnostic", "max"),
        stronger_first_stage_sensitivity=("stronger_first_stage_sensitivity", "max"),
        controls=("controls", "max"),
    )
    originals = originals.merge(membership, left_on="base_item_id", right_index=True, validate="one_to_one")
    originals["candidate_id"] = ""
    originals["review_id"] = ""
    originals["control_type"] = "none"
    originals["wording_kind"] = "original"
    originals["action_text"] = originals["situation_action_text"].astype(str)
    keep = [
        "base_item_id",
        "candidate_id",
        "review_id",
        "control_type",
        "wording_kind",
        "action_text",
        "representative_primary",
        "enriched_diagnostic",
        "stronger_first_stage_sensitivity",
        "controls",
    ]
    result = pd.concat([originals[keep], candidates[keep]], ignore_index=True)
    if result.duplicated(["wording_kind", "base_item_id", "candidate_id"]).any():
        raise RuntimeError("Verdict scoring population has duplicate stable rows.")
    return result.sort_values(["wording_kind", "base_item_id", "candidate_id"]).reset_index(drop=True)


def collapse_reason_scores(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    """Collapse the existing two-mapping reason judgments without redefining them."""
    frame = scores.copy()
    if model == "llama" and {
        "mapping_averaged_semantic_margin",
        "mapping_agreement",
        "semantic_sign",
        "wording_kind",
    } <= set(frame.columns):
        result = frame.copy()
        result["reason_mapping_consistent"] = result["mapping_agreement"].astype(str).str.lower().eq("true")
        result["reason_margin"] = pd.to_numeric(
            result["mapping_averaged_semantic_margin"], errors="raise"
        )
        result["reason_verdict"] = pd.to_numeric(result["semantic_sign"], errors="raise").astype(int)
        if result.duplicated(["base_item_id", "candidate_id", "wording_kind"]).any():
            raise RuntimeError("Repaired Llama wording results contain duplicate stable rows.")
        return result[
            [
                "base_item_id",
                "candidate_id",
                "wording_kind",
                "reason_mapping_consistent",
                "reason_margin",
                "reason_verdict",
            ]
        ]
    if model == "llama":
        frame = frame.loc[
            frame["dataset"].isin(["claim2_original", "claim2_accepted_rewrite"])
            & frame["prompt_contract"].isin(["ab_standard", "ab_reversed"])
            & frame["trial_type"].isin(["format", "exact_repeat"])
        ].copy()
        # Original exact repeats are deterministic; retain the first frozen repeat.
        frame = frame.sort_values("trial_index").drop_duplicates(
            ["base_item_id", "candidate_id", "prompt_contract"]
        )
    elif model == "gemma":
        frame = frame.loc[frame["prompt_contract"].isin(["ab_standard", "ab_reversed"])].copy()
    else:
        raise ValueError("model must be llama or gemma")
    pivot = frame.pivot_table(
        index=["base_item_id", "candidate_id", "wording_kind"]
        if "wording_kind" in frame
        else ["base_item_id", "candidate_id", "dataset"],
        columns="mapping",
        values="semantic_margin",
        aggfunc="first",
    ).reset_index()
    if not {"standard", "reversed"} <= set(pivot.columns):
        raise RuntimeError("Existing reason scores lack both frozen mappings.")
    pivot["reason_mapping_consistent"] = np.sign(pivot["standard"]) == np.sign(pivot["reversed"])
    pivot["reason_margin"] = (pivot["standard"] + pivot["reversed"]) / 2.0
    pivot["reason_verdict"] = np.where(
        pivot["reason_mapping_consistent"], np.where(pivot["reason_margin"] >= 0, 1, -1), 0
    )
    if "dataset" in pivot:
        pivot["wording_kind"] = np.where(
            pivot["dataset"].eq("claim2_original"), "original", "candidate"
        )
    return pivot[
        [
            "base_item_id",
            "candidate_id",
            "wording_kind",
            "reason_mapping_consistent",
            "reason_margin",
            "reason_verdict",
        ]
    ]


def classify_reason_verdict_pairs(
    verdict_rows: pd.DataFrame,
    reason_rows: pd.DataFrame,
) -> pd.DataFrame:
    required_verdict = {
        "base_item_id",
        "candidate_id",
        "wording_kind",
        "mapping_consistent",
        "semantic_verdict",
    }
    if not required_verdict <= set(verdict_rows.columns):
        raise RuntimeError("Verdict result columns are incomplete.")
    originals_v = verdict_rows.loc[verdict_rows["wording_kind"].eq("original")].set_index(
        "base_item_id"
    )
    candidates_v = verdict_rows.loc[verdict_rows["wording_kind"].eq("candidate")].copy()
    originals_r = reason_rows.loc[reason_rows["wording_kind"].eq("original")].set_index(
        "base_item_id"
    )
    candidates_r = reason_rows.loc[reason_rows["wording_kind"].eq("candidate")].copy()
    joined = candidates_v.merge(
        candidates_r,
        on=["base_item_id", "candidate_id", "wording_kind"],
        how="left",
        validate="one_to_one",
    )
    if joined["reason_verdict"].isna().any():
        raise RuntimeError("Fresh verdict rows do not resolve to existing reason rows.")
    joined["original_verdict"] = joined["base_item_id"].map(originals_v["semantic_verdict"])
    joined["original_verdict_mapping_consistent"] = joined["base_item_id"].map(
        originals_v["mapping_consistent"]
    )
    joined["original_reason_verdict"] = joined["base_item_id"].map(originals_r["reason_verdict"])
    joined["original_reason_mapping_consistent"] = joined["base_item_id"].map(
        originals_r["reason_mapping_consistent"]
    )
    if joined[["original_verdict", "original_reason_verdict"]].isna().any().any():
        raise RuntimeError("Original rows are missing for one or more candidate pairs.")
    joined["pair_mapping_consistent"] = (
        joined["mapping_consistent"].astype(bool)
        & joined["original_verdict_mapping_consistent"].astype(bool)
        & joined["reason_mapping_consistent"].astype(bool)
        & joined["original_reason_mapping_consistent"].astype(bool)
    )
    joined["reason_changed"] = joined["reason_verdict"] != joined["original_reason_verdict"]
    joined["verdict_changed"] = joined["semantic_verdict"] != joined["original_verdict"]
    joined["pair_class"] = np.select(
        [
            ~joined["reason_changed"] & ~joined["verdict_changed"],
            joined["reason_changed"] & ~joined["verdict_changed"],
            ~joined["reason_changed"] & joined["verdict_changed"],
            joined["reason_changed"] & joined["verdict_changed"],
        ],
        list(PAIR_CLASSES),
        default="MAPPING_INCONSISTENT",
    )
    joined.loc[~joined["pair_mapping_consistent"], "pair_class"] = "MAPPING_INCONSISTENT"
    return joined


def bind_model_identity(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """Bind one verified model label without duplicating an existing column."""
    bound = rows.copy()
    if "model" in bound.columns:
        observed = set(bound["model"].dropna().astype(str).unique())
        if observed != {model}:
            raise RuntimeError(
                f"Verdict rows have unexpected model identity: expected {model!r}, "
                f"observed {sorted(observed)!r}."
            )
        return bound.loc[:, ["model", *[column for column in bound.columns if column != "model"]]]
    bound.insert(0, "model", model)
    return bound


def item_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    valid = pair_rows.loc[pair_rows["pair_mapping_consistent"]].copy()
    summary = valid.groupby("base_item_id").agg(
        rewrites=("candidate_id", "nunique"),
        any_reason_change=("reason_changed", "max"),
        any_complete_verdict_change=("verdict_changed", "max"),
    ).reset_index()
    summary["both"] = summary["any_reason_change"] & summary["any_complete_verdict_change"]
    summary["neither"] = ~summary["any_reason_change"] & ~summary["any_complete_verdict_change"]
    summary["reason_only"] = summary["any_reason_change"] & ~summary["any_complete_verdict_change"]
    summary["verdict_only"] = ~summary["any_reason_change"] & summary["any_complete_verdict_change"]
    return summary


def original_item_bootstrap(
    pair_rows: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    by_item = pair_rows.groupby("base_item_id")[value_column].mean()
    values = by_item.to_numpy(float)
    if not len(values):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "items": 0}
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(replicates)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "items": int(len(values)),
    }


def association_diagnostic(items: pd.DataFrame) -> dict[str, Any]:
    table = pd.crosstab(items["any_reason_change"], items["any_complete_verdict_change"]).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    odds, p = fisher_exact(table.to_numpy())
    return {
        "table": table.to_numpy().astype(int).tolist(),
        "row_order": ["reason_stable", "reason_changed"],
        "column_order": ["verdict_stable", "verdict_changed"],
        "fisher_odds_ratio": float(odds),
        "fisher_p_value_diagnostic_only": float(p),
    }


def bootstrap_item_rate_difference(
    items: pd.DataFrame,
    left: str,
    right: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = items[left].astype(float).to_numpy() - items[right].astype(float).to_numpy()
    if not len(values):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "items": 0}
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(replicates)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "items": int(len(values)),
    }


def analyze_verdict_arm(repo_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_config(config_path)
    output = repo / config["paths"]["output_root"]
    checkpoint = output / "checkpoints"
    paths = {model: checkpoint / f"verdict_{model}.parquet" for model in ("llama", "gemma")}
    metadata = {model: checkpoint / f"verdict_{model}.metadata.json" for model in paths}
    for model in paths:
        if not paths[model].is_file() or not metadata[model].is_file():
            raise RuntimeError(f"Both frozen verdict model checkpoints are required; missing {model}.")
        receipt = json.loads(metadata[model].read_text(encoding="utf-8"))
        if receipt.get("result_sha256") != sha256_file(paths[model]) or int(receipt.get("batch_size", -1)) != 1:
            raise RuntimeError(f"Verdict checkpoint verification failed for {model}.")

    pair_path = output / "verdict_pair_results_v2.parquet"
    item_path = output / "verdict_item_summary_v2.csv"
    report_path = output / "VERDICT_ARM_REPORT_V2.md"
    result_path = output / "VERDICT_ARM_RESULT_V2.json"
    completed = (pair_path, item_path, report_path, result_path)
    if any(path.exists() for path in completed):
        if not all(path.is_file() for path in completed):
            raise RuntimeError("Verdict analysis has a partial versioned output; refusing overwrite.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "VERDICT_ARM_COMPLETE"
            or result.get("verdict_pair_results_sha256") != sha256_file(pair_path)
            or result.get("verdict_item_summary_sha256") != sha256_file(item_path)
            or result.get("report_sha256") != sha256_file(report_path)
        ):
            raise RuntimeError("Completed verdict analysis failed hash verification.")
        return {**result, "result_sha256": sha256_file(result_path), "resumed": True}

    llama_reason = collapse_reason_scores(
        pd.read_csv(
            repo
            / "outputs"
            / "claim2-behavioral-viability"
            / "v005"
            / "inference"
            / "wording_results_private.csv",
            dtype=str,
        ).fillna(""),
        "llama",
    )
    gemma_raw = pd.concat(
        [
            pd.read_parquet(
                repo
                / config["paths"]["claim2_behavioral_root"]
                / "checkpoints"
                / "gemma_candidate_behavior.parquet"
            ),
            pd.read_parquet(
                repo
                / config["paths"]["claim2_behavioral_root"]
                / "checkpoints"
                / "gemma_original_repeat_control.parquet"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    gemma_reason = collapse_reason_scores(gemma_raw, "gemma")

    all_pairs = []
    all_items = []
    summaries: dict[str, Any] = {}
    stats = config["statistics"]
    for model, reason in (("llama", llama_reason), ("gemma", gemma_reason)):
        verdict = pd.read_parquet(paths[model])
        pairs = bind_model_identity(classify_reason_verdict_pairs(verdict, reason), model)
        pairs["control_type"] = pairs["control_type"].astype(str)
        for flag in (
            "representative_primary",
            "enriched_diagnostic",
            "stronger_first_stage_sensitivity",
            "controls",
        ):
            pairs[flag] = pairs[flag].astype(str).str.lower().eq("true")
        all_pairs.append(pairs)

        ordinary = pairs.loc[pairs["control_type"].eq("none")].copy()
        valid_ordinary = ordinary.loc[ordinary["pair_mapping_consistent"]].copy()
        items = item_summary(ordinary)
        membership = ordinary.groupby("base_item_id").agg(
            representative_primary=("representative_primary", "max"),
            enriched_diagnostic=("enriched_diagnostic", "max"),
            stronger_first_stage_sensitivity=("stronger_first_stage_sensitivity", "max"),
        ).reset_index()
        items = items.merge(membership, on="base_item_id", validate="one_to_one")
        items.insert(0, "model", model)
        all_items.append(items)
        difference = bootstrap_item_rate_difference(
            items,
            "any_reason_change",
            "any_complete_verdict_change",
            replicates=int(stats["bootstrap_replicates"]),
            seed=int(stats["seed"]) + (100000 if model == "llama" else 110000),
        )
        controls = {}
        for kind in ("identity", "trivial_restatement", "semantic_change_positive"):
            group = pairs.loc[pairs["control_type"].eq(kind) & pairs["pair_mapping_consistent"]]
            controls[kind] = {
                "pairs": int(len(group)),
                "complete_verdict_stability": float((~group["verdict_changed"]).mean()) if len(group) else np.nan,
                "complete_verdict_change_rate": float(group["verdict_changed"].mean()) if len(group) else np.nan,
                "reason_stability": float((~group["reason_changed"]).mean()) if len(group) else np.nan,
                "reason_change_rate": float(group["reason_changed"].mean()) if len(group) else np.nan,
            }
        summaries[model] = {
            "mapping_consistency_rate": float(verdict["mapping_consistent"].astype(bool).mean()),
            "ordinary_pairs": int(len(ordinary)),
            "ordinary_mapping_consistent_pairs": int(len(valid_ordinary)),
            "ordinary_mapping_inconsistent_pairs": int(len(ordinary) - len(valid_ordinary)),
            "ordinary_originals": int(ordinary["base_item_id"].nunique()),
            "mapping_consistent_originals": int(len(items)),
            "ordinary_pair_complete_verdict_change_rate": float(valid_ordinary["verdict_changed"].mean()),
            "ordinary_pair_reason_change_rate": float(valid_ordinary["reason_changed"].mean()),
            "originals_with_any_complete_verdict_change": int(items["any_complete_verdict_change"].sum()),
            "originals_with_any_reason_change": int(items["any_reason_change"].sum()),
            "original_item_reason_minus_verdict_rate": difference,
            "pair_class_counts": pairs["pair_class"].value_counts().to_dict(),
            "controls": controls,
            "association": association_diagnostic(items),
            "population_original_item_rates": {
                name: _population_item_rates(
                    items.loc[items[name]],
                    replicates=int(stats["bootstrap_replicates"]),
                    seed=int(stats["seed"]) + index,
                )
                for index, name in enumerate(
                    (
                        "representative_primary",
                        "enriched_diagnostic",
                        "stronger_first_stage_sensitivity",
                    )
                )
            },
        }

    pair_result = pd.concat(all_pairs, ignore_index=True)
    item_result = pd.concat(all_items, ignore_index=True)
    pair_result.to_parquet(pair_path, index=False)
    item_path.write_bytes(item_result.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    interpretation = _verdict_interpretation(summaries)
    report = _verdict_report(summaries, interpretation)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    result = {
        "status": "VERDICT_ARM_COMPLETE",
        "analysis_revision": 2,
        "supersedes_result": "VERDICT_ARM_RESULT.json",
        "superseded_result_sha256": (
            sha256_file(output / "VERDICT_ARM_RESULT.json")
            if (output / "VERDICT_ARM_RESULT.json").is_file()
            else None
        ),
        "interpretation": interpretation,
        "models": summaries,
        "verdict_pair_results_sha256": sha256_file(pair_path),
        "verdict_item_summary_sha256": sha256_file(item_path),
        "report_sha256": sha256_file(report_path),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {**result, "result_sha256": sha256_file(result_path)}


def _population_item_rates(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"originals": int(len(frame))}
    rng = np.random.default_rng(seed)
    for column in (
        "any_reason_change",
        "any_complete_verdict_change",
        "both",
        "neither",
    ):
        values = frame[column].astype(float).to_numpy()
        if not len(values):
            result[column] = np.nan
            result[f"{column}_ci_low"] = np.nan
            result[f"{column}_ci_high"] = np.nan
            continue
        draws = np.asarray(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)]
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        result[column] = float(values.mean())
        result[f"{column}_ci_low"] = float(low)
        result[f"{column}_ci_high"] = float(high)
    return result


def _verdict_interpretation(summaries: dict[str, Any]) -> str:
    differences = [summaries[model]["original_item_reason_minus_verdict_rate"] for model in ("llama", "gemma")]
    if all(row["estimate"] >= 0.10 and row["ci_low"] > 0 for row in differences):
        return "AGGREGATION_COMPENSATES_FOR_UNSTABLE_COMPONENTS"
    if all(row["estimate"] <= -0.10 and row["ci_high"] < 0 for row in differences):
        return "INSTABILITY_ENTERS_DURING_AGGREGATION_OR_REPORTING"
    if all(
        summaries[model]["association"]["fisher_odds_ratio"] > 1
        and summaries[model]["association"]["fisher_p_value_diagnostic_only"] < 0.05
        for model in ("llama", "gemma")
    ):
        return "REASON_INSTABILITY_PROPAGATES_TO_COMPLETE_VERDICT"
    if all(
        summaries[model]["mapping_consistent_originals"] > 0
        and summaries[model]["originals_with_any_complete_verdict_change"]
        / summaries[model]["mapping_consistent_originals"]
        < 0.10
        and summaries[model]["originals_with_any_reason_change"]
        / summaries[model]["mapping_consistent_originals"]
        < 0.10
        for model in ("llama", "gemma")
    ):
        return "EXISTING_REWRITES_DO_NOT_STRESS_COMPLETE_JUDGMENT"
    return "REASON_VERDICT_LOCALIZATION_MIXED_OR_UNRESOLVED"


def _fmt(value: Any) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.3f}"


def _verdict_report(summaries: dict[str, Any], interpretation: str) -> str:
    lines = [
        "# Constrained complete-verdict arm",
        "",
        "Status: development-only constrained five-category verdict analysis. The report was written before any intervention outcome was opened.",
        "",
        f"Prespecified interpretation: **{interpretation}**.",
        "",
        "Only pairs for which the original and rewrite are mapping-consistent at both the reason and complete-verdict levels enter change rates and item-level inference. Excluded pairs are reported separately.",
        "",
        "| Model | Row mapping consistency | Ordinary pairs total | Valid pairs | Excluded pairs | Valid originals | Pair reason-change | Pair verdict-change | Any reason change | Any verdict change | Reason-minus-verdict item rate (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("llama", "gemma"):
        row = summaries[model]
        diff = row["original_item_reason_minus_verdict_rate"]
        lines.append(
            f"| {model} | {_fmt(row['mapping_consistency_rate'])} | {row['ordinary_pairs']} | {row['ordinary_mapping_consistent_pairs']} | {row['ordinary_mapping_inconsistent_pairs']} | {row['mapping_consistent_originals']} | {_fmt(row['ordinary_pair_reason_change_rate'])} | {_fmt(row['ordinary_pair_complete_verdict_change_rate'])} | {row['originals_with_any_reason_change']} | {row['originals_with_any_complete_verdict_change']} | {_fmt(diff['estimate'])} [{_fmt(diff['ci_low'])}, {_fmt(diff['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Frozen populations at the original-item level",
            "",
            "| Model | Population | Originals | Any reason change (95% CI) | Any verdict change (95% CI) | Both (95% CI) | Neither (95% CI) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama", "gemma"):
        for population, rates in summaries[model]["population_original_item_rates"].items():
            lines.append(
                f"| {model} | {population} | {rates['originals']} | "
                f"{_fmt(rates['any_reason_change'])} [{_fmt(rates['any_reason_change_ci_low'])}, {_fmt(rates['any_reason_change_ci_high'])}] | "
                f"{_fmt(rates['any_complete_verdict_change'])} [{_fmt(rates['any_complete_verdict_change_ci_low'])}, {_fmt(rates['any_complete_verdict_change_ci_high'])}] | "
                f"{_fmt(rates['both'])} [{_fmt(rates['both_ci_low'])}, {_fmt(rates['both_ci_high'])}] | "
                f"{_fmt(rates['neither'])} [{_fmt(rates['neither_ci_low'])}, {_fmt(rates['neither_ci_high'])}] |"
            )
    lines.extend(["", "## Pair classifications and association diagnostics", ""])
    for model in ("llama", "gemma"):
        association = summaries[model]["association"]
        lines.append(
            f"- {model}: pair classes={json.dumps(summaries[model]['pair_class_counts'], sort_keys=True)}; "
            f"original-item table={association['table']} with rows {association['row_order']} and "
            f"columns {association['column_order']}; Fisher odds ratio="
            f"{_fmt(association['fisher_odds_ratio'])}, diagnostic p="
            f"{_fmt(association['fisher_p_value_diagnostic_only'])}."
        )
    lines.extend(["", "## Hidden controls", ""])
    for model in ("llama", "gemma"):
        for kind, result in summaries[model]["controls"].items():
            lines.append(
                f"- {model} {kind}: n={result['pairs']}, complete-verdict stability={_fmt(result['complete_verdict_stability'])}, reason stability={_fmt(result['reason_stability'])}."
                f" Complete-verdict change={_fmt(result['complete_verdict_change_rate'])}, reason change={_fmt(result['reason_change_rate'])}."
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The original item is the uncertainty unit. Pair rows are descriptive. Fisher exact tests are diagnostics only. This constrained symbolic verdict arm does not exhaust free-text moral verdicts and cannot distinguish internal aggregation from final reporting when both produce the same symbolic outcome.",
            "",
        ]
    )
    return "\n".join(lines)
