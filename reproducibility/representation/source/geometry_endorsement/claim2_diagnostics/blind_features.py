from __future__ import annotations

import io
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .input_audit import load_contract, stage_outcome_blind_probe_inputs
from .jury import (
    aggregate_item_features,
    apply_eligibility,
    evaluate_model_competence,
    gpu_preflight,
    hard_votes,
    load_jury_model,
    model_access_preflight,
)
from .mechanical import aggregate_mechanical_features
from .prompt_contracts import (
    BlindFeatureReader,
    PromptContract,
    build_messages,
    build_target_messages,
    contracts_from_config,
    render_prompt,
)
from .reporting import sha256_file, write_json_once, write_once, write_text_once
from .sequence_scoring import score_one_prompt, score_one_prompt_with_activation, unload_model


def _repo_and_output(config_path: str | Path) -> tuple[Path, Path, Any]:
    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    return repo, output, contract


def _read_csv(path: Path) -> pd.DataFrame:
    payload = BlindFeatureReader().read_bytes(path)
    return pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False, engine="pyarrow", compression="zstd")
    return stream.getvalue()


def _tokenizer_sha256(tokenizer: Any) -> str:
    backend = (
        tokenizer.backend_tokenizer.to_str()
        if getattr(tokenizer, "backend_tokenizer", None) is not None
        else tokenizer.__class__.__qualname__
    )
    payload = json.dumps(
        {
            "backend": backend,
            "chat_template": str(getattr(tokenizer, "chat_template", "")),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_score_contract_sha256(frame: pd.DataFrame) -> str:
    columns = [
        value for value in [
            "dataset", "item_id", "base_item_id", "candidate_id",
            "trial_type", "trial_index", "prompt_contract",
            "rendered_prompt_sha256", "prompt_token_ids_sha256",
            "first_candidate_token_ids", "second_candidate_token_ids",
        ]
        if value in frame.columns
    ]
    ordered = frame[columns].astype(str).sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _resume_parquet_checkpoint(
    path: Path,
    metadata_path: Path,
    *,
    expected_status: str,
    expected_model_id: str,
    expected_revision: str,
    artifact_field: str = "artifact_sha256",
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    present = (path.is_file(), metadata_path.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError(f"Partial checkpoint cannot be resumed: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != expected_status:
        raise RuntimeError(f"Checkpoint status mismatch: {path}")
    if metadata.get("model_id", metadata.get("id")) != expected_model_id:
        raise RuntimeError(f"Checkpoint model mismatch: {path}")
    if metadata.get("model_revision", metadata.get("revision")) != expected_revision:
        raise RuntimeError(f"Checkpoint revision mismatch: {path}")
    if metadata.get("parameter_sha256_before") != metadata.get("parameter_sha256_after"):
        raise RuntimeError(f"Checkpoint parameter fingerprints differ: {path}")
    if int(metadata.get("behavioral_prompt_batch_size", -1)) != 1:
        raise RuntimeError(f"Checkpoint was not scored at prompt batch size one: {path}")
    if sha256_file(path) != metadata.get(artifact_field):
        raise RuntimeError(f"Checkpoint artifact hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if int(metadata.get("rows", len(frame))) != len(frame):
        raise RuntimeError(f"Checkpoint row count mismatch: {path}")
    if (
        len(str(metadata.get("tokenizer_sha256", ""))) != 64
        or len(str(metadata.get("chat_template_sha256", ""))) != 64
        or _prompt_score_contract_sha256(frame)
        != metadata.get("prompt_score_contract_sha256")
    ):
        raise RuntimeError(f"Checkpoint tokenizer/prompt binding mismatch: {path}")
    return frame, metadata


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def build_claim1_competence(repo: Path) -> pd.DataFrame:
    audit = _read_csv(repo / "outputs" / "claim1-first-pass-semantic-stratification" / "completed_exploratory_audit.csv")
    consensus = _read_csv(repo / "outputs" / "human-review-derived-sets" / "claim1_consensus_cells.csv")
    consensus = consensus[_bool(consensus["consensus_clear"])].copy()
    source = audit.set_index("review_id", verify_integrity=True)
    rows = []
    for row in consensus.to_dict("records"):
        position = str(row["cell_position"])
        if len(position) != 4 or position[:2] not in {"s1", "s2"} or position[2] != "_":
            raise RuntimeError(f"Unexpected Claim 1 cell position: {position}")
        board = source.loc[str(row["review_id"])]
        situation = str(board[f"situation_{position[1]}"])
        consideration = str(board[f"consideration_{position[3]}"])
        relation = str(row["stored_relation"])
        if relation not in {"Supports", "Opposes"}:
            raise RuntimeError(f"Unexpected stored relation: {relation}")
        rows.append({
            "dataset": "claim1_consensus_competence",
            "item_id": f"claim1:{row['review_id']}:{position}",
            "situation": situation,
            "consideration": consideration,
            "gold_label": 1 if relation == "Supports" else -1,
        })
    result = pd.DataFrame(rows)
    if len(result) != 297 or result["item_id"].duplicated().any():
        raise RuntimeError("Claim 1 competence set must contain 297 unique consensus-clear cells.")
    return result


def build_claim2_blind_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "population_base_items_private.csv")
    candidates = _read_csv(repo / "outputs" / "claim2-behavioral-viability" / "v005" / "population_candidates_private.csv")
    if len(base) != 112 or base["base_item_id"].duplicated().any():
        raise RuntimeError("Blind Claim 2 base population must contain 112 unique originals.")
    accepted = candidates[_bool(candidates["primary_analysis_eligible"])].copy()
    if len(accepted) != 234 or accepted["candidate_id"].duplicated().any():
        raise RuntimeError("Blind accepted primary rewrite population must contain 234 unique rows.")
    return base, accepted


def _score_record(
    loaded: Any,
    situation: str,
    consideration: str,
    prompt_contract: PromptContract,
    *,
    target_prompt_config: dict[str, Any] | None = None,
    random_string: str | None = None,
) -> dict[str, Any]:
    messages = (
        build_target_messages(situation, consideration, prompt_contract, target_prompt_config, random_string=random_string, model_id=loaded.model_id)
        if target_prompt_config is not None
        else build_messages(situation, consideration, prompt_contract, random_string=random_string)
    )
    prompt_ids, _, prompt_sha = render_prompt(loaded.tokenizer, messages)
    score = score_one_prompt(loaded, prompt_ids, prompt_contract.candidates, batch_size=1)
    semantic_margin = prompt_contract.semantic_margin(score.margin)
    return {
        "prompt_contract": prompt_contract.id,
        "mapping": "standard" if prompt_contract.supports_index == 0 else "reversed",
        "first_candidate_logp": score.first_logp,
        "second_candidate_logp": score.second_logp,
        "literal_first_minus_second_margin": score.margin,
        "semantic_margin": semantic_margin,
        "semantic_verdict": int(np.sign(semantic_margin)),
        "first_candidate_token_ids": json.dumps(score.first_tokens),
        "second_candidate_token_ids": json.dumps(score.second_tokens),
        "first_candidate_token_length": len(score.first_tokens),
        "second_candidate_token_length": len(score.second_tokens),
        "length_normalized_semantic_margin": prompt_contract.semantic_margin(score.length_normalized_margin),
        "rendered_prompt_sha256": prompt_sha,
        "prompt_token_ids_sha256": score.prompt_token_sha256,
        "behavioral_prompt_batch_size": 1,
    }


def _score_jury_model(loaded: Any, competence: pd.DataFrame, originals: pd.DataFrame, contracts: tuple[PromptContract, ...]) -> pd.DataFrame:
    mappings = [value for value in contracts if value.id in {"ab_standard", "ab_reversed"}]
    rows = []
    for record in competence.to_dict("records"):
        for mapping in mappings:
            rows.append({
                "model_id": loaded.model_id,
                "family": loaded.family,
                "dataset": record["dataset"],
                "item_id": record["item_id"],
                "base_item_id": "",
                "gold_label": record["gold_label"],
                **_score_record(loaded, record["situation"], record["consideration"], mapping),
            })
    for record in originals.to_dict("records"):
        for mapping in mappings:
            rows.append({
                "model_id": loaded.model_id,
                "family": loaded.family,
                "dataset": "claim2_original",
                "item_id": f"claim2:{record['base_item_id']}",
                "base_item_id": record["base_item_id"],
                "gold_label": np.nan,
                **_score_record(loaded, record["situation_action_text"], record["consideration_text"], mapping),
            })
    return pd.DataFrame(rows)


def _load_target_prompt_config(repo: Path) -> dict[str, Any]:
    raw = yaml.safe_load((repo / "configs" / "claim2_behavioral_m1_v1.yaml").read_text(encoding="utf-8"))
    return raw["prompt"]


def _score_target_blind(
    loaded: Any,
    competence: pd.DataFrame,
    originals: pd.DataFrame,
    rewrites: pd.DataFrame,
    contracts: tuple[PromptContract, ...],
    config: dict[str, Any],
    repo: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_config = _load_target_prompt_config(repo)
    rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for record in competence.to_dict("records"):
        for prompt_contract in contracts:
            rows.append({
                "dataset": "claim1_consensus_competence",
                "item_id": record["item_id"],
                "base_item_id": "",
                "candidate_id": "",
                "trial_type": "format",
                "trial_index": 0,
                "gold_label": record["gold_label"],
                **_score_record(loaded, record["situation"], record["consideration"], prompt_contract, target_prompt_config=prompt_config),
            })
    exact_trials = int(config["mechanical"]["trials"])
    random_strings = list(map(str, config["mechanical"]["random_strings"]))
    geometry_contracts = [value for value in contracts if value.id in {"ab_standard", "ab_reversed", "12_standard", "12_reversed"}]
    probe_path = repo / config["paths"]["output_dir"] / "frozen_inputs" / "llama_m1_probe_parameters.npz"
    with np.load(probe_path, allow_pickle=False) as probe:
        direction = probe["difference_in_means_direction"].astype(np.float32)
        midpoint = float(probe["difference_in_means_midpoint"])
        dim_mu = float(probe["dim_scaler_mu"])
        dim_sigma = float(probe["dim_scaler_sigma"])
        logistic_coef = probe["logistic_coef"].astype(np.float32).reshape(-1)
        logistic_intercept = float(probe["logistic_intercept"].reshape(-1)[0])
        selected_layer = int(probe["selected_layer"])
    for record in originals.to_dict("records"):
        situation = record["situation_action_text"]
        consideration = record["consideration_text"]
        base = {"item_id": f"claim2:{record['base_item_id']}", "base_item_id": record["base_item_id"], "candidate_id": "", "gold_label": np.nan}
        for trial in range(exact_trials):
            rows.append({**base, "dataset": "claim2_original", "trial_type": "exact_repeat", "trial_index": trial, **_score_record(loaded, situation, consideration, contracts[0], target_prompt_config=prompt_config)})
        for trial, random_string in enumerate(random_strings):
            for mapping in contracts[:2]:
                rows.append({**base, "dataset": "claim2_original", "trial_type": "irrelevant_string", "trial_index": trial, "random_string": random_string, **_score_record(loaded, situation, consideration, mapping, target_prompt_config=prompt_config, random_string=random_string)})
        for prompt_contract in contracts:
            rows.append({**base, "dataset": "claim2_original", "trial_type": "format", "trial_index": 0, **_score_record(loaded, situation, consideration, prompt_contract, target_prompt_config=prompt_config)})
        for prompt_contract in geometry_contracts:
            messages = build_target_messages(situation, consideration, prompt_contract, prompt_config, model_id=loaded.model_id)
            prompt_ids, _, prompt_sha = render_prompt(loaded.tokenizer, messages)
            activation_score = score_one_prompt_with_activation(loaded, prompt_ids, prompt_contract.candidates, selected_layer=selected_layer, batch_size=1)
            activation = activation_score.activation
            raw_dim = float(activation @ direction - midpoint)
            standardized_dim = (raw_dim - dim_mu) / dim_sigma
            raw_logistic = float(activation @ logistic_coef + logistic_intercept)
            geometry_rows.append({
                "base_item_id": record["base_item_id"],
                "mapping": prompt_contract.id,
                "geometry_score": -abs(standardized_dim),
                "signed_dim_score": standardized_dim,
                "raw_dim_margin": raw_dim,
                "raw_logistic_margin": raw_logistic,
                "selected_layer": selected_layer,
                "prompt_token_index": activation_score.prompt_token_index,
                "rendered_prompt_sha256": prompt_sha,
                "prompt_token_ids_sha256": activation_score.score.prompt_token_sha256,
                "model_revision": loaded.model_revision,
                "behavioral_prompt_batch_size": 1,
            })
    for record in rewrites.to_dict("records"):
        for prompt_contract in contracts:
            rows.append({
                "dataset": "claim2_accepted_rewrite",
                "item_id": f"candidate:{record['candidate_id']}",
                "base_item_id": record["base_item_id"],
                "candidate_id": record["candidate_id"],
                "trial_type": "format",
                "trial_index": 0,
                "gold_label": np.nan,
                **_score_record(loaded, record["candidate_situation_action"], record["unchanged_named_consideration"], prompt_contract, target_prompt_config=prompt_config),
            })
    return pd.DataFrame(rows), pd.DataFrame(geometry_rows)


def _prompt_competence_table(target_scores: pd.DataFrame) -> pd.DataFrame:
    competence = target_scores[target_scores["dataset"].eq("claim1_consensus_competence")].copy()
    result: list[dict[str, Any]] = []
    for contract_id, frame in competence.groupby("prompt_contract", sort=False):
        margin = frame["semantic_margin"].astype(float)
        normalized = frame["length_normalized_semantic_margin"].astype(float)
        gold = frame["gold_label"].astype(int)
        confidence = margin.abs()
        result.append({
            "row_type": "contract",
            "prompt_contract": contract_id,
            "comparison_left": "",
            "comparison_right": "",
            "items": len(frame),
            "semantic_accuracy": float((np.sign(margin) == gold).mean()),
            "length_normalized_accuracy": float((np.sign(normalized) == gold).mean()),
            "mean_absolute_margin": float(confidence.mean()),
            "p10_absolute_margin": float(confidence.quantile(.10)),
            "median_absolute_margin": float(confidence.median()),
            "p90_absolute_margin": float(confidence.quantile(.90)),
            "invalid_rate": float((~np.isfinite(margin)).mean()),
            "verdict_agreement": np.nan,
            "semantic_margin_spearman": np.nan,
        })
    pivot = competence.pivot(index="item_id", columns="prompt_contract", values="semantic_margin")
    for left, right, label in [
        ("ab_standard", "ab_reversed", "A/B mapping"),
        ("12_standard", "12_reversed", "1/2 mapping"),
        ("ab_standard", "direct_semantic", "A/B versus direct"),
        ("12_standard", "direct_semantic", "1/2 versus direct"),
    ]:
        result.append({
            "row_type": "comparison",
            "prompt_contract": label,
            "comparison_left": left,
            "comparison_right": right,
            "items": len(pivot),
            "semantic_accuracy": np.nan,
            "length_normalized_accuracy": np.nan,
            "mean_absolute_margin": np.nan,
            "p10_absolute_margin": np.nan,
            "median_absolute_margin": np.nan,
            "p90_absolute_margin": np.nan,
            "invalid_rate": float((~np.isfinite(pivot[[left, right]])).any(axis=1).mean()),
            "verdict_agreement": float((np.sign(pivot[left]) == np.sign(pivot[right])).mean()),
            "semantic_margin_spearman": float(pivot[left].corr(pivot[right], method="spearman")),
        })
    return pd.DataFrame(result)


def _prompt_report(target_scores: pd.DataFrame) -> str:
    table = _prompt_competence_table(target_scores)
    rows = [
        "# Prompt contract report",
        "",
        "Outcome-blind independent Claim 1 competence results. No Claim 2 flip outcome was loaded.",
        "",
        "| Contract | N | Accuracy | Length-normalized accuracy | Mean |margin| | P10 |margin| | Median |margin| | P90 |margin| | Invalid rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in table[table["row_type"].eq("contract")].to_dict("records"):
        rows.append(
            f"| {record['prompt_contract']} | {record['items']} | "
            f"{record['semantic_accuracy']:.6f} | "
            f"{record['length_normalized_accuracy']:.6f} | "
            f"{record['mean_absolute_margin']:.6f} | "
            f"{record['p10_absolute_margin']:.6f} | "
            f"{record['median_absolute_margin']:.6f} | "
            f"{record['p90_absolute_margin']:.6f} | "
            f"{record['invalid_rate']:.6f} |"
        )
    rows += ["", "## Pairwise competence consistency", "", "| Comparison | Verdict agreement | Margin correlation |", "|---|---:|---:|"]
    for record in table[table["row_type"].eq("comparison")].to_dict("records"):
        rows.append(
            f"| {record['prompt_contract']} | "
            f"{record['verdict_agreement']:.6f} | "
            f"{record['semantic_margin_spearman']:.6f} |"
        )
    rows += [
        "",
        "- A/B and 1/2 mapping consistency are the corresponding standard-versus-reversed verdict-agreement rows above.",
        "- Abstention/invalid rate is computed explicitly; any scoring exception fails the mode instead of becoming a silent abstention.",
        "- Full summed and length-normalized confidence distributions remain in the frozen target-score checkpoint.",
        "",
        "The existing frozen Llama result is not reformulated or replaced by these diagnostics.",
    ]
    return "\n".join(rows)


def _augment_jury_features(
    item_features: pd.DataFrame,
    votes: pd.DataFrame,
    jury_scores: pd.DataFrame,
    target_scores: pd.DataFrame,
    ordered_models: list[str],
) -> pd.DataFrame:
    result = item_features.copy()
    vote_wide = votes.pivot(index="base_item_id", columns="model_id", values="vote")
    for model_id in ordered_models:
        column = "loo_" + "".join(character if character.isalnum() else "_" for character in model_id) + "_vote_entropy"
        remaining = vote_wide.drop(columns=[model_id], errors="ignore")
        fraction = remaining.gt(0).sum(axis=1) / remaining.notna().sum(axis=1).replace(0, np.nan)
        entropy = fraction.map(lambda value: 0.0 if value in {0.0, 1.0} else (np.nan if pd.isna(value) else float(-(value * np.log2(value) + (1 - value) * np.log2(1 - value)))))
        result = result.merge(entropy.rename(column), left_on="base_item_id", right_index=True, how="left", validate="one_to_one")
    if len(ordered_models) == 7:
        from .jury import five_of_seven_subsets

        for index, subset in enumerate(five_of_seven_subsets(ordered_models)):
            # Preserve all 21 prespecified columns even when a frozen panel
            # member is ineligible. Missing members remain explicit NaNs; they
            # are never silently substituted.
            available = vote_wide.reindex(columns=list(subset))
            fraction = available.gt(0).sum(axis=1) / available.notna().sum(axis=1).replace(0, np.nan)
            strength = np.maximum(fraction, 1.0 - fraction)
            result = result.merge(strength.rename(f"five_of_seven_{index:02d}_majority_fraction"), left_on="base_item_id", right_index=True, how="left", validate="one_to_one")
    competence = jury_scores[jury_scores["dataset"].eq("claim1_consensus_competence")]
    calibration = competence.groupby("model_id")["semantic_margin"].agg(["mean", "std"])
    originals = jury_scores[jury_scores["dataset"].eq("claim2_original")].groupby(["base_item_id", "model_id"], as_index=False)["semantic_margin"].mean()
    originals = originals.join(calibration, on="model_id")
    originals["standardized_margin"] = (originals["semantic_margin"] - originals["mean"]) / originals["std"].replace(0, np.nan)
    soft = originals.groupby("base_item_id")["standardized_margin"].agg(calibration_standardized_soft_mean="mean", calibration_standardized_soft_dispersion="std").reset_index()
    result = result.merge(soft, on="base_item_id", how="left", validate="one_to_one")
    target = target_scores[
        target_scores["dataset"].eq("claim2_original")
        & target_scores["trial_type"].eq("format")
        & target_scores["prompt_contract"].isin(["ab_standard", "ab_reversed"])
    ].groupby("base_item_id", as_index=False)["semantic_margin"].mean()
    target["target_sign"] = np.sign(target["semantic_margin"])
    majority = votes.groupby("base_item_id")["vote"].agg(lambda values: np.sign(np.nansum(values))).rename("jury_majority_sign").reset_index()
    target = target.merge(majority, on="base_item_id", how="left", validate="one_to_one")
    target["target_versus_jury_disagreement"] = target["target_sign"].ne(target["jury_majority_sign"])
    target["target_in_jury_minority"] = target["target_versus_jury_disagreement"] & target["jury_majority_sign"].ne(0)
    return result.merge(target[["base_item_id", "target_versus_jury_disagreement", "target_in_jury_minority"]], on="base_item_id", how="left", validate="one_to_one")


def generate_blind_features(config_path: str | Path) -> dict[str, Any]:
    repo, output, contract = _repo_and_output(config_path)
    stage_outcome_blind_probe_inputs(config_path)
    config = contract.raw
    from .geometry_diagnostics import audit_frozen_claim1_mapping

    stable_join = _read_csv(
        output / "frozen_inputs" / "claim1_claim2_join.csv"
    )
    owner_base = _read_csv(
        output / "frozen_inputs" / "base_item_manifest_private.csv"
    )
    compact_path = (repo / config["paths"]["llama_compact_vectors"]).resolve()
    compact_metadata_path = (
        repo / config["paths"]["llama_compact_metadata"]
    ).resolve()
    compact_metadata = json.loads(compact_metadata_path.read_text(encoding="utf-8"))
    probe_path = output / "frozen_inputs" / "llama_m1_probe_parameters.npz"
    mapping_audit, mapping_report = audit_frozen_claim1_mapping(
        stable_join,
        owner_base,
        compact_path,
        probe_path,
        model_revision=str(compact_metadata["model_revision"]),
        selected_layer=int(compact_metadata["selected_layer"]),
        compact_artifact_sha256=sha256_file(compact_path),
        probe_artifact_sha256=sha256_file(probe_path),
    )
    mapping_audit_sha = write_once(
        output / "frozen_claim1_mapping_audit.parquet",
        _parquet_bytes(mapping_audit),
    )
    mapping_report_sha = write_json_once(
        output / "FROZEN_CLAIM1_MAPPING_AUDIT.json", mapping_report
    )
    gpu = gpu_preflight()
    contracts = contracts_from_config(config)
    competence = build_claim1_competence(repo)
    originals, rewrites = build_claim2_blind_inputs(repo)
    access = model_access_preflight(config["models"]["jury"])
    access_path = output / "MODEL_ACCESS_PREFLIGHT.csv"
    write_once(access_path, access.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    usable = access[access["usable"].astype(bool)]
    minimum = int(config["models"]["minimum_usable_jury_families"])
    if usable["family"].nunique() < minimum:
        raise RuntimeError(f"Insufficient jury diversity: {usable['family'].nunique()} usable families; require {minimum}.")

    checkpoint_dir = output / "checkpoints" / "jury"
    model_frames = []
    runtime_rows = []
    for entry in config["models"]["jury"]:
        safe_name = entry["family"]
        checkpoint = checkpoint_dir / f"{safe_name}.parquet"
        metadata_path = checkpoint.with_suffix(".metadata.json")
        resumed = _resume_parquet_checkpoint(
            checkpoint,
            metadata_path,
            expected_status="MODEL_SCORE_COMPLETE",
            expected_model_id=entry["id"],
            expected_revision=entry["revision"],
        )
        if resumed is not None:
            print(f"[resume] {entry['id']} from {checkpoint}", flush=True)
            frame, metadata = resumed
            model_frames.append(frame)
            runtime_rows.append(metadata)
            continue
        if not bool(usable.loc[usable["id"].eq(entry["id"]), "usable"].any()):
            runtime_rows.append({**entry, "status": "PREFLIGHT_UNUSABLE", "runtime_seconds": 0.0})
            continue
        loaded = None
        started = time.monotonic()
        try:
            print(f"[load] {entry['id']}@{entry['revision']}", flush=True)
            loaded = load_jury_model(entry)
            import torch
            torch.cuda.reset_peak_memory_stats()
            from causal_token_smoke.fingerprint import parameter_sha256
            parameter_before = parameter_sha256(loaded.model)
            frame = _score_jury_model(loaded, competence, originals, contracts)
            if len(frame) != 818 or not frame["behavioral_prompt_batch_size"].eq(1).all():
                raise RuntimeError(
                    f"Unexpected jury score topology for {entry['id']}: "
                    f"rows={len(frame)}"
                )
            parameter_after = parameter_sha256(loaded.model)
            if parameter_before != parameter_after:
                raise RuntimeError(f"Model parameters changed while scoring {entry['id']}.")
            digest = write_once(checkpoint, _parquet_bytes(frame))
            metadata = {
                **entry,
                "status": "MODEL_SCORE_COMPLETE",
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "tokenizer_sha256": _tokenizer_sha256(loaded.tokenizer),
                "chat_template_sha256": hashlib.sha256(loaded.tokenizer.chat_template.encode("utf-8")).hexdigest(),
                "prompt_score_contract_sha256": _prompt_score_contract_sha256(frame),
                "dtype": loaded.dtype,
                "attention_implementation": loaded.attention_implementation,
                "gpu": loaded.device_name,
                "gpu_memory_mib": loaded.gpu_memory_mib,
                "runtime_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "rows": len(frame),
                "artifact_sha256": digest,
                "behavioral_prompt_batch_size": 1,
                "parameter_sha256_before": parameter_before,
                "parameter_sha256_after": parameter_after,
            }
            write_json_once(metadata_path, metadata)
            model_frames.append(frame)
            runtime_rows.append(metadata)
            print(f"[saved] {entry['id']} rows={len(frame)} sha256={digest}", flush=True)
        except Exception as exc:
            metadata = {**entry, "status": "MODEL_SCORE_FAILED", "error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.monotonic() - started}
            runtime_rows.append(metadata)
            print(f"[failed] {entry['id']}: {metadata['error']}", flush=True)
        finally:
            if loaded is not None:
                unload_model(loaded)
    completed_families = {row["family"] for row in runtime_rows if row["status"] == "MODEL_SCORE_COMPLETE"}
    if len(completed_families) < minimum:
        raise RuntimeError(f"Only {len(completed_families)} jury families completed; require {minimum}. No substitutes were used.")
    jury_scores = pd.concat(model_frames, ignore_index=True)
    for model_id, frame in jury_scores.groupby("model_id", sort=False):
        if len(frame) != 818 or not frame["behavioral_prompt_batch_size"].eq(1).all():
            raise RuntimeError(f"Resumed jury score topology is invalid for {model_id}.")

    target_checkpoint = output / "checkpoints" / "target_llama_blind.parquet"
    geometry_checkpoint = output / "checkpoints" / "target_llama_geometry.parquet"
    target_metadata_path = output / "checkpoints" / "target_llama_blind.metadata.json"
    expected_target_rows = (
        len(competence) * len(contracts)
        + len(originals) * int(config["mechanical"]["trials"])
        + len(originals) * len(config["mechanical"]["random_strings"]) * 2
        + len(originals) * len(contracts)
        + len(rewrites) * len(contracts)
    )
    expected_geometry_rows = len(originals) * 4
    target_present = (
        target_checkpoint.is_file(),
        geometry_checkpoint.is_file(),
        target_metadata_path.is_file(),
    )
    if target_present == (True, True, True):
        target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
        if (
            target_metadata.get("status") != "TARGET_BLIND_FEATURES_COMPLETE"
            or target_metadata.get("model_id") != config["models"]["target_llama"]["id"]
            or target_metadata.get("model_revision") != config["models"]["target_llama"]["revision"]
            or int(target_metadata.get("behavioral_prompt_batch_size", -1)) != 1
            or target_metadata.get("claim2_outcomes_loaded") is not False
            or target_metadata.get("parameter_sha256_before")
            != target_metadata.get("parameter_sha256_after")
            or sha256_file(target_checkpoint) != target_metadata.get("score_sha256")
            or sha256_file(geometry_checkpoint) != target_metadata.get("geometry_sha256")
        ):
            raise RuntimeError("Target Llama blind checkpoint failed resume verification.")
        target_scores = pd.read_parquet(target_checkpoint)
        geometry_scores = pd.read_parquet(geometry_checkpoint)
        if (
            len(target_scores) != int(target_metadata.get("score_rows", -1))
            or len(geometry_scores) != int(target_metadata.get("geometry_rows", -1))
            or len(target_scores) != expected_target_rows
            or len(geometry_scores) != expected_geometry_rows
            or _prompt_score_contract_sha256(target_scores)
            != target_metadata.get("prompt_score_contract_sha256")
            or len(str(target_metadata.get("tokenizer_sha256", ""))) != 64
            or len(str(target_metadata.get("chat_template_sha256", ""))) != 64
        ):
            raise RuntimeError("Target Llama blind checkpoint row count changed.")
    elif any(target_present):
        raise RuntimeError("Partial target Llama blind checkpoint cannot be resumed.")
    else:
        target_entry = {"family": "llama31_target", **config["models"]["target_llama"]}
        target_entry["id"] = target_entry.pop("id")
        loaded = load_jury_model(target_entry)
        started = time.monotonic()
        try:
            print(f"[load] target {target_entry['id']}@{target_entry['revision']}", flush=True)
            import torch
            torch.cuda.reset_peak_memory_stats()
            from causal_token_smoke.fingerprint import parameter_sha256
            parameter_before = parameter_sha256(loaded.model)
            target_scores, geometry_scores = _score_target_blind(loaded, competence, originals, rewrites, contracts, config, repo)
            if (
                len(target_scores) != expected_target_rows
                or len(geometry_scores) != expected_geometry_rows
            ):
                raise RuntimeError(
                    "Target blind score topology changed: "
                    f"scores={len(target_scores)}/{expected_target_rows}, "
                    f"geometry={len(geometry_scores)}/{expected_geometry_rows}"
                )
            parameter_after = parameter_sha256(loaded.model)
            if parameter_before != parameter_after:
                raise RuntimeError("Target Llama parameters changed during blind feature generation.")
            target_sha = write_once(target_checkpoint, _parquet_bytes(target_scores))
            geometry_sha = write_once(geometry_checkpoint, _parquet_bytes(geometry_scores))
            write_json_once(target_metadata_path, {
                "status": "TARGET_BLIND_FEATURES_COMPLETE",
                "model_id": loaded.model_id,
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "tokenizer_sha256": _tokenizer_sha256(loaded.tokenizer),
                "chat_template_sha256": hashlib.sha256(loaded.tokenizer.chat_template.encode("utf-8")).hexdigest(),
                "prompt_score_contract_sha256": _prompt_score_contract_sha256(target_scores),
                "dtype": loaded.dtype,
                "attention_implementation": loaded.attention_implementation,
                "gpu": loaded.device_name,
                "gpu_memory_mib": loaded.gpu_memory_mib,
                "runtime_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "score_rows": len(target_scores),
                "geometry_rows": len(geometry_scores),
                "score_sha256": target_sha,
                "geometry_sha256": geometry_sha,
                "behavioral_prompt_batch_size": 1,
                "claim2_outcomes_loaded": False,
                "parameter_sha256_before": parameter_before,
                "parameter_sha256_after": parameter_after,
            })
            print(f"[saved] target blind scores={len(target_scores)} geometry={len(geometry_scores)}", flush=True)
        finally:
            unload_model(loaded)

    competence_scores = jury_scores[jury_scores["dataset"].eq("claim1_consensus_competence")].copy()
    competence_table = evaluate_model_competence(competence_scores[["model_id", "item_id", "mapping", "semantic_margin", "literal_first_minus_second_margin", "gold_label"]])
    competence_table = competence_table.merge(access[["id", "family"]].rename(columns={"id": "model_id"}), on="model_id", validate="one_to_one")
    competence_table = apply_eligibility(competence_table, config["models"]["jury_eligibility"])
    write_once(output / "jury_competence.csv", competence_table.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    eligible_models = set(competence_table.loc[competence_table["eligible"], "model_id"])
    eligible_families = set(
        competence_table.loc[competence_table["eligible"], "family"].astype(str)
    )
    if len(eligible_families) < minimum:
        raise RuntimeError(
            f"Only {len(eligible_families)} jury families passed the frozen "
            f"competence rule; require {minimum}. No substitutes were used."
        )
    if not target_scores["behavioral_prompt_batch_size"].eq(1).all():
        raise RuntimeError("Target blind features contain non-batch-one scoring.")
    if not np.isfinite(target_scores["semantic_margin"].astype(float)).all():
        raise RuntimeError("Target blind features contain a non-finite margin.")
    repeats = target_scores[
        target_scores["dataset"].eq("claim2_original")
        & target_scores["trial_type"].eq("exact_repeat")
    ]
    repeat_range = repeats.groupby("base_item_id")["semantic_margin"].agg(
        lambda values: float(values.astype(float).max() - values.astype(float).min())
    )
    if len(repeat_range) != 112 or int(repeats.groupby("base_item_id").size().min()) != int(config["mechanical"]["trials"]):
        raise RuntimeError("Exact-repeat diagnostic is incomplete.")
    maximum_repeat_difference = float(repeat_range.max())
    if maximum_repeat_difference > float(config["scoring"]["exact_repeat_tolerance"]):
        raise RuntimeError(
            "Target exact-repeat diagnostic exceeds the frozen tolerance: "
            f"{maximum_repeat_difference}"
        )
    original_scores = jury_scores[jury_scores["dataset"].eq("claim2_original") & jury_scores["model_id"].isin(eligible_models)].copy()
    original_scores["mapping"] = original_scores["prompt_contract"].map({"ab_standard": "standard", "ab_reversed": "reversed"})
    votes = hard_votes(original_scores[["model_id", "family", "base_item_id", "mapping", "semantic_margin"]])
    item_features = aggregate_item_features(votes, excluded_target_model=config["models"]["target_llama"]["id"])
    item_features = _augment_jury_features(
        item_features,
        votes,
        jury_scores[jury_scores["model_id"].isin(eligible_models)],
        target_scores,
        [entry["id"] for entry in config["models"]["jury"]],
    )
    mechanical_source = target_scores[target_scores["dataset"].eq("claim2_original")].copy()
    mechanical_source["mapping"] = mechanical_source["prompt_contract"]
    mechanical = aggregate_mechanical_features(mechanical_source[["base_item_id", "trial_type", "trial_index", "mapping", "semantic_margin"]])
    from .geometry_diagnostics import compare_claim1_claim2_prompts, mapping_invariance
    claim1_prompt = yaml.safe_load(
        (repo / "configs" / "m1_vertical_slice.yaml").read_text(encoding="utf-8")
    )["prompt"]
    claim2_prompt = _load_target_prompt_config(repo)
    prompt_rows, prompt_comparison = compare_claim1_claim2_prompts(
        mapping_audit,
        geometry_scores,
        claim1_prompt_specification=claim1_prompt,
        claim2_prompt_specification=claim2_prompt,
    )
    prompt_rows_sha = write_once(
        output / "claim1_claim2_prompt_comparison.csv",
        prompt_rows.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    prompt_comparison_sha = write_json_once(
        output / "CLAIM1_CLAIM2_PROMPT_COMPARISON.json", prompt_comparison
    )
    geometry_features = mapping_invariance(geometry_scores[["base_item_id", "mapping", "geometry_score"]]).merge(
        geometry_scores.groupby("base_item_id", as_index=False).agg(
            frozen_dim_score=("geometry_score", "first"),
            signed_dim_score=("signed_dim_score", "first"),
            logistic_probe_margin=("raw_logistic_margin", "first"),
        ), on="base_item_id", validate="one_to_one"
    )
    artifacts = {
        "jury_model_scores.parquet": write_once(output / "jury_model_scores.parquet", _parquet_bytes(jury_scores)),
        "jury_item_features.parquet": write_once(output / "jury_item_features.parquet", _parquet_bytes(item_features)),
        "mechanical_features.parquet": write_once(output / "mechanical_features.parquet", _parquet_bytes(mechanical)),
        "geometry_diagnostic_features.parquet": write_once(output / "geometry_diagnostic_features.parquet", _parquet_bytes(geometry_features)),
    }
    prompt_competence_sha = write_once(
        output / "prompt_contract_competence.csv",
        _prompt_competence_table(target_scores).to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8"),
    )
    prompt_report_sha = write_text_once(output / "PROMPT_CONTRACT_REPORT.md", _prompt_report(target_scores))
    feature_freeze = {
        "schema_version": 1,
        "status": "BLIND_FEATURES_FROZEN",
        "diagnostic_contract_sha256": sha256_file(output / "DIAGNOSTIC_CONTRACT.json"),
        "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "claim2_outcomes_loaded": False,
        "usable_jury_families": sorted(completed_families),
        "eligible_jury_families": sorted(eligible_families),
        "eligible_jury_models": sorted(eligible_models),
        "feature_artifact_sha256": artifacts,
        "prompt_contract_report_sha256": prompt_report_sha,
        "prompt_contract_competence_sha256": prompt_competence_sha,
        "target_exact_repeat_maximum_margin_difference": maximum_repeat_difference,
        "target_exact_repeat_tolerance": float(config["scoring"]["exact_repeat_tolerance"]),
        "frozen_claim1_mapping_audit_sha256": mapping_audit_sha,
        "frozen_claim1_mapping_report_sha256": mapping_report_sha,
        "claim1_claim2_prompt_rows_sha256": prompt_rows_sha,
        "claim1_claim2_prompt_comparison_sha256": prompt_comparison_sha,
        "model_runtime": runtime_rows,
        "gpu": gpu,
    }
    freeze_sha = write_json_once(output / "BLIND_FEATURE_FREEZE.json", feature_freeze)
    return {**feature_freeze, "blind_feature_freeze_sha256": freeze_sha}
