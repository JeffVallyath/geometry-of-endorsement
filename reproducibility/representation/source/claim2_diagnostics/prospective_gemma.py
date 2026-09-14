from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .blind_features import (
    _load_target_prompt_config,
    _parquet_bytes,
    _prompt_score_contract_sha256,
    _score_record,
    _tokenizer_sha256,
)
from .input_audit import load_contract
from .jury import aggregate_item_features, hard_votes, load_jury_model
from .predictor_models import (
    item_bootstrap_metric_delta,
    run_frozen_grouped_ladder,
    summarize_ladder,
)
from .prompt_contracts import build_target_messages, contracts_from_config, render_prompt
from .reporting import artifact_manifest, sha256_file, write_json_once, write_once, write_text_once
from .sequence_scoring import score_one_prompt_with_activation, unload_model


GEMMA_ID = "google/gemma-2-9b-it"
FROZEN_LLAMA_RESULT = "Claim 1 geometry did not improve the preregistered held-out prediction metric beyond original text and native confidence in the representative Llama population."


def _context(config_path: str | Path):
    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    input_manifest = json.loads((output / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    frozen_root = Path(os.environ.get("CLAIM2_FROZEN_GEOMETRY_ROOT", input_manifest["frozen_geometry"]["path"]))
    return repo, output, contract, frozen_root


def _resume_checkpoint(
    artifact_path: Path,
    metadata_path: Path,
    *,
    status: str,
    revision: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    present = (artifact_path.is_file(), metadata_path.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise RuntimeError(f"Partial Gemma checkpoint cannot be resumed: {artifact_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != status
        or metadata.get("model_id") != GEMMA_ID
        or metadata.get("model_revision") != revision
        or metadata.get("tokenizer_revision") != revision
        or int(metadata.get("batch_size", -1)) != 1
        or sha256_file(artifact_path) != metadata.get("artifact_sha256")
        or metadata.get("parameter_sha256_before")
        != metadata.get("parameter_sha256_after")
    ):
        raise RuntimeError(f"Gemma checkpoint metadata/hash mismatch: {artifact_path}")
    frame = pd.read_parquet(artifact_path)
    if len(frame) != int(metadata.get("rows", -1)):
        raise RuntimeError(f"Gemma checkpoint row count mismatch: {artifact_path}")
    if len(str(metadata.get("tokenizer_sha256", ""))) != 64:
        raise RuntimeError(f"Gemma tokenizer hash is missing: {artifact_path}")
    expected_prompt = metadata.get("prompt_score_contract_sha256")
    if expected_prompt is not None and _prompt_score_contract_sha256(frame) != expected_prompt:
        raise RuntimeError(f"Gemma prompt-contract hash mismatch: {artifact_path}")
    return frame, metadata


def freeze_gemma_contract(config_path: str | Path) -> dict[str, Any]:
    repo, output, contract, frozen_root = _context(config_path)
    prohibited = list(repo.glob("outputs/**/gemma*claim2*outcome*")) + list(repo.glob("outputs/**/claim2*gemma*outcome*"))
    if prohibited:
        raise RuntimeError(f"Possible prior Gemma Claim 2 outcomes exist: {prohibited}")
    probe_path = output / "frozen_inputs" / "gemma_m1_probe_parameters.npz"
    if sha256_file(probe_path) != "d3265c5dd23c20203c71ed79c8d84a789d36c1dafcca23f5f7b0b9409605a028":
        raise RuntimeError("Frozen Gemma probe hash mismatch.")
    paths = contract.raw["paths"]
    freeze = {
        "schema_version": 1,
        "status": "GEMMA_PROSPECTIVE_CONTRACT_FROZEN",
        "prior_gemma_claim2_outcomes_inspected": False,
        "local_prior_outcome_artifact_matches": [],
        "diagnostic_contract_sha256": sha256_file(output / "DIAGNOSTIC_CONTRACT.json"),
        "human_population_manifest_sha256": sha256_file(repo / paths["human_manifest"]),
        "candidate_population_sha256": sha256_file(repo / paths["gate2_dir"] / "population_candidates_private.csv"),
        "base_population_sha256": sha256_file(repo / paths["gate2_dir"] / "population_base_items_private.csv"),
        "model_id": GEMMA_ID,
        "model_revision": contract.raw["models"]["prospective_gemma"]["revision"],
        "tokenizer_revision": contract.raw["models"]["prospective_gemma"]["revision"],
        "claim1_probe_sha256": sha256_file(probe_path),
        "selected_layer": int(contract.raw["models"]["prospective_gemma"]["selected_layer"]),
        "behavioral_prompt_batch_size": 1,
        "dtype": "bfloat16",
        "prompt_contracts": ["ab_standard", "ab_reversed"],
        "candidate_scoring": contract.raw["scoring"]["candidate_score"],
        "primary_outcome": "mapping_averaged_semantic_sign_flip",
        "secondary_outcome": "strict_mapping_consistent_flip",
        "viability": contract.raw["gemma"],
        "predictors": {
            "M2": ["original_text", "native_gemma_confidence"],
            "M3": ["original_text", "native_gemma_confidence", "frozen_gemma_dim_geometry"],
            "M5": ["original_text", "native_gemma_confidence", "leave_gemma_out_jury"],
            "M8": ["original_text", "native_gemma_confidence", "leave_gemma_out_jury", "frozen_gemma_dim_geometry"],
        },
        "jury_rule": "frozen eligible panel excluding Gemma",
        "outer_assignments": "reuse frozen representative-primary and enriched assignments; mapping-consistent subset inherits representative-primary assignments",
        "outer_seeds": contract.raw["inference"]["frozen_outer_seeds"],
        "inner_folds": contract.raw["inference"]["frozen_inner_folds"],
        "bootstrap_replicates": contract.raw["inference"]["bootstrap_replicates"],
        "permutation_replicates": contract.raw["inference"]["permutation_replicates"],
        "m8_vs_m5_bootstrap_seed_rule": "inference.random_seed + 5000 + population_index",
        "geometry_access_before_behavioral_viability": False,
        "stop_if_viability_fails": True,
        "frozen_geometry_root_used_for_assignments_only": str(frozen_root),
    }
    digest = write_json_once(output / "GEMMA_CLAIM2_PROSPECTIVE_FREEZE.json", freeze)
    return {**freeze, "freeze_sha256": digest}


def _score_gemma_originals(loaded: Any, originals: pd.DataFrame, contracts, prompt_config, repeats: int) -> pd.DataFrame:
    rows = []
    for record in originals.to_dict("records"):
        for repeat in range(repeats):
            for prompt_contract in contracts:
                rows.append({
                    "base_item_id": record["base_item_id"], "candidate_id": "", "wording_kind": "original",
                    "repeat_index": repeat, "stratum": record["stratum"], "control_type": "none",
                    **_score_record(loaded, record["situation_action_text"], record["consideration_text"], prompt_contract, target_prompt_config=prompt_config),
                })
    return pd.DataFrame(rows)


def _score_gemma_candidates(loaded: Any, candidates: pd.DataFrame, contracts, prompt_config) -> pd.DataFrame:
    rows = []
    for record in candidates.to_dict("records"):
        for prompt_contract in contracts:
            rows.append({
                "base_item_id": record["base_item_id"], "candidate_id": record["candidate_id"], "wording_kind": "candidate",
                "repeat_index": 0, "stratum": record["stratum"], "control_type": record["control_type"],
                "population_role": record["population_role"],
                **_score_record(loaded, record["candidate_situation_action"], record["unchanged_named_consideration"], prompt_contract, target_prompt_config=prompt_config),
            })
    return pd.DataFrame(rows)


def _validate_original_repeat_gate(
    original_scores: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate every frozen repeat-control invariant before scoring rewrites."""
    tolerance = float(config["gemma"]["original_repeat_tolerance"])
    expected_rows = 112 * int(config["gemma"]["original_repeat_count"]) * 2
    if len(original_scores) != expected_rows:
        raise RuntimeError(
            "Gemma original-repeat topology changed: "
            f"observed={len(original_scores)}, expected={expected_rows}."
        )
    required_columns = {
        "base_item_id",
        "prompt_contract",
        "repeat_index",
        "rendered_prompt_sha256",
        "behavioral_prompt_batch_size",
        "semantic_margin",
        "first_candidate_token_length",
        "second_candidate_token_length",
    }
    missing = sorted(required_columns - set(original_scores.columns))
    if missing:
        raise RuntimeError(f"Gemma original-repeat columns are missing: {missing}")
    batch_one = bool(original_scores["behavioral_prompt_batch_size"].eq(1).all())
    finite_margins = bool(
        np.isfinite(pd.to_numeric(original_scores["semantic_margin"], errors="coerce")).all()
    )
    positive_answer_lengths = bool(
        pd.to_numeric(
            original_scores["first_candidate_token_length"], errors="coerce"
        ).gt(0).all()
        and pd.to_numeric(
            original_scores["second_candidate_token_length"], errors="coerce"
        ).gt(0).all()
    )
    prompt_repeat_counts = original_scores.groupby(
        ["base_item_id", "prompt_contract"]
    )["rendered_prompt_sha256"].nunique()
    prompt_bytes_repeat_exact = bool(prompt_repeat_counts.eq(1).all())
    repeat_counts = original_scores.groupby(
        ["base_item_id", "prompt_contract"]
    )["repeat_index"].nunique()
    repeat_topology_exact = bool(
        len(repeat_counts) == 112 * 2
        and repeat_counts.eq(int(config["gemma"]["original_repeat_count"])).all()
    )
    ranges = original_scores.groupby(
        ["base_item_id", "prompt_contract"]
    )["semantic_margin"].agg(
        lambda values: float(values.max() - values.min())
    )
    maximum = float(ranges.max())
    passed = bool(
        batch_one
        and finite_margins
        and positive_answer_lengths
        and prompt_bytes_repeat_exact
        and repeat_topology_exact
        and maximum <= tolerance
    )
    receipt = {
        "schema_version": 1,
        "status": (
            "GEMMA_ORIGINAL_REPEAT_GATE_PASSED"
            if passed
            else "GEMMA_ORIGINAL_REPEAT_GATE_FAILED"
        ),
        "gate_passed": passed,
        "rows": len(original_scores),
        "expected_rows": expected_rows,
        "batch_size_one": batch_one,
        "finite_semantic_margins": finite_margins,
        "positive_answer_token_lengths": positive_answer_lengths,
        "prompt_bytes_repeat_exact": prompt_bytes_repeat_exact,
        "repeat_topology_exact": repeat_topology_exact,
        "maximum_absolute_margin_difference": maximum,
        "tolerance": tolerance,
        "candidate_scoring_started": False,
        "geometry_loaded": False,
    }
    return receipt


def _summarize_behavior(original_scores: pd.DataFrame, candidate_scores: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any]):
    tolerance = float(config["gemma"]["original_repeat_tolerance"])
    expected_original_rows = 112 * int(config["gemma"]["original_repeat_count"]) * 2
    expected_candidate_rows = len(candidates) * 2
    if len(original_scores) != expected_original_rows or len(candidate_scores) != expected_candidate_rows:
        raise RuntimeError(
            "Gemma behavioral score topology changed: "
            f"original={len(original_scores)}/{expected_original_rows}, "
            f"candidate={len(candidate_scores)}/{expected_candidate_rows}"
        )
    score_frames = [original_scores, candidate_scores]
    all_valid = all(
        frame["behavioral_prompt_batch_size"].eq(1).all()
        and np.isfinite(frame["semantic_margin"].astype(float)).all()
        and pd.to_numeric(frame["first_candidate_token_length"], errors="coerce").gt(0).all()
        and pd.to_numeric(frame["second_candidate_token_length"], errors="coerce").gt(0).all()
        for frame in score_frames
    )
    prompt_repeat_counts = original_scores.groupby(
        ["base_item_id", "prompt_contract"]
    )["rendered_prompt_sha256"].nunique()
    prompt_bytes_repeat_exact = bool(prompt_repeat_counts.eq(1).all())
    all_valid = all_valid and prompt_bytes_repeat_exact
    repeat_gate = _validate_original_repeat_gate(original_scores, config)
    if not repeat_gate["gate_passed"]:
        raise RuntimeError(
            "Gemma original repeat gate failed before behavioral summarization."
        )
    ranges = original_scores.groupby(["base_item_id", "prompt_contract"])["semantic_margin"].agg(lambda values: float(values.max() - values.min()))
    original = original_scores[original_scores["repeat_index"].eq(0)].pivot(index="base_item_id", columns="prompt_contract", values="semantic_margin").reset_index()
    original["original_margin"] = (original["ab_standard"] + original["ab_reversed"]) / 2.0
    original["original_mapping_agreement"] = np.sign(original["ab_standard"]) == np.sign(original["ab_reversed"])
    candidate = candidate_scores.pivot(index=["base_item_id", "candidate_id", "stratum", "control_type", "population_role"], columns="prompt_contract", values="semantic_margin").reset_index()
    candidate["candidate_margin"] = (candidate["ab_standard"] + candidate["ab_reversed"]) / 2.0
    candidate["candidate_mapping_agreement"] = np.sign(candidate["ab_standard"]) == np.sign(candidate["ab_reversed"])
    candidate = candidate.merge(original[["base_item_id", "original_margin", "original_mapping_agreement"]], on="base_item_id", validate="many_to_one")
    candidate["primary_semantic_flip"] = np.sign(candidate["candidate_margin"]) != np.sign(candidate["original_margin"])
    candidate["strict_mapping_consistent_flip"] = candidate["primary_semantic_flip"] & candidate["candidate_mapping_agreement"] & candidate["original_mapping_agreement"]
    identity = candidate[candidate["control_type"].eq("identity")]
    identity_flips = int(identity["primary_semantic_flip"].sum())
    trivial = candidate[candidate["control_type"].eq("trivial_restatement")]
    semantic_change = candidate[candidate["control_type"].eq("semantic_change_positive")]
    primary = candidate[candidate["population_role"].eq("primary_accepted")]
    representative = primary[primary["stratum"].eq("representative")]
    distinct = int(representative.loc[representative["primary_semantic_flip"], "base_item_id"].nunique())
    gate_passed = (
        identity_flips == 0
        and distinct >= int(config["gemma"]["minimum_distinct_representative_flipping_originals"])
        and (all_valid or not bool(config["gemma"]["require_all_behavioral_wordings_format_valid"]))
    )
    report = {
        "status": "GEMMA_BEHAVIORAL_VIABLE" if gate_passed else "GEMMA_BEHAVIORAL_NOT_VIABLE_STOP",
        "original_repeat_max_abs_difference": float(ranges.max()),
        "original_repeat_tolerance": tolerance,
        "identity_control_sign_flips": identity_flips,
        "trivial_restatement_control_sign_flips": int(trivial["primary_semantic_flip"].sum()),
        "semantic_change_control_sign_flips": int(semantic_change["primary_semantic_flip"].sum()),
        "representative_distinct_flipping_originals": distinct,
        "representative_flip_rows": int(representative["primary_semantic_flip"].sum()),
        "enriched_distinct_flipping_originals": int(primary.loc[primary["stratum"].eq("disagreement") & primary["primary_semantic_flip"], "base_item_id"].nunique()),
        "all_behavioral_wordings_format_valid": bool(all_valid),
        "original_prompt_bytes_repeat_exact": prompt_bytes_repeat_exact,
        "original_mapping_agreement_rate": float(original["original_mapping_agreement"].mean()),
        "candidate_mapping_agreement_rate": float(candidate["candidate_mapping_agreement"].mean()),
        "behavioral_score_rows": {
            "original": len(original_scores),
            "candidate": len(candidate_scores),
        },
        "geometry_loaded": False,
        "gate_passed": gate_passed,
        "original_repeat_gate_status": repeat_gate["status"],
    }
    return candidate, report


def _gemma_geometry(loaded, originals, contracts, prompt_config, probe_path: Path) -> pd.DataFrame:
    with np.load(probe_path, allow_pickle=False) as probe:
        direction = probe["difference_in_means_direction"].astype(np.float32)
        midpoint = float(probe["difference_in_means_midpoint"])
        mu = float(probe["dim_scaler_mu"]); sigma = float(probe["dim_scaler_sigma"])
        layer = int(probe["selected_layer"])
    rows = []
    for record in originals.to_dict("records"):
        for prompt_contract in contracts:
            messages = build_target_messages(record["situation_action_text"], record["consideration_text"], prompt_contract, prompt_config, model_id=GEMMA_ID)
            prompt_ids, _, prompt_sha = render_prompt(loaded.tokenizer, messages)
            scored = score_one_prompt_with_activation(loaded, prompt_ids, prompt_contract.candidates, selected_layer=layer, batch_size=1)
            raw = float(scored.activation @ direction - midpoint)
            rows.append({"base_item_id": record["base_item_id"], "mapping": prompt_contract.id, "raw_dim_margin": raw, "frozen_gemma_dim_geometry": -abs((raw-mu)/sigma), "rendered_prompt_sha256": prompt_sha, "selected_layer": layer})
    return pd.DataFrame(rows)


def _finalize_reports(output: Path, gemma_status: str, detail: str) -> str:
    llama = json.loads((output / "LLAMA_EXPLORATORY_RESULT.json").read_text(encoding="utf-8"))
    decision = [
        "# Decision report", "", FROZEN_LLAMA_RESULT, "",
        "All Llama failure diagnostics are exploratory and post-outcome. Gemma is reported separately as prospective.", "",
        "## Llama diagnostic categories", "",
        *[f"- `{value}`" for value in llama["categories"]], "",
        "## Prospective Gemma", "", f"- Status: `{gemma_status}`.", f"- {detail}", "",
        "The primary Llama set contains 13 flipping originals; the mapping-consistent secondary contains only 4.",
    ]
    write_text_once(output / "DECISION_REPORT.md", "\n".join(decision))
    draft = (output / "PROSPECTIVE_REPLICATION_SPEC_DRAFT.md").read_text(encoding="utf-8")
    write_text_once(output / "PROSPECTIVE_REPLICATION_SPEC.md", draft + f"\n## Prospective Gemma disposition\n\n- `{gemma_status}`: {detail}\n")
    manifest = artifact_manifest(output)
    return write_json_once(output / "artifact_manifest.json", manifest)


def run_prospective_gemma(config_path: str | Path) -> dict[str, Any]:
    repo, output, contract, frozen_root = _context(config_path)
    freeze = freeze_gemma_contract(config_path)
    if freeze["prior_gemma_claim2_outcomes_inspected"] is not False:
        raise RuntimeError("Prospective Gemma status failed.")
    config = contract.raw
    contracts = tuple(value for value in contracts_from_config(config) if value.id in {"ab_standard", "ab_reversed"})
    prompt_config = _load_target_prompt_config(repo)
    originals = pd.read_csv(repo / config["paths"]["gate2_dir"] / "population_base_items_private.csv")
    candidates = pd.read_csv(repo / config["paths"]["gate2_dir"] / "population_candidates_private.csv")
    candidates = candidates[candidates["inference_eligible"].astype(str).str.lower().eq("true")]
    entry = {"family": "gemma2_prospective_target", **config["models"]["prospective_gemma"]}
    original_path = output / "checkpoints" / "gemma_original_repeat_control.parquet"
    candidate_path = output / "checkpoints" / "gemma_candidate_behavior.parquet"
    original_metadata_path = original_path.with_suffix(".metadata.json")
    candidate_metadata_path = candidate_path.with_suffix(".metadata.json")
    original_resumed = _resume_checkpoint(
        original_path,
        original_metadata_path,
        status="GEMMA_ORIGINAL_REPEAT_COMPLETE",
        revision=config["models"]["prospective_gemma"]["revision"],
    )
    if original_resumed is not None:
        original_scores, original_metadata = original_resumed
        original_parameter_hash = original_metadata["parameter_sha256"]
    else:
        loaded = load_jury_model(entry)
        try:
            import torch
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            from causal_token_smoke.fingerprint import parameter_sha256
            before = parameter_sha256(loaded.model)
            original_scores = _score_gemma_originals(loaded, originals, contracts, prompt_config, int(config["gemma"]["original_repeat_count"]))
            after = parameter_sha256(loaded.model)
            if before != after:
                raise RuntimeError("Gemma parameters changed during original-repeat scoring.")
            original_parameter_hash = before
            artifact_sha = write_once(original_path, _parquet_bytes(original_scores))
            write_json_once(original_metadata_path,{"status":"GEMMA_ORIGINAL_REPEAT_COMPLETE","artifact_sha256":artifact_sha,"parameter_sha256":before,"parameter_sha256_before":before,"parameter_sha256_after":after,"model_id":loaded.model_id,"model_revision":loaded.model_revision,"tokenizer_revision":loaded.tokenizer_revision,"tokenizer_sha256":_tokenizer_sha256(loaded.tokenizer),"prompt_score_contract_sha256":_prompt_score_contract_sha256(original_scores),"dtype":loaded.dtype,"gpu":loaded.device_name,"gpu_memory_mib":loaded.gpu_memory_mib,"attention_implementation":loaded.attention_implementation,"runtime_seconds":time.monotonic()-started,"peak_vram_bytes":int(torch.cuda.max_memory_allocated()),"batch_size":1,"rows":len(original_scores)})
        finally:
            unload_model(loaded)
    repeat_gate = _validate_original_repeat_gate(original_scores, config)
    repeat_gate_sha = write_json_once(
        output / "GEMMA_ORIGINAL_REPEAT_GATE.json", repeat_gate
    )
    if not repeat_gate["gate_passed"]:
        raise RuntimeError(
            "Gemma original repeat gate failed before candidate scoring; "
            f"receipt SHA-256: {repeat_gate_sha}"
        )
    candidate_resumed = _resume_checkpoint(
        candidate_path,
        candidate_metadata_path,
        status="GEMMA_CANDIDATE_BEHAVIOR_COMPLETE",
        revision=config["models"]["prospective_gemma"]["revision"],
    )
    if candidate_resumed is not None:
        candidate_scores, candidate_metadata = candidate_resumed
        candidate_parameter_hash = candidate_metadata["parameter_sha256"]
    else:
        loaded = load_jury_model(entry)
        try:
            import torch
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            from causal_token_smoke.fingerprint import parameter_sha256
            before = parameter_sha256(loaded.model)
            candidate_scores = _score_gemma_candidates(loaded, candidates, contracts, prompt_config)
            after = parameter_sha256(loaded.model)
            if before != after:
                raise RuntimeError("Gemma parameters changed during candidate scoring.")
            candidate_parameter_hash = before
            artifact_sha = write_once(candidate_path, _parquet_bytes(candidate_scores))
            write_json_once(candidate_metadata_path,{"status":"GEMMA_CANDIDATE_BEHAVIOR_COMPLETE","artifact_sha256":artifact_sha,"parameter_sha256":before,"parameter_sha256_before":before,"parameter_sha256_after":after,"model_id":loaded.model_id,"model_revision":loaded.model_revision,"tokenizer_revision":loaded.tokenizer_revision,"tokenizer_sha256":_tokenizer_sha256(loaded.tokenizer),"prompt_score_contract_sha256":_prompt_score_contract_sha256(candidate_scores),"dtype":loaded.dtype,"gpu":loaded.device_name,"gpu_memory_mib":loaded.gpu_memory_mib,"attention_implementation":loaded.attention_implementation,"runtime_seconds":time.monotonic()-started,"peak_vram_bytes":int(torch.cuda.max_memory_allocated()),"batch_size":1,"rows":len(candidate_scores)})
        finally:
            unload_model(loaded)
    candidate_outcomes, report = _summarize_behavior(original_scores, candidate_scores, candidates, config)
    if original_parameter_hash != candidate_parameter_hash:
        raise RuntimeError("Gemma parameter fingerprints differ across behavioral phases.")
    report["prospective_freeze_sha256"] = sha256_file(output / "GEMMA_CLAIM2_PROSPECTIVE_FREEZE.json")
    report["original_scoring_parameter_sha256"] = original_parameter_hash
    report["candidate_scoring_parameter_sha256"] = candidate_parameter_hash
    report["original_repeat_gate_sha256"] = repeat_gate_sha
    report_sha = write_json_once(output / "GEMMA_BEHAVIORAL_VIABILITY.json", report)
    if not report["gate_passed"]:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(
            ["Llama frozen", "Gemma prospective"],
            [13, report["representative_distinct_flipping_originals"]],
            color=["#4c78a8", "#e45756"],
        )
        axis.set_ylabel("Representative originals with >=1 flip")
        axis.set_title("Behavioral viability only; Gemma geometry not loaded")
        figure.tight_layout()
        figure.savefig(
            output / "figures" / "figure_7_llama_vs_gemma.png", dpi=160
        )
        plt.close(figure)
        write_text_once(output / "GEMMA_PROSPECTIVE_REPORT.md", f"# Gemma prospective report\n\nStatus: `{report['status']}`. Geometry was not loaded. Behavioral report SHA-256: `{report_sha}`.\n")
        manifest_sha = _finalize_reports(output, report["status"], "The frozen behavioral viability gate failed, so Gemma geometry was never loaded.")
        return {**report, "behavioral_report_sha256": report_sha, "artifact_manifest_sha256": manifest_sha}

    probe_path = output / "frozen_inputs" / "gemma_m1_probe_parameters.npz"
    geometry_path = output / "checkpoints" / "gemma_original_geometry.parquet"
    geometry_metadata_path = geometry_path.with_suffix(".metadata.json")
    geometry_resumed = _resume_checkpoint(
        geometry_path,
        geometry_metadata_path,
        status="GEMMA_GEOMETRY_COMPLETE_AFTER_VIABILITY",
        revision=config["models"]["prospective_gemma"]["revision"],
    )
    if geometry_resumed is not None:
        geometry, geometry_metadata = geometry_resumed
        if geometry_metadata.get("behavioral_gate_report_sha256") != report_sha:
            raise RuntimeError("Gemma geometry checkpoint is not bound to this viability report.")
    else:
        loaded = load_jury_model(entry)
        try:
            import torch
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            from causal_token_smoke.fingerprint import parameter_sha256
            before = parameter_sha256(loaded.model)
            geometry = _gemma_geometry(loaded, originals, contracts, prompt_config, probe_path)
            after = parameter_sha256(loaded.model)
            if before != after:
                raise RuntimeError("Gemma parameters changed during geometry extraction.")
            artifact_sha=write_once(geometry_path, _parquet_bytes(geometry))
            write_json_once(geometry_metadata_path,{"status":"GEMMA_GEOMETRY_COMPLETE_AFTER_VIABILITY","artifact_sha256":artifact_sha,"parameter_sha256":before,"parameter_sha256_before":before,"parameter_sha256_after":after,"model_id":loaded.model_id,"model_revision":loaded.model_revision,"tokenizer_revision":loaded.tokenizer_revision,"tokenizer_sha256":_tokenizer_sha256(loaded.tokenizer),"dtype":loaded.dtype,"gpu":loaded.device_name,"gpu_memory_mib":loaded.gpu_memory_mib,"attention_implementation":loaded.attention_implementation,"runtime_seconds":time.monotonic()-started,"peak_vram_bytes":int(torch.cuda.max_memory_allocated()),"behavioral_gate_report_sha256":report_sha,"batch_size":1,"rows":len(geometry)})
        finally:
            unload_model(loaded)
    geometry_item = geometry.groupby("base_item_id", as_index=False)["frozen_gemma_dim_geometry"].mean()
    original_item = original_scores[original_scores["repeat_index"].eq(0)].groupby("base_item_id", as_index=False)["semantic_margin"].mean().rename(columns={"semantic_margin": "native_gemma_margin"})
    original_item["native_gemma_confidence"] = original_item["native_gemma_margin"].abs()
    jury_scores = pd.read_parquet(output / "jury_model_scores.parquet")
    feature_freeze = json.loads((output / "BLIND_FEATURE_FREEZE.json").read_text(encoding="utf-8"))
    eligible = set(feature_freeze["eligible_jury_models"]) - {GEMMA_ID}
    jury_original = jury_scores[jury_scores["dataset"].eq("claim2_original") & jury_scores["model_id"].isin(eligible)].copy()
    jury_original["mapping"] = jury_original["prompt_contract"].map({"ab_standard":"standard","ab_reversed":"reversed"})
    jury_feature = aggregate_item_features(hard_votes(jury_original[["model_id","family","base_item_id","mapping","semantic_margin"]]), excluded_target_model=GEMMA_ID)[["base_item_id","leave_target_out_vote_entropy"]].rename(columns={"leave_target_out_vote_entropy":"leave_gemma_out_jury"})
    primary = candidate_outcomes[candidate_outcomes["population_role"].eq("primary_accepted")]
    population_specs = {
        "representative_primary": primary[primary["stratum"].eq("representative")],
        "representative_mapping_consistent_secondary": primary[primary["stratum"].eq("representative") & primary["original_mapping_agreement"] & primary["candidate_mapping_agreement"]],
        "enriched_geometry_confidence_diagnostic": primary[primary["stratum"].eq("disagreement")],
    }
    with np.load(frozen_root / "original_text_embeddings.npz", allow_pickle=False) as arrays:
        embedding_ids=list(map(str,arrays["base_item_id"])); embedding_values=arrays["embedding"].astype(np.float32)
    embedding_index={value:index for index,value in enumerate(embedding_ids)}
    summaries=[]
    prediction_frames=[]
    hyperparameter_frames=[]
    for population_index, (population, rows) in enumerate(population_specs.items()):
        items=rows.groupby("base_item_id",as_index=False).agg(k_i=("primary_semantic_flip","sum"),n_i=("candidate_id","size"),consideration_cluster_id=("base_item_id","first"))
        frozen_item_map=pd.read_csv(frozen_root/"population_items_private.csv").drop_duplicates("base_item_id")[["base_item_id","board_id","consideration_cluster_id"]]
        base=originals[["base_item_id"]].merge(frozen_item_map,on="base_item_id",how="left",validate="one_to_one")
        items=items.drop(columns="consideration_cluster_id").merge(base,on="base_item_id",validate="one_to_one").merge(original_item,on="base_item_id",validate="one_to_one").merge(geometry_item,on="base_item_id",validate="one_to_one").merge(jury_feature,on="base_item_id",validate="one_to_one")
        embeddings=embedding_values[[embedding_index[value] for value in items["base_item_id"].astype(str)]]
        assignment_population="representative_primary" if population=="representative_mapping_consistent_secondary" else population
        assignments=pd.read_csv(frozen_root/"populations"/assignment_population/"outer_fold_assignments.csv")
        assignments=assignments[assignments["base_item_id"].astype(str).isin(set(items["base_item_id"].astype(str)))]
        models={"M0":[],"M2":["original_text","native_gemma_confidence"],"M3":["original_text","native_gemma_confidence","frozen_gemma_dim_geometry"],"M5":["original_text","native_gemma_confidence","leave_gemma_out_jury"],"M8":["original_text","native_gemma_confidence","leave_gemma_out_jury","frozen_gemma_dim_geometry"]}
        predictions,hyperparameters=run_frozen_grouped_ladder(items,embeddings,assignments,models,c_grid=list(map(float,config["inference"]["regularization_c"])),pca_grid=list(map(int,config["inference"]["text_pca_dimensions"])),inner_folds=int(config["inference"]["frozen_inner_folds"]))
        predictions.insert(0,"population",population)
        hyperparameters.insert(0,"population",population)
        prediction_frames.append(predictions)
        hyperparameter_frames.append(hyperparameters)
        summary=summarize_ladder(predictions.drop(columns="population"),bootstrap_replicates=int(config["inference"]["bootstrap_replicates"]),permutation_replicates=int(config["inference"]["permutation_replicates"]),seed=int(config["inference"]["random_seed"]))
        averaged=predictions.groupby(["base_item_id","model"],as_index=False).agg(k_i=("k_i","first"),n_i=("n_i","first"),predicted_probability=("predicted_probability","mean"))
        m5=averaged[averaged["model"].eq("M5")]
        m8=averaged[averaged["model"].eq("M8")]
        pair_seed=int(config["inference"]["random_seed"])+5000+population_index
        pair_boot=item_bootstrap_metric_delta(m5,m8,int(config["inference"]["bootstrap_replicates"]),pair_seed)
        m5_loss=float(summary.loc[summary["model"].eq("M5"),"binomial_log_loss_per_rephrasing"].iloc[0])
        m8_loss=float(summary.loc[summary["model"].eq("M8"),"binomial_log_loss_per_rephrasing"].iloc[0])
        summary["delta_log_loss_M8_vs_M5"]=np.nan
        summary["delta_M8_vs_M5_ci_low"]=np.nan
        summary["delta_M8_vs_M5_ci_high"]=np.nan
        summary["M8_vs_M5_bootstrap_seed"]=np.nan
        m8_mask=summary["model"].eq("M8")
        summary.loc[m8_mask,"delta_log_loss_M8_vs_M5"]=m5_loss-m8_loss
        summary.loc[m8_mask,"delta_M8_vs_M5_ci_low"]=float(np.quantile(pair_boot,.025))
        summary.loc[m8_mask,"delta_M8_vs_M5_ci_high"]=float(np.quantile(pair_boot,.975))
        summary.loc[m8_mask,"M8_vs_M5_bootstrap_seed"]=pair_seed
        summary.insert(0,"population",population); summaries.append(summary)
    gemma_ladder=pd.concat(summaries,ignore_index=True)
    write_once(output/"gemma_predictor_ladder.csv",gemma_ladder.to_csv(index=False,lineterminator="\n").encode("utf-8"))
    gemma_predictions=pd.concat(prediction_frames,ignore_index=True)
    gemma_hyperparameters=pd.concat(hyperparameter_frames,ignore_index=True)
    write_once(output/"gemma_nested_predictions.parquet",_parquet_bytes(gemma_predictions))
    write_once(output/"gemma_nested_hyperparameters.parquet",_parquet_bytes(gemma_hyperparameters))
    import matplotlib.pyplot as plt
    llama_ladder=pd.read_csv(output/"predictor_ladder.csv")
    llama_primary=llama_ladder[llama_ladder["population"].eq("representative_primary") & llama_ladder["model"].isin(["M2","M3"])][["model","binomial_log_loss_per_rephrasing"]]
    gemma_primary=gemma_ladder[gemma_ladder["population"].eq("representative_primary") & gemma_ladder["model"].isin(["M2","M3"])][["model","binomial_log_loss_per_rephrasing"]]
    figure,axis=plt.subplots(figsize=(7,4)); x=np.arange(2); width=.35
    axis.bar(x-width/2,llama_primary.set_index("model").loc[["M2","M3"],"binomial_log_loss_per_rephrasing"],width,label="Llama frozen")
    axis.bar(x+width/2,gemma_primary.set_index("model").loc[["M2","M3"],"binomial_log_loss_per_rephrasing"],width,label="Gemma prospective")
    axis.set_xticks(x,["M2","M3"]); axis.set_ylabel("Held-out log loss"); axis.legend(); axis.set_title("Llama versus prospective Gemma")
    figure.tight_layout(); figure.savefig(output/"figures"/"figure_7_llama_vs_gemma.png",dpi=160); plt.close(figure)
    report_text=f"# Gemma prospective report\n\nStatus: `GEMMA_PROSPECTIVE_ANALYSIS_COMPLETE`. The behavioral viability gate passed before frozen Gemma geometry was loaded.\n\n- Representative distinct flipping originals: {report['representative_distinct_flipping_originals']}.\n- Identity-control flips: {report['identity_control_sign_flips']}.\n- Original-repeat maximum difference: {report['original_repeat_max_abs_difference']}.\n"
    write_text_once(output/"GEMMA_PROSPECTIVE_REPORT.md",report_text)
    manifest_sha=_finalize_reports(output,"GEMMA_PROSPECTIVE_ANALYSIS_COMPLETE","Behavioral viability passed before frozen Gemma geometry was loaded; see gemma_predictor_ladder.csv.")
    return {"status":"GEMMA_PROSPECTIVE_ANALYSIS_COMPLETE","behavioral_report_sha256":report_sha,"artifact_manifest_sha256":manifest_sha}
