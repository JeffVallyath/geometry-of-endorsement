from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_of_truth.truth.reproduction.cache import ChunkStore, concatenate_chunks

from .support.split_stress_test import CON, SIT

from .analysis import analyze
from .config import M1Config, load_config
from .manifests import (
    build_pilot_manifests,
    common_text_training_frame,
    file_sha256,
    load_materialized_pilot,
)
from .model import extract_batch, load_model
from .prompts import prompt_contract_hash
from .reference import compare_result
from .reporting import write_json_atomic, write_results


FINAL_ARTIFACTS = (
    "M1_VERTICAL_SLICE_REPORT.md",
    "m1_vertical_slice_results.json",
    "m1_layerwise_probe_performance.png",
    "m1_probe_parameters.npz",
    "environment.json",
    "run.log",
    "cache_index.csv",
    "cache_manifest.json",
    "M1_RUNTIME_VALIDATION.json",
    "reference_comparison.csv",
)

TRUTH_V2_PROTOCOL = "TRUTH_CONTROL_V2_NEUTRAL_MAPPING"
TRUTH_V2_SCIENTIFIC_COMMIT = "f1513cfce983abf81c1334b5d96e0bd9c280f2d8"
TRUTH_V2_CONFIG_SHA256 = (
    "61e9c98297fe85af4eec1ea064122155740cf8420bd818180371388322d95ee2"
)
TRUTH_V2_PROMPT_SHA256 = (
    "5b271d42d849f9c17cd939afbbb5b2c7b0281c4fd97a7286a3045bf0566b3b58"
)
TRUTH_V2_GEMMA_MODEL_ID = "google/gemma-2-9b-it"
TRUTH_V2_GEMMA_MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
TRUTH_V2_GEMMA_SCIENTIFIC_COMMIT = "a62aaf2398f53ef44e6b863127bb243f14675493"
TRUTH_V2_GEMMA_CONFIG_SHA256 = "8987e53657dbdb238d8c10ad97da3d513cd7d48d043224a42d70d4e0c7bae017"
TRUTH_V2_MODEL_CONTRACTS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "scientific_commit": TRUTH_V2_SCIENTIFIC_COMMIT,
        "config_sha256": TRUTH_V2_CONFIG_SHA256,
        "prompt_sha256": TRUTH_V2_PROMPT_SHA256,
        "resolved_revision": None,
    },
    TRUTH_V2_GEMMA_MODEL_ID: {
        "scientific_commit": TRUTH_V2_GEMMA_SCIENTIFIC_COMMIT,
        "config_sha256": TRUTH_V2_GEMMA_CONFIG_SHA256,
        "prompt_sha256": TRUTH_V2_PROMPT_SHA256,
        "resolved_revision": TRUTH_V2_GEMMA_MODEL_REVISION,
    },
}
TRUTH_V2_DATASET_SHA256 = {
    "cities": "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e",
    "neg_cities": "66c5c84de98ca15570138c205ab0e3907b57750083fac72a4eacafc3d9ebcea7",
}
TRUTH_V2_CHECKS = {
    "native_semantic_auroc_strictly_above_0.70",
    "complete_group_permutation_p_below_0.05",
    "selected_primary_ci_above_zero",
    "selected_transfer_ci_above_zero",
    "all_four_mapping_transfer_T_positive",
    "selected_layer_clear",
    "at_least_one_immediate_neighbor_clear",
    "layer_selected_from_primary_development_only",
}
TRUTH_V2_FINAL_FILES = {
    "TRUTH_CONTROL_V2_REPORT.md",
    "truth_control_v2_results.json",
    "layerwise_signed_separation.png",
    "environment.json",
    "run.log",
    "split_manifest.csv",
    "cache_manifest.json",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("m1_vertical_slice")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    for handler in (
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _truth_control_summary(
    path: Path,
    expected_model_id: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != TRUTH_V2_PROTOCOL:
        raise RuntimeError("M1 requires a TRUTH_CONTROL_V2_NEUTRAL_MAPPING result.")

    environment_path = path.parent / "environment.json"
    private_manifest_path = path.parent / "final_artifact_manifest.json"
    public_manifest_path = path.parent / "manifest.json"
    manifest_path = (
        private_manifest_path
        if private_manifest_path.is_file()
        else public_manifest_path
    )
    public_bundle = not private_manifest_path.is_file() and public_manifest_path.is_file()
    if not environment_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "Truth v2 import requires sibling environment.json and "
            "a retained artifact manifest."
        )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_model_id = str(environment.get("model", {}).get("id", ""))
    if expected_model_id is not None and source_model_id != expected_model_id:
        raise RuntimeError(
            f"M1 model {expected_model_id} requires its own truth control; "
            f"received {source_model_id or 'an unidentified model'}."
        )
    try:
        truth_contract = TRUTH_V2_MODEL_CONTRACTS[source_model_id]
    except KeyError as exc:
        raise RuntimeError(
            f"No frozen truth-control import contract exists for {source_model_id!r}."
        ) from exc

    null = payload["permutation_null"]
    null_values = np.asarray(null["values"], dtype=np.float64)
    if null_values.ndim != 1 or not np.isfinite(null_values).all():
        raise RuntimeError("Truth v2 permutation values must be a finite vector.")
    permutations = int(null["permutations"])
    observed = float(null["observed_T"])
    exceedances = int(np.sum(null_values >= observed))
    exact_p = float((1 + exceedances) / (permutations + 1))

    development = payload["development_layer_sweep"]
    if not development:
        raise RuntimeError("Truth v2 development layer sweep is empty.")
    selected_layer = int(payload["selected_layer"])
    recomputed_selected = int(
        max(
            development,
            key=lambda row: (float(row["selection_score"]), -int(row["layer"])),
        )["layer"]
    )
    candidates = [int(layer) for layer in payload["confirmatory_layer_candidates"]]
    layer_results = payload["confirmatory_layer_results"]
    selected_result = layer_results[str(selected_layer)]
    selected_test = selected_result["test"]
    selected_intervals = selected_result["group_bootstrap_ci"]

    native_values = payload["native_gate_values"]
    expected_native_keys = {
        f"{scheme}:{mapping}"
        for scheme in ("primary", "transfer")
        for mapping in ("overall", "standard", "reversed")
    }
    expected_mapping_keys = {
        f"{split}:{scheme}:{mapping}"
        for split in ("train", "dev", "test")
        for scheme in ("primary", "transfer")
        for mapping in ("standard", "reversed")
    }

    repository = environment.get(
        "repository",
        {
            "commit": truth_contract["scientific_commit"],
            "working_tree_clean": True,
            "status_porcelain": [],
        },
    )
    model = environment["model"]
    env_datasets = {
        str(row["name"]): str(row["sha256"])
        for row in environment["datasets"]
    }
    artifact_hashes = (
        final_manifest["public_files"] if public_bundle else final_manifest["files"]
    )
    result_digest = file_sha256(path)
    environment_digest = file_sha256(environment_path)
    source_checks = payload.get("checks", {})
    checks = {
        "protocol_exact": payload.get("protocol") == TRUTH_V2_PROTOCOL,
        "full_mode": payload.get("mode") == "full",
        "source_disposition_pass": payload.get("terminal_disposition")
        == "TRUTH_CONTROL_V2_PASS",
        "m1_development_authorized": (
            public_bundle or payload.get("m1_development_authorized") is True
        ),
        "v1_disposition_preserved": payload.get("v1_disposition_changed") is False,
        "source_check_set_exact": set(source_checks) == TRUTH_V2_CHECKS,
        "all_source_checks_true": set(source_checks) == TRUTH_V2_CHECKS
        and all(value is True for value in source_checks.values()),
        "eight_training_partitions": payload.get("training_partitions") == 8,
        "development_selection_recomputed": selected_layer == recomputed_selected,
        "candidate_layers_are_selected_and_neighbors": selected_layer in candidates
        and len(candidates) >= 2
        and len(candidates) == len(set(candidates))
        and all(abs(layer - selected_layer) <= 1 for layer in candidates),
        "selected_layer_result_matches": int(selected_result["layer"])
        == selected_layer,
        "primary_test_T_matches": bool(
            np.isclose(
                float(payload["primary_test_T"]), observed, rtol=0.0, atol=1e-12
            )
            and np.isclose(
                float(selected_test["primary"]["overall"]["T"]),
                observed,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "permutation_count_exact": permutations == 1000
        and len(null_values) == permutations,
        "permutation_exceedance_recomputed": int(
            null["count_greater_equal_observed"]
        )
        == exceedances,
        "permutation_p_recomputed": bool(
            np.isclose(
                float(null["p_greater_equal"]), exact_p, rtol=0.0, atol=1e-15
            )
        ),
        "permutation_p_below_0_05": exact_p < 0.05,
        "native_gate_keys_exact": set(native_values) == expected_native_keys,
        "native_semantic_aurocs_strictly_above_0_70": set(native_values)
        == expected_native_keys
        and all(float(value) > 0.70 for value in native_values.values()),
        "primary_ci_above_zero": float(selected_intervals["primary"]["low"])
        > 0.0
        and int(selected_intervals["primary"]["replicates"]) == 2000,
        "transfer_ci_above_zero": float(selected_intervals["transfer"]["low"])
        > 0.0
        and int(selected_intervals["transfer"]["replicates"]) == 2000,
        "all_mapping_transfer_cells_positive": all(
            float(selected_test[scheme][mapping]["T"]) > 0.0
            for scheme in ("primary", "transfer")
            for mapping in ("standard", "reversed")
        ),
        "selected_and_neighbor_clear": selected_result["clear_adjacent_effect"]
        is True
        and any(
            layer_results[str(layer)]["clear_adjacent_effect"] is True
            for layer in candidates
            if layer != selected_layer and abs(layer - selected_layer) == 1
        ),
        "auroc_is_descriptive_only": payload.get("auroc_role")
        == "descriptive_only_not_a_permutation_gate",
        "mapping_cells_complete": set(payload["mapping_counts"])
        == expected_mapping_keys
        and all(int(value) > 0 for value in payload["mapping_counts"].values()),
        "config_hash_frozen": payload.get("config_sha256")
        == truth_contract["config_sha256"]
        and environment["config"].get("sha256") == truth_contract["config_sha256"]
        and environment["config"].get("mode") == "full",
        "prompt_hash_frozen": payload.get("prompt_contract_sha256")
        == truth_contract["prompt_sha256"],
        "dataset_hashes_frozen": payload.get("dataset_sha256")
        == TRUTH_V2_DATASET_SHA256
        and env_datasets == TRUTH_V2_DATASET_SHA256,
        "scientific_commit_frozen_and_clean": repository.get("commit")
        == truth_contract["scientific_commit"]
        and repository.get("working_tree_clean") is True
        and repository.get("status_porcelain") == [],
        "bf16_exact_model_unquantized": model.get("id") == source_model_id
        and model.get("torch_dtype") == "bfloat16"
        and model.get("quantization") == "none"
        and model.get("resolved_revision") == payload.get("model_revision")
        and (
            truth_contract["resolved_revision"] is None
            or model.get("resolved_revision") == truth_contract["resolved_revision"]
        ),
        "artifact_manifest_protocol_exact": (
            final_manifest.get("schema_version") == 1
            if public_bundle
            else final_manifest.get("protocol") == TRUTH_V2_PROTOCOL
        ),
        "artifact_manifest_complete": (
            {"v2_results.json", "environment.json", "split_manifest.csv"}
            <= set(artifact_hashes)
            if public_bundle
            else TRUTH_V2_FINAL_FILES.issubset(artifact_hashes)
        ),
        "result_artifact_hash_verified": artifact_hashes.get(path.name)
        == result_digest,
        "environment_artifact_hash_verified": artifact_hashes.get("environment.json")
        == environment_digest,
    }
    return {
        "terminal_disposition": "PASS" if all(checks.values()) else "FAIL",
        "source_terminal_disposition": payload.get("terminal_disposition"),
        "protocol": payload.get("protocol"),
        "checks": checks,
        "selected_layer": selected_layer,
        "model_id": source_model_id,
        "primary_test_T": float(payload["primary_test_T"]),
        "exact_permutation_p": float(null["p_greater_equal"]),
        "directional_consensus_C": float(payload["directional_consensus_C"]),
        "scientific_commit": repository.get("commit"),
        "source_results_sha256": result_digest,
        "source_results_path": str(path),
        "source_environment_sha256": environment_digest,
        "source_environment_path": str(environment_path),
        "source_final_manifest_path": str(manifest_path),
        "provenance_mode": (
            "public_aggregate_bundle" if public_bundle else "full_retained_bundle"
        ),
    }


def _signature(
    config: M1Config,
    *,
    key: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_sha256: str,
    pilot_manifest_sha256: str,
    record_ids: list[str],
    repository_commit: str,
) -> str:
    payload = {
        "cache_schema": "m1_activation_cache_v2",
        "extraction_config_sha256": config.extraction_digest,
        "feature_key": key,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": chat_template_sha256,
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "record_ids_sha256": hashlib.sha256(
            "\n".join(record_ids).encode()
        ).hexdigest(),
        "prompt_contract_sha256": prompt_contract_hash(config.section("prompt")),
        "repository_commit": repository_commit,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _records(
    frame: pd.DataFrame, scheme: str, representation: str
) -> list[dict[str, str]]:
    return [
        {
            "item_id": str(row["item_id"]),
            "situation": str(row[SIT]),
            "consideration": str(row[CON]),
            "scheme": scheme,
            "representation": representation,
        }
        for _, row in frame.iterrows()
    ]


def _chunk_ranges(row_count: int, chunk_size: int) -> list[np.ndarray]:
    rows = np.arange(row_count)
    return [
        rows[start : start + chunk_size]
        for start in range(0, row_count, chunk_size)
    ]


def _extract_feature_set(
    *,
    key: str,
    scheme: str,
    representation: str,
    frame: pd.DataFrame,
    loaded: Any,
    config: M1Config,
    target: Path,
    persistent: Path | None,
    pilot_manifest_sha256: str,
    repository_commit: str,
    logger: logging.Logger,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
    record_ids = frame["item_id"].astype(str).tolist()
    records = _records(frame, scheme, representation)
    signature = _signature(
        config,
        key=key,
        model_revision=loaded.model_revision,
        tokenizer_revision=loaded.tokenizer_revision,
        chat_template_sha256=hashlib.sha256(
            loaded.tokenizer.chat_template.encode()
        ).hexdigest(),
        pilot_manifest_sha256=pilot_manifest_sha256,
        record_ids=record_ids,
        repository_commit=repository_commit,
    )
    local_root = target / "cache" / key
    persistent_root = persistent / "cache" / key if persistent else None
    store = ChunkStore(local_root, signature, persistent_root)
    chunks: list[dict[str, np.ndarray]] = []
    index_rows: list[dict[str, Any]] = []
    peak_vram = 0
    ranges = _chunk_ranges(len(frame), int(config.raw["cache"]["chunk_size"]))
    batch_size = int(config.raw["cache"]["batch_size"])
    for chunk_index, rows in enumerate(ranges):
        expected_ids = [record_ids[index] for index in rows]
        cached = store.load(chunk_index, expected_ids)
        if cached is None:
            batch_results = []
            prompt_hashes: list[str] = []
            for start in range(0, len(rows), batch_size):
                batch_indices = rows[start : start + batch_size]
                result = extract_batch(
                    loaded,
                    [records[index] for index in batch_indices],
                    config.section("prompt"),
                    str(config.raw["cache"]["activation_dtype"]),
                )
                batch_results.append(result)
                prompt_hashes.extend(
                    hashlib.sha256(text.encode()).hexdigest()
                    for text in result.rendered_prompts
                )
                peak_vram = max(peak_vram, result.peak_vram_bytes)
            store.save(
                chunk_index,
                expected_ids,
                activations=np.concatenate(
                    [result.activations for result in batch_results]
                ),
                supports_logp=np.concatenate(
                    [result.supports_logp for result in batch_results]
                ),
                opposes_logp=np.concatenate(
                    [result.opposes_logp for result in batch_results]
                ),
                native_margin=np.concatenate(
                    [result.native_margin for result in batch_results]
                ),
                predicted_labels=np.concatenate(
                    [result.predicted_labels for result in batch_results]
                ),
                prompt_token_indices=np.concatenate(
                    [result.prompt_token_indices for result in batch_results]
                ),
                prompt_sha256=np.asarray(prompt_hashes),
                mapping_names=np.asarray(
                    [
                        name
                        for result in batch_results
                        for name in result.mapping_names
                    ]
                ),
                candidate_token_lengths=np.concatenate(
                    [result.candidate_token_lengths for result in batch_results]
                ),
            )
            cached = store.load(chunk_index, expected_ids)
            if cached is None:
                raise RuntimeError(f"Saved cache chunk cannot be reloaded: {key}/{chunk_index}")
            logger.info(
                "Wrote and verified %s chunk %d/%d",
                key,
                chunk_index + 1,
                len(ranges),
            )
        else:
            logger.info(
                "Restored %s chunk %d/%d",
                key,
                chunk_index + 1,
                len(ranges),
            )
        chunks.append(cached)
        chunk_name = store.filename(chunk_index)
        entry = store.manifest["chunks"][chunk_name]
        for local_index, row_index in enumerate(rows):
            source = frame.iloc[int(row_index)]
            activation_shape = list(cached["activations"][local_index].shape)
            index_rows.append(
                {
                    "item_id": str(source["item_id"]),
                    "feature_key": key,
                    "scheme": scheme,
                    "representation": representation,
                    "situation_id": str(source["situation_id"]),
                    "consideration_cluster_id": str(
                        source["consideration_cluster_id"]
                    ),
                    "board_id": (
                        ""
                        if pd.isna(source["board_id"])
                        else str(source["board_id"])
                    ),
                    "reference_label": int(source["label"]),
                    "answer_mapping": str(cached["mapping_names"][local_index]),
                    "prompt_contract_sha256": prompt_contract_hash(
                        config.section("prompt")
                    ),
                    "rendered_prompt_sha256": str(
                        cached["prompt_sha256"][local_index]
                    ),
                    "model_revision": loaded.model_revision,
                    "tokenizer_revision": loaded.tokenizer_revision,
                    "chat_template_sha256": hashlib.sha256(
                        loaded.tokenizer.chat_template.encode()
                    ).hexdigest(),
                    "layer_indices": f"0..{activation_shape[0] - 1}",
                    "token_position": int(
                        cached["prompt_token_indices"][local_index]
                    ),
                    "activation_dtype": str(
                        cached["activations"][local_index].dtype
                    ),
                    "activation_shape": "x".join(map(str, activation_shape)),
                    "pilot_manifest_sha256": pilot_manifest_sha256,
                    "repository_commit": repository_commit,
                    "cache_file": f"{key}/activation_chunks/{chunk_name}",
                    "cache_file_sha256": entry["sha256"],
                    "cache_signature": signature,
                }
            )
    # If every chunk was restored from Drive, ChunkStore loaded the persistent
    # ledger in memory but did not need to save a chunk. Materialize the same
    # ledger locally so the aggregate final manifest is always complete.
    store._write_manifest()
    combined = concatenate_chunks(chunks)
    if [str(value) for value in combined["record_ids"]] != record_ids:
        raise RuntimeError(f"Cache order differs from the pilot manifest for {key}.")
    return combined, index_rows, peak_vram


def _runtime_validation(
    *,
    loaded: Any,
    frame: pd.DataFrame,
    config: M1Config,
    cached_primary: dict[str, np.ndarray],
) -> dict[str, Any]:
    validation_batch_size = min(
        int(config.raw["cache"]["batch_size"]), len(frame)
    )
    records = _records(
        frame.iloc[:validation_batch_size], "primary", "joint"
    )
    activation_dtype = str(config.raw["cache"]["activation_dtype"])
    production_batch = extract_batch(
        loaded, records, config.section("prompt"), activation_dtype
    )
    single_item_runs = [
        extract_batch(
            loaded, [record], config.section("prompt"), activation_dtype
        )
        for record in records
    ]
    first_repeat = extract_batch(
        loaded, [records[0]], config.section("prompt"), activation_dtype
    )
    tolerance = 2e-3
    cached_slice = slice(0, validation_batch_size)
    production_prompt_hashes = np.asarray(
        [
            hashlib.sha256(prompt.encode()).hexdigest()
            for prompt in production_batch.rendered_prompts
        ]
    )
    single_prompt_hashes = np.asarray(
        [
            hashlib.sha256(result.rendered_prompts[0].encode()).hexdigest()
            for result in single_item_runs
        ]
    )
    single_activations = np.stack(
        [result.activations[0] for result in single_item_runs]
    ).astype(np.float32)
    production_activations = production_batch.activations.astype(np.float32)
    cross_batch_delta = np.abs(single_activations - production_activations)
    cross_batch_cosines = []
    for single, production in zip(
        single_activations, production_activations, strict=True
    ):
        single_flat = single.reshape(-1).astype(np.float64)
        production_flat = production.reshape(-1).astype(np.float64)
        denominator = np.linalg.norm(single_flat) * np.linalg.norm(
            production_flat
        )
        cross_batch_cosines.append(
            float(np.dot(single_flat, production_flat) / denominator)
            if denominator
            else 1.0
        )
    single_margins = np.asarray(
        [result.native_margin[0] for result in single_item_runs],
        dtype=np.float32,
    )
    single_predictions = np.asarray(
        [result.predicted_labels[0] for result in single_item_runs],
        dtype=np.int8,
    )
    expected_layers = int(loaded.model.config.num_hidden_layers)
    checks = {
        "model_loaded": True,
        "expected_hidden_state_layers": bool(
            production_batch.activations.shape[1] == expected_layers
            and all(
                result.activations.shape[1] == expected_layers
                for result in single_item_runs
            )
        ),
        "sequence_candidate_scores_finite": bool(
            np.isfinite(production_batch.supports_logp).all()
            and np.isfinite(production_batch.opposes_logp).all()
            and all(
                np.isfinite(result.supports_logp).all()
                and np.isfinite(result.opposes_logp).all()
                for result in single_item_runs
            )
        ),
        "candidate_lengths_positive": bool(
            (production_batch.candidate_token_lengths > 0).all()
            and all(
                (result.candidate_token_lengths > 0).all()
                for result in single_item_runs
            )
        ),
        "deterministic_repeat_atol_0_002": bool(
            np.allclose(
                single_item_runs[0].activations,
                first_repeat.activations,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "same_batch_cache_activations_atol_0_002": bool(
            np.allclose(
                np.asarray(cached_primary["activations"])[cached_slice],
                production_batch.activations,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "same_batch_cache_candidate_scores_atol_0_002": bool(
            np.allclose(
                np.asarray(cached_primary["supports_logp"])[cached_slice],
                production_batch.supports_logp,
                rtol=0.0,
                atol=tolerance,
            )
            and np.allclose(
                np.asarray(cached_primary["opposes_logp"])[cached_slice],
                production_batch.opposes_logp,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "same_batch_cache_predictions_exact": bool(
            np.array_equal(
                np.asarray(cached_primary["predicted_labels"])[cached_slice],
                production_batch.predicted_labels,
            )
        ),
        "same_batch_cache_prompt_hashes_exact": bool(
            np.array_equal(
                np.asarray(cached_primary["prompt_sha256"])[
                    cached_slice
                ].astype(str),
                production_prompt_hashes,
            )
        ),
        "prompt_text_batch_shape_invariant": bool(
            np.array_equal(single_prompt_hashes, production_prompt_hashes)
        ),
        "final_prompt_positions_batch_shape_invariant": bool(
            np.array_equal(
                np.asarray(
                    [
                        result.prompt_token_indices[0]
                        for result in single_item_runs
                    ]
                ),
                production_batch.prompt_token_indices,
            )
            and np.array_equal(
                np.asarray(cached_primary["prompt_token_indices"])[cached_slice],
                production_batch.prompt_token_indices,
            )
        ),
        "semantic_mapping_names_batch_shape_invariant": bool(
            [result.mapping_names[0] for result in single_item_runs]
            == production_batch.mapping_names
            and np.array_equal(
                np.asarray(cached_primary["mapping_names"])[
                    cached_slice
                ].astype(str),
                np.asarray(production_batch.mapping_names),
            )
        ),
        "semantic_decisions_batch_shape_invariant": bool(
            np.array_equal(single_predictions, production_batch.predicted_labels)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "validation_contract": "m1_runtime_validation_v2_fixed_batch_reproduction",
        "validation_batch_size": validation_batch_size,
        "first_item_id": str(frame.iloc[0]["item_id"]),
        "rendered_prompt_sha256": str(single_prompt_hashes[0]),
        "final_prompt_token_index": int(
            single_item_runs[0].prompt_token_indices[0]
        ),
        "hidden_state_layers": int(single_item_runs[0].activations.shape[1]),
        "activation_tolerance": tolerance,
        "cross_batch_activation_comparison_role": (
            "descriptive_only; BF16 kernels are validated by deterministic "
            "same-batch cache reproduction, while prompt positions, mappings, "
            "and semantic decisions must remain batch-shape invariant"
        ),
        "cross_batch_activation_max_abs": float(cross_batch_delta.max()),
        "cross_batch_activation_mean_abs": float(cross_batch_delta.mean()),
        "cross_batch_activation_p99_abs": float(
            np.quantile(cross_batch_delta, 0.99)
        ),
        "cross_batch_activation_min_cosine": float(min(cross_batch_cosines)),
        "cross_batch_native_margin_max_abs": float(
            np.max(
                np.abs(
                    single_margins
                    - production_batch.native_margin.astype(np.float32)
                )
            )
        ),
        "contains_source_text": False,
        "peak_vram_bytes": max(
            production_batch.peak_vram_bytes,
            first_repeat.peak_vram_bytes,
            *(result.peak_vram_bytes for result in single_item_runs),
        ),
    }


def _environment(
    *,
    repo_root: Path,
    config: M1Config,
    loaded: Any,
    manifest_metadata: dict[str, Any],
    repository_commit: str,
) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pandas",
        "scikit-learn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    status = _git(repo_root, "status", "--porcelain")
    return {
        "repository": {
            "commit": repository_commit,
            "branch": _git(repo_root, "branch", "--show-current"),
            "working_tree_clean": status == "",
            "status_porcelain": status.splitlines(),
        },
        "config": {
            "path": str(config.path),
            "sha256": config.digest,
            "mode": config.mode,
        },
        "model": {
            "id": config.raw["model"]["id"],
            "requested_revision": config.raw["model"]["revision"],
            "resolved_revision": loaded.model_revision,
            "tokenizer_revision": loaded.tokenizer_revision,
            "chat_template_sha256": hashlib.sha256(
                loaded.tokenizer.chat_template.encode()
            ).hexdigest(),
            "torch_dtype": loaded.activation_torch_dtype,
            "quantization": config.raw["model"]["quantization"],
        },
        "pilot_manifest": manifest_metadata,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "gpu": {
                "name": loaded.device_name,
                "memory_total_mib": loaded.gpu_memory_mib,
            },
        },
    }


def _persist_final(output_dir: Path, persistent: Path | None) -> dict[str, str]:
    if persistent is None:
        return {}
    persistent.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in FINAL_ARTIFACTS:
        source = output_dir / name
        if not source.exists():
            continue
        target = persistent / name
        partial = target.with_suffix(target.suffix + ".part")
        shutil.copy2(source, partial)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        os.replace(partial, target)
        digests[name] = digest
    write_json_atomic(
        {"files": digests},
        persistent / "final_artifact_manifest.json",
    )
    return digests


def dry_run_plan(
    config: M1Config,
    *,
    output_dir: str | Path,
    persistent_dir: str | Path | None,
    manifest_dir: str | Path | None,
    m0_dir: str | Path,
    truth_results: str | Path,
) -> dict[str, Any]:
    return {
        "status": "DRY_RUN_ONLY",
        "config": str(config.path),
        "config_sha256": config.digest,
        "mode": config.mode,
        "model": config.raw["model"],
        "output_dir": str(Path(output_dir).resolve()),
        "persistent_dir": (
            str(Path(persistent_dir).resolve()) if persistent_dir else None
        ),
        "manifest_dir": str(Path(manifest_dir).resolve()) if manifest_dir else None,
        "m0_dir": str(Path(m0_dir).resolve()),
        "truth_results": str(Path(truth_results).resolve()),
        "gpu_network_or_gated_data_touched": False,
    }


def _recommended_action(disposition: str) -> str:
    return {
        "M1_VERTICAL_SLICE_SIGNAL_SUPPORTED": (
            "Preserve this pilot unchanged and wait for the independent M0 audit/"
            "confirmatory manifest before any confirmatory replication."
        ),
        "M1_VERTICAL_SLICE_LINEAR_DECODING_ONLY": (
            "Treat the relation as predictively decodable but not intervention-ready; "
            "test a prespecified simple-direction replication without opening confirmatory data."
        ),
        "M1_VERTICAL_SLICE_VALID_NULL": (
            "Stop this exact representation/protocol branch; any continuation needs a "
            "materially new, preregistered observation rather than a tuned retry."
        ),
        "M1_VERTICAL_SLICE_INCONCLUSIVE": (
            "Increase only the nonconfirmatory development sample under the frozen "
            "protocol; do not inspect the confirmatory set."
        ),
        "M1_PIPELINE_NOT_VALIDATED": (
            "Repair the named failed validation gate and rerun the controls before "
            "interpreting any moral result."
        ),
        "M1_SMOKE_ONLY_NOT_EMPIRICAL": (
            "Run the full frozen configuration only after this smoke artifact passes."
        ),
    }[disposition]


def run(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    truth_results: str | Path,
    persistent_dir: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    m0_dir: str | Path | None = None,
    dry_run: bool = False,
    require_smoke_mode: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if require_smoke_mode and config.mode != "smoke":
        raise RuntimeError("--smoke-test requires run.mode=smoke.")
    if dry_run:
        return dry_run_plan(
            config,
            output_dir=output_dir,
            persistent_dir=persistent_dir,
            manifest_dir=manifest_dir,
            m0_dir=m0_dir or Path(__file__).resolve().parents[3],
            truth_results=truth_results,
        )

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    m0_root = Path(m0_dir).resolve() if m0_dir else repo_root
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    persistent = Path(persistent_dir).resolve() if persistent_dir else None
    manifests = (
        Path(manifest_dir).resolve()
        if manifest_dir
        else ((persistent / "manifests") if persistent else (target / "manifests"))
    )
    truth_path = Path(truth_results).resolve()
    if not truth_path.is_file():
        raise RuntimeError(
            f"Truth-control results are missing: {truth_path}. Run the full truth "
            "positive control first."
        )
    logger = _logger(target)
    logger.info(
        "Starting %s (%s); config=%s",
        config.raw["run"]["name"],
        config.mode,
        config.digest,
    )
    repository_commit = _git(repo_root, "rev-parse", "HEAD")
    repository_clean_at_start = _git(repo_root, "status", "--porcelain") == ""
    if not repository_clean_at_start:
        raise RuntimeError(
            "The execution checkout is dirty. Run a pinned clean commit so cache "
            "provenance cannot diverge from the recorded code."
        )

    truth_summary = _truth_control_summary(
        truth_path,
        expected_model_id=str(config.raw["model"]["id"]),
    )
    manifest_metadata = build_pilot_manifests(
        config,
        m0_root,
        manifests,
    )
    frame, manifest_metadata = load_materialized_pilot(
        config,
        m0_root,
        manifests,
    )
    pilot_manifest_sha256 = file_sha256(manifests / "pilot_manifest.json")
    select_boards = pd.read_csv(manifests / "pilot_select_boards.csv")
    eval_boards = pd.read_csv(manifests / "pilot_eval_boards.csv")
    logger.info(
        "Frozen %d/%d/%d pilot rows and %d evaluation boards before extraction; manifest=%s",
        manifest_metadata["counts"]["pilot_train"],
        manifest_metadata["counts"]["pilot_select"],
        manifest_metadata["counts"]["pilot_eval"],
        manifest_metadata["board_counts"]["pilot_eval"],
        pilot_manifest_sha256,
    )

    loaded = load_model(config)
    logger.info(
        "Loaded %s at %s on %s (%d MiB), dtype=%s",
        config.raw["model"]["id"],
        loaded.model_revision,
        loaded.device_name,
        loaded.gpu_memory_mib,
        loaded.activation_torch_dtype,
    )
    specs = (
        ("primary_joint", "primary", "joint"),
        ("primary_situation", "primary", "situation"),
        ("primary_consideration", "primary", "consideration"),
        ("transfer_joint", "transfer", "joint"),
    )
    extracted: dict[str, dict[str, np.ndarray]] = {}
    cache_index_rows: list[dict[str, Any]] = []
    peak_vram = 0
    for key, scheme, representation in specs:
        combined, index_rows, feature_peak = _extract_feature_set(
            key=key,
            scheme=scheme,
            representation=representation,
            frame=frame,
            loaded=loaded,
            config=config,
            target=target,
            persistent=persistent,
            pilot_manifest_sha256=pilot_manifest_sha256,
            repository_commit=repository_commit,
            logger=logger,
        )
        extracted[key] = combined
        cache_index_rows.extend(index_rows)
        peak_vram = max(peak_vram, feature_peak)

    cache_index = pd.DataFrame(cache_index_rows)
    cache_index_partial = target / "cache_index.csv.part"
    cache_index.to_csv(cache_index_partial, index=False, lineterminator="\n")
    os.replace(cache_index_partial, target / "cache_index.csv")
    cache_manifest = {
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "repository_commit": repository_commit,
        "feature_sets": {
            key: {
                "manifest_sha256": file_sha256(
                    target / "cache" / key / "cache_manifest.json"
                ),
                "manifest": json.loads(
                    (
                        target / "cache" / key / "cache_manifest.json"
                    ).read_text(encoding="utf-8")
                ),
            }
            for key, _, _ in specs
        },
    }
    write_json_atomic(cache_manifest, target / "cache_manifest.json")
    runtime_validation = _runtime_validation(
        loaded=loaded,
        frame=frame,
        config=config,
        cached_primary=extracted["primary_joint"],
    )
    runtime_validation.update(
        {
            "pilot_manifest_sha256": pilot_manifest_sha256,
            "repository_commit": repository_commit,
            "model_revision": loaded.model_revision,
            "tokenizer_revision": loaded.tokenizer_revision,
            "chat_template_sha256": hashlib.sha256(
                loaded.tokenizer.chat_template.encode()
            ).hexdigest(),
        }
    )
    peak_vram = max(peak_vram, int(runtime_validation["peak_vram_bytes"]))
    write_json_atomic(
        runtime_validation,
        target / "M1_RUNTIME_VALIDATION.json",
    )
    if runtime_validation["status"] != "PASS":
        raise RuntimeError(
            "M1 runtime validation failed; do not interpret or continue the run."
        )

    required_cache_columns = {
        "item_id",
        "situation_id",
        "consideration_cluster_id",
        "reference_label",
        "answer_mapping",
        "prompt_contract_sha256",
        "rendered_prompt_sha256",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "layer_indices",
        "token_position",
        "activation_dtype",
        "activation_shape",
        "pilot_manifest_sha256",
        "repository_commit",
        "cache_file_sha256",
    }
    pipeline_checks = {
        **{
            f"manifest_{name}": bool(value)
            for name, value in manifest_metadata["checks"].items()
        },
        "repository_clean_at_start": repository_clean_at_start,
        "runtime_validation_passed": runtime_validation["status"] == "PASS",
        "cache_index_row_count": len(cache_index) == len(frame) * len(specs),
        "cache_provenance_columns_complete": required_cache_columns
        <= set(cache_index.columns),
        "cache_provenance_values_complete": not cache_index[
            sorted(required_cache_columns)
        ]
        .isna()
        .any()
        .any(),
        "primary_mapping_consistent_across_representations": all(
            np.array_equal(
                extracted["primary_joint"]["mapping_names"],
                extracted[key]["mapping_names"],
            )
            for key in ("primary_situation", "primary_consideration")
        ),
        "candidate_sequence_lengths_positive": all(
            bool((value["candidate_token_lengths"] > 0).all())
            for value in extracted.values()
        ),
        "confirmatory_manifest_used_for_id_exclusion_only": True,
        "no_source_text_in_cache_index": not {
            SIT,
            CON,
            "rendered_prompt",
        }
        & set(cache_index.columns),
    }
    features = {
        key: np.asarray(value["activations"]) for key, value in extracted.items()
    }
    native_margins = {
        "primary": np.asarray(extracted["primary_joint"]["native_margin"]),
        "transfer": np.asarray(extracted["transfer_joint"]["native_margin"]),
    }
    mapping_names = {
        "primary": np.asarray(extracted["primary_joint"]["mapping_names"]),
        "transfer": np.asarray(extracted["transfer_joint"]["mapping_names"]),
    }
    results, probes = analyze(
        frame=frame,
        features=features,
        native_margins=native_margins,
        mapping_names=mapping_names,
        select_board_manifest=select_boards,
        eval_board_manifest=eval_boards,
        truth_control=truth_summary,
        pipeline_checks=pipeline_checks,
        config=config,
        text_baseline_train=(
            None
            if config.mode == "smoke"
            else common_text_training_frame(config, m0_root)
        ),
    )
    probes.update(
        {
            "pilot_manifest_sha256": np.asarray(pilot_manifest_sha256),
            "repository_commit": np.asarray(repository_commit),
            "model_revision": np.asarray(loaded.model_revision),
            "tokenizer_revision": np.asarray(loaded.tokenizer_revision),
            "chat_template_sha256": np.asarray(
                hashlib.sha256(
                    loaded.tokenizer.chat_template.encode()
                ).hexdigest()
            ),
        }
    )
    probe_partial = target / "m1_probe_parameters.npz.part"
    with probe_partial.open("wb") as handle:
        np.savez(handle, **probes)
    os.replace(probe_partial, target / "m1_probe_parameters.npz")

    elapsed = time.monotonic() - started
    cache_bytes = sum(
        path.stat().st_size for path in (target / "cache").rglob("*.npz")
    )
    results.update(
        {
            "runtime": {
                "repository_commit": repository_commit,
                "model_id": config.raw["model"]["id"],
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "chat_template_sha256": hashlib.sha256(
                    loaded.tokenizer.chat_template.encode()
                ).hexdigest(),
                "gpu_name": loaded.device_name,
                "gpu_memory_mib": loaded.gpu_memory_mib,
                "model_dtype": loaded.activation_torch_dtype,
                "activation_dtype": config.raw["cache"]["activation_dtype"],
                "peak_vram_bytes": int(peak_vram),
                "elapsed_seconds": float(elapsed),
                "cache_bytes": int(cache_bytes),
            },
            "manifest": {
                "pilot_manifest_sha256": pilot_manifest_sha256,
                "m0_manifest_sha256": manifest_metadata[
                    "m0_manifest_sha256"
                ],
                "source_row_id_sha256": manifest_metadata[
                    "source_row_id_sha256"
                ],
                "split_manifest_sha256": manifest_metadata["manifest_sha256"],
                "board_manifest_sha256": manifest_metadata[
                    "board_manifest_sha256"
                ],
                "contains_source_text": False,
            },
            "config_sha256": config.digest,
            "prompt_contract_sha256": prompt_contract_hash(
                config.section("prompt")
            ),
            "prompt_specification": config.section("prompt"),
            "cache_index_sha256": file_sha256(target / "cache_index.csv"),
            "cache_manifest_sha256": file_sha256(
                target / "cache_manifest.json"
            ),
            "probe_parameters_sha256": file_sha256(
                target / "m1_probe_parameters.npz"
            ),
            "persistent_dir": str(persistent) if persistent else None,
            "failed_experiments": [],
            "reproduction_commands": [
                (
                    "python scripts/run_m1_development.py "
                    f"--config {config.path} --truth-results {truth_path} "
                    f"--m0-dir {m0_root} --output-dir {target}"
                    + (
                        f" --persistent-dir {persistent} --manifest-dir {manifests}"
                        if persistent
                        else f" --manifest-dir {manifests}"
                    )
                )
            ],
            "recommended_next_action": _recommended_action(
                results["terminal_disposition"]
            ),
        }
    )
    environment = _environment(
        repo_root=repo_root,
        config=config,
        loaded=loaded,
        manifest_metadata=manifest_metadata,
        repository_commit=repository_commit,
    )
    reference_comparison = None
    if config.mode == "full":
        reference_comparison = compare_result(results)
        reference_comparison.to_csv(
            target / "reference_comparison.csv", index=False, lineterminator="\n"
        )
        results["public_reproduction_reference"] = {
            "checks": int(len(reference_comparison)),
            "passed": int(reference_comparison["pass"].sum()),
            "all_passed": bool(reference_comparison["pass"].all()),
            "comparison_file": "reference_comparison.csv",
        }
    write_json_atomic(environment, target / "environment.json")
    write_results(results, target)
    logger.info(
        "Terminal disposition: %s", results["terminal_disposition"]
    )
    for handler in logger.handlers:
        handler.flush()
    persisted = _persist_final(target, persistent)
    logger.info("Persisted %d final artifacts", len(persisted))
    if reference_comparison is not None and not bool(reference_comparison["pass"].all()):
        failed = reference_comparison.loc[
            ~reference_comparison["pass"], "quantity"
        ].tolist()
        raise RuntimeError(
            f"M1 reproduction differs from the retained public reference for {failed}"
        )
    return results
