from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from geometry_of_truth.m1.config import load_config
from geometry_of_truth.m1.runner import (
    TRUTH_V2_CHECKS,
    TRUTH_V2_CONFIG_SHA256,
    TRUTH_V2_DATASET_SHA256,
    TRUTH_V2_GEMMA_CONFIG_SHA256,
    TRUTH_V2_GEMMA_MODEL_ID,
    TRUTH_V2_GEMMA_MODEL_REVISION,
    TRUTH_V2_GEMMA_SCIENTIFIC_COMMIT,
    TRUTH_V2_PROMPT_SHA256,
    TRUTH_V2_PROTOCOL,
    TRUTH_V2_SCIENTIFIC_COMMIT,
    _truth_control_summary,
    dry_run_plan,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layer_result(layer: int, primary_t: float = 2.0) -> dict:
    return {
        "layer": layer,
        "directional_consensus_C": 0.8,
        "test": {
            "primary": {
                "overall": {"T": primary_t},
                "standard": {"T": 1.9},
                "reversed": {"T": 1.8},
            },
            "transfer": {
                "overall": {"T": 1.7},
                "standard": {"T": 1.6},
                "reversed": {"T": 1.5},
            },
        },
        "group_bootstrap_ci": {
            "primary": {"low": 0.4, "high": 2.4, "replicates": 2000},
            "transfer": {"low": 0.3, "high": 2.2, "replicates": 2000},
        },
        "clear_adjacent_effect": True,
    }


def truth_v2_payload() -> dict:
    mapping_counts = {
        f"{split}:{scheme}:{mapping}": 10
        for split in ("train", "dev", "test")
        for scheme in ("primary", "transfer")
        for mapping in ("standard", "reversed")
    }
    return {
        "protocol": TRUTH_V2_PROTOCOL,
        "terminal_disposition": "TRUTH_CONTROL_V2_PASS",
        "mode": "full",
        "selected_layer": 14,
        "training_partitions": 8,
        "development_layer_sweep": [
            {"layer": 13, "selection_score": 0.8},
            {"layer": 14, "selection_score": 1.0},
            {"layer": 15, "selection_score": 0.9},
        ],
        "confirmatory_layer_candidates": [13, 14, 15],
        "confirmatory_layer_results": {
            "13": _layer_result(13, 1.8),
            "14": _layer_result(14, 2.0),
            "15": _layer_result(15, 1.7),
        },
        "primary_test_T": 2.0,
        "directional_consensus_C": 0.8,
        "permutation_null": {
            "permutations": 1000,
            "observed_T": 2.0,
            "values": [-1.0] * 1000,
            "count_greater_equal_observed": 0,
            "p_greater_equal": 1 / 1001,
        },
        "native_gate_values": {
            f"{scheme}:{mapping}": 0.9
            for scheme in ("primary", "transfer")
            for mapping in ("overall", "standard", "reversed")
        },
        "checks": {name: True for name in TRUTH_V2_CHECKS},
        "auroc_role": "descriptive_only_not_a_permutation_gate",
        "v1_disposition_changed": False,
        "m1_development_authorized": True,
        "model_revision": "model-revision",
        "config_sha256": TRUTH_V2_CONFIG_SHA256,
        "prompt_contract_sha256": TRUTH_V2_PROMPT_SHA256,
        "dataset_sha256": dict(TRUTH_V2_DATASET_SHA256),
        "mapping_counts": mapping_counts,
    }


def write_truth_v2_bundle(tmp_path: Path, payload: dict | None = None) -> Path:
    payload = truth_v2_payload() if payload is None else payload
    path = tmp_path / "truth_control_v2_results.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    environment = {
        "repository": {
            "commit": TRUTH_V2_SCIENTIFIC_COMMIT,
            "working_tree_clean": True,
            "status_porcelain": [],
        },
        "config": {"sha256": TRUTH_V2_CONFIG_SHA256, "mode": "full"},
        "model": {
            "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "resolved_revision": "model-revision",
            "torch_dtype": "bfloat16",
            "quantization": "none",
        },
        "datasets": [
            {"name": name, "sha256": digest, "rows": 100}
            for name, digest in TRUTH_V2_DATASET_SHA256.items()
        ],
    }
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = {
        "TRUTH_CONTROL_V2_REPORT.md": "0" * 64,
        "truth_control_v2_results.json": _digest(path),
        "layerwise_signed_separation.png": "0" * 64,
        "environment.json": _digest(environment_path),
        "run.log": "0" * 64,
        "split_manifest.csv": "0" * 64,
        "cache_manifest.json": "0" * 64,
    }
    (tmp_path / "final_artifact_manifest.json").write_text(
        json.dumps({"protocol": TRUTH_V2_PROTOCOL, "files": files}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_truth_v2_gate_recomputes_every_frozen_condition(tmp_path) -> None:
    path = write_truth_v2_bundle(tmp_path)
    summary = _truth_control_summary(path)
    assert summary["terminal_disposition"] == "PASS"
    assert all(summary["checks"].values())
    assert summary["selected_layer"] == 14
    assert summary["exact_permutation_p"] == pytest.approx(1 / 1001)


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (
            lambda payload: payload["checks"].__setitem__(
                "selected_primary_ci_above_zero", False
            ),
            "all_source_checks_true",
        ),
        (
            lambda payload: payload["permutation_null"].__setitem__(
                "p_greater_equal", 0.001
            ),
            "permutation_p_recomputed",
        ),
        (
            lambda payload: payload["development_layer_sweep"][0].__setitem__(
                "selection_score", 2.0
            ),
            "development_selection_recomputed",
        ),
        (
            lambda payload: payload["native_gate_values"].__setitem__(
                "primary:standard", 0.70
            ),
            "native_semantic_aurocs_strictly_above_0_70",
        ),
    ],
)
def test_truth_v2_gate_rejects_tampering(tmp_path, mutation, failed_check) -> None:
    payload = truth_v2_payload()
    mutation(payload)
    path = write_truth_v2_bundle(tmp_path, payload)
    summary = _truth_control_summary(path)
    assert summary["terminal_disposition"] == "FAIL"
    assert summary["checks"][failed_check] is False


def test_truth_v2_gate_rejects_wrong_environment_commit(tmp_path) -> None:
    path = write_truth_v2_bundle(tmp_path)
    environment_path = tmp_path / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["repository"]["commit"] = "0" * 40
    environment_path.write_text(json.dumps(environment) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "final_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["environment.json"] = _digest(environment_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    summary = _truth_control_summary(path)
    assert summary["terminal_disposition"] == "FAIL"
    assert summary["checks"]["scientific_commit_frozen_and_clean"] is False


def test_truth_v2_gate_rejects_manifest_digest_mismatch(tmp_path) -> None:
    path = write_truth_v2_bundle(tmp_path)
    manifest_path = tmp_path / "final_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    summary = _truth_control_summary(path)
    assert summary["terminal_disposition"] == "FAIL"
    assert summary["checks"]["result_artifact_hash_verified"] is False


def test_truth_v1_payload_is_not_accepted(tmp_path) -> None:
    path = tmp_path / "truth_baseline_results.json"
    path.write_text(json.dumps({"terminal_disposition": "PASS"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="TRUTH_CONTROL_V2_NEUTRAL_MAPPING"):
        _truth_control_summary(path)



def test_dry_run_touches_no_gpu_network_or_gated_data(tmp_path) -> None:
    config = load_config("configs/m1_development_smoke.yaml")
    plan = dry_run_plan(
        config,
        output_dir=tmp_path / "out",
        persistent_dir=tmp_path / "drive",
        manifest_dir=tmp_path / "manifests",
        m0_dir=tmp_path / "m0",
        truth_results=tmp_path / "missing-is-okay-for-dry-run.json",
    )
    assert plan["status"] == "DRY_RUN_ONLY"
    assert plan["gpu_network_or_gated_data_touched"] is False
    assert plan["model"]["quantization"] == "none"
    assert plan["model"]["min_gpu_memory_mib"] == 23000


def test_gemma_truth_v2_bundle_is_model_specific(tmp_path) -> None:
    path = write_truth_v2_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_sha256"] = TRUTH_V2_GEMMA_CONFIG_SHA256
    payload["model_revision"] = TRUTH_V2_GEMMA_MODEL_REVISION
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    environment_path = tmp_path / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["repository"]["commit"] = TRUTH_V2_GEMMA_SCIENTIFIC_COMMIT
    environment["config"]["sha256"] = TRUTH_V2_GEMMA_CONFIG_SHA256
    environment["model"].update(
        {
            "id": TRUTH_V2_GEMMA_MODEL_ID,
            "resolved_revision": TRUTH_V2_GEMMA_MODEL_REVISION,
        }
    )
    environment_path.write_text(
        json.dumps(environment, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = tmp_path / "final_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name] = _digest(path)
    manifest["files"]["environment.json"] = _digest(environment_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = _truth_control_summary(
        path, expected_model_id=TRUTH_V2_GEMMA_MODEL_ID
    )
    assert summary["terminal_disposition"] == "PASS"
    assert summary["model_id"] == TRUTH_V2_GEMMA_MODEL_ID
    assert all(summary["checks"].values())
    with pytest.raises(RuntimeError, match="requires its own truth control"):
        _truth_control_summary(
            path,
            expected_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
