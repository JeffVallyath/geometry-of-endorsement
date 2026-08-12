from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from geometry_of_truth.m1.config import ConfigError, load_config
from geometry_of_truth.m1.reference import compare_result, load_reference
from geometry_of_truth.m1.runner import _truth_control_summary


ROOT = Path(__file__).resolve().parents[1]


def _reference_shaped_result() -> dict:
    reference = load_reference(ROOT)
    metrics = reference["metrics"]
    return {
        "terminal_disposition": reference["terminal_disposition"],
        "selection": {"selected_layer": reference["selected_layer"]},
        "runtime": {
            "model_revision": reference["model"]["revision"],
            "tokenizer_revision": reference["model"]["tokenizer_revision"],
        },
        "sample_counts": {
            "pilot_train": reference["data"]["pilot_train_rows"],
            "pilot_select": reference["data"]["pilot_select_rows"],
            "pilot_eval": reference["data"]["pilot_eval_rows"],
            "pilot_eval_boards": reference["data"]["pilot_eval_boards"],
        },
        "manifest": {
            "split_manifest_sha256": {
                "pilot_train": reference["manifest_hashes"]["pilot_train"],
                "pilot_select": reference["manifest_hashes"]["pilot_select"],
                "pilot_eval": reference["manifest_hashes"]["pilot_eval"],
            },
            "board_manifest_sha256": {
                "pilot_select": reference["manifest_hashes"]["pilot_select_boards"],
                "pilot_eval": reference["manifest_hashes"]["pilot_eval_boards"],
            },
        },
        "prompt_contract_sha256": reference["prompt_contract_sha256"],
        "evaluations": {
            "difference_in_means": {
                "auroc": metrics["difference_in_means_relation_auroc"],
                "checkerboard_interaction_mean": metrics["difference_in_means_I_b"],
            },
            "logistic": {
                "auroc": metrics["logistic_relation_auroc"],
                "checkerboard_interaction_mean": metrics["logistic_I_b"],
            },
            "sbert_interaction": {
                "checkerboard_interaction_mean": metrics["sbert_interaction_I_b"]
            },
            "native_answer_margin": {
                "auroc": metrics["native_answer_margin_auroc"]
            },
            "situation_only_activation": {
                "checkerboard_interaction_mean": metrics["situation_only_I_b"]
            },
            "consideration_only_activation": {
                "checkerboard_interaction_mean": metrics["consideration_only_I_b"]
            },
            "separate_encoding_additive": {
                "checkerboard_interaction_mean": metrics["separate_encoding_additive_I_b"]
            },
        },
    }


def test_public_truth_bundle_passes_the_m1_gate() -> None:
    summary = _truth_control_summary(
        ROOT / "artifacts" / "truth" / "v2_results.json",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    assert summary["terminal_disposition"] == "PASS"
    assert summary["provenance_mode"] == "public_aggregate_bundle"
    assert all(summary["checks"].values())


def test_public_truth_bundle_fails_when_the_result_changes(tmp_path: Path) -> None:
    source = ROOT / "artifacts" / "truth"
    payload = json.loads((source / "v2_results.json").read_text(encoding="utf-8"))
    payload["selected_layer"] = int(payload["selected_layer"]) + 1
    (tmp_path / "v2_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    for name in ("environment.json", "manifest.json", "split_manifest.csv"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    summary = _truth_control_summary(
        tmp_path / "v2_results.json",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    assert summary["terminal_disposition"] == "FAIL"


def test_m1_config_pins_the_retained_model_revision(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "m1_development.yaml")
    assert config.raw["model"]["revision"] == (
        "0e9e39f249a16976918f6564b8830bc894c89659"
    )
    changed = copy.deepcopy(config.raw)
    changed["model"]["revision"] = "main"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ConfigError, match="immutable revision"):
        load_config(path)


def test_reference_comparison_accepts_the_retained_values() -> None:
    comparison = compare_result(_reference_shaped_result())
    assert len(comparison) == 23
    assert bool(comparison["pass"].all())


def test_reference_comparison_exposes_metric_drift() -> None:
    result = _reference_shaped_result()
    result["evaluations"]["difference_in_means"]["checkerboard_interaction_mean"] += 0.01
    comparison = compare_result(result)
    failed = comparison.loc[~comparison["pass"], "quantity"].tolist()
    assert failed == ["DIM I_b"]
