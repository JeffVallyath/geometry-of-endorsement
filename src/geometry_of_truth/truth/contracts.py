from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import pandas as pd

from geometry_of_truth.common.artifacts import load_json, repository_root, verify_files


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
RANDOM_SEED = 314159
CONFIG_SHA256 = "61e9c98297fe85af4eec1ea064122155740cf8420bd818180371388322d95ee2"
PROMPT_SHA256 = "5b271d42d849f9c17cd939afbbb5b2c7b0281c4fd97a7286a3045bf0566b3b58"
CACHE_SIGNATURE = "c019ed735ef51cf028099e4668db9790f38a93bfc6aef3abbaeb85a7bd14dc28"
CACHE_MANIFEST_SHA256 = "ebe75e27e136872bb5933fbcff31acbe4db039af877ff780a2878c4b45a43b7d"
EXPECTED = {
    "semantic_rows": 2992,
    "groups": 748,
    "selected_layer": 14,
    "primary_T": 1.9244548082351685,
    "consensus_C": 0.9986950159072876,
    "permutation_p": 1 / 1001,
    "transfer_T": 1.8401652574539185,
}


def load_bundle(root: str | Path | None = None) -> dict[str, Any]:
    repo = repository_root(root)
    artifact_root = repo / "artifacts" / "truth"
    manifest = load_json(artifact_root / "manifest.json")
    checks = verify_files(artifact_root, manifest["public_files"])
    split = pd.read_csv(artifact_root / "split_manifest.csv")
    bundle = {
        "root": repo,
        "artifacts": artifact_root,
        "manifest": manifest,
        "checks": pd.DataFrame(checks),
        "v1": load_json(artifact_root / "v1_diagnostic.json"),
        "v2": load_json(artifact_root / "v2_results.json"),
        "environment": load_json(artifact_root / "environment.json"),
        "split": split,
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    split = bundle["split"]
    primary = split[split["scheme"] == "primary"]
    v2 = bundle["v2"]
    transfer = v2["confirmatory_layer_results"][str(v2["selected_layer"])]["test"]["transfer"]["overall"]["T"]
    observed = {
        "semantic_rows": len(primary),
        "groups": int(primary["group_id"].nunique()),
        "selected_layer": int(v2["selected_layer"]),
        "primary_T": float(v2["primary_test_T"]),
        "consensus_C": float(v2["directional_consensus_C"]),
        "permutation_p": float(v2["permutation_null"]["p_greater_equal"]),
        "transfer_T": float(transfer),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"Frozen Truth contract mismatch  {observed}")
    if bundle["v1"]["frozen_v1_disposition"] != "TRUTH_CONTROL_V1_STRICT_FAIL":
        raise RuntimeError("Truth v1 disposition drifted")
    if v2["config_sha256"] != CONFIG_SHA256 or v2["prompt_contract_sha256"] != PROMPT_SHA256:
        raise RuntimeError("Truth protocol hash drifted")
    if v2["cache_signature"] != CACHE_SIGNATURE:
        raise RuntimeError("Truth activation cache signature drifted")


def provenance(bundle: dict[str, Any]) -> pd.DataFrame:
    environment = bundle["environment"]
    runtime = environment.get("runtime", {})
    return pd.DataFrame([
        ("Model", MODEL_ID),
        ("Resolved model revision", MODEL_REVISION),
        ("Random seed", RANDOM_SEED),
        ("Python", platform.python_version()),
        ("Retained Python", runtime.get("python")),
        ("Retained CUDA", runtime.get("gpu")),
    ], columns=["item", "value"])


def retained_environment(bundle: dict[str, Any]) -> pd.DataFrame:
    environment = bundle["environment"]
    runtime = environment["runtime"]
    rows = [
        ("Python", runtime["python"]),
        ("Platform", runtime["platform"]),
        ("GPU", runtime["gpu"]["name"]),
        ("GPU memory MiB", runtime["gpu"]["memory_total_mib"]),
    ]
    rows.extend((name, version) for name, version in sorted(runtime["packages"].items()))
    return pd.DataFrame(rows, columns=["component", "retained run"])


def number_lineage(bundle: dict[str, Any]) -> pd.DataFrame:
    split = bundle["split"]
    primary = split[split["scheme"] == "primary"]
    v2 = bundle["v2"]
    selected_result = v2["confirmatory_layer_results"][str(v2["selected_layer"])]
    selected = selected_result["test"]
    rows = [
        ("Semantic proposition rows", len(primary), "split_manifest.csv", "count where scheme equals primary"),
        ("Proposition groups", int(primary["group_id"].nunique()), "split_manifest.csv", "distinct group_id where scheme equals primary"),
        ("Selected layer", v2["selected_layer"], "v2_results.json", "selected_layer"),
        ("Primary signed T", v2["primary_test_T"], "v2_results.json", "primary_test_T"),
        ("Primary group-bootstrap CI low", selected_result["group_bootstrap_ci"]["primary"]["low"], "v2_results.json", "confirmatory_layer_results.14.group_bootstrap_ci.primary.low"),
        ("Primary group-bootstrap CI high", selected_result["group_bootstrap_ci"]["primary"]["high"], "v2_results.json", "confirmatory_layer_results.14.group_bootstrap_ci.primary.high"),
        ("Directional consensus C", v2["directional_consensus_C"], "v2_results.json", "directional_consensus_C"),
        ("Monte Carlo permutation p", v2["permutation_null"]["p_greater_equal"], "v2_results.json", "permutation_null.p_greater_equal"),
        ("Held out 1/2 signed T", selected["transfer"]["overall"]["T"], "v2_results.json", "confirmatory_layer_results.14.test.transfer.overall.T"),
        ("Held out 1/2 group-bootstrap CI low", selected_result["group_bootstrap_ci"]["transfer"]["low"], "v2_results.json", "confirmatory_layer_results.14.group_bootstrap_ci.transfer.low"),
        ("Held out 1/2 group-bootstrap CI high", selected_result["group_bootstrap_ci"]["transfer"]["high"], "v2_results.json", "confirmatory_layer_results.14.group_bootstrap_ci.transfer.high"),
        ("Truth v1 disposition", bundle["v1"]["frozen_v1_disposition"], "v1_diagnostic.json", "frozen_v1_disposition"),
    ]
    return pd.DataFrame(rows, columns=["reported quantity", "value", "source artifact", "field or calculation"])
