from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_of_truth.common.artifacts import load_json, repository_root, verify_files


DATASET_REVISION = "d439ca90825e5b4e5ef97798d9b5950e16ba7065"
EXPECTED = {
    "binary_rows": 183023,
    "reversing_considerations": 3437,
    "boards": 13923,
    "pairs": 6073,
    "viability_rows": 110656,
    "viability_situations": 19068,
    "consideration_mean_pp": 7.294793288,
    "situation_mean_pp": 0.526936606,
    "l1_overlap": 2,
    "l3_overlap": 0,
    "u1_rows": 7081,
    "u1_comparisons": 1865,
    "candidates": 1090,
    "required_fraction": 800 / 1090,
}


def load_bundle(root: str | Path | None = None) -> dict[str, Any]:
    repo = repository_root(root)
    artifact_root = repo / "artifacts" / "leakage"
    manifest = load_json(artifact_root / "manifest.json")
    checks = verify_files(artifact_root, manifest["public_files"])
    results = load_json(artifact_root / "results.json")
    bundle = {"root": repo, "artifacts": artifact_root, "manifest": manifest, "checks": pd.DataFrame(checks), "results": results}
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    r = bundle["results"]
    observed = {
        "binary_rows": r["audit"]["binary_rows"],
        "reversing_considerations": r["audit"]["both_valence_exact_considerations"],
        "boards": r["audit"]["candidate_checkerboards"],
        "pairs": r["audit"]["consideration_pairs"],
        "viability_rows": r["audit"]["viability_rows"],
        "viability_situations": r["audit"]["viability_situations"],
        "consideration_mean_pp": r["stress_test"]["consideration_mean_pp"],
        "situation_mean_pp": r["stress_test"]["situation_mean_pp"],
        "l1_overlap": r["strict_split"]["overlaps"]["l1_normalized"],
        "l3_overlap": r["strict_split"]["overlaps"]["l3_semantic_cluster"],
        "u1_rows": r["sensitivity"]["coverage"]["U1"]["rows"],
        "u1_comparisons": r["sensitivity"]["coverage"]["U1"]["within_situation_comparisons"],
        "candidates": r["candidate_audit"]["ranked_candidates"],
        "required_fraction": r["candidate_audit"]["required_acceptance_fraction"],
    }
    exact_keys = set(observed) - {"consideration_mean_pp", "situation_mean_pp", "required_fraction"}
    exact_pass = all(observed[key] == EXPECTED[key] for key in exact_keys)
    float_pass = all(
        np.isclose(observed[key], EXPECTED[key], rtol=0, atol=1e-12)
        for key in {"consideration_mean_pp", "situation_mean_pp", "required_fraction"}
    )
    if not exact_pass or not float_pass:
        raise RuntimeError(f"Frozen leakage contract mismatch  {observed}")
    if r["dataset"]["row_level_text_included"] is not False:
        raise RuntimeError("Public leakage artifact contains row-level text")


def provenance(bundle: dict[str, Any]) -> pd.DataFrame:
    r = bundle["results"]
    return pd.DataFrame([
        ("Dataset", r["dataset"]["name"]),
        ("Configuration", r["dataset"]["config"]),
        ("Pinned revision", r["dataset"]["revision"]),
        ("License", r["dataset"]["license"]),
        ("Row-level text included", r["dataset"]["row_level_text_included"]),
    ], columns=["item", "value"])


def number_lineage(bundle: dict[str, Any]) -> pd.DataFrame:
    r = bundle["results"]
    rows = [
        ("Reversing exact consideration strings", r["audit"]["both_valence_exact_considerations"], "results.json", "audit.both_valence_exact_considerations"),
        ("Candidate checkerboards", r["audit"]["candidate_checkerboards"], "results.json", "audit.candidate_checkerboards"),
        ("Distinct consideration pairs", r["audit"]["consideration_pairs"], "results.json", "audit.consideration_pairs"),
        ("L0 distinct forms", r["normalization"]["counts"]["L0"], "results.json", "normalization.counts.L0"),
        ("L3 algorithmic clusters", r["normalization"]["counts"]["L3"], "results.json", "normalization.counts.L3"),
        ("Viability rows", r["audit"]["viability_rows"], "results.json", "audit.viability_rows"),
        ("Viability situations", r["audit"]["viability_situations"], "results.json", "audit.viability_situations"),
        ("L1 normalized cross split overlaps", r["strict_split"]["overlaps"]["l1_normalized"], "results.json", "strict_split.overlaps.l1_normalized"),
        ("L3 cluster cross split overlaps", r["strict_split"]["overlaps"]["l3_semantic_cluster"], "results.json", "strict_split.overlaps.l3_semantic_cluster"),
        ("Consideration restoration mean in percentage points", r["stress_test"]["consideration_mean_pp"], "results.json", "stress_test.consideration_mean_pp"),
        ("Situation restoration mean in percentage points", r["stress_test"]["situation_mean_pp"], "results.json", "stress_test.situation_mean_pp"),
        ("U1 rows", r["sensitivity"]["coverage"]["U1"]["rows"], "results.json", "sensitivity.coverage.U1.rows"),
        ("U1 within-situation comparisons", r["sensitivity"]["coverage"]["U1"]["within_situation_comparisons"], "results.json", "sensitivity.coverage.U1.within_situation_comparisons"),
        ("U3 rows", r["sensitivity"]["coverage"]["U3"]["rows"], "results.json", "sensitivity.coverage.U3.rows"),
        ("U3 terminal disposition", r["sensitivity"]["terminal_disposition"], "results.json", "sensitivity.terminal_disposition"),
        ("Ranked checkerboard candidates", r["candidate_audit"]["ranked_candidates"], "results.json", "candidate_audit.ranked_candidates"),
        ("Required acceptance fraction", r["candidate_audit"]["required_acceptance_fraction"], "results.json", "candidate_audit.required_acceptance_fraction"),
    ]
    return pd.DataFrame(rows, columns=["reported quantity", "value", "source artifact", "field or calculation"])
