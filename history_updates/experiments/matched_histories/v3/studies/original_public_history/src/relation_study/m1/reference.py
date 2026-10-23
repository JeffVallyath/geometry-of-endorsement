from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_reference(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(root).resolve() if root else repository_root()
    path = base / "artifacts" / "m1" / "development_reference.json"
    manifest = json.loads(
        (path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["public_files"][path.name]
    if observed != expected:
        raise RuntimeError("The retained M1 development reference failed its hash check.")
    return json.loads(path.read_text(encoding="utf-8"))


def _value(result: dict[str, Any], dotted_path: str) -> Any:
    current: Any = result
    for part in dotted_path.split("."):
        current = current[part]
    return current


def compare_result(result: dict[str, Any], reference: dict[str, Any] | None = None) -> pd.DataFrame:
    frozen = reference or load_reference()
    tolerance = float(frozen["comparison"]["reported_value_absolute_tolerance"])
    exact_checks = [
        ("terminal disposition", "terminal_disposition", frozen["terminal_disposition"]),
        ("selected layer", "selection.selected_layer", frozen["selected_layer"]),
        ("model revision", "runtime.model_revision", frozen["model"]["revision"]),
        ("tokenizer revision", "runtime.tokenizer_revision", frozen["model"]["tokenizer_revision"]),
        ("pilot train rows", "sample_counts.pilot_train", frozen["data"]["pilot_train_rows"]),
        ("pilot select rows", "sample_counts.pilot_select", frozen["data"]["pilot_select_rows"]),
        ("pilot eval rows", "sample_counts.pilot_eval", frozen["data"]["pilot_eval_rows"]),
        ("pilot eval boards", "sample_counts.pilot_eval_boards", frozen["data"]["pilot_eval_boards"]),
        ("pilot train manifest", "manifest.split_manifest_sha256.pilot_train", frozen["manifest_hashes"]["pilot_train"]),
        ("pilot select manifest", "manifest.split_manifest_sha256.pilot_select", frozen["manifest_hashes"]["pilot_select"]),
        ("pilot eval manifest", "manifest.split_manifest_sha256.pilot_eval", frozen["manifest_hashes"]["pilot_eval"]),
        ("pilot select board manifest", "manifest.board_manifest_sha256.pilot_select", frozen["manifest_hashes"]["pilot_select_boards"]),
        ("pilot eval board manifest", "manifest.board_manifest_sha256.pilot_eval", frozen["manifest_hashes"]["pilot_eval_boards"]),
        ("prompt contract", "prompt_contract_sha256", frozen["prompt_contract_sha256"]),
    ]
    metric_checks = [
        ("DIM relation AUROC", "evaluations.difference_in_means.auroc", "difference_in_means_relation_auroc"),
        ("DIM I_b", "evaluations.difference_in_means.checkerboard_interaction_mean", "difference_in_means_I_b"),
        ("logistic relation AUROC", "evaluations.logistic.auroc", "logistic_relation_auroc"),
        ("logistic I_b", "evaluations.logistic.checkerboard_interaction_mean", "logistic_I_b"),
        ("SBERT I_b", "evaluations.sbert_interaction.checkerboard_interaction_mean", "sbert_interaction_I_b"),
        ("native answer margin AUROC", "evaluations.native_answer_margin.auroc", "native_answer_margin_auroc"),
        ("situation-only I_b", "evaluations.situation_only_activation.checkerboard_interaction_mean", "situation_only_I_b"),
        ("consideration-only I_b", "evaluations.consideration_only_activation.checkerboard_interaction_mean", "consideration_only_I_b"),
        ("separate-encoding additive I_b", "evaluations.separate_encoding_additive.checkerboard_interaction_mean", "separate_encoding_additive_I_b"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, expected in exact_checks:
        observed = _value(result, path)
        rows.append(
            {
                "quantity": name,
                "reproduced": observed,
                "reference": expected,
                "tolerance": "exact",
                "pass": observed == expected,
            }
        )
    for name, path, reference_key in metric_checks:
        observed = float(_value(result, path))
        expected = float(frozen["metrics"][reference_key])
        rows.append(
            {
                "quantity": name,
                "reproduced": observed,
                "reference": expected,
                "tolerance": tolerance,
                "pass": abs(observed - expected) <= tolerance,
            }
        )
    return pd.DataFrame(rows)


def require_reference_agreement(result: dict[str, Any]) -> pd.DataFrame:
    comparison = compare_result(result)
    if not bool(comparison["pass"].all()):
        failed = comparison.loc[~comparison["pass"], "quantity"].tolist()
        raise RuntimeError(f"M1 reproduction differs from the retained reference for {failed}")
    return comparison
