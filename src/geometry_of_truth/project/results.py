from __future__ import annotations

from typing import Any

import pandas as pd


def stage_table(status: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(status["stages"])[["stage", "question", "status", "evidence"]]


def development_design(status: dict[str, Any]) -> pd.DataFrame:
    result = status["moral_relation_development"]
    return pd.DataFrame([
        ("Model", result["model"]),
        ("Resolved model revision", result["model_revision"]),
        ("Selected layer", result["selected_layer"]),
        ("Training rows", result["training_rows"]),
        ("Selection rows", result["selection_rows"]),
        ("Selection boards", result["selection_boards"]),
        ("Evaluation rows", result["evaluation_rows"]),
        ("Evaluation boards", result["evaluation_boards"]),
        ("Primary endpoint", result["primary_endpoint"]),
    ], columns=["item", "value"])


def development_results(status: dict[str, Any]) -> pd.DataFrame:
    result = status["moral_relation_development"]
    dim = result["difference_in_means"]
    logistic = result["logistic"]
    controls = result["controls"]
    return pd.DataFrame([
        ("Difference in means", "I_b", dim["I_b"], "Primary activation direction"),
        ("Logistic activation probe", "I_b", logistic["I_b"], "Learned linear activation readout"),
        ("SBERT interaction", "I_b", controls["sbert_interaction_I_b"], "Frozen matched-text baseline"),
        ("Situation only", "I_b", controls["situation_only_I_b"], "No joint state"),
        ("Consideration only", "I_b", controls["consideration_only_I_b"], "No joint state"),
        ("Separate encoding additive", "I_b", controls["separate_encoding_additive_I_b"], "Additive control"),
        ("Native answer margin", "AUROC", controls["native_answer_margin_auroc"], "Model output baseline"),
        ("Difference in means", "AUROC", dim["relation_auroc"], "Relation decoding"),
        ("Logistic activation probe", "AUROC", logistic["relation_auroc"], "Relation decoding"),
    ], columns=["method", "metric", "value", "role"])


def development_intervals(status: dict[str, Any]) -> pd.DataFrame:
    result = status["moral_relation_development"]
    rows = []
    for key, label in (
        ("difference_in_means", "Difference in means"),
        ("logistic", "Logistic activation probe"),
    ):
        cell = result[key]
        rows.append((
            label,
            cell["I_b"],
            cell["permutation_p"],
            cell["delta_over_sbert_ci_low"],
            cell["delta_over_sbert_ci_high"],
        ))
    return pd.DataFrame(rows, columns=[
        "method",
        "I_b",
        "permutation p",
        "Delta I_b over SBERT CI low",
        "Delta I_b over SBERT CI high",
    ])


def source_lineage(status: dict[str, Any]) -> pd.DataFrame:
    result = status["moral_relation_development"]
    return pd.DataFrame([
        ("Development result JSON", result["source_result_sha256"]),
        ("Independent audit archive", result["source_audit_archive_sha256"]),
    ], columns=["source", "SHA-256"])
