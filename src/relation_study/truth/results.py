from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import EXPECTED, PROMPT_SHA256, RANDOM_SEED


def prompt_examples() -> pd.DataFrame:
    return pd.DataFrame([
        {"illustrative proposition": "Paris is the capital of France.", "semantic label": "true", "answer mapping": "true to B, false to A", "correct output": "B", "verbalizer": "A/B"},
        {"illustrative proposition": "Paris is the capital of France.", "semantic label": "true", "answer mapping": "true to 2, false to 1", "correct output": "2", "verbalizer": "1/2 held out"},
    ])


def v1_diagnostics(v1: dict[str, Any]) -> pd.DataFrame:
    d = v1["diagnostics"]
    return pd.DataFrame([
        ("standard vs reversed semantic direction cosine", d["semantic_direction_cosine_standard_vs_reversed"]),
        ("reversed instruction mapped macro AUROC", d["reversed_instruction_mapped_macro_auroc"]),
        ("reversed literal True minus False macro AUROC", d["reversed_literal_true_minus_false_macro_auroc"]),
        ("cross prompt semantic DIM macro AUROC", d["cross_prompt_semantic_dim_macro_auroc"]),
    ], columns=["quantity", "value"])


def design_counts(split: pd.DataFrame) -> pd.DataFrame:
    primary = split[split["scheme"] == "primary"]
    return pd.DataFrame([
        ("semantic proposition rows", len(primary)),
        ("affirmative city rows", int((primary["dataset"] == "cities").sum())),
        ("negated city rows", int((primary["dataset"] == "neg_cities").sum())),
        ("proposition groups", int(primary["group_id"].nunique())),
        ("training rows per verbalizer", int((primary["split"] == "train").sum())),
        ("development rows per verbalizer", int((primary["split"] == "dev").sum())),
        ("test rows per verbalizer", int((primary["split"] == "test").sum())),
        ("all A/B and 1/2 cache records", len(split)),
    ], columns=["unit", "count"])


def mapping_checks(split: pd.DataFrame, v2: dict[str, Any]) -> pd.DataFrame:
    mapping_counts = {
        ":".join(map(str, key)): int(value)
        for key, value in split.groupby(["split", "scheme", "mapping"]).size().items()
    }
    checks = {
        "binary semantic labels": set(split["label"].unique()) == {0, 1},
        "standard and reversed mappings only": set(split["mapping"].unique()) == {"standard", "reversed"},
        "both mappings in every split and verbalizer": split.groupby(["split", "scheme"])["mapping"].nunique().eq(2).all(),
        "groups stay inside one split": split.groupby("group_id")["split"].nunique().max() == 1,
        "affirmative and negated labels are complementary": split.groupby(["source_row", "scheme"])["label"].sum().eq(1).all(),
        "prompt contract hash": v2["prompt_contract_sha256"] == PROMPT_SHA256,
        "mapping counts": mapping_counts == v2["mapping_counts"],
    }
    return pd.DataFrame([{"assertion": key, "pass": bool(value)} for key, value in checks.items()])


def direction_method() -> pd.DataFrame:
    return pd.DataFrame([
        ("fit", "unit mean true minus mean false", "A/B training rows inside partition k"),
        ("center", "midpoint of projected class means", "the same training partition"),
        ("scale", "population SD of centered projections", "the same training partition"),
        ("score", "centered projection divided by training SD", "held out development or test rows"),
        ("partition effect", "mean true z minus mean false z", "macro average across datasets"),
        ("signed statistic", "mean partition effect", "eight train oriented directions"),
        ("consensus", "norm of the mean unit direction", "eight unit directions"),
    ], columns=["step", "definition", "data used"])


def layer_sweep(v2: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "layer": int(row["layer"]),
            "standard development T": row["primary_dev_by_mapping"]["standard"]["T"],
            "reversed development T": row["primary_dev_by_mapping"]["reversed"]["T"],
            "selection score": row["selection_score"],
        }
        for row in v2["development_layer_sweep"]
    ])


def partition_effects(v2: dict[str, Any]) -> pd.DataFrame:
    selected = v2["confirmatory_layer_results"][str(v2["selected_layer"])]["test"]
    rows = []
    for scheme in ("primary", "transfer"):
        for mapping in ("overall", "standard", "reversed"):
            cell = selected[scheme][mapping]
            for index, value in enumerate(cell["partition_effects"]):
                rows.append({"verbalizer": "A/B" if scheme == "primary" else "1/2", "mapping": mapping, "partition": index, "signed effect": value, "cell T": cell["T"]})
    return pd.DataFrame(rows)


def consensus(v2: dict[str, Any]) -> pd.DataFrame:
    k = int(v2["training_partitions"])
    c = float(v2["directional_consensus_C"])
    mean_pairwise = (k * c * c - 1) / (k - 1)
    return pd.DataFrame([
        ("training partition directions", k, "Each direction has unit norm"),
        ("implied mean pairwise cosine", mean_pairwise, "Derived from C for unit vectors"),
        ("directional consensus C", c, "Norm of the mean direction"),
    ], columns=["component", "value", "meaning"])


def bootstrap_intervals(v2: dict[str, Any]) -> pd.DataFrame:
    selected = v2["confirmatory_layer_results"][str(v2["selected_layer"])]
    rows = []
    for scheme, symbols in (("primary", "A/B"), ("transfer", "1/2")):
        interval = selected["group_bootstrap_ci"][scheme]
        point = selected["test"][scheme]["overall"]["T"]
        rows.append((
            symbols,
            point,
            interval["low"],
            interval["high"],
            interval["replicates"],
            interval["confidence_level"],
        ))
    return pd.DataFrame(rows, columns=[
        "answer symbols",
        "T",
        "CI low",
        "CI high",
        "bootstrap replicates",
        "confidence level",
    ])


def permutation_summary(v2: dict[str, Any]) -> pd.DataFrame:
    null = v2["permutation_null"]
    return pd.DataFrame([
        ("permutations", null["permutations"]),
        ("seed", RANDOM_SEED + 4000),
        ("unit", "proposition group sign flip"),
        ("observed T", null["observed_T"]),
        ("null values at least observed", null["count_greater_equal_observed"]),
        ("add one p", null["p_greater_equal"]),
    ], columns=["field", "value"])


def transfer(v2: dict[str, Any]) -> pd.DataFrame:
    test = v2["confirmatory_layer_results"][str(v2["selected_layer"])]["test"]
    rows = []
    for scheme, symbols in (("primary", "A/B"), ("transfer", "1/2")):
        for mapping in ("standard", "reversed", "overall"):
            cell = test[scheme][mapping]
            rows.append({"symbols": symbols, "mapping": mapping, "signed T": cell["T"], "orientation positive": cell["T"] > 0, "macro AUROC": cell["ensemble_macro_dataset_auroc"]})
    return pd.DataFrame(rows)


def validation(v2: dict[str, Any]) -> pd.DataFrame:
    selected = v2["confirmatory_layer_results"][str(v2["selected_layer"])]["test"]
    actual = {
        "selected layer": v2["selected_layer"],
        "eight partition signed T": v2["primary_test_T"],
        "directional consensus C": v2["directional_consensus_C"],
        "Monte Carlo permutation p": v2["permutation_null"]["p_greater_equal"],
        "held out 1/2 T": selected["transfer"]["overall"]["T"],
    }
    expected = {
        "selected layer": EXPECTED["selected_layer"],
        "eight partition signed T": EXPECTED["primary_T"],
        "directional consensus C": EXPECTED["consensus_C"],
        "Monte Carlo permutation p": EXPECTED["permutation_p"],
        "held out 1/2 T": EXPECTED["transfer_T"],
    }
    rows = [{"quantity": key, "expected": expected[key], "observed": actual[key], "pass": actual[key] == expected[key]} for key in expected]
    rows.append({"quantity": "terminal disposition", "expected": "TRUTH_CONTROL_V2_PASS", "observed": v2["terminal_disposition"], "pass": v2["terminal_disposition"] == "TRUTH_CONTROL_V2_PASS"})
    table = pd.DataFrame(rows)
    if not table["pass"].all():
        raise RuntimeError("Truth validation failed")
    return table
