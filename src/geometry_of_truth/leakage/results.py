from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def audit(results: dict[str, Any]) -> pd.DataFrame:
    a = results["audit"]
    return pd.DataFrame([
        ("Eligible binary Supports or Opposes rows", a["binary_rows"]),
        ("Situations containing both valences", a["both_valence_situations"]),
        ("Exact considerations with both valences", a["both_valence_exact_considerations"]),
        ("Candidate checkerboards", a["candidate_checkerboards"]),
        ("Distinct consideration pairs", a["consideration_pairs"]),
    ], columns=["quantity", "count"])


def protections(results: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        ("Unseen identity strict split", "recognized consideration identities and held out situations", "cluster and situation double holdout", "unseen identity generalization"),
        ("Reciprocal checkerboard interaction", "fixed consideration preference", results["controls"]["primary_checkerboard_endpoint"], "relation sensitivity after additive priors cancel"),
    ], columns=["control", "shortcut blocked", "mechanism", "claim protected"])


def normalization(results: dict[str, Any]) -> pd.DataFrame:
    counts = results["normalization"]["counts"]
    rules = {
        "L0": "raw exact forms",
        "L1": "case, punctuation, spacing, article, and light plural normalization",
        "L2": "standard Value, Right, and Duty prefix removal",
        "L3": "character 3 to 5 gram leader clustering at threshold 0.85 and cap 25",
    }
    rows = []
    prior = None
    for stage in ("L0", "L1", "L2", "L3"):
        count = counts[stage]
        rows.append((stage, rules[stage], count, 0 if prior is None else prior - count))
        prior = count
    return pd.DataFrame(rows, columns=["stage", "rule", "count", "collapsed from prior"])


def split_lineage(results: dict[str, Any]) -> pd.DataFrame:
    a = results["audit"]
    s = results["strict_split"]
    mixed = s["dataset_binary_rows"] - s["train_rows"] - s["test_rows"]
    return pd.DataFrame([
        ("Shared source", a["binary_rows"], "exclude Either", a["binary_rows"]),
        ("Audit viability", a["binary_rows"], "keep L1 identities with both valences", a["viability_candidate_rows"]),
        ("Audit viability", a["viability_candidate_rows"], "keep situations with both labels", a["viability_rows"]),
        ("Frozen manifest", a["binary_rows"], f"drop {s['duplicate_rows_dropped']} identical rows", s["dataset_binary_rows"]),
        ("Frozen manifest", s["dataset_binary_rows"], "25 percent L3 cluster and 30 percent situation double holdout", f"{s['train_rows']} train, {s['test_rows']} test, {mixed} mixed boundary"),
    ], columns=["lane", "input", "rule", "output"])


def overlap_checks(results: dict[str, Any]) -> pd.DataFrame:
    overlaps = results["strict_split"]["overlaps"]
    labels = (
        ("Row identity", "row_id", "Prohibited in U0"),
        ("Exact consideration", "exact_consideration", "Prohibited in U0"),
        ("L1 normalized consideration", "l1_normalized", "Removed in U1"),
        ("L2 prefix stripped consideration", "l2_prefix_stripped", "Prohibited in U0"),
        ("L3 algorithmic identity cluster", "l3_semantic_cluster", "Prohibited in U0"),
        ("Situation", "situation", "Prohibited in U0"),
    )
    return pd.DataFrame([
        (label, overlaps[key], disposition)
        for label, key, disposition in labels
    ], columns=["boundary", "cross split overlaps", "disposition"])


def stress_draws(results: dict[str, Any]) -> pd.DataFrame:
    s = results["stress_test"]
    rows = []
    for seed, strict, dc, ds in zip(s["seeds"], s["strict_scores"], s["consideration_deltas"], s["situation_deltas"]):
        rows.append((seed, strict, strict + dc, 100 * dc, strict + ds, 100 * ds))
    return pd.DataFrame(rows, columns=["seed", "strict score", "consideration overlap score", "consideration difference pp", "situation overlap score", "situation difference pp"])


def uncertainty(results: dict[str, Any]) -> pd.DataFrame:
    draws = stress_draws(results)
    rows = []
    for label, column in (("Restore consideration identities", "consideration difference pp"), ("Restore situations", "situation difference pp")):
        values = draws[column].to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        se = sd / np.sqrt(n)
        critical = float(stats.t.ppf(0.975, n - 1))
        rows.append((label, n, mean, sd, se, mean - critical * se, mean + critical * se))
    return pd.DataFrame(rows, columns=["intervention", "draws", "mean pp", "sample SD pp", "standard error pp", "95 percent CI low pp", "95 percent CI high pp"])


def sensitivity(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for level in ("U0", "U1", "U2", "U3"):
        c = results["sensitivity"]["coverage"][level]
        rows.append((level, c["rows"], c["situations"], c["within_situation_comparisons"], c["within_consideration_comparisons"]))
    return pd.DataFrame(rows, columns=["set", "rows", "situations", "within situation comparisons", "within consideration comparisons"])


def candidate_supply(results: dict[str, Any]) -> pd.DataFrame:
    c = results["candidate_audit"]
    return pd.DataFrame([
        ("Ranked candidate checkerboards", c["ranked_candidates"]),
        ("Human confirmed target", c["target_confirmed"]),
        ("Required acceptance fraction", c["required_acceptance_fraction"]),
    ], columns=["quantity", "value"])
