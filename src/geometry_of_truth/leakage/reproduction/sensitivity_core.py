"""Run the deterministic leakage sensitivity audit and write its artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .build_confirmatory_split import manifest_hash
from .split_stress_test import CON, OPPOSES, SIT, SUPPORTS, VAL, l1_normalize, l2_normalize
from .sensitivity_retrieval import (
    ANNOTATION_FIELDS, DEFAULT_CONFIG, OUTPUTS, QUEUE, ROOT, config_load,
    load_frames, retrieve, sha_file, verify_inputs,
)


def checkerboards(df: pd.DataFrame):
    conf = pd.read_csv(ROOT / "manifest_confirmatory.csv")
    raw_map, l2_map = df.groupby(CON).l3.first().to_dict(), df.groupby("l2").l3.first().to_dict()
    def resolve(value):
        if value in raw_map:
            return int(raw_map[value])
        normalized = l2_normalize(str(value))
        if normalized in l2_map:
            return int(l2_map[normalized])
        raise RuntimeError(f"unresolved checkerboard consideration {value!r}")
    conf["cluster_A"] = conf.consideration_A.map(resolve)
    conf["cluster_B"] = conf.consideration_B.map(resolve)
    membership = defaultdict(set)
    for _, row in conf.iterrows():
        membership[int(row.cluster_A)].add(int(row["rank"]))
        membership[int(row.cluster_B)].add(int(row["rank"]))
    return conf, membership


def unit_counts(frame: pd.DataFrame) -> dict[str, int]:
    sit = frame.groupby([SIT, VAL]).size().unstack(fill_value=0)
    con = frame.groupby(["l3", VAL]).size().unstack(fill_value=0)
    return {
        "rows": len(frame),
        "situations": int(frame[SIT].nunique()),
        "consideration_clusters": int(frame.l3.nunique()),
        "within_situation_comparisons": int((sit.get(SUPPORTS, 0) * sit.get(OPPOSES, 0)).sum()),
        "within_consideration_comparisons": int((con.get(SUPPORTS, 0) * con.get(OPPOSES, 0)).sum()),
        "within_situation_groups": int(((sit.get(SUPPORTS, 0) > 0) & (sit.get(OPPOSES, 0) > 0)).sum()),
        "within_consideration_groups": int(((con.get(SUPPORTS, 0) > 0) & (con.get(OPPOSES, 0) > 0)).sum()),
    }


def affected_situation_pairs(frame: pd.DataFrame, risky: set[int]) -> int:
    return unit_counts(frame)["within_situation_comparisons"] - unit_counts(
        frame[~frame.l3.isin(risky)]
    )["within_situation_comparisons"]


def impacts(test: pd.DataFrame, membership):
    output = {}
    for cluster, group in test.groupby("l3"):
        cluster = int(cluster)
        impacted = 0
        for situation in group[SIT].unique():
            whole = test[test[SIT] == situation]
            part = group[group[SIT] == situation]
            total = int((whole[VAL] == SUPPORTS).sum() * (whole[VAL] == OPPOSES).sum())
            safe_s = int((whole[VAL] == SUPPORTS).sum() - (part[VAL] == SUPPORTS).sum())
            safe_o = int((whole[VAL] == OPPOSES).sum() - (part[VAL] == OPPOSES).sum())
            impacted += total - safe_s * safe_o
        output[cluster] = {
            "test_rows_affected": len(group),
            "test_situations_affected": int(group[SIT].nunique()),
            "within_situation_pairs_affected": impacted,
            "within_consideration_pairs_affected": int(
                (group[VAL] == SUPPORTS).sum() * (group[VAL] == OPPOSES).sum()
            ),
            "checkerboards_affected": len(membership.get(cluster, set())),
        }
    return output


def make_queue(ledger: pd.DataFrame, impact):
    prior = {}
    if QUEUE.is_file():
        old = pd.read_csv(QUEUE, dtype=str, keep_default_na=False)
        if "pair_id" in old:
            prior = {
                str(row.pair_id): {field: str(row.get(field, "")) for field in ANNOTATION_FIELDS}
                for _, row in old.iterrows()
            }
    queue = ledger.copy()
    for field in next(iter(impact.values())):
        queue[field] = queue.test_cluster.map(lambda x: impact[int(x)][field])
    for field in ANNOTATION_FIELDS:
        queue[field] = queue.pair_id.map(lambda x: prior.get(str(x), {}).get(field, ""))
    priority = {"automatic_high_confidence": 0, "ambiguous_high_risk": 1, "safe_outside": 2}
    queue["_priority"] = queue.risk_band.map(priority)
    return queue.sort_values(
        ["_priority", "within_situation_pairs_affected", "test_rows_affected",
         "embedding_cosine", "pair_id"],
        ascending=[True, False, False, False, True],
    ).drop(columns="_priority").reset_index(drop=True)


def audit_state(queue: pd.DataFrame, config: dict[str, Any]):
    labels = set(config["human_audit"]["labels"])
    values = queue.adjudicated_label.astype(str).str.strip().str.lower()
    invalid = set(values) - labels - {""}
    if invalid:
        raise RuntimeError(f"invalid audit labels {invalid}")
    counts = {label: int((values == label).sum()) for label in sorted(labels)}
    per_band = {
        band: int(((queue.risk_band == band) & values.isin(labels)).sum())
        for band in ("automatic_high_confidence", "ambiguous_high_risk", "safe_outside")
    }
    sufficient = sum(counts.values()) >= config["human_audit"]["minimum_adjudicated_pairs_for_calibration"]
    sufficient = sufficient and all(
        count >= config["human_audit"]["minimum_per_risk_band"] for count in per_band.values()
    )
    return {
        "question": config["human_audit"]["question"],
        "adjudicated_counts": counts,
        "adjudicated_total": sum(counts.values()),
        "per_band": per_band,
        "calibration_sufficient": bool(sufficient),
        "annotators": sorted(str(x) for x in queue.annotator_id.unique() if str(x).strip()),
        "blindness_contract": config["human_audit"]["blind_fields_excluded"],
    }


def classify_domain(text: str, config: dict[str, Any]) -> str:
    normalized = " " + l1_normalize(text) + " "
    for category, words in config["domain_proxy"]["ordered_categories"].items():
        if any(" " + word.strip() + " " in normalized for word in words):
            return category
    return config["domain_proxy"]["fallback"]


def coverage(frame: pd.DataFrame, config: dict[str, Any]):
    output = unit_counts(frame)
    valence, vrd = frame[VAL].value_counts(), frame.vrd.value_counts()
    domains = {s: classify_domain(str(s), config) for s in frame[SIT].unique()}
    row_domains = frame[SIT].map(domains).value_counts()
    situation_domains = pd.Series(domains).value_counts()
    output.update({
        "valence_rows": {str(k): int(v) for k, v in sorted(valence.items())},
        "supports_fraction": float(valence.get(SUPPORTS, 0) / len(frame)) if len(frame) else 0.0,
        "vrd_rows": {str(k): int(v) for k, v in sorted(vrd.items())},
        "domain_proxy_method": config["domain_proxy"]["method"],
        "domain_rows": {str(k): int(v) for k, v in sorted(row_domains.items())},
        "domain_situations": {str(k): int(v) for k, v in sorted(situation_domains.items())},
    })
    return output


def exposure(frame: pd.DataFrame, risky: set[int], membership, board_total: int):
    total = unit_counts(frame)
    exposed = frame[frame.l3.isin(risky)]
    board_ids = set().union(*(membership.get(x, set()) for x in risky)) if risky else set()
    con = exposed.groupby(["l3", VAL]).size().unstack(fill_value=0) if len(exposed) else pd.DataFrame()
    affected = {
        "rows": len(exposed),
        "situations": int(exposed[SIT].nunique()),
        "consideration_clusters": int(exposed.l3.nunique()),
        "within_situation_comparisons": affected_situation_pairs(frame, risky) if risky else 0,
        "within_consideration_comparisons": int(
            (con.get(SUPPORTS, 0) * con.get(OPPOSES, 0)).sum()
        ) if len(exposed) else 0,
        "reciprocal_checkerboards": len(board_ids),
    }
    denominators = {
        key: total[key] for key in (
            "rows", "situations", "consideration_clusters",
            "within_situation_comparisons", "within_consideration_comparisons"
        )
    }
    denominators["reciprocal_checkerboards"] = board_total
    return {
        "affected": affected,
        "denominators": denominators,
        "fractions": {
            key: affected[key] / denominators[key] if denominators[key] else 0.0
            for key in denominators
        },
    }


def save_plot(cov, exp, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["U0", "U1", "U2", "U3"]
    x = [cov[label]["within_situation_comparisons"] for label in labels]
    rows = [exp[label]["fractions"]["rows"] for label in labels]
    pairs = [exp[label]["fractions"]["within_situation_comparisons"] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, rows, marker="o", label="row exposure")
    ax.plot(x, pairs, marker="s", label="within-situation comparison exposure")
    for label, xi, yi in zip(labels, x, pairs):
        ax.annotate(label, (xi, yi))
    ax.set_xlabel("Retained within-situation comparisons")
    ax.set_ylabel("Estimated possible residual exposure fraction")
    ax.set_title("Leakage sensitivity coverage tradeoff")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": "sensitivity_analysis.py"})
    plt.close(fig)


def inventory(paths):
    output = {}
    for path in paths:
        metadata = {"sha256": sha_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            sample = pd.read_csv(path, nrows=1)
            if "row_id" in sample:
                ids = pd.read_csv(path, dtype={"row_id": str}).row_id
                metadata.update({"rows": len(ids), "row_hash": manifest_hash(ids)})
        output[path.relative_to(ROOT).as_posix()] = metadata
    return output
