"""Top-level orchestration for the leakage sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import pandas as pd

from .sensitivity_retrieval import DEFAULT_CONFIG, OUTPUTS, QUEUE, ROOT, config_load, load_frames, retrieve, verify_inputs
from .sensitivity_core import audit_state, checkerboards, coverage, exposure, impacts, inventory, make_queue, save_plot


def disposition(human, cov, config):
    threshold = config["terminal_thresholds"]
    retention = cov["U2"]["within_situation_comparisons"] / cov["U0"]["within_situation_comparisons"]
    coverage_ok = (
        cov["U2"]["rows"] >= threshold["minimum_conservative_rows"]
        and retention >= threshold["minimum_conservative_within_situation_comparison_retention"]
    )
    if not human["calibration_sufficient"] or not coverage_ok:
        return "ULTRACLEAN_INCONCLUSIVE", (
            f"human calibration sufficient={human['calibration_sufficient']}; "
            f"U2 rows={cov['U2']['rows']}; within-situation retention={retention:.4f}"
        )
    return "ULTRACLEAN_SENSITIVITY_READY", "human calibration and conservative coverage gates passed"


def report_text(result):
    reproduction = result["strict_split_reproduction"]
    lines = [
        "# ValuePrism leakage sensitivity",
        "",
        f"Terminal disposition: {result['terminal_disposition']}",
        "",
        "## Current strict-split reproduction",
        "",
        f"Split contracts reproduce: {reproduction['assertions_pass']}.",
        "",
        "| Quantity | Value |", "|---|---:|",
    ]
    for key in ("strict_train_rows", "strict_test_rows", "strict_train_situations",
                "strict_test_situations", "strict_train_consideration_clusters",
                "strict_test_consideration_clusters"):
        lines.append(f"| {key} | {reproduction[key]:,} |")
    lines += [
        "",
        "Overlap assertions: " + ", ".join(f"{k}={v}" for k, v in reproduction["overlaps"].items()) + ".",
        "",
        "## Retrieval and fixed threshold methods",
        "",
        "Independent views are normalized character 3-5-gram TF-IDF cosine, prefix-stripped token-set similarity, prefix-stripped light-lemma character similarity, and pinned all-MiniLM-L6-v2 sentence embeddings on CPU. Selection uses no labels, activations, probes, baselines, or eventual model performance.",
        "",
        f"Configuration SHA-256: {result['config_sha256']}.",
        "",
        "## Human annotation rule",
        "",
        result["human_audit"]["question"],
        "",
        f"Adjudicated pairs: {result['human_audit']['adjudicated_total']}; calibration sufficient: {result['human_audit']['calibration_sufficient']}.",
        "",
        "## Cleanliness-coverage tradeoff",
        "",
        "| Subset | Rows | Situations | Clusters | Within-situation pairs | Within-consideration pairs | Residual row exposure | Residual pair exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("U0", "U1", "U2", "U3"):
        cov, exp = result["coverage"][label], result["estimated_residual_exposure"][label]["fractions"]
        lines.append(
            f"| {label} | {cov['rows']:,} | {cov['situations']:,} | {cov['consideration_clusters']:,} | "
            f"{cov['within_situation_comparisons']:,} | {cov['within_consideration_comparisons']:,} | "
            f"{exp['rows']:.4%} | {exp['within_situation_comparisons']:.4%} |"
        )
    lines += [
        "",
        "The JSON records Supports/Opposes balance, Value/Right/Duty composition, the fixed lexical domain proxy, and uncertainty-relevant grouping counts for every subset.",
        "",
        "## Worst-case contamination bounds",
        "",
        "For each bounded unit-averaged accuracy, maximum possible leakage inflation is the directly computed exposed-unit fraction in the JSON. These are possible-bias bounds, not realized performance estimates. The suspect-checkerboard manifest permits later removal and recomputation of the continuous reciprocal interaction statistic.",
        "",
        "The prior +0.073 +/- 0.019 overlap effect is deliberate leakage injection, not estimated residual contamination and not a worst-case bound.",
        "",
        "## Recommendation",
        "",
        result["recommendation"],
        "",
        f"Disposition basis: {result['terminal_disposition_basis']}.",
        "",
        "## Exact reproduction command",
        "",
        "Set HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and CUDA_VISIBLE_DEVICES empty, then run:",
        "",
        "python -m geometry_of_truth.leakage.reproduction.sensitivity_analysis --config configs/valueprism_sensitivity.json",
        "",
        "## Artifact hashes",
        "",
    ]
    for name, meta in sorted(result["artifacts"].items()):
        lines.append(f"- {name}: {meta['sha256']}")
    return "\n".join(lines) + "\n"


def run(config_path: Path):
    config, config_hash = config_load(config_path)
    input_hashes = verify_inputs(config)
    df, train, test, reproduction = load_frames(config)
    conf, membership = checkerboards(df)
    ledger, retrieval = retrieve(config, train, test)
    queue = make_queue(ledger, impacts(test, membership))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE, index=False, encoding="utf-8-sig", lineterminator="\n")
    human = audit_state(queue, config)

    high = set(queue.loc[queue.risk_band == "automatic_high_confidence", "test_cluster"].astype(int))
    ambiguous = set(queue.loc[queue.risk_band == "ambiguous_high_risk", "test_cluster"].astype(int)) - high
    safe = set(queue.test_cluster.astype(int)) - high - ambiguous
    subsets = {
        "U0": test.copy(),
        "U1": test[~test.l3.isin(high)].copy(),
        "U2": test[~test.l3.isin(high | ambiguous)].copy(),
    }
    subsets["U3"] = subsets["U2"].copy()
    for label, path in OUTPUTS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        subsets[label][["row_id"]].to_csv(path, index=False, lineterminator="\n")

    reasons = {}
    for cluster in high:
        pairs = sorted(queue.loc[(queue.test_cluster == cluster) & (queue.risk_band == "automatic_high_confidence"), "pair_id"])
        reasons[cluster] = ("U1", "automatic_high_confidence:" + "|".join(pairs))
    for cluster in ambiguous:
        pairs = sorted(queue.loc[(queue.test_cluster == cluster) & (queue.risk_band == "ambiguous_high_risk"), "pair_id"])
        reasons[cluster] = ("U2", "ambiguous_high_risk:" + "|".join(pairs))
    exclusions = [
        {"row_id": row.row_id, "test_cluster": int(row.l3),
         "first_excluded_at": reasons[int(row.l3)][0], "reason": reasons[int(row.l3)][1]}
        for _, row in test.iterrows() if int(row.l3) in reasons
    ]
    exclusion_path = ROOT / "manifests" / "valueprism_sensitivity_exclusions.csv"
    pd.DataFrame(exclusions).sort_values(["first_excluded_at", "test_cluster", "row_id"]).to_csv(
        exclusion_path, index=False, lineterminator="\n"
    )
    suspect = []
    for _, row in conf.iterrows():
        touched = {int(row.cluster_A), int(row.cluster_B)} & (high | ambiguous)
        if touched:
            suspect.append({
                "rank": int(row["rank"]), "suspect_cluster_count": len(touched),
                "suspect_test_clusters": "|".join(str(x) for x in sorted(touched)),
                "reason_tiers": "|".join(sorted({
                    "automatic_high_confidence" if x in high else "ambiguous_high_risk"
                    for x in touched
                })),
            })
    suspect_path = ROOT / "manifests" / "valueprism_sensitivity_suspect_checkerboards.csv"
    pd.DataFrame(suspect).sort_values("rank").to_csv(suspect_path, index=False, lineterminator="\n")

    cov = {label: coverage(frame, config) for label, frame in subsets.items()}
    residual = {"U0": high | ambiguous, "U1": ambiguous, "U2": set(), "U3": set()}
    exp = {label: exposure(subsets[label], residual[label], membership, len(conf)) for label in subsets}
    plot_path = ROOT / "figures" / "cleanliness_coverage_tradeoff.png"
    save_plot(cov, exp, plot_path)
    manifest_paths = list(OUTPUTS.values()) + [exclusion_path, suspect_path]
    artifacts = inventory(manifest_paths + [QUEUE, plot_path, config_path])
    terminal, basis = disposition(human, cov, config)
    recommendation = (
        "Use U1 as the automatic high-confidence sensitivity set. "
        "Complete the blind human audit before treating the ambiguous risk band as calibrated."
        if terminal == "ULTRACLEAN_INCONCLUSIVE"
        else "Report the nested U0 through U3 manifests as sensitivity sets under the calibrated audit rule."
    )
    result = {
        "schema_version": 1, "terminal_disposition": terminal,
        "terminal_disposition_basis": basis, "recommendation": recommendation,
        "config_sha256": config_hash,
        "frozen_input_hashes": input_hashes, "strict_split_reproduction": reproduction,
        "retrieval": retrieval, "human_audit": human,
        "risk_cluster_counts": {"automatic_high_confidence": len(high),
                                "ambiguous_high_risk": len(ambiguous), "safe_outside": len(safe)},
        "coverage": cov, "estimated_residual_exposure": exp,
        "worst_case_accuracy_style_score_inflation": {
            label: exp[label]["fractions"] for label in subsets
        },
        "deliberate_leakage_injection_effect": {
            "effect": 0.073, "standard_error": 0.019,
            "interpretation": "deliberate leakage-injection effect; not residual contamination"
        },
        "suspect_checkerboards": len(suspect), "artifacts": artifacts,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
    }
    result_path = ROOT / "reports" / "valueprism_sensitivity.json"
    report_path = ROOT / "reports" / "VALUEPRISM_SENSITIVITY.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report_text(result), encoding="utf-8")
    artifact_manifest = {
        "schema_version": 1, "config_sha256": config_hash,
        "artifacts": inventory(manifest_paths + [QUEUE, plot_path, config_path, result_path, report_path]),
    }
    artifact_path = ROOT / "reports" / "valueprism_sensitivity_artifact_manifest.json"
    artifact_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal_disposition": terminal, "basis": basis,
                      "risk_cluster_counts": result["risk_cluster_counts"],
                      "coverage": cov}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
