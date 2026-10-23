from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .provenance import write_json_atomic


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def write_results(results: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cleaned = _clean_json(results)
    write_json_atomic(cleaned, target / "truth_control_v2_results.json")
    (target / "TRUTH_CONTROL_V2_REPORT.md").write_text(
        render_report(cleaned), encoding="utf-8"
    )
    plot_results(cleaned, target / "layerwise_signed_separation.png")


def plot_results(results: dict[str, Any], path: str | Path) -> None:
    sweep = results["development_layer_sweep"]
    layers = [int(row["layer"]) for row in sweep]
    selection = [float(row["selection_score"]) for row in sweep]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(layers, selection, color="#1f77b4", label="A/B dev selection score")
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.axvline(
        int(results["selected_layer"]),
        color="#777777",
        linestyle=":",
        label="dev-selected layer",
    )
    for key, row in results["confirmatory_layer_results"].items():
        layer = int(key)
        primary = float(row["test"]["primary"]["overall"]["T"])
        transfer = float(row["test"]["transfer"]["overall"]["T"])
        axis.scatter([layer], [primary], color="#2ca02c", marker="o")
        axis.scatter([layer], [transfer], color="#d62728", marker="s")
    axis.scatter([], [], color="#2ca02c", marker="o", label="A/B test T (selected/neighbors)")
    axis.scatter([], [], color="#d62728", marker="s", label="1/2 test T (selected/neighbors)")
    axis.set(
        xlabel="Transformer layer",
        ylabel="Signed standardized separation T",
        title="Truth control v2: development selection and frozen test layers",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(results: dict[str, Any]) -> str:
    selected = results["confirmatory_layer_results"][str(results["selected_layer"])]
    primary = selected["test"]["primary"]["overall"]
    transfer = selected["test"]["transfer"]["overall"]
    primary_ci = selected["group_bootstrap_ci"]["primary"]
    transfer_ci = selected["group_bootstrap_ci"]["transfer"]
    null = results["permutation_null"]
    lines = [
        "# Truth Control v2 Report",
        "",
        f"**Terminal disposition:** {results['terminal_disposition']}",
        "",
        "Truth v1 remains TRUTH_CONTROL_V1_STRICT_FAIL; this report does not overwrite it.",
        "",
        "## Primary signed result",
        "",
        f"- Development-selected layer: {results['selected_layer']}",
        f"- Independent training partitions: {results['training_partitions']}",
        f"- Primary A/B test T: {primary['T']:.6f}",
        f"- Primary 95% grouped CI: [{primary_ci['low']:.6f}, {primary_ci['high']:.6f}]",
        f"- Held-out 1/2 test T: {transfer['T']:.6f}",
        f"- Held-out 1/2 95% grouped CI: [{transfer_ci['low']:.6f}, {transfer_ci['high']:.6f}]",
        f"- Directional consensus C: {results['directional_consensus_C']:.6f}",
        f"- Descriptive A/B ensemble macro-AUROC: {primary['ensemble_macro_dataset_auroc']:.6f}",
        f"- Descriptive 1/2 ensemble macro-AUROC: {transfer['ensemble_macro_dataset_auroc']:.6f}",
        "",
        "## Complete group-preserving null",
        "",
        f"- Permutations: {null['permutations']}",
        f"- Null mean +/- sd: {null['mean']:.6f} +/- {null['std']:.6f}",
        f"- Exact p(null >= observed): {null['p_greater_equal']:.6f}",
        "- Every null run refits all layers and repeats development-only layer selection.",
        "- AUROC is descriptive only and does not determine this gate.",
        "",
        "## Frozen checks",
        "",
    ]
    lines.extend(
        f"- {name}: {value}" for name, value in results["checks"].items()
    )
    lines.extend(
        [
            "",
            "Full mapping-cell transfers, native semantic answer controls, null values, "
            "layer-selection curves, artifact hashes, and provenance are retained in "
            "truth_control_v2_results.json and environment.json.",
        ]
    )
    return "\n".join(lines) + "\n"
