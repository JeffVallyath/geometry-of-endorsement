from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .support.metrics import format_both_ways


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def write_json_atomic(value: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_text(
        json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, target)


def write_results(results: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cleaned = clean_json(results)
    write_json_atomic(cleaned, target / "m1_vertical_slice_results.json")
    (target / "M1_VERTICAL_SLICE_REPORT.md").write_text(
        render_report(cleaned), encoding="utf-8"
    )
    plot_layerwise(cleaned, target / "m1_layerwise_probe_performance.png")


def plot_layerwise(results: dict[str, Any], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["layerwise_difference_in_means"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    layers = [row["layer"] for row in rows]
    axes[0].plot(
        layers,
        [row["within_situation_macro"] for row in rows],
        label="within situation",
    )
    axes[0].plot(
        layers,
        [row["within_consideration_macro"] for row in rows],
        label="within consideration",
    )
    axes[1].plot(
        layers,
        [row["checkerboard_interaction_mean"] for row in rows],
        label="standardized I_b (primary)",
    )
    axes[1].plot(
        layers,
        [row["checkerboard_signed_board_mean"] for row in rows],
        label="signed B_b (secondary)",
    )
    for axis in axes:
        axis.axhline(0.5, color="black", linewidth=1, alpha=0.4)
        axis.axvline(
            results["selection"]["selected_layer"],
            color="#777777",
            linestyle=":",
            label="selected layer",
        )
        axis.set(xlabel="Transformer layer")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_title("Mirrored paired metrics")
    axes[0].set(ylabel="Accuracy", ylim=(-0.02, 1.02))
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[1].set(ylabel="Board statistic", title="Reciprocal checkerboards")
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        metadata={
            "Description": (
                "M1 layerwise metrics; pilot_manifest_sha256="
                + results["manifest"]["pilot_manifest_sha256"]
            )
        },
    )
    plt.close(figure)


def _metric_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Scorer | Within situation | Within consideration | I_b (primary) | B_b (secondary) | Pairwise | Both-ways (descriptive) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["evaluations"].items():
        lines.append(
            f"| {name} | {row['within_situation_macro']:.4f} | "
            f"{row['within_consideration_macro']:.4f} | "
            f"{row['checkerboard_interaction_mean']:.4f} | "
            f"{row['checkerboard_signed_board_mean']:.4f} | "
            f"{row['checkerboard_pairwise']:.4f} | "
            f"{format_both_ways(row['checkerboard_both_ways'])} |"
        )
    return lines


def _layer_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Layer | Within situation | Within consideration | I_b (primary) | B_b (secondary) | Pairwise | Both-ways (descriptive) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["layerwise_difference_in_means"]:
        lines.append(
            f"| {row['layer']} | {row['within_situation_macro']:.4f} | "
            f"{row['within_consideration_macro']:.4f} | "
            f"{row['checkerboard_interaction_mean']:.4f} | "
            f"{row['checkerboard_signed_board_mean']:.4f} | "
            f"{row['checkerboard_pairwise']:.4f} | "
            f"{format_both_ways(row['checkerboard_both_ways'])} |"
        )
    return lines


def _transfer_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Frozen probe | Within situation | Within consideration | I_b (primary) | B_b (secondary) | Pairwise | Both-ways (descriptive) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results["held_out_template_transfer"].items():
        lines.append(
            f"| {name} | {row['within_situation_macro']:.4f} | "
            f"{row['within_consideration_macro']:.4f} | "
            f"{row['checkerboard_interaction_mean']:.4f} | "
            f"{row['checkerboard_signed_board_mean']:.4f} | "
            f"{row['checkerboard_pairwise']:.4f} | "
            f"{format_both_ways(row['checkerboard_both_ways'])} |"
        )
    return lines


def _permutation_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "| Probe | Metric | Real | Null mean | Null 95% interval | p(null >= real) | Rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family, metrics in results["permutation_nulls"].items():
        for metric, row in metrics.items():
            lines.append(
                f"| {family} | {metric} | {row['observed']:.4f} | "
                f"{row['mean']:.4f} | [{row['interval_95'][0]:.4f}, "
                f"{row['interval_95'][1]:.4f}] | "
                f"{row['empirical_p_greater_equal']:.6f} | "
                f"{row['rank_ascending']}/{row['n'] + 1} |"
            )
    return lines


def _error_lines(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for family, groups in results["error_analysis"].items():
        lines.append(f"### {family}")
        lines.append("")
        for name in ("both_right", "both_wrong_or_partial"):
            lines.append(f"- {name}:")
            for row in groups[name][:10]:
                lines.append(
                    f"  - {row['board_id']}: gaps "
                    f"{row['gap_1']:.4f}, {row['gap_2']:.4f}"
                )
            if not groups[name]:
                lines.append("  - None.")
        lines.append("")
    return lines


def render_report(results: dict[str, Any]) -> str:
    runtime = results["runtime"]
    manifest = results["manifest"]
    truth = results["truth_positive_control"]
    selection = results["selection"]
    lines = [
        "# M1 Vertical Slice Report",
        "",
        f"**Terminal disposition:** `{results['terminal_disposition']}`",
        "",
        f"**Disposition basis:** {results['disposition_rationale']}",
        "",
        f"**Claim boundary:** {results['claim_boundary']}",
        "",
        "## Frozen execution",
        "",
        f"- Repository commit: `{runtime['repository_commit']}`",
        *(
            [
                f"- Analysis-only replay commit: `{results['reanalysis']['analysis_commit']}`",
                f"- Source disposition preserved: `{results['reanalysis']['source_original_disposition']}`",
                f"- Reanalysis protocol: `{results['reanalysis']['protocol']}`; Llama loaded: {results['reanalysis']['llama_loaded']}",
            ]
            if "reanalysis" in results else []
        ),
        f"- Model: `{runtime['model_id']}` at `{runtime['model_revision']}`",
        f"- Tokenizer revision: `{runtime['tokenizer_revision']}`",
        f"- Chat template hash: `{runtime['chat_template_sha256']}`",
        f"- GPU: {runtime['gpu_name']} ({runtime['gpu_memory_mib']} MiB)",
        f"- Model dtype: `{runtime['model_dtype']}`; cached activation dtype: `{runtime['activation_dtype']}`",
        f"- Peak allocated VRAM: {runtime['peak_vram_bytes']} bytes",
        f"- Runtime: {runtime['elapsed_seconds']:.1f} seconds; cache: {runtime['cache_bytes']} bytes",
        f"- Activation position: final non-padding prompt token before the assistant answer",
        f"- Selected layer: {selection['selected_layer']}; logistic C: {selection['logistic_c']}",
        "",
        "## Manifest and prompt contract",
        "",
        f"- Pilot manifest hash: `{manifest['pilot_manifest_sha256']}`",
        f"- Train/select/eval rows: {results['sample_counts']['pilot_train']} / "
        f"{results['sample_counts']['pilot_select']} / {results['sample_counts']['pilot_eval']}",
        f"- Evaluation checkerboards: {results['sample_counts']['pilot_eval_boards']}",
        f"- Prompt contract hash: `{results['prompt_contract_sha256']}`",
        f"- Cache index hash: `{results['cache_index_sha256']}`",
        f"- Cache manifest hash: `{results['cache_manifest_sha256']}`",
        f"- Probe-parameter hash: `{results['probe_parameters_sha256']}`",
        f"- Primary symbols: A/B with deterministic item-hash reversal",
        f"- Held-out symbols: 1/2 with a separately frozen template",
        f"- Verbalizers: sequence-level teacher-forced candidate log probabilities",
        "",
        "Exact frozen prompt specification:",
        "",
        "```json",
        json.dumps(results["prompt_specification"], indent=2, sort_keys=True),
        "```",
        "",
        "## Truth positive control",
        "",
        f"- Strict imported disposition: `{truth['terminal_disposition']}`",
        f"- Source truth disposition: `{truth['source_terminal_disposition']}`",
        f"- Protocol: `{truth['protocol']}`",
        f"- M1 truth gate pass: {results['validation']['truth_positive_control_passed']}",
        f"- Development-selected truth layer: {truth['selected_layer']}",
        f"- Primary signed cross-fit T: {truth['primary_test_T']:.6f}",
        f"- Exact group-preserving permutation p: {truth['exact_permutation_p']:.9f}",
        f"- Directional consensus C: {truth['directional_consensus_C']:.6f}",
        f"- Truth scientific commit: `{truth['scientific_commit']}`",
        f"- Imported truth results SHA-256: `{truth['source_results_sha256']}`",
        f"- Gate details: {truth['checks']}",
        "",
        "## Moral relation results",
        "",
        "Primary: standardized reciprocal interaction contrast `I_b`. Secondary: signed board score `B_b`. Pairwise and both-ways are descriptive only; both-ways reference levels are chance 0.258 and additive 0.000.",
        "",
        *_metric_table(results),
        "",
        "### Full post-freeze DIM layer curve",
        "",
        *_layer_table(results),
        "",
        "The same curve is plotted in `m1_layerwise_probe_performance.png`.",
        "",
        "## Permutation nulls and dyadic-robust uncertainty",
        "",
        *_permutation_table(results),
        "",
    ]
    for family in results["permutation_nulls"]:
        delta = results["dyadic_delta_ib_over_sbert"][family]
        transfer_delta = results["held_out_template_delta_ib_over_sbert"][family]
        lines.extend(
            [
                f"- {family} minus frozen sbert_interaction, Delta I_b: {delta['point']:.4f}, "
                f"dyadic-robust 95% CI [{delta['low']:.4f}, {delta['high']:.4f}], "
                f"SE {delta['standard_error']:.4f}.",
                f"- {family} held-out-template Delta I_b: {transfer_delta['point']:.4f}, "
                f"dyadic-robust 95% CI [{transfer_delta['low']:.4f}, "
                f"{transfer_delta['high']:.4f}].",
            ]
        )
    lines.extend(
        [
            "",
            "## Controls, transfer, and calibration",
            "",
            "### Held-out 1/2 template (frozen probes, no refit)",
            "",
            *_transfer_table(results),
            "",
            f"- Exact invariants: {results['exact_invariants']}",
            f"- Pipeline checks: {results['pipeline_checks']}",
            f"- Concentration checks: {results['concentration_checks']}",
            f"- Answer-map controls: {results['mapping_controls']}",
            f"- Calibration: {results['calibration']}",
            f"- Board-family analysis: {results['board_family_analysis']}",
            "",
            "## Instrument validation method",
            "",
            "The evaluator treats clean execution and plausible tables as insufficient evidence. It checks quantities with mathematically required values: situation-only within-situation and consideration-only within-consideration equal 0.500; additive I_b, tie-neutral signed B_b, and additive both-ways equal 0.000; score scales are fitted once on pilot_select; and checkerboard uncertainty uses an endpoint-swap-invariant dyadic estimator. Any failed invariant forces PIPELINE_NOT_VALIDATED.",
            "",
            "## Error analysis",
            "",
            "Only deterministic board IDs and score gaps are shown; no ValuePrism text "
            "is included.",
            "",
            *_error_lines(results),
            "",
            "## Failed experiments or implementation problems",
            "",
        ]
    )
    failures = results.get("failed_experiments") or []
    lines.extend([f"- {item}" for item in failures] or ["- None recorded."])
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            *[f"- `{command}`" for command in results["reproduction_commands"]],
            "",
            "## Recommended next action",
            "",
            results["recommended_next_action"],
            "",
        ]
    )
    return "\n".join(lines)
