"""Build the linear-probe figures from artifacts/figures/figure_data.json.

CPU only; no model inference, no dataset access. Captions live in
figures/CAPTIONS.md and carry the statistical detail the panels do not.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "figures" / "figure_data.json"
OUT = ROOT / "figures"

INK = "#1a1a1a"
MUTED = "#5a5a5a"
GRID = "#dcdcdc"

# One encoding for every scorer, used identically in every figure.
C_DIRECTION = "#2166ac"
C_LOGISTIC = "#b2182b"
C_ANSWER = "#5a5a5a"
C_BASELINE = "#a6a6a6"
C_NULL = "#c9c9c9"
C_MARK = "#b2182b"

SCORER_COLOR = {
    "difference_in_means": C_DIRECTION,
    "logistic": C_LOGISTIC,
    "native_answer_margin": C_ANSWER,
    "sbert_interaction": C_BASELINE,
    "held_out_template": C_DIRECTION,
    "deterministic_random": C_BASELINE,
    "situation_only_activation": C_BASELINE,
    "consideration_only_activation": C_BASELINE,
    "separate_encoding_additive": C_BASELINE,
}


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "lines.linewidth": 1.4,
            "text.color": INK,
        }
    )


def panel_label(ax, letter: str, text: str = "") -> None:
    """Single left-aligned title carrying the panel letter and its heading.

    One title slot rather than two: a long centre title and a left-aligned
    letter collide on narrow panels, and manual offsets reopen a gap between
    rows.
    """
    head = r"$\bf{(" + letter + r")}$"
    ax.set_title(f"{head}  {text}" if text else head, loc="left", fontsize=8.5)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    # Omit the PDF CreationDate so repeated builds are byte-identical.
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  figures/{name}.pdf, figures/{name}.png")


def fig_layer_sweep(data: dict) -> None:
    """I_b (primary) and AUROC (secondary) across layers. Development data."""
    models = [data["models"]["llama"], data["models"]["gemma"]]
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.0), sharex="col", sharey="row",
        height_ratios=(1.5, 1.0),
    )
    letters = [["a", "b"], ["c", "d"]]
    for col, m in enumerate(models):
        layers = [e["layer"] for e in m["layerwise"]]
        sel = m["selected_layer"]

        ax = axes[0, col]
        ax.plot(layers, [e["I_b"] for e in m["layerwise"]], color=C_DIRECTION)
        ax.axhline(0.0, color=INK, lw=0.6, ls=(0, (4, 3)))
        ax.axvline(sel, color=C_MARK, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(-0.3, 2.75)
        ax.set_ylabel("Checkerboard interaction $I_b$" if col == 0 else "")
        ax.tick_params(labelleft=(col == 0))
        ax.annotate(
            f"layer {sel}",
            xy=(sel, 2.6),
            xytext=(4, 0),
            textcoords="offset points",
            color=C_MARK,
            fontsize=7,
            va="top",
        )
        panel_label(ax, letters[0][col], m["label"])
        if col == 0:
            ax.annotate(
                "development data",
                xy=(0.03, 0.93),
                xycoords="axes fraction",
                fontsize=6.5,
                style="italic",
                color=MUTED,
                va="top",
            )

        ax = axes[1, col]
        ax.plot(layers, [e["auroc"] for e in m["layerwise"]], color=C_DIRECTION)
        ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (4, 3)))
        ax.axvline(sel, color=C_MARK, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(0.45, 0.90)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_ylabel("AUROC" if col == 0 else "")
        ax.tick_params(labelleft=(col == 0))
        ax.set_xlabel("Layer")
        ax.set_xlim(0, max(layers))
        panel_label(ax, letters[1][col])

    fig.tight_layout()
    save(fig, "fig1_layer_sweep")


def _dot_panel(ax, model, keys, open_keys=()):
    ev = model["evaluations"]
    y = np.arange(len(keys))[::-1]
    for pos, key in zip(y, keys):
        e = ev[key]
        color = SCORER_COLOR[key]
        is_open = key in open_keys
        if e.get("ci_low") is not None:
            ax.errorbar(
                e["I_b"], pos,
                xerr=[[e["I_b"] - e["ci_low"]], [e["ci_high"] - e["I_b"]]],
                fmt="o", ms=4.5, color=color, ecolor=color,
                elinewidth=1.1, capsize=2.5, capthick=1.1,
            )
        else:
            ax.plot(
                e["I_b"], pos,
                marker="D" if is_open else "o", ms=4.5 if is_open else 4,
                mfc="white" if is_open else color, mec=color, ls="none",
            )
        ax.annotate(
            f"{e['I_b'] + 0.0:.2f}".replace("-0.00", "0.00"),
            xy=(e["I_b"], pos), xytext=(0, 7), textcoords="offset points",
            ha="center", fontsize=6.5, color=color,
        )
    ax.axvline(0.0, color=INK, lw=0.6, ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([ev[k]["label"] for k in keys])
    ax.grid(axis="y", visible=False)
    return y


def fig_scorer_comparison(data: dict) -> None:
    """Interaction by scorer at the selected layer, with 95% CIs."""
    models = [data["models"]["llama"], data["models"]["gemma"]]
    keys = [
        "native_answer_margin",
        "difference_in_means",
        "logistic",
        "sbert_interaction",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, m, letter in zip(axes, models, "ab"):
        _dot_panel(ax, m, keys)
        ax.set_xlabel("Checkerboard interaction $I_b$   (95% CI)")
        ax.set_xlim(-0.25, 3.0)
        ax.set_ylim(-0.5, len(keys) - 0.3)
        panel_label(ax, letter, f"{m['label']}, layer {m['selected_layer']}")
    fig.tight_layout()
    save(fig, "fig2_scorer_comparison")


def fig_permutation_null(data: dict) -> None:
    """Observed interaction against the situation-flip null."""
    models = [data["models"]["llama"], data["models"]["gemma"]]
    probes = [
        ("difference_in_means", "Support/opposition direction", C_DIRECTION),
        ("logistic", "Logistic activation probe", C_LOGISTIC),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.9), sharex=True)
    letters = [["a", "b"], ["c", "d"]]

    # One bin grid and a density y-axis for every panel: B differs by 50x
    # between the columns, so raw counts would not be comparable, and the
    # observed line must not fall on the axis boundary.
    def null_for(m, key):
        large = m.get("permutation_null_large", {}).get("probes")
        return (large or {}).get(key) or m["permutation_null"][key]

    extremes = [v for m in models for key, _, _ in probes
                for v in (list(null_for(m, key)["values"]) + [null_for(m, key)["observed"]])]
    span = max(extremes) - min(extremes)
    lo, hi = min(extremes) - 0.04 * span, max(extremes) + 0.06 * span
    bins = np.linspace(lo, hi, 49)

    for col, m in enumerate(models):
        for row, (key, label, color) in enumerate(probes):
            ax = axes[row, col]
            null = null_for(m, key)
            values = np.asarray(null["values"], dtype=float)
            ax.hist(values, bins=bins, density=True,
                    color=C_NULL, edgecolor="white", linewidth=0.3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
            ax.axvline(null["observed"], color=color, lw=1.6)
            k = null.get("k_ge_observed")
            tail = f"$p$ = {null['p']:.4f}"
            if k is not None:
                tail += f"  ($k$ = {k})"
            ax.annotate(
                f"observed {null['observed']:.2f}\n{tail}",
                xy=(null["observed"], ax.get_ylim()[1]),
                xytext=(-5, -3), textcoords="offset points",
                ha="right", va="top", fontsize=6.5, color=color,
            )
            ax.annotate(
                f"{label}\n$B$ = {null['B']:,}",
                xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=6.5, color=color, va="top",
            )
            ax.set_ylabel("Null density" if col == 0 else "")
            if row == len(probes) - 1:
                ax.set_xlabel("Checkerboard interaction $I_b$")
            ax.grid(axis="x", visible=False)
            panel_label(ax, letters[row][col], m["label"] if row == 0 else "")
    fig.tight_layout()
    save(fig, "fig3_permutation_null")


def fig_truth_control(data: dict) -> None:
    """Factual True/False positive control."""
    t = data["truth_control"]["llama"]
    gemma = data["models"]["gemma"]
    sweep = t["dev_sweep"]
    layers = [e["layer"] for e in sweep]
    sel = t["selected_layer"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), width_ratios=(1.65, 1.0))

    ax = axes[0]
    ax.plot(layers, [e["standard_T"] for e in sweep], color=C_DIRECTION,
            label="Standard A/B mapping")
    ax.plot(layers, [e["reversed_T"] for e in sweep], color=C_ANSWER,
            ls=(0, (3, 2)), label="Reversed mapping")
    ax.axhline(0.0, color=INK, lw=0.6, ls=(0, (4, 3)))
    ax.axvline(sel, color=C_MARK, lw=0.8, ls=(0, (2, 2)))
    ax.annotate(f"layer {sel}", xy=(sel, 2.62), xytext=(4, 0),
                textcoords="offset points", color=C_MARK, fontsize=7, va="top")
    ax.annotate("development data", xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=6.5, style="italic", color=MUTED, va="top")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Standardized separation $T$")
    ax.set_xlim(0, max(layers))
    ax.set_ylim(-0.15, 2.75)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07))
    panel_label(ax, "a", "Layer sweep, both answer mappings")

    ax = axes[1]
    labels = ["Llama\nprimary", "Llama\ntransfer", "Gemma\nprimary"]
    sublabels = [f"layer {sel}", "held-out 1/2", f"layer {gemma['truth_control_layer']}"]
    points = [t["primary_T"], t["transfer_T"], gemma["truth_control_T"]]
    cis = [t["primary_ci"], t["transfer_ci"], None]
    colors = (C_DIRECTION, C_ANSWER, C_DIRECTION)
    x = np.arange(len(points))
    for xi, point, ci, color in zip(x, points, cis, colors):
        if ci is not None:
            ax.errorbar(xi, point,
                        yerr=[[point - ci["low"]], [ci["high"] - point]],
                        fmt="o", ms=5, color=color, elinewidth=1.2,
                        capsize=3, capthick=1.2)
        else:
            ax.plot(xi, point, marker="D", ms=4.5, mfc="white", mec=color, ls="none")
        ax.annotate(f"{point:.3f}", xy=(xi, point), xytext=(11, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=color)
    for xi, sub in zip(x, sublabels):
        ax.annotate(sub, xy=(xi, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 3.05)
    ax.set_ylim(1.70, 2.05)
    ax.set_ylabel("Standardized separation $T$")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "b", "Held-out test, 95% CI")

    fig.tight_layout()
    save(fig, "fig4_truth_control")


def fig_specificity(data: dict) -> None:
    """Relation effect beside the confound baselines and structural checks."""
    models = [data["models"]["llama"], data["models"]["gemma"]]
    keys = [
        "difference_in_means",
        "held_out_template",
        "sbert_interaction",
        "deterministic_random",
        "situation_only_activation",
        "consideration_only_activation",
        "separate_encoding_additive",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, m, letter in zip(axes, models, "ab"):
        _dot_panel(ax, m, keys, open_keys={"held_out_template"})
        for boundary in (len(keys) - 2.5, len(keys) - 4.5):
            ax.axhline(boundary, color=GRID, lw=0.7)
        ax.set_xlabel("Checkerboard interaction $I_b$")
        ax.set_xlim(-0.8, 3.0)
        ax.set_ylim(-0.6, len(keys) - 0.3)
        panel_label(ax, letter, f"{m['label']}, layer {m['selected_layer']}")
        if letter == "b":
            for text, pos in (
                ("relation", len(keys) - 1.5),
                ("confound\nbaselines", len(keys) - 3.5),
                ("zero by\nconstruction", 1.0),
            ):
                ax.annotate(text, xy=(1.0, pos), xycoords=("axes fraction", "data"),
                            xytext=(7, 0), textcoords="offset points",
                            fontsize=6, color=MUTED, va="center")
    fig.tight_layout()
    save(fig, "fig5_specificity")


def main() -> None:
    style()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    print("Writing figures:")
    fig_layer_sweep(data)
    fig_scorer_comparison(data)
    fig_permutation_null(data)
    fig_truth_control(data)
    fig_specificity(data)


if __name__ == "__main__":
    main()
