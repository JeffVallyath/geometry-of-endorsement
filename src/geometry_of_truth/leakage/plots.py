from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from .results import stress_draws


def stress_test(results: dict[str, Any]):
    draws = stress_draws(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, score, title in (
        (axes[0], "consideration overlap score", "Restore consideration identities"),
        (axes[1], "situation overlap score", "Restore situations"),
    ):
        for _, row in draws.iterrows():
            ax.plot([0, 1], [row["strict score"], row[score]], marker="o", alpha=0.8)
        ax.set_xticks([0, 1], ["strict", "restored overlap"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("paired accuracy")
    fig.tight_layout()
    return fig
