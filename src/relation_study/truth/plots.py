from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from .results import layer_sweep


def layer_selection(v2: dict[str, Any]):
    sweep = layer_sweep(v2)
    ax = sweep.plot(x="layer", y=["standard development T", "reversed development T", "selection score"], marker="o", figsize=(10, 4))
    ax.axvline(v2["selected_layer"], color="black", linestyle="--", label="selected layer")
    ax.set_ylabel("signed standardized separation T")
    ax.set_title("Development layer selection")
    ax.legend()
    return ax


def permutation_null(v2: dict[str, Any]):
    null = v2["permutation_null"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(null["values"], bins=35, color="#7ea6e0", edgecolor="white")
    ax.axvline(null["observed_T"], color="#c43d3d", linewidth=2, label=f"observed T = {null['observed_T']:.3f}")
    ax.set_xlabel("complete null T")
    ax.set_ylabel("count")
    ax.set_title("Group preserving permutation null")
    ax.legend()
    return ax
