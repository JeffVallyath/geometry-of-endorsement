"""The single required comparison figure.

Two panels, one row:
  left   primary endpoint -- the four planned contrasts (learned recipe minus
         FIELD_PLUS_LATEST_ERRATUM) with 98.75% paired-root bootstrap intervals.
         A zero line is drawn. An interval crossing zero does NOT prove
         equivalence; point performance and uncertainty are both shown.
  right  quality/time -- all-origin question accuracy against measured mean
         update/refresh time per case, every declared arm plotted. Untrained
         baselines are marked distinctly so a baseline win stays visible.

No arm is hidden, reordered by outcome, or dropped for performing badly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEARNED = ("INV_PAIR_NLL", "FREE_PAIR_CONSISTENCY", "CANONICAL_REPLAY")


def read_csv(path):
    with Path(path).open(encoding="utf8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tables = Path(args.tables)
    headline = read_csv(tables / "headline_results.csv")
    timing = {(r["actor"], r["condition"]): r for r in read_csv(tables / "timing_memory.csv")}
    primary = json.loads((tables / "primary_contrasts.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- left: the four planned contrasts
    contrasts = primary.get("contrasts") or []
    if contrasts:
        labels = [f"{c['actor']}\n{c['recipe']}" for c in contrasts]
        eff = [c["effect"] for c in contrasts]
        lo = [c["effect"] - c["lower"] for c in contrasts]
        hi = [c["upper"] - c["effect"] for c in contrasts]
        y = range(len(contrasts))
        ax1.errorbar(eff, y, xerr=[lo, hi], fmt="o", capsize=4, color="#1f77b4")
        ax1.set_yticks(list(y)); ax1.set_yticklabels(labels, fontsize=9)
        ax1.axvline(0, color="#444", lw=1, ls="--")
        ax1.set_xlabel("learned − FIELD_PLUS_LATEST_ERRATUM\n(complete 34-question bundle, all four origins)")
        ax1.set_title("Primary endpoint: four planned contrasts\n98.75% paired-root bootstrap", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "PRIMARY CONTRASTS UNAVAILABLE\n"
                           f"{primary.get('status', {}).get('status', 'INCOMPLETE')}\n"
                           "missing cells identified in\ntables/primary_contrasts.json",
                 ha="center", va="center", fontsize=10, color="#b00")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title("Primary endpoint", fontsize=10)

    # ---- right: quality vs measured update time, every arm
    for r in headline:
        acc = num(r["all_origin_question_accuracy"])
        t = timing.get((r["actor"], r["condition"]))
        if acc is None or t is None:
            continue
        secs = num(t["mean_update_refresh_seconds"]) or num(t["mean_compile_seconds"])
        if secs is None:
            continue
        learned = r["method"] in LEARNED
        ax2.scatter(secs, acc,
                    marker="o" if learned else "s",
                    color="#1f77b4" if learned else "#d62728",
                    s=55, alpha=.85,
                    edgecolors="black" if r["method"] == "FIELD_PLUS_LATEST_ERRATUM" else "none",
                    linewidths=1.4)
        ax2.annotate(f"{r['actor'][0]}:{r['condition']}", (secs, acc), fontsize=6,
                     xytext=(4, 3), textcoords="offset points")

    ax2.set_xlabel("mean update / refresh seconds per case (measured)")
    ax2.set_ylabel("all-origin question accuracy")
    ax2.set_title("Quality vs measured update cost\nblue circles learned, red squares no-training\n"
                  "(black edge = primary baseline)", fontsize=10)
    ax2.grid(alpha=.25)

    fig.suptitle("Update-method comparison on the supplied fixed cases", fontsize=12)
    fig.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
