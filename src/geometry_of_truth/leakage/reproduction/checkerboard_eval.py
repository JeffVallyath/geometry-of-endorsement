"""Evaluate reciprocal checkerboards and bootstrap leakage effects.

The reciprocal construction cancels additive situation and consideration
terms, so fixed single-input preferences cannot solve both comparisons.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .split_stress_test import (
    SIT, CON, VAL, SUPPORTS, OPPOSES, SEED,
    load_binary, make_split, score_model, cap,
)

MODELS = ["consideration_only", "situation_only", "pair_text", "nn_consideration"]
SEED_RNG = np.random.default_rng(SEED)


# ==========================================================================
# exact checkerboard mining, restricted to a single frame
# ==========================================================================

def mine_exact_boards(df: pd.DataFrame, con_key: str, max_per_pair: int = 3):
    """Enumerate exact 4-cell reciprocal structures fully contained in `df`.

    Returns a list of dicts with the four (row-index) cells, so a scorer can be
    evaluated on precisely those cells and nothing else.
    """
    # (c_supports, c_opposes) -> {situation: (idx_support_row, idx_oppose_row)}
    ordered: dict[tuple, dict] = defaultdict(dict)
    for s, g in df.groupby(SIT, sort=False):
        sup = g.loc[g[VAL] == SUPPORTS, [con_key]]
        opp = g.loc[g[VAL] == OPPOSES, [con_key]]
        if sup.empty or opp.empty:
            continue
        # first row per key, so each (situation, key) cell is unambiguous
        sup_first = {k: i for i, k in zip(sup.index[::-1], sup[con_key][::-1])}
        opp_first = {k: i for i, k in zip(opp.index[::-1], opp[con_key][::-1])}
        for a, ia in sup_first.items():
            for b, ib in opp_first.items():
                if a != b:
                    ordered[(a, b)][s] = (ia, ib)

    boards, seen = [], set()
    for (a, b), sits_ab in ordered.items():
        if (a, b) in seen or (b, a) in seen:
            continue
        sits_ba = ordered.get((b, a))
        if not sits_ba:
            continue
        seen.add((a, b))
        n = 0
        for s1, (i_s1_c1, i_s1_c2) in sits_ab.items():
            for s2, (i_s2_c2, i_s2_c1) in sits_ba.items():
                if s1 == s2 or n >= max_per_pair:
                    continue
                boards.append({
                    "c1": a, "c2": b, "s1": s1, "s2": s2,
                    # cell -> row index. s1: c1 supports, c2 opposes.
                    "i_s1_c1": i_s1_c1, "i_s1_c2": i_s1_c2,
                    # s2: c2 supports, c1 opposes
                    "i_s2_c2": i_s2_c2, "i_s2_c1": i_s2_c1,
                })
                n += 1
    return boards


def evaluate_boards(boards: list[dict], score_by_index: dict) -> dict:
    """pairwise: fraction of the 2 required comparisons the model gets right.
       both:     fraction of boards it solves in BOTH directions."""
    if not boards:
        return {"pairwise": float("nan"), "both": float("nan"), "n_boards": 0}
    right = both = 0.0
    per_board = []
    for b in boards:
        # Ties get half credit. Without this a model that scores both cells
        # identically (situation_only does, by construction) reads 0.000
        # instead of the 0.500 the theory row predicts, and the empirical
        # rows stop being comparable to it.
        d1 = score_by_index[b["i_s1_c1"]] - score_by_index[b["i_s1_c2"]]
        d2 = score_by_index[b["i_s2_c2"]] - score_by_index[b["i_s2_c1"]]
        a1 = 1.0 if d1 > 0 else (0.5 if d1 == 0 else 0.0)
        a2 = 1.0 if d2 > 0 else (0.5 if d2 == 0 else 0.0)
        # 'both ways' requires two STRICT wins. A tie is not a solved board,
        # so it gets no credit here even though it gets half credit pairwise.
        solved = float(d1 > 0 and d2 > 0)
        right += a1 + a2
        both += solved
        per_board.append((a1 + a2, solved))
    return {
        "pairwise": right / (2 * len(boards)),
        "both": both / len(boards),
        "n_boards": len(boards),
        "_per_board": per_board,
    }


def bootstrap_ci(per_board, idx_fn, rng, n_boot=2000, alpha=0.05):
    arr = np.array([idx_fn(p) for p in per_board], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    stats = arr[draws].mean(axis=1)
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


# ==========================================================================
# main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=60_000)
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_binary()
    out: dict = {}

    train, test, _ = make_split(df, "strict", np.random.default_rng(SEED))
    train = cap(train, args.max_train, rng)
    print(f"\nstrict split: train {len(train):,}  test {len(test):,}")

    # ---------------- exact boards inside the test frame ----------------
    print("\n" + "=" * 78)
    print("EXACT RECIPROCAL CHECKERBOARDS  (all four cells inside the test set)")
    print("=" * 78)
    # Mine on exact consideration strings so the structural key matches the scorer input.
    test_r = test.reset_index(drop=True)
    boards = mine_exact_boards(test_r, CON)
    print(f"exact 4-cell boards: {len(boards):,}")
    print(f"distinct consideration pairs: "
          f"{len({(b['c1'], b['c2']) for b in boards}):,}")
    out["n_exact_boards"] = len(boards)

    if not boards:
        print("none found in this test frame, widen the split or lower max_per_pair")
        return

    print("\n" + "-" * 78)
    hdr = (f"{'model':<22}{'pairwise':>12}{'95% CI':>18}"
           f"{'both ways':>12}{'95% CI':>18}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'ADDITIVE MODEL (theory)':<22}{0.5:>12.3f}{'exact':>18}{0.0:>12.3f}{'exact':>18}")

    for m in MODELS:
        scores = score_model(m, train, test_r)
        by_idx = {i: s for i, s in enumerate(scores)}
        res = evaluate_boards(boards, by_idx)
        ci_p = bootstrap_ci(res["_per_board"], lambda p: p[0] / 2, rng)
        ci_b = bootstrap_ci(res["_per_board"], lambda p: p[1], rng)
        out.setdefault("boards", {})[m] = {
            "pairwise": res["pairwise"], "pairwise_ci": ci_p,
            "both": res["both"], "both_ci": ci_b,
        }
        print(f"{m:<22}{res['pairwise']:>12.3f}"
              f"{f'[{ci_p[0]:.3f},{ci_p[1]:.3f}]':>18}"
              f"{res['both']:>12.3f}"
              f"{f'[{ci_b[0]:.3f},{ci_b[1]:.3f}]':>18}")

    print("\nBoth-ways results require the additive baseline and chance level for interpretation.")

    # ---------------- leakage effects with paired bootstrap ----------------
    print("\n" + "=" * 78)
    print("LEAKAGE EFFECTS, paired bootstrap over test situations")
    print("=" * 78)

    test_sits, test_clusters = set(test[SIT]), set(test["l3"])
    pool_s = df[df[SIT].isin(test_sits) & ~df["l3"].isin(test_clusters)]
    pool_c = df[~df[SIT].isin(test_sits) & df["l3"].isin(test_clusters)]
    n = len(train)

    def build(specs):
        """specs: list of (pool, fraction). Total size always == n, so every
        condition is matched in size AND in how much base data it gives up."""
        parts, taken = [], 0
        for pool, frac in specs:
            k = min(len(pool), int(frac * n))
            if k:
                parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
                taken += k
        parts.append(train.iloc[rng.choice(len(train), n - taken, replace=False)])
        return pd.concat(parts, ignore_index=True)

    variants = {
        "T0  no overlap": build([]),
        "TS  +situations (30%)": build([(pool_s, 0.30)]),
        "TC  +considerations (30%)": build([(pool_c, 0.30)]),
        "TSC +both (15% each)": build([(pool_s, 0.15), (pool_c, 0.15)]),
    }

    def within_situation_per_group(te, scores):
        t = te.copy()
        t["__s"] = scores
        vals = []
        for _, g in t.groupby(SIT):
            sup = g.loc[g[VAL] == SUPPORTS, "__s"].values
            opp = g.loc[g[VAL] == OPPOSES, "__s"].values
            if len(sup) and len(opp):
                d = sup[:, None] - opp[None, :]
                vals.append(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
        return np.array(vals)

    print(f"training size held at {n:,} for every condition\n")
    hdr = f"{'training set':<28}{'pair_text':>12}{'95% CI':>18}{'delta vs T0':>14}"
    print(hdr)
    print("-" * len(hdr))

    base = None
    for label, tr in variants.items():
        per_g = within_situation_per_group(test, score_model("pair_text", tr, test))
        mean = per_g.mean()
        draws = rng.integers(0, len(per_g), size=(2000, len(per_g)))
        ci = np.quantile(per_g[draws].mean(axis=1), [0.025, 0.975])
        if base is None:
            base, base_g = mean, per_g
            delta = "-"
        else:
            d = per_g - base_g          # paired: same situations, same order
            dd = np.quantile(d[draws].mean(axis=1), [0.025, 0.975])
            delta = f"{mean - base:+.3f} [{dd[0]:+.3f},{dd[1]:+.3f}]"
        out.setdefault("leakage", {})[label] = {
            "mean": mean, "ci": list(ci), "delta": delta, "n_train": len(tr)
        }
        print(f"{label:<28}{mean:>12.3f}{f'[{ci[0]:.3f},{ci[1]:.3f}]':>18}{delta:>28}")

    print("\nTSC now adds 15% from each pool, so total added data matches")
    print("TS and TC at 30%. All four conditions are size- and composition-matched.")

    # ---------------- cluster audit sample ----------------
    impact = df.groupby("l3").size().rename("rows")
    forms = df.groupby("l3")["l2"].nunique().rename("n_forms")
    merged = pd.concat([impact, forms], axis=1)
    merged = merged[merged.n_forms > 1].sort_values("rows", ascending=False)
    rows = []
    for cid in merged.index[:400]:
        rows.append({
            "cluster": cid,
            "rows_affected": int(merged.loc[cid, "rows"]),
            "n_forms": int(merged.loc[cid, "n_forms"]),
            "forms": " | ".join(sorted(df.loc[df["l3"] == cid, CON].unique())[:12]),
            "VERDICT_keep_or_split": "",
        })
    pd.DataFrame(rows).to_csv("cluster_audit_queue.csv", index=False)
    cov = merged.head(400)["rows"].sum() / merged["rows"].sum()
    print(f"\nwrote cluster_audit_queue.csv, top 400 merged clusters by impact,")
    print(f"covering {cov:.1%} of all rows in multi-form clusters.")
    out["cluster_audit_coverage"] = float(cov)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
