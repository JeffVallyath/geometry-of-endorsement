"""
Relation-purity checks for the ValuePrism split.

Four things the stress test left open, plus the highest-purity subset.

1. CLUSTER AUDIT — connected-component clustering chains transitively (A~B, B~C
   merges A and C even if unrelated). Report the size distribution and print the
   largest clusters so over-merging is visible rather than assumed.

2. TWO PAIRED METRICS — the mirror pair. Together they make BOTH single-input
   shortcuts impossible by construction rather than merely weak:

     within-situation:     fix the situation, compare a supporting vs an opposing
                           CONSIDERATION.  situation_only == 0.500 exactly.
     within-consideration: fix the consideration, compare a supporting vs an
                           opposing SITUATION. consideration_only == 0.500 exactly.

   Restricting to both-valence considerations is NOT sufficient on its own — a
   consideration can carry both labels while still being Supports 90% of the
   time, so its wording stays predictive. The mirror metric is what closes it.

3. CHECKERBOARDS — the purest relational subset. Find (s1,s2,c1,c2) with

              c1        c2
      s1   Supports  Opposes
      s2   Opposes   Supports

   No situation-only, consideration-only, or additive model can solve this; it
   is literally XOR over the two inputs.

4. RETAINED vs DISCARDED — the strict split throws away rows whose endpoints
   land on opposite sides. Check the survivors are not a biased subsample.

Run:
    python relation_purity.py --out purity.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .split_stress_test import (
    SIT, CON, VAL, SUPPORTS, OPPOSES, SEED,
    load_binary, make_split, score_model, cap,
)

MODELS = ["consideration_only", "situation_only", "pair_text", "nn_consideration"]


# ==========================================================================
# paired metrics
# ==========================================================================

def paired_accuracy(test: pd.DataFrame, scores: np.ndarray, fix: str, vary: str):
    """P(score(supporting) > score(opposing)) holding `fix` constant.

    fix=SIT  -> within-situation      (situation_only is 0.500 by construction)
    fix=CON  -> within-consideration  (consideration_only is 0.500 by construction)
    """
    t = test.copy()
    t["__s"] = scores
    wins = total = 0.0
    groups = 0
    for _, g in t.groupby(fix):
        sup = g.loc[g[VAL] == SUPPORTS, "__s"].values
        opp = g.loc[g[VAL] == OPPOSES, "__s"].values
        if len(sup) == 0 or len(opp) == 0:
            continue
        groups += 1
        d = sup[:, None] - opp[None, :]
        wins += (d > 0).sum() + 0.5 * (d == 0).sum()
        total += d.size
    return (wins / total if total else float("nan")), int(total), groups


# ==========================================================================
# checkerboards
# ==========================================================================

def find_checkerboards(df: pd.DataFrame, con_key: str = "l3"):
    """Return reciprocal discordant consideration pairs and the situations
    realizing them. Uses the clustered consideration key so orthographic
    variants do not create fake checkerboards."""
    # (c_supporting, c_opposing) -> situations where that ordering holds
    ordered: dict[tuple, list] = defaultdict(list)
    for s, g in df.groupby(SIT):
        sup = g.loc[g[VAL] == SUPPORTS, con_key].unique()
        opp = g.loc[g[VAL] == OPPOSES, con_key].unique()
        for a in sup:
            for b in opp:
                if a != b:
                    ordered[(a, b)].append(s)

    boards = []
    seen = set()
    for (a, b), sits_ab in ordered.items():
        if (a, b) in seen:
            continue
        sits_ba = ordered.get((b, a))
        if not sits_ba:
            continue
        seen.add((a, b))
        seen.add((b, a))
        boards.append({
            "c1": a, "c2": b,
            "n_situations_c1_supports": len(sits_ab),
            "n_situations_c2_supports": len(sits_ba),
            "example_s1": sits_ab[0], "example_s2": sits_ba[0],
        })
    return boards, ordered


# ==========================================================================
# main
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=60_000)
    ap.add_argument("--max-test", type=int, default=25_000)
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_binary()
    out: dict = {}

    # ---------------- 1. cluster audit ----------------
    print("\n" + "=" * 78)
    print("1. CLUSTER AUDIT — is connected-component chaining over-merging?")
    print("=" * 78)
    sizes = df.groupby("l3")["l2"].nunique()
    dist = Counter()
    for n in sizes:
        dist["1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else
             "21-100" if n <= 100 else ">100"] += 1
    print(f"{len(sizes):,} clusters over {df['l2'].nunique():,} L2 forms")
    for k in ["1", "2-5", "6-20", "21-100", ">100"]:
        print(f"  {k:>8} distinct forms: {dist[k]:,} clusters")
    out["cluster_size_distribution"] = dict(dist)

    big = sizes.sort_values(ascending=False).head(5)
    print("\n5 largest clusters (chaining risk is visible here):")
    for cid, n in big.items():
        forms = sorted(df.loc[df["l3"] == cid, "l2"].unique())[:8]
        print(f"  cluster {cid}  ({n} forms)  e.g. {forms}")
    out["largest_cluster_sizes"] = big.to_dict()
    if big.iloc[0] > 100:
        print("\n  WARNING: largest cluster exceeds 100 forms. Chaining is likely.")
        print("  Lower the threshold or switch to average-linkage with a size cap.")

    # ---------------- 2. checkerboards ----------------
    print("\n" + "=" * 78)
    print("2. CHECKERBOARDS — the XOR subset no additive model can solve")
    print("=" * 78)
    boards, _ = find_checkerboards(df)
    out["n_checkerboard_pairs"] = len(boards)
    print(f"reciprocal discordant consideration pairs: {len(boards):,}")
    if boards:
        boards.sort(key=lambda b: -(b["n_situations_c1_supports"] + b["n_situations_c2_supports"]))
        print("\ntop 5 by situation coverage:")
        for b in boards[:5]:
            f1 = sorted(df.loc[df["l3"] == b["c1"], CON].unique())[0]
            f2 = sorted(df.loc[df["l3"] == b["c2"], CON].unique())[0]
            print(f"  '{f1}'  vs  '{f2}'")
            print(f"     {b['n_situations_c1_supports']} situations where the first supports,"
                  f" {b['n_situations_c2_supports']} where the second does")
        involved = {b["c1"] for b in boards} | {b["c2"] for b in boards}
        rows = df[df["l3"].isin(involved)]
        out["checkerboard_clusters"] = len(involved)
        out["checkerboard_rows"] = len(rows)
        print(f"\nclusters involved: {len(involved):,}   rows touched: {len(rows):,}")

    # ---------------- 3. the mirror metrics ----------------
    print("\n" + "=" * 78)
    print("3. PAIRED METRICS — within-situation AND within-consideration")
    print("=" * 78)

    conditions = {}
    train, test, disc = make_split(df, "strict", np.random.default_rng(SEED))
    conditions["strict"] = (train, test)

    bv_cons = {c for c, g in df.groupby("l3")[VAL]
               if SUPPORTS in set(g) and OPPOSES in set(g)}
    conditions["strict + both-valence only"] = (train, test[test["l3"].isin(bv_cons)])

    cb = {b["c1"] for b in boards} | {b["c2"] for b in boards}
    if cb:
        conditions["strict + checkerboard only"] = (train, test[test["l3"].isin(cb)])

    for label, (tr, te) in conditions.items():
        tr_c, te_c = cap(tr, args.max_train, rng), cap(te, args.max_test, rng)
        print(f"\n[{label}]   train {len(tr_c):,}   test {len(te_c):,}")
        hdr = f"{'model':<22}{'within-situation':>20}{'within-consideration':>24}"
        print(hdr)
        print("-" * len(hdr))
        entry = {"n_train": len(tr_c), "n_test": len(te_c)}
        for m in MODELS:
            sc = score_model(m, tr_c, te_c)
            # The group key MUST match the model's input granularity or the
            # invariant breaks. within-consideration groups by the EXACT
            # string, because that is what consideration_only reads. Grouping
            # by cluster instead let the model vary inside the group and the
            # sanity check silently failed at 0.715.
            a_sit, n_sit, g_sit = paired_accuracy(te_c, sc, SIT, CON)
            a_con, n_con, g_con = paired_accuracy(te_c, sc, CON, SIT)
            entry[m] = {"within_situation": a_sit, "within_consideration": a_con}
            flag = ""
            if m == "situation_only" and abs(a_sit - 0.5) > 1e-9:
                flag = "  <-- INVARIANT VIOLATED (must be exactly 0.500)"
            if m == "consideration_only" and abs(a_con - 0.5) > 1e-9:
                flag = "  <-- INVARIANT VIOLATED (must be exactly 0.500)"
            print(f"{m:<22}{a_sit:>20.3f}{a_con:>24.3f}{flag}")
        entry["pairs"] = {"within_situation": n_sit, "within_consideration": n_con,
                          "groups_sit": g_sit, "groups_con": g_con}
        print(f"{'':<22}{n_sit:>20,}{n_con:>24,}   (pairs)")
        conditions[label] = None
        out.setdefault("paired_metrics", {})[label] = entry

    print("\nread: situation_only MUST be 0.500 in the left column and")
    print("      consideration_only MUST be 0.500 in the right column.")
    print("      Any model above chance in BOTH columns cannot be explained")
    print("      by either input alone.")

    # ---------------- 4. retained vs discarded ----------------
    print("\n" + "=" * 78)
    print("4. RETAINED vs DISCARDED — is the strict subset biased?")
    print("=" * 78)
    in_ts = df[SIT].isin(set(test[SIT]))
    in_tk = df["l3"].isin(set(test["l3"]))
    discarded = df[in_ts ^ in_tk]
    retained = pd.concat([train, test])
    print(f"retained {len(retained):,}   discarded {len(discarded):,}")
    comp = {}
    for name, sub in (("retained", retained), ("discarded", discarded)):
        comp[name] = {
            "pct_supports": round(100 * (sub[VAL] == SUPPORTS).mean(), 1),
            "mean_consideration_chars": round(sub[CON].str.len().mean(), 1),
            "mean_situation_chars": round(sub[SIT].str.len().mean(), 1),
            "vrd_mix": {k: round(100 * v, 1)
                        for k, v in sub["vrd"].value_counts(normalize=True).items()},
        }
    print(pd.DataFrame(comp).to_string())
    out["retained_vs_discarded"] = comp

    deltas = [abs(comp["retained"]["pct_supports"] - comp["discarded"]["pct_supports"])]
    print("\nverdict:", "no material skew detected" if max(deltas) < 5
          else f"CHECK — supports rate differs by {max(deltas):.1f} points")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
