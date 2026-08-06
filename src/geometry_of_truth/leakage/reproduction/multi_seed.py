"""Repeat fixed leakage interventions across situation and cluster split draws."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .split_stress_test import (
    SIT, CON, VAL, SUPPORTS, OPPOSES,
    load_binary, make_split, score_model, cap,
)
from .checkerboard_eval import mine_exact_boards, evaluate_boards

MODELS = ["consideration_only", "situation_only", "pair_text", "nn_consideration"]
MERGE_THRESHOLD = 0.85


def paired_acc(test, scores, fix):
    t = test.copy()
    t["__s"] = scores
    w = n = 0.0
    for _, g in t.groupby(fix):
        sup = g.loc[g[VAL] == SUPPORTS, "__s"].values
        opp = g.loc[g[VAL] == OPPOSES, "__s"].values
        if len(sup) and len(opp):
            d = sup[:, None] - opp[None, :]
            w += (d > 0).sum() + 0.5 * (d == 0).sum()
            n += d.size
    return w / n if n else float("nan")


def missed_merge_rate(df, train, test, vec, Xall, form_index):
    """Fraction of test rows whose consideration has a near-identical (>=0.85)
    counterpart on the train side sitting in a DIFFERENT cluster."""
    tr = train.groupby("l2").agg(cluster=("l3", "first")).reset_index()
    te = test.groupby("l2").agg(cluster=("l3", "first"), rows=("l2", "size")).reset_index()
    if tr.empty or te.empty:
        return float("nan")
    Xtr = Xall[[form_index[f] for f in tr["l2"]]]
    Xte = Xall[[form_index[f] for f in te["l2"]]]
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xtr)
    dist, idx = nn.kneighbors(Xte)
    sim = 1.0 - dist[:, 0]
    near_cluster = tr["cluster"].values[idx[:, 0]]
    miss = (sim >= MERGE_THRESHOLD) & (near_cluster != te["cluster"].values)
    return te.loc[miss, "rows"].sum() / te["rows"].sum()


def run_seed(df, seed, max_train, vec, Xall, form_index):
    rng = np.random.default_rng(seed)
    train, test, discarded = make_split(df, "strict", np.random.default_rng(seed))
    res = {
        "usable_situations": int(test[SIT].nunique()),
        "test_rows": int(len(test)),
        "discarded_rows": int(discarded),
        "missed_merge_rate": float(missed_merge_rate(df, train, test, vec, Xall, form_index)),
    }

    tr = cap(train, max_train, rng)
    te = test.reset_index(drop=True)
    boards = mine_exact_boards(te, CON)
    res["n_boards"] = len(boards)

    for m in MODELS:
        s = score_model(m, tr, te)
        res[f"within_sit__{m}"] = paired_acc(te, s, SIT)
        res[f"within_con__{m}"] = paired_acc(te, s, CON)
        res[f"both_ways__{m}"] = (evaluate_boards(boards, dict(enumerate(s)))["both"]
                                  if boards else float("nan"))

    # leakage injection, size- and composition-matched
    test_sits, test_clusters = set(test[SIT]), set(test["l3"])
    pool_s = df[df[SIT].isin(test_sits) & ~df["l3"].isin(test_clusters)]
    pool_c = df[~df[SIT].isin(test_sits) & df["l3"].isin(test_clusters)]
    n = len(tr)

    def build(specs):
        parts, taken = [], 0
        for pool, frac in specs:
            k = min(len(pool), int(frac * n))
            if k:
                parts.append(pool.iloc[rng.choice(len(pool), k, replace=False)])
                taken += k
        parts.append(tr.iloc[rng.choice(len(tr), n - taken, replace=False)])
        return pd.concat(parts, ignore_index=True)

    base = paired_acc(te, score_model("pair_text", build([]), te), SIT)
    res["leak_T0"] = base
    res["leak_delta_situations"] = paired_acc(
        te, score_model("pair_text", build([(pool_s, 0.30)]), te), SIT) - base
    res["leak_delta_considerations"] = paired_acc(
        te, score_model("pair_text", build([(pool_c, 0.30)]), te), SIT) - base
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=40_000)
    ap.add_argument("--out")
    args = ap.parse_args()

    df = load_binary()

    # one shared vectoriser so similarity is comparable across seeds
    forms = sorted(df["l2"].unique())
    form_index = {f: i for i, f in enumerate(forms)}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xall = vec.fit_transform(forms)

    rows = []
    for seed in range(args.seeds):
        print(f"seed {seed} ...", flush=True)
        r = run_seed(df, seed, args.max_train, vec, Xall, form_index)
        r["seed"] = seed
        rows.append(r)

    res = pd.DataFrame(rows).set_index("seed")

    def block(title, keys, fmt="{:.3f}"):
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)
        print(f"{'quantity':<34}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
        print("-" * 74)
        for k in keys:
            if k not in res:
                continue
            v = res[k].astype(float)
            print(f"{k:<34}{fmt.format(v.mean()):>10}{fmt.format(v.std(ddof=1)):>10}"
                  f"{fmt.format(v.min()):>10}{fmt.format(v.max()):>10}")

    block("SPLIT GEOMETRY", ["usable_situations", "test_rows", "discarded_rows",
                             "n_boards"], "{:,.0f}")
    block("RESIDUAL LEAKAGE IN THE SPLIT", ["missed_merge_rate"])
    block("LEAKAGE EFFECTS (pair_text, within-situation)",
          ["leak_T0", "leak_delta_situations", "leak_delta_considerations"])
    block("WITHIN-SITUATION", [f"within_sit__{m}" for m in MODELS])
    block("WITHIN-CONSIDERATION  (consideration_only must be 0.500 every seed)",
          [f"within_con__{m}" for m in MODELS])
    block("CHECKERBOARD BOTH-WAYS  (additive baselines must be 0.000 every seed)",
          [f"both_ways__{m}" for m in MODELS])

    print("\n" + "=" * 76)
    print("INVARIANT CHECK ACROSS ALL SEEDS")
    print("=" * 76)
    ok = True
    for k, want in [("within_sit__situation_only", 0.5),
                    ("within_con__consideration_only", 0.5),
                    ("both_ways__consideration_only", 0.0),
                    ("both_ways__situation_only", 0.0),
                    ("both_ways__nn_consideration", 0.0)]:
        v = res[k].astype(float)
        good = bool(np.allclose(v, want, atol=1e-9))
        ok &= good
        print(f"  {k:<38} == {want}   {'PASS' if good else 'FAIL ' + str(v.tolist())}")
    print("\nall invariants hold on every seed" if ok
          else "\nINVARIANT BROKEN, investigate before trusting anything above")

    if args.out:
        res.to_json(args.out, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
