"""
Split stress test for ValuePrism — does the split actually block the shortcuts?

The point: you can demonstrate leakage, and demonstrate that a split blocks it,
using TEXT-ONLY baselines. No activations, no GPU, no model downloads. The
activation probe slots into the same harness later as one more scorer.

Two experiments.

PART A — split stress test.
  Build the same probe under six leakage regimes, from "random rows" to
  "neither the situation nor the semantic consideration cluster appears in
  training", and evaluate every one with the same within-situation metric.
  If the text baselines look strong on the naive split and collapse on the
  strict split, the naive split contained exploitable leakage. That is a
  measured claim, not an assertion.

PART B — controlled leakage injection.
  Hold ONE strict test set fixed. Build four training sets of equal size:
      T0   no overlap
      TS   test situations reintroduced
      TC   near-duplicate test considerations reintroduced
      TSC  both
  Any performance rise is attributable to the reintroduced shortcut, because
  nothing else changed. This is stronger than comparing two split methods.

Metric throughout is within-situation pairwise accuracy:
      P( score(s, c_supports) > score(s, c_opposes) )
Because the situation is identical on both sides, a situation-only model scores
exactly 0.500 by construction. That is the harness's own sanity check.

Run:
    python split_stress_test.py
    python split_stress_test.py --max-train 40000 --out stress.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

DATASET, CONFIG = "allenai/ValuePrism", "valence"
DATASET_REVISION = "d439ca90825e5b4e5ef97798d9b5950e16ba7065"
SIT, CON, VAL = "situation", "text", "valence"
SUPPORTS, OPPOSES = "supports", "opposes"
SEED = 0


# ==========================================================================
# normalization ladder
# ==========================================================================

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_ARTICLE = re.compile(r"^(the|a|an)\s+")
# VRD type is baked into the string ("Duty to respect X", "Right to life").
# It encodes the TYPE, not the content, so strip it before comparing content.
_VRD_PREFIX = re.compile(
    r"^(the\s+)?(duty|right|value)\s+(to|of|for|not\s+to)\s+", re.I
)


def l1_normalize(text: str) -> str:
    """Case, punctuation, whitespace, leading article, crude plural."""
    t = _PUNCT.sub(" ", str(text).lower().strip())
    t = _WS.sub(" ", t).strip()
    t = _ARTICLE.sub("", t)
    parts = t.split()
    if parts and len(parts[-1]) > 3 and parts[-1].endswith("s") and not parts[-1].endswith("ss"):
        parts[-1] = parts[-1][:-1]
    return " ".join(parts)


def l2_normalize(text: str) -> str:
    """L1 + strip the Value/Right/Duty prefix.

    The prefix encodes the TYPE, not the content, so 'Right to privacy' and
    'privacy' should compare as the same consideration.

    NOTE: an earlier version also sorted tokens. That was a mistake — sorting
    scatters short function words ('to', 'of', 'and') into positions where
    character n-grams read them as similarity, which fed runaway merging. Token
    sorting is available separately as l2_sorted() for exact-match rules only,
    never as input to the fuzzy clustering.
    """
    t = _VRD_PREFIX.sub("", str(text).strip())
    return l1_normalize(t)


def l2_sorted(text: str) -> str:
    """Token-sorted L2. Safe for exact-match grouping ('respect for privacy'
    == 'privacy respect'), unsafe as input to character-n-gram similarity."""
    return " ".join(sorted(l2_normalize(text).split()))


def l3_cluster(
    forms: list[str],
    counts: dict[str, int] | None = None,
    threshold: float = 0.85,
    k: int = 20,
    max_cluster: int = 25,
) -> dict[str, int]:
    """Leader (canopy) clustering over L2 forms — no transitive chaining.

    Why not connected components: A~B and B~C merges A with C even when A and C
    are unrelated. On this data that produced a single cluster holding 45% of
    all considerations. Here every member is compared to its cluster's LEADER
    only, so membership requires direct similarity, and a hard size cap bounds
    the damage from any remaining over-merging.

    Frequent forms become leaders first, so clusters form around the canonical
    surface form rather than a rare typo.

    Char n-grams catch orthographic near-duplicates. They will NOT catch
    semantic synonyms with no shared substrings ('respecting privacy' vs
    'confidentiality'). A sentence-embedding pass is the upgrade; this is the
    no-dependency floor.
    """
    counts = counts or {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(forms)
    nn = NearestNeighbors(n_neighbors=min(k, len(forms)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    order = sorted(range(len(forms)), key=lambda i: -counts.get(forms[i], 0))
    label = [-1] * len(forms)
    size: dict[int, int] = {}
    next_id = 0

    leader_of: dict[int, int] = {}
    for i in order:
        if label[i] != -1:
            continue
        cid = next_id
        next_id += 1
        label[i] = cid
        leader_of[cid] = i
        size[cid] = 1
        # only DIRECT neighbours of this leader may join — this is what
        # prevents chaining
        for d, j in zip(dist[i], idx[i]):
            if j == i or label[j] != -1:
                continue
            if (1.0 - d) >= threshold and size[cid] < max_cluster:
                label[j] = cid
                size[cid] += 1

    # ---- ONE bounded merge pass over cluster leaders ----
    # Pure leader clustering trades chaining for MISSES: two genuine duplicates
    # split whenever neither is in the other's neighbour list. The missed-merge
    # audit measured that at 6.0% of test rows, with cases like
    # 'respect others property' vs 'respect for others property' at 0.959.
    #
    # One pass only, leaders compared to leaders, size cap still enforced. That
    # recovers direct misses without reopening the transitive-chaining hole,
    # because a merged cluster's leader does not then go looking for more.
    merged_into: dict[int, int] = {}

    def root(c: int) -> int:
        while c in merged_into:
            c = merged_into[c]
        return c

    for cid, li in sorted(leader_of.items(), key=lambda kv: -size[kv[0]]):
        if cid in merged_into:
            continue
        for d, j in zip(dist[li], idx[li]):
            other = root(label[j])
            if other == root(cid) or (1.0 - d) < threshold:
                continue
            a, b = root(cid), other
            if size[a] + size[b] <= max_cluster:
                merged_into[b] = a
                size[a] += size[b]

    return {f: int(root(l)) for f, l in zip(forms, label)}


# ==========================================================================
# data
# ==========================================================================

def load_binary() -> pd.DataFrame:
    from datasets import load_dataset

    print(f"loading {DATASET} [{CONFIG}] ...")
    revision = os.environ.get("VALUEPRISM_DATASET_REVISION", DATASET_REVISION)
    ds = load_dataset(DATASET, CONFIG, revision=revision)
    df = pd.concat([s.to_pandas() for s in ds.values()], ignore_index=True)
    df[VAL] = df[VAL].astype(str).str.strip().str.lower()
    df = df[df[VAL].isin([SUPPORTS, OPPOSES])].copy()

    df["l1"] = df[CON].map(l1_normalize)
    df["l2"] = df[CON].map(l2_normalize)

    forms = sorted(df["l2"].unique())
    counts = df["l2"].value_counts().to_dict()
    print(f"clustering {len(forms):,} L2 forms (char n-gram, leader clustering) ...")
    mapping = l3_cluster(forms, counts=counts)
    df["l3"] = df["l2"].map(mapping)

    print(f"  L0 exact         {df[CON].nunique():,}")
    print(f"  L1 normalized    {df['l1'].nunique():,}")
    print(f"  L2 prefix-strip  {df['l2'].nunique():,}")
    print(f"  L3 clustered     {df['l3'].nunique():,}")
    biggest = df.groupby("l3")["l2"].nunique().max()
    print(f"  largest cluster  {biggest} forms  (cap is 25)")
    return df


# ==========================================================================
# splits
# ==========================================================================

def make_split(df: pd.DataFrame, condition: str, rng: np.random.Generator):
    """Return (train, test, n_discarded). Test rows always come from
    situations carrying both valences, so the within-situation metric is
    defined on them."""
    bv_sits = {
        s for s, g in df.groupby(SIT)[VAL]
        if SUPPORTS in set(g) and OPPOSES in set(g)
    }
    pool = df[df[SIT].isin(bv_sits)]

    if condition == "random":
        mask = rng.random(len(df)) < 0.2
        test = df[mask & df[SIT].isin(bv_sits)]
        return df[~mask], test, 0

    if condition == "situation":
        sits = np.array(sorted(bv_sits))
        test_sits = set(rng.choice(sits, size=int(0.2 * len(sits)), replace=False))
        return df[~df[SIT].isin(test_sits)], pool[pool[SIT].isin(test_sits)], 0

    key = {"consid_exact": CON, "consid_norm": "l1", "consid_semantic": "l3"}.get(condition)
    if key is not None:
        keys = np.array(sorted(df[key].unique().tolist()))
        test_keys = set(rng.choice(keys, size=int(0.25 * len(keys)), replace=False))
        train = df[~df[key].isin(test_keys)]
        test = pool[pool[key].isin(test_keys)]
        return train, test, 0

    if condition == "strict":
        # bipartite: hold out BOTH endpoints. Rows with one endpoint on each
        # side must be discarded — that discard count is the real cost.
        keys = np.array(sorted(df["l3"].unique().tolist()))
        test_keys = set(rng.choice(keys, size=int(0.25 * len(keys)), replace=False))
        sits = np.array(sorted(bv_sits))
        test_sits = set(rng.choice(sits, size=int(0.35 * len(sits)), replace=False))

        in_ts, in_tk = df[SIT].isin(test_sits), df["l3"].isin(test_keys)
        train = df[~in_ts & ~in_tk]
        test = df[in_ts & in_tk]
        discarded = int((in_ts ^ in_tk).sum())
        return train, test, discarded

    raise ValueError(condition)


def assert_no_leak(train: pd.DataFrame, test: pd.DataFrame, condition: str) -> list[str]:
    """The split's own tests. A split you didn't assert is a split you don't have."""
    problems = []
    checks = {
        "situation": [(SIT, "situation")],
        "consid_exact": [(CON, "exact consideration")],
        "consid_norm": [("l1", "normalized consideration")],
        "consid_semantic": [("l3", "semantic cluster")],
        "strict": [(SIT, "situation"), ("l3", "semantic cluster")],
    }.get(condition, [])
    for col, label in checks:
        overlap = set(train[col]) & set(test[col])
        if overlap:
            problems.append(f"{label} overlap: {len(overlap):,}")
    return problems


# ==========================================================================
# text-only scorers
# ==========================================================================

def _fit_lr(train_text, y, test_text):
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000
    )
    Xtr = vec.fit_transform(train_text)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, y)
    return clf.predict_proba(vec.transform(test_text))[:, 1]


def score_model(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y = (train[VAL] == SUPPORTS).astype(int).values

    if name == "consideration_only":
        return _fit_lr(train[CON], y, test[CON])

    if name == "situation_only":
        return _fit_lr(train[SIT], y, test[SIT])

    if name == "pair_text":
        return _fit_lr(
            train[SIT] + " || " + train[CON], y, test[SIT] + " || " + test[CON]
        )

    if name == "nn_consideration":
        # copy the label tendency of the most lexically similar training
        # consideration — the memorization baseline, in its purest form
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        Xtr = vec.fit_transform(train[CON])
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xtr)
        _, idx = nn.kneighbors(vec.transform(test[CON]))
        return y[idx[:, 0]].astype(float)

    raise ValueError(name)


def within_situation_accuracy(test: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    """P(score(supporting) > score(opposing)) over all within-situation pairs.
    Situation-only models score exactly 0.5 here by construction."""
    t = test.copy()
    t["__score"] = scores
    wins = total = 0.0
    for _, g in t.groupby(SIT):
        sup = g.loc[g[VAL] == SUPPORTS, "__score"].values
        opp = g.loc[g[VAL] == OPPOSES, "__score"].values
        if len(sup) == 0 or len(opp) == 0:
            continue
        diff = sup[:, None] - opp[None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += diff.size
    return (wins / total if total else float("nan")), int(total)


# ==========================================================================
# main
# ==========================================================================

CONDITIONS = [
    ("random", "random rows — situations AND considerations cross splits"),
    ("situation", "no exact situation crosses"),
    ("consid_exact", "no exact consideration string crosses"),
    ("consid_norm", "no normalized consideration crosses"),
    ("consid_semantic", "no semantic cluster crosses"),
    ("strict", "neither situation nor semantic cluster crosses"),
]
MODELS = ["consideration_only", "situation_only", "pair_text", "nn_consideration"]


def cap(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    return df if len(df) <= n else df.iloc[rng.choice(len(df), n, replace=False)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=60_000)
    ap.add_argument("--max-test", type=int, default=25_000)
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_binary()
    results: dict = {"part_a": {}, "part_b": {}}

    # ---------------- PART A ----------------
    print("\n" + "=" * 78)
    print("PART A — SPLIT STRESS TEST   (within-situation pairwise accuracy)")
    print("=" * 78)
    header = f"{'condition':<18}" + "".join(f"{m[:15]:>17}" for m in MODELS) + f"{'test pairs':>12}"
    print(header)
    print("-" * len(header))

    for cond, desc in CONDITIONS:
        train, test, discarded = make_split(df, cond, np.random.default_rng(SEED))
        problems = assert_no_leak(train, test, cond)
        train = cap(train, args.max_train, rng)
        test = cap(test, args.max_test, rng)

        row, entry = {}, {"description": desc, "n_train": len(train), "n_test": len(test),
                          "n_discarded": discarded, "leak_assertions": problems or "clean"}
        for m in MODELS:
            try:
                acc, pairs = within_situation_accuracy(test, score_model(m, train, test))
            except Exception as e:                       # noqa: BLE001
                acc, pairs = float("nan"), 0
                entry.setdefault("errors", {})[m] = str(e)
            row[m] = acc
            entry[m] = acc
            entry["test_pairs"] = pairs

        results["part_a"][cond] = entry
        line = f"{cond:<18}" + "".join(f"{row[m]:>17.3f}" for m in MODELS)
        print(line + f"{entry.get('test_pairs', 0):>12,}")
        if problems:
            print(f"{'':<18}  !! LEAK: {problems}")
        if discarded:
            print(f"{'':<18}  discarded {discarded:,} rows (one endpoint each side)")

    print("\nread: situation_only must be 0.500 everywhere (harness sanity check).")
    print("      consideration_only and nn_consideration falling as you move down")
    print("      the table IS the leakage, measured.")

    # ---------------- PART B ----------------
    print("\n" + "=" * 78)
    print("PART B — CONTROLLED LEAKAGE INJECTION   (one fixed strict test set)")
    print("=" * 78)

    train0, test, _ = make_split(df, "strict", np.random.default_rng(SEED))
    test = cap(test, args.max_test, rng)
    test_sits, test_clusters = set(test[SIT]), set(test["l3"])

    pool_s = df[df[SIT].isin(test_sits) & ~df["l3"].isin(test_clusters)]
    pool_c = df[~df[SIT].isin(test_sits) & df["l3"].isin(test_clusters)]

    n = min(len(train0), args.max_train)
    print(f"training-set size held constant at {n:,} rows for every condition\n")

    def build(extra: pd.DataFrame, frac: float) -> pd.DataFrame:
        k = min(len(extra), int(frac * n))
        add = extra.iloc[rng.choice(len(extra), k, replace=False)] if k else extra.iloc[:0]
        base = train0.iloc[rng.choice(len(train0), n - k, replace=False)]
        return pd.concat([base, add], ignore_index=True)

    variants = {
        "T0  no overlap": build(df.iloc[:0], 0.0),
        "TS  +test situations": build(pool_s, 0.30),
        "TC  +test considerations": build(pool_c, 0.30),
        "TSC +both": pd.concat(
            [build(pool_s, 0.15), build(pool_c, 0.15).iloc[: n // 2]], ignore_index=True
        ).iloc[:n],
    }

    hdr = f"{'training set':<28}" + "".join(f"{m[:15]:>17}" for m in MODELS)
    print(hdr)
    print("-" * len(hdr))
    for label, tr in variants.items():
        entry, row = {"n_train": len(tr)}, {}
        for m in MODELS:
            acc, _ = within_situation_accuracy(test, score_model(m, tr, test))
            row[m] = entry[m] = acc
        results["part_b"][label] = entry
        print(f"{label:<28}" + "".join(f"{row[m]:>17.3f}" for m in MODELS))

    print("\nread: a rise from T0 is attributable to the reintroduced shortcut,")
    print("      because training-set size and everything else is held fixed.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
