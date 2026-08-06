"""Audit ValuePrism counts, reversals, normalization, and strict-set viability."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

DATASET = "allenai/ValuePrism"
CONFIG = "valence"          # ships train/val/test splits

# If auto-detection picks the wrong columns, hard-code them here.
COLUMN_OVERRIDES: dict[str, str] = {
    "situation": "situation",
    "consideration": "text",   # NOT "vrd" -- that's the Value/Right/Duty type (3 values)
    "valence": "valence",
    "vrd_type": "vrd",
}

SUPPORTS = "supports"
OPPOSES = "opposes"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    revision = os.environ.get("VALUEPRISM_DATASET_REVISION")
    if revision:
        print(f"loading {DATASET} [{CONFIG}] at pinned revision {revision} ...")
        ds = load_dataset(DATASET, CONFIG, revision=revision)
    else:
        print(f"loading {DATASET} [{CONFIG}] ...")
        ds = load_dataset(DATASET, CONFIG)
    frames = []
    for split_name, split in ds.items():
        df = split.to_pandas()
        df["__split"] = split_name
        frames.append(df)
        print(f"  {split_name}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Guess which columns hold situation / consideration / valence."""
    cols = {c.lower(): c for c in df.columns}

    def pick(role: str, candidates: list[str], fallback_contains: str) -> str:
        if role in COLUMN_OVERRIDES:
            return COLUMN_OVERRIDES[role]
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        for lower, original in cols.items():
            if fallback_contains in lower:
                return original
        sys.exit(
            f"could not find a '{role}' column. Columns present: {list(df.columns)}\n"
            f"Set it explicitly in COLUMN_OVERRIDES at the top of this file."
        )

    detected = {
        "situation": pick("situation", ["situation", "context", "scenario"], "situat"),
        "consideration": pick(
            "consideration", ["text", "consideration", "vrd_text", "value"], "text"
        ),
        "valence": pick("valence", ["valence", "label", "stance"], "valen"),
    }
    # sanity: the consideration column must have many distinct values, not 3
    n_distinct = df[detected["consideration"]].nunique()
    if n_distinct < 100:
        sys.exit(
            f"'{detected['consideration']}' has only {n_distinct} distinct values -- that is "
            f"almost certainly the Value/Right/Duty TYPE column, not the consideration text. "
            f"Set COLUMN_OVERRIDES['consideration'] explicitly. Columns: {list(df.columns)}"
        )
    return detected


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"^(the|a|an)\s+")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Deliberately simple and transparent. The point is to measure the delta,
    not to be the final near-duplicate rule."""
    t = str(text).lower().strip()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _ARTICLES.sub("", t)
    # crude plural strip on the final token only
    parts = t.split()
    if parts and len(parts[-1]) > 3 and parts[-1].endswith("s") and not parts[-1].endswith("ss"):
        parts[-1] = parts[-1][:-1]
    return " ".join(parts)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def valence_sets(df: pd.DataFrame, key: str, valence_col: str) -> dict:
    out = defaultdict(set)
    for k, v in zip(df[key], df[valence_col]):
        out[k].add(v)
    return out


def both_valence_keys(sets: dict) -> set:
    return {k for k, vs in sets.items() if SUPPORTS in vs and OPPOSES in vs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write results as JSON")
    args = ap.parse_args()

    df = load()
    cols = detect_columns(df)
    print(f"\ncolumns -> {cols}")
    print(f"all columns: {list(df.columns)}")
    print("\nsample row:")
    print(df.iloc[0].to_dict())

    sit, con, val = cols["situation"], cols["consideration"], cols["valence"]

    # normalize valence labels to lowercase strings
    df[val] = df[val].astype(str).str.strip().str.lower()

    results: dict = {}

    # ---------------- 1. headline reproduction ----------------
    print("\n" + "=" * 70)
    print("1. HEADLINE NUMBERS (exact string)")
    print("=" * 70)

    results["n_rows"] = len(df)
    results["n_situations"] = df[sit].nunique()
    results["n_considerations_exact"] = df[con].nunique()
    results["valence_counts"] = df[val].value_counts().to_dict()

    print(f"rows                    {results['n_rows']:,}   (expect 218,408)")
    print(f"unique situations       {results['n_situations']:,}   (expect 31,028)")
    print(f"unique considerations   {results['n_considerations_exact']:,}")
    print(f"valence counts          {results['valence_counts']}")
    print(f"annotations/situation   {results['n_rows'] / max(results['n_situations'], 1):.2f}")

    binary = df[df[val].isin([SUPPORTS, OPPOSES])].copy()
    results["n_rows_binary"] = len(binary)
    print(f"\nSupports/Opposes rows   {len(binary):,}  (Either excluded from here on)")

    sit_sets = valence_sets(binary, sit, val)
    bv_situations = both_valence_keys(sit_sets)
    results["n_both_valence_situations"] = len(bv_situations)
    pct = 100 * len(bv_situations) / max(results["n_situations"], 1)
    print(f"both-valence situations {len(bv_situations):,}  ({pct:.1f}%)   (expect 20,032 / 65%)")

    con_sets = valence_sets(binary, con, val)
    bv_considerations = both_valence_keys(con_sets)
    results["n_both_valence_considerations_exact"] = len(bv_considerations)
    print(f"both-valence considerations (exact)  {len(bv_considerations):,}   (expect 3,444)")

    # ---------------- 2. normalization delta ----------------
    print("\n" + "=" * 70)
    print("2. NORMALIZATION DELTA  (near-duplicate leakage risk)")
    print("=" * 70)

    binary["__norm"] = binary[con].map(normalize)
    results["n_considerations_normalized"] = binary["__norm"].nunique()

    norm_sets = valence_sets(binary, "__norm", val)
    bv_norm = both_valence_keys(norm_sets)
    results["n_both_valence_considerations_normalized"] = len(bv_norm)

    collapsed = results["n_considerations_exact"] - results["n_considerations_normalized"]
    print(f"considerations exact       {results['n_considerations_exact']:,}")
    print(f"considerations normalized  {results['n_considerations_normalized']:,}"
          f"   ({collapsed:,} exact strings collapsed)")
    print(f"both-valence exact         {len(bv_considerations):,}")
    print(f"both-valence normalized    {len(bv_norm):,}")

    # how many normalized clusters contain >1 distinct exact string?
    cluster_sizes = binary.groupby("__norm")[con].nunique()
    multi = cluster_sizes[cluster_sizes > 1]
    results["n_norm_clusters_multi_surface"] = int(len(multi))
    print(f"\nnormalized clusters with >1 surface form: {len(multi):,}")
    print("  -> these are the leakage risk. Grouping by exact string lets a")
    print("     paraphrase of a held-out consideration stay in training.")
    if len(multi):
        print("\n  10 largest clusters:")
        for norm_key in multi.sort_values(ascending=False).head(10).index:
            forms = sorted(binary.loc[binary["__norm"] == norm_key, con].unique())[:4]
            print(f"    '{norm_key}'  <- {forms}")

    # ---------------- 3. viability structure ----------------
    print("\n" + "=" * 70)
    print("3. STRICT WITHIN-SITUATION EVALUATION, does it survive?")
    print("=" * 70)
    print("A situation is usable if, holding it fixed, it has at least one")
    print("Supports and one Opposes row whose consideration is a held-out")
    print("both-valence string. Situation-only scores exactly 50% by")
    print("construction; lexical shortcuts are excluded by the split.\n")

    for label, bv_set, key_col in (
        ("exact-string grouping", bv_considerations, con),
        ("normalized grouping", bv_norm, "__norm"),
    ):
        sub = binary[binary[key_col].isin(bv_set)]
        usable, rows_in_usable, pair_count = [], 0, 0
        for s, grp in sub.groupby(sit):
            vs = set(grp[val])
            if SUPPORTS in vs and OPPOSES in vs:
                usable.append(s)
                rows_in_usable += len(grp)
                n_sup = int((grp[val] == SUPPORTS).sum())
                n_opp = int((grp[val] == OPPOSES).sum())
                pair_count += n_sup * n_opp

        key = "strict_" + label.split()[0]
        results[key] = {
            "usable_situations": len(usable),
            "rows": int(rows_in_usable),
            "within_situation_pairs": int(pair_count),
            "candidate_rows_before_situation_filter": int(len(sub)),
        }
        print(f"[{label}]")
        print(f"  rows with a held-out consideration      {len(sub):,}")
        print(f"  situations usable for within-situation  {len(usable):,}")
        print(f"  rows inside those situations            {rows_in_usable:,}")
        print(f"  supporting x opposing pairs             {pair_count:,}")
        print()

    # ---------------- summary ----------------
    strict_n = results["strict_normalized"]["usable_situations"]
    print("=" * 70)
    if strict_n >= 2000:
        verdict = f"HEALTHY - {strict_n:,} usable situations. Design proceeds as written."
    elif strict_n >= 500:
        verdict = (f"WORKABLE - {strict_n:,} usable situations. Enough for the primary "
                   f"evaluation but report per-situation CIs; consider relaxing the "
                   f"consideration hold-out for a secondary analysis.")
    else:
        verdict = (f"PROBLEM - only {strict_n:,} usable situations. The strict evaluation "
                   f"is underpowered. Options: relax to consideration-level hold-out "
                   f"without the within-situation constraint, or use the two controls "
                   f"as separate evaluations rather than intersected.")
    print("VERDICT:", verdict)
    print("=" * 70)
    results["verdict"] = verdict

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
