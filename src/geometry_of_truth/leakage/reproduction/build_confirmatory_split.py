"""Build deterministic strict, development, training, and checkerboard manifests.

Selection uses structural criteria only. Content hashes fix ordering and row
identity without consulting model or baseline performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from .split_stress_test import SIT, CON, VAL, SUPPORTS, OPPOSES, load_binary
from .checkerboard_eval import mine_exact_boards

TARGET_CONFIRMED = 800


def row_id(situation: str, text: str, valence: str, vrd: str) -> str:
    """Content-addressed, stable across reloads and dataset re-downloads."""
    h = hashlib.sha256("\x1f".join([situation, text, valence, vrd]).encode("utf-8"))
    return h.hexdigest()[:16]


def manifest_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()



# ==========================================================================
# checkerboard interaction contrast
# ==========================================================================

def interaction_contrast(board: dict, score_by_id: dict) -> float:
    """I_b = [f(s1,c1) - f(s1,c2)] + [f(s2,c2) - f(s2,c1)]

    Cells are arranged so both brackets should be POSITIVE for a model that
    tracks the relation. For any additive f(s,c) = a(s) + b(c):
        bracket 1 = b(c1) - b(c2)
        bracket 2 = b(c2) - b(c1)
        I_b       = 0        exactly, with a(s) cancelling inside each bracket

    So this is an exact interaction test like both-ways, but continuous -- it
    can detect a systematic relational shift when too few boards cross both
    decision boundaries for the binary rate to move.
    """
    return ((score_by_id[board["id_s1_c1"]] - score_by_id[board["id_s1_c2"]])
            + (score_by_id[board["id_s2_c2"]] - score_by_id[board["id_s2_c1"]]))


# ==========================================================================
# selection
# ==========================================================================

def select_boards(boards, frame, n_candidates, cluster_cap):
    """Greedy diversity-maximising selection over a content-hash ordering.

    The ordering is a sha256 of the board's four row ids: deterministic,
    reproducible, and provably unrelated to how any model scores the board.
    """
    for b in boards:
        b["_order"] = hashlib.sha256(
            "".join(sorted([b["id_s1_c1"], b["id_s1_c2"],
                            b["id_s2_c1"], b["id_s2_c2"]])).encode()).hexdigest()
    boards = sorted(boards, key=lambda b: b["_order"])

    used_pairs, used_sits = set(), set()
    cluster_use: Counter = Counter()
    vrd_use: Counter = Counter()
    chosen, rejected = [], Counter()

    for b in boards:
        pair = tuple(sorted((b["c1"], b["c2"])))
        if pair in used_pairs:
            rejected["duplicate consideration pair"] += 1
            continue
        if b["s1"] in used_sits or b["s2"] in used_sits:
            rejected["situation already used"] += 1
            continue
        if (cluster_use[b["cl1"]] >= cluster_cap
                or cluster_use[b["cl2"]] >= cluster_cap):
            rejected["cluster cap reached"] += 1
            continue

        chosen.append(b)
        used_pairs.add(pair)
        used_sits.update([b["s1"], b["s2"]])
        cluster_use[b["cl1"]] += 1
        cluster_use[b["cl2"]] += 1
        vrd_use[b["vrd1"]] += 1
        vrd_use[b["vrd2"]] += 1
        if len(chosen) >= n_candidates:
            break

    return chosen, rejected, vrd_use


def main() -> None:
    global TARGET_CONFIRMED
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=1090)
    ap.add_argument("--cluster-cap", type=int, default=14)
    ap.add_argument("--target", type=int, default=TARGET_CONFIRMED,
                    help="confirmed boards needed after human review")
    args = ap.parse_args()
    TARGET_CONFIRMED = args.target

    df = load_binary().reset_index(drop=True)
    df["row_id"] = [row_id(s, t, v, r) for s, t, v, r
                    in zip(df[SIT], df[CON], df[VAL], df["vrd"])]
    # ValuePrism contains literal duplicate rows -- identical situation,
    # consideration, valence and type. A repeated identical row is not new
    # evidence, so drop them. Recorded in the manifest for reproducibility.
    n_dupes = int(df["row_id"].duplicated().sum())
    df = df.drop_duplicates("row_id").reset_index(drop=True)
    dup = int(df["row_id"].duplicated().sum())
    print(f"\nrows {len(df):,} after dropping {n_dupes:,} exact duplicate rows")

    print("mining all exact boards dataset-wide ...")
    raw = mine_exact_boards(df, CON)
    print(f"  {len(raw):,} boards")

    boards = []
    for b in raw:
        r1, r2 = df.loc[b["i_s1_c1"]], df.loc[b["i_s1_c2"]]
        r3, r4 = df.loc[b["i_s2_c2"]], df.loc[b["i_s2_c1"]]
        boards.append({
            **b,
            "cl1": r1["l3"], "cl2": r2["l3"],
            "vrd1": r1["vrd"], "vrd2": r2["vrd"],
            "id_s1_c1": r1["row_id"], "id_s1_c2": r2["row_id"],
            "id_s2_c2": r3["row_id"], "id_s2_c1": r4["row_id"],
        })

    chosen, rejected, vrd_use = select_boards(
        boards, df, args.candidates, args.cluster_cap)

    print("\n" + "=" * 74)
    print("SELECTION  (structural criteria only -- no model scores consulted)")
    print("=" * 74)
    print(f"selected {len(chosen):,} candidate boards "
          f"(target {TARGET_CONFIRMED} after human review)")
    for k, v in rejected.most_common():
        print(f"  rejected -- {k}: {v:,}")
    print(f"\nunique situations used: {len({s for b in chosen for s in (b['s1'], b['s2'])}):,}")
    print(f"unique consideration pairs: {len({tuple(sorted((b['c1'], b['c2']))) for b in chosen}):,}")
    print(f"VRD mix across board cells: {dict(vrd_use)}")

    # ---------------- manifests ----------------
    rows = []
    for rank, b in enumerate(chosen):
        rows.append({
            # Rank fixes review order. Rejections are backfilled in the same order.
            "rank": rank,
            "audit_priority": ("tier1_audit_first" if rank < TARGET_CONFIRMED
                               else "tier2_backfill"),
            "pool": "primary" if rank < TARGET_CONFIRMED else "reserve",
            "board_id": b["_order"][:12],
            "consideration_A": b["c1"], "consideration_B": b["c2"],
            "situation_1": b["s1"], "situation_2": b["s2"],
            "cluster_A": b["cl1"], "cluster_B": b["cl2"],
            "type_A": b["vrd1"], "type_B": b["vrd2"],
            "row_id_s1_A": b["id_s1_c1"], "row_id_s1_B": b["id_s1_c2"],
            "row_id_s2_A": b["id_s2_c1"], "row_id_s2_B": b["id_s2_c2"],
        })
    conf = pd.DataFrame(rows)
    conf.to_csv("manifest_confirmatory.csv", index=False, encoding="utf-8-sig")

    board_row_ids = set(conf[["row_id_s1_A", "row_id_s1_B",
                              "row_id_s2_A", "row_id_s2_B"]].values.ravel())
    conf_sits = set(conf["situation_1"]) | set(conf["situation_2"])
    conf_clusters = set(conf["cluster_A"]) | set(conf["cluster_B"])

    # Exclude board rows and situations from checkerboard training. Fixed consideration effects cancel in the interaction contrast, so consideration clusters may overlap.
    train = df[~df["row_id"].isin(board_row_ids) & ~df[SIT].isin(conf_sits)]
    train[["row_id"]].to_csv("manifest_train.csv", index=False)

    # Development boards exclude candidate situations.
    dev = [b for b in boards
           if b["s1"] not in conf_sits and b["s2"] not in conf_sits]
    pd.DataFrame([{"board_id": b["_order"][:12],
                   "consideration_A": b["c1"], "consideration_B": b["c2"],
                   "situation_1": b["s1"], "situation_2": b["s2"]}
                  for b in dev]).to_csv("manifest_dev.csv", index=False,
                                        encoding="utf-8-sig")

    # Hold out both consideration clusters and situations for the strict test.
    rng_strict = np.random.default_rng(20260803)
    all_clusters = np.array(sorted(df["l3"].unique()))
    held = set(rng_strict.choice(all_clusters, int(0.25 * len(all_clusters)),
                                 replace=False))
    bv_sits = {s for s, g in df.groupby(SIT)[VAL]
               if SUPPORTS in set(g) and OPPOSES in set(g)}
    sits = np.array(sorted(bv_sits))
    held_sits = set(rng_strict.choice(sits, int(0.30 * len(sits)), replace=False))

    in_s, in_c = df[SIT].isin(held_sits), df["l3"].isin(held)
    strict_test = df[in_s & in_c]
    strict_train = df[~in_s & ~in_c]
    strict_test[["row_id"]].to_csv("manifest_strict_test.csv", index=False)
    strict_train[["row_id"]].to_csv("manifest_strict_train.csv", index=False)
    strict_ok = (len(set(strict_train["l3"]) & set(strict_test["l3"])) == 0
                 and len(set(strict_train[SIT]) & set(strict_test[SIT])) == 0)

    # Use one shared training manifest for both public analyses.
    common = df[
        ~df["l3"].isin(held)                    # strict-test clusters out
        & ~df[SIT].isin(held_sits)              # strict-test situations out
        & ~df["row_id"].isin(board_row_ids)     # exact board cells out
        & ~df[SIT].isin(conf_sits)              # confirmatory situations out
    ]
    common[["row_id"]].to_csv("manifest_train_common.csv", index=False)
    common_ok = (
        len(set(common["l3"]) & set(strict_test["l3"])) == 0
        and len(set(common["row_id"]) & board_row_ids) == 0
        and len(set(common[SIT]) & conf_sits) == 0
    )
    # cluster overlap with the checkerboards is expected and permitted
    cb_cluster_overlap = len(set(common["l3"]) & conf_clusters)

    meta = {
        "schema_version": 1,
        "duplicate_rows_dropped": n_dupes,
        "dataset": "allenai/ValuePrism",
        "config": "valence",
        "cleaning_rule_version": "l1+l2(prefix-strip)+l3(leader,cap25,merge-pass)",
        "common_training_manifest": {
            "purpose": "one shared training set for the strict split and checkerboard analyses",
            "n_rows": int(len(common)),
            "excludes": [
                "strict-test consideration clusters",
                "strict-test situations",
                "checkerboard cells",
                "checkerboard situations",
            ],
            "permits": "other rows sharing a checkerboard consideration cluster",
            "checkerboard_cluster_overlap": cb_cluster_overlap,
            "assertions_pass": bool(common_ok),
            "hash_rows": manifest_hash(common["row_id"]),
        },
        "strict_generalization": {
            "n_test_rows": int(len(strict_test)),
            "n_train_rows": int(len(strict_train)),
            "cluster_overlap_prohibited": True,
            "situation_overlap_prohibited": True,
            "assertions_pass": bool(strict_ok),
            "hash_test_rows": manifest_hash(strict_test["row_id"]),
        },
        "candidate_review": {
            "selection_rule": (
                "Review candidates in deterministic rank order and accept passing boards "
                "in that order until the target is reached."
            ),
            "candidates": len(chosen),
            "target": TARGET_CONFIRMED,
            "max_tolerable_rejection_rate": round(
                1 - TARGET_CONFIRMED / max(len(chosen), 1), 3
            ),
        },
        "target_confirmed_boards": TARGET_CONFIRMED,
        "candidates_selected": len(chosen),
        "cluster_cap": args.cluster_cap,
        "boards_available_dataset_wide": len(boards),
        "n_train_rows": int(len(train)),
        "n_dev_boards": len(dev),
        "hash_confirmatory_rows": manifest_hash(board_row_ids),
        "hash_train_rows": manifest_hash(train["row_id"]),
    }
    with open("MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 74)
    print("MANIFESTS WRITTEN")
    print("=" * 74)
    print(f"  manifest_confirmatory.csv  {len(conf):,} boards "
          f"({TARGET_CONFIRMED} primary + {len(conf)-TARGET_CONFIRMED} reserve)")
    print(f"  manifest_train.csv         {len(train):,} rows")
    print(f"  manifest_dev.csv           {len(dev):,} development boards")
    print(f"  MANIFEST.json              hashes + fixed parameters")
    print(f"\n  confirmatory row hash  {meta['hash_confirmatory_rows'][:32]}...")
    print(f"  train row hash         {meta['hash_train_rows'][:32]}...")

    # ---------------- contract assertions ----------------
    print("\n" + "=" * 74)
    print("LEAKAGE CONTRACT")
    print("=" * 74)
    tr = df[df["row_id"].isin(set(train["row_id"]))]
    checks = [
        ("no confirmatory row in train", len(set(tr["row_id"]) & board_row_ids) == 0),
        ("no confirmatory situation in train", len(set(tr[SIT]) & conf_sits) == 0),
        ("no duplicate row_ids", dup == 0),
        ("every board cell resolves to a real row",
         board_row_ids <= set(df["row_id"])),
        ("one board per consideration pair",
         len({tuple(sorted((b['c1'], b['c2']))) for b in chosen}) == len(chosen)),
        ("no situation reused across boards",
         len({s for b in chosen for s in (b['s1'], b['s2'])}) == 2 * len(chosen)),
        (f"candidates >= target ({TARGET_CONFIRMED})", len(chosen) >= TARGET_CONFIRMED),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    overlap = len(set(tr["l3"]) & conf_clusters)
    print(f"  INFO  consideration-cluster overlap with train: {overlap:,} clusters "
          f"(permitted by design)")
    if not all(ok for _, ok in checks):
        raise SystemExit("\nCONTRACT VIOLATED. Manifests are not usable")
    print("\nall contract assertions hold.")
    print("Candidate boards require human review before analysis.")


if __name__ == "__main__":
    main()
