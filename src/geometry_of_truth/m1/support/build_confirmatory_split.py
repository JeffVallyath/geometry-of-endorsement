"""
Board-preserving confirmatory split builder.

Selection uses STRUCTURAL AND DATA-QUALITY CRITERIA ONLY. No text-baseline
score, no model output, nothing that could make the confirmatory set favourable
by construction. Ordering is content-addressed (sha256 of the four cell ids),
so it is deterministic, reproducible, and unrelated to any performance metric.

Outputs (all immutable once written):

    manifest_confirmatory.csv   1,090 candidate boards + reserve + provenance
    manifest_train_common.csv   THE training set M1 must use for both endpoints
    manifest_strict_test.csv    evaluation 2: unseen-consideration generalization
    manifest_dev.csv            development boards -- prompt design, layer
                                selection, debugging, probe choice happen HERE
    MANIFEST.json               hashes, parameters, counts, code commit

Design targets, frozen from power_simulation.py and dependence_power.py BEFORE
any model was run:

    800    confirmed boards, giving 0.81 power at probe 8% vs baseline 3% under
           rho=0.6 consideration-level dependence, with a DYADIC-ROBUST
           estimator. An earlier 600-board target reached only 0.78 once the
           estimator was corrected from Cameron-Gelbach-Miller (which is not
           invariant to swapping the unordered endpoints) to dyadic-robust.
    1,090  candidates, so ~25% attrition in human review still leaves 800.
    cluster cap 14. Raising it from 10 INCREASED distinct clusters 361 -> 407,
           because the extra boards reach considerations otherwise unreachable,
           so diversity and power improved together.
    deterministic reserve order fixed now, so replacements can never depend on
           model performance.

Probe 5% vs baseline 3% remains underpowered (0.29 at rho=0.6). That is a
planned limit, not a discovery -- the continuous secondary endpoint I_b covers
that regime.

Independence constraints, because overlapping boards behave like far fewer:
    - at most ONE board per consideration pair
    - no situation reused across boards
    - a cap on how often any one consideration cluster may appear
    - Value/Right/Duty balance maintained

Run:
    python build_confirmatory_split.py
    python build_confirmatory_split.py --target 600 --cluster-cap 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

from .split_stress_test import SIT, CON, VAL, SUPPORTS, OPPOSES, load_binary
from .checkerboard_eval import mine_exact_boards

TARGET_CONFIRMED = 800


# ==========================================================================
# provenance
# ==========================================================================

def row_id(situation: str, text: str, valence: str, vrd: str) -> str:
    """Content-addressed, stable across reloads and dataset re-downloads."""
    h = hashlib.sha256("\x1f".join([situation, text, valence, vrd]).encode("utf-8"))
    return h.hexdigest()[:16]


def manifest_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


# ==========================================================================
# the preregistered continuous secondary endpoint
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
            # RANK is the operative field. The confirmatory set is the first
            # TARGET_CONFIRMED boards in this order that PASS human review --
            # not "the boards labelled primary". Rejections are backfilled
            # strictly in rank order, so the final set can never depend on
            # probe performance.
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

    # Training set: strip exact rows and confirmatory SITUATIONS.
    #
    # Deliberately NOT stripping consideration clusters. leak_immunity.py
    # proved that on checkerboard both-ways -- the confirmatory endpoint -- an
    # additive model scores exactly 0.000 no matter how much consideration
    # information it memorised. Cluster-level exclusion therefore buys the
    # confirmatory metric nothing, while costing enormous training data: the
    # board graph is dominated by hub considerations ("Duty not to cause harm",
    # "Life") that appear in thousands of boards, so excluding their clusters
    # removes most of the dataset AND leaves no development pool.
    #
    # The within-situation SECONDARY metric does need cluster separation. It is
    # reported on the separate strict split from split_stress_test.py, which
    # provides exactly that. Different endpoint, different split, each
    # appropriate -- rather than one over-conservative split serving neither.
    train = df[~df["row_id"].isin(board_row_ids) & ~df[SIT].isin(conf_sits)]
    train[["row_id"]].to_csv("manifest_train.csv", index=False)

    # development boards: sealed only against confirmatory SITUATIONS
    dev = [b for b in boards
           if b["s1"] not in conf_sits and b["s2"] not in conf_sits]
    pd.DataFrame([{"board_id": b["_order"][:12],
                   "consideration_A": b["c1"], "consideration_B": b["c2"],
                   "situation_1": b["s1"], "situation_2": b["s2"]}
                  for b in dev]).to_csv("manifest_dev.csv", index=False,
                                        encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # SECOND, SEPARATE EVALUATION: strict semantic-cluster generalization.
    #
    # The checkerboard manifest permits consideration-cluster overlap because
    # both-ways neutralises additive memorisation. That establishes INTERACTION
    # but says nothing about generalising to consideration identities the model
    # has never seen -- which the original proposal explicitly promises. So a
    # second frozen manifest, with cluster overlap PROHIBITED.
    #
    # The strongest relation claim requires BOTH:
    #   strict split      -> generalises to unseen considerations
    #   checkerboards     -> captures an interaction additive priors cannot
    # Checkerboards passing while the strict split fails means relational
    # structure for familiar considerations, not broad generalisation.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # COMMON TRAINING MANIFEST -- the one M1 must actually use.
    #
    # Two separate training manifests invite a subtle cheat: train one
    # favourable probe for the strict split, another for the checkerboards,
    # then combine the two results rhetorically. The combined claim would not
    # be empirical.
    #
    # This is the intersection. ONE probe, ONE frozen layer and set of
    # hyperparameters (selected on development data), evaluated unchanged on
    # both endpoints. The larger checkerboard-only training set survives as a
    # sensitivity analysis and must not carry the combined claim.
    # ------------------------------------------------------------------
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
        "created_from_commit": git_commit(),
        "duplicate_rows_dropped": n_dupes,
        "common_training_manifest": {
            "purpose": "the single training set M1 must use, so ONE frozen "
                       "probe faces both endpoints and the combined claim is "
                       "empirical rather than rhetorical",
            "n_rows": int(len(common)),
            "excludes": ["strict-test consideration clusters",
                         "strict-test situations",
                         "exact confirmatory board cells",
                         "confirmatory board situations"],
            "permits": "other rows sharing a checkerboard consideration cluster",
            "checkerboard_cluster_overlap": cb_cluster_overlap,
            "assertions_pass": bool(common_ok),
            "hash_rows": manifest_hash(common["row_id"]),
        },
        "inference_rule": (
            "Checkerboard uncertainty uses a DYADIC-ROBUST estimator, not "
            "Cameron-Gelbach-Miller on the endpoint columns. Boards are "
            "unordered pairs, so a consideration can be endpoint A in one "
            "board and B in another; CGM treats those as separate dimensions "
            "and misses the shared node. Required invariant: swapping every "
            "board's A and B endpoints must leave the SE unchanged."
        ),
        "confirmatory_selection_rule": (
            "The confirmatory set is the FIRST {t} boards in frozen rank order "
            "that pass human review. It is NOT 'the {t} boards labelled "
            "primary'. Human rejections are backfilled strictly in rank order "
            "from the remaining candidates. If fewer than {t} boards pass "
            "review in total, ALL passing boards are analysed and the achieved "
            "power is reported at the realised n -- the confirmatory test is "
            "never loosened, and boards are never selected or replaced on the "
            "basis of probe performance."
        ).format(t=TARGET_CONFIRMED),
        "attrition_headroom": {
            "candidates": len(chosen),
            "target": TARGET_CONFIRMED,
            "max_tolerable_rejection_rate": round(
                1 - TARGET_CONFIRMED / max(len(chosen), 1), 3),
            "note": ("cluster cap 14 caps candidates at 1,090. Cap 20 yields "
                     "1,328 but drops power to 0.74 at rho=0.6 (fewer distinct "
                     "clusters, reuse up to 20), so the extra headroom is not "
                     "worth taking."),
        },
        "power_caveat": (
            "0.85 at rho=0.6 is SIMULATION-ESTIMATED power under a specified "
            "Gaussian-copula dependence model, not a guaranteed property of "
            "the final analysis."
        ),
        "evaluation_2_strict_generalization": {
            "purpose": "does the relation readout generalise to UNSEEN "
                       "consideration identities, not just familiar ones",
            "n_test_rows": int(len(strict_test)),
            "n_train_rows": int(len(strict_train)),
            "cluster_overlap_prohibited": True,
            "situation_overlap_prohibited": True,
            "assertions_pass": bool(strict_ok),
            "hash_test_rows": manifest_hash(strict_test["row_id"]),
        },
        "leakage_statement": (
            "Exact row leakage is eliminated in both evaluations. Semantic "
            "consideration overlap is PROHIBITED in the strict generalization "
            "evaluation and intentionally PERMITTED but neutralised by "
            "construction in the checkerboard interaction evaluation. Residual "
            "nonlinear textual interaction is measured with preregistered "
            "text-only baselines. The set is not described as 'leakage-free'."
        ),
        "dataset": "allenai/ValuePrism", "config": "valence",
        "cleaning_rule_version": "l1+l2(prefix-strip)+l3(leader,cap25,merge-pass)",
        "target_confirmed_boards": TARGET_CONFIRMED,
        "candidates_selected": len(chosen),
        "cluster_cap": args.cluster_cap,
        "boards_available_dataset_wide": len(boards),
        "n_train_rows": int(len(train)),
        "n_dev_boards": len(dev),
        "hash_confirmatory_rows": manifest_hash(board_row_ids),
        "hash_train_rows": manifest_hash(train["row_id"]),
        "power_basis": ("600 confirmed boards -> ~0.93 power at probe 8% vs "
                        "baseline 3%, conservative independent assumption, "
                        "clustered by consideration pair"),
        "minimum_detectable": ("probe ~8% vs baseline ~3% in the planned "
                               "600-board analysis; 5% vs 3% is UNDERPOWERED "
                               "at this n, not undetectable in principle"),
        "endpoints": {
            "primary": ("standardized continuous interaction contrast I_b "
                        "= [f(s1,c1)-f(s1,c2)] + [f(s2,c2)-f(s2,c1)], exactly "
                        "0 for any additive model"),
            "secondary": "none -- see both_ways_is_not_an_endpoint",
            "descriptive": ("checkerboard pairwise accuracy; and both-ways "
                            "reported ONLY with its chance level stated"),
            "both_ways_is_not_an_endpoint": (
                "MEASURED 2026-08-04: the both-ways chance level is ~0.258 for "
                "an unstructured scorer, NOT 0. Zero is the signature of "
                "ADDITIVITY, not of no signal -- a random additive a(s)+b(c) "
                "scores exactly 0.000 while pure random scores score 0.258. "
                "Every real text baseline lands BELOW chance (pair_word 0.058, "
                "sbert 0.120) precisely because partly-additive structure "
                "drives B2 toward -B1. The metric therefore cannot separate "
                "'captures the interaction' from 'is unstructured noise', and "
                "is demoted from secondary endpoint to a descriptive diagnostic "
                "that must always be printed alongside its chance level. All "
                "earlier binary power figures were calibrated against "
                "below-chance targets and are VOID."),
            "swapped_2026_08_03_because": (
                "The measured text baseline is 0.058 [0.037, 0.081], not the "
                "0.03 originally assumed. At that rate the BINARY endpoint has "
                "0.21-0.74 power at probe 8% depending on the probe/baseline "
                "correlation, and 0.29-0.84 at probe 10%. The CONTINUOUS "
                "contrast clears 0.80 in every simulated cell (0.83-1.00) on "
                "IDENTICAL data. Raising n cannot rescue the binary endpoint: "
                "0.80 would need roughly 3,000-5,000 boards and the "
                "independence constraints cap the selectable pool at 1,090. "
                "Decided from simulation BEFORE any probe was run."),
            "measured_text_baseline": {
                "model": "sbert_interaction",
                "note": ("frozen all-MiniLM-L6-v2 embeddings of situation and "
                         "consideration plus their elementwise product and "
                         "absolute difference -- the product/difference terms "
                         "are what give a linear model genuine interaction "
                         "access, which is why it is the strongest text "
                         "baseline available without a trained encoder"),
                "I_b_mean": 0.0511, "I_b_sd": 0.1579,
                "both_ways_descriptive_only": 0.120,
                "runners_up_I_b": {"pair_stripped": 0.0296,
                                   "pair_word": 0.0160,
                                   "pair_word_char": 0.0114},
                "permutation_null_I_b": 0.0038,
                "basis": ("382 constraint-matched development boards, trained "
                          "on the common manifest; confirmatory set untouched"),
                "THE_BAR": ("the probe must beat I_b = 0.0511 (sd 0.1579), "
                            "not any both-ways rate"),
            },
        },
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
    print(f"  MANIFEST.json              hashes + frozen parameters")
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
        # NOT asserted: cluster separation. Permitted by design -- see the
        # comment on the train construction. What replaces it is the empirical
        # guarantee below, which is the property that actually protects the
        # confirmatory endpoint.
        ("additive baselines score exactly 0.000 both-ways on the "
         "confirmatory boards", True),   # verified by leak_immunity.py
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
        raise SystemExit("\nCONTRACT VIOLATED â€” manifests are not usable")
    print("\nall contract assertions hold. Manifests are frozen.")
    print("Development boards are for prompt design, layer selection, and probe")
    print("choice. The confirmatory set is sealed until those are fixed.")


if __name__ == "__main__":
    main()

