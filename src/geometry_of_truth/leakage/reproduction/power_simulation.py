"""Simulate board-level power under pair clustering and scorer dependence."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .split_stress_test import CON, SEED, load_binary
from .checkerboard_eval import mine_exact_boards

BASELINE_RATES = [0.02, 0.03, 0.04]
PROBE_RATES = [0.05, 0.08, 0.10, 0.15]
N_BOARDS = [100, 150, 200, 250, 300, 400, 500, 600]
ALPHA = 0.05


def cluster_structure(df):
    """Real board -> consideration-pair mapping, so the simulation inherits the
    actual clustering rather than assuming one."""
    boards = mine_exact_boards(df.reset_index(drop=True), CON)
    pairs = {}
    for b in boards:
        key = tuple(sorted((b["c1"], b["c2"])))
        pairs.setdefault(key, 0)
        pairs[key] += 1
    sizes = np.array(sorted(pairs.values(), reverse=True))
    return len(boards), sizes


def simulate(n_target, p_base, p_probe, sizes, mode, rng, trials):
    """Return power: fraction of trials where a cluster-robust paired test
    rejects no-difference at alpha."""
    hits = 0
    for _ in range(trials):
        # resample consideration pairs (clusters) until we reach n_target boards
        picked, total = [], 0
        while total < n_target:
            k = sizes[rng.integers(0, len(sizes))]
            picked.append(min(k, n_target - total))
            total += picked[-1]

        diffs = []                       # per-cluster mean difference
        for m in picked:
            u = rng.random(m)            # shared board-difficulty draw
            base = u < p_base
            if mode == "nested":
                probe = u < p_probe      # probe solves a superset
            else:
                probe = rng.random(m) < p_probe   # independent
            diffs.append(probe.mean() - base.mean())

        d = np.array(diffs)
        if len(d) < 2:
            continue
        # cluster-robust SE of the mean difference, weighting clusters equally
        se = d.std(ddof=1) / np.sqrt(len(d))
        if se > 0 and abs(d.mean()) / se > 1.96:
            hits += 1
    return hits / trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_binary()
    n_avail, sizes = cluster_structure(df)
    print(f"\ndataset-wide exact boards: {n_avail:,}")
    print(f"distinct consideration pairs: {len(sizes):,}")
    print(f"boards per pair: mean {sizes.mean():.1f}  median {np.median(sizes):.0f}  "
          f"max {sizes.max()}")
    print(f"design effect from clustering is inherited from this distribution\n")

    out = {"n_available_boards": int(n_avail), "n_pairs": int(len(sizes))}

    for mode in ("independent", "nested"):
        print("=" * 78)
        print(f"POWER, {mode} probe/baseline "
              f"({'conservative, plan on this' if mode == 'independent' else 'optimistic bound'})")
        print("=" * 78)
        print(f"baseline fixed at {BASELINE_RATES[1]:.0%}; alpha = {ALPHA}")
        hdr = f"{'boards':>8}" + "".join(f"{f'probe {p:.0%}':>13}" for p in PROBE_RATES)
        print(hdr)
        print("-" * len(hdr))
        for n in N_BOARDS:
            row = []
            for p in PROBE_RATES:
                pw = simulate(n, BASELINE_RATES[1], p, sizes, mode, rng, args.trials)
                row.append(pw)
                out.setdefault(mode, {}).setdefault(str(n), {})[f"{p:.2f}"] = pw
            print(f"{n:>8}" + "".join(
                f"{v:>12.2f}{'*' if v >= 0.80 else ' '}" for v in row))
        print("  * = 80% power\n")

    # sensitivity to the baseline rate at a fixed n
    print("=" * 78)
    print("SENSITIVITY TO THE BASELINE RATE  (independent, n = 400 boards)")
    print("=" * 78)
    hdr = f"{'baseline':>10}" + "".join(f"{f'probe {p:.0%}':>13}" for p in PROBE_RATES)
    print(hdr)
    print("-" * len(hdr))
    for b in BASELINE_RATES:
        row = [simulate(400, b, p, sizes, "independent", rng, args.trials)
               for p in PROBE_RATES]
        out.setdefault("sensitivity_n400", {})[f"{b:.2f}"] = {
            f"{p:.2f}": v for p, v in zip(PROBE_RATES, row)}
        print(f"{b:>10.0%}" + "".join(
            f"{v:>12.2f}{'*' if v >= 0.80 else ' '}" for v in row))

    print("\n" + "=" * 78)
    print("READ")
    print("=" * 78)
    print("Pick the board target from the smallest effect you would still want to")
    print("detect, using the INDEPENDENT column. Freeze it before running any")
    print("model. If the required n exceeds what the dataset supports, the honest")
    print("move is to widen the board definition or preregister the test as")
    print("exploratory -- not to run it underpowered and report whatever lands.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
