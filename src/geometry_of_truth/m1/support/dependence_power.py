"""
Dependence-aware power, using the ACTUAL board graph.

The earlier simulation clustered by consideration pair. With one board per
pair that is identical to clustering by board -- i.e. no clustering at all.
The real dependence runs through individual consideration clusters appearing
in several different pairs:

    board 1:  Autonomy vs Care
    board 2:  Autonomy vs Safety      <- different pair, shared endpoint

Errors on those two boards may correlate through Autonomy. This script:

1. builds the real bipartite board/consideration graph from the frozen
   confirmatory manifest;
2. reports connected components and the effective sample size;
3. reruns power with a consideration-level random effect and a
   dependence-aware standard error.

If the graph turns out to be one giant component, component bootstrap is
useless (effective n = 1) and the correct tool is MULTIWAY clustering
(Cameron-Gelbach-Miller): V = V_A + V_B - V_AB, clustering on each
consideration endpoint and subtracting the intersection.

Run:
    python dependence_power.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ALPHA = 0.05


# ==========================================================================
# graph structure
# ==========================================================================

def components(boards):
    """Connected components of the board graph, boards joined when they share
    a consideration cluster."""
    by_cluster = defaultdict(list)
    for i, b in enumerate(boards):
        by_cluster[b[0]].append(i)
        by_cluster[b[1]].append(i)

    parent = list(range(len(boards)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for members in by_cluster.values():
        for m in members[1:]:
            union(members[0], m)

    comp = defaultdict(list)
    for i in range(len(boards)):
        comp[find(i)].append(i)
    return list(comp.values())


def multiway_se(values, cluster_a, cluster_b):
    """Cameron-Gelbach-Miller two-way clustered SE of the mean.

    WARNING: not appropriate here. CGM assumes two genuinely separate,
    non-nested clustering DIMENSIONS. Our boards are UNORDERED pairs -- the
    same consideration can be endpoint A in one board and endpoint B in
    another -- so clustering on the A column and the B column separately
    misses that cross-column sharing. Kept only to demonstrate the failure.
    """
    def cl_var(keys):
        s = pd.Series(values).groupby(pd.Series(keys)).sum()
        n, g = len(values), len(set(keys))
        if g < 2:
            return 0.0
        return s.var(ddof=1) * g / (n ** 2)

    inter = [f"{a}|{b}" for a, b in zip(cluster_a, cluster_b)]
    v = cl_var(cluster_a) + cl_var(cluster_b) - cl_var(inter)
    return float(np.sqrt(max(v, 0.0)))


def dyadic_se(values, node_a, node_b):
    """Dyadic-robust SE (Fafchamps-Gubert style) for the mean over dyads.

        V = (1/n^2) * sum_i sum_j  d~_i d~_j * 1{dyads i and j share a node}

    Two boards are allowed to covary whenever they share EITHER consideration,
    regardless of which column it sits in. That makes the estimator invariant
    to swapping endpoint A and endpoint B, which is the invariant CGM fails.
    """
    d = np.asarray(values, dtype=float)
    d = d - d.mean()
    n = len(d)
    if n < 2:
        return 0.0

    # accumulate via node sums instead of an n^2 loop:
    #   sum over pairs sharing >=1 node
    #     = sum_v (sum_{i in v} d_i)^2  -  sum over pairs sharing BOTH nodes
    # (inclusion-exclusion: a dyad sharing both nodes is counted twice)
    by_node: dict = {}
    for i, (a, b) in enumerate(zip(node_a, node_b)):
        by_node.setdefault(a, []).append(i)
        if b != a:
            by_node.setdefault(b, []).append(i)

    total = 0.0
    for members in by_node.values():
        s = d[members].sum()
        total += s * s

    # subtract the double-counted term: dyads sharing both endpoints
    key = [tuple(sorted((a, b))) for a, b in zip(node_a, node_b)]
    for _, idx in pd.Series(range(n)).groupby(pd.Series(key)).groups.items():
        s = d[list(idx)].sum()
        total -= s * s

    return float(np.sqrt(max(total, 0.0)) / n)


# ==========================================================================
# simulation on the real graph
# ==========================================================================

def simulate(boards, p_base, p_probe, rho, rng, trials, mode="independent"):
    """Gaussian copula so the MARGINAL rates are preserved exactly.

    z_b = sqrt(rho) * (mean of the two consideration effects) + sqrt(1-rho) * e_b

    with every component standard normal, so z_b is standard normal and
    thresholding at Phi^-1(p) yields rate p exactly, with intra-consideration
    correlation rho. An earlier version mixed uniforms, which concentrates the
    latent toward 0.5, silently crushed the baseline rate, and made dependence
    look like it INCREASED power.

    rho = 0 reproduces independent boards.
    """
    from scipy.stats import norm

    ca = np.array([b[0] for b in boards])
    cb = np.array([b[1] for b in boards])
    clusters = np.unique(np.concatenate([ca, cb]))
    cidx = {c: i for i, c in enumerate(clusters)}
    ia = np.array([cidx[c] for c in ca])
    ib = np.array([cidx[c] for c in cb])

    t_base, t_probe = norm.ppf(p_base), norm.ppf(p_probe)
    # averaging two independent standard normals gives sd 1/sqrt(2); rescale
    # so the cluster component is itself standard normal
    scale = np.sqrt(2.0)

    def latent():
        u = rng.standard_normal(len(clusters))
        shared = (u[ia] + u[ib]) / 2 * scale
        return np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(len(boards))

    hits = 0
    for _ in range(trials):
        z = latent()
        base = (z < t_base).astype(float)
        # nested: probe sees the same difficulty, so it solves a superset
        # independent: probe has its own draw with the same cluster structure
        zp = z if mode == "nested" else latent()
        probe = (zp < t_probe).astype(float)
        d = probe - base

        se = dyadic_se(d, ca, cb)
        if se > 0 and abs(d.mean()) / se > 1.96:
            hits += 1
    return hits / trials


def swap_invariance_check(boards, rng, reps=200):
    """The diagnostic: swap every board's A and B endpoints. A correct
    estimator for unordered dyads is unchanged. CGM is not."""
    ca = np.array([b[0] for b in boards])
    cb = np.array([b[1] for b in boards])
    rows = []
    for _ in range(reps):
        d = rng.standard_normal(len(boards))
        flip = rng.random(len(boards)) < 0.5           # swap a random half
        sa = np.where(flip, cb, ca)
        sb = np.where(flip, ca, cb)
        rows.append({
            "cgm": multiway_se(d, ca, cb),
            "cgm_swapped": multiway_se(d, sa, sb),
            "dyadic": dyadic_se(d, ca, cb),
            "dyadic_swapped": dyadic_se(d, sa, sb),
        })
    r = pd.DataFrame(rows)
    return {
        "cgm_mean": r["cgm"].mean(),
        "cgm_max_abs_change": (r["cgm"] - r["cgm_swapped"]).abs().max(),
        "dyadic_mean": r["dyadic"].mean(),
        "dyadic_max_abs_change": (r["dyadic"] - r["dyadic_swapped"]).abs().max(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_confirmatory.csv")
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--baseline", type=float, default=0.03)
    ap.add_argument("--out")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    primary = man[man["pool"] == "primary"]
    boards = list(zip(primary["cluster_A"], primary["cluster_B"]))
    print(f"\nconfirmatory primary boards: {len(boards):,}")

    comps = components(boards)
    sizes = sorted((len(c) for c in comps), reverse=True)
    print("\n" + "=" * 72)
    print("BOARD GRAPH â€” boards joined when they share a consideration cluster")
    print("=" * 72)
    print(f"connected components: {len(comps):,}")
    print(f"largest component:    {sizes[0]:,} boards "
          f"({100*sizes[0]/len(boards):.1f}% of the set)")
    print(f"component sizes:      {sizes[:10]}{' ...' if len(sizes) > 10 else ''}")
    singletons = sum(1 for s in sizes if s == 1)
    print(f"singleton components: {singletons:,}")

    giant = sizes[0] / len(boards) > 0.5
    if giant:
        print("\n  GIANT COMPONENT. Component bootstrap would give an effective")
        print("  n near 1 and is unusable. Using multiway (two-way) clustering")
        print("  on the two consideration endpoints instead.")
    else:
        print("\n  No giant component -- component bootstrap is viable.")

    cl = pd.concat([primary["cluster_A"], primary["cluster_B"]])
    print(f"\ndistinct consideration clusters: {cl.nunique():,}")
    print(f"reuse: max {cl.value_counts().max()}  median "
          f"{int(cl.value_counts().median())}")

    # ---------------- estimator choice: the swap invariance test ----------
    rng = np.random.default_rng(0)
    print("\n" + "=" * 72)
    print("ESTIMATOR CHECK â€” swap endpoint A and B; a correct dyadic estimator")
    print("is unchanged, because the pairs are UNORDERED")
    print("=" * 72)
    inv = swap_invariance_check(boards, rng)
    print(f"  CGM two-way     mean SE {inv['cgm_mean']:.5f}   "
          f"max |change| under swap {inv['cgm_max_abs_change']:.5f}")
    print(f"  dyadic-robust   mean SE {inv['dyadic_mean']:.5f}   "
          f"max |change| under swap {inv['dyadic_max_abs_change']:.5f}")
    ok = inv["dyadic_max_abs_change"] < 1e-12
    print(f"\n  dyadic invariant to swap: {'PASS' if ok else 'FAIL'}")
    if inv["cgm_max_abs_change"] > 1e-12:
        print("  CGM is NOT invariant -> it treats 'endpoint A' and 'endpoint B'")
        print("  as separate dimensions and misses cross-column node sharing.")
        print("  Using the dyadic-robust estimator for all inference below.")

    # ---------------- power under real dependence ----------------
    print("\n" + "=" * 72)
    print("POWER ON THE REAL GRAPH  (baseline 3%, multiway clustered SE)")
    print("=" * 72)
    print("rho = share of variance from a consideration-level random effect")
    hdr = f"{'rho':>6}" + "".join(f"{f'probe {p:.0%}':>13}" for p in (0.05, 0.08, 0.10, 0.15))
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for rho in (0.0, 0.3, 0.6):
        row = []
        for p in (0.05, 0.08, 0.10, 0.15):
            pw = simulate(boards, args.baseline, p, rho, rng, args.trials)
            row.append(pw)
            out.setdefault(f"rho_{rho}", {})[f"{p:.2f}"] = pw
        print(f"{rho:>6.1f}" + "".join(
            f"{v:>12.2f}{'*' if v >= 0.80 else ' '}" for v in row))
    print("  * = 80% power    rho=0.0 reproduces the independent-board case")

    print("\n" + "=" * 72)
    print("READ")
    print("=" * 72)
    print("If power at rho=0.3-0.6 stays near the rho=0 column, the shared-")
    print("consideration dependence is not materially eroding the test and the")
    print("600-board target holds. If it drops, either raise the board target or")
    print("tighten the cluster cap so fewer boards share endpoints.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"components": len(comps), "largest": sizes[0],
                       "distinct_clusters": int(cl.nunique()), "power": out},
                      f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

