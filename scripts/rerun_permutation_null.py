"""Re-run the M1 permutation null at a larger B, faithful to the original.

Null (m1_vertical_slice.probes.grouped_label_flip): every training SITUATION
independently keeps or reverses all of its labels. Both probes are refit on the
flipped training labels, the select split refits its own scaler (ddof=1) for
that draw, and I_b is recomputed on the 125 held-out evaluation boards. The
reported p-value is one-sided: (k + 1) / (B + 1) with k the number of draws at
or above the observed value.

Llama, layer 19, CPU only.

    python scripts/rerun_permutation_null.py --draws 10000

Inputs
------
artifacts/figures/m1_situation_groups.csv
    item_id -> situation_id, the grouping the null preserves. Extracted from
    results/cache_index.csv in the M1 vertical-slice audit archive
    (sha256 9b3561887259c6346ce8d046bf79bd601ec33c2c995edbf2641c6305a16950cf).
    Opaque ids only; no source text.

--activations (default below)
    The compact export holding the 2,300 x 4096 layer-19 activations. Too large
    for the repository; set GOE_ARCHIVE_DIR or pass --activations. Its digest is
    pinned here and verified on load, and recorded into the output so a null can
    be traced back to its exact input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "artifacts" / "figures" / "m1_situation_groups.csv"
DEFAULT_OUT = ROOT / "artifacts" / "figures" / "permutation_null_llama_B10000.json"

DEFAULT_ACTIVATIONS = str(Path(os.environ.get("GOE_ARCHIVE_DIR", "."))
                          / "claim2_m1_compact_transfer_a56997d2875f.zip")
ACTIVATIONS_SHA256 = "0b4ac61072932015ec6c99a4f8fe94bf06461c59eafc965aaca1ad40f9137553"
ACTIVATIONS_MEMBER = "claim2_m1_compact.npz"

SEED = 20260803
LOGISTIC_C = 0.1
MAX_ITER = 2000

# Published values from the source run, checked against the refit on load.
PUBLISHED = {"difference_in_means": 1.6470105763501846, "logistic": 2.0835848847098010}

_G: dict = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(activations: str = DEFAULT_ACTIVATIONS) -> dict:
    if _G:
        return _G
    path = Path(activations)
    if not path.is_file():
        raise SystemExit(
            f"activation archive not found: {path}\n"
            f"Expected sha256 {ACTIVATIONS_SHA256}. It is not stored in the "
            f"repository; see the module docstring."
        )
    digest = sha256(path)
    if digest != ACTIVATIONS_SHA256:
        raise SystemExit(
            f"activation archive digest mismatch\n  expected {ACTIVATIONS_SHA256}\n"
            f"  found    {digest}"
        )

    d = np.load(io.BytesIO(zipfile.ZipFile(path).read(ACTIVATIONS_MEMBER)), allow_pickle=False)
    H = d["activation"].astype(np.float32)
    item = d["item_id"]
    split = d["split"]
    lab = d["reference_label"].astype(np.int8)
    board = d["board_id"]

    with GROUPS.open(encoding="utf-8") as fh:
        mapping = {r["item_id"]: r["situation_id"] for r in csv.DictReader(fh)}
    unknown = [i for i in item if i not in mapping]
    if unknown:
        raise SystemExit(f"{len(unknown)} items missing from {GROUPS.name}")
    situation = np.array([mapping[i] for i in item])

    tr, se, ev = split == "pilot_train", split == "pilot_select", split == "pilot_eval"
    _G.update(
        Xtr=H[tr], ytr=lab[tr], gtr=situation[tr],
        Xse=H[se], Xev=H[ev], yev=lab[ev],
        board_masks=[(board[ev] == b) for b in sorted(set(board[ev].tolist()))],
        activations_sha256=digest,
        groups_sha256=sha256(GROUPS),
    )
    return _G


def dim_direction(X, y):
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    direction = mu1 - mu0
    direction = direction / np.linalg.norm(direction)
    return direction, float(0.5 * (mu1 + mu0) @ direction)


def i_b(scores_eval, scores_select, g):
    z = (scores_eval - float(scores_select.mean())) / float(scores_select.std(ddof=1))
    yev = g["yev"]
    return float(np.mean([z[m & (yev == 1)].sum() - z[m & (yev == 0)].sum()
                          for m in g["board_masks"]]))


def evaluate(ytr, g, seed):
    d, mid = dim_direction(g["Xtr"], ytr)
    dim = i_b(g["Xev"] @ d - mid, g["Xse"] @ d - mid, g)
    clf = LogisticRegression(C=LOGISTIC_C, solver="liblinear", dual=True,
                             max_iter=MAX_ITER, random_state=seed).fit(g["Xtr"], ytr)
    return dim, i_b(clf.decision_function(g["Xev"]), clf.decision_function(g["Xse"]), g)


def flip(y, groups, rng):
    out = y.astype(np.int8).copy()
    for grp in np.unique(groups):
        if bool(rng.integers(0, 2)):
            mask = groups == grp
            out[mask] = 1 - out[mask]
    return out


def one(perm: int):
    g = load()
    rng = np.random.default_rng(SEED + perm)
    y = flip(g["ytr"], g["gtr"], rng)
    if len(np.unique(y)) != 2:
        return None
    return evaluate(y, g, SEED + perm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--activations", default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    g = load(args.activations)
    observed = dict(zip(("difference_in_means", "logistic"), evaluate(g["ytr"], g, SEED)))
    for name, value in observed.items():
        print(f"observed {name:<20} {value:.10f}   published {PUBLISHED[name]:.10f}")
    sys.stdout.flush()

    from multiprocessing import Pool
    with Pool(processes=min(20, os.cpu_count() or 4)) as pool:
        draws = [r for r in pool.imap_unordered(one, range(args.draws), chunksize=20)
                 if r is not None]
    arr = np.asarray(draws, dtype=float)

    result = {
        "B_requested": args.draws,
        "B_valid": int(arr.shape[0]),
        "seed": SEED,
        "p_value": "one-sided empirical, (k + 1) / (B + 1), k = draws >= observed",
        "null_definition": (
            "situation-level label flip on the training split; both probes refit; "
            "select scaler refit per draw (ddof=1); I_b recomputed on the 125 "
            "held-out evaluation boards"
        ),
        "inputs": {
            "activations_archive": Path(args.activations).name,
            "activations_sha256": g["activations_sha256"],
            "activations_member": ACTIVATIONS_MEMBER,
            "situation_groups": str(GROUPS.relative_to(ROOT)).replace("\\", "/"),
            "situation_groups_sha256": g["groups_sha256"],
        },
        "published_observed": PUBLISHED,
        "probes": {},
    }
    for name, column in (("difference_in_means", 0), ("logistic", 1)):
        values = arr[:, column]
        obs = observed[name]
        k = int((values >= obs).sum())
        k_two = int((np.abs(values) >= abs(obs)).sum())
        result["probes"][name] = {
            "observed": obs,
            "k_ge_observed": k,
            "B": int(values.size),
            "p_empirical": (k + 1) / (values.size + 1),
            "p_two_sided_sensitivity": (k_two + 1) / (values.size + 1),
            "null_mean": float(values.mean()),
            "null_sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
        block = result["probes"][name]
        print(f"{name:<20} k={k:<5} B={values.size}  p={block['p_empirical']:.5f}"
              f"  (two-sided {block['p_two_sided_sensitivity']:.5f})")

    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
