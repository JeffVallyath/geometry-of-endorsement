# Pre-update route results

Start with RESULTS.md and figures/SCREEN_CORRECTIONS.png. The frozen historical
gate failed; no final cases were run. All owned instances are closed.

The package includes two immutable source/runtime capsules, four terminal exports
(including both failed-attempt exports), exact source/input/package hashes,
runtime and shutdown receipts, all raw scores, safety tensors, CPU analysis,
descriptive tables, two SVG/PNG figures, and the external-review packet.

## Offline CPU replay

Use Python 3.11 or newer and NumPy 2.2.6. No model, GPU, network, credential,
Torch, Transformers, or Lambda access is needed. After extracting the results ZIP:

```powershell
python -m pip install -r CPU_REQUIREMENTS.txt
python tools/replay_bundle.py --bundle . --work D:/Downloads/PREUPDATE_ROUTE_CPU_REPLAY
```

Choose a fresh work directory. The verifier checks every delivered member,
extracts and verifies the immutable capsule and all four raw exports, then runs
the frozen CPU gate from the included source and compares it with the reported
result. It never changes inputs or launches inference. The original CPU audit
used Python 3.11.15/NumPy 2.3.5; an independent bundled replay is also checked with
the frozen runtime's NumPy 2.2.6. The gate is deterministic integer/Boolean logic.

To recreate the two figures after replay, without rerunning science:

```powershell
python tools/screen_figures.py --source . --out D:/Downloads/PREUPDATE_ROUTE_FIGURE_REPLAY
```

Optional PNG rendering uses resvg-py 0.5.0 via the same script's `--render` flag.
SVG generation itself uses only the standard library. Rendering was isolated
from the scientific environment.

## Inventory and interpretation

- MANIFEST.json declares each delivered path, byte count and SHA256.
- capsules/puf1-r2.zip contains the successful FHV source, fixed inputs and
  analysis, runtime pins, implemented hooks and model-free checks.
- raw/r2_seed0.zip and raw/r2_seed1.zip contain the complete screen and technical
  data. Each scientific/safety/<case>/TENSORS.npz has a matching index; mediator
  values are original BF16 bit patterns in int16 arrays. Load with allow_pickle=False.
- raw/r1_seed*.zip preserve complete and partial failed-attempt work. The r2
  ledger already carries r1 work; do not add it a second time.
- SCREEN_DAMAGE.csv and SCREEN_CORRECTIONS.csv are descriptive historical-panel
  tables. TECHNICAL_FOUR_CELLS.csv is not final-population mechanistic evidence.
- A runtime config may reference the original machine's paths. These are
  provenance, not paths required by the portable CPU replay.
- The 128 frozen final inputs exist in the capsule but have no neural outcomes.
  Final repair/damage intervals, repeat discrepancies and a repeated-positive
  worked example are inapplicable on the declared failed-gate branch.

No pretrained weights or credentials are included. No commit, push, paper edit,
or successor experiment was performed. Source worktree:
`D:/geometry-of-endorsement/preupdate-route-v1`, branch
`codex/preupdate-route-v1`, base `2de7cf71183031e19c129b7b316cf828ac4bf04f`;
task files remain uncommitted as instructed.
