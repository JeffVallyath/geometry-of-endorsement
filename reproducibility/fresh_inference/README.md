# Optional model execution for the prospective matched-history study

Internal experiment ID: `V10`; source package: `STATE_SUFFICIENCY_CONFIRMATION`.

The portable single-case entry point is packaged and its CPU scientific smoke passes.
Fresh pretrained execution is **not verified**. No new pretrained inference is
authorized in this integration pass, so no pretrained forward smoke was run.

The completed `STATE_SUFFICIENCY_CONFIRMATION_RESULTS_AND_REPRODUCTION.zip`
contains the actual `reproduction/scripts/ssc1_eval.py`, `ssc1_prepare.py`,
`ssc1_boundary.py` and related scientific routines. It also contains the exact
selected Gemma/Qwen `setter.npz` files under
`reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/weights/`,
plus the `canonical` adapter and model configuration. The
[supplied manifest](supplied/MANIFEST.json) now binds the byte-preserved scientific
subset: 368 files, including all frozen case-data partitions and ten selected checkpoints.
All declared members of the source archive were checked before selection. No
pretrained weights or provider controllers were imported.

## CPU smoke and optional execution

The saved-score replay environment is sufficient for all normal paper commands.
The original scientific update functions additionally import CPU PyTorch, even
when no model is loaded. Install this optional dependency before the smoke:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
python reproducibility/fresh_inference/run.py
```

Expected: `CPU_SCIENTIFIC_PACKAGE_SMOKE_PASS`, ten checkpoint hashes and realized
bases checked, both models' eight histories converge to the same rewritten facts,
original compile/serve imports succeed, and zero neural forward passes. The check
does not import a model loader or initialize CUDA. It works from an arbitrary
working directory using the absolute script path. Other included benchmark cases can be
selected with `--root-id`.

The optional neural path reuses `ssc1_eval.evaluate_root` without changing it.
It compiles all conditions before opening the question file, preserves the
original read-order guard, exact saved basis loader, BF16 backbone, FP32 workspace
and readout, immutable caches, and single-question serving. It regenerates one
completed-study benchmark case, not a new prospective study or the full primary aggregate.
The original full-study technical qualification is not repeated by this interface.

On a separately authorized compatible GPU host, the original direct dependencies
are in [requirements-inference.txt](supplied/reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS/requirements-inference.txt).
Execution also requires Python 3.12.3. Obtain Gemma from `google/gemma-2-9b-it`
at `11c9b309abf73637e4b6f9a3fa1e92e615547819`, or Qwen from `Qwen/Qwen3-8B`
at `b968826d9c46dd6066d109eabc6255188de91218`, after accepting upstream terms.
Use the upstream Hugging Face `snapshot_download` API with that exact `revision`
and a user-owned `cache_dir`; never `main`. The runner only opens a local cache
and never downloads a model automatically.

```sh
python reproducibility/fresh_inference/run.py --execute --actor qwen --root-id SSC1-FINAL-0006 --cache /path/to/licensed/hub-cache --output /path/to/new-output
```

This invocation is documented, **not executed or certified**. The original
Wren/Orla example belongs to this benchmark case. The supported way to verify its measured
answer pattern remains the independent saved-score reconstruction. Matching a
fresh output to the original is still an outstanding acceptance check.

The selected causal extension additionally needs the actual `continuation.bindings`,
`continuation.gates`, `canonical.adapter` and `intervention_adapter` modules
referenced by its retained adapter. The completed extension archive includes its
own `implementation/adapter.py`, but not a self-contained copy of all those
dependencies. The predecessor completed bundles must supply their exact versions.

A CPU structural check cannot establish a successful pretrained forward pass.
Do not advertise the recovered source or older notebook interfaces as a tested
fresh reproduction.
Use the [supported main saved-score commands](../state_sufficiency/independent_verification/README.md)
and [causal replays](../cache_crossover/README.md) meanwhile.

Any eventual optional runner must retrieve models only from the exact licensed
upstream revisions, accept a user-owned model cache, preserve the study's
single-question serving and scientific gates, and have no private machine/cloud
dependency. It must not rent compute, train a new editor or silently replace the
original checkpoint with a historical one.

## Integration incident and regression guard

The first CPU smoke failed because the saved setter module imports `torch` during
address normalization; the lighter saved-score environment does not include it.
The optional CPU dependency is now explicit. The smoke regression checks actual
source imports, input rewriting and saved parameter arrays from an arbitrary
directory, while rejecting CUDA initialization or model-loader imports. This was
a packaging failure, not a scientific outcome, and no experiment was retried.
