# Reproducibility

The CPU replay checks artifact hashes and scientific claims, rebuilds the supported
tables from saved measurements, and renders the figures. It does not download a
model, generate new answers, or train an editor. No GPU or model-account access
is needed for these commands.

## Run the saved-evidence replay

Use a complete repository checkout or extracted repository archive, Python 3.11
or newer, and enough disk space for a separate output directory. The tested
environment uses Python 3.13.12 and the package versions in
[requirements-replay.lock](reproducibility/requirements-replay.lock). Run from the
repository root:

```sh
python -m pip install -r reproducibility/requirements-replay.lock -e .
python -m repro verify
python -m repro evidence
python -m repro tables --output reproduced
python -m repro figures --output reproduced
```

`verify` reports the number of checked files and experiments. `evidence` reports
the declared number bindings and scientific qualification checks, and fails on
unresolved evidence by default. Tables go to `reproduced/tables/`; plots go to
`reproduced/figures/`. Use a fresh output directory for another replay. Existing
outputs and committed evidence are not overwritten.

The tests additionally require pytest and nbformat for notebook validation. Install the test dependencies and run:

```sh
python -m pip install -r reproducibility/requirements-test.lock
python -m pytest
```

The supported suite is `tests/`. Historical tests in study source snapshots
require their original layouts and dependencies. The editable checkout is the
supported installation: a code-only wheel does not contain the evidence archive.

## Experiment index

Every `tables` entry below is included in `python -m repro tables`. Exact file
identities and unavailable inputs are in the [artifact manifest](reproducibility/manifest.json).
[Results and Claims](docs/RESULTS_AND_CLAIMS.md) supplies the interpretation.

| Evidence | CPU output or destination | Material boundary |
|---|---|---|
| Relation readouts and factual control | `tables`, `figures`; [figure data](artifacts/figures/figure_data.json) | Aggregate replay; no new probe fitting |
| Rewrite instability | `tables`; [predictions and freezes](reproducibility/representation/rewrite_fragility/) | Individual review judgments are not distributed |
| Relation transfer and matched steering | `tables`; [scores and source](reproducibility/representation/relation_transfer/) | Original corpus text must be acquired separately |
| Fingerprint validation | `tables`; [validation arrays](reproducibility/representation/causal_fingerprint/) | Replays the recorded invalid instrument, not a passed one |
| Broader-state information | `tables`; [frozen-plan predictions](reproducibility/representation/full_state_report/) | [Controlled readouts](reproducibility/representation/richer_state/) and elicitation comparisons are aggregate-only |
| Counterfactual consequences | `tables`; [behavioral scores](reproducibility/representation/counterfactual_fidelity/) | Fitted direction and later-layer witness parameters are unavailable; witness evidence is aggregate-only |
| Compression, decision-rule edits, applicability | [Terminal summaries](reproducibility/representation/background/) | Source-bound summaries, not additional numerical replays |
| Initial and matched shared-state editors | `tables`; [initial](reproducibility/relational_editing/v1/) and [matched](reproducibility/relational_editing/v2/) packages | Qualification and development rows replay recorded summaries |
| Repetition, restoration, and broader training consequences | `tables`; [update](reproducibility/relational_editing/v3/) and [coverage](reproducibility/relational_editing/v4/) packages | Recomputes saved behavioral statistics; does not repeat training |
| Source independence and fixed readers | `tables`; [source-independence package](reproducibility/relational_editing/v6/) | Uses original per-scene/per-root sufficient statistics, not raw model probabilities |

The prepared fixed-readout predecessor had no fresh efficacy measurement. Its
scientific definitions are inherited by the executed source-independence study;
it is not distributed as a second result package. Unfinished successors are
excluded from the completed-result index.

## What the CPU checks establish

The original representation figures render from committed aggregate data.
Saved null arrays reproduce displayed permutation summaries; refitting the
original permutation experiment is a different operation.

Earlier editing tables use their original scientific analyzers. Fresh
single-edit, program, control, energy and workflow statistics are recomputed from
saved question or episode records. Qualification tables use recorded
qualification summaries, and development tables use recorded checkpoint
summaries. These commands do not independently recheck omitted physical witness
tensors or repeat training. Both original seeds and all measured controls are
retained; row ordering is aligned by scientific identity when comparing tables.

The source-independence package retains the original per-scene and per-root
sufficient statistics, paired contrasts, program inventory, final decisions,
synthetic population manifests and selected weights. Replay recomputes means,
grouped intervals and conjunctions from those statistics. Questions, origins,
answer-code draws and training seeds remain grouped within their original
scene/root. This does not independently prove the original physical cache
identity or reproduce raw label probabilities from model activations.

The full source-independent bundle uses every question in its declared bundle;
the original question subset remains a separate result. Identical wrong answers
do not satisfy source independence. Repetition alone cannot convert a low-quality
single edit into success. Original qualification amendments and post-outcome
descriptive interval additions remain disclosed.

## Optional model reproduction and distribution boundaries

The original [ValuePrism notebook](notebooks/01_reproduce_valueprism_pipeline.ipynb),
[Llama development notebook](notebooks/02_reproduce_llama_m1.ipynb), and
[factual-control notebook](notebooks/geometry_of_truth.ipynb) retain their
documented FULL/ANALYSIS interfaces. They may require licensed data, gated model
access, a compatible activation cache and GPU memory. Their setup cells pin the
original reproduction snapshots; they are not the CPU interface for this
saved-evidence replay.

The later-study source directories retain scientific construction, evaluation,
training/editor and analysis routines with provenance and portability notices.
These source snapshots are not complete, turnkey GPU reruns. The supported public reproduction level is
the saved-evidence replay specified above, not a claim of end-to-end backbone
reproduction.

Some representation files are selected source snapshots, not a complete original
GPU package. Counterfactual fitted direction and later-layer witness parameters
are unavailable; the witness also lacks per-world logits for independent replay.
Controlled and elicitation readouts lack the full prediction ledgers needed to
recompute their supporting summaries. The manifest distinguishes these boundaries
from the independently replayed behavioral and frozen-plan endpoints.

Large activation and physical-state archives are not included. An artifact
without a public retrieval location is unavailable to an external reproducer,
even when its hash is known. Those boundaries are recorded in the manifest;
a recorded hash alone does not provide access to an omitted file.
For the older Llama permutation refit, the hash-pinned compact activation
archive named by [the refitting script](scripts/rerun_permutation_null.py) remains
external. Gemma's original larger-null activation cache was not retained.

Synthetic study data, source code, fitted parameters and third-party text have
different distribution constraints. See [Data notice](DATA_NOTICE.md) for model
licenses, corpus acquisition and the review-privacy boundary.

Original model-execution versions and unrecovered environment fields are recorded
in the [editing environment record](reproducibility/relational_editing/runtime-environments.json)
and [representation boundaries](reproducibility/representation/boundaries.json).
