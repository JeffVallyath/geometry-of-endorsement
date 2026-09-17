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
python -m repro.state_sufficiency --output reproduced
```

`verify` reports the number of checked files and experiments. `evidence` reports
the declared number bindings and scientific qualification checks, and fails on
unresolved evidence by default. Tables go to `reproduced/tables/`; plots go to
`reproduced/figures/`. Use a fresh output directory for another replay. Existing
outputs and committed evidence are not overwritten.

Figure 7 uses the prospective matched-history study's saved primary means and multiplicity-adjusted intervals
directly, without recomputing uncertainty. Its full-precision projection is
[v10_source_history.json](artifacts/figures/v10_source_history.json), including
source hashes and fixed opportunity denominators. The earlier shared-state study's descriptive Figure 7 is
preserved as Supplementary Figure S3, with unchanged counts in
[consequence_comparisons.json](artifacts/figures/consequence_comparisons.json).
Both regenerate through `python -m repro figures`; the selected repeat-checked
Wren/Orla example remains linked separately from the main caption.

The tests additionally require pytest and nbformat for notebook validation. Install the test dependencies and run:

```sh
python -m pip install -r reproducibility/requirements-test.lock
python -m pytest
```

The supported suite is `tests/`. Historical tests in study source snapshots
require their original layouts and dependencies. The editable checkout is the
supported installation: a code-only wheel does not contain the evidence archive.

## Experiment index

For a claim-first route with figures and expected outputs, start with
[PAPER_GUIDE.md](PAPER_GUIDE.md).

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
| Counterfactual consequences | `tables`; [behavioral scores](reproducibility/representation/counterfactual_fidelity/) | Fitted direction and later-layer readout parameters are unavailable; supporting readout evidence is aggregate-only |
| Compression, decision-rule edits, applicability | [Terminal summaries](reproducibility/representation/background/) | Source-bound summaries, not additional numerical replays |
| Initial and matched shared-state editors | `tables`; [initial](reproducibility/relational_editing/v1/) and [matched](reproducibility/relational_editing/v2/) packages | Qualification and development rows replay recorded summaries |
| Repetition, restoration, and broader training consequences | `tables`; [update](reproducibility/relational_editing/v3/) and [coverage](reproducibility/relational_editing/v4/) packages | Recomputes saved behavioral statistics; does not repeat training |
| Source independence and fixed readers | `tables`; [source-independence package](reproducibility/relational_editing/v6/) | Uses original per-scene/per-case sufficient statistics, not raw model probabilities |
| Later constructive editing | `tables`; [combined-update summary (`V7`)](reproducibility/state_sufficiency/v7/) | Source-bound aggregate rates and gate flags, not training or scene-level reconstruction |
| Source-history sufficiency, retrospective census and archival corroboration | `tables`; [retrospective census (`V9`)](reproducibility/state_sufficiency/v9/) and [fixed-criterion archival check (`V6`)](reproducibility/state_sufficiency/v6/) summaries | Selected source-table consistency, not independent reconstruction of qualifying checks from scores |
| Source-history sufficiency, prospective confirmation aggregates | `tables`; [prospective confirmation (`V10`)](reproducibility/state_sufficiency/v10/) | Reconstructs primary rates, intervals and strong-witness counts from saved per-case counts |
| Source-history sufficiency, independent per-example verification | [Collaborator package](reproducibility/state_sufficiency/independent_verification/README.md); commands below | Reconstructs semantic labels, eligibility, witnesses and intervals from saved scores; no original analysis imports or neural rerun |
| Six-case causal cache intervention | `python -m repro.cache_crossover --output reproduced/causal-panel`; [package](reproducibility/cache_crossover/README.md) | Saved-score normalization and all classifications/controls; selected cases |
| Matched one-case causal extension | `python scripts/export_cache_crossover.py --output reproduced/causal-extension` | Both formats and strict direct checks; separate retrospective case |
| Public-data Qwen/Ripple extension | `python -m repro.in_context_updates --output reproduced/qwen`; [package](reproducibility/in_context_updates/README.md) | Frozen analyzer replay of all saved responses with both parsers, including qualification and repeats |
| Matched-history update-method comparison | `python -m repro.update_methods --output reproduced/update-methods`; [package](reproducibility/update_method_comparison/README.md) | All saved primary scores, four contrasts and intervals, four CSVs; no new inference |
| MEMIT/GPT-J MQuAKE qualification | [Compact source records](reproducibility/memit_qualification/README.md) | Pretrained inference ran but failed reference qualification; no FINAL result; summary-level verification |

The prepared fixed-readout predecessor had no fresh efficacy measurement. Its
scientific definitions are inherited by the executed source-independence study;
it is not distributed as a second result package. The prospective matched-history
study (`V10`) is complete at controlled
generated-case scope. Public-data in-context validation is completed and replayable.
Conventional parameter-editing validation remains unresolved. The historical
AlphaEdit route stopped before pretrained inference; the later MEMIT qualification
failure is a distinct executed outcome, not that earlier exclusion.

## What the CPU checks establish

A benchmark case (`root` in saved records) groups the same intended final facts
with their starting histories and questions. Some earlier studies instead group
measurements by synthetic situation (`scene`); their units are kept separate.
A qualifying history-dependent question, called a `witness`, passes the relevant
current-fact and reference checks but receives different answers across histories.
The exact checks and statistical denominators are given in each package guide.

The original representation figures render from committed aggregate data.
Saved null arrays reproduce displayed permutation summaries; refitting the
original permutation experiment is a different operation.

Earlier editing tables use their original scientific analyzers. Fresh
single-edit, program, control, energy and workflow statistics are recomputed from
saved question or episode records. Qualification tables use recorded
qualification summaries, and development tables use recorded checkpoint
summaries. These commands do not independently recheck omitted stored-activation
tensors or repeat training. Both original seeds and all measured controls are
retained; row ordering is aligned by scientific identity when comparing tables.

The source-independence package retains the original per-scene and per-case
sufficient statistics, paired contrasts, program inventory, final decisions,
synthetic population manifests and selected weights. Replay recomputes means,
grouped intervals and conjunctions from those statistics. Questions, starting histories,
answer-code draws and training seeds remain grouped within their original
situation or benchmark case. This does not independently prove the original physical cache
identity or reproduce raw label probabilities from model activations.

The full source-independent bundle uses every question in its declared bundle;
the original question subset remains a separate result. Identical wrong answers
do not satisfy source independence. Repetition alone cannot convert a low-quality
single edit into success. Original qualification amendments and post-outcome
descriptive interval additions remain disclosed.

## Three levels of reproduction

Aggregate saved-evidence replay reconstructs tables from retained summaries and
benchmark case counts. Independent per-example reconstruction starts instead from saved
model outputs, relations, questions and reference responses. A full fresh neural
rerun would load the models and generate new outputs. This checkout supports
the first two levels for the prospective study; the independent verification package supplies the second.

The [compact source-history package](reproducibility/state_sufficiency/README.md)
records original archive/member hashes and exact selection rules. The command
`python -m repro.state_sufficiency --output reproduced` exports its derived tables;
the same checks also run inside `python -m repro tables`. It recomputes the prospective study's case bootstrap, seed-averaged rates and
counts of cases qualifying under at least one seed. Checks of the retrospective
census and archived study validate selected aggregate identities and denominators,
not their original score-level qualifying checks.

The [independent collaborator package](reproducibility/state_sufficiency/independent_verification/README.md)
retains the relevant fields from the prospective study's raw journals, complete
case records, references and saved repeat passes. The references either present
current facts from the start or apply the same update to already-correct facts. A new standalone
analysis derives Boolean labels, response validity, eligibility, primary and
strong witnesses, per-condition and per-case quantities and confidence intervals without
importing the original experimental analyzer. Run the calculation first, then
compare with the separately stored reported results:

```sh
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
python -m pytest tests/test_independent_state_sufficiency.py
```

The independent calculation never opens expected results. No primary
classification inputs are omitted at the saved-output level. Full-vocabulary
argmax membership is established from the retained candidate probabilities and
the remaining probability-mass bound, rather than assumed from a validity flag.
The complete original archives are not copied into this checkout; every included
projection has source-member hashes, exact transformations and omitted-field
descriptions. Six traceable examples cover witnesses, eligible non-witnesses and
excluded cases. The retained repeat panel covers native and learned conditions,
not independent textual-correction repeats.

Fresh forward passes, cache-tensor regeneration, tokenizer execution and
prospective execution-order attestation are not performed. Those boundaries
remain distinct from saved per-example reconstruction. The retrospective census
and archival qualifying-check packages remain aggregate-level here. No private resource, billing,
provider or machine logs are included. Original archive hashes identify sources,
not public download locations or attestation of the original model run.

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

The [fresh-inference audit](reproducibility/fresh_inference/README.md) records
recovered execution code/checkpoints and the remaining smoke-test boundary.
No new pretrained inference is part of this integration pass.

The old `scripts/package_figures.py` is explicitly a historical Figures 1 through 5
packager. Use `python -m repro figures` for the complete current figure set.
The project-status notebook is historical; its original pins are unchanged.

Some representation files are selected source snapshots, not a complete original
GPU package. Counterfactual fitted direction and later-layer readout parameters
are unavailable; that readout also lacks per-context logits for independent replay.
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
