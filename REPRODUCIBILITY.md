# Reproducibility

> **Historical version.** These commands and reproduction levels describe an
> earlier release. Start with the [collaborator README](README.md), then use the
> [current reproduction guide](history_updates/docs/reproduction.md) and
> [current paper map](history_updates/docs/paper_map.md). Historical content follows.

The supported route reconstructs saved results on CPU. It never downloads a model,
opens a cloud account, generates pretrained outputs or retrains an editor.
Scientific inputs stay fixed. Each output directory must be new.

## Environment and commands

The CPU dependency pins target the tested Python 3.13.12, Windows x86-64 environment.
The source requires Python 3.12 or later; other combinations remain unverified.
Use an isolated environment outside the release and activate it before the commands
below. Run from the release root, including when the path contains spaces.

Reserve at least 5 GiB of free space on the output volume for each paper-analysis
or test workflow (10 GiB if running both concurrently), plus space for the Python
environment on its own volume. Compressed evidence expands during reconstruction.
The paper runner checks a conservative 5 GiB floor before creating output; this
is a preflight check, not a reservation against other processes consuming space.
An absolute `--output` path on another drive is supported. Preserve failed output
directories and choose a new path when retrying after an environment failure.

```sh
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m repro.release_manifest
python -m repro.paper_results --output reproduced/paper
python -m repro.paper_results --test --output reproduced/tests
```

Installation needs the normal Python package index and the official CPU PyTorch
wheel index. Analysis subprocesses have network access and pretrained/service
imports blocked. CPU-only PyTorch is used for synthetic tensors and parameter
arithmetic. These guards catch accidental use; they are not a hostile-code sandbox.

The paper command runs 16 retained analysis/check steps; MQMH has a separate saved-response reconstruction command and dedicated test. Logs and `CHECKS.json`
are written under the requested output. Each step has a 1,200-second bound.
Failure stops the sequence and preserves prior outputs. A failed scientific
qualification recorded in the evidence is an expected result, not a runner error.
The test command runs the same retained root test suite as pytest, with offline
restrictions and isolated temporary files.

Public manifests verify the actual delivered bytes and closed study inventories.
The root manifest excludes itself from its digest list. After installation, its
normal mode permits only `reproduced/`, Python bytecode caches and the package's
own editable-install metadata as local products. `--strict` permits none of these
and is intended for a pristine extracted tree. ZIP integrity requires comparison
with the separately supplied archive checksum. Hashes alone do not validate a
scientific claim or certify anonymity.

## Reproduction levels

Level 1 views saved results or redraws aggregates. Level 2 reconstructs statistics
and gates from scores or sufficient statistics. Level 3 regenerates pretrained
outputs. Level 4 retrains learned components and reruns the complete study.

| Study | Level 1 | Level 2 | Level 3 | Level 4 / specific missing material |
|---|---|---|---|---|
| <a id="reproduction-relation"></a>Relation readout and factual controls | Aggregate reconciliation | Stored null-tail recount and available Llama gates | Source/configs inspectable; complete execution unavailable | Gemma activation membership and MiniLM fitted count/revision supplied. Licensed rows, exact MiniLM fitted membership, current Llama fit provenance, activations and complete Gemma config/control inputs remain unavailable |
| <a id="reproduction-consequences"></a>Consequence recovery | Saved results | Saved-margin reconstruction, 145 comparisons | Source inspectable; complete execution unavailable | Refitted direction and later-layer witness weights missing. One exact prior corpus supplied; the second remains absent for the overlap audit |
| Primary matched histories | Verified | Verified saved-score semantic/gate reconstruction and case bootstrap | Supplied portable path available but not executed or verified | Selected weights, paired-stage source, explicit optimizer/schedule settings and FIT/CAL records supplied; public backend integration and fresh retraining remain unverified |
| Original stronger controls | Verified | Verified post-hoc reconstruction from original scores | Inherits primary neural limit | No new fit is part of this reanalysis |
| Fresh Gemma confirmation and utility | Verified | Verified primary, utility and repeat reconstruction | Supplied source available but unexecuted; model/tokenizer not included | Frozen editor reuse; retraining is not part of this study |
| Smaller-model textual breadth and stateless 32B | Verified | Verified saved-score accounting and recorded repeats | Supplied source available but unexecuted; backbones/tokenizers not included | No learned component is fitted in these textual studies |
| Selected crossover and matched extension | Verified | Verified saved-journal classifications and controls | Original measurement source available but unexecuted; native caches absent | Inherits selected-editor execution/retraining limits |
| Question-blind refresh | Saved results and intervals | Primary gates and harms reconstructed from 512 case/seed score blocks; 14 selected-repeat blocks and four tables reproduced exactly | Fresh execution unavailable in this release | Public score/token/alias projections supplied. Native caches and the full measurement runtime are not included; original intervals are retained, not regenerated |
| Separate update-method comparison | Verified | Verified 208,896 scores, 6,144 telemetry rows and paired contrasts | Not verified; exact executed revision/environment absent | Planned additional panels and complete external baseline adapter absent; completed origin panel retained |
| MQuAKE-derived public multi-hop (MQMH) | Verified from saved responses | Reconstructs both models, full denominators, subject-clustered intervals, strict parser and repeats | `python -m repro.mqmh --output reproduced/mqmh` reconstructs the retained responses; no model/tokenizer needed | No new inference; residual pretraining exposure unknown; shared cohort; no parameter-editing transfer claim |
| Ripple-derived public-data extension | Verified | Verified both parsers, all retained phases, locality and negative repeats | Supplied source available but unexecuted; model/tokenizer not included | No training; broader preparation pool and immutable upstream dataset checkout unavailable |
| Carrier-reader and mediation follow-ups | Verified saved tables/reports | Narrower only: supplied effect/score accounting and qualification checks | Native tensor replay unavailable | Private activation/native cache tensors absent |
| Selected editor parameters | Public manifests and arrays | Original CPU arithmetic/interface tests; reference recipe and selection records available | Real-tokenizer and pretrained execution unverified | Training source is an inert reference, not a supported retraining command. Original optimizer/RNG snapshots are withheld and would be required for exact continuation |

The [MiniLM training discussion](docs/appendix/A03_text_features.md#training-provenance)
is separate from evaluation-board agreement. The Gemma record establishes
107,368 fitted text-comparator rows, versus 1,500 activation-probe training rows.
Its encoder revision is recorded; its exact fitted-row list remains unavailable.
The current Llama fit's provenance remains unestablished. Published scores are
the recorded run values. See the [training records](artifacts/probe_weights/README.md#recovered-gemma-provenance).

## Topic-specific commands and outputs

Run the MQMH reconstruction with `python -m repro.mqmh --output reproduced/mqmh`; it checks the retained hashes and reproduces `summary.json`, target counts, case 1617, and `CHECKS.json` without inference. Run its regression checks with `pytest tests/test_mqmh.py`. Each study guide contains its executable command. [PAPER_MAP.md](PAPER_MAP.md)
is generated from [paper_map.json](paper_map.json) and supplies the evidence,
settings, command and expected files for every mapped topic, paper figure and
supplementary table. The paper command runs the same commands under its output
root, with independent analysis preceding comparison to expected primary results.

All integer counts, identities, cohort memberships, gate decisions and seeds use
exact checks. Original study tolerances apply to derived floats. No tolerance was
relaxed to accommodate packaging. The retained overlap test explicitly skips
because one of its two required earlier corpora remains unavailable; a skip is not
a completed check.

The [figure guide](figures/main/README.md) follows the paper's five main-text and
two appendix figures. Run `python -m repro.main_figures --output
reproduced/main-figures` to render PDF, SVG and PNG versions from the verified
measurements. Display wording and layout preserve the underlying values and
intervals. PDF and direct PNG rasterization can differ in antialiasing; PDFs are
the authoritative vector figures. The command performs no model inference.

## Scope and acceptance

[STATUS.md](STATUS.md) records final local acceptance and its qualifications.
The release contains only retained studies and required dependencies. It preserves
negative outcomes, ineligible denominators, selection rules, amendments and harms.
Technical recovery and scientific repeat remain distinct in the study records.
Local artifact inspection does not verify an eventual hosting account or guarantee
that distinctive public content cannot be matched to another copy.
