# Paper guide

The main question is whether the same verified current facts are sufficient for
the same later answer, regardless of the path to those facts. Downstream
propagation failures have prior art. The matched-history test is the centerpiece;
method comparisons and the public-data extension provide supporting evidence.

Read by scientific question: [representation](#representation-and-decoding),
[updating information](docs/RESULTS_AND_CLAIMS.md#7-can-we-change-a-relation-before-knowing-which-question-will-be-asked),
[current facts and later behavior](#matched-history-sufficiency),
[public data](#public-data-extension), [causal intervention](#causal-cache-intervention),
[question-blind repair](#question-blind-repair), and
[alternative update procedures](#update-method-comparison).
The [short terminology lookup](docs/TERMINOLOGY.md) maps exact record names to these ideas.

Use the [locked CPU environment](REPRODUCIBILITY.md#run-the-saved-evidence-replay).
Commands below run from the checkout root and require fresh output directories.
They do not contact a model service, require a cloud account, or run inference.

## Current evidence

This is the current completed-evidence map as of September 19, 2026. Each row
keeps its own experimental unit; selected examples and repeats are not added
to population denominators.

| Study | Current outcome | Evidence and boundary |
|---|---|---|
| Prospective matched histories | All eight primary model/method cells retain history dependence | [Original primary study](#matched-history-sufficiency); basic factual/reference criterion |
| Fresh Gemma confirmation | 22 qualifying target/seed observations in 14 of 64 new designs | [Fresh study](#fresh-confirmation-and-held-out-validation); stronger fixed controls, uncertain validation utility |
| Question-blind repair and repeats | 36/83 repairs across 28 of 256 designs; 50% threshold failed | [Population and successful-case verification](#question-blind-repair); harms and full reference retained |
| Textual breadth | Basic recurrence in all six newly measured smaller-model method rows | [Breadth package](#textual-breadth-and-full-forward-32b); historical bridges failed |
| Stateless Qwen3-32B | 128/1,152 and 123/1,152 primary comparisons; strongest cross-format 0/576 for each method | [Full-forward package](reproducibility/qwen32_full_forward/README.md); existing 64 designs newly measured, not newly sampled |
| Carrier-reader and mediation | Functional transfer and partial mediation in selected cases; compact complete mechanism not established | [Completed mechanism follow-ups](reproducibility/completed_followups/README.md#carrier-reader-and-mediation); failed reserved-pair and reader qualification retained |
| Method-selection reanalysis | No comparison meets conventional equivalence and history-effect separation together | [Negative follow-up](reproducibility/completed_followups/README.md#negative-method-selection-reanalysis); distinct from Figure 10 |
| ValuePrism extension | Latest released benchmark-reuse status: technical stop before inference | [Released status only](reproducibility/completed_followups/README.md#valueprism-released-status); no scientific gate result or active outcomes |

The [September 15 project snapshot](artifacts/project/status.json) and
[status notebook](notebooks/00_project_status.ipynb) retain their original bytes
for historical replay. They are not the current external-validation status.

## Representation and decoding

The background establishes a readable support/opposition relation and the limits
of treating a successful direct intervention as a reusable semantic update.
[Figures 1 through 6 and supplementary plots](figures/CAPTIONS.md) link to
[representation evidence](reproducibility/representation/) and
[earlier editing evidence](reproducibility/relational_editing/).

```sh
python -m repro tables --output reproduced/all-tables
python -m repro figures --output reproduced/all-figures
```

Expected: all supported historical tables and ten main plus three supplementary
figures, each in PNG and PDF. Some background findings are aggregate-only;
the [study index](REPRODUCIBILITY.md#experiment-index) identifies them separately.

## Matched-history sufficiency

[Figure 7](figures/fig07_source_history.png) presents the prospective main result.
The [independent verification package](reproducibility/state_sufficiency/independent_verification/README.md)
contains saved scores, complete inputs, semantic mappings, references and repeats.

```sh
python -m repro.state_sufficiency --output reproduced/main
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
```

Expected: eight primary model/method cells reconstructed, all adjusted intervals
above zero; independent comparison matches eight cells, 896 benchmark case rows and 16,128
joint-query rows. A benchmark case is a target fact configuration
with several starting histories and questions. Joint questions combine facts.
The first command starts from case-level counts; the independent command checks
which questions meet the factual/reference requirements and still receive
different answers across histories.
Neither regenerates neural outputs or retrains editors.

## Post-hoc robustness

The [CPU audit](reproducibility/state_sufficiency/README.md#post-hoc-cpu-robustness-audit)
reuses those same frozen scores. It matches starting histories on changed-fact
count and exact prefix-token length, then checks stronger logical controls.
It is a post-hoc analysis, separate from the prospective primary test above.

```sh
python -m repro.cpu_robustness --output reproduced/cpu-audit
python -m pytest tests/test_cpu_robustness.py
```

The replay derives answers and control flags from
[saved response panels](reproducibility/state_sufficiency/independent_verification/inputs/responses/)
and [complete case designs](reproducibility/state_sufficiency/independent_verification/inputs/designs/).
Each panel is `inputs/responses/{model}/{root_id}/core.json.gz`. The replay uses
`physical[].logps`, `bindings` and `contexts[context].prefix_ids`.
Changed-fact counts and logical answers come from
the design's starting facts, assignment commands and question specifications.
The [input manifest](reproducibility/state_sufficiency/independent_verification/MANIFEST.json)
identifies the exact original archive, member and journal projection for each file.

| Claim | Canonical saved audit field | Replayed output |
|---|---|---|
| 653 original witnesses; 430 retain disagreement after matching | `RESULTS.csv`: `G0_witness`, `G0_M_witness` | `summary.json`: `original_witnesses`, `controls.G0.matched` |
| Same-pair control retains 188 opportunities across 53/64 designs | `G2a_M_witness`, unique `root_id` | `controls.G2a.matched`, `controls.G2a.designs` |
| Cross-pair control retains 82 across 34 designs | `G2b_M_witness`, unique `root_id` | `controls.G2b.matched`, `controls.G2b.designs` |
| Both controls retain 64 across 29 designs | `G2ab_M_witness`, unique `root_id` | `controls.G2ab.matched`, `controls.G2ab.designs` |
| Cross-pair survival is concentrated in learned updates | `G2b_M_witness` grouped by `method` | `controls.G2b.learned/textual`; `by_condition.csv` |

All fields above refer to the canonical
[RESULTS.csv](reproducibility/state_sufficiency/cpu_audit/RESULTS.csv).
[GATE_SPEC.json](reproducibility/state_sufficiency/cpu_audit/GATE_SPEC.json)
maps each target to its same-pair and cross-pair checks.
[SURVIVORS.json](reproducibility/state_sufficiency/cpu_audit/SURVIVORS.json)
retains the surviving examples and manual-review records.
The archived [audit report](reproducibility/state_sufficiency/cpu_audit/CPU_AUDIT_REPORT.md)
preserves the per-condition and group tables in a documented privacy derivative.
Machine-local paths were removed; original bytes remain externally hash-bound.
Use the portable command above for this checkout.

Expected: all 13,824 updating opportunities and 608,256 Boolean flag checks
agree. `verification.json` records the check; `summary.json` and
`by_condition.csv` give the counts without pooling designs as independent cases.
Cross-code sensitivity counts each semantic question once, on draw 0, and
requires the same history pair and direction of disagreement under both codes.

The [intake manifest](reproducibility/state_sufficiency/cpu_audit/MANIFEST.json)
binds the latest ZIP and retained provenance by hash. The earlier same-instance
package is superseded for the G2 interpretation, not invalidated or counted as
independent evidence. The
[technical recovery note](reproducibility/state_sufficiency/README.md#floating-point-recovery)
explains the original equality-check issue separately from the scientific result.

## Fresh confirmation and held-out validation

History-dependent answers survived the strongest controls in 14 of 64 new Gemma
cases. Those controls were fixed before outcomes. The practical benefit of
spreading validation across histories remains uncertain. This separate study
tests one learned-update family and leaves V10 as the original primary study.

The [method and replay guide](docs/FRESH_HISTORY_REPRODUCTION.md) explains what
changes, what stays fixed and what counts as a failure. The
[fully repeated Ada/Bryn example](reproducibility/fresh_history_validation/examples/CASE_0017.md)
connects a qualifying failure to its exact inputs and saved responses.

```sh
python -m repro.fresh_history --output reproduced/fresh-history
python -m pytest tests/test_fresh_history.py tests/test_saved_replay_portability.py
```

Expected: all cases and both stored seeds reconstruct, including 22 qualifying
target/seed observations. Four complete repeated cases retain identical scores
and answer/token bindings. The [result table](docs/FRESH_HISTORY_REPRODUCTION.md#results-and-statistical-units)
keeps the case denominators, uncertainty intervals and inconclusive utility
result together. Repeats and seeds remain measurements within cases.

Separate post-hoc diagnostics check withheld-target misses and exact token
inventories without changing either frozen endpoint.

## Pre-update deletion screen

Deleting the proposed messages increased downstream errors and failed four of
five admission requirements. The 128 fresh final cases were never run. The
conclusion concerns this fixed low-harm repair procedure; other upstream
influences remain unresolved.

The [compact package](reproducibility/preupdate_route/README.md) explains the
intervention and retains the original screen, case-level tables and figures.

```sh
python -m repro.preupdate_route
python -m pytest tests/test_preupdate_route.py
```

Expected: `PASS` for table consistency and the saved failed-gate decision.
Joint errors are 75/1,152 for the original computation, 129/1,152 for deletion,
and 72/1,152 for the scrambled control. Deletion introduces 38/768 direct errors.
The package's [full replay instructions](reproducibility/preupdate_route/README.md#full-saved-score-replay-with-the-external-archive)
use the hash-bound external archive to reconstruct the screen from raw saved
scores. The full archive stays outside normal Git.

## Causal cache intervention

[Figure 8](figures/fig08_cache_crossover.png) illustrates the
[six-case experiment](reproducibility/cache_crossover/README.md).

```sh
python -m repro.cache_crossover --output reproduced/causal-panel
```

Expected: 24 classifications and 288 identity controls. Under the main format,
five of six selected positive examples follow the history supplying the later-layer
activations; one is mixed. Under the alternate format, one follows that history,
two are mixed and three lack adequate original separation. Selection prevents
interpreting these counts as prevalence.

## Matched one-case extension

[Figure 9](figures/fig09_cache_crossover_extension.png) presents
[the matched extension](reproducibility/cache_crossover/README.md#one-case-extension).

```sh
python scripts/export_cache_crossover.py --output reproduced/causal-extension
```

Expected: the history supplying the later-layer activations controls the answer in both formats;
eight strict direct checks pass, with 108 baseline replays and 24 exact controls.
This is one retrospective case, separate from the six-case panel and all
prevalence denominators. Both commands normalize and classify saved raw scores.

## Question-blind repair

A fixed refresh of stored values partially corrected downstream answers before
the question was known. The [prospective result](reproducibility/question_blind_repair/README.md#prospective-result)
keeps the failed strong-repair threshold, introduced errors and better full
current-record reference visible. The [selected-case repeat](reproducibility/question_blind_repair/README.md#post-hoc-successful-case-reproducibility)
adds reproducibility, with the original population and intervals unchanged.

Start with the [Esme/Vera example](reproducibility/question_blind_repair/WORKED_EXAMPLE.md).
Its successful Yes/No repair appears beside the failed A/B-format repair.
The [caption and method notes](figures/CAPTION_DETAILS.md#question-blind-repair)
explain the manipulation and the correct-donor reverse effect.

```sh
python -m repro.question_blind_repair
python -m pytest tests/test_question_blind_repair.py
```

The compact check verifies source hashes, table accounting and scope. The
[full offline replay](reproducibility/question_blind_repair/README.md#offline-replay)
requires both hash-bound archives and compares the delivered scientific products
byte-for-byte. It makes no model calls.

## Public-data extension

The [Qwen/Ripple package](reproducibility/in_context_updates/README.md) retains
both frozen parsers and the same recorded responses. Its
[generated table](reproducibility/in_context_updates/public_data_table.md)
keeps the denominator and parser sensitivity visible without another figure.

```sh
python -m repro.in_context_updates --output reproduced/qwen
```

Expected: primary 3/98 qualifying history-dependent questions across three benchmark cases, alternative 2/98,
65/98 primary eligible questions (42/98 under the alternative), and accuracy on unrelated facts
170/272 without history versus 94/272 with history. Both repeated benchmark cases have
zero qualifying questions; all 86 repeated responses agree with the final pass.
Qualifying questions have different answers after different histories despite
correct current-fact and clean-reference checks.
These are in-context updates with fixed weights, not parameter editing.

## Update-method comparison

The [completed matched-history comparison](reproducibility/update_method_comparison/README.md)
and [Figure 10](figures/fig10_update_methods.png) show how update procedures change
downstream coherence, including uncertainty, seed variation and absolute rates.

```sh
python -m repro.update_methods --output reproduced/update-methods
```

Expected: 208,896 score rows and 6,144 telemetry rows reconstruct four primary
contrasts and the headline, per-case, per-scene and timing/memory CSVs. Both Qwen
improvements have intervals above zero; Gemma's intervals touch or cross zero.
No method establishes generally reliable performance on all questions. This is
supporting evidence, not the central matched-history sufficiency result.

## Textual breadth and full-forward 32B

The [textual breadth package](reproducibility/text_scope/README.md) compares
correction sentences and restated current facts on the original designs.
Basic history dependence recurs in the completed smaller-model rows. Failed
historical bridges prevent the planned cross-row comparisons. The cached 32B
path was not measured; it is not a zero result.

The separate [stateless 32B run](reproducibility/qwen32_full_forward/README.md)
completed those existing designs with full-prompt reads. Its strongest
cross-format criterion remains zero. The repeated
[Mina/Wren Yes/No example](reproducibility/qwen32_full_forward/WORKED_EXAMPLE.md)
passes narrower controls but fails qualification in the other format.
Neither study establishes a model-size trend.

```sh
python -m repro.text_scope --output reproduced/text-scope
python -m repro.qwen32_full_forward --output reproduced/qwen32
python -m repro.completed_followups
```

The first two commands reconstruct saved scores without inference. The last
checks the compact original reports and tables for the other completed follow-ups;
it is not a new tensor or neural replay.

## Reproduction levels and release boundaries

Inspecting figures, reconstructing saved measurements, regenerating neural
outputs and retraining are distinct levels. The first two are the supported
paper interfaces here. See [fresh-inference status](reproducibility/fresh_inference/README.md)
for the recovered portable entry point, exact implementation/checkpoints and
passed CPU smoke. Its optional pretrained path has not been run or certified.
The original historical notebook pins do not certify the current main result.

This attributed checkout preserves existing authorship, licenses and
upstream terms. The [privacy boundary](docs/PRIVACY_BOUNDARY.md) distinguishes
sanitized current files from unchanged historical Git objects.
