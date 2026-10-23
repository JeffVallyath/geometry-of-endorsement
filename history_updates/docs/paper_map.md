# Paper-to-evidence map

Start with the [Figure 1 synthetic walkthrough](../../examples/synthetic/README.md)
or the [public walkthrough](../../examples/public/README.md). Read the
[paper](../../papers/final-paper.pdf) alongside this map. The
[experiment overview](../../docs/experiment_overview.md) separates the study units.
Browse the results under [results/tables](../results/tables).

Tables are generated from completed observation replays. Each JSON table records
its source outputs and units. CSV, Markdown and LaTeX views use the same rows.
The [study index](../metadata/study_index.json) maps each retained source path to
its included code or data file. The output names below are relative to a replay's
tables/ directory. This map follows the final paper with 13 tables and eight
figures. Output filenames remain stable and do not necessarily match the paper's
current numbering.

The [machine-readable map](paper_map.json) resolves each table, figure and
quantitative-passage group to its included analyzers and observation files.

## Tables

| Paper item | Section command | Generated output | Observations and calculation |
|---|---|---|---|
| Table 1 | matched_histories | docs/methods.md (repository); retained eligibility checks | Qualitative definitions of the single-fact and two reference checks, not an additional numerical table |
| Table 2 | matched_histories | table_02 | The fresh synthetic example, matched to its recomputed witness row and all 20 checks per history |
| Table 3 | matched_histories | table_01 | Saved multi-hop responses, frozen alias rules, whole-case intervals and guarded relevance contrasts |
| Table 4 | external_state | table_06 | Cache refresh, saved-direct-answer recomposition and neural-compiler observations on the same cohort |
| Table 5 | interpreter_checks | aggregate table_10 | Mean complete-case percentages over both seeds: 98% dictionary only, 90% with final-fact text, 56% with original history |
| Table 6 | matched_histories | table_03 | Per-case synthetic witness classifications; both activation seeds remain linked |
| Table 7 | matched_histories | table_04 | Parsed responses for the Levada case under all four histories and both references |
| Table 8 | cache_interventions | table_05 | Complete cache-refresh score recount, including lost qualification and introduced errors |
| Table 9 | external_state | table_07 | Complete-case correctness from all three repair-model analyses, with separate seeds and baselines |
| Table 10 | external_state | table_08 | Original whole-subject paired bootstrap calculations |
| Table 11 | external_state | table_09 | Every distinct current-query refusal and incorrect accepted answer |
| Table 12 | interpreter_checks | table_10 | Both unchanged interpreters with no additional text, original history and final-fact comparison text |
| Table 13, trained readers and MeLLo | interpreter_checks | table_11 | The seven current reader rows, including both seeds of each trained reader |
| Table 13, corrected GMeLLo; Appendix G contrasts | gmello_comparison | gmello_corrected; gmello_contrasts | CORE (six facts), FULL (all facts) and TEXT (text edits); both inherited repair seeds and two primary intervals |
| Historical result, not a current paper row | interpreter_checks | legacy_gmello | The superseded implementation's 35/128 observation, kept separately |

Table 13 uses the three GMeLLo rows from gmello_corrected alongside table_11.
The repair rows also present in gmello_corrected are shared comparators, not
additional rows to duplicate. The old and corrected implementations have
separate source records and labels; the corrected replay checks all 34,710
linked reads.

## Figures

The supplied paper preserves the approved figure designs. This map is the
current paper-to-data cross-reference. A schematic is distinguished from a
measured plot.

| Figure | Evidence binding | Reconstruction level |
|---|---|---|
| 1: Matched histories | table_02; claims.json, synthetic_worked_example | Synthetic histories H5/H6, their checks, 455-token contexts and recorded answers; author-designed layout |
| 2: Synthetic rates | table_03 | Rates and intervals reconstructed from saved scores |
| 3: Cache crossover | claims.json, cache_crossover/example | Saved probabilities and direct checks replayed |
| 4: External state | Exact dictionary implementation and methods.md | Architecture schematic; preserved in the supplied PDF |
| 5: Repair results | table_07 | Complete-case counts reconstructed; trained bars average the two seeds |
| 6: Single cache sites | claims.json, cache_sites | Recorded individual-site effects and all four direction/format conditions |
| 7: Leave-one-out sites | claims.json, cache_sites | Recorded leave-one-out effects on the same case and scale |
| 8: Cache refresh | claims.json, cache_refresh | Original paired changes, intervals, corrections and harms |

Editable source for the final PDF's complete figure layouts was not supplied.
Available scientific renderers and their generated data are retained. The
package does not substitute an old manuscript or redesign the supplied figures.
The complete command also writes the retained source renderings for Figures 2,
3 and 8 under figures/, with exact plotting inputs in plot_data.json. The fresh
confirmation and larger-model analyzers retain their own generated plots.

Replay-generated figure_bindings.json and paper-alignment labels retain the
earlier manuscript's identifiers. In particular, its Figure 1 binding refers to
the public worked example, now Table 7; use this map's synthetic-example binding
for the final paper's Figure 1. No recorded measurement or renderer is changed
by this documentation cross-reference.

## Quantitative passages

claims.json retains the detailed measurements behind the prose, beyond the
compact table views. The mappings below name its top-level entries.

| Paper passage | Claim collection | Coverage |
|---|---|---|
| Abstract, introduction and conclusion | paper_averages; cache_refresh; public_repair | The rounded 99% versus 38% complete-case averages, 43% internal repair rate and all-model witness-repair statement |
| Section 2 and Appendix A | synthetic_worked_example | History matching, token lengths, direct checks, predictions and probabilities |
| Section 3.1 and Appendix B | synthetic_primary | Eight model/update cells, intervals, denominators and affected cases |
| Section 3.1 and Appendix B, stricter controls | stricter_controls; fresh_confirmation | Original, structurally matched and fully controlled counts; prospective cases and both seeds |
| Section 3.2 and Appendix C | public_matched_histories | Unconditional and guarded counts, long chains, canonical-only sensitivity and repeats |
| Section 3.3 and Appendices B-C | paired_scale; synthetic_larger_model; gemma_native | Shared-prompt size comparison and separately labeled larger-model measurements |
| Section 4.1 and Appendix D | cache_crossover; cache_sites; cache_reader_checks; cache_reader_effects | Selected cases, direct checks, individual sites, reserved cases and reader interventions; reader fractions checked against saved log probabilities |
| Section 4.2 and Appendix D | cache_refresh | Repaired and new witnesses, changed qualification, total answer errors, controls and intervals |
| Section 5.2 and Appendix E | synthetic_recomposition; synthetic_compiler | Exact cohort joins, supplied-operation recomposition, compiler/intake accuracy and reuse across editor seeds |
| Appendix E | synthetic_compiler | History-visible interpreter/restatement arms, new cases, graph tasks, public questions and failures |
| Section 5.3 and Appendix F | public_repair; paper_averages | All conditions/seeds, paired gains, witness repairs, controls and averages |
| Section 5.4 and Appendix G, exposure | history_exposure | Subject losses, paired intervals, question-level changes and retained unfavorable outcomes |
| Appendix G, reader choice | common_state_readers; common_state_matched_snapshots | Complete cases, controls, paired contrasts and exactly matched original/current questions |
| Appendix F training and shared questions | training_sample_counts; common_state_training_exposure_join; common_state_cost_timing_workload | All 12,288 presentations, 7,507 distinct pairs, 1,024 source cases, retained order, control overlap and measured calls/tokens/times |
| Appendix G token matching | history_exposure_token_matching | Exact regrouping of all 12,198 candidate contexts; 5,798 equal-length history/final-fact pairs |
| Appendix G original-snapshot errors | common_state_matched_snapshots | The 768 changed-answer reads and 226/440 updated-entity answers on original inputs, with both seeds separate |
| Limitations, internal-repair error change | cache_refresh; table_05 | The net reduction of 597 errors among 73,728 combined answers, about 0.8 percentage points; distinct from witness creation and repair |
| Appendix G, component errors | common_state_behavior_component_audit | Wrong-operation/right-endpoint coincidences, graph/fallback decisions and refusal categories |
| Appendix G, corrected GMeLLo and Table 13 | corrected_gmello | Primary CORE/FULL contrasts, secondary TEXT results, original snapshots, controls and witness cohorts |

Hyperparameters and model/data revisions remain in the included configurations
and training records. They describe the recorded methods; replay does not train
models again. See [reproduction.md](reproduction.md) for fresh-model availability.
Related-work citations do not correspond to local experimental claims.
