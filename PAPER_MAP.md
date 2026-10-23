# Paper and evidence map

> **Historical version.** This page maps an earlier manuscript, not the current
> paper. Use the [current paper-to-results map](history_updates/docs/paper_map.md)
> and [collaborator README](README.md). The historical content below is preserved.

Generated from [paper_map.json](paper_map.json) by `python scripts/generate_paper_map.py`.

Start with the [measured Ada and Bryn walkthrough](reproducibility/fresh_history_validation/examples/CASE_0017.md).
Figures 1–5 belong to the main text; Appendix Figures 6–7 belong to the appendix.
Additional repository analyses are identified separately; reproduction limits are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Manuscript appendix crosswalk

A01–A13 are repository topic IDs, distinct from the manuscript appendix letters.

| Repository topic | Manuscript location | Coverage |
|---|---|---|
| [A01](docs/appendix/A01_relation_construction.md) | Sections 2.1–2.2; Appendices A.1–A.2; Table 2 | Manuscript methods |
| [A02](docs/appendix/A02_controls.md) | Section 2.1; Appendices A.4–A.5; Table 3 | Manuscript controls plus the earlier specification |
| [A03](docs/appendix/A03_text_features.md) | Section 2.1; Appendix A.3 | Text-only features and separate fitted-population records |
| [A04](docs/appendix/A04_counterfactual_calibration.md) | Section 3; Appendices B.1–B.5; Figure 3 | Calibration, recovery and qualification |
| [A05](docs/appendix/A05_learned_editors.md) | Section 4.1; Appendices C.1–C.7 | Paired training, selection and the primary protocol |
| [A06](docs/appendix/A06_stronger_controls.md) | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | Original-output controls and fresh confirmation |
| [A07](docs/appendix/A07_textual_breadth.md) | Related to Section 4.4 and Appendix D.3 | Additional smaller-model studies; current 32B results are in the main matched-history guide |
| [A08](docs/appendix/A08_update_methods.md) | Related to Section 4.1 and Appendix C.4 | Additional paired method comparison, not the primary study |
| [A09](docs/appendix/A09_crossover.md) | Section 6.1; Appendix E.1; Figure 5 | Selected and matched crossovers |
| [A10](docs/appendix/A10_refresh_accounting.md) | Section 6.2; Appendix E.2; Appendix Figure 7 | Repair, harm, denominators and repeats |
| [A11](docs/appendix/A11_ripple_accounting.md) | Appendix F.2; Table 5 | Secondary RippleEdits-derived extension and parser sensitivity |
| [A12](docs/appendix/A12_mechanism.md) | Related to Sections 6 and 8 | Additional carrier-reader and mediation analyses; no separate manuscript appendix section |
| [A13](docs/06_mqmh_public_multihop.md) | Section 5; Appendix F.1; Table 1 | MQuAKE-derived public multi-hop matched-history evaluation |

## Figures

| Figure | Manuscript location | Asset | Evidence guide |
|---|---|---|---|
| Figure 1: Matched-history design | Section 1 | [PDF](figures/main/matched_history_design.pdf) · [PNG](figures/main/matched_history_design.png) | [Guide](docs/04_matched_histories.md) · `figure-1` |
| Figure 2: Relation readout | Section 2.2 | [PDF](figures/main/construct_validity.pdf) · [PNG](figures/main/construct_validity.png) | [Guide](docs/02_reading_relation.md) · `figure-2` |
| Figure 3: Direct-answer calibration and consequence recovery | Section 3.2 | [PDF](figures/main/consequence_qualification.pdf) · [PNG](figures/main/consequence_qualification.png) | [Guide](docs/03_changing_direct_answer.md) · `figure-3` |
| Figure 4: Prospective matched-history rates | Section 4.2 | [PDF](figures/main/matched_history.pdf) · [PNG](figures/main/matched_history.png) | [Guide](docs/04_matched_histories.md) · `figure-4` |
| Figure 5: Causal crossover | Section 6.1 | [PDF](figures/main/selected_crossover.pdf) · [PNG](figures/main/selected_crossover.png) | [Guide](docs/appendix/A09_crossover.md) · `figure-5` |
| Appendix Figure 6: Fresh confirmation and utility comparison | Appendix D.2 | [PDF](figures/main/fresh_confirmation_utility.pdf) · [PNG](figures/main/fresh_confirmation_utility.png) | [Guide](docs/appendix/A06_stronger_controls.md) · `figure-6` |
| Appendix Figure 7: Refresh benefits and new errors | Appendix E.2 | [PDF](figures/main/repair_benefit_damage.pdf) · [PNG](figures/main/repair_benefit_damage.png) | [Guide](docs/appendix/A10_refresh_accounting.md) · `figure-7` |

## Mapped claims, tables, and examples

| ID | Item | Type | Manuscript location | Guide | Study |
|---|---|---|---|---|---|
| `claim-readout` | Context-sensitive relation readout | claim | Section 2; Appendix A | [docs/02_reading_relation.md](docs/02_reading_relation.md) | `relation` |
| `claim-readout-null` | Relation permutation controls | claim | Section 2; Appendix A | [docs/02_reading_relation.md](docs/02_reading_relation.md) | `relation` |
| `claim-truth` | Factual readout controls | claim | Section 2; Appendix A | [docs/02_reading_relation.md](docs/02_reading_relation.md) | `relation` |
| `claim-text-comparator` | Text-only baseline and training data | claim | Section 2.1; Appendix A.3 | [docs/appendix/A03_text_features.md](docs/appendix/A03_text_features.md) | `relation` |
| `claim-recovery` | Direct-answer calibration and consequence recovery | claim | Section 3; Appendix B | [docs/03_changing_direct_answer.md](docs/03_changing_direct_answer.md) | `consequences` |
| `claim-failed-qualification` | Consequence-assay qualification | claim | Section 3; Appendix B | [docs/03_changing_direct_answer.md](docs/03_changing_direct_answer.md) | `consequences` |
| `claim-primary` | Prospective matched-history results | claim | Section 4; Appendix C | [docs/04_matched_histories.md](docs/04_matched_histories.md) | `primary` |
| `claim-selected-weights` | Paired-editor training and checkpoint selection | claim | Section 4.1; Appendix C.4 | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | `editors` |
| `claim-posthoc` | Matched-history controls on the original outputs | claim | Section 4.3; Appendix D.1 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `controls` |
| `claim-confirmation` | Fresh prospective confirmation | claim | Section 4.3; Appendix D.2 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `claim-utility` | Held-out update-validation comparison | claim | Section 4.3; Appendix D.2 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `claim-fresh-diagnostics` | Selected-case diagnostics | claim | Section 4.3; Appendix D.2 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `claim-small-breadth` | Smaller-model textual breadth | claim | Additional repository analysis; related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `claim-breadth-failures` | Earlier textual-method comparisons | claim | Additional repository analysis; related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `claim-stateless32` | Stateless 32B matched-history results | claim | Section 4.4; Appendix D.3; Table 4 | [docs/04_matched_histories.md](docs/04_matched_histories.md) | `qwen32` |
| `claim-six-crossovers` | Selected-case activation crossover | claim | Section 6.1; Appendix E.1 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover` |
| `claim-matched-crossover` | Matched crossover under stronger controls | claim | Section 6.1; Appendix E.1 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover-extension` |
| `claim-repair` | Refresh benefits and new errors | claim | Section 6.2; Appendix E.2 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `claim-refresh-repeats` | Refresh repeats and reverse transfers | claim | Section 6.2; Appendix E.2 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `claim-method-comparison` | Paired update-method comparison | claim | Additional repository analysis; related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `claim-public-data` | Public-data history dependence | claim | Appendix F.2; Table 5 | [docs/06_public_data_extension.md](docs/06_public_data_extension.md) | `ripple` |
| `claim-parser-sensitivity` | Parser sensitivity and eligibility | claim | Appendix F.2; Table 5 | [docs/06_public_data_extension.md](docs/06_public_data_extension.md) | `ripple` |
| `claim-locality-repeats` | Locality and negative repeat controls | claim | Appendix F.2; Table 5 | [docs/06_public_data_extension.md](docs/06_public_data_extension.md) | `ripple` |
| `claim-carrier` | Carrier-transfer follow-up | claim | Additional repository analyses; related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `claim-mediation` | Reader mediation and restoration controls | claim | Additional repository analyses; related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A01` | Relation construction, splits and statistics | supplementary_topic | Sections 2.1–2.2; Appendices A.1–A.2; Table 2 | [docs/appendix/A01_relation_construction.md](docs/appendix/A01_relation_construction.md) | `relation` |
| `A01-table-1` | Development-only split construction: table 1 | appendix_table | Sections 2.1–2.2; Appendices A.1–A.2; Table 2 | [docs/appendix/A01_relation_construction.md](docs/appendix/A01_relation_construction.md) | `relation` |
| `A02` | Permutation and factual controls | supplementary_topic | Section 2.1; Appendices A.4–A.5; Table 3 | [docs/appendix/A02_controls.md](docs/appendix/A02_controls.md) | `relation` |
| `A02-table-1` | Relation permutation nulls: table 1 | appendix_table | Section 2.1; Appendices A.4–A.5; Table 3 | [docs/appendix/A02_controls.md](docs/appendix/A02_controls.md) | `relation` |
| `A02-data-table-1` | split manifest | appendix_table | Section 2.1; Appendices A.4–A.5; Table 3 | [docs/appendix/A02_controls.md](docs/appendix/A02_controls.md) | `relation` |
| `A03` | Text-only baseline and training data | supplementary_topic | Section 2.1; Appendix A.3 | [docs/appendix/A03_text_features.md](docs/appendix/A03_text_features.md) | `relation` |
| `A04` | Calibration and consequence recovery | supplementary_topic | Section 3; Appendices B.1–B.5; Figure 3 | [docs/appendix/A04_counterfactual_calibration.md](docs/appendix/A04_counterfactual_calibration.md) | `consequences` |
| `A04-table-1` | Inputs and fixed comparisons: table 1 | appendix_table | Section 3; Appendices B.1–B.5; Figure 3 | [docs/appendix/A04_counterfactual_calibration.md](docs/appendix/A04_counterfactual_calibration.md) | `consequences` |
| `A05` | Paired-editor training and checkpoint selection | supplementary_topic | Section 4.1; Appendices C.1–C.7 | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | `editors` |
| `A05-table-1` | What the learned update changes: table 1 | appendix_table | Section 4.1; Appendices C.1–C.7 | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | `editors` |
| `A05-table-2` | What the learned update changes: table 2 | appendix_table | Section 4.1; Appendices C.1–C.7 | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | `editors` |
| `A05-table-3` | Objectives, seeds and actual selection: table 3 | appendix_table | Section 4.1; Appendices C.1–C.7 | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | `editors` |
| `A06` | A06. Stronger controls and fresh confirmation | supplementary_topic | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A06-table-1` | Original-output decomposition: table 1 | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `controls` |
| `A06-table-2` | Fresh cases and fixed criteria: table 2 | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A06-data-table-1` | full opportunity table | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `controls` |
| `A06-data-table-2` | acceptance and errors | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A06-data-table-3` | per-seed contrasts | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A06-data-table-4` | all retention levels | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A06-data-table-5` | both directions of policy disagreement | appendix_table | Section 4.3; Appendices D.1–D.2; Appendix Figure 6 | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | `fresh` |
| `A07` | Smaller-model textual breadth | supplementary_topic | Related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `A07-table-1` | Question and method: table 1 | appendix_table | Related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `A07-table-2` | Primary results: table 2 | appendix_table | Related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `A07-table-3` | Stronger-control accounting: table 3 | appendix_table | Related to Section 4.4 and Appendix D.3 | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | `textual-breadth` |
| `A08` | Update methods and downstream coherence | supplementary_topic | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-table-1` | Paired contrasts: table 1 | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-table-2` | Paired contrasts: table 2 | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-1` | per-design outcomes | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-2` | per-scene outcomes | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-3` | timing and fallback accounting | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-4` | behavior | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-5` | geometry | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A08-data-table-6` | checkpoint-selection metadata | appendix_table | Related to Section 4.1 and Appendix C.4 | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | `update-methods` |
| `A09` | Crossover details | supplementary_topic | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover-extension` |
| `A09-table-1` | Intervention and fixed inputs: table 1 | appendix_table | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover` |
| `A09-table-2` | Six-case panel and selection: table 2 | appendix_table | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover` |
| `A09-data-table-1` | complete 24-row panel | appendix_table | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover` |
| `A09-data-table-2` | joint table | appendix_table | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover-extension` |
| `A09-data-table-3` | direct table | appendix_table | Section 6.1; Appendix E.1; Figure 5 | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | `crossover-extension` |
| `A10` | Refresh accounting | supplementary_topic | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-table-1` | Fixed intervention and controls: table 1 | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-table-2` | Eligibility and repair: table 2 | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-table-3` | Benefit, harm and uncertainty: table 3 | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-data-table-1` | ALL_SEED_FORMAT_TARGETS.csv | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-data-table-2` | All condition/family rows | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-data-table-3` | FOUR_ENDPOINTS.csv | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-data-table-4` | worked-example questions | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A10-data-table-5` | Donor agreement | appendix_table | Section 6.2; Appendix E.2; Appendix Figure 7 | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | `refresh` |
| `A11` | Public-data accounting and parser sensitivity | supplementary_topic | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A11-table-1` | Candidate construction and parser rules: table 1 | appendix_table | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A11-table-2` | Eligibility, disagreement and qualification: table 2 | appendix_table | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A11-table-3` | Eligibility, disagreement and qualification: table 3 | appendix_table | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A11-table-4` | Final accounting and witnesses: table 4 | appendix_table | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A11-table-5` | Complete phase and repeat accounting: table 5 | appendix_table | Appendix F.2; Table 5 | [docs/appendix/A11_ripple_accounting.md](docs/appendix/A11_ripple_accounting.md) | `ripple` |
| `A12` | Selected-case carrier and mediation evidence | supplementary_topic | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A12-table-1` | Carrier-reader study: table 1 | appendix_table | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A12-table-2` | Loss and restoration controls: table 2 | appendix_table | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A12-data-table-1` | complete source tables | appendix_table | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A12-data-table-2` | all 2,172 score rows | appendix_table | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `A12-data-table-3` | full result tables | appendix_table | Related to Sections 6 and 8 | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | `mechanism` |
| `worked-example-0017` | Ada and Bryn: a measured matched-history example | worked_example | Section 4.3; Appendix D.2 | [reproducibility/fresh_history_validation/examples/CASE_0017.md](reproducibility/fresh_history_validation/examples/CASE_0017.md) | `fresh` |
| `claim-mqmh` | Public multi-hop history dependence | claim | Section 5; Appendix F.1; Table 1 | [docs/06_mqmh_public_multihop.md](docs/06_mqmh_public_multihop.md) | `mqmh` |
| `table-mqmh` | MQuAKE-derived public multi-hop results | main_table | Section 5; Table 1 | [docs/06_mqmh_public_multihop.md](docs/06_mqmh_public_multihop.md) | `mqmh` |
| `example-mqmh-1617` | Matched-history example: source case 1617 | worked_example | Section 5; Appendix F.1 | [docs/06_mqmh_public_multihop.md](docs/06_mqmh_public_multihop.md) | `mqmh` |

## Study replay index

Each study guide documents evidence, settings, commands, expected outputs, and reproduction limits.

| Study | Guide | Reproduction level | Command |
|---|---|---|---|
| `primary` | [docs/04_matched_histories.md](docs/04_matched_histories.md) | 2 counts plus independent saved-score reconstruction | `python -m repro.state_sufficiency --output reproduced/primary` |
| `relation` | [docs/02_reading_relation.md](docs/02_reading_relation.md) | 1 verified; 2 stored null tails and available factual gates only | `python -m repro.relation_readout --output reproduced/relation` |
| `consequences` | [docs/03_changing_direct_answer.md](docs/03_changing_direct_answer.md) | 2 saved margins; fresh execution unavailable | `python -m repro.counterfactual --output reproduced/consequences` |
| `crossover` | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | 2 saved journals; selected cases | `python -m repro.cache_crossover --output reproduced/crossover` |
| `fresh` | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | 2 saved-score confirmation, utility, sensitivities and repeats | `python -m repro.fresh_history --output reproduced/fresh` |
| `refresh` | [docs/appendix/A10_refresh_accounting.md](docs/appendix/A10_refresh_accounting.md) | 2 primary and selected-repeat score reconstruction; original intervals retained, not regenerated | `python -m repro.question_blind_repair --output reproduced/refresh` |
| `editors` | [docs/appendix/A05_learned_editors.md](docs/appendix/A05_learned_editors.md) | 2 original CPU parameter/interface checks; paired-stage source and FIT/CAL records supplied as an inert reference; fresh retraining unverified | `python -m repro.learned_editors --output reproduced/editors` |
| `controls` | [docs/appendix/A06_stronger_controls.md](docs/appendix/A06_stronger_controls.md) | 2 original saved scores; post-hoc controls | `python -m repro.cpu_robustness --output reproduced/controls` |
| `textual-breadth` | [docs/appendix/A07_textual_breadth.md](docs/appendix/A07_textual_breadth.md) | 2 saved scores; failed bridges and unmeasured rows retained | `python -m repro.text_scope --output reproduced/textual-breadth` |
| `qwen32` | [docs/04_matched_histories.md](docs/04_matched_histories.md) | 2 stateless saved outputs; existing designs | `python -m repro.qwen32_full_forward --output reproduced/qwen32` |
| `crossover-extension` | [docs/appendix/A09_crossover.md](docs/appendix/A09_crossover.md) | 2 saved journal and original baseline bindings; one selected pair | `python -m repro.cache_crossover_extension --output reproduced/crossover-extension` |
| `update-methods` | [docs/appendix/A08_update_methods.md](docs/appendix/A08_update_methods.md) | 2 completed origin panel saved scores and telemetry | `python -m repro.update_methods --output reproduced/update-methods` |
| `ripple` | [docs/06_public_data_extension.md](docs/06_public_data_extension.md) | 2 both frozen parsers and all retained phases | `python -m repro.in_context_updates --output reproduced/ripple` |
| `mechanism` | [docs/appendix/A12_mechanism.md](docs/appendix/A12_mechanism.md) | 1 saved tables with score/effect accounting checks; native tensors unavailable | `python -m repro.completed_followups --output reproduced/mechanism` |
| `mqmh` | [docs/06_mqmh_public_multihop.md](docs/06_mqmh_public_multihop.md) | 2 saved-response reconstruction with subject-clustered bootstrap | `python -m repro.mqmh --output reproduced/mqmh` |

## Figure rendering

Render the five main-text and two appendix figures from retained measurements:

```sh
python -m repro.main_figures --output reproduced/main-figures
```
