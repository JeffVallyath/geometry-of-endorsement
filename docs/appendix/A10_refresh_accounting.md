# Refresh accounting

Repository topic A10. See manuscript Section 6.2, Appendix E.2 and Appendix Figure 7.

## Fixed intervention and controls

The Gemma study tests 256 designs, eight earlier histories and two learned-editor seeds. Each case/seed has 18 scheduled joint targets across two answer formats. All 9,216 target observations remain in the primary denominator.

The input is a learned-updated context. A complete current-record reference is built using the same learned update. Before opening any question, the intervention copies reference values into aligned trailing positions. Layer indices are zero-based and inclusive.

| Condition | Stored state used for answering |
|---|---|
| A | Original learned-updated context |
| B | Reference values substituted at layers 16–23 |
| C | Comparison substitution at layers 24–31 |
| D | Full current-record reference, using origin zero within each seed |
| RB / RC | Selected reverse transfers from the updated context into the full reference |

Keys, values outside the mask, model weights and the recipient prompt stay fixed for partial transfers. Every question uses an independent copy. The full reference is required to construct both partial refreshes. The experiment reports no compute advantage over that reference.

The primary analysis uses end-offset alignment. Its separately recorded exact-absolute-position subset contains zero cases. Empty suffixes use identity behavior; unmappable positions remain failures under the [frozen analysis rules](../../reproducibility/question_blind_repair/supplied/primary/ANALYSIS_RULES.md).

## Eligibility and repair

The original qualifying cohort is locked from condition A before analyzing interventions. A qualifying failure requires correct current-fact and logical controls, correct clean references and a valid disagreement between structurally matched histories. Full-vocabulary validity and the inherited answer-mass checks apply.

A repair makes the target valid and correct across all eight histories. It also preserves all 12 direct questions, two same-pair logical functions and six other-pair questions in the same format. Wrong consensus and lost qualification receive no repair credit. At most one of the nine semantic targets per seed/format can satisfy the complete witness rule.

| Population accounting | Value |
|---|---:|
| Designs / baseline-witness designs | 256 / 63 |
| Baseline qualifying observations | 83 |
| Targeted / comparison repairs | 36 / 2 |
| Designs with targeted repair | 28 |
| Targeted repair fraction within original cohort | 36/83 = 43.37% |
| Original qualifying failures retained | 35 |
| Original observations losing qualification without repair | 12 |
| Newly qualifying failures after targeted refresh | 40 |
| All qualifying failures after targeted refresh | 35 + 40 = 75 |

The original cohort partitions as 83 = 36 + 35 + 12. The post-treatment total of 75 uses a different cohort and is not a repair rate. Exact rows are in [ALL_SEED_FORMAT_TARGETS.csv](../../reproducibility/question_blind_repair/supplied/primary/replay/ALL_SEED_FORMAT_TARGETS.csv).

## Benefit, harm and uncertainty

| Joint-answer accounting | Value |
|---|---:|
| Scheduled responses per condition | 73,728 |
| Errors in A / B / D | 3,490 / 2,893 / 2,000 |
| B joint corrections / introduced errors | 944 / 347 |
| B direct corrections / introduced errors | 84 / 76 |
| B net joint-error reduction | 597/73,728 = 0.81 percentage points |

Condition B damages a previously complete case in 11 seed-zero cases and three seed-one cases. [All condition/family rows](../../reproducibility/question_blind_repair/supplied/primary/tables/ALL_SEED_FORMAT_FAMILIES.csv) retain invalid responses and harm for B, C and D. The [claim ledger](../../reproducibility/question_blind_repair/supplied/verification/CLAIM_LEDGER.json) includes oppositional-question damage.

The four endpoints are efficacy, net joint-error change, new direct errors and new joint errors. Efficacy averages the B-minus-C baseline-witness repair indicator over all 36 scheduled targets per design. Harm uses all scheduled responses of the relevant family. Invalid responses remain errors.

The frozen bootstrap samples 256 case designs with replacement. Each draw preserves all within-design seeds, histories, questions and conditions. It uses 10,000 draws, NumPy `default_rng(2609184202)` and quantiles 0.00625 and 0.99375. The resulting 98.75% intervals form the four-comparison family. Exact estimates and intervals are in [FOUR_ENDPOINTS.csv](../../reproducibility/question_blind_repair/supplied/primary/tables/FOUR_ENDPOINTS.csv).

Strong repair requires at least 20 distinct baseline-witness cases, positive efficacy in both seeds and a positive efficacy lower bound. It also requires at least 50% of baseline observations repaired, a negative net-error upper bound, no seed-level net increase and both harm upper bounds at most one percentage point. Only the 50% condition fails. Selected repeats do not change these intervals or thresholds.

## Reverse transfers and repeat selection

The original reverse screen retains all 32 fixed cases. Ten have an eligible targeted reverse effect, representing both semantic directions. Nineteen cases have a clean-reference failure. The original corroboration threshold is at least eight eligible cases out of 32. These results remain distinct from the later two-case reverse illustration.

The outcome-independent repeat selection combines four hash-selected cases with the first four baseline-witness cases in frozen order. Deduplication has no fill rule. That selection correction occurred before model outcomes. Eight cases and 16 seed blocks pass the recorded comparison. This original repeat panel contains zero full-criterion B repairs.

The later extension selects the first four successful cases in frozen order plus two forward/reverse overlap cases. Six successful-case designs and one technical anchor produce 14 case/seed blocks and 14,076 scored reads. All retained score and binding fields match exactly in the recorded comparison. The selected six designs contain nine baseline qualifying observations, eight B repairs and zero C repairs. They add zero independent designs.

The numerical comparison tolerance is `1e-6`. Only `batch_seconds`, `qualification_seconds`, `preprocessing_seconds` and `compile_seconds` are excluded. Answers, validity, tokens and non-timing bindings are exact checks. [REPEAT_COMPARISON.json](../../reproducibility/question_blind_repair/supplied/verification/REPEAT_COMPARISON.json) retains all 14 comparisons. [SELECTED_RECOUNT.json](../../reproducibility/question_blind_repair/supplied/verification/SELECTED_RECOUNT.json) retains selected failures and harm.

## Worked example and interpretation

Case `QBR1-FINAL-0127`, seed 1, repairs the illustrated Yes/No target but fails full repair in the A/B format. The five [worked-example questions](../../reproducibility/question_blind_repair/supplied/verification/WORKED_EXAMPLE.csv) are independently asked. Their direct and other logical controls stay correct.

For the illustrated SAME question, conditions A/history1, A/history2, B/history2, C/history2, D, RB/history2 and RC/history2 answer `Yes, No, Yes, No, Yes, No, Yes`. In case `QBR1-FINAL-0146`, seed 0, reverse errors occur at origins 6 and 7 despite correct donor responses. [Donor agreement](../../reproducibility/question_blind_repair/supplied/verification/DONOR_AGREEMENT.csv) therefore distinguishes causal influence from simple copying of an already wrong answer.

## Replay boundary

```sh
python -m repro.question_blind_repair --output reproduced/refresh
```

This command reconstructs the primary gates and harms from 512 case/seed score blocks.
It also compares 14 selected-repeat blocks and rebuilds four selected-case tables.
The [original decoder and comparator](../../reproducibility/question_blind_repair/supplied/verification/tools/replay.py)
and all scientific counting rules are unchanged. The new wrapper reads the
[public score projections](../../reproducibility/question_blind_repair/scores/DATA.json).

`PRIMARY_RECOUNT.json` and `SELECTED_RECOUNT.json` contain the reconstructed counts.
`REPEAT_SCORE_COMPARISON.json` checks the retained score, token and alias fields.
Native-cache receipts and raw journals are not reconstructed from that projection.
The four CSV outputs reproduce the retained selected-case tables byte-for-byte.
`RESULTS.json` also reconciles the original intervals without rerunning their bootstrap.
The [reproduction matrix](../../REPRODUCIBILITY.md#reproduction-levels) states the remaining execution limits.
