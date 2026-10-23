# What was tested

Passing the updated-fact checks does not ensure that downstream answers are
independent of history. The studies below test that finding, its internal
mechanisms and a separate external-state remedy.

| Study | Input and manipulation | Held fixed | Measurement and result | Implementation and boundary |
|---|---|---|---|---|
| Original synthetic | 64 fictional cases; eight histories; textual corrections or activation setters | Final facts and downstream question | 18 scheduled combined questions per case; rates 2.4%-6.3% across eight model/update settings | [Primary analyzer](../history_updates/experiments/matched_histories/v3/studies/original_public_history/src/repro/state_sufficiency.py); activation seeds are averaged within case |
| Stricter synthetic | Reanalysis of those original cases | All direct and other-combined checks; compared pairs match changed-fact count and token length | 29/64 cases (45%) pooled over settings; 64 surviving question/setting observations | [Controls](../history_updates/experiments/matched_histories/v3/studies/original_public_history/src/repro/cpu_robustness.py); controls selected after inspecting the initial results |
| Fresh synthetic | 64 new cases with the controls fixed before evaluation | Original frozen Gemma activation setters, both seeds | 14/64 cases (22%); 0.95% of scheduled questions | [Frozen gates](../history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/analyze.py); Figure 1 teaches one of these cases |
| Public MQuAKE | 128 cases, three target phrasings each, four histories; position and relevance of outdated facts | Corrections, final facts and question | 94/384 questions for Qwen2.5-7B; 101/384 for Gemma-2-9B | [Parser and guards](../history_updates/experiments/interpreter_checks/v3/studies/history_exposure/sources/ppr1/reference/mqmh.py); supplied contextual facts, not weight editing |
| Larger models | Same public cohort with paired Qwen wrapper; larger Gemma and synthetic Qwen settings | Cohort and, where specified, wrapper | Public paired Qwen 57/384 versus 9/384; Gemma-2-27B 41/384 | [Paired-scale study](../history_updates/experiments/matched_histories/v3/studies/paired_native_scale/MQMH_SCALE_V1/implementation/study.py); Gemma sizes did not share one wrapper |
| Cache interventions | Selected synthetic disagreements and a 256-case refresh panel | Questions, relevant checks and frozen intervention selection | Refresh repairs 36/83 witnesses, creates 40 new ones; total combined-answer errors fall from 3,490 to 2,893 | [Cache refresh](../history_updates/experiments/matched_histories/v3/studies/original_public_history/src/repro/question_blind_repair.py); repair counts and total errors are different measures |
| External state | Independently update current dictionaries; interpreter selects an operation for exact execution | Final dictionary under matched histories | Public complete cases average about 99%, versus 38% from corrected text; synthetic bridge repairs all 83 witnesses | [Recorded execution](../history_updates/experiments/external_state/v3/src/repro/example.py), [full results](../history_updates/results/tables/table_07.md); trained and untrained whole-system comparison on an inspected cohort |
| History exposure | Add historical or current-only text to the trained reader | Dictionary, question, weights, candidates and executor | Mean complete cases: 98% dictionary only, 90% current-only extra text, 56% historical text | [Results](../history_updates/results/tables/table_10.md); all new historical-text errors are refusals |
| Learned readers | QA writes an answer; PROGRAM writes an operation; ranker chooses one | Prepared current state, training examples/order and budget | Complete cases: QA 126/128 and 122/128; PROGRAM 126/128 in each seed; ranker 124/128 and 127/128 | [Full comparison](../history_updates/results/tables/table_11.md); PROGRAM shares the same exact executor, QA shares the prepared state |
| Corrected GMeLLo | Independent local graph; six-fact, full-fact or text-edit variants | Available facts and linked repair comparators | CORE 91/128, FULL 89/128, secondary TEXT 46/128 complete cases | [Full comparison](../history_updates/results/tables/gmello_corrected.md), [contrasts](../history_updates/results/tables/gmello_contrasts.md); untrained local reimplementation, not the published full-Wikidata system |

The original GMeLLo 35/128 result is retained only in the explicitly superseded
[legacy table](../history_updates/results/tables/legacy_gmello.md). It is not the
current Table 13 result. The learned-reader near-tie is not evidence of equivalence
or of unique superiority for ranking.

The [paper map](../history_updates/docs/paper_map.md) links every table, figure
and quantitative passage to its source analyses. The
[methods](../history_updates/docs/methods.md) define the checks and distinguish
state isolation from correct answering.
