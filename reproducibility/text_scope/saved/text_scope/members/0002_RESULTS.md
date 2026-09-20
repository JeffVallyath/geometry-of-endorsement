# V10 textual-update scope: final results

Three of four fixed new model/template rows completed all 64 roots and both fresh-process repeat passes. Qwen32/T4 is NOT_MEASURED because its cached/full DEV predictions disagreed; it is not a zero result.

EC = EXISTING_CORRECTION; LSW = LATEST_SAME_WORDING. A certifies both operands across histories; C certifies native and same-method no-op references; E=A&C; W=E&D. All rates retain 1,152 joint opportunities per row/method. Complete-case success uses all 34 questions across all eight histories, over 64 roots. Historical rows are reconstructed, not fresh inference.

| Model/template | Method | A | C | E | W / 1,152 | W/E | Strong | Complete case |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma9/C1 (historical) | EC | 100.00% | 78.12% | 78.12% | 69 (5.99%) | 7.67% | 20 | 15.62% |
| Gemma9/C1 (historical) | LSW | 99.48% | 78.30% | 77.95% | 64 (5.56%) | 7.13% | 22 | 15.62% |
| Qwen8/T4 (historical) | EC | 85.42% | 71.01% | 62.85% | 71 (6.16%) | 9.81% | 36 | 0.00% |
| Qwen8/T4 (historical) | LSW | 88.02% | 70.66% | 63.89% | 73 (6.34%) | 9.92% | 33 | 1.56% |
| Gemma9/T4 | EC | 100.00% | 84.29% | 84.29% | 104 (9.03%) | 10.71% | 41 | 20.31% |
| Gemma9/T4 | LSW | 100.00% | 83.59% | 83.59% | 93 (8.07%) | 9.66% | 38 | 15.62% |
| Llama8/T4 | EC | 76.30% | 63.45% | 52.26% | 61 (5.30%) | 10.13% | 0 | 0.00% |
| Llama8/T4 | LSW | 76.30% | 63.37% | 52.43% | 68 (5.90%) | 11.26% | 0 | 0.00% |
| Qwen32/T4 | EC | NA | NA | NA | NA (technical) | NA | NA | NA |
| Qwen32/T4 | LSW | NA | NA | NA | NA (technical) | NA | NA | NA |
| Qwen4/T4 | EC | 91.67% | 72.31% | 67.36% | 115 (9.98%) | 14.82% | 82 | 3.12% |
| Qwen4/T4 | LSW | 96.61% | 72.14% | 70.31% | 110 (9.55%) | 13.58% | 72 | 4.69% |

Primary witnesses were observed in 6/6 measured new method cells, spanning Gemma9/T4, Llama8/T4, Qwen4/T4. This extends the observed phenomenon within the fixed T4 assay; it does not establish population prevalence or a quantitative difference from the unbridged historical runs.

For Llama specifically, certified E coverage is 52.26% / 52.43%, and all-origin joint correctness is 58.77% / 57.99% (EC / LSW). Strong witnesses are 0 / 0, and complete-case success is 0.00% / 0.00%. Its primary witnesses do not erase these coverage, confidence and downstream-correctness limitations.

## New primary uncertainty

Root-block percentile bootstrap: 10,000 draws, PCG64 seed 2609141002. The planned eight-cell family is retained despite Qwen32 being unmeasured. Historical corrected intervals remain their original family, not pooled with these results.

| Model/template | Method | W rate | 95% interval | Family-eight 99.375% interval | Witness roots |
|---|---|---:|---|---|---:|
| Gemma9/T4 | EC | 9.03% | [6.77%, 11.46%] | [5.99%, 12.59%] | 36/64 |
| Gemma9/T4 | LSW | 8.07% | [6.08%, 10.07%] | [5.47%, 11.02%] | 41/64 |
| Llama8/T4 | EC | 5.30% | [3.99%, 6.68%] | [3.47%, 7.36%] | 36/64 |
| Llama8/T4 | LSW | 5.90% | [4.51%, 7.29%] | [3.99%, 7.81%] | 38/64 |
| Qwen32/T4 | EC | NA | NA | NA | NA |
| Qwen32/T4 | LSW | NA | NA | NA | NA |
| Qwen4/T4 | EC | 9.98% | [7.47%, 12.76%] | [6.42%, 13.72%] | 44/64 |
| Qwen4/T4 | LSW | 9.55% | [7.12%, 12.24%] | [6.08%, 13.26%] | 42/64 |

## Planned comparisons

| Planned paired comparison | Method | Result |
|---|---|---|
| Gemma9/T4 minus Gemma9/C1 (historical) | EC | NA: Historical bridge failed |
| Gemma9/T4 minus Gemma9/C1 (historical) | LSW | NA: Historical bridge failed |
| Qwen4/T4 minus Qwen8/T4 (historical) | EC | NA: Historical bridge failed |
| Qwen4/T4 minus Qwen8/T4 (historical) | LSW | NA: Historical bridge failed |
| Qwen32/T4 minus Qwen8/T4 (historical) | EC | NA: Missing Qwen32 row and failed historical bridge |
| Qwen32/T4 minus Qwen8/T4 (historical) | LSW | NA: Missing Qwen32 row and failed historical bridge |

The six planned contrasts retain their planned family (99.1666667% corrected intervals if assessable); no paired estimates, superiority claims or jointly certified old/new intersections are reported because the bridges failed. The Qwen three-way intersection is also unavailable.

## Interpretation and limits

The measurements test factual readability versus correct all-history downstream use on the same 64 existing designs. Coverage, conditional W, all-origin joint accuracy, whole-battery and complete-case correctness must be read together. Low coverage and invalid answers remain in the primary denominator; neither is a reason to drop an actor.

The new primary rows stand on their own technical checks, but failed historical numerical bridges prevent quantitative template or Qwen-ladder conclusions. This is not a size-only causal law, new-domain evidence, a mechanism replication, a repair, or a pass of the older H3/G3 contract.

The DEV gate was explicitly amended after all DEV outputs, before new FINAL outputs: an erroneous reuse of historical replay numerical caps was replaced by the inherited E0 decision/cache-integrity criterion. The original failures and all numerical discrepancies are retained; no DEV scores were rerun. This is an exposed, authorized amendment, not unchanged preregistration. Qwen32 still failed the amended criterion. A separate Llama metadata-sentinel repair did not alter task rendering or scores.

Both historical bridges remain failed. The previous premature global shutdown and its replacement allocation are disclosed in operator lineage and incident records. Resource, score and forward budgets include both allocations and technical work, without resets.

All three measured rows completed independent repeat0 and repeat1. Numerical/discrete differences are reported in runtime/checks/REPEATS_*.json, never averaged into FINAL or used to regenerate a preferred answer. Query-level sensitivity, answer-code, candidate-mass and distributional-history records accompany the main tables.

A zero-width zero-count bootstrap does not establish invariance. Any supplied zero-root upper bound assumes independent-root sampling and applies to the event “any witness,” not to 1,152 correlated questions. Candidate-normalized probabilities are not calibrated beliefs.

See runtime/analysis_final for full numeric tables and root/query records, runtime/results for raw scores and both repeats, and operator/ for resource and custody evidence. No model weights or activation dataset are distributed.
