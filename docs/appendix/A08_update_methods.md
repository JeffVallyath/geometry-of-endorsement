# Update methods and downstream coherence

Repository topic A08 is an additional paired update-method comparison. It is
related to manuscript Section 4.1 and Appendix C.4, but is not reported there as
the primary matched-history study.

## Method and denominator

This separate study measures complete-case reliability after updating facts. Each model has 64 fixed designs, four starting histories and 34 questions per history. A design succeeds only when all 136 answers are valid and correct. Both learned seeds are retained and averaged within each design.

The constrained learned update and the unrestricted consistency-trained update are compared with a factual-update baseline. That baseline attempts an internal fact refresh and supplies the latest textual correction. Its machine identifier is `FIELD_PLUS_LATEST_ERRATUM`. The saved baseline falls back in 192 of 256 contexts for each model.

The complete recorded panel also includes the original context, rebuilding, existing correction, latest same wording, latest erratum and two canonical-replay seeds. All 12 conditions per model remain in the score records and tables. The delivered data cover the matched-origin panel. Planned additional panels were not delivered as completed measurements.

The models are `google/gemma-2-9b-it` at revision `11c9b309abf73637e4b6f9a3fa1e92e615547819` and `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`. [Model settings](../../reproducibility/update_method_comparison/inputs/models.json) retain BF16 backbones, FP32 edit workspace, layer conventions and scorer settings.

## Paired contrasts

Each design contributes its seed-mean learned success minus one baseline success indicator. Models remain separate. The paired bootstrap resamples 64 designs with replacement, preserving both seed effects within each design. It uses 10,000 draws, seed `260913902`, and quantiles 0.00625 and 0.99375. The four contrasts form a 98.75% interval family. An interval touching or crossing zero is not evidence of equivalence.

| Model | Learned update | Effect, percentage points | 98.75% interval |
|---|---|---:|---:|
| Gemma | Constrained | 15.625 | [0, 30.46875] |
| Gemma | Consistency-trained | 13.28125 | [-0.78125, 27.34375] |
| Qwen | Constrained | 31.25 | [20.3125, 43.75] |
| Qwen | Consistency-trained | 39.84375 | [28.90625, 50.78125] |

Both Qwen intervals exclude zero. Gemma is mixed. Its constrained method succeeds on 30/64 designs in seed zero and 12/64 in seed one.

| Model | Baseline | Constrained, seed mean | Consistency-trained, seed mean |
|---|---:|---:|---:|
| Gemma | 11/64 | 32.8125% | 30.46875% |
| Qwen | 1/64 | 32.8125% | 41.40625% |

The gains coexist with limited absolute reliability. The supported conclusion is to measure factual-update performance and downstream coherence separately. This study does not establish a method-selection reversal from the matched-history test.

## Evidence and cost accounting

The [complete result](../../reproducibility/update_method_comparison/results.json) contains all condition-level rates and seed-specific contrasts. The saved tables retain [per-design outcomes](../../reproducibility/update_method_comparison/supplied/completed/tables/per_root_results.csv), [per-scene outcomes](../../reproducibility/update_method_comparison/supplied/completed/tables/per_scene_results.csv) and [timing and fallback accounting](../../reproducibility/update_method_comparison/supplied/completed/tables/timing_memory.csv). Durations are historical measurements, not new hardware benchmarks.

The three seed diagnostics retain [behavior](../../reproducibility/update_method_comparison/supplied/completed/tables/seed_diag_behavior.csv), [geometry](../../reproducibility/update_method_comparison/supplied/completed/tables/seed_diag_geometry.csv) and [checkpoint-selection metadata](../../reproducibility/update_method_comparison/supplied/completed/tables/seed_diag_selection.csv). Their metadata are study-specific. Do not substitute them for the prospective matched-history checkpoint selection.

## Reconstruct

```sh
python -m repro.update_methods --output reproduced/update-methods
```

The CPU command verifies 208,896 score rows and 6,144 telemetry rows. It checks exact frozen questions, terminal/source labels, score-derived predictions, ties, candidate mass, correctness and full design coverage. It then reconstructs all four contrasts and compares every cell of the headline, per-design, per-scene and timing tables.

The saved full-vocabulary argmax flag remains a recorded measurement. Two candidate probabilities cannot independently certify the highest-scoring token outside the candidate set. The delivery also lacks an attested executed code revision and complete execution environment. Saved-score reconstruction is verified here; fresh neural reproduction is not.
