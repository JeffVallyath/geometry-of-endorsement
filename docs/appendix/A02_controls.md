# Permutation and factual controls

Repository topic A02. See manuscript Section 2.1, Appendices A.4–A.5 and Table 3.

## Relation permutation nulls

The implemented relation null flips every training label within a situation together. Each situation independently retains or reverses its binary orientation. Rows remain grouped. Both the difference-in-means and logistic readouts are refitted after each draw. The already selected layer and logistic regularization stay fixed. The selection-score scaler is refitted in each draw, and interaction is evaluated on the unchanged 125 evaluation boards.

The retained development implementation uses 200 draws and `default_rng(seed + 404)`, with base seed `20260803`. The [figure data](../../artifacts/figures/figure_data.json) retain both models' original draws and tail summaries. These are development nulls conditional on the selected layer, not a new confirmatory study.

A separate larger Llama calculation retains 10,000 draws with recorded seed `20260803`. Its [full null record](../../artifacts/figures/permutation_null_llama_B10000.json) stores both probes' values. The one-sided Monte Carlo p-value is `(k + 1)/(B + 1)`, counting draws greater than or equal to the observed interaction.

The [large-null kernels](../../reproducibility/relation_readout/source/large_null.py) use float32 class means, fixed layer 19 and logistic `C=0.1`, `solver="liblinear"`, `dual=True`, `max_iter=2000`. Draw index `i` uses a separate `default_rng(20260803 + i)` and the same seed for the logistic fit. A draw collapsing to one class is omitted; all 10,000 requested draws were valid. The stored two-sided sensitivity counts `abs(null) >= abs(observed)` with the same plus-one correction.

| Llama large null | Draws at least observed | One-sided p |
|---|---:|---:|
| Difference in means | 131/10,000 | 132/10,001 = 0.01319868 |
| Logistic | 0/10,000 | 1/10,001 = 0.00009999 |

The stored two-sided sensitivities are 0.02449755 and 0.00009999. The observed difference-in-means value in this recalculation is 1.6470104456; the original published aggregate is 1.6470105764. Both are retained. This release recounts the stored null tails, not the unavailable activation-level refits. The original 200-draw record remains available alongside the larger calculation.

## Factual positive control

The control uses aligned `cities` and `neg_cities` statements from the immutable upstream sources in [truth_control_v2.yaml](../../configs/truth_control_v2.yaml). It contains 2,992 semantic rows grouped into 748 city/proposition units. The [split manifest](../../artifacts/truth/split_manifest.csv) records both answer schemes without redistributing full source statements.

The grouped split is 60% training, 20% development and 20% test, with seed `314159`. Recorded primary rows are 1,796 train, 600 development and 596 test. The control fits eight training partitions. Layer selection maximizes the smaller signed standardized separation across standard and reversed primary A/B mappings; an exact tie chooses the lower layer. The 1/2 answer wording is held out from selection.

The second-version null flips semantic signs within complete training groups. Each of 1,000 draws repeats fitting and all-layer development selection, then evaluates one untouched primary test. Draw index `i` uses `default_rng(318159 + i)`, from the run seed plus 4,000. It uses the inclusive one-sided `(k + 1)/(B + 1)` tail. No draw reaches the observed Llama statistic, giving `1/1001`. The [full control record](../../artifacts/truth/v2_results.json) retains all null values and the distribution of selected layers.

Positive-control requirements include native semantic AUROC strictly above 0.70 in both mappings and both answer schemes, permutation p below 0.05, positive primary and transfer 95% intervals, positive signed separation in all four mapping/scheme cells, and a clear immediate neighboring layer. Test-group bootstrap intervals use 2,000 resamples. AUROC is a descriptive control, not the permutation endpoint.

Llama selects layer 14, with primary signed separation 1.9244548 and transfer separation 1.8401653. The retained aggregate bundle allows the layer choice, tail count, interval gates, mapping coverage and scientific input identities to be checked. The Gemma figure record reports its own control at layer 25 with signed separation 1.9443935; a complete corresponding control bundle is not supplied for the same replay.

## Earlier factual-control specification

The first factual-control version remains `TRUTH_CONTROL_V1_STRICT_FAIL`. Its reversed instruction-mapped AUROC is about 0.000349 despite strong literal True/False semantics. The [diagnostic record](../../artifacts/truth/v1_diagnostic.json) preserves both its rowwise permutation result, 25/101, and proposition-pair-preserving result, 433/1001. Their fitting/orientation uses training data with the original selected layer fixed. They are different nulls from the second version's complete layer-reselection procedure.

These controls support interpretation of the readout assay. They do not establish moral understanding or a causal role for the relation direction in later reasoning.
