# Relation construction, splits and statistics

Repository topic A01. See manuscript Section 2.1, Appendices A.1–A.2 and Table 2.

## Source rows and checkerboards

The source is `allenai/ValuePrism`, configuration `valence`, revision `d439ca90825e5b4e5ef97798d9b5950e16ba7065`. The loader lowercases and trims the valence field, then retains only `supports` and `opposes`. It combines the upstream splits before applying the fixed study manifests. Stable row identities and duplicate filtering are implemented in [manifests.py](../../src/relation_study/m1/manifests.py).

A checkerboard contains `(s1,c1,supports)`, `(s1,c2,opposes)`, `(s2,c1,opposes)` and `(s2,c2,supports)`. Each situation and consideration appears once under each label. The source builder rejects unresolved cells and overlap with the reserved confirmatory board IDs.

The leakage audit uses exact strings, normalized strings, prefix-stripped strings and bounded character-ngram clusters. Normalization removes punctuation, excess whitespace, a leading article and a simple terminal plural. The next level strips the Value/Right/Duty prefix. Clustering uses character 3–5-grams, cosine similarity at least 0.85 and a maximum cluster size of 25. It compares members directly with a frequent-form leader, followed by one bounded leader-merge pass. This is a lexical near-duplicate control, not proof that all semantic synonyms are grouped.

The [retained audit aggregates](../../artifacts/leakage/results.json) include filtering counts, split overlap checks and five-seed leakage stress tests. The strict split contains 116,000 training rows and 7,394 test rows after 421 duplicate removals. Those are the broader split-audit counts, not the smaller activation-probe training sample below.

## Development-only split construction

| Split | Items | Checkerboards | Role |
|---|---:|---:|---|
| Training | 1,500 | Not a board endpoint | Fit activation readouts |
| Selection | 300 | 75 | Select layer, regularization and score scaling |
| Evaluation | 500 | 125 | Report held-out development results |

The construction seed is `20260803`. A hash assigns each authoritative consideration cluster to one of two selection/evaluation lanes. Boards are hash-ordered. The builder selects evaluation first, then selection, excluding shared situations and clusters. It rejects reused situations and consideration pairs within a selected panel.

Training is drawn from the original common-training manifest after removing reserved situations, clusters and confirmatory rows. Hash ordering chooses 750 supports and 750 opposes items. All three splits are exactly label-balanced. Checks enforce row, situation and consideration-cluster disjointness, unique label-independent item IDs, no confirmatory rows or boards, and distinct selection/evaluation boards.

The supplied [Gemma membership files](../../artifacts/probe_weights/m1_relation/provenance/probe_membership.json)
record its realized activation-probe assignments. They retain identifiers and
groupings, not source text or reference labels. They are distinct from both the
original construction inputs and the MiniLM fitted-row population.

## Prompts, extraction and readouts

The [Llama configuration](../../configs/m1_development.yaml) contains the exact prompts. Primary answers use A/B; the held-out wording uses 1/2. Each scheme includes standard and reversed label mappings. Assignment is deterministic from item identity, without consulting the target label. Native-answer scoring sums candidate-token log probabilities, including multi-token answers when needed.

The extraction code reads the final prompt token before an answer is generated. It drops the embedding entry from `hidden_states` and indexes post-block outputs from zero. Selected layers are 19 for Llama-3.1-8B-Instruct and 27 for Gemma-2-9B-it. BF16 backbones are unquantized; cached activations use float16. The exact model revisions are retained in [figure data](../../artifacts/figures/figure_data.json) and [model specifications](../../src/relation_study/m1/model_specs.py).

The difference-in-means fit averages each class in float64, casts the means to float32, subtracts them and normalizes to unit length. Its score is the float32 projection minus the projected class midpoint. The logistic fit uses `solver="liblinear"`, `dual=True`, L2 regularization, a fixed random seed and at most 3,000 iterations. The selection grid is `C = 0.01, 0.1, 1, 10`.

Layer selection maximizes mean mirrored pairwise accuracy: the mean of within-situation and within-consideration macro scores. Exact ties choose the lower layer. Logistic regularization is selected at that layer using the same criterion; ties choose the smaller C. Evaluation and held-out answer wording do not determine either selection.

## Interaction, controls and uncertainty

For each scorer, the selection split fixes its mean and sample standard deviation (`ddof=1`). Degenerate or nonfinite scale is an error. Standardized scores define

`I_b = f(s1,c1) - f(s1,c2) + f(s2,c2) - f(s2,c1)`.

This contrast cancels any additive situation-plus-consideration function. Higher scores mean supports. Situation-only and consideration-only activation controls average scores within the relevant identity. The additive control sums rounded component scores. Pairwise ties receive half credit. Groups lacking one label are excluded from the pairwise metric; one-class AUROC is missing rather than invented.

The primary mean interaction and paired advantage over the text comparator use normal 95% intervals with the [dyadic-robust standard error](../../src/relation_study/m1/support/dependence_power.py). Boards may covary whenever they share either consideration endpoint. The estimator subtracts the double-counted contribution of boards sharing both endpoints. It is invariant to swapping endpoint columns. Descriptive grouped/bootstrap metrics use the configured 2,000 replicates and remain distinct from this primary interval.

Missing or incomplete board inputs fail construction checks. The development interpretation requires at least 75 usable evaluation boards. Exact additive and isolated-input invariants are tested. The [probe containers](../../artifacts/probe_weights/README.md) preserve all retained numerical arrays, including the original directions used in the later consequence experiment.

## Human-audit boundary

The supplied procedure calls for two independent reviewers who have not seen item-level model or AI-preflight results. First-stage review covers the 125 development boards. Reviewers assess each of four source cells as clear, ambiguous or invalid. They also assess whether each consideration keeps the same ordinary meaning across situations. They judge the plausibility of the stored relation, not whether the action is morally right.

Primary source clarity requires two independent clear judgments. Uncertainty excludes a cell from independent consensus. A second-stage holder/object stability review applies only after first-stage submissions are locked. Adjudication is a separate secondary population and preserves the original disagreement status. No completed protected judgments or audited confirmatory outcome are provided here. The numeric readout results retain their development-only status.

## Reproduction limits

The release provides aggregate replay, saved null arrays, scientific implementation,
the retained Llama configuration and fitted probe parameters. Gemma's supplied
activation memberships and text-fit metadata add provenance, not a fresh refit.
The [reproduction matrix](../../REPRODUCIBILITY.md#reproduction-relation) records
the remaining execution prerequisites and data restrictions.
