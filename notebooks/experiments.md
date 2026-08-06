# Notebook experiment log

## ValuePrism shortcut restoration

| Field | Frozen value |
| --- | --- |
| Hypothesis | Consideration identity exposure raises text-only within-situation paired accuracy more than situation exposure |
| Baseline | Strict-style cluster and situation holdout |
| Changed variable | Replace 30 percent of the capped training sample with rows exposing one held-out identity type |
| Held fixed | Training size, test rows, features, classifier, metric |
| Seeds | 0, 1, 2, 3, 4 |
| Result | Consideration exposure plus 7.29 percentage points, situation exposure plus 0.53 points |
| Uncertainty | Five-draw Student-t interval from 4.90 to 9.69 points for consideration exposure |
| Conclusion | Consideration identity is a material shortcut for this text baseline |

## Factual positive control

| Field | Frozen value |
| --- | --- |
| Hypothesis | A truth direction fitted on neutral A/B prompts transfers across reversed mappings and held-out 1/2 symbols |
| Baseline | Preserved countersemantic True/False attempt |
| Changed variable | Replace literal truth words with neutral answer symbols |
| Held fixed | Factual source family, grouped split, mean-difference direction, held-out test |
| Model | Meta Llama 3.1 8B Instruct at revision 0e9e39f249a16976918f6564b8830bc894c89659 |
| Result | Layer 14, A/B T=1.92, 1/2 T=1.84, C=0.999, p=1/1001 |
| Conclusion | The activation procedure recovers factual class information across answer mappings |

## Moral-relation development slice

| Field | Frozen value |
| --- | --- |
| Hypothesis | Joint situation-consideration activations carry reciprocal relation information beyond additive and matched-text controls |
| Baselines | Situation-only, consideration-only, separate-encoding additive, SBERT interaction, native answer margin |
| Model | Meta Llama 3.1 8B Instruct at revision 0e9e39f249a16976918f6564b8830bc894c89659 |
| Split | 1,500 train rows, 300 selection rows, 500 evaluation rows |
| Result | Layer 19, DIM I_b=1.65, logistic I_b=2.08, SBERT I_b=0.28 |
| Relation AUROC | Native 0.721, DIM 0.732, logistic 0.780 |
| Conclusion | Development evidence supports a relation signal. Human-audited confirmation and rephrasing-flip prediction remain open |

## Reconstruction contract repair

| Field | Value |
| --- | --- |
| Hypothesis | FULL currently reads a nonexistent single row_id column from the four-cell confirmatory manifest |
| Baseline | Published v1.1.0 comparison path |
| Changed variable | Flatten the four row ID columns, deduplicate row membership, and hash with the manifest builder's sorted rule |
| Held fixed | Frozen manifest, candidate set, row identity function, and expected hash |
| Result | Synthetic regression covers all four columns and repeated row IDs; public suite passes |
| Conclusion | The comparison now operationally verifies the stored confirmatory row-membership contract |

## Analysis environment lock

| Field | Value |
| --- | --- |
| Hypothesis | ANALYSIS can drift when numerical packages load before exact versions are installed |
| Baseline | Ranged package installation in v1.1.0 |
| Changed variable | Install the CPU analysis lock before importing NumPy, pandas, scikit-learn, PyYAML, or plotting code |
| Held fixed | Activation cache contract, seed 314159, estimator, layer selection, and exact result comparison |
| Result | Setup source now selects the analysis or GPU lock before experiment imports |
| Conclusion | ANALYSIS and FULL use mode-specific dependency locks without forcing CUDA packages into the CPU-only path |
