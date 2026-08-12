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
| Hypothesis | A truth direction fitted on neutral A/B prompts transfers across reversed mappings and a held-out 1/2 prompt scheme |
| Baseline | Preserved countersemantic True/False attempt |
| Changed variable | Replace literal truth words with neutral symbols in v2; the held-out condition also changes A/B to 1/2 and uses a second user template |
| Held fixed | Factual source family, grouped split, mean-difference procedure, held-out test groups |
| Model | Meta Llama 3.1 8B Instruct at revision 0e9e39f249a16976918f6564b8830bc894c89659 |
| Result | Layer 14, A/B T=1.92, 1/2 T=1.84, C=0.999, p=1/1001 |
| Conclusion | The activation procedure recovers factual class information across answer mappings and the combined held-out vocabulary and prompt-template shift; answer symbols alone are not isolated |

## Notebook interface correction

| Field | Value |
| --- | --- |
| Hypothesis | Shared setup code can expose unsupported modes and overstate what the held-out Truth transfer isolates |
| Baseline | Published v1.1.1 notebook sources |
| Changed variable | Correct transfer wording, validate notebook-specific modes, and state the two distinct confirmatory-manifest hash contracts |
| Held fixed | Frozen prompts, activations, splits, results, model revision, and reconstruction thresholds |
| Result | Truth declares DEMO, ANALYSIS, and FULL; ValuePrism declares DEMO and FULL; the exact manifest file SHA is distinguished from the row-membership hash |
| Conclusion | The notebook interface now matches the implemented conditions without adding a post-result experimental control |

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

## Public ValuePrism and Llama replay

| Field | Frozen value |
| --- | --- |
| Experiment ID | 2026-08-11-public-reproduction |
| Hypothesis | The public pipeline can reconstruct the frozen M0 manifests, then rerun the Llama development test without opening confirmatory results |
| Baseline | Retained ValuePrism aggregate and the finalized Llama development result with source SHA-256 85c281974299682cd544d341d73174997ddc696b73a0912bfb274162404f34e0 |
| Changed variable | Execution environment and checkout only |
| Held fixed | ValuePrism revision, Llama revision, seed 20260803, split sizes, prompts, layer selection, probes, baselines, and tie-neutral analysis |
| Comparison | Exact agreement for 14 protocol fields and absolute tolerance 0.0005 for 9 reported measurements |
| CPU verification | 41 tests passed and 1 PyTorch-dependent test skipped on the local Python 3.13 environment |
| Local M0 attempt | The cached ValuePrism rebuild spent 900 seconds in deterministic L3 clustering without producing the terminal manifest hashes, so it does not count as a completed reproduction |
| Conclusion | The open development experiment now has a pinned public runner and two generated Colab entry points. A GPU replay remains required before reporting an independent numerical reproduction |

## Colab note reduction

| Field | Value |
| --- | --- |
| Change | Shortened the ValuePrism Markdown from 232 to 126 words and the Llama Markdown from 272 to 122 words |
| Held fixed | Every code cell, frozen parameter, output contract, and evidence boundary |
| Result | The notebooks now use short wording drawn from the README and results document |
