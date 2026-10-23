# Paired-editor training and checkpoint selection

Repository topic A05. See manuscript Section 4.1 and Appendices C.1–C.7,
especially C.4 for the learned procedures.

## What the learned update changes

The updater receives a source prefix and explicit commands assigning binary values to addressed record clauses. It receives no downstream question, evaluator label or terminal-world dictionary. Each command acts at a fixed post-block site in a detached FP32 workspace. The result is cast back to the BF16 backbone representation.

The [original setter](../../reproducibility/learned_editors/canonical/setters.py) has a rank-16 mutable subspace and two value-conditioned linear heads. For centered clause states `h`, basis `U` and complement `r = h - h U Uᵀ`, it constructs per-token features and their clause mean. A head predicts 16 coordinates for the requested bit. The output is the preserved complement plus the predicted coordinates projected through `Uᵀ`, with the center restored.

| Setter | Features read by its head | Preserved part |
|---|---|---|
| Constrained | Complement `r` and its clause mean | Orthogonal complement |
| Unrestricted | Full centered state `h` and its clause mean | Orthogonal complement |

Each head has shape `[2 × hidden width, 16]`. The two heads, two biases, center, raw basis and realized basis are retained. The loader uses the saved realized basis exactly. QR with sign normalization is a diagnostic on the raw basis, not a replacement for the saved basis. Setter strength is exactly one, with no additional clipping or multiplier.

| Backbone | Width | Editor site, zero-based post-block | Revision |
|---|---:|---:|---|
| `google/gemma-2-9b-it` | 3,584 | 15 | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| `Qwen/Qwen3-8B` | 4,096 | 13 | `b968826d9c46dd6066d109eabc6255188de91218` |

These editor sites differ from the relation-readout layers in Section 2. Qwen disables thinking in its chat template. Backbone weights stay frozen. [Shared model settings](../../reproducibility/update_method_comparison/inputs/models.json) retain precision and scoring configuration.

## Objectives, seeds and actual selection

The supplied training description distinguishes two paired procedures:

| Procedure identifier | Reported objective | Learning rate |
|---|---|---:|
| `INV_PAIR_NLL` | Paired full-vocabulary label negative log likelihood; consistency weight zero | 0.0003 |
| `FREE_PAIR_CONSISTENCY` | Paired label loss plus symmetric KL on the conditional two-label distribution; consistency weight one | 0.001 |

The [paired-stage training record](../../reproducibility/learned_editors/training_reference/README.md)
includes the actual training, loss and calibration source, plus the original 64 FIT
and 32 CAL pairs. Each retained condition completed one epoch with 384 groups,
3,072 paired question slots and 384 optimizer steps.

The original function uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8` and zero
weight decay. It invokes no learning-rate scheduler. Norm regularization has weight
0.01, and gradient clipping uses norm one. Warm starts initialize fresh optimizer
moments. The two-origin gradient uses three forwards and two backwards before one
optimizer step. Original 24-question bundle weights are preserved within each
four-question group.

The source snapshots are inspection material, not an installed retraining pipeline.
The [reproduction matrix](../../REPRODUCIBILITY.md#reproduction-levels) distinguishes
the supplied recipe from the remaining fresh-execution integration work.

Selection is based on the calibration split, with seeds zero and one retained for each procedure and backbone. [SELECTED.json](../../reproducibility/learned_editors/weights/SELECTED.json) preserves every selected epoch, feasibility flag and calibration row. Calibration performance includes complete-battery correctness, changed-answer accuracy, invariant harm and no-op preservation.

Gemma's unrestricted procedure at seed zero selects paired-stage epoch zero. Its actual metadata retain the earlier `FREE_COMPLETE_SINGLE` checkpoint at epoch two, with 1,440 prior steps and no recorded consistency-weight field. The other seven learned selections use paired-stage epoch one. This distinction identifies the weights used; it does not say that paired training was never attempted. Missing consistency weight is not interpreted as zero.

Canonical replay in the separate method comparison uses `INV_PAIR_NLL` at seed zero and `INV_PAIR_CONSISTENCY` at seed one for each backbone. The two extra seed-one files are replay dependencies, not extra main comparison procedures. The release includes ten selected parameter bundles and 15 required initialization bundles. [PARAMETERS.json](../../reproducibility/learned_editors/PARAMETERS.json) records all 200 arrays and their scientific roles.

## Commands, prompts and read ordering

The [typed source/update interface](../../reproducibility/learned_editors/inputs.py) takes half-open Unicode character spans covering complete source clauses. Tokenization maps them to contiguous overlapping token positions. A writer receives only prefix token IDs, addressed positions and requested bits. Source text and all fixed demonstrations are retained in [prompts.json](../../reproducibility/learned_editors/prompts.json).

Learned updates preserve command order, including repetitions, no-ops and restorations. Existing-correction text retains every correction sentence. Latest-correction text retains the last command per exact address in source-record order. Canonical replay also normalizes the command history, then recompiles the original source using its selected setter. Neither normalization consults evaluator truth.

The implementation recompiles the prefix with hooks. Each question uses a separate clone of the resulting cache. Tests reject master-cache mutation, prefix-boundary drift, partial overlapping addresses and evaluator fields passed to the updater. This is a typed application boundary, not a Python security sandbox. Recompilation and retained source/history storage belong in cost accounting.

## Primary matched-history design and question battery

The prospective study uses 64 terminal designs per backbone. Three distinct addressed records produce all eight binary starting assignments. The same three commands make every history converge to the same complete current table. Origin zero already equals that current table and provides the same-method no-op reference.

Each context has 34 questions: direct reads, opposition reads and the joint functions SAME, BOTH and EITHER. The two answer formats are literal No/Yes and a deterministically assigned A/B mapping. There are 17 questions per format, including nine joint targets. Exact rendered wording, public labels, semantic specifications, source prefixes and commands are together in each [design file](../../reproducibility/state_sufficiency/independent_verification/inputs/designs/SSC1-FINAL-0000.json). The independent checker verifies all 64 designs, not just this example.

The joint templates ask whether two actors take the same side, both support a proposal, or at least one supports it. Direct and opposition templates ask whether a named actor favors or opposes the proposal. All histories receive the same question text and answer mapping. The [protocol](../../reproducibility/state_sufficiency/independent_verification/inputs/protocol.json) fixes context counts, references and repeat selection.

## Scoring, eligibility and denominator

Candidate scores are full-vocabulary FP32 log probabilities, with Gemma's final logit softcap of 30 retained. Ties and nonfinite results cannot create a witness. In the primary saved records, the larger candidate probability exceeds all omitted vocabulary mass by more than `1e-6`. This independently certifies that the full-vocabulary argmax lies in the answer set. Token objects, question digests, mappings and recorded flags are checked against those probabilities.

The portable historical input helper separately exposes forced-choice validity without requiring vocabulary-argmax membership. Its synthetic test preserves that documented behavior. The primary release checker uses the stronger saved-score certificate above; the helper's interface is not evidence that every historical neural path has been revalidated.

Eligibility requires both direct operands to be valid and correct in all eight histories. Both the native-current and same-method no-op references must answer the joint target correctly. A witness then requires at least two distinct valid joint answers across histories. Invalid responses receive no witness credit.

Each design retains all 18 scheduled joint opportunities, including ineligible ones. Learned-seed rates are averaged within design. A design is affected if either learned seed has a witness, so affected-design counts and seed-mean rates are distinct. The case bootstrap resamples 64 designs, preserving within-design dependence. It uses 10,000 draws, seed `2609141002`, and 99.375% intervals for the eight model/procedure comparisons.

The additional high-margin criterion retains the primary witness and requires all joint answers valid, all direct correct probabilities at least 0.95, candidate mass at least 0.90 for the relevant operands/joint/reference reads, and joint correct probabilities spanning at least 0.95 to at most 0.05. [Stronger logical controls and fresh confirmation](A06_stronger_controls.md) remain separate analyses.

## Verify the retained parameters and implementation

```sh
python -m repro.learned_editors --output reproduced/editors
```

The command verifies the public inventory and runs 18 original CPU checks. They load all ten selected bundles, compare both setter kinds against independent NumPy arithmetic at the original `2e-5` absolute/relative tolerance, verify frozen bases, and exercise source/update and cache-copy boundaries with synthetic inputs. The model backbones and real tokenizers are never loaded.

Saved-score replay is available independently of this parameter check. The supplied
source does not establish fresh backbone execution or retraining. Original optimizer
snapshots are withheld; they are needed for exact continuation, not for a fresh
warm start. The compact helper deliberately rejects the external factual-refresh
baseline whose complete adapter is absent.
