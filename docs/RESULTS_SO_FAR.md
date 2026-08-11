# Results so far

This record covers completed development work through August 10, 2026. It separates measured results from review designs, unopened evidence, and planned analyses.

The project has two fundamental claims. Claim 1 asks whether the model's internal state encodes the relation between the situation and consideration. Claim 2 asks whether that same state predicts which judgments change under rephrasing that holds meaning fixed. Verdict, intervention, and mechanism analyses are downstream follow-ups rather than additional fundamental claims.

The current result is narrow. Llama 3.1 8B Instruct and Gemma 2 9B it contain a linearly readable signal for whether a named ValuePrism consideration supports or opposes an action in context. No completed experiment shows that the original signal predicts later rephrasing instability, changes a model judgment when intervened upon, or propagates to a complete action verdict.

## Evidence status

| Evidence item | Status |
| --- | --- |
| Reciprocal checkerboard supply | Established |
| Strict identity split and leakage stress test | Established |
| Llama factual positive control | Passed |
| Gemma factual positive control | Passed |
| Llama relation signal | Supported on development data |
| Gemma relation signal | Supported on development data |
| Human-confirmed semantic populations | Pending |
| Dedicated explicit task-competence report | Pending |
| Rephrasing prediction beyond text and confidence | Not run |
| Complete-verdict propagation | Not run |
| Real-checkpoint activation access | Compatible |
| Causal relation effect | Not evaluated |
| Sealed confirmation | Not opened |

## ValuePrism supply

The ValuePrism source contains 218,406 rows. Removing the third `Either` class leaves 183,023 binary `Supports` or `Opposes` rows. Among them, 3,437 exact consideration strings receive both labels in different situations.

Those repeated strings support 13,923 possible reciprocal checkerboards across 6,073 consideration pairs. A deterministic ranking retained 1,090 boards, 492 automatic consideration clusters, and 2,180 unique situations. Every situation in the ranked pool is unique.

The 3,437 strings are repeated consideration names. They are not 3,437 fully specified moral reasons. Holder identity, normative object, threat, and action-specific application often remain implicit in the short label.

## Identity separation and leakage

The split pipeline represents consideration identity at four levels.

| Level | Rule | Forms or clusters |
| --- | --- | --- |
| L0 | Raw consideration text | 17,678 |
| L1 | Case, punctuation, spacing, articles, and simple plurals | 15,707 |
| L2 | L1 plus standard Value, Right, and Duty prefix removal | 14,601 |
| L3 | Bounded character-fragment leader clustering | 10,191 |

L3 is an algorithmic identity approximation. Human semantic judgment enters later.

After removal of 421 exact duplicate rows, the strict split assigns 116,000 rows to training and 7,394 rows to testing, while another 59,208 rows cross only one required identity boundary and remain outside both sets. Training and test share zero row identities, exact consideration strings, L2 identities, L3 clusters, or situations. Two disclosed L1 collisions remain, and the U1 sensitivity set removes both.

Across five strict-style draws, exposing held out consideration identities in 30 percent of a fixed training sample raised text-only paired accuracy by 7.29 percentage points on average. The 95 percent Student t interval ran from 4.90 to 9.69 points. Restoring held out situations changed the same baseline by 0.53 points, with an interval from negative 0.81 to 1.86 points.

The automatic filter eventually destroys the comparison graph. U0 retains 7,394 rows and 2,009 within-situation comparisons. U1 retains 7,081 rows and 1,865 comparisons. U2 falls to 587 rows and 23 comparisons. U3 currently equals U2 because no human adjudications have entered it.

The leakage experiment shows that familiar consideration wording supplies a measurable shortcut for the text baseline. It does not estimate residual leakage in the activation probe.

## Checkerboard invariant

Each board contains two situations and two considerations whose four labeled pairings permit exact cancellation of fixed situation scores, fixed consideration scores, and their additive combination.

| Situation | Consideration 1 | Consideration 2 |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

For a score `f`, the interaction is

    I_b = [f(s1,c1) - f(s1,c2)] + [f(s2,c2) - f(s2,c1)]

Any additive score `f(s,c) = a(s) + b(c)` gives `I_b = 0`. Fixed situation and consideration terms cancel exactly. A positive interaction requires pairing-dependent information, though a nonlinear wording pattern can still supply it, so held out consideration identities, the frozen SBERT comparison, answer-mapping controls, factual positive controls, and cross-model replication address different parts of that alternative explanation.

## Factual positive controls

The first factual control failed. It used `True` and `False` as answer symbols while reversing their meanings in the instructions, and Llama often followed the familiar words instead of the temporary mapping. The project retains that run as `TRUTH_CONTROL_V1_STRICT_FAIL`.

The replacement used neutral `A` and `B` symbols, reversed mappings, grouped proposition splits, and a held out `1` and `2` condition across 748 proposition groups. Within each answer scheme, 1,796 records fitted the directions, 600 selected the layer, and 596 remained untouched for final evaluation.

| Model | Selected layer | Separation T | 95 percent interval | Null result |
| --- | --- | --- | --- | --- |
| Llama | 14 | 1.9245 | 1.8930 to 1.9556 | 0 of 1,000, add-one value 1/1001 |
| Gemma | 25 | 1.9444 | 1.9207 to 1.9666 | add-one value 1/1001 |

Llama reached 1.8402 on the held out `1` and `2` condition, with an interval from 1.8070 to 1.8743. That condition changes the answer vocabulary and part of the surrounding prompt. It is broader than a token-only mapping test.

The controls establish that the extraction and direction-fitting pipeline can recover a known semantic distinction across reversed mappings and a modest prompt change in both model families. They do not establish moral geometry.

## Llama development result

| Field | Value |
| --- | --- |
| Checkpoint | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Revision | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Training rows | 1,500 |
| Selection rows | 300 across 75 boards |
| Evaluation rows | 500 across 125 boards |
| Selected activation layer | 19 |
| Terminal disposition | `M1_VERTICAL_SLICE_SIGNAL_SUPPORTED` |

Layer 19 won the selection sweep. The final evaluation then reused the fitted directions, midpoints, and scales.

| Readout | Within-situation accuracy | Within-consideration accuracy | Relation AUROC | Checkerboard interaction |
| --- | --- | --- | --- | --- |
| Native answer margin | 0.8220 | 0.8094 | 0.721 | 1.6089 |
| Difference in means | 0.7800 | 0.7962 | 0.732 | 1.6470 |
| Logistic activation probe | 0.8560 | 0.8332 | 0.780 | 2.0836 |
| Frozen SBERT comparison | 0.5560 | 0.5885 | n/a | 0.2842 |

The native answer margin is a behavioral output and has no selected hidden-state layer. Difference in means remains the primary geometric object because it directly connects the mean `Opposes` activation with the mean `Supports` activation. Logistic regression remains the stronger secondary decoder.

The difference-in-means advantage over SBERT was 1.3628, with a 95 percent interval from 1.0885 to 1.6370. The logistic advantage was 1.7994, with an interval from 1.4970 to 2.1017. The held out `1` and `2` difference-in-means interaction reached 2.1569. Situation-only, consideration-only, and separately encoded additive controls each produced interaction zero.

### Tie correction

The original board finalization counted exact ties as wrong in both directions for one secondary signed exact-board statistic named `B_b`, while leaving the primary interaction and fitted directions unchanged.

| Readout | Original signed value | Corrected signed value |
| --- | --- | --- |
| Situation-only control | negative 1.000 | 0.000 |
| Consideration-only control | negative 0.008 | 0.000 |
| Separate additive control | negative 0.008 | 0.000 |

The corrected primary values remained 0.560 for signed DIM, 1.6470 for DIM interaction, 0.712 for signed logistic, 2.0836 for logistic interaction, 0.640 for signed native margin, and 1.6089 for native interaction after reanalysis. The correction restored the exact additive invariant and changed the terminal disposition from `M1_PIPELINE_NOT_VALIDATED` to `M1_VERTICAL_SLICE_SIGNAL_SUPPORTED`.

A cache-only finalization performed the repair. It loaded no Llama checkpoint, fitted no new direction, and preserved the selected layer and primary output values. The corrected bundle is authoritative, while the original artifact remains retained.

| Artifact | SHA-256 |
| --- | --- |
| Corrected bundle | `d1e57f79cd93e7ae7a4db44076dcbe6785c685c31e908e717012c0f509f9ac53` |
| Corrected result JSON | `85c281974299682cd544d341d73174997ddc696b73a0912bfb274162404f34e0` |
| Original bundle | `9b3561887259c6346ce8d046bf79bd601ec33c2c995edbf2641c6305a16950cf` |

## Gemma development result

| Field | Value |
| --- | --- |
| Checkpoint | `google/gemma-2-9b-it` |
| Revision | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Selected activation layer | 27 |

Gemma used its own prompt implementation, activation space, direction, midpoint, logistic readout, and scale. It did not inherit a fitted activation object from Llama.

| Readout | Within-situation accuracy | Within-consideration accuracy | Checkerboard interaction |
| --- | --- | --- | --- |
| Native answer margin | 0.8920 | 0.8597 | 2.4068 |
| Difference in means | 0.8720 | 0.8776 | 2.3221 |
| Logistic activation probe | 0.8680 | 0.8425 | 2.1499 |
| Frozen SBERT comparison | 0.5560 | 0.5885 | 0.2842 |

The difference-in-means advantage over SBERT reached 2.0378, with a 95 percent interval from 1.6608 to 2.4149. The held out `1` and `2` interaction reached 2.2617. All isolated and additive controls again produced interaction zero.

Gemma reduces the chance that the Llama result depends on one model family, prompt implementation, or fitted direction, while its larger native answer interaction makes native confidence an essential baseline for the rephrasing study.

| Artifact | SHA-256 |
| --- | --- |
| Development bundle | `bb155a092e3c33b04d6031975c2bbf43f2f57513188451022193d2ab8a9db3d8` |
| Result JSON | `952ec81d00ee4b3f70e96b8988d2d52d101cbf7d1663994d980c24c9635de055` |

## Explicit task competence remains open

The native margins and relation AUROC provide behavioral evidence, but they do not replace a dedicated report covering explicit ValuePrism task accuracy, answer mapping, calibration, format compliance, and repeat stability for both models. The report is pending.

Once human source labels exist, the report should stratify performance by consensus-clear, ambiguous, invalid, and disputed relations. Representation claims will then rest beside a direct account of whether each model can answer the task reliably.

## Claim 1 construct and review status

The development result concerns the contextual `Supports` versus `Opposes` direction of a repeated named consideration. It does not establish a universal moral axis or one invariant fully instantiated reason across situations.

The project owner approved the revised construct on August 10, 2026, before any human review label or sealed confirmatory result was observed. The earlier form is superseded and remains preserved as protocol history. The reduced v2 design now has versioned instruments, blank sheets, package hashes, a stable identifier join, implementation tests, and a prospective freeze receipt.

The first review stage asks for four cell judgments and two repeated-consideration judgments per board, with each cell receiving `CLEAR`, `AMBIGUOUS`, or `INVALID` and each repeated consideration receiving `STABLE`, `SHIFTED`, or `UNCERTAIN` for ordinary meaning. Failure flags and short explanations appear only when a judgment is not clear or stable.

After first-stage submissions lock, holder and normative-object stability are reviewed only for a consideration pair whose two cells are unanimously clear and whose ordinary sense is unanimously stable. This per-consideration trigger supports a strict Claim 2 item population without requiring the unrelated second consideration to pass the same source audit. No labels exist.

The 125-board development audit is post hoc with respect to the known aggregate activation result. Its semantic strata provide construct characterization and robustness analysis. A semantic population for sealed confirmation can remain prospective if the rubric, sample size, testing hierarchy, and candidate order freeze before model outcomes open.

No human study labels exist yet. Reviewers who saw item-level diagnostic preflight classifications or corresponding model outcomes cannot serve as independent reviewers for the same primary items unless that exposure is recorded and the affected judgments stay outside the independent-consensus population.

## Confirmatory planning

The power package operates on the real 1,090-board reuse graph. It supports the continuous DIM interaction, DIM-minus-SBERT advantage, logistic secondary analysis, crossed consideration effects, node-residual resampling, dyadic robust simulation, and beta-binomial review-yield calculations.

The planner currently returns `BOARD_POWER_INPUTS_INCOMPLETE`. It still needs the semantic acceptance rate, authoritative development strata, the smallest effect worth detecting, alpha, desired power, and a reserve for unusable boards.

A separate 100-board probability sample will estimate review yield and workload. It shares zero boards with the development review. The former target of 800 accepted boards remains provisional until the project freezes the missing statistical inputs.

## Claim 2 candidate package

The Claim 2 pilot uses the same open 125-board Llama development population and contains 80 representative base items plus 32 disjoint geometry-confidence disagreement items, with one selected cell per board and four ordinary rewrite candidates per item.

| Candidate type | Count | Intended use |
| --- | --- | --- |
| Ordinary rewrite | 448 | Preservation review and susceptibility measurement |
| Identity control | 11 | Coarse detection of obvious rejection errors |
| Trivial restatement | 11 | Coarse detection of overstrict review |
| Single-fact change | 11 | Coarse detection of missed semantic changes |

Each reviewer receives 481 rows in an independent order. Eleven controls per type can expose glaring failures, but one error changes a type-specific rate by about 9 percentage points. Reports must show exact counts, uncertainty intervals, individual failures, and reviewer-specific patterns.

Claude Haiku 4.5 revision `claude-haiku-4-5-20251001` generated 459 responses across 23 initial batches. Two outputs repeated their source wording and received one retry each. The merged import contains every job and reports zero final parsing failures.

The initial generation cost was $1.280928, the two retries cost $0.033067, and the resulting formal total was $1.313995 under the Claude Code 2.1.221 client. The provider exposed no deterministic seed control, and the client did not report the realized temperature.

The candidate set remains unchanged. The original v1 review package is preserved and marked superseded before review because it required many component ratings without a matching downstream analysis and allowed unresolved critical uncertainty to pass under one acceptance path before any reviewer response existed. Reviewers will not complete v1.

The frozen v2 question asks whether the rewrite preserves all facts and relationships that could reasonably affect whether the unchanged consideration supports or opposes the action. Reviewers choose `PRESERVED`, `CHANGED`, or `UNCERTAIN`. A changed or uncertain judgment requires diagnostic flags and a short explanation.

Primary rewrite acceptance requires two independent `PRESERVED` judgments. Any uncertainty excludes the rewrite from that population. Adjudicated cases remain separate and retain their original disagreement status.

The existing provider, Claim 2, M1, and AISteer regression suite reported 51 passing tests and 60 third-party scikit-learn deprecation warnings before the v2 review redesign. The focused v2 and existing Claim 2 regression suite now passes 29 tests with the same 60 third-party warnings. The broader local suite requires PyTorch, which is absent from the current CPU-only verification environment.

## Claim 2 statistical target

The 80 representative base items form the primary sampling stratum. The 32 disagreement items form an enriched diagnostic stratum and cannot estimate natural flip prevalence.

All four rewrites of one base item share its original hidden state, prompt text, native confidence, geometry score, and sampling history. They are repeated behavioral trials. They are not independent geometry observations.

For base item `i`, the analysis records the number of accepted rewrites `n_i`, the number that flip the model's reason judgment `k_i`, and susceptibility `k_i/n_i`. Inclusion must follow a frozen minimum accepted-rewrite rule. Grouped held out prediction keeps every rewrite from one base item together.

The primary semantic population requires an ordinary candidate, unanimous source-cell clarity, and unanimous rewrite preservation. The strict item population additionally requires clarity in the paired cell containing the same consideration, ordinary-sense stability, holder stability, and normative-object stability for that sampled consideration. A full strict-board population supplies an additional conservative sensitivity analysis.

The stored ValuePrism relation, human source clarity, original model judgment, and rephrased model judgment remain separate fields. A flip compares the model with itself. Agreement with ValuePrism does not define stability.

## Predictive specificity controls

The primary prediction comparison adds relation geometry to original prompt text and native confidence. Generic hidden-state measurements test whether any improvement belongs to the relation direction rather than general item difficulty.

| Control family | Purpose |
| --- | --- |
| Activation norm | Tests general magnitude |
| Hidden-state principal components | Tests broad activation variation |
| Random directions | Tests arbitrary projections |
| Orthogonal random directions | Tests directions separated from DIM |
| Label-permuted DIM directions | Tests dependence on relation labels |
| Factual direction when technically compatible | Tests an unrelated semantic direction in the same space |

The plan freezes 20 random directions, 20 orthogonal directions, and 20 label-permuted DIM directions. Truth or polarity controls enter only when an already frozen direction occupies the same activation space under a compatible extraction contract.

The two control families differ. Prediction controls test feature specificity. Intervention controls test causal specificity.

## Real-checkpoint intervention compatibility

AISteer360 accepts a supplied activation direction at one selected layer and prompt position. Gate 1 loaded the exact pretrained checkpoints in BF16 on one CUDA device, used the original prompts, evaluated the original teacher-forced candidate scorer, and changed the decoder block output after attention and MLP residual updates. Under the Hugging Face convention, block `L` output is the state exposed as `hidden_states[L+1]`, which matches the activation used by the relation probe at the final prompt token.

| Model | Layer | Runtime | Peak allocated VRAM | Result |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B Instruct | 19 | 189.14 seconds | 16,392,586,752 bytes | Compatible |
| Gemma 2 9B it | 27 | 210.34 seconds | 19,003,006,464 bytes | Compatible |

Compatibility passed. Both models cleared nine engineering checks. The baseline intervention produced zero target delta. The nonzero intervention matched the realized BF16 vector exactly. Untargeted positions stayed unchanged at the modified block output, candidate log probabilities remained finite, and every model-parameter fingerprint matched after execution.

Gate 1 establishes precise activation access. Gates 2 through 4 must still test signed dose response, answer mapping reversal, norm-matched random vectors, orthogonal vectors, the wrong layer, the wrong token, the logistic direction, and damage at moderate intervention strength.

## Preserved provenance

| Object | Commit or SHA-256 |
| --- | --- |
| Llama implementation | `c7e5b5a134ad093457daa7f49df56b30a514c848` |
| Gemma implementation | `87c058990b6f1da04ee8157581e55f2a8398e406` |
| AISteer feature | `18436209873133f86488d733a64645b8eed5aaa8` |
| Preservation implementation | `212d6f0aeb3afaa3a09d5e57b0bcaf28e146db62` |
| Frozen vectors | `0024b9ed004e36a5922cccf25e6dce3a9c90fc177886d92f6a826845758f27d2` |
| Preservation archive | `b98f7e350cf736f3bdd695d82f3514f7bf5862be8e755a67341b50297a0b0d28` |
| Claim 2 implementation | `f2197c56cec6e03f1ecb3aa8111da63997c8ba20` |

The Claim 2 generation record also binds the 459 jobs, 23 batches, two retries, exact model revision, client version, costs, package hashes, and reviewer-sheet hashes. The public status notebook remains an August 5 snapshot and should not be used as the current project ledger.

## Remaining evidence

The reduced v2 rubrics, blank sheets, stable identifier join, reviewer eligibility rules, prospective assertions, and v1 preservation hashes are frozen. The next gate is independent review. Calibration must finish before each reviewer begins study rows.

The human reviews can then run in parallel. Claim 1 measures source relation clarity. Claim 2 measures rewrite preservation. Holder and object review follows only for eligible consideration pairs after first-stage labels lock.

Accepted representative rewrites will measure rephrasing susceptibility and identical-prompt repeat noise. Grouped prediction will then compare text plus native confidence with text plus native confidence plus relation geometry. The 32 disagreement items will test whether geometry and confidence fail on different cases.

The intervention sweep, candidate-yield audit, power decision, sealed Llama confirmation, Gemma replication, complete-verdict arm, and deferred mechanism analyses follow their gates in [the project plan](PROJECT_PLAN.md).

If geometry predicts reason-level susceptibility but complete verdicts remain stable, the evidence will locate compensation after individual reason evaluation. That result would motivate direct measurement of how the model combines competing reasons, rather than another expansion of the representation probe.
