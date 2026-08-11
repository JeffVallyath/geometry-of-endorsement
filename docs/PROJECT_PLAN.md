# Project plan and decisions

This document records the prospective research program, owner-approved decisions, human-review design, analysis populations, and result branches as of August 10, 2026, while completed measurements remain in [Results so far](RESULTS_SO_FAR.md).

The human-review redesign described here has owner approval. Its technical freeze remains pending until the versioned instruments, blank sheets, manifests, hashes, joins, tests, and freeze receipt exist.

## Research claims

| Claim | Fundamental research question | Current state |
| --- | --- | --- |
| Claim 1 | Does the model's internal state encode the relation between the situation and consideration | Supported on development data |
| Claim 2 | Does that same state predict which judgments change under rephrasing that holds meaning fixed | Not run |

Complete-verdict analysis, causal intervention, and mechanism characterization are downstream tests. They locate, probe, or extend results under the two claims and retain separate evidence standards. They do not create additional fundamental research questions.

## Variables that must remain separate

| Variable | Source | Role |
| --- | --- | --- |
| Stored ValuePrism relation | Dataset | Target used to train and evaluate the relation readout |
| Human source clarity | Independent Claim 1 reviewers | Establishes whether the stored relation has a clear contextual interpretation |
| Original model reason judgment | Tested model | Baseline behavior for one named consideration |
| Rephrased model reason judgment | Tested model | Defines a reason-level flip relative to the original answer |
| Complete action verdict | Tested model and later stance coding | Measures the all-things-considered response |

A Claim 2 flip compares the model's rephrased reason judgment with its own original reason judgment. Agreement with the ValuePrism relation does not define a flip.

## Construct and evidence boundary

The naturalistic checkerboard repeats a named consideration, but the short label does not guarantee one fully specified normative proposition with an invariant holder and object across both situations.

The primary Claim 1 construct is the contextual direction of a repeated named consideration, while the exact-reason-like sensitivity requires stable ordinary sense, holder, and normative object for the relevant consideration pair. A role-explicit benchmark remains an optional later branch.

The 125-board development review occurs after the aggregate development result became known. Semantic analyses of those boards are post hoc construct characterization and robustness analyses. The same rules can remain prospective for sealed confirmation when the rubric, testing hierarchy, candidate order, and sample size freeze before confirmatory outcomes open.

## Decision state

| Decision | State on August 10, 2026 |
| --- | --- |
| DIM as primary relation geometry | Owner-approved |
| Logistic probe as secondary decoder | Owner-approved |
| Llama as primary confirmatory model | Owner-approved |
| Gemma as prespecified replication | Owner-approved |
| Contextual direction as primary Claim 1 construct | Owner-approved |
| Independent human consensus for primary semantic eligibility | Owner-approved |
| Reduced v2 human-review design | Owner-approved, technical freeze pending |
| Original v1 packages | Preserved and superseded before review |
| Candidate rewrite text | Frozen and unchanged |
| Smallest effect of scientific interest | Open |
| Alpha, desired power, and reserve | Open |
| Confirmatory board count | Open pending review yield and power inputs |
| Complete-verdict sample size | Open pending reason-level flip prevalence |

## Claim 1 first-stage review

Two reviewers independently inspect each source checkerboard. They receive the situation text, named considerations, and stored relations. They do not receive model answers, activations, confidence, probe scores, candidate strata, intervention output, or the other reviewer's labels, and each package randomizes board order independently.

For each of the four cells, the reviewer answers the following question.

> Using the stated facts and the ordinary meaning of the consideration, is the stored Supports or Opposes relation a clear and plausible reading?

| Response | Definition |
| --- | --- |
| `CLEAR` | The situation is intelligible, the consideration applies, and the stored relation is a natural reading without a substantial competing interpretation |
| `AMBIGUOUS` | The consideration is relevant, but missing detail or competing readings prevent a clear relation judgment |
| `INVALID` | The situation is incoherent, the consideration does not apply, or the stored relation is not a plausible reading |

For each of the two repeated considerations, the reviewer answers the following question.

> Does the consideration retain the same ordinary meaning across the two situations?

| Response | Definition |
| --- | --- |
| `STABLE` | The phrase keeps the same ordinary sense across both situations |
| `SHIFTED` | The phrase changes sense enough to alter the comparison |
| `UNCERTAIN` | The reviewer cannot establish stable ordinary meaning |

An `AMBIGUOUS`, `INVALID`, `SHIFTED`, or `UNCERTAIN` answer opens conditional failure fields. During the first-stage review, allowed flags cover unclear situation text, consideration inapplicability, competing relation readings, implausible stored relation, shifted meaning, uncertain reference, changed applicability, malformed wording, and other. Notes stay conditional. A short explanation becomes required when any failure flag appears.

The form collects no confidence score. The primary categories already encode the uncertainty needed by the analysis.

## Claim 1 second-stage review

Holder and normative-object coding begins only after first-stage submissions lock, so the second-stage package never uses provisional source categories or unlocked reviewer labels during selection or package generation. Selection happens per repeated consideration, not per complete board.

A consideration pair enters the second stage only when both of its source cells receive two independent `CLEAR` judgments and both reviewers mark its ordinary sense `STABLE`. The second-stage package reveals only the source text and the selected consideration pair. It does not reveal model measurements, candidate strata, first-stage disagreement details, or derived analysis populations.

Each reviewer independently records holder stability and normative-object stability as `STABLE`, `SHIFTED`, or `UNCERTAIN`. These labels determine the exact-reason-like sensitivity populations. They are not descriptive fields without an endpoint.

The per-consideration trigger preserves usable Claim 2 items when the unrelated second consideration on the checkerboard fails. A complete Claim 1 strict board still requires both consideration pairs to pass.

## Claim 2 rewrite review

Claim 2 reviewers receive one source and candidate rewrite pair at a time, together with the unchanged named consideration. They answer the following question.

> Does the proposed rewrite preserve all facts and relationships that could reasonably affect whether the unchanged consideration supports or opposes the action?

The wording asks reviewers to compare semantic content directly without predicting the model's response or deciding which moral answer is correct for either version of the text.

| Response | Definition |
| --- | --- |
| `PRESERVED` | Relevant actors, action, target, intent, causal facts, consequences, certainty, scope, moral stakes, specificity, and consideration target remain substantively fixed |
| `CHANGED` | At least one substantive difference could reasonably alter the Supports versus Opposes judgment |
| `UNCERTAIN` | Ambiguity, awkward wording, or incomplete comparison prevents a confident preservation judgment |

A `CHANGED` or `UNCERTAIN` answer opens conditional diagnostic fields. Reviewers select one or more of the following technical categories.

1. Actor, action, target, or intent
2. Facts, certainty, scope, or specificity
3. Consequences, severity, or moral stakes
4. Consideration target or applicability
5. Unclear or malformed wording
6. Other

A short explanation is required for every changed or uncertain judgment. Accepted candidates require two independent `PRESERVED` judgments. Any uncertainty excludes a candidate from the independent-consensus population.

The v2 instrument does not recreate the old component-level acceptance rule, and the old package remains unchanged as protocol history without receiving reviewer responses in the study. A lenient v2 sensitivity will exist only if a defined robustness question requires it.

## Reviewer assignment and eligibility

Separate Claim 1 and Claim 2 reviewer pairs are preferred because they reduce correlated judgment errors and prevent source-clarity impressions from carrying into rewrite comparison across the two tasks during independent review. Reviewer availability can require overlap. When overlap occurs, the project records it, separates sessions, randomizes row order independently, and blocks access to consensus labels or derived categories from the other review.

Anyone who saw item-level AI preflight classifications, model outcomes, geometry scores, native confidence, or intervention results for an item cannot supply one of its two independent primary judgments unless the exposure is documented and the label remains outside the primary population.

Calibration uses separate worked examples that never enter the study manifests. The calibration material explains each category, demonstrates difficult boundaries, and tests package handling before reviewers receive the study sheets.

Identity, trivial-restatement, and single-fact-change controls remain hidden inside the Claim 2 package. Their 11 cases per type provide coarse checks for obvious review failure. Reports show exact counts, interval estimates, individual errors, and reviewer-specific patterns.

## Independent consensus and adjudication

Independent agreement is an observed result. Adjudication is a later decision.

Primary source clarity requires Reviewer 1 `CLEAR` and Reviewer 2 `CLEAR`. Primary rewrite preservation requires Reviewer 1 `PRESERVED` and Reviewer 2 `PRESERVED`. Original labels remain immutable after submission lock.

Adjudication begins only after both independent submissions lock. Adjudicated cases enter a separately named secondary population and retain their original disagreement fields. They never become independent-consensus cases in reports or manifests.

The agreement report covers raw agreement, complete confusion matrices, category-specific agreement, uncertainty rates, control performance, reviewer-specific patterns, and reasons for disagreement before any adjudicated analysis or revised population count appears.

## Stable identifier contract

The Claim 1 to Claim 2 join uses stable identifiers. Text matching is forbidden.

Each of the 112 Claim 2 base items must resolve without text matching to exactly one board identifier, situation position, consideration position, source row identifier, stored ValuePrism relation, and Claim 1 cell judgment. The sampled item must belong to the expected 125-board `pilot_eval` population.

Validation fails when an item has no source cell, maps to more than one source cell, disagrees on either position, carries a mismatched source row, or falls outside the expected board population. The candidate text and source mappings remain unchanged during the v2 review redesign.

## Analysis populations

Human filtering does not erase the sampling design. The 80 representative items and 32 geometry-confidence disagreement items remain separate after every review gate.

| Population | Frozen definition | Purpose |
| --- | --- | --- |
| Representative primary | Representative stratum, ordinary candidate, source cell independently consensus-clear, rewrite independently consensus-preserved | Estimate acceptance, instability, and held out prediction in the sampled development population |
| Disagreement diagnostic | Enriched disagreement stratum with the same semantic gates | Compare geometry with native confidence where their vulnerability rankings differ |
| Strict Claim 2 item | Primary semantic gates plus clear paired cell, stable sense, stable holder, and stable object for the sampled consideration | Test the exact-reason-like interpretation for the item being predicted |
| Strict Claim 1 board | All four cells clear, both considerations stable in sense, holder, and object | Support the strongest checkerboard-level interpretation |
| Full-board Claim 2 sensitivity | Claim 2 accepted rewrite whose complete source board meets the strict Claim 1 rule | Supply an additional conservative sensitivity analysis |
| Adjudicated secondary | Cases accepted only after adjudication, with original disagreement retained | Test whether formal dispute resolution changes the conclusion |
| Broad descriptive | Meaning-preserving rewrites stratified by clear, ambiguous, invalid, and disputed source status | Describe excluded material without carrying the strongest semantic claim |

The disagreement stratum cannot estimate natural flip prevalence because selection deliberately enriched it for geometry-confidence conflict. Reports do not pool it with the representative sample for prevalence or primary prediction.

## Statistical unit and target

The primary geometry observation is one original base item, whose four rewrites share the original hidden state, prompt text, native confidence, relation score, and selection history throughout analysis.

For item `i`, the analysis records accepted rewrite count `n_i`, flip count `k_i`, and susceptibility `k_i/n_i`. A frozen minimum accepted-rewrite rule determines whether an item enters susceptibility analysis. Rewrites from one base item remain together in every training, validation, bootstrap, and held out split. Grouping is mandatory.

The primary prediction model compares original prompt text plus native confidence against the same inputs plus original DIM geometry. Probability metrics such as log loss and Brier score carry primary weight. AUROC and area under the precision-recall curve remain secondary.

Generic activation controls include activation norm, hidden-state principal components, 20 random directions, 20 orthogonal random directions, and 20 label-permuted DIM directions under the same grouped evaluation contract. A factual direction enters only under a compatible, previously frozen activation contract.

Identical-prompt repeats estimate behavioral noise. Accepted rewrite flips must be interpreted against that baseline before the project sizes a larger Claim 2 study.

## Weakness and conflict

The current weakness feature is DIM boundary proximity, computed from the standardized distance between the original DIM score and its decision boundary, with smaller absolute distance indicating greater geometric weakness before any rewrite appears.

The 32-item diagnostic stratum captures one conflict type, disagreement between geometry-based vulnerability and native-confidence vulnerability. Broader conflict measures remain unfinished. Probe-label disagreement, probe-behavior disagreement, and disagreement among value-specific directions require separate prospective definitions before use.

## Explicit task competence

Before strong interpretation of the internal relation signal, each model needs a dedicated behavioral report that covers answer accuracy, calibration, answer mapping, format compliance, and repeat stability on the explicit ValuePrism relation task.

Human source labels later support relation-clear, ambiguous, invalid, and disputed strata. This report remains separate from Claim 2 stability because a model can answer the stored relation incorrectly yet remain stable across every accepted rewrite.

## Freeze contents

The technical v2 freeze binds the following objects.

1. Claim 1 first-stage rubric and definitions
2. Claim 1 second-stage holder and object rubric
3. Claim 2 rewrite rubric and diagnostic flags
4. Reviewer instructions and eligibility rules
5. Calibration material
6. Blank reviewer sheets
7. Base-item, candidate, control, and board manifests
8. Representative and disagreement strata
9. Stable identifier join contract
10. Independent-consensus and adjudicated-secondary rules
11. Strict Claim 1 board and strict Claim 2 item rules
12. Statistical unit, outcome target, and grouped split rules
13. Package locking code and tests
14. v1 package hashes and superseded status
15. Prospective-state assertions

The prospective-state record states that no human study labels, Claim 2 rephrasing outcomes, susceptibility estimates, or sealed confirmatory outputs were observed, while also recording unchanged candidate text, no candidate regeneration, and byte-identical preservation of v1.

Required tests reject uncertainty from primary acceptance, reject a preserved rewrite when its source cell lacks independent consensus clarity, verify strict item and strict board rules, confirm reviewer sheets contain no hidden fields, prove every stable identifier join, preserve the 80 and 32 strata, and verify v1 hash stability.

## Ordered gates

### Gate 1

Implement and freeze both v2 instruments, their packages, joins, analysis configuration, tests, and receipt. Mark v1 superseded before distribution.

### Gate 2

Run Claim 1 first-stage review and Claim 2 rewrite review in parallel. Lock both independent submissions before computing consensus or opening protected outcomes.

### Gate 3

Generate second-stage holder and object sheets for eligible consideration pairs. Lock those submissions, verify the stable identifier join, and materialize the prospective analysis manifests.

### Gate 4

Measure accepted-rewrite susceptibility and identical-prompt repeat noise. Analyze the 80 representative items separately from the 32 disagreement items.

### Gate 5

Run grouped held out prediction with text and native confidence as the baseline. Add DIM geometry, then compare generic activation and unrelated-direction controls.

### Gate 6

Complete the independent 100-board yield audit, freeze the smallest effect of scientific interest, alpha, desired power, reserve, and confirmatory sample size, then open the sealed Llama result only after those decisions and review populations lock. Gemma follows as the prespecified replication.

## Follow-up verdict analysis

The complete-verdict study begins after the reason-level pilot establishes usable flip prevalence, and it retains support, opposition, permission, uncertainty, conditionality, and refusal instead of forcing every answer into `Supports` or `Opposes`.

The analysis asks whether original relation state predicts complete-verdict changes and whether observed reason changes propagate to those changes. Separate patterns locate different failure stages.

| Pattern | Interpretation target |
| --- | --- |
| Reason changes and verdict remains stable | Later composition compensates for one unstable reason |
| Reason remains stable and verdict changes | Instability arises during composition or final reporting |
| Both change together | Reason-level instability may propagate into the final stance |
| Neither changes | Stable behavior under the tested rewrites |

## Follow-up causal analysis

Real-checkpoint activation access has passed for Llama layer 19 and Gemma layer 27. Scientific causal testing remains unopened.

The next gates test symmetric signed doses in frozen projection-standard-deviation units, semantic margins under answer-map reversal, literal answer-token margins, mapping compliance, norm-matched random directions, orthogonal directions, the wrong layer, the wrong token, logistic geometry, entropy, unrelated damage, runtime, VRAM, hashes, and parameter preservation.

A causal claim requires a signed, smooth, mapping-robust effect that exceeds direction and location controls at moderate strength. Prediction flips, hook execution, or movement only at destructive strength do not satisfy that criterion.

## Deferred mechanism work

The following branches remain visible but deferred until the core prediction result is known.

| Branch | Question |
| --- | --- |
| J-space | Does joint relation structure occupy a distinct interaction subspace |
| Truth and polarity comparisons | Does the relation direction reflect truth, answer polarity, or task-general response structure |
| Value-group transfer | Does geometry transfer across held out families of considerations |
| Shared axis versus low rank | Is one direction sufficient, or does the representation require several dimensions |
| LEACE and INLP | Does removing relation information alter behavior or expose distributed encoding |
| Sparse autoencoders and activation patching | Can more localized mechanisms recover the same effect |
| Checkpoint and smaller-model transfer | How stable is the effect across scale and training history |
| Verdict steering | Can a validated intervention influence the complete action verdict selectively |

These branches characterize the mechanism after the prediction gate. They do not substitute for evidence that the original state predicts future rephrasing behavior.

## Result branches

| Observed result | Scientific consequence |
| --- | --- |
| Relation signal weakens on human-clear sources | Recast development evidence as dataset-wide decoding with limited construct validity |
| Relation signal survives but geometry does not predict flips | Keep the representation result and reject the proposed warning-signal claim under the tested design |
| Geometry predicts representative-item susceptibility beyond text and confidence | Support prospective reason-level fragility prediction and proceed to verdict propagation |
| Geometry wins only in the 32-item disagreement stratum | Treat the finding as diagnostic and redesign representative sampling before a prevalence claim |
| Accepted rewrites do not exceed repeat noise | Stop larger Claim 2 scaling and analyze behavioral nondeterminism or task design |
| Reason changes occur while verdicts remain stable | Study compensation during reason composition |
| Verdicts change while measured reasons remain stable | Study unmeasured reasons, composition, and reporting |
| Causal effects fail specificity controls | Retain correlational geometry and reject causal use under the tested intervention family |
| Causal effects pass at moderate strength | Support selective use of the measured direction and test transfer to verdicts |

Negative outcomes terminate only the tested representation, review rule, predictor, or intervention family. They do not establish that moral reasoning is theoretically inaccessible.

## Minimum and extended papers

The minimum paper requires the human-audited Claim 1 analysis, explicit task-competence report, representative Claim 2 viability result, grouped incremental prediction test, specificity controls, legitimate power decision, sealed Llama confirmation, and Gemma replication.

The extended paper adds complete-verdict propagation, causal efficacy, dimensionality, value-group transfer, and mechanism localization. Those additions wait until the core prediction result identifies which branch carries useful information.

The next implementation should emit machine-readable population manifests before model inference begins. That artifact will let every reported table trace back to independent review labels, sampling stratum, stable source identifiers, and the exact prospective rule that admitted each item.
