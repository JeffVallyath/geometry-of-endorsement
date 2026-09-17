# Figure captions

Methods, measurements, and interpretation for the representation and editing figures.

## Scope

The representation figure set reports development evidence about **decodability**: whether a
linear readout of the model's activations tracks how a consideration bears on an
action. All five figures agree across Llama and Gemma.

That agreement is real for this question, and it stops here. Later causal and
reliability results do not replicate symmetrically across studies. Qualification
failures and bounded positive outcomes differ by experiment; the
[results page](../docs/RESULTS_AND_CLAIMS.md) keeps their scopes separate.

## Shared definitions

**The checkerboard.** A checkerboard crosses two situations with two
considerations. Each consideration flips between Supports and Opposes across the
two situations, so a preference for one situation or one consideration cancels.

**Checkerboard interaction $I_b$.** Each scorer is standardized using a mean and
standard deviation frozen on the pilot select split. The four standardized cell
scores are then combined:

```math
I_b = \tilde{s}_{11} - \tilde{s}_{12} - \tilde{s}_{21} + \tilde{s}_{22}.
```

Any score that is additive in situation and consideration gives $I_b = 0$
exactly. The measure therefore responds only to the situation-by-consideration
interaction. It is a standardized difference-in-differences,
expressed in selection-split standard deviations. Reported values are the mean
over the evaluation checkerboards.

**Splits.** Probes are fitted on the pilot train split, which has 1,500 rows. The
layer and the standardization constants are chosen on the pilot select split,
which has 300 rows across 75 checkerboards. Every reported metric is computed on
the pilot evaluation split, which has 500 rows across 125 checkerboards.

The evaluation split did not participate in choosing the layer. No panel in this
set touches the audited confirmatory boards or the 7,394-row strict test set.
Every figure is development evidence.

**Confidence intervals.** Intervals on the representation study's checkerboard interaction estimates are
95% *normal* intervals built from the frozen dyadic-robust standard error. They
are not bootstrap intervals. Figure 4 is the one exception: its intervals are
group bootstrap intervals.

---

## Figure 1: Decodability across layers

<a id="figure-1--decodability-across-layers"></a>

A support/opposition direction is fitted independently at every layer, and each
layer is then scored on the evaluation split. Panels **(a, b)** show the
checkerboard interaction. Panels **(c, d)** show AUROC as the secondary
quantity. The left column is Llama-3.1-8B-Instruct, which has 32 layers; the
right column is Gemma-2-9B-it, which has 42. Rows share a y-axis, so the two
models can be compared directly.

Both models sit at chance through the early layers and then rise to a plateau.
The rise falls around layers 10 to 15 in Llama. In Gemma it comes later, around
layers 17 to 25.

The dashed red line marks the frozen layer used in every later figure. That layer
was chosen on the pilot select split, using the prespecified criterion
`mean_mirrored_pairwise`, before the evaluation split was scored.

The selected layer is demonstrably not the peak of the plotted curves.

In Llama, both curves peak at layer 21. There the AUROC is 0.7348 and the
checkerboard interaction is 1.6528. The selected layer 19 gives 0.7320 and
1.6470 instead.

In Gemma, both curves peak at layer 28, at 0.8306 and 2.3366. The selected layer
27 gives 0.8291 and 2.3221.

In each model the selection rule gave up a little of the plotted quantity
relative to that quantity's own maximum. That is what selection on a separate
split should look like. The curves are development-split evaluation metrics,
plotted to show shape, and they had no part in choosing the layer.

## Figure 2: Scorers at the selected layer

<a id="figure-2--scorers-at-the-selected-layer"></a>

Four scorers are compared at the selected layer: the model answer margin, the
support/opposition direction, the logistic activation probe, and the frozen
MiniLM comparator. Panel **(a)** is Llama at layer 19 and panel **(b)** is Gemma
at layer 27. Error bars are 95% normal intervals from the dyadic-robust standard
error.

The frozen MiniLM comparator sees the situation and the consideration as text and
never sees activations. It reaches a checkerboard interaction of 0.28. That is
small, but it is not zero, so wording alone carries a little interaction signal.

The preregistered gate is therefore stated as a margin over that comparator,
rather than as a claim that the comparator scores nothing. The support/opposition
direction exceeds the comparator by 1.363, with an interval of 1.089 to 1.637, in
Llama. In Gemma it exceeds the comparator by 2.038, with an interval of 1.661 to
2.415. Both intervals exclude zero. Both runs record this as one of the four
gates the frozen direction cleared.

Scorer colours and ordering are held fixed across every figure in this set.

## Figure 3: Checkerboard interaction against the permutation null

<a id="figure-3--checkerboard-interaction-against-the-permutation-null"></a>

The grey histogram is the null distribution of the checkerboard interaction, and
the coloured line is the observed value. Panels **(a, c)** are Llama and panels
**(b, d)** are Gemma. The top row is the support/opposition direction and the
bottom row is the logistic activation probe. All four panels use one common bin
grid and a density y-axis, because the number of draws differs between the
columns and raw counts would not be comparable.

**How the null is built.** Each draw visits every training situation
independently and either keeps or reverses all of that situation's labels. Llama
has 1,454 training situations across its 1,500 training rows. The draw preserves
the rows themselves, the grouping into situations, and the label composition
within each situation. What it randomizes is the orientation that the support/opposition
direction can learn.

Both probes are then refit from scratch on the flipped training labels. The
select split refits its own standardization constants for that draw. The
checkerboard interaction is recomputed on the 125 held-out evaluation boards.

**How the p-value is defined.** The reported value is a one-sided empirical
permutation p-value, $(k+1)/(B+1)$, where $k$ counts draws at or above the
observed value. This is the prespecified test in both the original run and the
re-run. A two-sided version of the same 10,000 draws gives 0.0245 for the Llama
support/opposition direction, so the conclusion does not turn on sidedness.

**The Llama panels show the later refit-null analysis.** They use 10,000 draws
with the procedure described above. The historical source summary used 200 draws
for improvement over the text comparator, and both Llama probes reported 0.005,
the resolution floor. That historical test and the later refitted-interaction
null are not interchangeable; the change cannot be interpreted simply as a more
precise estimate of the same tail. In the later analysis, the logistic activation
probe gives 0.0001 with no draw reaching the observed value. The support/opposition
direction gives 0.0132, with 131 draws at or above it. Both are below 0.05.

**The Gemma panels are still at 200 draws.** That is low resolution, but it is not
uniformly censored. Only Gemma's logistic result sits at the floor of 1/201.
Gemma's support/opposition direction has 2 draws at or above the observed value,
giving a measured 0.0149. Gemma was not re-run because its layer-27 activations
were not retained.

**Why the two rows differ.** The support/opposition direction's null has a
standard deviation of 1.01, with 46% of draws beyond an absolute interaction of
1, and it is visibly bimodal. The logistic probe's null has a standard deviation
of 0.46, with 2.5% of draws beyond that threshold, and is unimodal near zero.

The consequence is that the support/opposition direction gives the weaker of the
two permutation tests, even though its point estimate is comparable. Read the
logistic panels as the sharper evidence. This experiment does not isolate why the
two nulls differ; regularization is one plausible contributor, but it is not
separated here from the other differences between the estimators.

## Figure 4: Factual True/False positive control

<a id="figure-4--factual-truefalse-positive-control"></a>

This control applies the same extraction and direction-fitting machinery to a
task whose ground truth is already known, using the `cities` and `neg_cities`
sources. It acts as a gate: the representation-study runner requires each model to pass its own
truth control before the relation result is reported.

Panel **(a)** is the development layer sweep under both answer mappings. Panel
**(b)** is the held-out test, with 95% group bootstrap intervals over 2,000
replicates.

The standardized separation $T$ is a True-minus-False mean difference measured in
training-projection standard deviations, averaged over eight training partitions.
Its scale is therefore set by the training projections themselves.

An earlier version of this control failed strict review. The probe there was
reading the answer token rather than the semantics. The symptom was a
reversed-instruction AUROC of 0.0003 while the reversed-*literal*
True-minus-False AUROC stayed at 0.9997.

The version shown here uses neutral symbols whose meaning is reversed across
examples, and it tests transfer to a held-out answer format. In panel **(a)**,
the standard and reversed mappings diverge after roughly the selected layer 14.
That divergence is the residue of the confound the redesign was meant to remove.

The Gemma point is drawn as an open marker at layer 25 because only its point
estimate was retained. No interval is available for it.

## Figure 5: Specificity

<a id="figure-5--specificity"></a>

Each panel places the relation effect beside its controls, in the same units and
on the same axis. Panel **(a)** is Llama at layer 19 and panel **(b)** is Gemma
at layer 27. The rows are grouped into three bands.

**Relation.** The first band holds the fitted support/opposition direction, and
the same frozen direction evaluated under a held-out answer template. The
template result is 2.16 in Llama and 2.26 in Gemma, drawn as an open marker
because no interval was retained. The effect therefore survives a change of
answer format, which is the control against a scorer that is really reading the
answer token or the surface form of the prompt.

**Confound baselines.** The second band holds the frozen MiniLM comparator and
the deterministic random item scores. The comparator reaches 0.28 and the random
item scores reach -0.26. Both are empirical: they are what the estimator returns
for wording alone, and for a meaningless per-item score.

The random baseline needs care in description. Its implementation hashes each
item identifier to a scalar in the range 0 to 1 and scores the item with that
number. No vector in activation space is ever sampled, so this baseline says
nothing about arbitrary directions in the residual stream.

**Zero by construction.** The third band holds situation-only activations,
consideration-only activations, and an additive separate encoding. All three give
an interaction of exactly 0, analytically, because the checkerboard interaction
annihilates any score that is additive in situation and consideration. These rows
are not evidence about the model. They are a validity check that the estimator
behaves as the algebra says, and they are kept in a separate band for that
reason.

**What this figure does not cover.** Two gaps remain, and both need runs outside
this representation study (`M1`). Specificity against other semantic directions, such as truth, sentiment, or
actor identity, is not testable here, because this study fits only the relation
direction; that comparison belongs to the later intervention work. Specificity
against arbitrary activation-space directions is not supplied either, because this study
contains no such control.

So the figure establishes that the effect is not wording, not answer format, and
not an artefact of the board algebra. It does not establish that an arbitrary
direction in the residual stream would fail to produce it.

## Figure 6: Direct answers and broader counterfactual consequences

<a id="figure-6--direct-answers-and-broader-counterfactual-consequences"></a>

![Direct-margin hits compared with broader consequence recovery](fig06_answer_consequences.png)

The matched Gemma comparison uses the same 96 test contexts at target fraction
0.75: the requested answer-margin change is that fraction of the change produced
by rewriting the fact in the text. Rank-one relation steering and whole-state interpolation reach the requested
direct-answer margin within tolerance in 100% of these contexts. This is a margin
criterion, not perfect hard-answer accuracy or a full strong-target pass: some
targets fall below the frozen absolute-strength requirement. The direction fitted
to the matched setting also failed qualification; the displayed rank-one arm uses
the original frozen direction.

The right panel measures recovery of the natural change on held-out questions,
including complementary, paraphrased, and unchanged relations. Recovery is a
normalized score, not a proportion of correct answers; zero is the unchanged
starting state and one is the natural change. Negative recovery moves farther
from the natural pattern. Error bars are the saved 95% context-bootstrap intervals.
The natural changed-state patch is a reference intervention, not margin-tuned.
Llama failed natural-reference qualification and is not included in this comparison.

[Saved results](../reproducibility/representation/counterfactual_fidelity/results.json),
[per-world measurements](../reproducibility/representation/counterfactual_fidelity/scores/final_gemma.jsonl.gz),
and [plot data](../artifacts/figures/consequence_comparisons.json) retain the exact
estimates. The supported table replay checks the behavioral measurements; it does
not recreate the omitted fitted direction or the supporting later-layer readout.

## Figure 7: Correct current facts can still leave downstream answers dependent on source history

<a id="figure-7--correct-current-facts-can-still-leave-downstream-answers-dependent-on-source-history"></a>

![Matched histories and eight qualifying-question rates with adjusted intervals above zero](fig07_source_history.png)

**A, matched-history design.** Different starting histories receive the same
updates, producing identical intended current facts before the same fresh
downstream question is asked. The prospective matched-history study (`V10`)
compares all 8 starting assignments to three
addressed records and checks complete intended final-table equality, including
untouched facts. Relevant direct facts are checked after updating. The question
and response interface are fixed across histories. This is a behavioral test:
neither equality of hidden activations nor physical erasure of history is assumed.

**B, prospective matched-history result.** Each row is one model and update group:
Constrained learned editor (`INV_PAIR_NLL`), Unrestricted consistency-trained
editor (`FREE_PAIR_CONSISTENCY`), Existing textual correction
(`EXISTING_CORRECTION`), or Latest-value wording (`LATEST_SAME_WORDING`). Dots and
squares distinguish learned and textual updates, not a ranking of methods.
Points copy the saved primary mean qualifying-question rates; intervals copy the exact
multiplicity-adjusted 99.375% case-bootstrap intervals (10,000 draws, fixed seed
2609141002, correction across 8 cells). All eight intervals lie above zero.

Each cell has 64 benchmark cases and 18 fixed joint-question opportunities per benchmark case
(9 semantic questions under two answer-code draws), or 1,152 opportunities per
seed or textual condition. Ineligible questions remain in the denominator.
The two learned-seed scores are averaged within each benchmark case before averaging over
cases; models and update groups are not pooled. A question qualifies when its
individual facts are answered correctly and validly in every history, both
references answer correctly, and valid answers to the question combining those
facts differ across histories. The references supply current facts from the start
(`native-final`) or apply the same update to already-correct facts (`no-op`). This rate
is not benchmark case prevalence, a filter-conditional rate, or an individual-answer error
rate. These intervals support recurrence within the generated-case sampling
scheme, not universal failure or a pairwise method comparison.

The result demonstrates **history-dependent answers despite correct
individual-fact checks**. It does not establish that updated facts are absent internally,
that separate physical old/new stores exist, or that a specific neural mechanism
has been identified. Correct facts with a history-sensitive reader remain possible.

The selected Qwen Wren/Orla case is kept separately as an
[illustrative repeat-checked example](../reproducibility/state_sufficiency/independent_verification/examples/EXAMPLES.md#repeat_checked_positive),
not a prevalence estimate. Its direct facts are correct across histories but
the fixed same-side question receives different answers; the exact joint query
matches both fresh repeat passes. Operand scores are from the core, not all
independently repeated. See also the
[main-text account](../docs/RESULTS_AND_CLAIMS.md#a-repeat-checked-illustration-and-provenance-limits)
and [saved example](../reproducibility/state_sufficiency/v10/selected_repeat_checked_example.json).

[Saved primary output](../reproducibility/state_sufficiency/v10/primary_results.json),
[specification](../reproducibility/state_sufficiency/v10/specification_summary.json),
and [full-precision plot data](../artifacts/figures/v10_source_history.json)
bind every plotted estimate and interval. Regenerate with `python -m repro figures`
using [the plotting code](../src/repro/source_history_figure.py). No new inference,
estimand, or uncertainty calculation is performed by the renderer. The earlier
archival shared-state figure (`V6`) is retained as [Supplementary Figure S3](#supplementary-figure-s3--v6-descriptivearchival-source-history-census).

## Figure 8: Swapping later activations changes the joint answer

<a id="figure-8--swapping-later-activations-changes-the-joint-answer"></a>

![One saved Gemma crossover example](fig08_cache_crossover.png)

In this selected Gemma example, the current records say Dion and Orla both support
Bridge. The original histories give very different answers to whether both support
it. Combining the first 16 layers' stored activations from one history with the
remaining 26 from the other shifts the joint answer toward the latter history.
The activations are combined before the question; model weights stay fixed.

Bars show normalized probability of semantic Yes under the main answer format.
Their widths use the unrounded saved-score values. The adjacent columns show
normalized probabilities of the two correct direct answers, which remain near
one. Numerical labels are rounded to six decimal places. Colors identify the
history supplying the later activations, not whether the answer is correct.

This is one selected example, not a prevalence estimate. Across the six selected
cases, the main format gives five clear later-state effects and one mixed result.
The alternate format gives one clear effect, two mixed results and three cases
with insufficient original separation. The swap establishes coarse causal
influence, not a unique mechanism or where the history information first arose.

[Unrounded plot data](../artifacts/figures/cache_crossover.json),
[the six-case table](../reproducibility/cache_crossover/six_cases.csv), and
[saved-score replay and provenance](../reproducibility/cache_crossover/README.md)
link the figure to the completed archive. The example is `SSC1-FINAL-0026`,
question `SSC1-FINAL-0026-d1-extra2`.

## Figure 9 - Changing the later stored activations changes the downstream answer

![The selected extension under both answer formats](fig09_cache_crossover_extension.png)

One retrospectively selected Gemma history pair combines the strong behavioral
controls and the causal crossover result. Both histories require 2 fact changes
and have 517-token prefixes. The current records say Ada and Gita both oppose
Theater, so the correct answer to whether at least one supports it is No.

Both answer formats show the downstream answer following the history supplying
the later stored activations. The swap combines the first 16 layers from one
history with the remaining 26 from the other before the question. Model weights
stay fixed. Bars use unrounded normalized probabilities of semantic Yes; labels
are rounded to six decimal places. Colors identify the history supplying the
later layers, not correctness. The coded format maps its answer tokens back to
semantic Yes/No separately for each question.

All 8 strict checks of separately measured direct facts pass. Their minimum
correct-answer probability is 0.999230; the largest change from the corresponding
original state is 0.00008612. The two formats test the same example,
not independent cases or a prevalence estimate. The intervention covers a large
block of stored computation. Earlier computation may have contributed to those
activations; a unique circuit or stored obsolete fact remains unidentified.

[Source tables](../artifacts/cache_crossover/extension_joint.csv),
[direct checks](../artifacts/cache_crossover/extension_direct.csv),
[plot data](../artifacts/figures/cache_crossover_extension.json), and
[replay and provenance](../reproducibility/cache_crossover/README.md#one-case-extension)
provide the full values. The original six-case results remain separate in the
earlier crossover figure and [appendix](../docs/CACHE_CROSSOVER_APPENDIX.md).

## Figure 10. Update procedures and downstream coherence

![Update-method comparison](fig10_update_methods.png)

The completed matched-history comparison tests 64 benchmark cases per model.
A benchmark case succeeds only when all 34 questions are correct under every one of four
histories ending at the same facts. Panel A shows percentage-point changes
relative to the factual-update baseline: attempted internal fact refresh plus
the latest textual correction (`FIELD_PLUS_LATEST_ERRATUM`).
Points average both learned seeds inside each benchmark case; error bars are 98.75% paired
case-bootstrap intervals. Triangles retain the separate seed effects.
"Constrained" denotes `INV_PAIR_NLL`; "consistency-trained" denotes
`FREE_PAIR_CONSISTENCY`. Models are not pooled.

Both Qwen intervals exclude zero. Gemma's constrained interval touches zero and
its consistency-trained interval crosses zero. Panel B shows absolute complete-case success on the full percentage scale: neither learned method's seed mean
reaches half the benchmark cases in either model. The baseline used its recorded fallback
in 192/256 contexts per model. The improvements do not establish a universal
consistency cure. This comparison supports treating downstream coherence as a
separate evaluation objective; the matched-history sufficiency test remains the
paper's centerpiece.

[Saved rows, tables and reconstruction](../reproducibility/update_method_comparison/README.md)
and [plotting code](../src/repro/update_methods_figure.py) reproduce both panels
through `python -m repro figures` without model inference.

## Supplementary Figure S1: Broader single-edit training and later sequences

<a id="supplementary-figure-s1--broader-single-edit-training-and-later-sequences"></a>

![Matched effects of broader single-edit training](supplementary/figS1_editing_coverage.png)

This is a matched comparison within the coverage study, not a trend across studies.
Both editors were trained on individual edits; the broader condition was trained
on more consequences of each edit. The evaluation asks whether all questions about
the resulting program are correct after two or three updates, averaging those two
outcomes within each scene. Reversed orders are not counted as additional scenes.

A scene is a synthetic situation tested under several update sequences. It is
the unit resampled for this comparison, not an individual answer.

The constrained editor limits changes to preserve other addressed relations; free
overwrite is the less restricted comparison. Each row shows broader minus narrower
training for the same editor and model. Positive values favor broader training.
The dot and interval average both training seeds within each scene. The triangles
show the separate seed effects, not extra independent observations.

Intervals are the saved 98.75% paired whole-scene bootstrap intervals. Gemma has
64 scenes and Qwen 32. Three intervals exclude zero in favor of broader training;
Gemma's free-overwrite interval crosses zero. This relative improvement does not
establish a pass of the full absolute joint-control requirements.

The [original comparison table](../reproducibility/relational_editing/v4/expected/T10_coverage_contrasts.csv)
contains the estimates, all seed-specific intervals, and the other study contrasts.
The [plot data](../artifacts/figures/editing_comparisons.json) retain the selected
values without recomputing uncertainty.

## Supplementary Figure S2: Changing how the same edited state is queried

<a id="supplementary-figure-s2--changing-how-the-same-edited-state-is-queried"></a>

![Paired changes from alternative fixed readers](supplementary/figS2_fixed_readers.png)

Each row compares an alternative question with the original question on the same
saved edited states. The paraphrase asks whether two people hold matching positions.
Here a scene is the synthetic situation shared by the compared question formats.
The explicit-rule version adds an explanation: the answer is yes if both support
or both oppose the proposal, and no otherwise. That extra instruction changes the
readout interface; it is not evidence of better answers under the original wording.

The endpoint is same-side question accuracy after a three-edit sequence, averaged
within each scene and then across the two original training seeds. The panel uses
the study's fixed early ordering. Intervals are the saved 99.375% paired whole-scene
bootstrap intervals. Gemma has 64 scenes and Qwen 32. Seed effects are shown
separately, and neither model nor study populations are pooled.

None of these corrected intervals establishes an improvement. Several are wholly
negative; the others cross zero, which is not evidence of equivalence. This
comparison is distinct from the full source-independence test: a change in
question accuracy does not show that overwritten starting relations have ceased
to affect the answers.

The [paired scene records](../reproducibility/relational_editing/v6/scores/reader_paired_contrasts.json.gz),
[exact reader prompts](../reproducibility/relational_editing/v6/data/READERS.json),
and [plot data](../artifacts/figures/editing_comparisons.json) give the full comparisons.

## Supplementary Figure S3: Descriptive history dependence in archived cases

<a id="supplementary-figure-s3--v6-descriptivearchival-source-history-census"></a>

![Descriptive history dependence in archived benchmark cases](supplementary/figS3_source_history_readable.png)

This descriptive figure uses the earlier shared-state editing study (`V6`),
not the prospective confirmation. Previously Figure 7, its exact counts and
provenance are preserved here, with clearer display labels. The prospective
study (`V10`) supplies the main Figure 7 evidence.

The [original PNG](supplementary/figS3_v6_source_history.png) and
[original PDF](supplementary/figS3_v6_source_history.pdf) remain byte-for-byte
unchanged. The display above uses the same counts with reader-facing labels.

The top panel shows a saved synthetic design: 4 starting states receive the same
requested final assignments before fresh questions are supplied. The lower panel
uses every benchmark case under the original reader: Gemma has 64 benchmark cases and Qwen
32, with both editors and both original training seeds shown separately.

A case passes all individual-fact checks only when all 16 direct/opposes questions are correct
across every starting state. The red segment counts such benchmark cases where at least
one of the 18 joint both/either/same questions still changes its answer with the
overwritten starting values. The teal segment has correct individual-fact answers and no
joint disagreement; its joint answers may still be consistently wrong. Gray
marks benchmark cases with at least one incorrect individual-fact answer. The right-hand counts
use cases passing all individual-fact checks as their denominator; the bars use all cases.

These are descriptive counts derived from saved per-case sufficient statistics,
not a new primary test, an uncertainty interval, or a pooled model/seed estimate.
They isolate observed answer disagreements, not a claim about an identified
internal mechanism. The top panel is a design example, not a sequence of model answers.
This descriptive census is distinct from the separate fixed-criterion replay of the same archived study.

[Case statistics](../reproducibility/relational_editing/v6/scores/source_root_statistics.json.gz),
[synthetic source worlds](../reproducibility/relational_editing/v6/data/gemma_source_roots.jsonl),
and [unchanged plot data](../artifacts/figures/consequence_comparisons.json) provide
the population and exact counts. Regenerate through `python -m repro figures`
using [the archival plotting code](../src/repro/consequence_figures.py).
