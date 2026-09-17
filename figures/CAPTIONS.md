# Figure captions

Main-paper captions; exact procedures, numerical detail, and artifact links are
in the [figure methods and evidence appendix](CAPTION_DETAILS.md).

## Scope

The five representation figures report development evidence about decodability
in Llama and Gemma, not confirmation or a complete mechanism. Later studies have
experiment-specific qualifications and outcomes; see the
[results page](../docs/RESULTS_AND_CLAIMS.md).

## Shared definitions

A *checkerboard* crosses situations and considerations whose support/opposition
labels reverse. Its interaction cancels fixed preferences for either input:
higher positive values indicate stronger contextual separation, measured in
selection-split standard deviations. Relation readouts use 125 evaluation
checkerboards (500 rows), separate from fitting and layer selection; they do not
use the confirmatory or strict test sets.
[Exact definitions and splits](CAPTION_DETAILS.md#shared-definitions).

## Figure 1: Decodability across layers

<a id="figure-1--decodability-across-layers"></a>

We test where a linear readout can recover whether a consideration supports or
opposes an action, fitting a readout at each layer and evaluating every layer on
the same checkerboards.

Panels **(a, b)** show contextual separation; **(c, d)** show AUROC. Llama
(left) rises around layers 10–15 and Gemma (right) around 17–25, then both
plateau. Red lines mark layers 19 and 27, selected on separate data rather than
at the peaks of the plotted curves. The shared evaluation set contains 125
checkerboards (500 rows); axes match across models. These are development
readouts, not confirmatory evidence or causal effects.

[Methods and evidence](CAPTION_DETAILS.md#figure-1-decodability-across-layers).

## Figure 2: Scorers at the selected layer

<a id="figure-2--scorers-at-the-selected-layer"></a>

Does the activation readout capture the contextual relation better than wording
alone? We compare the direction, a logistic activation probe, the model's answer
margin, and a frozen MiniLM text comparator on the same checkerboards.

Panel **(a)** shows Llama at layer 19 and **(b)** Gemma at layer 27. The direction
exceeds the text comparator by 1.363 [1.089, 1.637] and 2.038 [1.661, 2.415],
respectively; the comparator itself has a small, nonzero interaction of 0.28.
Error bars and quoted intervals are 95% normal intervals using dyadic-robust
standard errors over 125 evaluation checkerboards. This is development evidence
of improvement over the tested text baseline, not proof that wording carries
no signal.

[Methods and evidence](CAPTION_DETAILS.md#figure-2-scorers-at-the-selected-layer).

## Figure 3: Checkerboard interaction against the permutation null

<a id="figure-3--checkerboard-interaction-against-the-permutation-null"></a>

Could the relation signal arise after randomizing which orientation counts as
support versus opposition? We keep examples and situation groups fixed, randomly
reverse labels by situation, and refit the readouts before evaluating on the
unchanged checkerboards.

Grey histograms show randomized interactions; colored lines mark observed
values. Panels **(a, c)** are Llama and **(b, d)** Gemma; rows show the direction
and logistic probe, respectively. All four one-sided p-values are below 0.05,
with sharper evidence from the logistic probe. Histograms share density scales;
Llama uses 10,000 draws and Gemma 200, so p-value resolution differs. The
historical comparator-improvement test is a different analysis, not an earlier
estimate of this same tail.

[Methods and evidence](CAPTION_DETAILS.md#figure-3-checkerboard-interaction-against-the-permutation-null).

## Figure 4: Factual True/False positive control

<a id="figure-4--factual-truefalse-positive-control"></a>

Can the readout distinguish known true and false statements rather than answer
tokens? We reverse the meanings of neutral answer symbols and test transfer to
a held-out answer format.

Panel **(a)** shows the development layer sweep under both mappings; **(b)**
shows held-out separation, with both models passing the factual control.
Separation is the True-minus-False mean difference in training-projection
standard deviations; error bars are 95% group-bootstrap intervals, while Gemma's
open marker has no retained interval. Passing qualifies this control, not every
layer: later-layer mapping divergence and the earlier failed literal-label
control remain important limitations.

[Methods and evidence](CAPTION_DETAILS.md#figure-4-factual-truefalse-positive-control).

## Figure 5: Specificity

<a id="figure-5--specificity"></a>

Could wording or answer format explain the relation effect? We hold the
checkerboards and scale fixed while comparing the learned direction with
answer-template transfer, text-only and random-item baselines, and separate-input
controls.

Panels **(a, b)** show Llama and Gemma. The relation band retains a large point estimate
under the new answer template; the empirical-baseline band is much smaller.
The separate-input band is zero by construction because additive scores cancel.
Error bars are 95% normal dyadic-robust intervals over 125 evaluation checkerboards;
open transfer markers have no retained intervals. Random item scores are not
random activation directions, and algebraic zeros are not neural evidence; other
semantic directions are not tested here.

[Methods and evidence](CAPTION_DETAILS.md#figure-5-specificity).

## Figure 6: Direct answers and broader counterfactual consequences

<a id="figure-6--direct-answers-and-broader-counterfactual-consequences"></a>

![Direct-margin hits compared with broader consequence recovery](fig06_answer_consequences.png)

Does hitting a direct-answer target recreate the consequences of changing the
underlying relation? We compare relation-direction steering with whole-state
interpolation toward a genuinely changed context, matching the direct-answer
target on the same 96 Gemma worlds.

Both interventions reach the requested margin in 100% of worlds (left), but
steering moves held-out answers farther from the natural change while
whole-state interpolation closely recovers it (right). The changed-state patch
is an untuned reference. Recovery is normalized, not accuracy: zero is unchanged,
one is the natural change, and negative values move farther away; bars are 95%
world-bootstrap intervals. Margin attainment is not perfect hard-answer accuracy
or a full strong-target pass; the matched direction failed qualification, so the
original frozen direction is used, and Llama's unqualified reference is excluded.

[Methods and evidence](CAPTION_DETAILS.md#figure-6-direct-answers-and-broader-counterfactual-consequences).

## Figure 7: Correct current facts can still leave downstream answers dependent on source history

<a id="figure-7--correct-current-facts-can-still-leave-downstream-answers-dependent-on-source-history"></a>

![Matched histories and eight V10 primary witness rates with adjusted intervals above zero](fig07_source_history.png)

Different starting histories receive the same updates and reach the same intended
current facts. After verifying the relevant facts directly, we ask the same
downstream question with the same response interface across histories.

Panel **A** shows this design; **B** reports prospective results for Gemma and
Qwen under two learned and two textual update procedures. A witness requires
correct direct facts in every history and correct reference answers, but differing
valid downstream answers. All eight multiplicity-adjusted intervals lie above
zero. Each condition uses 64 cases with a fixed set of downstream questions;
questions failing the witness criteria remain in the denominator. Intervals are
multiplicity-adjusted 99.375% root-bootstrap intervals, not pairwise method comparisons. This is
recurring behavioral dependence on overwritten history, not case prevalence,
individual-answer error, evidence of absent internal facts, or an identified
neural mechanism; recurrence is limited to the generated-case sampling scheme.

[Methods and evidence](CAPTION_DETAILS.md#figure-7-correct-current-facts-can-still-leave-downstream-answers-dependent-on-source-history).

## Figure 8: Swapping later activations changes the joint answer

<a id="figure-8--swapping-later-activations-changes-the-joint-answer"></a>

![One saved Gemma crossover example](fig08_cache_crossover.png)

Can history-dependent information in later activations change the answer?
In this selected Gemma example, both histories end with Dion and Orla supporting
Bridge, yet disagree on whether both support it. Before asking the same question,
we combine earlier activations from one history with later activations from
the other; model weights stay fixed.

Bars show normalized Yes probabilities: the joint answer follows the later-layer
donor, while adjacent columns show the two correct direct answers remaining
near one. Colors identify the donor, not correctness. This is one selected case,
not a prevalence estimate or a unique mechanism: the main format yields five
clear effects among six selected cases, while the alternate format yields only
one, with two mixed and three insufficiently separated cases.

[Methods and evidence](CAPTION_DETAILS.md#figure-8-swapping-later-activations-changes-the-joint-answer).

## Figure 9 - Changing the later stored activations changes the downstream answer

![The selected extension under both answer formats](fig09_cache_crossover_extension.png)

Does the activation-swap effect persist across answer formats with matched
histories? One retrospectively selected Gemma pair has equal update counts and
prompt lengths and the same current facts: Ada and Gita both oppose Theater.
The fixed question asks whether at least one supports it; the correct answer
is No.

Under both formats, the answer follows the history donating the later
activations, while separately measured direct facts remain correct. Bars show
normalized semantic Yes probabilities, not correctness; coded answers are mapped
back to Yes/No for each question. All 8 strict direct-fact checks pass, with
minimum correct-answer probability 0.999230. These formats test one example,
not independent cases or prevalence; the large activation swap does not identify
a unique circuit or where history information originated.

[Methods and evidence](CAPTION_DETAILS.md#figure-9---changing-the-later-stored-activations-changes-the-downstream-answer).

## Figure 10. Update procedures and downstream coherence

![Update-method comparison](fig10_update_methods.png)

Does the update procedure improve complete-case downstream accuracy? We compare
two learned procedures with fact refresh plus latest textual correction on the
same 64 cases per model, each with four histories ending at the same facts.

Panel **A** shows changes from baseline: both Qwen intervals exclude zero;
Gemma's constrained interval touches zero and its consistency-trained interval
crosses zero. Panel **B** shows absolute success, requiring all 34 questions
correct under every history; neither learned method's seed mean reaches half
the cases. Bars are 98.75% paired root-bootstrap intervals with seeds averaged
within roots; triangles show separate seed effects. Gains are relative to the
executed baseline, which used fallback in 192/256 contexts per model, not a
universal consistency cure.

[Methods and evidence](CAPTION_DETAILS.md#figure-10-update-procedures-and-downstream-coherence).

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
