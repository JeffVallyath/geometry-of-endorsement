# Results and Claims

The main result is that correctly reported current facts do not always determine
a model's later answers. Different starting histories can still produce different
answers to the same question after the facts have been brought into agreement.

The evidence answers six questions: is the relation represented; can it be
updated; do current facts determine later behavior; does the effect appear on
public data; can stored activations causally change it; and do alternative update
procedures improve downstream coherence? The sections below retain the supporting
measurements and limitations. Study codes are secondary provenance labels.

This is the current results page. The point of the tables is to show what each
experiment measured, not to make the argument depend on following every detail
of the analysis. Exact methods, saved results and reproduction limits are in
[Reproducibility](../REPRODUCIBILITY.md).

A **benchmark case** is a target fact configuration with its starting histories
and questions (`root` in saved records). An **individual-fact question** asks
about one relation; a **joint question** combines relations. A **qualifying
history-dependent question**, also called a **witness** in the records, passes
the study's factual and reference checks but receives different answers across
histories. The exact checks are stated with each result. These counting units
must not be confused: several questions and histories belong to the same case.

## Part I. Establishing the relation signal

### 1. Do the models represent support and opposition in context?

Yes. In both Llama and Gemma, we can read a support/opposition signal from the
model's activations that depends on the particular situation and reason.

The main test uses a **checkerboard**: paired situations and considerations
whose Supports/Opposes labels reverse. We standardize the scores and combine them so that a fixed
preference for a situation or consideration cancels out. Larger positive values
mean stronger context-sensitive separation on this measure.

| Development measurement | Llama | Gemma |
|---|---:|---:|
| Support/opposition direction | 1.6470 | 2.3221 |
| Logistic activation probe | 2.0836 | 2.1499 |
| Text-only comparison | 0.2842 | 0.2842 |
| Selected layer | 19 | 27 |

The direction is a linear readout fitted to separate Supports from Opposes.
The logistic probe is another fitted linear readout; the text-only comparison
asks how much of the result the wording can explain without model activations.

Before interpreting this as a new relation signal, we checked that the same
pipeline could recover factual True/False structure. Standardized separation
was 1.9245 in Llama and 1.9444 in Gemma. The answer symbols had their meanings
swapped across examples, so the model could not succeed just by favoring a
particular answer token. The earlier literal-label control failed and remains
a diagnostic of the answer-format problem, not another positive result.

These are development measurements. They support a readable relation signal,
but they do not establish correct moral judgment or tell us the complete
mechanism behind the answer. The original plots and statistical details are in
[Figure captions](../figures/CAPTIONS.md).

## Part II. What does the direction tell us?

### 2. Does distance along the direction measure how firmly the model holds its judgment?

Not reliably. Our original hypothesis was that examples nearer the decision
boundary would be easier to flip through meaning-preserving rephrasing.
The important test is whether the direction adds prediction beyond the text
and the model's own answer confidence.

| Analysis | Improvement from adding the direction score | Interval |
|---|---:|---:|
| Llama representative originals | -0.003870 | [-0.024731, 0.010818] |
| Gemma analysis planned before its outcomes | 0.002753 | [-0.000507, 0.007011] |

The intervals show uncertainty in the estimates. The improvement is measured
in log loss, which evaluates predicted probabilities.
Positive means adding the direction helped; negative means it made prediction
worse. Neither interval gives reliable evidence of an improvement.

So the direction can tell us which side the model is on without reliably
telling us how resistant that judgment is to rephrasing. This is not proof that
every activation-based predictor of instability must fail. Individual human
judgments are not distributed. The public package does not independently verify
completion of human review, and these development results are not being presented
as a separate confirmatory test.

### 3. Is the signal specifically moral, and is its effect unique?

No to the first question, and mostly no to the second on the answer we tested.
The ValuePrism direction transfers without refitting to ordinary argument
support/attack relations. Transfer back to ValuePrism is also observed.

Removing measured answer-token, sentiment and factual-truth components does
not eliminate the relation signal. This supports a more general counts-for/
counts-against interpretation. It does not mean the representations are completely
independent of one another.

Steering the relation direction changes the model's support/opposition answer.
But matched truth and sentiment interventions often change that same answer
as much or more. Each number below is the relation response slope minus the
comparison direction's slope. Positive favors relation steering; negative favors
the comparison.

| Comparison direction | Llama ValuePrism | Llama argument relations | Gemma ValuePrism | Gemma argument relations |
|---|---:|---:|---:|---:|
| Factual truth | -0.015 | -0.125 | -0.247 | -0.681 |
| Sentiment | -0.542 | -0.392 | 0.602 | -0.113 |

An exploratory follow-up matched how far the interventions moved the activation
state. Its results were mixed, and some comparisons reversed which direction
was stronger. It did not establish a general advantage for relation steering.

This motivated the idea of **more concepts than control knobs**: distinct
representations may share ways of influencing an answer. It is not proof of a
universal shared control mechanism. The separate refusal reconstruction is
retained as background, not as a test of that universal claim.

### 4. Does the same direction have a stable effect across contexts?

Not automatically. We tested more than whether the target answer moved by
recording a direction's effects across a panel of downstream answers. That
pattern is what we call its **causal fingerprint**.

The first measurement check was invalid. Random directions could look reliable
simply because they were projected through similar average gradients. That
rewarded repeatable projection geometry without establishing semantic structure.

| Final validation measurement | Llama | Gemma |
|---|---:|---:|
| Agreement between gradient predictions and small interventions | 0.9926 | 0.9984 |
| Median similarity of the same direction's fingerprint across contexts | 0.8435 | 0.4594 |

The first row is a correlation: it asks whether we can predict a small
intervention's local effect. The second asks whether the same direction produces
a similar pattern in different contexts. These are different requirements.

Local predictions were accurate in both models, but cross-context stability was
much weaker in Gemma. The random-direction numerical controls also failed their
checks. The final instrument remained invalid for the intended cross-model
claim; these results do not establish the proposed shared control geometry.

## Part III. Looking beyond the single score

### 5. Does the broader activation contain information the direction and answer leave out?

Yes, with an important distinction between the experiments. Controlled contexts
vary who holds a stance and what it concerns, then change the question. Broader
activation readouts recover relation information that the single support/
opposition score does not fully describe.

On natural disagreements, the direction usually follows the model's own answer.
A controlled task initially looked as though the model represented the right
relation but could not report it. That gap largely disappeared when we supplied
fixed examples showing how to answer. We should therefore not treat the original
gap as evidence of knowledge the model could not express.

The separate question is whether the broader activation improves prediction
after accounting for the text and the model's answer. The replication analysis
was planned in advance and found the following log-loss improvements:

| Replication population | Llama | Gemma |
|---|---:|---:|
| Examples grouped by situation | 0.064 | 0.038 |
| Examples grouped by checkerboard | 0.073 | 0.031 |

These populations came from different pre-existing development partitions, not
the strict test. They are also separate from the earlier retrospective analysis.
That earlier analysis had an invalid baseline: checkerboard position determined
the label. Removing position was a correction made after seeing the results,
not a successful test of the original plan.

The controlled and prompting observations are available as summary results;
the replication planned in advance can be checked from saved predictions. A separate
base-versus-instruction-tuned comparison is retained as background: decoding can
remain strong even when the detailed direction geometry changes.

## Part IV. Does changing the answer recreate the semantic change?

### 6. Does steering reproduce what happens when we actually change the relation?

In the measured Gemma comparison, no. Steering could reach the direct-answer
target while moving the other answers away from the pattern produced by
actually changing the relation.

The test uses paired synthetic contexts that differ only in one person's stance
toward one target. The direct question sets intervention strength. Held-out
questions then ask about the complement, rephrase the focal relation, or ask
about relations that should not change.

We measure how closely the intervention reproduces the natural pattern across
those other questions, using scales fixed on development data. A negative
recovery score means it moved farther from that pattern than the unchanged
starting state.

| Gemma intervention | Recovery of the natural change | Interval |
|---|---:|---:|
| Steering along the relation direction | -0.851 | [-1.005, -0.707] |
| Moving the whole activation toward the changed context, to the same answer target | 0.914 | [0.899, 0.926] |
| Replacing it with the changed context's activation | 0.981 | [0.980, 0.983] |

Relation steering reached the requested direct-answer margin in 100% of the
final contexts, yet did not recover the broader change. The margin measures the
model's relative preference for the answer, not just which answer won.

There is an important qualification. Reaching those margins did not satisfy
every requirement set before the experiment: some targets were below the required
absolute strength, and the direction fitted to the matched setting failed its
qualification check. The measured answer changes are not a full pass of those
stricter requirements.

A later-layer readout gave supporting summary evidence, but its result is not
independently replayed in this package. Llama's natural changed-context reference
was unstable across equivalent wording, so it did not qualify for this comparison.
That reference failure is not a negative editing result.

**Changing the target answer did not reproduce the broader consequences of
changing the underlying relation.**

![Direct-answer margins and broader counterfactual recovery](../figures/fig06_answer_consequences.png)

The [matched consequence figure](../figures/CAPTIONS.md#figure-6--direct-answers-and-broader-counterfactual-consequences)
separates direct-margin attainment from recovery of the held-out natural pattern.

## Part V. Editing a shared state

### 7. Can we change a relation before knowing which question will be asked?

Yes in bounded tests. Later work also established useful composition under fixed
criteria across Gemma and Qwen, while complete source-history sufficiency failed.
The later editors change an addressed relation in the context before seeing
the downstream question. Fresh questions then test what the edited state supports.

The initial editor produced substantial effects, but did not meet all the
reliability requirements. Some model and task branches failed qualification
before they could support the intended comparison.

A matched comparison then produced strong single edits under revised checks
of the model's starting ability. That was an explicitly amended qualification,
not a pass under the original protocol. Repeating and undoing edits still failed.

Later editors could repeat changes and restore earlier answers. A less
restricted editor, called free overwrite, also worked, so that capability was
not unique to the constrained design. Training on a broader set of consequences
of single edits then improved some unseen sequences of changes. At that stage,
reliable joint control and full replication across models were not established.

[Supplementary Figure S1](../figures/CAPTIONS.md#supplementary-figure-s1--broader-single-edit-training-and-later-sequences)
shows the matched comparison between broader and narrower single-edit training,
including both seed effects and the uncertainty in each model/editor comparison.

One prepared study defined fixed ways of asking the readout questions but
produced no new efficacy measurements. Preparation alone is neither a positive
nor a negative result. The next experiment asked a harder question. If we reach
the same final relations from different starting relations, do the answers
still depend on what was overwritten?

The earlier shared-state editing study (`V6`) below uses the ordinary question wording and the constrained editor,
which is designed to change the requested relation while keeping the others stable.
Repeating an edit should leave its result unchanged; restoration should recover
answers that were correct in the starting state. Joint updates should not
introduce errors on relations that were supposed to stay unchanged.

| Model and training seed | Requested change correct | Most disagreement after repeats | Lowest restoration rate | New errors on unchanged relations after joint updates |
|---|---:|---:|---:|---:|
| Gemma seed 0 | 90.6250% | 0.1302% | 98.1527% | 8.0729% |
| Gemma seed 1 | 92.9688% | 0.1302% | 99.0699% | 8.9193% |
| Qwen seed 0 | 88.5417% | 0.5208% | 98.9993% | 5.4688% |
| Qwen seed 1 | 93.7500% | 0.2604% | 96.2126% | 6.9010% |

These are point estimates on the tested program subset, not population
guarantees. Both original training seeds are shown and must meet the requirements;
success from a single seed does not replace success across both.

The best rate of getting the whole declared answer set correct consistently
across starting states was 29.6875%. No tested setting in that study met all the requirements
for independence from the starting relations. Consistent wrong answers do not
count as success.

![Archived answers combining facts disagree despite correct individual facts](../figures/supplementary/figS3_source_history_readable.png)

The [archival Supplementary Figure S3](../figures/CAPTIONS.md#supplementary-figure-s3--v6-descriptivearchival-source-history-census)
shows descriptive case counts where all individual-fact answers are correct across starting
states but some joint answers still vary with the overwritten history.

The comparison with the model answering genuinely rewritten text is also
imperfect: those answers can be wrong, and their accuracy is not a hard upper
limit. Some single-edit checks are relative to that comparison; passing them
does not mean every question in a scene is answered correctly.

Giving the reader extra instructions changes how the state is queried; it is
not the same as improving answers under the original wording. Descriptive
intervals added after seeing the outcomes do not change the original primary
tests or their thresholds.

[Supplementary Figure S2](../figures/CAPTIONS.md#supplementary-figure-s2--changing-how-the-same-edited-state-is-queried)
compares the fixed readers on the same saved edited states. It keeps a change in
question wording separate from a change in the editor itself.

### Later constructive editing results

The combined-update study (`V7`) improved usable updates without resolving
whether current facts determine later answers. The constrained update trained
on answer likelihood, the constrained consistency-trained update, and the
unrestricted consistency-trained update met the retained-operation and joint-update point
criteria in both backbones and both seeds. Those criteria protect single edits,
repetition, restoration and changed-answer accuracy/preservation after paired
or triple updates. They do not require every joint answer to be correct.

For the simpler constrained likelihood-trained update, the triple-update results were:

| Model | Seed | Changed-answer accuracy | New errors on unchanged relations |
|---|---:|---:|---:|
| Gemma | 0 | 96.88% | 2.21% |
| Gemma | 1 | 91.28% | 4.82% |
| Qwen | 0 | 99.09% | 1.30% |
| Qwen | 1 | 95.57% | 2.80% |

Each program panel used 32 scenes per backbone, overlapping the single-edit
population. These are scene-averaged point estimates, not guarantees. No
candidate passed all source-history independence requirements together. The objective-specific
consistency advantage was established only for Qwen's unrestricted construction;
it was not a general cure or evidence of constrained-editor superiority.
The [compact constructive summary](../reproducibility/state_sufficiency/v7/constructive_summary.json)
retains the source rates and all operating-point flags, including failures.

### Update procedures and downstream coherence

Learned updates improve downstream answers clearly in Qwen; Gemma's results
are mixed. The completed matched-history comparison uses 64 benchmark cases per
model. Each benchmark case has four histories ending at the same facts and 34 questions
per history. Complete-case success requires every answer to be correct.

The factual-update baseline combines an attempted internal fact update with the
latest textual correction. A
constrained learned update (`INV_PAIR_NLL`) and a consistency-trained update
(`FREE_PAIR_CONSISTENCY`) are compared with that same baseline. Both original
seeds are averaged within each benchmark case before the paired bootstrap.

| Model | Update procedure | Change in complete-case success (percentage points) | 98.75% interval |
| --- | --- | ---: | ---: |
| Gemma | Constrained | +15.625 | [0.000, 30.469] |
| Gemma | Consistency-trained | +13.281 | [-0.781, 27.344] |
| Qwen | Constrained | +31.250 | [20.312, 43.750] |
| Qwen | Consistency-trained | +39.844 | [28.906, 50.781] |

Both Qwen intervals exclude zero. Gemma's constrained interval touches zero
and its consistency-trained interval crosses zero. Seed variation is retained
in [Figure 10](../figures/fig10_update_methods.png).

Complete-case performance remains weak: the baseline succeeds on 11/64 Gemma
benchmark cases and 1/64 Qwen benchmark cases, and neither learned method's seed-mean success rate
reaches half the benchmark cases in either model. The baseline used its recorded fallback
in 192/256 contexts for each model. These are effects relative to that executed
baseline, not proof that internal fact refresh succeeded in every context.

The comparison supports separating local update quality from downstream
coherence. It does not establish a universal consistency cure or priority for
the broad observation that local edits can fail downstream. The main test below
asks the narrower question of whether the old path still predicts an answer
after current facts are verified. [Saved-score reconstruction](../reproducibility/update_method_comparison/README.md)
recovers the complete tables and uncertainty without new inference.

## Part VI. Do current facts screen off source history?

### The criterion and its scope

Source-history sufficiency asks whether the declared current facts are enough
to account for a fixed downstream response. Distinct starting records receive
commands that lead to the same complete intended final records. The question
and response interface are held fixed. Once the current facts and the question
are known, knowing what used to be true should not change the expected answer.

Two references check whether the model can answer without a conflicting past.
The **current-facts-from-the-start reference** (`native-final`) presents the final
facts directly. The **already-correct update reference** (`no-op`) applies the same
update procedure to facts that already have their requested values.

Writing the old history as H, declared current records as S, question as Q and
response as Y, the criterion is:

$$
P(Y \mid H,S,Q)=P(Y \mid S,Q).
$$

This is not ordinary consequence accuracy. Identical wrong answers can satisfy
history invariance while failing the task. The controlled witnesses instead
require correct direct reads and competent references, then ask whether the
same joint question receives different answers across histories. Equality of
intended records does not assert equality of hidden states. Correct updated
facts combined with a history-sensitive reader remain compatible with failure.

### Quantifying a history discrepancy

For a binary candidate-answer interface, let p_h be the normalized probability
of the same answer label after history h. The best common Bernoulli probability
has worst-history discrepancy

$$
t_{\mathrm{hist}}=\frac{\max_h p_h-\min_h p_h}{2}.
$$

The midpoint of the extrema attains the bound. Thus a discrepancy of 0.10
requires a probability spread of at least 0.20; a hard-answer flip is a separate
condition. This is standard minimax/conditional-sufficiency mathematics, not a
new probability theorem. The previously cited audited finite countermodel is
not included in the available delivery, so that additional formal claim is not
retained here. Candidate-normalized scores are an observable response interface,
not calibrated beliefs. A selected large discrepancy is an illustration, not a
typical effect or an individual-answer error rate.

## Part VII. How robust is the finding?

### How often did the effect occur in the earlier cases?

<a id="v9-retrospective-census"></a>

The retrospective history-dependence census (`V9`) replaced a selected example with
population summaries. The strong view requires correct direct operands in every
compared starting history, correct native-final and same-method no-op references, and
direct correct-label probability at least 0.90. The following counts use
t_hist at least 0.10. Each condition has 1,088 fixed opportunities per model,
from 64 shared case designs and 17 questions under the first answer-code draw.

| Condition | Gemma qualifying / eligible questions | Gemma affected benchmark cases / all benchmark cases | Qwen qualifying / eligible questions | Qwen affected benchmark cases / all benchmark cases |
|---|---:|---:|---:|---:|
| Constrained likelihood-trained seed 0 | 15/989 | 12/64 | 18/896 | 13/64 |
| Constrained likelihood-trained seed 1 | 66/915 | 38/64 | 29/877 | 23/64 |
| Unrestricted consistency-trained seed 0 | 40/941 | 28/64 | 19/892 | 18/64 |
| Unrestricted consistency-trained seed 1 | 37/962 | 28/64 | 32/893 | 20/64 |
| Existing textual correction | 50/956 | 35/64 | 63/850 | 44/64 |
| Latest-value wording | 54/961 | 35/64 | 59/861 | 42/64 |

An affected benchmark case contains at least one qualifying question. Case prevalence is
not a per-question failure rate, and the shared designs are not independent
samples across backbones. This census is retrospective, not prospective
confirmation. It preserves the original early process-wide question-loading deviation
and failed new-rule reader; the saved outputs do not repair that protocol history.
The [source summary](../reproducibility/state_sufficiency/v9/primary_summary.json)
retains the full and eligible denominators and additional thresholds.

### The adverse coherence control

Ordinary-reader logical coherence and matched-history sufficiency have different
null models. The retrospective study's mean logical-coherence distances, without
filtering cases, were as follows in probability percentage points. The saved
analysis labels are `MINIMAL` and `LOCAL`.

| Context or update | Gemma | Qwen |
|---|---:|---:|
| Current facts supplied from the start | 2.5374 | 4.4566 |
| Already-correct plus constrained likelihood-trained seed 1 | 1.0311 | 0.9963 |
| Actual edits with the same constrained seed 1 | 2.1122 | 1.7068 |
| Existing textual correction, actual edits | 2.2094 | 3.7405 |

Every updating condition's unfiltered mean was below its native-final mean.
These point comparisons do not establish pairwise statistical significance.
Actual editing can increase discrepancy relative to the same-method no-op while
remaining below the native-final control. The evidence therefore does not support
an editing-specific origin of logical incoherence. This adverse result remains
unchanged by the prospective study. The [coherence summary](../reproducibility/state_sufficiency/v9/coherence_summary.json)
includes all updating conditions, not only the displayed rows.

### Does the same criterion find the effect in archived data?

<a id="v6-fixed-definition-archival-corroboration"></a>

The same half-range discrepancy measure was applied retrospectively to an earlier
shared-state editing study (`V6`) and its
recorded population. Its strong reference-qualified view requires correct
direct operands across all four histories with normalized correct-answer
probability at least 0.90, plus correct native and same-method no-op answers.
At t_hist at least 0.10:

| Method | Gemma affected benchmark cases / all benchmark cases | Qwen affected benchmark cases / all benchmark cases |
|---|---:|---:|
| Constrained seed 0 | 33/64 | 12/32 |
| Constrained seed 1 | 39/64 | 12/32 |
| Unrestricted seed 0 | 30/64 | 18/32 |
| Unrestricted seed 1 | 39/64 | 16/32 |
| Textual correction | 37/64 | 21/32 |

Both answer-code draws are included, giving 34 opportunities per benchmark case.
Textual correction has 87/1,863 eligible questions in Gemma and 41/748 in Qwen,
out of 2,176 and 1,088 full opportunities. Most eligible questions have small
discrepancies. The [compact table](../reproducibility/state_sufficiency/v6/fixed_certificate_summary.json)
retains question denominators for every listed condition. The retrospective census uses a single answer format; its
counts must not be directly compared with these combined-draw headlines.

Exact rendered source inputs do not overlap with the retrospective census, but model families, semantic
task family and software ancestry are shared. One Gemma unrestricted learned
checkpoint is byte-identical across the studies. This is archival cross-study
corroboration, not a fully independent replication or a prospective result.
The control that repeats the same directly written input has zero qualifying history discrepancies; it does
not establish invariance to distinct benign histories or repeatability of every
intervention. This replay did not recompute coherence.

### Does the effect recur in a study planned before its outcomes?

<a id="v10-prospective-confirmation"></a>

![Correct current facts can still leave downstream answers dependent on source history](../figures/fig07_source_history.png)

[Figure 7](../figures/CAPTIONS.md#figure-7--correct-current-facts-can-still-leave-downstream-answers-dependent-on-source-history)
now presents the matched-history design and the eight prospective primary rates
and adjusted intervals below. The earlier descriptive census (`V6`) remains in
Supplementary Figure S3; it is not the prospective aggregate evidence.

The prospective matched-history study (`V10`) tested fresh generated cases using
the same familiar question functions fixed before execution.
All eight starting assignments to three addressed records receive the same
commands, with complete intended final-table equality checked, including untouched
facts. A question combining facts qualifies only if its two individual-fact answers are
correct and valid in every history, its native-final and same-method already-correct
references are correct, and valid joint hard answers differ across histories.

Each benchmark case score is the witness count divided by 18 joint-question opportunities,
including ineligible questions in the denominator. The battery has 34 questions;
the joint subset comprises nine semantic questions under two answer-code draws.
Learned seeds are averaged within benchmark cases. Each backbone uses 64 shared designs
and is analyzed separately. Intervals use 10,000 case-bootstrap draws, fixed
seed 2609141002, and multiplicity-adjusted 99.375% coverage across eight cells.

| Model | Update group | Qualifying-question rate | 99.375% interval | Cases with a qualifying question |
|---|---|---:|---:|---:|
| Gemma | Constrained learned editor | 6.337% | [4.514, 8.290]% | 50/64 |
| Gemma | Unrestricted consistency-trained editor | 5.122% | [3.526, 6.771]% | 41/64 |
| Gemma | Existing textual correction | 5.990% | [3.472, 8.920]% | 30/64 |
| Gemma | Latest-value wording | 5.556% | [2.951, 8.594]% | 25/64 |
| Qwen | Constrained learned editor | 2.474% | [1.172, 4.123]% | 24/64 |
| Qwen | Unrestricted consistency-trained editor | 2.387% | [1.172, 3.993]% | 25/64 |
| Qwen | Existing textual correction | 6.163% | [3.841, 8.594]% | 38/64 |
| Qwen | Latest-value wording | 6.337% | [4.080, 8.854]% | 40/64 |

All corrected intervals are above zero. The rate measures fixed opportunities
for a controlled cross-history witness, not an individual-answer error rate or
a filter-conditional rate. Learned-group benchmark case prevalence is the union over both
seeds, so it has more witness opportunities than a single textual condition.
It must not be compared as an equal-budget search. The bootstrap supports
recurrence within the generated-case sampling scheme, not universal failure.
The [primary table](../reproducibility/state_sufficiency/v10/primary_results.json)
and [per-case counts](../reproducibility/state_sufficiency/v10/root_counts.json)
support aggregate reconstruction of these rates and intervals.

### Strong margins and repeatability

The predeclared strong diagnostic requires every operand correct-answer
probability to be at least 0.95, relevant candidate mass at least 0.90, and joint
correct-answer probability at least 0.95 in one history and at most 0.05 in another.

| Model | Update group | Strong qualifying questions / full opportunities | Cases with a strong qualifying question |
|---|---|---:|---:|
| Gemma | Constrained learned editor | 49/2304 | 21/64 |
| Gemma | Unrestricted consistency-trained editor | 24/2304 | 16/64 |
| Gemma | Existing textual correction | 20/1152 | 14/64 |
| Gemma | Latest-value wording | 22/1152 | 15/64 |
| Qwen | Constrained learned editor | 32/2304 | 17/64 |
| Qwen | Unrestricted consistency-trained editor | 7/2304 | 4/64 |
| Qwen | Existing textual correction | 36/1152 | 23/64 |
| Qwen | Latest-value wording | 33/1152 | 21/64 |

These counts sum learned-seed opportunities; they are not seed-averaged primary
rates. Strong witnesses occur in every individual learned seed and textual
condition. The all-measured-direct sensitivity also retains witnesses in every
condition, but does not certify the complete hidden state.

The fixed repeatability panel covered native-final and four learned conditions,
all eight starting histories, eight fixed questions and eight preselected benchmark cases per model.
The recorded score differences were zero across the core and both fresh passes
on this panel. Textual-correction procedures were not independently repeated.
The retained review checked score/journal bindings and recorded cache identities;
these compact summaries do not regenerate the original cache tensors.

### A repeat-checked illustration and provenance limits

In Qwen's unrestricted consistency seed 0, benchmark case `SSC1-FINAL-0006`, question
`d0-02` asks whether Wren and Orla take the same side on Garden. The final records
say Wren supports Garden and Orla opposes it. The correct answer is No, and both
direct facts are read correctly in all eight histories, with minimum normalized
correct-label probability 0.9994579196. Native and same-editor no-op references
also answer correctly. Joint answers are No, No, No, No, Yes, Yes, No, Yes.
Their half-range discrepancy is 0.4999465918, not a task-error rate.

This exact joint question belongs to the repeat panel and matches both fresh
passes. The operand measurements come from the core; the repeats do not cover
every operand/query. This is a selected tail example, not a prevalence estimate.
The [saved example](../reproducibility/state_sufficiency/v10/selected_repeat_checked_example.json)
contains its probabilities and flags.

The supplied provenance supports a prospective primary analysis, not independent
clock-time attestation or a public preregistration registry. Pre-final amendments
added native answer mass to the strong diagnostic, strengthened replay requirements
and corrected a synthetic figure without changing the primary estimand. Both
backbones used single-question chunking fixed before final execution after a
Gemma batch-versus-single discrepancy. The prospective study prevents questions
from being opened before updates are complete. That fixes the earlier census
procedure for this study; it does not repair the earlier study retrospectively.

The update procedures recompute the full stored context. This study does not establish cheap in-place cache
editing, a speed advantage, or transfer to new question functions introduced
after writing. Repetition, restoration and composition remain earlier-study
results. The compact aggregate replay reconstructs summaries from saved benchmark case
counts; it cannot independently recheck witness eligibility against omitted
raw journals or reproduce the original forward passes. The separate
[collaborator verification package](../reproducibility/state_sufficiency/independent_verification/README.md)
now supplies scientific per-example projections of the saved journals and
independently reconstructs eligibility, witnesses and intervals from those
scores. It does not reproduce fresh neural outputs or attest the original run.

### Changing which history the answer follows

Swapping part of Gemma's stored internal computation changed which history
controlled a later answer, while direct fact answers stayed correct. Earlier
experiments showed that answers could depend on old history despite correct
current-fact readout. This experiment adds causal evidence by changing the stored
activations directly and measuring how the answer changes.

We used six selected examples from the prospective study where different histories led to different
answers despite the same intended current facts. Before asking the question,
we combined stored activations from the earlier layers of one history with those
from the later layers of the other. The split was after the first 16 layers;
the remaining 26 layers came from the other history. We tested both directions
of the swap and asked the same question. Model weights stayed fixed.

| Answer format | Answer followed the later layers' history | Mixed result | Original answers too similar for a clear comparison |
|---|---:|---:|---:|
| Main | 5/6 | 1/6 | 0/6 |
| Alternate | 1/6 | 2/6 | 3/6 |

In the clearest example, swapping the later layers' stored activations switched
the answer toward the history that supplied them. The original histories assigned
normalized probabilities of about 0.966 and 0.017 to Yes. Combining the earlier
layers of the first history with the later layers of the second gave about 0.025.
The reverse combination gave about 0.977. The direct facts remained correctly
reported in both combinations.

Across the six positive examples under the main answer format, the lowest
normalized probability assigned to a correct direct answer was about 0.994.
The largest change was about 0.0014, relative to the
original history supplying the later layers. These bounds are specific to the
main format. Under the alternate format, the corresponding values were about
0.9881 and 0.00657. Direct answers stayed correct under both formats.

Checks of the original states reproduced the saved answers within the fixed
tolerances. Recombining a history with itself and copying a complete stored state
left scores unchanged in all 288 checks. The six matched comparison examples
had original answers too similar for a clear comparison under the main format.
Under the alternate format, five had that same limitation and one was mixed.
These comparison cases therefore do not establish that stored history is irrelevant.

The later stored computation can carry enough information to sustain a
history-sensitive answer under this intervention. Because the swap covers many
layers at once, it leaves open where that information first arose and which
smaller components matter. This is coarse causal evidence, not identification
of a unique mechanism or an old-fact circuit.

The six examples were selected for an existing history effect, so their counts
are not prevalence estimates. The alternate-format results also limit claims
of robustness for this panel. The separate one-case extension below combines
the strong behavioral controls and the causal result on the same example.
The separate Qwen/Ripple study remains an in-context update result.

The [crossover figure](../figures/fig08_cache_crossover.png) shows the clearest example.
The [saved-score replay](../reproducibility/cache_crossover/README.md) supplies the
six-case table, controls, unrounded values and archive provenance.

### One-case extension with matched histories

One retrospectively selected Gemma history pair now combines strong behavioral
controls with the causal crossover result. Under both tested answer formats,
changing which history supplied the later stored activations changed the
downstream answer, while separately measured direct facts remained correct.

Both histories required 2 fact changes and had 517-token prefixes. They reached
the same intended current records: Ada and Gita both oppose Theater. Asked
whether at least one supports Theater, the original states strongly disagreed
under both formats. The correct answer is No. The experiment used the same split
as the earlier crossover: the first 16 layers from one history and the remaining
26 from the other, combined before the question. This is learned activation
updating with fixed model weights, not parameter editing.

| Stored activations | Literal Yes/No format | Coded answer format |
|---|---:|---:|
| Original history 1 | 0.9976 | 0.9876 |
| Earlier-layer activations from history 1 + later-layer activations from history 2 | 0.0336 | 0.0235 |
| Earlier-layer activations from history 2 + later-layer activations from history 1 | 0.9979 | 0.9887 |
| Original history 2 | 0.0450 | 0.0289 |

The table shows normalized probability of semantic Yes. Both formats met the
criterion for the answer to follow the later layers' history. All 8 strict
direct-readout checks passed. Across those checks, the minimum probability of
the correct answer was about 0.99923 and the largest change from the original
state supplying the later layers was about 0.0000861. Saved-score replay matched
all 108 baseline responses, and all 24 same-history recombination and full-copy
controls were exact.

This closes the earlier gap between the strongest behavioral controls and the
causal result. The two formats test the same example, not independent cases.
The example was selected retrospectively and is kept separate from the six-case
panel and all prevalence denominators. The intervention swaps a large block of
stored computation. Earlier computation may have contributed to those later
activations, so the result leaves a unique circuit or stored obsolete fact
unidentified. Full mechanism identification remains unresolved; the mechanism
branch is closed for the current paper.

The [selected-case figure](../figures/fig09_cache_crossover_extension.png) shows
both formats. [Extension evidence and replay](../reproducibility/cache_crossover/README.md#one-case-extension)
provide the source records, strict checks and provenance. The
[original-panel appendix](CACHE_CROSSOVER_APPENDIX.md) retains every earlier
classification without pooling the extension into that panel.

### Public-data in-context extension

The [saved-response replay](../reproducibility/in_context_updates/README.md)
now reconstructs the following results from the frozen inputs and original
journals. It compares all phases against the supplied analysis, without inference.

Qwen sometimes used an obsolete value to answer a question even after it
correctly reported the current facts. Different histories began with different
obsolete records and then received the same authoritative current records.
In the qualifying cases, Qwen also answered the question combining those facts
correctly when given only the current records. The history changed the later
answer despite these controls. This extends the internal source-history finding
to public data.

The study used `Qwen2.5-7B-Instruct` and public RippleEdits-derived compositional
questions. Updates were supplied in the prompt while model weights stayed fixed.
The obsolete record remained explicitly present earlier in the chronological prompt.
The test asks whether Qwen uses the later authoritative records when answering.

The final set contained 34 benchmark cases and 98 questions. Both controls were satisfied
for 65/98 questions: the relevant direct current facts were reported correctly
under both histories, and clean references containing only the current records
answered the composed question correctly. A qualifying history-dependent question also required the
downstream answers to differ across histories.

The primary label-first parser found 3/98 qualifying questions across 3 benchmark cases. In every qualifying question,
the incorrect downstream answer was exactly the superseded value from that
history. The denominator is the full set of questions, including those that
failed the controls. Under the original union-parser sensitivity, 2/98
qualifying questions remained. This sensitivity check applies a different parser to the
same recorded answers.

The label-first rule gives a unique source label priority over aliases naming other entities.
The union rule pools labels and aliases and abstains when they identify more
than one entity. Both use full-string, type-restricted matching, never substring
rescue or a gold-answer tie-break. The
[parser explanation](../reproducibility/in_context_updates/README.md#what-the-parsers-mean)
details the unchanged strict correctness rule and pre-execution amendment.

For example, `popular:622` assigns Wyatt Russell's mother to Goldie Hawn and
Goldie Hawn's birthplace to Clinch County. These are counterfactual benchmark
assignments, not real-world biographical claims. History B previously assigned
Sinking Spring Farm as the birthplace; history D previously assigned Kimberley.
Both histories then received the same authoritative current records.

| Question | After history B | After history D |
|---|---|---|
| Who is Wyatt Russell's mother? | Goldie Hawn | Goldie Hawn |
| Where was Goldie Hawn born? | Clinch County | Clinch County |
| Where was Wyatt Russell's mother born? | Sinking Spring Farm | Clinch County |

The current-record-only references answered the composed question with Clinch
County. Thus the model could read both required facts and combine them correctly
without the obsolete history, but after history B it returned that history's
superseded value. This example illustrates the result; it is not a prevalence
estimate.

The two preselected fresh-process repeats reproduced exactly. Neither repeated
benchmark case contained a qualifying history-dependent question.

The history context also hurt answers to questions meant to remain unaffected.
Without chronological context, 170/272 answers were correct, versus 94/272 with
the prompt containing obsolete history B. These were 8 questions about unrelated facts, repeated across 34
contexts, not 272 independent facts. The drop shows broader context interference
and limits claims of selective preservation.

These qualifying questions show behavioral source-history dependence under in-context
updating. The neural mechanism and frequency in deployment remain unresolved.

## Part VIII. What can we conclude?

Prior work already shows that a successful local update need not propagate to
later answers. The central test here instead holds verified current facts and
the later question fixed across different histories. The completed supporting
[update-method comparison](../reproducibility/update_method_comparison/README.md)
shows clear downstream gains in Qwen and mixed results in Gemma. It motivates
evaluating downstream coherence separately, without replacing that central test.

The support/opposition relation is readable, transferable and causally useful.
Its magnitude did not reliably measure commitment, and moving the direct answer
did not necessarily reproduce the consequences of changing the relation.
Question-independent editors then achieved useful updates, repetition, restoration
and bounded composition. Those constructive results remain intact.

The current central result is a controlled separation between correctly readable
current facts and downstream behavior determined only by those facts. The
retrospective census (`V9`), fixed-criterion archival check (`V6`), and prospective
confirmation (`V10`) support this across Gemma and Qwen under learned and textual
updates. Declared current records can be insufficient to account for the measured
downstream response. Readable state and operative state name this explanatory
distinction, not two physically separate objects.

The causal swaps strengthen this behavioral result: changing the later stored
activations changed which history the answer followed while direct facts stayed
correct. The matched one-case extension supports this under both tested formats,
without turning a selected example into a prevalence estimate.

This does not establish absent internal facts, physical erasure of history,
a unique neural mechanism, calibrated beliefs, deployment failure rates, universal
model-editing failure, first priority, or new probability mathematics. Correct
facts plus a history-sensitive reader remain compatible with the observations.
The ordinary-reader coherence adverse control also remains visible.

The completed Qwen study supplies public-data in-context evidence for the
distinction between correctly reported facts and history-dependent answers.
Conventional parameter-editor validation
is pending. The early AlphaEdit route stopped before pretrained inference and
produced no scientific outcome.
The later MEMIT/GPT-J MQuAKE Phase 1 did run pretrained inference, but failed
prospective reference qualification before FINAL. The run stopped at qualification,
so it provides no final evidence about source-history dependence. Broader external
generality remains open. The closest-prior-work comparison is still being finalized.

[The retained qualification records](../reproducibility/memit_qualification/README.md)
document that executed failure separately from the earlier unexecuted route.
