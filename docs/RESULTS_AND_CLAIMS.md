# Results and Claims

The project began with a fairly simple question: does a model represent whether
a reason supports or opposes an action in its particular situation? It then
became a question about what we have found when that representation lets us
change the model's answer.

This is the current results page. The point of the tables is to show what each
experiment measured, not to make the argument depend on following every detail
of the analysis. Exact methods, saved results and reproduction limits are in
[Reproducibility](../REPRODUCIBILITY.md).

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

![Direct-answer margins and broader counterfactual recovery](../figures/fig6_answer_consequences.png)

The [matched consequence figure](../figures/CAPTIONS.md#figure-6--direct-answers-and-broader-counterfactual-consequences)
separates direct-margin attainment from recovery of the held-out natural pattern.

## Part V. Editing a shared state

### 7. Can we change a relation before knowing which question will be asked?

Yes in bounded tests, but reliably combining changes remains unresolved.
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
of single edits then improved some unseen sequences of changes. Reliable joint
control and full replication across models were still not established.

[Supplementary Figure S1](../figures/CAPTIONS.md#supplementary-figure-s1--broader-single-edit-training-and-later-sequences)
shows the matched comparison between broader and narrower single-edit training,
including both seed effects and the uncertainty in each model/editor comparison.

One prepared study defined fixed ways of asking the readout questions but
produced no new efficacy measurements. Preparation alone is neither a positive
nor a negative result. The next experiment asked a harder question. If we reach
the same final relations from different starting relations, do the answers
still depend on what was overwritten?

The table below uses the ordinary question wording and the constrained editor,
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
across starting states was 29.6875%. No tested setting met all the requirements
for independence from the starting relations. Consistent wrong answers do not
count as success.

![Joint-answer disagreement despite correct atomic relations](../figures/fig7_source_history.png)

The [source-history figure](../figures/CAPTIONS.md#figure-7--joint-answers-retain-overwritten-source-history)
shows descriptive root counts where all atomic answers are correct across starting
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

## Part VI. What can we conclude?

### 8. What does the evidence establish, and what is still missing?

The support/opposition direction is readable, transferable and causally useful.
But its magnitude is not a reliable measure of commitment, and moving it does
not necessarily recreate the consequences of changing the relation itself.
The broader activation contains information that the single score leaves out.

Learned editors make stronger changes before seeing the question and can repeat
and restore them in bounded tests. That progress does not establish that they
recreate the same internal state as rewritten text, erase the starting history,
or reliably combine arbitrary updates. It also does not establish a uniquely
located semantic mechanism or reliable behavior under other ways of asking.
An unfinished successor is not a completed result.

A semantic intervention earns a stronger interpretation when its effects propagate
through the later consequences of the edited fact while preserving unrelated
information. The experiments here measure that progression from readable relation
signals to reusable state updates, and identify the points where the stronger
interpretation still fails.
