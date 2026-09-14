# Geometry of Endorsement

**Geometry of Endorsement studies what internal directions in language models tell us about how the model represents and uses semantic information. The project began with moral support and opposition, then followed that signal through transfer, causal intervention, richer internal state, and learned updates to shared context.**

For the short logical progression, see [Project Strategy](PROJECT_STRATEGY.md). The quantitative record is in [Results and Claims](docs/RESULTS_AND_CLAIMS.md). Human-review procedures are described in [Human Review](docs/HUMAN_REVIEW.md), and reproduction instructions are in [Reproducibility](REPRODUCIBILITY.md).

A language model can give a reasonable moral answer once and still be unstable. It may be tracking the situation in a useful way, or it may be leaning on familiar wording. Rephrasing the same case gives us one way to tell how much of the original judgment survives.

The project started from two questions:

1. Does the model internally track whether a particular moral reason supports or opposes an action in context?
2. If it does, can that internal state tell us which judgments are likely to change when the same case is rewritten without changing what happened?

The first question produced a clear signal. Following it further changed the project. The direction transferred outside morality, could influence the model’s answers, and captured only part of a richer relational state. That eventually led to a more demanding question: if we intervene on the model, do the later consequences behave as though the underlying relation itself had changed?

The main starting dataset is ValuePrism. Each example gives a situation, an action, and a consideration such as autonomy, fairness, privacy, or harm. The dataset records whether that consideration **Supports** or **Opposes** the action in that situation.

The useful feature is that the same named consideration can point in different directions across different cases.

For example, autonomy might support respecting a patient’s refusal of treatment. In another situation, an appeal to personal freedom might support letting a toddler run into a dangerous street. The surrounding situation determines how the consideration applies.

We organize examples into four-part comparisons where two considerations reverse direction across two situations. We call this a checkerboard.

| **Situation** | **Reason A** | **Reason B** |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

A fixed preference for one situation cancels across the checkerboard. A fixed preference for one consideration cancels too. A model score that still separates Supports from Opposes therefore has to respond to the relation between the consideration and the particular situation.

The evaluation also includes several controls. Familiar situation and consideration identities are held out from the strict test. Text-only comparisons check how much of the label is already available from wording. Answer labels are swapped to expose output-token preferences. A factual-truth control checks that the activation pipeline can recover a semantic distinction already known to be linearly represented. The main relation result was then reproduced in Gemma after first being developed in Llama.

## Where human review came into this

Checkerboards control several statistical shortcuts, but their semantic interpretation still depends on the examples making sense to a human reader.

A separate human-review process checked whether the support/opposition labels and rewritten cases preserved the intended meaning. The public repository includes the review procedure and calibration material. The purpose of this review is semantic quality control: reviewers judge whether the relation and rewrite are coherent, without deciding which moral position is universally correct.

## What changed after the first result

The original second hypothesis treated distance from the Supports/Opposes boundary as a possible measure of commitment. Cases near the boundary seemed like they might be easier to flip under meaning-preserving rewrites.

That prediction did not hold reliably. The internal score separated Supports from Opposes, while its magnitude added little beyond the original text and the model’s own answer confidence when predicting rephrasing flips.

This narrowed what the direction could mean. It tracks which side of the relation the model occupies much better than how firmly that judgment will survive a rewrite.

Around the same time, category and cross-dataset tests broadened the interpretation of the signal. Directions learned from ValuePrism transferred to ordinary argument support/attack datasets and back again. The signal also survived measured removal of answer-token, sentiment, and factual-truth components. The most useful interpretation became a general **counts-for / counts-against** relation embedded in a broader evaluative representation.

## From representation to control

Once the relation direction was readable, the next question was whether changing it would change behavior.

Steering the model along the direction moved Supports/Opposes answers in the expected direction. That established causal access to the answer-producing computation.

The behavior of other semantic directions complicated the picture. Truth and sentiment directions could often move the same Supports/Opposes endpoint, including under matched intervention sizes. Several distinct internal signals could therefore reach overlapping downstream behavior.

This motivated a broader measurement of what an intervention does across the model.

## What does a direction actually do?

A **causal fingerprint** records how a small intervention changes a fixed panel of downstream measurements.

The first fingerprint design failed its own validation because random directions could look reliable under the original diagnostic. The corrected analysis separated two questions: whether a local intervention effect can be predicted at one prompt, and whether the same semantic direction keeps a similar causal role across different prompts.

Local gradient predictions were highly accurate in both models. Cross-context stability differed sharply. Llama’s fingerprints were relatively stable, while Gemma’s were much less so. The planned cross-model claim about one stable compressed control geometry therefore remained unsupported.

That result changed the role of the one-dimensional direction in the project. It remained a useful readout and intervention axis, but the broader internal state became increasingly important.

## What exactly is the direction tracking?

Controlled examples with several people and targets let us ask whether the direction follows the specific relation being queried.

It does. When the question switches from one person to another, or from one target to another, the frozen direction follows the addressed relation. This gives a sharper interpretation than generic positive or negative tone.

The same experiments also show why a scalar relation score cannot describe the entire relational structure of the context.

## The direction, the broader state, and the explicit answer

On ordinary ValuePrism disagreements, the one-dimensional score usually follows the model’s own answer.

A controlled experiment initially appeared to show a deeper gap: the internal direction tracked the intended relation while zero-shot answers were poor. Fixed demonstrations of the expected answer format largely removed that discrepancy. The model could often report the relation once the task format was made clear.

The broader activation state still carried additional relation information. Readouts from the full activation improved prediction after accounting for the text and the model’s explicit answer. A frozen-plan replication reproduced this incremental information in both models on separate pre-existing populations.

The emerging picture was a relational state with several useful views: a scalar support/opposition coordinate, an explicit answer, and additional information distributed through the broader activation.

## Testing whether an intervention recreates the semantic change

This gave us a stronger way to evaluate steering.

Controlled context pairs differ in exactly one relation. For example, the original context might say that Alice supports a proposal and the changed context says that Alice opposes it. That change has consequences across several later questions. Questions about Alice’s support should flip, complementary questions should move consistently, paraphrases should agree, and unrelated relations should remain stable.

In Gemma, rank-one steering could reliably hit the requested direct-answer margin while failing this broader consequence test. These measured margin hits did not satisfy the full frozen strong-target requirements; the direction fitted to the matched setting also failed qualification. Moving the full activation toward the state produced by the genuinely changed context recovered far more of the expected pattern. Replacing the activation with the natural changed-context state recovered almost all of it.

The same direct answer can therefore be reached through internal changes with very different downstream consequences.

Llama could not support the same comparison because its natural changed-context reference was unstable across equivalent wordings. That reference failure limits the comparison to Gemma rather than counting as an editing failure.

## Editing a shared state before future questions

The next stage moved the intervention earlier.

Instead of editing a representation after the downstream question was already present, the editor changes an addressed relation in the shared context first. The edited context is then reused for questions that were unknown when the edit was made.

This turns the problem into a state-update task. A successful edit has to support fresh questions about the changed relation while preserving unrelated information.

The first shared-state editors produced strong direct changes, although their reliability under repeated and reversed use was limited. Later constructions improved single-edit performance and made repeated assignment and restoration much more stable.

## Repeated, restored, and combined updates

The later studies treat an update more like an assignment to a reusable state.

Repeating the same requested value should keep the state stable. Restoring an earlier value should recover the corresponding answers. Several updates should combine without damaging relations that were never touched.

On Gemma, learned overwrite operators achieved strong single edits together with near-idempotent repetition and high restoration rates. A less constrained overwrite method also worked, which showed that those capabilities did not belong uniquely to one parameterization.

Broader single-edit consequence supervision improved several unseen two- and three-update programs, especially in Qwen. Joint semantic consistency remained the hard part. Equality and other derived questions could fail even when the individual edited relations were answered correctly.

One planned readout study stopped after preparation and produced no efficacy result. Its successor evaluated the harder question directly.

## Does the final state forget where it started?

The last study included in this preview reaches the same requested final relations from several different starting relations.

If an update has fully established the new relational state, later answers should depend on that final state. The overwritten starting values should stop affecting the result.

The editors retained useful single-edit, repetition, and restoration behavior across Gemma and Qwen. Fixed attempts to clarify the downstream questions did not reliably improve the remaining errors.

The stronger source-independence test still failed. Across different starting states, most individual questions were often answered correctly, yet complete sets of joint consequences were much less reliable. Many disagreements remained even when the model answered the relevant individual relations correctly in every starting state.

A concrete example captures the issue. Suppose the final edited state says that two people both oppose a project. The model may correctly answer that each person opposes it in every version of the context. Its answer to whether the two people take the same side can still change depending on what their positions were before the edits.

This leaves a clear boundary around the current capability: repeated and reversible relational updates are possible in the tested setting, while some downstream consequences continue to carry information about overwritten history.

## What the project is arguing for now

The project began with a question about whether a support/opposition direction existed. That direction turned out to be readable, transferable, relation-specific, and causally useful.

Following its consequences exposed a richer problem. A model can reach the desired answer while the surrounding state behaves differently from a genuine change in the underlying relation. Learned shared-state editors recover substantially more reusable behavior, including repetition and restoration, while joint consequences still reveal residual dependence on the starting context.

The resulting evaluation principle is straightforward:

**A useful semantic update should support the consequences of the new fact across later questions, preserve unrelated information, survive repeated use, and behave consistently regardless of the value it replaced.**

That standard is stronger than checking whether one target answer moved. The experiments in this repository build toward that standard and show which parts are already achievable under controlled conditions, along with the parts that remain unresolved.
