# Geometry of Endorsement

**Geometry of Endorsement studies when a model that correctly reports updated information actually uses that information as the state governing its later decisions. The project began with a linearly readable support/opposition relation and progressed through causal intervention and learned context editing. The central finding is that these notions can end up coming apart: different histories can be updated to the same intended current facts, the model can report those facts correctly, yet later decisions can still depend on information that was supposedly replaced.**

For the short logical progression, see [Project Strategy](PROJECT_STRATEGY.md). The quantitative record is in [Results and Claims](docs/RESULTS_AND_CLAIMS.md). Human-review procedures are described in [Human Review](docs/HUMAN_REVIEW.md), and reproduction instructions are in [Reproducibility](REPRODUCIBILITY.md).

The project began with moral and evaluative relations: whether considerations such as fairness, autonomy, privacy, or harm count for or against an action. This gave us a controlled setting for a broader question: when a model reports that one of these relations has changed, is that new relation actually what ends up governing its later judgments?

For example, if an update makes fairness count against an action and the model reports that correctly, later moral judgments should no longer depend on whether fairness used to count for it.

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

In Gemma, an edit calibrated on the direct question could move that answer while still failing to reproduce the broader consequences of actually changing the relation. We first thought the edit might be failing because it was too simple, but making it higher-dimensional did not fix the problem. What ended up helping much more was tailoring the edit to the specific question being asked, which suggested the harder problem was making one reusable change that still works across future questions.

This changed the interpretation of the earlier result. The harder problem was carrying one semantic change across future questions. Since the question-matched intervention already uses information about the question being asked, the next step was to construct an edit before those future questions were known.

Llama could not support the same comparison because its natural changed-context reference was unstable across equivalent wordings. That reference failure limits the comparison to Gemma rather than counting as an editing failure.

## Editing a shared state before future questions

The next stage moved the intervention earlier.

Instead of editing a representation after the downstream question was already present, the editor changes an addressed relation in the shared context first. The edited context is then reused for questions that were unknown when the edit was made.

This turns the problem into a state-update task. A successful edit has to support fresh questions about the changed relation while preserving unrelated information.

The first shared-state editors produced strong direct changes, although their reliability under repeated and reversed use was limited. Later constructions improved single-edit performance and made repeated assignment and restoration much more stable.

## Repeated, restored, and combined updates

The later studies treat an update more like an assignment to a reusable state.

Repeating the same requested value should keep the state stable. Restoring an earlier value should recover the corresponding answers. Several updates should combine without damaging relations that were never touched.

Later experiments made this result substantially stronger. Editors trained only on individual assignments learned to repeat, restore, and combine two or three updates across both Gemma and Qwen while still being able to mostly preserve relations that were never changed.

The strongest simple recipe reached about 91–97% accuracy in Gemma and 96–99% in Qwen on relations that should change after three updates, with roughly 1–5% new errors on relations that should stay unchanged.

However, some questions combining several relations could still change depending on the model’s earlier state, even when the individual edited relations were all reported correctly. That led to the next question: are the reported current facts actually enough to explain the model’s later decisions?

## Are the reported current facts sufficient for later decisions?

We started with several versions of the same case that differed in their original relations, edited each one so they all ended with the same new relations, and then asked the exact same follow-up question.
We call this requirement **source-history sufficiency**: once the current facts are the same, the answer should no longer depend on what was true before the update.

Formally, if the current semantic state \(S\) is sufficient for answering question \(Q\), then the old history \(H\) should provide no additional information about the answer \(Y\):

$$
P(Y \mid H, S, Q) = P(Y \mid S, Q)
$$

Our experiments test this by reaching the same \(S\) from different histories and asking the same \(Q\).
This was not limited to a few selected examples. The same pattern appeared in both Gemma and Qwen, with both learned edits and ordinary text-based updates: the model could correctly report the current relations while later answers still depended on what those relations had been before.

One example from my own experiments helps make this more concrete. Ravi and Kira both end up opposing Canal, and the model reports those individual relations correctly in every history. But when asked whether at least one of them supports Canal, the answers across four histories are No, No, No, Yes.

The result is a separation between what the model can correctly report as its current facts and the information that still affects its later decisions.

## What the project is arguing for now

The project began by asking whether a support/opposition relation could be identified inside a model. Following that relation through readout, intervention, and learned editing led to a broader question:

**When an AI says it has updated a fact, has the information governing its later decisions actually changed too?**

The experiments demonstrate that the answer can be no. We can start from different histories, update them to the same final facts, and verify that the model reports those facts correctly. Its later decisions can still depend on what used to be true.

This exposes a gap between readable state and operative state. The model may correctly tell us its current facts while older information continues to influence how those facts are used.

For moral reasoning, the consequence is straightforward. A model might correctly report that fairness now counts against an action while later moral judgments still depend on whether fairness used to count for it. More generally, this matters whenever we want to know whether an AI has actually incorporated an update into the state driving its behavior.

**A model can appear to have updated what it knows while still making decisions from information that should no longer matter.**
