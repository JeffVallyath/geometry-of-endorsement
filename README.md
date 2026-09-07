**Geometry of Endorsement studies what internal directions in language models actually tell us about how the model represents and uses semantic information. We began with moral support and opposition, but the project has increasingly become about a broader question: when a direction lets us read and change a model’s behavior, what have we actually identified?**

For the full logical progression of my plan, read Project Strategy. For the empirical details, see [Results and Claims](docs/RESULTS_AND_CLAIMS.md). The reasoning and results from the human review are in Human Review, while Current Status tracks the experiments that are still on the table. Reproduction instructions live in Reproducibility.

A language model can give a reasonable moral answer once and still be unstable. It might understand the situation correctly, or it might just be reacting to familiar wording. Even if the first answer looks fine, we do not know whether that same judgment would survive another way of writing the same case.

The project had boiled down to two fundamental research questions.

1. Does the model internally track whether a particular moral reason supports or opposes an action in context?
2. If it does, can that internal state tell us which judgments are likely to become unstable when the same case is rewritten without changing what happened?

For the first question, the evidence towards this answer seems to be moving towards a yes. But following that representation further I thought of a different question: once we find a meaningful internal direction, what can we actually infer from this?

I found that the representation transfers beyond morality and can causally change model behavior. At the same time, its magnitude does not behave as a reliable measure of commitment, its causal effects overlap with other semantic directions, and its relationship to the model’s final answer is more complicated than we originally expected.

The main dataset is ValuePrism. Each example gives us a situation, an action, and a moral reason such as autonomy, fairness, privacy, or harm. The dataset also records whether that reason Supports or Opposes the action in that situation.

The useful part is that the same reason can point in different directions depending on the case.

For example, autonomy might support respecting a patient’s refusal of treatment. In another situation, an appeal to personal freedom might allow a toddler to run down the street, endangering themselves.

This provides us with a way to test whether the model is actually using the situation rather than following simple moral intuition from the words themselves.

We arrange examples into four-part comparisons where two reasons swap direction across two situations, a setup which we can call a checkerboard.

| **Situation** | **Reason A** | **Reason B** |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

This setup is useful because a fixed preference for one situation cancels out. A fixed preference for one reason also cancels out. So, if the model’s internal score still separates Supports from Opposes, it has to be responding to how the reason applies in that particular situation.

We also tried to rule out a few boring explanations before taking the result with anything more than a grain of salt. We keep familiar reason wording and repeated situations out of the test set so the model cannot just lean on examples it has effectively seen before. We compare the internal signal with a text-only model to see whether the wording itself already gives the answer away. We swap the answer labels to make sure the model is not just favoring one output token. Finally, we repeat the whole thing in Gemma so the result is not resting on one model, in this case Llama.

## Where human review came into this

The checkerboard demonstrates that the signal depends on the situation–reason pairing, but not that the pairing means what we claim. (ie does it make sense to a human reading the checkerboard/rewrites)

Because of this, Andrew and I independently reviewed the original examples along with the rewrites. We found some ambiguity, but a substantial clear subset remained, and the model’s Supports/Opposes signal became cleaner on examples we judged clearer.

## What changed after the first result

This brought us back to the second question we started with. Our second hypothesis was that distance from the Supports/Opposes boundary might measure how firmly the model held a judgment, making cases near the boundary easier to flip under meaning-preserving rewrites.

However, this did not work reliably. Rewrites sometimes changed the model’s judgment, but distance along the direction added little beyond the text and the model’s own confidence. So a direction can tell us which side the model is on without telling us how firmly it is there.

But from this, I figured out that representation turns out not to be specifically moral. Directions learned from ValuePrism transferred to ordinary argument support/attack tasks and back again. Even checking for potential confounds, supports/opposes information remained after removing answer-token, sentiment, and factual-truth components. So now, the better interpretation from this point on of what the model thinks is if something counts for or against something else rather than moral reasoning.

## Representation is not the same thing as control

At this point, we had fairly strong evidence that the Supports/Opposes direction represented something meaningful. The next question was whether changing it could actually change the model’s behavior.

It could. Steering the model’s activations along the direction moved its answers in the expected direction.

But truth and sentiment directions could often move the same Supports/Opposes behavior too. So, although the model appeared to distinguish these ideas internally, they did not necessarily have completely separate ways of influencing behavior.

This suggested what I call **“more concepts than control knobs”**: several internal distinctions may reuse some of the same downstream machinery. Andrew’s separate replication suggested that this sharing is not universal, however, as related families can converge on a common behavioral effect while unrelated ones remain distinct.

That raised a deeper problem: showing that a direction changes behavior does not necessarily tell us **what mechanism we have found**.

## What does a direction actually do?

To investigate this, we introduced a **causal fingerprint**: rather than only asking whether steering a direction makes “Supports” more likely, we look at the broader pattern of internal measurements and outputs that change with it.

In Llama, the same semantic directions tended to produce similar downstream effects across different contexts. Gemma behaved differently. At any individual prompt, we could predict the effect of an intervention extremely accurately, but the same direction was much less consistent across prompts.

So even the idea that a semantic direction has one fixed causal role turned out to be something that had to be tested rather than assumed.

## What exactly is the direction tracking?

We then went back to a more basic question: can the representation actually track **who supports what**?

Controlled examples with multiple people and different stances showed that it does. A direction learned from ValuePrism followed changes in the queried person and target rather than simply detecting whether the passage sounded supportive in an abstract sense.

This made the semantic interpretation sharper. But it also exposed another distinction.

## The direction, the broader state, and the answer are not the same thing

On ValuePrism examples where the model disagrees with the dataset relation, the one-dimensional direction usually follows the model’s own answer.

Controlled examples initially looked like the model had the relation right internally but somehow could not give the right answer: the frozen internal direction tracked the queried relation even while the model’s zero-shot answers were poor. But when we repeated the task using two fixed few-shot formats that made the required answer clearer, both models’ answers improved dramatically and closely matched the same frozen direction. So, the original gap was an **elicitation failure**. The prompting format was failing to reliably draw out behavior the model was capable of producing, rather than evidence that the model represented a relation it couldn’t report.

A separate analysis revealed a more interesting distinction. Instead of collapsing everything at that layer onto the single Supports/Opposes direction, we looked at the **broader activation state**, meaning the full set of values at that layer. This broader state still improved prediction of the ValuePrism relation after accounting for the text and the model’s own answer. We first found this retrospectively and then reproduced it under a plan frozen in advance on a separate pre-existing population in both models.

**This reveals that neither the one-dimensional direction nor the explicit answer completely describes the information present in the broader activation state.**

The direction is therefore better thought of as a useful coordinate through a richer representation than as the whole representation itself.

## What the project is arguing for now

We began by asking whether a meaningful semantic direction existed.

We now know that such a direction can transfer across domains, track the particular relation being queried, and causally change model behavior, while still failing to completely describe the model’s broader internal state or uniquely identify how that state produces behavior.

This leads to the central argument of the project:

**A direction that lets us read and control a behavior does not, by itself, tell us what semantic mechanism we have identified.**

Changing the target answer shows that the intervention has causal influence over the model’s behavior, while leaving open what exactly was changed internally: the full semantic representation, one useful coordinate within it, or a downstream pathway that other semantic interventions can also access.

From what I see here, the next step is therefore to test interventions by more than whether the target answer moves. We should ask whether steering produces the broader consequences we would expect if the underlying semantic fact had genuinely changed.
