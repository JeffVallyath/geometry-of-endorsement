# Geometry of Endorsement

A language model can give a reasonable moral answer once and still be unstable. It might understand the situation correctly, or it might just be reacting to familiar wording. Even if the first answer looks fine, we do not know whether that same judgment would survive another way of writing the same case.

The project boils down to two fundamental research questions.

1. Does the model internally track whether a particular moral reason supports or opposes an action in context?

2. If it does, can that internal state tell us which judgments are likely to become unstable when the same case is rewritten without changing what happened?

If you want the fuller version, start with the [project plan](docs/PROJECT_PLAN.md). It explains what each stage is testing and how the analysis fits together. Then read [results so far](docs/RESULTS_SO_FAR.md) for the actual numbers, completed checks, and what is still pending.

## The basic setup

The main dataset is ValuePrism. Each example gives us a situation, an action, and a moral reason such as autonomy, fairness, privacy, or harm. The dataset also records whether that reason Supports or Opposes the action in that situation.

The useful part is that the same reason can point in different directions depending on the case.

For example, autonomy might support respecting a patient’s refusal of treatment. In another situation, an appeal to personal freedom might allow a toddler to run down the street, endangering themselves.

This provides us with a way to test whether the model is actually using the situation rather than following simple moral intuition from the words themselves.

We arrange examples into four part comparisons where two reasons swap direction across two situations, a setup which we can call a checkerboard.

| Situation | Reason A | Reason B |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

This setup is useful because a fixed preference for one situation cancels out. A fixed preference for one reason also cancels out. So, if the model’s internal score still separates Supports from Opposes, it has to be responding to how the reason applies in that particular situation.

We also tried to rule out a few boring explanations before taking the result with anything more than a grain of salt. We keep familiar reason wording and repeated situations out of the test set so the model cannot just lean on examples it has effectively seen before. We compare the internal signal with a text-only model to see whether the wording itself already gives the answer away. We swap the answer labels to make sure the model is not just favoring one output token. Finally, we repeat the whole thing in Gemma so the result is not resting on one model, in this case Llama.

## What we have so far

Most of the recent work has gone into the first question.

The data split is built, and we confirmed that familiar reason wording really can make the task easier, so blocking that shortcut matters.

We also ran a factual sanity check first. Before trusting the activation method on something as potentially messy as moral reasoning, we wanted to see whether it could recover a much more obvious distinction where we already knew what the right answer was. So, we tested it on simple true and false statements. The method worked there, which at least gave us some confidence that the machinery itself was functioning before we tried to interpret the moral-reason signal.

After that, we tested Llama 3.1 8B Instruct.

The model showed a readable internal difference between cases where a reason Supports the action and cases where it Opposes the action.

We then repeated the experiment in Gemma 2 9B it. Gemma used its own prompt, its own internal states, and its own fitted score, and we still found the same general pattern.

So, from here, the first question currently has positive development evidence in two models.

The first question is not finished. The result we have so far comes from development data, which means it is evidence we found while building and checking the experiment rather than the final test of the claim. We also still need people to review the source examples and confirm that the Supports or Opposes labels are actually reasonable readings of the situations and not just LLM slop. Once that review is finished, we can define the clean set of examples and run a final test whose model results we have not looked at yet. If the same pattern survives then I would say for the most part, that would close out the work needed for the first question.

## Where the second question stands

The second experiment is already prepared.

We selected a little over one hundred original cases and generated several alternative wordings for each one. The point is to keep the case the same while changing how it is expressed.

Before we run the main analysis, there are two things that need to be checked.

First, is the original ValuePrism label actually a clear reading of the situation?

And second, did the rewrite really preserve the case?

Two reviewers will judge these independently before we look at the protected model outcomes.

## What we are doing next

Right now, there are four things that I would say are blocking the main experiment.

1. Finish the review of the original ValuePrism examples.

2. Finish the review of the proposed rewrites.

3. Combine those two reviews into one final approved set.

4. We also need to verify that Llama and Gemma can reliably perform the Supports versus Opposes task itself. Otherwise, it becomes much harder to interpret what their internal states mean.
