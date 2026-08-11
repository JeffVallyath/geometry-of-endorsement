# Project Plan

This document starts with where the README leaves off. This document explains how all the pieces of the project fit together, so it’s a bit longer than the README.

## What we are actually measuring

There are a few different things in the project that need to stay separate.

ValuePrism gives us a Supports or Opposes label for a moral reason in a situation. Human reviewers tell us whether that label is actually a clear reading of the example. Separately, Llama or Gemma gives its own answer about that reason.

For the rephrasing experiment, we are measuring whether the model stays consistent with itself. If it says a reason Supports the action in the original case but says it Opposes the action after an approved rewrite, that counts as a flip, and vice versa.

## Finishing the first question

The first question already has positive development evidence in Llama and Gemma, as mentioned in the read me, but there are still a few things left before we can realistically treat it as complete.

First, people need to review the source examples. For each situation and moral reason, reviewers judge whether the stored Supports or Opposes label is a clear reading, an ambiguous one, or just simply invalid. They also check whether the repeated moral reason keeps the same ordinary meaning across the paired situations. The 125-checkerboard (as defined in the read me, the 4-part comparisons) review happens after we already know the overall development result, so this part is best treated as a way to understand and stress-test that result.

We will also run a direct behavioral report for both Llama and Gemma. Before interpreting the internal state, we should show that the models can actually perform the Supports versus Opposes task reliably, handle both answer mappings, give stable answers across repeats, and follow the expected format.

The human review also tells us how many candidate examples are likely to survive the quality check. We need that estimate before deciding the size of the final test. Once we know the usable rate, we can fix in advance how large an effect would still matter, how much uncertainty we are willing to accept, and how many examples we need to test for that effect fairly. Those choices should be locked before we look at the final Llama results, so the test is not shaped around whatever outcome happens to appear.

## Running the second question

The second question begins with the rewrite review.

Each proposed rewrite is compared with its original case. Reviewers decide whether the rewrite preserved everything that could reasonably affect whether the same moral reason supports or opposes the action. A rewrite enters the main experiment only when both reviewers independently say that the case was preserved.

After the source review and rewrite review are locked down, the two can then be joined using permanent item IDs. This gives us the final approved set.

The first rephrasing study contains 112 original items, but they serve two different purposes. 80 were sampled to reflect the broader development set, so they are the main group for estimating how often judgments actually change and whether the internal score predicts that instability. The other 32 were deliberately chosen because the model’s own confidence and the internal score disagree about which cases look fragile. Those cases let us directly compare the two warning signals, but because they were selected for that disagreement, they cannot be used to estimate how common instability is overall.

## What counts as one prediction example

The original item is the main unit of analysis.

Each original has several rewrites, but all of those rewrites share the same original wording, original internal state, original confidence, and original internal score.

For each original item, we will count how many rewrites passed human review and how many of those caused the model to switch between Supports and Opposes. That gives us the item’s observed flip rate.

All rewrites from one original item stay together whenever we split the data for fitting or evaluation.

## How we measure a real flip

The model will be tested using both answer-label arrangements so that a preference for one literal output token cannot masquerade as a moral change.

We can then convert both answer formats back into the same Supports or Opposes meaning and compare the original answer with the rewrite.

We will also repeat identical prompts. If the model changes its answer even when nothing in the wording changes, that gives us a potential baseline for ordinary run-to-run variability.

## The main prediction test

The first predictor gets the original wording and the model’s original confidence.

The second predictor gets those same inputs plus the original internal score.

The central question is whether adding the internal score improves prediction on original items that were kept away from fitting.

If it does not beat wording and confidence, then the internal signal may be readable without giving us any extra warning about fragility.

If it does help, we still need to ask whether the moral-reason score is special. We will compare it with unrelated summaries of the same internal state, including random directions and directions built after shuffling the Supports and Opposes labels.

If many unrelated measurements work just as well, the result is probably closer to a general difficulty signal. If the real relation score performs better, the interpretation becomes much more specific.

## How the possible outcomes change the project

If the internal signal weakens on human-clear examples, then the first result should be narrowed to tracking the dataset more generally.

If approved rewrites rarely cause changes beyond ordinary repeat noise, then the current rephrasing design does not give us a useful instability target.

If flips happen but wording and confidence already explain them, then the representation result can survive while the proposed warning-signal claim fails.

If the internal score adds predictive value on the representative items and beats unrelated internal measurements, then we have the strongest version of the second result and can justify scaling it into the final study.
