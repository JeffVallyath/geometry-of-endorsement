# Results So Far

## Current status

| **Result** | **Status** |
| --- | --- |
| ValuePrism data preparation | Complete |
| Split blocking familiar reason wording and repeated situations | Complete |
| Factual sanity check | Passed in Llama and Gemma |
| Llama Supports/Opposes activation result | Positive development result |
| Gemma replication | Positive development result |
| Human-reviewed source results | Not available yet |
| Rephrasing instability result | Not run yet |
| Causal effect on the model’s judgment | Not tested yet |

## ValuePrism data

The original ValuePrism data contains 218,406 rows. After removing the third Either label, 183,023 rows remain where a moral reason is labeled either Supports or Opposes.

Among those rows, 3,437 exact reason labels receive both answers in different situations. In other words, the same named reason can support an action in one situation and oppose an action in another.

Those reversals can be combined into 13,923 possible checkerboard comparisons across 6,073 distinct pairs of reasons.

After ranking the available examples under the current data rules, we retained 1,090 candidate comparisons containing 2,180 unique situations.

These numbers mainly tell us that there is enough natural structure in ValuePrism to run the experiment, ideally without having to create the Supports/Opposes reversals ourselves.

## Blocking familiar wording

We wanted to make sure the model could not succeed simply because it had seen the same reason wording or the same situation elsewhere in the data.

After removing 421 exact duplicate rows, the main split contains 116,000 training rows and 7,394 test rows. Another 59,208 rows are kept out of both sets because they don’t satisfy the separation rules.

The training and test sets share no situations and no exact reason labels. We also grouped closely related versions of the same reason so obvious wording variations could not cross the split.

Across 5 repeated tests, deliberately letting the text-only model see familiar reason wording during training increased its Supports/Opposes prediction accuracy by an average of 7.29%.

The 95% CI was 4.90 to 9.69 points. This showed that familiar reason wording really can make the task easier, which is why we keep it out of the test set.

Doing the same thing with familiar situations improved the model by only 0.53% on average, with an 95% CI from negative 0.81% to 1.86%.

Essentially, this pretty much shows that the moral intuition off of just a word really is a shortcut in this dataset. Keeping it out of the test set matters.

We also tested increasingly aggressive automatic filtering. The main test set contains 7,394 rows and 2,009 within-situation comparisons. A slightly stricter version keeps 7,081 rows and 1,865 comparisons. The next automatic filter decimated the data to only 587 rows and 23 comparisons, which is unusable.

## Factual sanity check

**For each example, the Supports versus Opposes score is put on a common normalized scale. We then combine the four scores in a checkerboard so that any fixed preference for a situation or moral reason cancels to zero. The resulting checkerboard score therefore measures how strongly the model’s score changes with the particular situation-and-reason pairing. Zero means no such interaction under this measure, while larger positive values indicate a stronger context-sensitive pattern.**

Before interpreting moral activations, we tested the same general activation method on simple factual statements where the correct answer was already known.

The final factual setup used arbitrary answer symbols such as A and B instead of the words True and False. We then reversed which symbol meant True or False across prompt versions, so the model had to follow the instructed mapping rather than simply rely on the meaning of the answer word itself.

Llama selected layer 14. Along the learned True-versus-False activation direction, the two classes were separated by 1.9245 standardized units, with a 95% CI from 1.8930 to 1.9556. None of 1,000 randomized labelings produced a separation this large. On a held-out prompt version using different answer symbols, Llama still reached 1.8402.

Gemma independently selected layer 25 and reached 1.9444, with a 95% CI from 1.9207 to 1.9666.

This tells us that the activation method can recover a known semantic distinction in both models before we use it on the harder Supports/Opposes task.

## Llama development result

The Llama experiment used 1,500 rows to fit the activation measurements, 300 rows across 75 checkerboard comparisons to choose the layer, and 500 rows across 125 comparisons for the final development evaluation.

Layer 19 was selected.

| **Measurement** | **AUROC** | **Four-part score** |
| --- | --- | --- |
| Model’s own answer strength | 0.721 | 1.6089 |
| Difference in means | 0.732 | 1.6470 |
| Logistic probe | 0.780 | 2.0836 |
| text-only comparison | n/a | 0.2842 |

The difference-in-means score beat the text-only comparison by 1.3628, with a 95% CI from 1.0885 to 1.6370.

The logistic probe beat the same comparison by 1.7994, with a 95% CI from 1.4970 to 2.1017.

The difference-in-means result also transferred to the held-out answer format, where the four-part score reached 2.1569.

We also tested simpler explanations for the result. A score based only on the situation, only on the moral reason, or just the sum of those two independent scores all collapse to zero in the checkerboard. That means the positive activation result cannot be explained by a fixed preference for the situation or the moral reason alone.

The current takeaway is that Llama contains a readable Supports/Opposes signal that depends on how the reason and situation are paired, rather than only on either one by itself.

## Gemma replication

Gemma repeated the experiment using its own prompt, activations, selected layer, and fitted measurements.

Layer 27 was selected.

| **Measurement** | **Within-situation accuracy** | **Within-reason accuracy** | **Four-part score** |
| --- | --- | --- | --- |
| Model’s own answer strength | 0.8920 | 0.8597 | 2.4068 |
| Difference in means | 0.8720 | 0.8776 | 2.3221 |
| Logistic probe | 0.8680 | 0.8425 | 2.1499 |
| Frozen text-only comparison | 0.5560 | 0.5885 | 0.2842 |

The difference-in-means score beat the text-only comparison by 2.0378, with a 95% CI from 1.6608 to 2.4149.

The held-out answer-format score reached 2.2617.

The simple situation-only and reason-only controls again produced zero.

So, the same general Supports/Opposes pattern appears in both Llama and Gemma rather than simply depending on one model.

## Intervention access

The intervention code can successfully modify the selected activation in the real Llama and Gemma checkpoints without changing the model parameters or breaking the output probabilities. This then establishes that we can reach and modify the intended internal state.
