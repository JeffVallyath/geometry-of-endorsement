# Same facts, different answer

Gemma-2-9B-it answers **No** under H5 and **Yes** under H6 after the same
activation intervention. The correct answer is No. All 160 stricter checks
pass across the eight histories, and both reference answers are No.

This is the measured example in Figure 1 and Table 2 of the
[final-paper copy](../../papers/final-paper.pdf). **H0-H7 are eight histories
within this one synthetic case, not the whole study.** The study also contains
other independently generated cases and a separate public benchmark.

## What starts differently

The record contains twelve fictional facts: four people's stances on three
proposals. Only these three Gallery facts vary. The
[complete twelve-fact table](facts.md) shows both starting records and all final facts.

| Person | H5 starts | H6 starts | Requested final stance |
|---|---|---|---|
| Bryn | In favor | Opposed | Opposed |
| Ari | Opposed | In favor | Opposed |
| Ada | Opposed | Opposed | In favor |

Every history receives the same three assignments, in this order:
Bryn on Gallery becomes opposed; Ari becomes opposed; Ada becomes in favor.
A request still runs when its fact is already correct.

Two facts change under H5, and two change under H6. Ari is not part of the
downstream question. His starting stance makes that changed-fact count equal.
Both contexts have 455 tokens before the question.

## What the model sees and what the hook receives

Read the complete [H5 model-visible input](H5.txt) and [H6 input](H6.txt).
They include the same two solved demonstrations, all record statements,
the question, allowed answer words and model-turn markers. These are frozen
prompt renderings from `PROMPTS.json.gz`, checked against the recorded full-prompt
hashes and token IDs. They are not a newly captured inference log.

The activation intervention adds no correction sentences to those prompts.
Its inputs are separate: [activation arguments and addressed tokens](activation-inputs.json).
For seed 0, the saved intervention edits the output of block 16, recorded as
zero-based site 15. It uses a rank-16 setter in an FP32 workspace and returns
the edited activations to the frozen backbone.

| Addressed sentence | H5 token positions | H6 token positions | Requested value |
|---|---|---|---|
| Bryn on Gallery | 331-340 | 331-339 | Opposed (0) |
| Ari on Gallery | 383-390 | 382-390 | Opposed (0) |
| Ada on Gallery | 411-418 | 411-418 | In favor (1) |

Positions are zero-based and inclusive. They are derived from the frozen
character spans with the pinned tokenizer, whose full token IDs match the
saved prompt IDs. The recorded hook reports the same addressed-token counts.
The JSON also retains the selected setter's hash and the exact character spans.

## The question and saved answers

> Do Ada and Bryn take the same side on Gallery?

The final facts put Ada in favor and Bryn opposed, so the answer is No.
The recorded normalized probability of No is 0.778670 under H5 and 0.017623
under H6. These probabilities are rescaled over the two allowed next-token
answers. This experiment scores tokens rather than generating an explanation.

| History | Facts changed | Tokens before question | Saved answer |
|---|---:|---:|---|
| H0 | 0 | 455 | No |
| H1 | 1 | 456 | No |
| H2 | 1 | 456 | Yes |
| H3 | 2 | 457 | No |
| H4 | 1 | 454 | Yes |
| H5 | 2 | 455 | No |
| H6 | 2 | 455 | Yes |
| H7 | 3 | 456 | No |

All eight full inputs are available as [H0](H0.txt), [H1](H1.txt), [H2](H2.txt),
[H3](H3.txt), [H4](H4.txt), [H5](H5.txt), [H6](H6.txt) and [H7](H7.txt).
The old facts imply No for H5 and Yes for H6. Those old answers are derived
from the records, not measured pre-update responses or answers fed to the model.

## What passes before a disagreement counts

The [full check table](checks.md) shows twelve direct checks and the other eight
combined checks for each history. That is 20 checks per history and 160 in all.
The direct checks cover six facts in both answer formats. The other combined
checks use the target question's answer format.

The clean reference starts with final facts and has no intervention. The no-op
reference is H0 with the same three interventions. Both answer No. Each question
gets its own clone of the prepared KV cache. The model weights are not reloaded
for each question, and no question receives another question's answer.

Three operations are used over three pairs of people, giving nine combined
questions. The table below is an explanation of those operations, not new data.

| First person | Second person | Same side? | Both in favor? | At least one in favor? |
|---|---|---|---|---|
| Opposed | Opposed | Yes | No | No |
| Opposed | In favor | No | No | Yes |
| In favor | Opposed | No | No | Yes |
| In favor | In favor | Yes | Yes | Yes |

Read [all exact questions and answer formats](questions.md). Each combined
question takes a turn as the target. Its other eight become checks, reusing
the saved responses. A failed check leaves the question in the denominator.
The replay includes a different target from this case whose stricter eligibility
fails, so a disagreement alone is visibly insufficient.

## Source and replay

Source case: `FHV1-FINAL-0017`; question: `FHV1-FINAL-0017-d0-08`;
model: `google/gemma-2-9b-it`; method: `INV_PAIR_NLL`; teaching seed: 0.
Both saved seeds remain in the compact score panel and are recounted.

[Source bindings](../records/BINDINGS.json), [starting and final facts](../records/synthetic/root.json),
[frozen plan](../records/synthetic/PLAN.json), [complete prompt records](../records/synthetic/PROMPTS.json.gz),
[saved scores](../records/synthetic/SCORES.json.gz), and
[existing scoring functions](../../history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/analyze.py)
make the example auditable. Run the [small CPU command](../README.md) to rebuild
the decisions, check tables and readable inputs.
