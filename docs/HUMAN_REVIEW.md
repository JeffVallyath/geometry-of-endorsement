# Human review

This guide governs the independent human review for Claim 1 and Claim 2. The source population and candidate text were fixed before the reduced v2 instrument was prepared on August 10, 2026. No human study label, Claim 2 model outcome, susceptibility estimate, or sealed confirmatory result was observed.

Do not complete any v1 sheet. The v1 files remain unchanged as protocol history.

## Purpose

Claim 1 asks whether the model's internal state encodes the relation between a situation and a named consideration. Human review identifies source cells where the stored `Supports` or `Opposes` relation is clear enough to carry that interpretation.

Claim 2 asks whether the same original state predicts which judgments change under rephrasing that holds meaning fixed. Human review identifies rewrites that preserve the facts and relationships relevant to the unchanged consideration.

Reviewers do not decide whether an action is morally right. They judge source clarity, ordinary meaning, and semantic preservation.

## Independence and eligibility

Two reviewers complete each task independently. Separate Claim 1 and Claim 2 reviewer pairs are preferred. A reviewer must not see item-level AI preflight classifications, model answers, activation scores, geometry, native confidence, predicted fragility, intervention output, another reviewer's answers, or derived consensus categories for the items they judge.

Record any exposure before beginning. An exposed review cannot enter the independent-consensus population for the affected items.

Work independently. Do not discuss individual rows until both submissions have been locked. Do not search for benchmark labels or model results, and keep filenames, row order, review IDs, columns, and visible source fields unchanged.

## Work sessions

Complete no more than 60 rows or 60 minutes in one block, whichever comes first. Take a break of at least 10 minutes before the next block and record each block in the metadata file. Stop when concentration drops or when several rows begin receiving the same answer without fresh comparison.

Calibration examples are separate from study rows. Complete [human review calibration](HUMAN_REVIEW_CALIBRATION.md) before beginning the study package. Do not copy calibration explanations into study answers.

## Claim 1 first stage

The first stage contains 125 development checkerboards. Each row presents two situations, two repeated considerations, and the four stored relations in their checkerboard positions. Make six required judgments.

For each of the four cells, answer the following question.

> Using the stated facts and the ordinary meaning of the consideration, is the stored `Supports` or `Opposes` relation a clear and plausible reading?

| Response | Rule |
| --- | --- |
| `CLEAR` | The situation is intelligible, the consideration applies, and the stored relation is a natural reading without a substantial competing interpretation |
| `AMBIGUOUS` | The consideration is relevant, but missing detail or competing readings prevent a clear relation judgment |
| `INVALID` | The situation is incoherent, the consideration does not apply, or the stored relation is not a plausible reading |

For each repeated consideration, answer the following question.

> Does the consideration retain the same ordinary meaning across the two situations?

| Response | Rule |
| --- | --- |
| `STABLE` | The phrase keeps the same ordinary sense across both situations |
| `SHIFTED` | The phrase changes sense enough to alter the comparison |
| `UNCERTAIN` | The available wording does not establish stable ordinary meaning |

Use the failure fields only when a cell is `AMBIGUOUS` or `INVALID`, or when a sense judgment is `SHIFTED` or `UNCERTAIN`. Select one or more allowed flags and write one short explanation. Leave both fields blank after `CLEAR` or `STABLE`.

Cell failure flags use `SITUATION_UNCLEAR`, `CONSIDERATION_INAPPLICABLE`, `COMPETING_RELATION_READINGS`, `STORED_RELATION_IMPLAUSIBLE`, `UNCERTAIN_REFERENCE`, `MALFORMED_WORDING`, or `OTHER`.

Sense failure flags use `SHIFTED_MEANING`, `UNCERTAIN_REFERENCE`, `CHANGED_APPLICABILITY`, `MALFORMED_WORDING`, or `OTHER`.

Do not infer an unstated motive, consequence, relationship, or factual condition. A relation can be morally debatable and still be `CLEAR` when the stored reading is ordinary and the competing interpretation is not substantial. Use `AMBIGUOUS` when two readings remain live after applying only the stated facts.

## Claim 1 second stage

Do not begin this stage until both first-stage submissions have been locked and the second-stage package has been generated mechanically.

A consideration pair enters this stage only when both source cells received two independent `CLEAR` judgments and both reviewers marked its ordinary sense `STABLE`. Judge holder stability and normative-object stability as `STABLE`, `SHIFTED`, or `UNCERTAIN`.

Holder stability asks whether the person or group whose right, welfare, duty, or value is at issue occupies the same abstract role across the two situations. Normative-object stability asks whether the protected or burdened choice, interest, property, life, relationship, or institution is the same kind of object across the two situations.

These fields determine the exact-reason-like sensitivity population. They do not change first-stage source clarity.

## Claim 2 rewrite review

The Claim 2 package contains 481 candidate rows from 112 base items. The 80 representative items remain separate from the 32 geometry-confidence disagreement items in analysis, while the reviewer sheets hide that selection information.

Each row presents an original situation and action, one proposed rewrite, and the unchanged named consideration. Answer the following question.

> Does the proposed rewrite preserve all facts and relationships that could reasonably affect whether the unchanged consideration supports or opposes the action?

| Response | Rule |
| --- | --- |
| `PRESERVED` | Relevant actors, action, target, intent, causal facts, consequences, certainty, scope, moral stakes, specificity, and consideration target remain substantively fixed |
| `CHANGED` | At least one substantive difference could reasonably alter the `Supports` or `Opposes` judgment |
| `UNCERTAIN` | Ambiguity, awkward wording, or incomplete comparison prevents a confident preservation judgment |

Use diagnostic fields only after `CHANGED` or `UNCERTAIN`. Select one or more allowed flags and write one short explanation. Leave both fields blank after `PRESERVED`.

Allowed flags use `ACTOR_ACTION_TARGET_INTENT`, `FACTS_CERTAINTY_SCOPE_SPECIFICITY`, `CONSEQUENCES_SEVERITY_MORAL_STAKES`, `CONSIDERATION_TARGET_APPLICABILITY`, `UNCLEAR_OR_MALFORMED`, or `OTHER`.

Compare the texts directly. Do not predict whether a model would flip. Do not decide whether either moral answer is correct. A smoother or more explicit sentence can remain `PRESERVED`; a rewrite that resolves an ambiguity, adds a causal link, changes certainty, changes who is affected, or narrows the action is `CHANGED` or `UNCERTAIN`.

## Primary rules and adjudication

Claim 1 primary source clarity requires two independent `CLEAR` judgments for the sampled cell. Claim 2 primary rewrite acceptance requires two independent `PRESERVED` judgments. Any uncertainty excludes the item from the independent-consensus population.

Independent disagreement remains an observed result. Adjudication begins only after both original submissions are locked. Adjudicated cases enter a separately named secondary population and retain their original disagreement status.

The stable join maps all 112 Claim 2 base items to one Claim 1 board, cell position, source row, stored relation, and review field without text matching. Validation fails on missing, duplicate, positional, row, or relation mismatches.

## Return and locking

Complete only the judgment, conditional diagnostic, explanation, and metadata fields. Do not reorder rows, rename files, add columns, alter source text, or replace review IDs. Return the CSV and metadata JSON together.

The locking command verifies every source field against the frozen blank template. Any source edit, missing row, duplicated ID, invalid response, misplaced diagnostic, incomplete attestation, or changed instruction hash fails closed and requires a new versioned submission.

## Public repository boundary

The files in [review templates](../review_templates/) contain headers only. They document the schemas and allow integrity tests without redistributing ValuePrism text. Row-filled blinded packages, reviewer answers, private ID maps, sampling strata, model measurements, and sealed material remain outside the public repository.
