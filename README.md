# Geometry of Endorsement

Language models can reverse a moral judgment after a rewrite that preserves the underlying situation. This project tests whether the model's original hidden state contains advance evidence of that instability.

The plan tests two claims. First, does the model's internal state encode the relation between the situation and consideration? Second, does that same state predict which judgments change under rephrasing that holds meaning fixed? Later verdict and intervention analyses remain follow-ups to these claims.

Three quantities remain separate throughout the project.

| Quantity | Meaning |
| --- | --- |
| Stored relation | ValuePrism records whether a consideration Supports or Opposes an action |
| Model reason judgment | The model states whether that named consideration Supports or Opposes the action |
| Complete verdict | The model gives its broader stance on the action, including support, opposition, permission, uncertainty, or refusal |

The stored relation supplies the representation target. It does not stand in for the model's own judgment or its complete verdict.

## The checkerboard test

ValuePrism contains consideration names that receive both labels across different situations. The project arranges two such reversals into a reciprocal checkerboard.

| Situation | Consideration A | Consideration B |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

A score based only on the situation cancels exactly. A fixed consideration score also cancels, as does their sum. A positive checkerboard interaction therefore requires the score to change with the pairing between the situation and the consideration.

Nonlinear wording cues can still produce that pattern. Held out consideration identities, a matched text comparison, reversed answer mappings, factual positive controls, and replication in another model family address that alternative in the development design.

## Development evidence

Llama 3.1 8B Instruct and Gemma 2 9B it both contain a linearly readable Supports versus Opposes signal at the final prompt token.

| Model and readout | Selected layer | Checkerboard interaction |
| --- | --- | --- |
| Llama native answer margin | n/a | 1.609 |
| Llama difference in means | 19 | 1.647 |
| Llama logistic probe | 19 | 2.084 |
| Gemma difference in means | 27 | 2.322 |
| Gemma logistic probe | 27 | 2.150 |
| Frozen SBERT comparison | n/a | 0.284 |

On the 500-row Llama development evaluation, relation AUROC reached 0.721 for the native margin, 0.732 for difference in means, and 0.780 for the logistic probe, while all three additive controls produced interaction zero on development data. Gemma fitted its own direction in its own activation space and reproduced the result at layer 27.

These are development results. Human semantic review, held out prediction of rephrasing instability, and sealed confirmation remain incomplete. [Results so far](docs/RESULTS_SO_FAR.md) preserves the numerical record and evidence boundary.

## The construct boundary

The checkerboards repeat a named consideration, not a fully specified moral proposition. A word such as autonomy can concern a patient's treatment in one situation and a speaker's expression in another. The holder and normative object can change even when the ordinary sense remains stable.

The supported development claim concerns the contextual direction of a repeated named consideration, while a stronger exact-reason-like claim requires human-confirmed stability of ordinary sense, holder, and normative object across the paired situations.

## The rephrasing test

The current pilot contains 112 original items from the open Llama development population. Eighty form a representative sample. Another 32 were selected because relation geometry and native confidence rank their vulnerability differently.

Each original has four proposed rewrites. The 448 ordinary candidates share 11 identity controls, 11 trivial restatements, and 11 single-fact changes. The candidate text is frozen, but the original review form has been superseded before distribution because an audit found excessive burden and an uncertainty rule that was too permissive before any human label or rephrasing outcome was recorded.

The reduced review asks whether each rewrite preserves the substantive facts and relationships relevant to whether the unchanged consideration supports or opposes the action. Two independent `PRESERVED` judgments are required for the primary population. Any `UNCERTAIN` judgment excludes the rewrite from that population.

Claim 1 reviewers separately judge whether the stored relation in the source cell is clear and plausible. The primary Claim 2 population joins unanimous source clarity with unanimous rewrite preservation through stable item identifiers. Human review labels, Claim 2 rephrasing outcomes, susceptibility estimates, and sealed confirmatory results remain unobserved.

Prediction uses the original state only. The baseline receives the original prompt text and native confidence. The geometry model receives those inputs plus the original relation score. Generic activation summaries, principal components, random directions, label-permuted directions, and other unrelated measurements test under grouped held out evaluation whether any improvement belongs specifically to relation geometry.

The primary prediction unit is one original item. Its four rewrites are repeated trials, not four independent geometry observations. The 80 representative items support prevalence and prediction estimates, while the enriched 32-item disagreement sample remains a separate diagnostic analysis.

## Causal compatibility

Both pretrained checkpoints passed the token-level intervention compatibility test. Llama uses layer 19 and Gemma uses layer 27. In each case, the intervention changes the chosen final-prompt-token block output, leaves every untargeted position unchanged at that intervention site, preserves all model parameters, and returns finite answer probabilities under BF16 execution for both checkpoint tests.

This establishes access to the intended activation. Signed dose response, answer remapping, direction controls, location controls, and damage measurements remain unrun, so causal efficacy has not been established.

## Repository guide

[Results so far](docs/RESULTS_SO_FAR.md) records completed measurements, controls, failures, revisions, and provenance. [Project plan](docs/PROJECT_PLAN.md) records the human-review design, analysis populations, decision gates, deferred mechanism work, and interpretation rules. [Human review](docs/HUMAN_REVIEW.md) provides the reviewer rules, response definitions, eligibility limits, fatigue controls, diagnostics, and locking procedure.

The August 5 project-status notebook remains a dated artifact and no longer defines the current frontier. The ValuePrism leakage and factual-control notebooks reproduce their respective analyses.

For a local code check, run the following commands.

~~~bash
python -m pip install -e .
python -m pytest -q
~~~

ValuePrism row text remains outside this repository under the dataset license. Private development outputs, row-level Gemma artifacts, row-filled human-review packages, reviewer answers, and sealed results also remain outside the public release. Header-only [review templates](review_templates/) expose the frozen schemas without study rows.

A successful held out prediction result would support using original-state relation geometry as an early screening signal for rephrasing fragility, while a null result would restrict the finding to readable prompt state and redirect the next study toward composition and verdict stages.
