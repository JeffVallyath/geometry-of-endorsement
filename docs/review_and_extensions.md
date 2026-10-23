# Questions worth taking forward

The existing results leave several questions open. For a review or extension,
link each question to a paper section, example or function.

| Category | What to record |
|---|---|
| Answered by existing evidence | The relevant result and its exact scope |
| Needs clearer explanation | The confusing sentence or missing example |
| Needs a new experiment | The hypothesis, missing evidence and smallest useful test |

## Safety and practical framing

Start with the two examples, Sections 2 and 3, and the limitations. The existing
result shows that reporting a correction correctly does not establish
history-independent downstream answers.

A proposal is to connect that failure to a concrete changing-record task, such
as a revised permission or a corrected planning constraint. The present tests
do not measure harmful behavior or safety failures in deployed agents.
The missing evidence is a controlled task linking the corrected fact to a
consequential decision, with matched histories, direct checks and clean references.
The [case and guard code](../CONTRIBUTING.md#code-navigation) provides a starting point.

## Broader examples and question structures

Compare the fictional stance task with the
[public two-hop and Levada examples](../examples/public/README.md).
The existing public results already cover relation chains, not just binary
stances. Their supplied facts and inspected benchmark cohort limit the inference.

A proposal is to test a domain or question structure that adds evidence beyond
both existing tasks. The missing evidence is performance under a predeclared
new case-generating process that preserves the relevant controls. Do not count
new phrasing of an existing example as independent replication.

## Internal measurement, separate from repair

Section 4, Appendix D and the cache intervention sources cover internal measurements.
The [code map](../CONTRIBUTING.md#code-navigation) locates the implementations.
The measured effects involve selected cases and tested cache regions. They do not identify every
possible cause or rule out every internal repair.

A proposal is to test whether an internal measurement predicts held-out
history-dependent answers. Measurement need not repair the behavior. The
missing evidence is prediction on independently selected cases with controls
for direct-answer correctness and ordinary errors. Specify selection, held-out
evaluation and what would falsify the prediction. Do not infer an unobserved
replacement circuit from the current intervention failures.

## Review notes

| Question | Category | Paper/code link | Evidence or next deliverable |
|---|---|---|---|
| Add a question here | Choose one category above | Add an exact link | State what would answer it |
