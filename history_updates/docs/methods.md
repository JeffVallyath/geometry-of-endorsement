# Methods

Start with the [synthetic](../../examples/synthetic/README.md) and
[public](../../examples/public/README.md) measured walkthroughs. The
[experiment overview](../../docs/experiment_overview.md) keeps their populations,
denominators and selection chronology separate.

## Terms used in the tests

| Term | Meaning |
|---|---|
| History | The starting factual statements and their order, not previous generated answers |
| Synthetic case | One set of eight starting histories, common assignments and questions |
| Public case | One MQuAKE fact chain with three target phrasings and four histories |
| Downstream question | A question combining an updated fact with other facts |
| Direct check | An independent question asking for one required fact |
| Clean reference | The final facts supplied from the start, without the update |
| No-op reference | Already-correct facts receiving the same update requests |
| Counted question | An eligible question whose matched histories yield the required valid disagreement |

## Clarifications from the collaborator discussion

The paper and saved records settle the following points. These clarify the
existing method rather than add new experimental controls.

Every synthetic history receives all three update requests, including no-ops.
Three binary starting stances create eight histories per case. The stricter
matching rule concerns how many facts actually change, not the number of requests.

For each target, the stricter synthetic protocol requires twelve direct checks
and the other eight combined questions in that target's answer format to pass
under all eight histories. Compared pairs also match changed-fact count and
pre-question token length. The two references are separate. Section 3.1 and
Appendix A describe these controls; they are not the public MQuAKE protocol.

The original stricter 29/64 (45%) and fresh 14/64 (22%) results count synthetic
cases. The public 94/384 and 101/384 results count questions. The original
stricter controls were chosen after the initial outcomes were inspected, then
fixed for the new synthetic cases. Both tasks have empirical results.

Each synthetic question receives a fresh clone of the saved context cache.
The base-model weights are shared, not reloaded for each question. No generated
answer is fed to the next question. See the exact
[question formats](../../examples/synthetic/questions.md) and
[model-visible inputs](../../examples/synthetic/H5.txt).

Text corrections and activation interventions do not edit model weights.
A passed direct check establishes correct reporting for that query. It is not
an independent measurement of internal knowledge. The
[activation inputs](../../examples/synthetic/activation-inputs.json) are separate
from the model-visible text.

## Matched histories

Each case starts from several records and receives the same updates. The final
facts and downstream question stay fixed. A history-dependent answer counts
only when the required fact checks and both references pass, yet matched
histories yield different downstream answers. Scheduled questions remain in
the denominator when they fail a check.

The synthetic task uses fictional people and proposals. The stricter analysis
also matches the number of changed facts and tokens. It checks the other
combined questions. Its prospective panel uses new cases with the controls
fixed in advance. Public multi-hop questions retain their original entity
aliases and the frozen response parser. Bootstrap resampling keeps each case's
questions, histories and applicable seeds together.

## Cache interventions

Cache crossover combines blocks from two histories that disagree. The question
and direct checks stay fixed. The analysis measures whether the answer follows
the donor of the later blocks. Site and reader-head interventions preserve the
recorded selection procedure and all reserved-case outcomes.

Cache refresh copies a specified region from a reference cache before the
question is known. Its accounting separates repaired witnesses, newly created
witnesses, lost qualification and ordinary answer corrections or harms. The
control region and direct-fact errors remain in the results.

## External current state

An update makes a new dictionary snapshot. Answering reads the current snapshot
without reading earlier records. Code enumerates permitted operations, a trained
interpreter scores them, and the same exact executor returns the selected
operation's result. A nonpositive best score produces a refusal.

Identical complete current-state inputs under fixed inference remove the
historical input route. Correctness is measured separately: an interpreter can
still refuse or select the wrong operation. An exact executor alone does not
ensure isolation if history reaches another part of the answering interface.

The synthetic bridge first combines the model's saved direct answers using
supplied operations. It preserves incorrect direct answers. A separate saved
neural compiler reads original text into typed operations. Its observations are
joined to both activation-editor seeds without counting reuse as a second
compiler replication.

Complete-case correctness requires every associated target and control to be
correct under all four nonreference histories. Target, control and original-
snapshot answer counts have separate denominators. Individual incorrect
accepted answers remain distinct from refusals.

## History access and interpreter choice

The history-exposure test holds the dictionary, question, candidate operations,
trained weights and executor fixed. It changes only the additional text visible
to the interpreter. The comparison text replaces obsolete values with current
ones in the same line structure and position. Exact token-length matches are
reported rather than assumed.

The common-state reader comparison trains one model to generate an answer and
another to generate an operation. Both read the same constructed dictionary.
The operation generator and ranking interpreter share exact execution. Their
current and original-snapshot answers are paired by case, history and question.

## Corrected graph comparison

The corrected local GMeLLo implementation constructs an independent RDF graph
from initial facts and applies the supplied writes. Its question-only model
extracts a relation chain using a fixed ontology and training demonstrations.
Code normalizes and executes that chain. A nonempty graph answer takes priority,
including when it is wrong and fallback would be correct.

CORE retrieves current facts for fallback. FULL uses the same graph computation
and gives fallback all current facts. TEXT reads edits from correction sentences
and is secondary. Both repair seeds, the four histories, two references,
original-snapshot targets and all diagnostics remain in the comparison.
The two primary contrasts resample whole subjects with both repair seeds linked.
