# A corrected music genre still leads to the old country

Qwen2.5-7B-Instruct reports both updated facts correctly, yet answers a two-hop
question differently when the same outdated statement changes position.
This is a saved, qualified MQuAKE case. The statements are counterfactual
benchmark stipulations, not claims about the people, works or religions named.

## A short two-hop example

The records associate Funky Drummer with a music genre and associate each genre
with a country. The correction assigns Funky Drummer to opéra comique.
The final chain is Funky Drummer, opéra comique, France.

The exact question is:

> From which country did the genre that includes Funky Drummer come from?

| History | Outdated statement | Its position | Saved answer |
|---|---|---|---|
| H0 | Funky Drummer's genre is funk | First | France |
| H1 | Funky Drummer's genre is funk | Last | United States of America |
| S0 | Eric Paslay's genre is funk | First | France |
| S1 | Eric Paslay's genre is funk | Last | France |
| Clean C | Neither statement is outdated | Final facts from the start | France |
| No-op C+ | Neither statement is outdated | Same corrections applied | France |

Every non-clean history receives these same two corrections:

```text
SET The type of music that Funky Drummer plays is opéra comique.
SET The type of music that Eric Paslay plays is opéra comique.
```

The final records also retain “funk was created in the country of United States
of America” and “opéra comique was created in the country of France.” Thus every
history ends with the same four facts.

The [full records and all checks](qwen-simple.md) show every starting history,
both references and every saved answer. The required checks ask Funky Drummer's
genre and the country associated with opéra comique. Both pass under all four
histories. The unrelated-entity check passes too.

Read the [saved native message payloads](qwen-simple-prompts.json), including
the system instruction, record text, corrections and each question. These are
selected records from the frozen native prompt plan. Chat-text and token-ID
hashes and input lengths are retained. No demonstrations were inserted into
this native wrapper.

Source case: `531`; counted phrasing: `J1`. Another phrasing in the same case
fails the two references and is not counted, despite changing answers. The
replay also supplies an [eligible non-disagreement](qwen-contrast.md) from case
963. Both contrasts keep the counting rule visible.

## The paper's three-hop example

The paper's Section 3.2 and Table 7 use William Levada's religion, its founder
and the founder's workplace. In these benchmark records the corrections set
both William Levada and the unrelated Albert of Saxony to Church of Scotland.
The final relevant chain is Church of Scotland, John Knox, Edinburgh.

The exact downstream question is:

> What place did the person who founded William Levada's religion work in?

| History/reference | Qwen2.5-7B | Gemma-2-9B |
|---|---|---|
| Related outdated statement first | Galilee | Edinburgh |
| Related outdated statement last | Edinburgh | Galilee |
| Unrelated outdated statement first | Edinburgh | Edinburgh |
| Unrelated outdated statement last | Edinburgh | Edinburgh |
| Final facts from the start | Edinburgh | Edinburgh |
| Already-correct facts with the same corrections | Edinburgh | Edinburgh |

All three required fact checks pass under every history for both models.
The remaining-chain check also passes. See the complete
[Qwen checks and records](qwen-paper.md) and [Gemma checks and records](gemma-paper.md),
with their [Qwen](qwen-paper-prompts.json) and [Gemma](gemma-paper-prompts.json)
saved message payloads. This is case `1617`, phrasing `J2`.

Galilee is the endpoint implied by the superseded Catholic Church, Jesus Christ,
Galilee chain present in the records. That correspondence is a diagnostic
observation. It is not a measured internal reasoning trace.

## What these comparisons change

H0 versus H1 changes the placement of a relevant outdated statement.
H versus S changes whether the outdated statement concerns the queried entity
or an unrelated entity. Both manipulations preserve the same corrections and
final facts. These four histories are not the synthetic task's eight binary
starting assignments. Synthetic stance records were generated separately.

These are native history-dependence diagnostics, not the external-repair
evaluation. The later repair uses a different wrapper. Its
[recorded Levada example](../records/public/recorded_case_1617.json) retains
separate current-state construction and saved repair outputs. Its two displayed
history rows share recorded calls; they are not two independent neural replications.
The [existing builder](../../history_updates/experiments/external_state/v3/src/repro/example.py)
checks state identity, ranker selection and exact execution.

The native question-level totals are 94/384 for Qwen2.5-7B and 101/384 for
Gemma-2-9B. The [CPU replay](../README.md) recomputes these from every saved
response using the original parser and guards. The 45% and 22% synthetic
case rates have different denominators; see the
[experiment overview](../../docs/experiment_overview.md).
