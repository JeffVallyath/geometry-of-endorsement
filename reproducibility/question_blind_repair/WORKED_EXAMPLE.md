# A repaired same-side answer and its reverse

Esme and Vera both support the Library proposal. Under one edited history, the
model correctly reports each person's position but says they take different
sides. Targeted refresh corrects that same-side answer. Copying values in the
reverse direction makes the full current-record reference answer incorrectly.

This illustration was selected post hoc and then reproduced in the extension.
Its reproducibility identifier is `QBR1-FINAL-0127`, stored seed `1`. The shown
histories have the same number of changed facts and the same prefix length.
Every question is asked independently from a context clone; no question sees
the model's previous answers.

| Independently asked question | Edited history 1 | Edited history 2 | Targeted refresh | Comparison refresh | Full reference | Reverse targeted | Reverse comparison |
|---|---|---|---|---|---|---|---|
| Does Esme favor the Library proposal? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Does Vera favor the Library proposal? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Do Esme and Vera take the same side on Library? | Yes | No | Yes | No | Yes | No | Yes |
| Is Library supported by both Esme and Vera? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Does at least one of them support Library? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Targeted and comparison refresh use edited history 2. Reverse transfers copy
from that history into the full reference. The correct answer is Yes throughout
this table. Full qualification also checks the other required questions and all
eight histories, not just the displayed slice.

The same semantic target has different outcomes across answer formats:

| Format | Baseline qualifies | Target correct across all histories after targeted refresh | Full targeted repair | Full comparison repair |
|---|---|---|---|---|
| Literal Yes/No | Yes | Yes | Yes | No |
| Counterbalanced A/B | Yes | No | No | No |

Caption: independently asked facts and logical answers under edited contexts,
partial refresh and reverse transfer. Targeted refresh corrects the same-side
answer, and reverse transfer makes it wrong again. This post-hoc-selected example
passes full repair in Yes/No but not A/B format.

The [source answer table](supplied/verification/WORKED_EXAMPLE.csv) and
[full target checks](supplied/verification/TARGETS.csv) are replayed from raw scores.
The original [outcome-independent illustration](supplied/primary/figures/WORKED_EXAMPLE.svg)
remains archived unchanged.

## When a correct donor causes an error

A separate selected case produces reverse errors from values copied from a
donor that originally answered correctly. For `QBR1-FINAL-0146`, seed `0`, the
eligible target is `QBR1-FINAL-0146-d0-extra1`.

| Donor history | Donor original answer | Full reference answer | Reverse targeted answer |
|---|---|---|---|
| 6 | Correct | Correct | Wrong |
| 7 | Correct | Correct | Wrong |

The [complete donor table](supplied/verification/DONOR_AGREEMENT.csv) retains
agreement and disagreement for every reverse read. The
[same-case record](supplied/verification/SAME_CASE_FORWARD_REVERSE.json)
links these errors to the eligible target. The copied values have a causal
effect that depends on the recipient computation; reverse transfer is not
universally faithful replay of a donor's old answer.
