# Fixed worked example: 0127, seed 1

Post-hoc selection specified in the handoff; zero new independent observations. Histories 1 and 2 each change one fact and have 519-token prefixes.

| question | A_history1 | A_history2 | B16_23_history2 | C24_31_history2 | D_reference | RB16_23_history2 | RC24_31_history2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Does Esme favor the Library proposal? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Do Esme and Vera take the same side on Library? | Yes | No | Yes | No | Yes | No | Yes |
| Does Vera favor the Library proposal? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Is the Library proposal supported by both Esme and Vera? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Does at least one of Esme and Vera support Library? | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Answer-format boundary (same semantic SAME target):

| Query | A witness | B all-history target correct | B full criterion | C full criterion |
|---|---|---|---|---|
| QBR1-FINAL-0127-d0-02 | True | True | True | False |
| QBR1-FINAL-0127-d1-02 | True | False | False | False |

D is A/origin-zero with the same learned update, prepared from current records before questions. The A/B-format target is not to be described as a full two-format repair. The separate 0146 example must retain reverse-error donors that originally answered correctly; see SAME_CASE_FORWARD_REVERSE.json and DONOR_AGREEMENT.csv.
