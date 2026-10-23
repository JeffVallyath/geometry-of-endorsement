# Section 3: Changing the direct answer

<a id="sec:changing-direct-answer"></a>

Changing a direct answer does not ensure that the other answers change as they
would after rewriting the underlying relation. In the retained Gemma comparison,
rank-one steering moves the consequence pattern farther from the natural
counterfactual. Interpolation and full activation replacement recover most of it.
Because neither tuned arm meets the prespecified full-dose rule, the comparison is
reported descriptively (qualification-limited), not as a qualified confirmatory test.

[Figure 3: consequence recovery](../figures/main/consequence_qualification.pdf)
corresponds to manuscript Section 3 and Appendix B.

## Method

Each input describes two people, two propositions and four explicit relations.
The counterfactual changes only the focal support/opposition relation. Two wordings
express each context. A six-question battery separates the direct calibration
question from three focal consequences and two unchanged-relation controls.

Steering adds a multiple of a relation direction to the final prompt-position
activation. Its magnitude is chosen using only the direct question. Whole-state
interpolation uses the natural base-to-counterfactual activation difference.
Full replacement copies the counterfactual activation for the same query.
Model weights, questions and the other context facts remain fixed.

The recovery score compares the five remaining answer-margin changes with the
natural counterfactual changes. A score of one matches that pattern; zero matches
the unchanged base. Negative values move farther from the counterfactual than
the unchanged base does. The direct calibration answer is excluded from this score.

## Result

The following values use all 96 final worlds where the intervention was applied.
The steering and interpolation rows use the target fraction 0.75.

| Intervention | Mean recovery | 95% world-bootstrap interval |
|---|---:|---:|
| Original relation direction | -0.8508 | [-1.0048, -0.7067] |
| Prompt-format-refitted relation direction | -1.1825 | [-1.3325, -1.0399] |
| Natural whole-state interpolation | 0.9136 | [0.8995, 0.9260] |
| Full activation replacement | 0.9813 | [0.9799, 0.9826] |

Both steering directions meet the calibration tolerance in 96 of 96 worlds and
produce the counterfactual direct answer in 95. The complete dose rule passes
in only 80 worlds, below the required 90% coverage. Natural whole-state
interpolation at the same target fraction also meets the calibration tolerance
in 96 of 96 worlds but passes the complete dose rule in only 81 (`dose_valid:
false`), so neither tuned arm is dose-qualified. Full replacement is an untuned
reference. The prompt-format-refitted
direction also fails its fit-population qualification. Llama fails natural-reference
qualification across the three candidate protocols, so it supplies no qualified
cross-model consequence comparison.

These outcomes make the Gemma comparison qualification-limited. The formal
[classification record](../reproducibility/representation/counterfactual_fidelity/results.json)
preserves `DIRECTION_NOT_QUALIFIED` for the primary refitted arm,
`STRONG_TARGET_UNREACHABLE` for the original-direction transport assessment,
and `ASSAY_NOT_QUALIFIED` for Llama. The transport label applies the complete dose
rule, not merely whether the direct calibration tolerance was reached.

[Repository topic A04](appendix/A04_counterfactual_calibration.md) gives the exact scoring,
calibration amendment, gates, sample counts and parameter availability.

## Reproduce

```sh
python -m repro.counterfactual --output reproduced/consequences
```

The command reconstructs the saved-score recovery panels across target fractions,
wordings and answer formats. It checks all 145 original numerical comparisons at
their original absolute tolerance of 2e-10. It writes `RESULTS.json` and
`VERIFICATION.json` into a new directory. It does not fit directions or generate
neural outputs.
