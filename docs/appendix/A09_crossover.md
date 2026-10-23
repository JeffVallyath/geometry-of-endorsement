# Crossover details

Repository topic A09. See manuscript Section 6.1, Figure 5 and Appendix E.1.

## Intervention and fixed inputs

Later stored activations determine the joint answer in five of six selected
witness cases under the main answer format. The alternate format is less
consistent. A separate matched-case extension passes both formats with stricter
direct checks.

The study uses the existing Gemma-2-9B-it checkpoint and constrained writer.
Each stored updated prefix is evaluated with the same questions and unchanged
model weights. The crossover combines all prefix key/value-cache entries from
two histories at a fixed layer boundary. The first 16 cache layers, zero-based
0 through 15, come from one history. The remaining 26, layers 16 through 41,
come from the other. The learned writer acts after block 15.

The downstream question is introduced only after combining those stored prefix
states. Each question starts from its own state copy. The manipulation includes
both transfer directions, both unchanged originals, same-history recombination
and full-state-copy controls.

| State code | Earlier layers | Later layers |
|---|---|---|
| AA | History A | History A |
| AB | History A | History B |
| BA | History B | History A |
| BB | History B | History B |

These machine identifiers specify the inputs to the replay. They do not identify
a unique storage location or a complete natural causal pathway.

## Six-case panel and selection

The panel inherits six witness/control pairs from the existing selection.
Pairs with unequal stored positions were excluded before this retained panel.
No new behavioral ranking was performed. The exact questions, label-token order,
primary-format flags, exclusions and case order remain in the
[case plan](../../reproducibility/cache_crossover/case_plan.json).
The main format is the selected primary question; it is not always the `d0` code.

| Selected witness case | Main format | Alternate format |
|---|---|---|
| `SSC1-FINAL-0034` | Clear later-state effect | Insufficient original separation |
| `SSC1-FINAL-0026` | Clear later-state effect | Clear later-state effect |
| `SSC1-FINAL-0033` | Mixed | Insufficient original separation |
| `SSC1-FINAL-0004` | Clear later-state effect | Insufficient original separation |
| `SSC1-FINAL-0057` | Clear later-state effect | Mixed |
| `SSC1-FINAL-0054` | Clear later-state effect | Mixed |

The six control cases all have insufficient main-format separation. Under the
alternate format, five have insufficient separation and one is mixed. The
[complete 24-row panel](../../artifacts/cache_crossover/original_panel.csv) keeps
both roles and formats. Selected-case counts do not estimate prevalence.

## Exact classification and direct checks

Let p be the probability of semantic Yes after normalizing the two candidate
log scores. Original separation is `abs(p_AA - p_BB)`. The earlier-state error
is the mean of `abs(p_BA - p_BB)` and `abs(p_AB - p_AA)`. The later-state error
is the mean of `abs(p_BA - p_AA)` and `abs(p_AB - p_BB)`.

A clear later-state effect requires separation at least 0.10, later-state error
at most 0.05, and earlier-state error at least 0.10. Both mixed states must give
the corresponding later original's hard answer, with valid scores and candidate
mass at least 0.90. Direct facts must also be preserved. The earlier-state label
reverses those comparisons. Separation below 0.10 takes the insufficient-separation
label; other unresolved cases are mixed.

For the main format, each direct answer must be correct, with correct-label
probability at least 0.95, candidate mass at least 0.90 and probability change
at most 0.02. The comparison reference is the original supplying the later layers.
The earlier panel's alternate-format rule checks correctness where both original
direct answers were correct. It does not impose the main format's full confidence
and drift rule. All reported direct-preservation checks pass their respective rules.

The journal contains 576 scored state/question cells and 1,296 baseline responses.
It retains both baseline passes, recorded comparisons with older scores, eligibility
checks and state-identity records. The 144 same-history and 144 full-copy comparisons
agree exactly. The allowed baseline differences remain 0.05 nat in log score,
0.005 in normalized probability and 0.005 in candidate mass.

Technical recovery reused 332 forwards and added 1,660, for 1,992 total. The earlier
engineering stop stays recorded. A deterministic binding correction repaired the
first pair's summaries without rerunning or changing its original scores.

## Matched one-case extension

The extension retrospectively selects `SSC1-FINAL-0035`, histories 3 and 5.
Both require two fact changes and have 517-token prefixes. The retained selection
record joins length/count matching with the earlier same-pair and cross-pair
logical controls. Its [candidate record](../../reproducibility/cache_crossover/extension/candidate.json)
keeps the original control entry and exact question identities.

Both answer formats show a clear later-state effect. All eight strict direct
checks pass, using the confidence, mass and drift thresholds above for both
formats. Minimum correct-label probability is 0.999230; maximum change is
0.00008612. The coded direct question about Ada reverses the label order used by
the other coded questions. The replay uses each question's own semantic mapping.

The extension retains 108 exact baseline replays and 24 exact same-history/full-copy
comparisons. Its 166 original neural forwards are historical accounting; CPU replay
performs none. Two formats on the same selected history pair remain one case.

The [joint table](../../artifacts/cache_crossover/extension_joint.csv) and
[direct table](../../artifacts/cache_crossover/extension_direct.csv) identify the
public journal and exact source row for every score. The earlier panel is unchanged.

## Reproduce

Run from the root into new output locations:

```sh
python -m repro.cache_crossover --output reproduced/crossover
python -m repro.cache_crossover_extension --output reproduced/matched-crossover
python scripts/plot_cache_crossover.py --output reproduced/crossover-plot
```

The first two commands independently normalize saved log scores, derive semantic
answers from the retained worlds, and apply the original gates. Saved expected
classifications are opened after reconstruction. The extension additionally checks
its baseline tokens and scores against the primary-study input. Original
floating-point comparisons retain their absolute tolerance of 1e-12.

The final command redraws a supplementary extension plot from those checked values.
The supplied manuscript Figure 5 is a separate preserved composition. CPU checks
verify recorded state-identity statements as data; they do not reconstruct or
independently attest unavailable native cache tensors.
