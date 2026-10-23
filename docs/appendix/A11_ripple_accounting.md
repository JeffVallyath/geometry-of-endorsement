# Public-data accounting and parser sensitivity

Repository topic A11. See manuscript Appendix F.2 and Table 5.

## Inputs, rendering and fixed model

The retained sample contains five development cases, six qualification cases and 34 final cases. Their edited-subject clusters are disjoint. The final cases contain 98 strong-chain composed questions. [SCOPE.json](../../reproducibility/in_context_updates/supplied/protocol/SCOPE.json) retains every selected ID, record order and scientific gate. The exact original upstream dataset checkout revision is unrecorded.

For each case, the current table includes the edited record and every supporting record required by its selected questions. Records are deduplicated by subject, relation and temporal address. Conflicting current targets are rejected. A root-wide topological order is fixed with deterministic address tie-breaking. The edited address may be the second hop of a compositional chain.

The two chronological streams contain one obsolete assignment followed by the complete current table. Their obsolete values and current value are three distinct entity IDs at the same address. Only the obsolete line differs between streams. Equal record counts do not imply equal token lengths; actual token counts are retained.

Five contexts are measured: no records, supplied facts, canonical current records, and the two chronological histories. The two clean references use the same current table and order with different headers. A last-write ledger confirms that both histories resolve to that table. Prompts contain no scoring aliases, eligibility flags, history labels or computed downstream answer.

The model/tokenizer is `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`. The recorded settings are BF16, SDPA, TF32 off, single-example greedy decoding, one beam, repetition penalty 1 and at most 16 new tokens. Each independent prompt starts a fresh key/value cache. The fixed system instruction is `Answer with only the entity name.` [Scientific freezes](../../reproducibility/in_context_updates/supplied/freezes/PARSER_FREEZE.json) and [model identities](../../reproducibility/in_context_updates/supplied/freezes/MODEL_RECEIPT.json) retain the exact settings and upstream file hashes.

## Candidate construction and parser rules

The registry is built from source entity labels and aliases within each declared object type. Candidates are fixed from source data, not model responses or the desired answer. The implementation normalizes registry strings and response strings identically.

The strict scorer first removes leading whitespace, including leading newlines, with `lstrip()`. It takes the first remaining line and trims its outer whitespace. It removes at most one trailing character from `.,;:!?`. Correctness is case-sensitive equality with a benchmark answer or alias processed by that same rule. The [implemented scorer](../../reproducibility/in_context_updates/supplied/source/icmh/parse.py) is authoritative for this leading-whitespace detail.

Entity extraction additionally applies NFKC normalization, case folding and whitespace collapse. It matches the entire resulting string within the declared object type. Extra prose receives no prefix or substring rescue.

| Rule | Label and alias handling |
|---|---|
| Primary label-first, LP1 | A unique source-label match wins. Multiple label matches abstain. Without a label match, aliases must identify one entity. |
| Alternative union | Pool labels and aliases, then require exactly one matching entity. |

Neither rule uses the gold answer to break a tie. LP1 resolves a label/alias collision in favor of the label owner, even when the alias owner would be the intended answer. Ambiguous label/label matches still abstain.

The label-first amendment was fixed before the retained model outputs. It changed candidate lists globally at 117 type/string keys. It did not alter the 45 case metadata objects or any of the five inference plans and scoring maps. The original union parser remains a descriptive sensitivity analysis. The pre-output structural simulations were feasibility checks, not model outcomes. A residual label collision involving `London` remains in case `popular:627`; the case and denominator were retained.

## Eligibility, disagreement and qualification

The source uses four flags:

| Flag | Required observation |
|---|---|
| A, per case | Every current record maps to its correct entity under both histories. |
| C, per question | Both clean references are strictly correct. |
| D, per question | The two history-conditioned answers map uniquely to distinct entities, with exactly one strictly correct. |
| W, per question | A, C and D all hold with the necessary responses measured. |

Two wrong answers, unresolved responses and different aliases for the same entity receive no witness credit. Missing measurements remain missing, not zero. Strict scores remain visible alongside entity extraction.

Qualification excludes chronological downstream questions. It tests readiness without exposing the target history effect. The fixed scientific gates and their measured outcomes are:

| Gate | Threshold | Primary | Alternative |
|---|---|---:|---:|
| Clean-reference correctness | At least 13/16 for each reference | 16/16 each | 16/16 each |
| Current records correct in both histories | At least 18/22 | 20/22 | 18/22 |
| Unresolved chronological atomic reads | At most 4/44 | 1/44 | 5/44, fails |
| Factual plus reference coverage | At least 10/16 across at least 3/6 cases | 13/16 across 4 cases | 5/16 across 2 cases, fails |

All 114 qualification generations are present, with no empty or truncated output. Final execution followed the primary gate. The alternative failures remain part of sensitivity interpretation; they do not retroactively substitute a different readiness rule.

## Final accounting and witnesses

| Measure | Primary | Alternative |
|---|---:|---:|
| Scheduled cases / questions | 34 / 98 | 34 / 98 |
| Eligible questions, A and C | 65 | 42 |
| Qualifying questions, W | 3 | 2 |

Twenty-four questions change at least one A/C/D/W or eligibility flag across parsers. Exact categories, per-case distributions and changed questions remain in the [complete result](../../reproducibility/in_context_updates/results.json) and [saved sensitivity analyses](../../reproducibility/in_context_updates/supplied/expected/final/PARSER_SENSITIVITY_COMPARISON.json.gz).

The primary witnesses are `popular:736 / Compositionality_II:2:0`, `popular:622 / Compositionality_II:2:0` and `random:491 / Compositionality_II:0:0`. Each incorrect qualifying answer equals that history's obsolete value. All other cases and failed gates remain in the denominator.

Locality uses eight prepared CounterFact questions across 34 contexts. Correctness is 170/272 without history and 94/272 under chronological history B. This is a repeated-context diagnostic, not 272 independent facts or a parametric-edit fidelity result.

## Complete phase and repeat accounting

| Phase | Cases | Recorded generations |
|---|---:|---:|
| Development | 5 | 207 |
| Qualification | 6 | 114 |
| Final | 34 | 1,430 |
| Preselected repeat, popular:670 | 1 | 35 |
| Preselected repeat, random:618 | 1 | 51 |

Both repeat schedules were fixed before final outcomes and run in separate fresh processes. All 86 recorded responses and classifications match under both parsers. Neither repeated case contains a witness. They add no independent cases and do not establish repeatability of the three positive cases.

The final denominator is the full restricted sample. Questions share cases, so no p-value or independent-question population bound is inferred from 3/98. The alternative parser's lower coverage is reported rather than hidden by changing eligibility.

## Reconstruct

```sh
python -m repro.in_context_updates --output reproduced/ripple
```

The frozen analyzers and both parsers are reused without scientific changes. They verify all five journal chains, reconstruct classifications and then open the saved expected analyses for comparison. Tests reject missing, extra and corrupted journal records. Public chains bind sanitized payloads; experimental prompts, generated text, token identities, durations and scoring fields are unchanged.

The retained builder and model worker document the original implementation. They are not a verified fresh-inference interface. Rebuilding the broader preparation pool requires source manifests absent from this compact release. No pretrained execution was performed for this artifact.
