# A06. Stronger controls and fresh confirmation

Repository topic A06. Manuscript Section 4.3 and Appendix D.1 cover the original
post-hoc analysis. Section 4.3, Appendix D.2 and Appendix Figure 6 also cover the fresh
prospective confirmation and separate utility comparison.

## Original-output decomposition

Matching requires the same number of changed facts and exactly the same tokenized
prefix length within a model, condition and seed. The target answer is excluded
from the additional logical correctness checks.

The same-pair check requires the other two joint-question functions for the target
pair to be correct under every history. The cross-pair check requires all three
functions on each of the other two pairs. Both also require all measured direct
facts and the original target references to be correct.

| Criterion | Retained coded opportunities | Shared cases |
|---|---:|---:|
| Original witnesses | 653 | 62/64 |
| Matching with original eligibility | 430 | 58/64 |
| Matching and same-pair controls | 188 | 53/64 |
| Matching and cross-pair controls | 82 | 34/64 |
| Matching and both controls | 64 | 29/64 |

These counts pool correlated observations across models, procedures, seeds and
formats, out of 13,824 coded opportunities. Of the 64 opportunities retained by
matching and both controls, 60 come from the two learned editors and 4 from the
two textual procedures (one per backbone and procedure). Cases are counted once within each reported union. They are descriptive
counts from the original saved outputs, not a new prospective prevalence estimate.
Matching leaves token order, lexical associations and clause position free to differ.

The [full opportunity table](../../reproducibility/state_sufficiency/cpu_audit/RESULTS.csv)
retains every eligibility flag, matching pair, seed and denominator.
[Gate mappings](../../reproducibility/state_sufficiency/cpu_audit/GATE_SPEC.json)
specify every target's controls; [survivor records](../../reproducibility/state_sufficiency/cpu_audit/SURVIVORS.json)
retain the complete selected response panels and manual-audit accounting.

The two-format sensitivity counts nine semantic targets per case. It requires the
same matched history pair to disagree in the same semantic direction under both
formats. Separate disagreements under different pairs do not qualify. With both
logical controls, this sensitivity retains 13 semantic observations across 12 cases.

## Fresh cases and fixed criteria

The fresh study uses Gemma-2-9B-it, the same frozen model revision and both stored
constrained-editor seeds. No editor was refitted. Its generator seed is 2609182101.
There are 64 final cases and eight separate technical cases. The final balance is
four person/proposal size combinations crossed with eight final truth assignments,
with two cases per cell. Collision checks precede inference.

All prompts, answer mappings, matching pairs and validation partitions were fixed
before the new outcomes. The primary criterion requires both logical-control sets
and matching. Ineligible targets stay in the denominator. The bootstrap resamples
case IDs jointly across seeds and formats: 10,000 draws, seed 2609182102 and a 95%
percentile interval.

| Stored seed | Scheduled coded targets | Eligible targets | Witnesses |
|---|---:|---:|---:|
| 0 | 1,152 | 509 | 12 |
| 1 | 1,152 | 146 | 10 |

The 22 observations span 14 cases. The first four positive cases in the frozen
source order were repeated completely. All 2,312 repeated logical records have
identical retained scores and answer/token bindings. Repeats stay within their
original case denominators.

[Scientific protocol excerpt](../../reproducibility/fresh_history_validation/protocol/PROTOCOL.md),
[analysis rules](../../reproducibility/fresh_history_validation/ANALYSIS_RULES.md),
and [generated inputs](../../reproducibility/fresh_history_validation/generated/INDEX.json)
provide the exact definitions and records.

## Held-out validation utility

Each relation pair withholds one of the three question functions, selected by the
fixed case/pair rule. The remaining six questions form the validation pool. Both
procedures share the same preliminary direct checks and receive six additional
scored questions. History-pair selection uses only pre-model matching information.

Procedure A asks all six validation questions at one anchor history. Procedure B
asks three questions at each of the two selected histories. Their acceptance
decisions and retention rankings were locked before withheld-outcome analysis.
Invalid responses count as task errors.

The named endpoint uses literal No/Yes at 50% retention. The common pools contain
59 and 57 cases for seeds 0 and 1. Each bootstrap sample reapplies eligibility and
ranking, preserving paired case IDs across seeds. Undefined draws make an interval
unavailable rather than being silently dropped.

Keep [acceptance and errors](../../reproducibility/fresh_history_validation/tables/UTILITY_ACCEPTANCE.csv),
[per-seed contrasts](../../reproducibility/fresh_history_validation/tables/UTILITY_SEED_CONTRASTS.csv),
[all retention levels](../../reproducibility/fresh_history_validation/tables/UTILITY_EQUAL_RETENTION.csv),
and [both directions of policy disagreement](../../reproducibility/fresh_history_validation/tables/POLICY_DISAGREEMENT_CASES.csv)
together. Equal query counts do not establish equal GPU cost because execution
shares compiled contexts.

## Post-hoc diagnostics

Thirteen full-criterion observations have a withheld target, spanning seven cases.
Both validators accept all thirteen corresponding case/seed/format combinations.
Nine also disagree at the validators' selected history pair. All 22 primary
observations retain a witness pair with equal token-ID multisets, including
multiplicity. Token order and relation assignments still differ.

These [diagnostics](../../reproducibility/fresh_history_validation/diagnostics/DIAGNOSTICS.json)
were added after outcomes and leave the frozen primary and utility endpoints
unchanged. They are separate from prospective confirmation.

## Replay and numerical rules

```sh
python -m repro.cpu_robustness --output reproduced/controls
python -m repro.fresh_history --output reproduced/fresh
```

Identities, classifications, counts and seed grouping require exact agreement.
Finite derived floats use the existing absolute tolerance of 1e-12 with zero
relative tolerance. The original fresh-process repeat threshold is 1e-6; the
observed repeat scores agree exactly. These commands reconstruct saved scores,
not neural activations or new answers.
