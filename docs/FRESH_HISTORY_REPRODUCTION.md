# Fresh matched-history confirmation and validation utility

History-dependent answers survived the strongest controls in 14 of 64 new Gemma
cases. The same study left the practical advantage of spreading validation
across histories inconclusive. The controls were fixed before new outcomes.
This is a separate prospective confirmation for one model and one learned-update
family; V10 remains the original primary study.

## Method

Each case starts from eight different histories. We apply the same stored
learned update so they reach the same intended current relations. Model weights,
the downstream questions and the response formats stay fixed. We first require
correct current-fact reports, reference answers and related logical answers.
A qualifying failure occurs when histories matched on changed-fact count and
prefix-token length still produce different valid answers to the target question.

The logical checks cover the other compositions of the target relation pair
and compositions of the other pairs. References contain the current facts
directly or apply the same update to an already-current state. The
[frozen protocol](../reproducibility/fresh_history_validation/protocol/PROTOCOL.md)
and [analysis rules](../reproducibility/fresh_history_validation/ANALYSIS_RULES.md)
give the exact criterion and prospective scope.

## Results and statistical units

| Measurement | Result | 95% paired case-bootstrap interval |
|---|---:|---:|
| Qualifying history-dependent answers | 22 target/seed observations across 14/64 cases | |
| Full-denominator rate | 0.9549% | [0.5208%, 1.4323%] |
| Withheld task-error difference at equal 50% retention, history-diverse minus question-diverse validation | -1.7447 percentage points | [-3.5346, +0.6504] points |

Ineligible targets remain in the full denominator. Seed rates are averaged
within each case; bootstrap draws resample whole cases across seeds and formats.
Seeds, answer formats, questions and repeats are repeated measurements within
cases. The fresh cases and original V10 cases remain separate populations.

Validation checks six consequence questions at one history or three questions
at each of two histories. Both rules share the same direct-check cost and withhold
the same three other questions per format. Decisions and rankings were locked
before withheld outcomes were analyzed. The named endpoint uses literal No/Yes
answers at 50% retention. Its interval spans zero, leaving the practical advantage
uncertain. Equal question budgets do not establish equal measured GPU cost.

Four positive cases were fully repeated. All 2,312 logical records had identical
scores and hard-answer/token bindings. The [complete Ada/Bryn example](../reproducibility/fresh_history_validation/examples/CASE_0017.md)
connects one failure to its exact inputs, controls and repeated responses.

The [frozen result table and per-seed accounting](../reproducibility/fresh_history_validation/README.md#canonical-result-table)
retain every seed, format and retention level. Later
[post-hoc diagnostics](../reproducibility/fresh_history_validation/README.md#post-hoc-diagnostics)
examine withheld-target misses and equal token inventories. They leave both
prospective endpoints unchanged.

## Saved-score replay

Use the [repository CPU environment](../REPRODUCIBILITY.md#run-the-saved-evidence-replay),
or the NumPy-only setup in the [frozen reproduction README](../reproducibility/fresh_history_validation/README.md#cpu-replay-and-equality).
From the checkout root, choose a new output directory:

```sh
python -m repro.fresh_history --output reproduced/fresh-history
python -m pytest tests/test_fresh_history.py tests/test_saved_replay_portability.py
```

Expected: `PASS`, 623 checked artifacts, 64 final cases, both seeds, and 2,312
strictly equal repeated logical records. The replay rebuilds primary results,
locked decisions, utility, diagnostics, tables and both SVGs from saved scores.
Classifications and identities must match exactly. Finite recalculated summary
values allow absolute error 1e-12; original scores and manifest hashes stay exact.
The [frozen equality rules](../reproducibility/fresh_history_validation/README.md#cpu-replay-and-equality)
also distinguish observed repeat equality from the original acceptance threshold.

The [scientific manifest](../reproducibility/fresh_history_validation/MANIFEST.json),
[source binding](../reproducibility/fresh_history_validation/SOURCE.json), original
README, worked example and frozen figures are unchanged. This reader guide sits
outside that hash-bound package so clearer explanations do not rewrite evidence.
