# Pre-update attention route: fixed screen failed

Disposition: **NO_ACTIONABLE_DIRECT_PREUPDATE_ROUTE**. The complete historical
admission panel failed its frozen gate. The study stopped without running any
of the 128 final cases or the conditional four-case repeats. This is a measured
scientific no-go, not an inference from the earlier technical failure.

## What was tested

Pinned Gemma-2-9B-it, both stored INV_PAIR_NLL seeds. During prefix construction,
K removed attention-value messages from all three addressed clauses to the
common trailing positions at layers 0 through 15. The existing post-block-15
setter was unchanged. The signed-permutation control used K's captured message
vectors, not answer scores. Hooks were removed before answering. No question,
gold answer, or clean-reference activation entered the intervention.

The resource-admission sample was the fixed 16 previously exposed cases:
14 prior positive cases and two hash-selected controls. Both seeds answered
all 34 questions in both preselected histories, with the prescribed references.
This enriched panel is not a fresh population sample. The 128 new final designs
and four separate technical designs were frozen before new neural results;
the final designs remained unrun.

## Gate and answer errors

| Frozen criterion | Observed | Disposition |
|---|---|---|
| K corrects at least 3 distinct positive cases while preserving controls | 3 | Pass |
| At least 1 such K correction for each seed | Seed 0: 0; seed 1: 3 | Fail |
| K corrects at least 2 more cases than the matched control | K: 3; control: 2; difference: 1 | Fail |
| Mean joint error falls relative to original N | 75 to 129 errors / 1,152 | Fail |
| New direct errors are at most 2 percentage points | 38 / 768 = 4.948% | Fail |

K corrected FHV1-FINAL-0033, -0050 and -0053 under the screen's bound-target
criterion, all with seed 1. The control corrected -0053 and -0060, also only
with seed 1. These are historical two-history corrections, not the unmeasured
eight-history final repair endpoint.

| Arm | Direct errors / 768 | Joint errors / 1,152 | Newly introduced joint errors / 1,152 |
|---|---:|---:|---:|
| Original N | 0 | 75 (6.510%) | 0 |
| Message removal K | 38 (4.948%) | 129 (11.198%) | 67 (5.816%) |
| Matched control SCTRL | 0 | 72 (6.250%) | 5 (0.434%) |

K corrected 13 previously erroneous joint responses but introduced 67 errors,
for a net increase of 54 responses, or 4.6875 percentage points. Opposes-query
errors were 1/256 for N, 27/256 for K and 1/256 for SCTRL. All scheduled N/K/SCTRL
panel responses were valid. Per-case/per-seed counts and invalids are retained
in SCREEN_DAMAGE.csv; all correction decisions are in SCREEN_CORRECTIONS.csv.

No final-sample bootstrap, baseline-W repair rate, or confirmatory intermediate
link estimate is reported. The technical cases exercised all four causal cells,
and their continuous scores are supplied in TECHNICAL_FOUR_CELLS.csv, explicitly
labeled technical-only. They do not substitute for the gated final study.
The requested repeated-positive final example does not exist on this branch.
The two figures instead show the complete historical screen, clearly labeled.

## Technical validity and recovery

The successful runtime completed all four technical cases per seed. The CPU
audit verified 3,072 technical hook records, 128 record-boundary checks across
technical and safety cases, and 2,048 exact signed-permutation norm certificates.
All checks used the actual treated suffix S. Maximum group-reduction RMS error
was 0.0018885 of applied message RMS, below the unchanged 0.02 criterion.
Complete native attention reconstruction passed both prescribed per-tensor
rounding limits; maximum absolute difference was 0.125, within its corresponding
native-scale bound. Maximum RMS difference was 0.0006671.

Actual post-setter record activations, source inputs, writer workspace and
outside-S attention rows remained exactly equal; nothing was clamped. The
same-code disabled hook reproduced 272 scored rows exactly. Sixteen clone/order
checks passed. Sixteen fixed short greedy checks had no first-token invalids
or disagreements with scoring; this limited subset is not a general generation
equivalence claim. Two archived cases replayed on both seeds with zero logp
discrepancy. Completed first-attempt technical scores also replayed exactly.

The first runtime attempt stopped before the safety panel because a diagnostic
incorrectly required bitwise equality of non-associative floating-point norm
reductions. The replacement certified exact per-node BF16 absolute-component
multisets and corrected a group-error diagnostic to S without relaxing its
tolerance or changing the intervention. Both failed-attempt exports are retained.
A later local coordinator import-shadowing error was fixed by binding the same
frozen scientific modules before the borrowed transport. No remote inference
process, scientific code byte, score, gate or budget was changed by that fix.
See provenance/INCIDENTS.md and provenance/ops/INCIDENTS_OPERATIONS.md.

## Scope of the conclusion

This fixed deletion procedure is not an actionable low-harm route under its
declared historical gate. The result does not disprove every upstream pathway,
invalidate earlier matched-history observations, identify a unique old-fact
code, or estimate prevalence on fresh cases. Removing these messages may also
remove useful computation; the current screen cannot separate that possibility
from an incomplete or non-useful proposed bypass. No intervention search,
weakened gate, replacement pilot, or successor experiment was started.

## Custody, resource use and shutdown

All archive members and capsule bindings passed verification. All 32 safety
case/seed blocks retain indexed native/treated boundary activations and mediator
values, comprising 1,920 arrays. Sparse hook/state summaries and every raw scored
response are retained. Technical cases have full scores and sparse captures,
not full attention archives. No pretrained model weights are in this package.

Across both attempts: 27,428 scored-read reservations and 841 prefix/capture/
generation forwards, including failed work exactly once. Conservative rounded
usage is 97 instance-minutes, with a compute estimate of $3.2172. Adding the
fixed stage-one ancillary reserve gives $5.2172 against the $12 stage-one cap.
This is a reserved estimate, not a reconciled provider invoice; the unused
whole-study ancillary reserve is documented separately, not charged per attempt.

First allocation was 2026-09-19 03:40:02Z. Both replacement instances were
provider-confirmed absent by 04:40:51Z, well before the original cutoff.
All four owned IDs across both attempts are absent. Only this study's closed
guard tasks were disabled; their files and one-command rollback were preserved.
No other instance, protected concurrent outcome, paper file, commit or push
was touched. See COST_AND_SHUTDOWN.json and GUARD_CLOSEOUT.json.

CPU replay instructions are in README.md. The external-review packet is ready;
no automatic successor is authorized.
