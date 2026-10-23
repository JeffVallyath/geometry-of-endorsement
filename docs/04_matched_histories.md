# 4. Matched histories

<a id="sec:matched-histories"></a>

Different earlier histories can change a downstream answer after a model correctly
reports the same current facts. The experiment tests **source-history sufficiency**:
with current facts and the question fixed, the answer should be independent of the
earlier history. This is a behavioral criterion, not a claim about separate physical
stores inside the model.

[Figure 1: matched-history design](../figures/main/matched_history_design.pdf) · [Figure 4: prospective results](../figures/main/matched_history.pdf)

[Walk through a measured example](../reproducibility/fresh_history_validation/examples/CASE_0017.md):
Ada and Bryn's updated facts and surrounding logical checks are correct, yet the
same downstream question receives different answers under two matched histories.
The walkthrough includes its selection and repeat qualifications and complete records.

## Method

A case specifies a fictional meeting record and three facts to update. All eight
starting assignments receive the same three commands. Every history then has the
same intended final record, including facts that were not updated. Two learned
activation editors and two textual procedures are tested on frozen Gemma-2-9B-it
and Qwen3-8B backbones.

Each completed update receives the same 34-question battery. Nine joint questions
combine pairs of facts and appear in two answer formats. Questions are introduced
only after the update; each reads a fresh copy of the updated context.

A joint opportunity is **eligible** when both required facts are read correctly
under all eight histories and two references answer the joint question correctly.
One reference receives the final facts from the start. The other applies the same
update to an already-correct history. An eligible opportunity is a **witness** when
valid joint answers differ across histories. Ties or invalid answers cannot create
a witness.

## Prospective result

| Model | Procedure | Witness rate | 99.375% interval | Affected cases |
|---|---|---:|---:|---:|
| Gemma | Constrained learned editor | 6.337% | [4.514, 8.290]% | 50/64 |
| Gemma | Consistency-trained editor | 5.122% | [3.526, 6.771]% | 41/64 |
| Gemma | Existing textual correction | 5.990% | [3.472, 8.920]% | 30/64 |
| Gemma | Latest-value wording | 5.556% | [2.951, 8.594]% | 25/64 |
| Qwen | Constrained learned editor | 2.474% | [1.172, 4.123]% | 24/64 |
| Qwen | Consistency-trained editor | 2.387% | [1.172, 3.993]% | 25/64 |
| Qwen | Existing textual correction | 6.163% | [3.841, 8.594]% | 38/64 |
| Qwen | Latest-value wording | 6.337% | [4.080, 8.854]% | 40/64 |

Table note: rates use the parameters selected by the frozen calibration rule.
Gemma's consistency-procedure seed 0 selected the pre-paired-training checkpoint;
the other seven main learned selections use paired-stage epoch 1. All eight
learned runs completed the paired epoch. See
[training and checkpoint selection](appendix/A05_learned_editors.md#objectives-seeds-and-actual-selection).

All eight adjusted intervals exclude zero. Each case's rate uses all 18 scheduled
joint opportunities, including ineligible ones. Learned-editor seed rates are
averaged within each case; a case is affected if either seed has a witness. That
union gives learned procedures more search opportunities than a single textual
condition. It is not an equal-budget prevalence comparison.

Intervals resample the 64 cases, preserving measurements within cases. The eight
comparisons use 10,000 draws and the frozen multiplicity adjustment. These estimates
describe the controlled case-generating procedure, not deployment frequency.

Full-precision results and all individual-condition counts are in
[the primary table](../reproducibility/state_sufficiency/v10/primary_results.json),
[case counts](../reproducibility/state_sufficiency/v10/root_counts.json), and
[strong-margin results](../reproducibility/state_sufficiency/v10/witness_summary.json).

## Stronger controls on the original outputs

<a id="sec:matched-histories-posthoc"></a>

The post-hoc analysis matches histories on both changed-fact count and exact prefix
length. It then requires correct logical answers on the target relation pair,
other pairs, or both. Pooled over both backbones, all four update procedures and
both learned-editor seeds, the complete conjunction retains 64 coded opportunities
in 29 of 64 cases; 60 of the 64 come from learned editors and 4 from textual
procedures.
The 29-case result is not length matching alone.

[Repository topic A06](appendix/A06_stronger_controls.md) gives the decomposition, correlated
counting units, cross-format rule and fresh-study distinction.

## Fresh confirmation and held-out utility

<a id="sec:fresh-confirmation"></a>

A separate 64-case Gemma cohort fixed the stronger criterion before its outcomes.
Both stored seeds of the constrained editor were used without refitting. The
criterion retains 22 target/seed observations in 14 cases: a case-averaged rate of
0.9549%, with a 95% interval of [0.5208%, 1.4323%].

A separate utility test compares six validation questions at one history with
three questions at each of two histories. At equal 50% retention, the named
withheld-error contrast is −1.7447 percentage points, with a 95% interval of
[−3.5346, +0.6504]. Practical advantage remains uncertain. The confirmation and
utility outcomes answer different questions.

[Appendix Figure 6](../figures/main/fresh_confirmation_utility.pdf), in manuscript
Appendix D.2, shows both outcomes.
[Complete fresh control counts](../reproducibility/fresh_history_validation/tables/PRIMARY_GATE_COUNTS.csv)
and [utility sensitivities](../reproducibility/fresh_history_validation/tables/UTILITY_EQUAL_RETENTION.csv)
retain all seeds, answer formats and retention levels.

## Stateless 32B model breadth

<a id="sec:qwen32-breadth"></a>

Qwen3-32B was measured on the existing 64 case designs. Each read used a fresh
full-prefix-plus-question forward pass through the unchanged backbone. No stored
prefix cache, appended answer or activation intervention was used.

| Textual update | Basic witnesses / scheduled | Affected cases / 64 | Fully certified semantic opportunities (cases) | Strongest two-format witnesses / 576 |
|---|---:|---:|---:|---:|
| Correction sentence | 128 / 1,152 | 43 | 80 (9) | 0 |
| Restated current values | 123 / 1,152 | 49 | 80 (9) | 0 |

Basic recurrence extends to this checkpoint. The strongest criterion also requires
all direct and other logical checks, correct references, and the same matched
history pair with the same disagreement direction in both answer formats.
Neither method has a witness under this criterion. Both retain a nonempty fully
certified set, so zero witnesses is not an empty-set result. The 576 scheduled
semantic opportunities are a different denominator from that certified subset.

The saved repeat checks cover 4,488 reads, including four complete selected cases.
Their token bindings and scores agree exactly. The
[worked example](../reproducibility/qwen32_full_forward/WORKED_EXAMPLE.json) passes
the single-format controls but fails the other-format reference. Its selection is
post hoc.

This is model breadth on existing designs, not independent case replication.
[Repository topic A07](appendix/A07_textual_breadth.md) separates the smaller-model results,
historical bridge failures and earlier unmeasured cached 32B row.

## Reproduce the supported results

Run from the repository root after installing the CPU dependencies. Each output
directory must be new.

```sh
python -m repro.state_sufficiency --output reproduced/primary
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
python -m repro.cpu_robustness --output reproduced/controls
python -m repro.fresh_history --output reproduced/fresh
python -m repro.qwen32_full_forward --output reproduced/qwen32
```

The first command reconstructs statistics from case counts. The independent
calculation reconstructs semantic answers, eligibility and witnesses from saved
scores before opening expected results for comparison. The remaining commands
reconstruct the post-hoc controls and fresh study. None generates new model outputs.

The independent comparison checks eight primary cells, 896 case-condition rows and
16,128 joint-query rows. The control audit checks 13,824 updating opportunities
and 608,256 Boolean flags. The fresh replay reconstructs its primary and utility
analyses and the four fully repeated cases.
The 32B command reconstructs all 36,992 final score records, checks 27,136
classification fields with independent code, and verifies the saved repeats.
