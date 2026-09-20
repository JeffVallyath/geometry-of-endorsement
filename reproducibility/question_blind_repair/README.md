# Question-blind partial repair and successful-case verification

A fixed refresh of stored values partially corrected history-dependent answers
in Gemma. The original prospective study established the effect. A later,
post-hoc-selected repeat reproduced actual repairs and reverse effects.

Current status, September 19, 2026: this computation branch is scientifically
closed with partial repair. The strong-repair threshold remains failed. This
entry supersedes earlier descriptions of this repair as unmeasured; dated
historical audits and the separate failed pre-update deletion screen are unchanged.

## Method

The input is an edited context plus a reference built from complete current
records using the same learned update. Before questions are opened, targeted
refresh replaces cached values at layers 16-23 and aligned trailing positions.
The comparison refresh uses layers 24-31 from the same reference. Keys, other
values, model weights and the recipient prompt stay fixed.

Each question is asked from an independent clone. We measure valid answers to
current facts and logical questions across eight starting histories. A qualifying
repair makes the target correct across every history while preserving the
required direct and other logical answers. Wrong consensus and loss of
qualification receive no repair credit.

Both partial refreshes require constructing the full current-record reference
before the questions. No compute or latency advantage was demonstrated.
The source protocol is
preserved in [ANALYSIS_RULES.md](supplied/primary/ANALYSIS_RULES.md).

## Prospective result

The targeted band repaired 36/83 baseline qualifying observations (43.4%) across
28 designs. The comparison band repaired 2/83. These observations are
target/seed/format measurements within the original 256 designs. The prespecified
50% strong-repair condition failed; the remaining conditions passed.

| Original accounting | Value |
|---|---:|
| Prospective designs | 256 |
| Baseline qualifying observations | 83 |
| Targeted / comparison repairs | 36 / 2 |
| Designs containing a targeted repair | 28 |
| Joint errors before / after targeted refresh | 3,490 / 2,893 |
| Joint errors under full reference | 2,000 |
| Joint-response denominator per arm | 73,728 |
| Targeted joint corrections / introduced errors | 944 / 347 |
| Targeted direct corrections / introduced errors | 84 / 76 |
| Original qualifying observations retained after treatment | 35 |
| Original observations losing qualification without repair | 12 |
| Newly qualifying observations after treatment | 40 |
| Total qualifying observations after treatment | 75 |

Joint-answer error fell by 0.81 percentage points overall. The cohort repair rate
and the total post-treatment qualifying count answer different questions.
The original cohort partitions into 36 repairs, 35 retained qualifying failures,
and 12 observations that cease to qualify. The 40 new qualifying observations
bring the post-treatment total to 75.

The original four-endpoint family and its multiplicity-adjusted 98.75% intervals
are preserved in [FOUR_ENDPOINTS.csv](supplied/primary/tables/FOUR_ENDPOINTS.csv).
Its efficacy contrast uses all scheduled target opportunities, not just the
83-observation baseline cohort. Intervals retain case-design resampling and the
original denominators. No uncertainty estimate is refit from selected repeats.
See the [primary summary](supplied/primary/replay/SUMMARY.json) and
[population figure](supplied/primary/figures/POPULATION_REPAIR_DAMAGE.svg).

## Post-hoc successful-case reproducibility

Fresh processes reproduced all retained score and binding fields exactly.
This extension deliberately selected successful cases after the primary outcomes.
It strengthens reproducibility of those cases, with zero new independent designs.

| Selected extension | Verified value |
|---|---:|
| Successful-case designs | 6 |
| Stored seeds per design | 2 |
| Separate technical-anchor designs | 1 |
| Total case/seed blocks, including anchor | 14 |
| Scored reads | 14,076 |
| Original qualifying observations in selected designs | 9 |
| Targeted / comparison repeated repairs | 8 / 0 |
| Selected reverse designs | 2 |
| Eligible targeted / comparison reverse observations | 3 / 0 |
| Maximum retained score difference | 0 |

Eight repairs means eight target/seed/format observations, not eight designs.
The selected fraction is not an updated efficacy estimate. The delivery status
`ALL_REQUIRED_GATES_PASS` concerns delivery and verification, not the failed
scientific strong-repair condition. All selected failures and introduced errors
remain in [SELECTED_RECOUNT.json](supplied/verification/SELECTED_RECOUNT.json).

The [worked example](WORKED_EXAMPLE.md) pairs the successful Yes/No illustration
with its failed A/B-format repair. It also explains why a correct donor can
produce a reverse error. This supports context-dependent causal influence;
the exact natural mechanism remains unresolved.

## Offline replay

Use the repository's configured Python environment, Python 3.11 or later.
The repair checker and delivered raw replay use only the standard library.
From the repository root:

```sh
python -m repro.question_blind_repair
python -m pytest -q -p no:cacheprovider tests/test_question_blind_repair.py
```

The compact check binds unchanged source bytes, reconciles primary counts and
intervals, and checks the selected tables and claim scope. Raw replay additionally
requires both original ZIPs. They remain external to normal Git:

```sh
python -m repro.question_blind_repair --verification-archive /path/to/QBR_REPAIR_VERIFICATION_RESULTS.zip --original-archive /path/to/QUESTION_BLIND_REPAIR_RESULTS.zip --output reproduced/question-blind-repair
```

Use a new output directory. The adapter verifies both outer digests and closed
manifests, both scientific capsules and the terminal exports. It then invokes
the delivered [standard-library replay](supplied/verification/tools/replay.py)
unchanged in a separate directory. Seven scientific products must match the
archived copies byte-for-byte:

| Products | Contents |
|---|---|
| `REPEAT_COMPARISON.json`, `SELECTED_RECOUNT.json` | Repeated bindings and selected-case accounting |
| `TARGETS.csv`, `FAMILIES.csv` | Target classifications and benefits/damage |
| `REVERSE_TARGETS.csv`, `DONOR_AGREEMENT.csv` | Reverse eligibility and donor/recipient answers |
| `WORKED_EXAMPLE.csv` | Independently asked example answers |

The adapter also checks the archived reference files against the original
terminal exports and recounts the already-reported primary classifications.
It preserves the original uncertainty family rather than recomputing it.
The output `VERIFICATION.json` records the checks and product hashes.

Only four frozen wall-clock fields are excluded from score/binding comparison:
`batch_seconds`, `qualification_seconds`, `preprocessing_seconds`, and
`compile_seconds`. Raw files retain exact hashes. The original numerical
tolerance stays at `1e-6`; the observed retained scores match exactly.
Answers, validity, tokens, masks and non-timing bindings are exact checks.
CPU receipt timestamps and local audit paths are provenance, not scientific products.

## External bindings and preservation

| External archive | SHA256 |
|---|---|
| `QUESTION_BLIND_REPAIR_RESULTS.zip` | `f95a0a20daca0bbc51fa8942b1d2ce03c275e9fca8d72abebd094454c10c7608` |
| `QBR_REPAIR_VERIFICATION_RESULTS.zip` | `590772eb9f4148a714bd7c22ce769925beb9890bdab0f81654a75934427ed188` |

[SOURCE.json](SOURCE.json) binds every compact member to its original archive.
[SOURCE_BINDINGS.json](supplied/handoff/SOURCE_BINDINGS.json) preserves the supplied
archive and capsule identities. The full raw replay is not self-contained in Git.
No model, credential or large ZIP is added to the repository.

The original [outcome-independent example figure](supplied/primary/figures/WORKED_EXAMPLE.svg)
and population figure remain byte-identical. The original repeat panel remains
distinct from the successful-case extension. This closeout performs saved-evidence
checks, not fresh inference or live-provider verification. Other studies and sealed
outcomes are outside its scope.
