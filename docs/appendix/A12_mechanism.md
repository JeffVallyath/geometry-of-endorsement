# Selected-case carrier and mediation evidence

Repository topic A12 contains additional carrier-reader and mediation analyses
related to manuscript Sections 6 and 8. These are distinct from the crossover and
refresh studies in Appendix E and have no separate manuscript appendix section.

## Question and shared setup

Transplanting stored values transfers joint answers on one selected history pair
while preserving direct factual reads. Follow-up interventions recover parts of
the transfer pathway. Compact-carrier search, reserved-pair qualification and
reader-set qualification each have important failures.

The discovery input is case `SSC1-FINAL-0035`, histories 3 and 5, with six original
questions and 517-token prefixes. Both studies use `google/gemma-2-9b-it`, revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`, and the original constrained writer
`INV_PAIR_NLL_s1`. The backbone uses native BF16 and the writer uses FP32.
Model weights, question text and answer mappings stay fixed.

Two directions and two answer formats give four joint-answer cells. They are
repeated measurements on the selected pair. The two additional reserved pairs
were also behavior-known before the carrier study; they were excluded from head
ranking. None of these sets is a population sample.

An answer transfer means adopting the opposite donor's joint answer. It can create
an error when the donor is wrong. Complete transfer requires four valid donor
answers with candidate mass at least 0.90 and eight preserved direct checks.
A direct check requires a valid correct answer, normalized correct-label score
at least 0.95, candidate mass at least 0.90, and recipient-score drift at most 0.02.
Scores are normalized over the two fixed labels; candidate mass is reported
separately.

## Carrier-reader study

The intervention copies donor value-cache entries before introducing a question.
The selected carrier uses layers 16 through 23, all eight native key/value groups,
and positions 426 through 516. Layer, head and position indices are zero-based;
ranges are inclusive. Keys and unselected recipient entries stay unchanged.
This footprint contains 5,824 native vectors of 256 BF16 values each.

Four parent footprints were tested in a fixed order. All passed the four-cell
criterion, so the order selected the value-only footprint named P1. Other parents
add keys or include the shared prefix positions 0 through 333. Complements are
exact differences within the broader shared-position key/value footprint at
layers 16 through 41. All four complements produce zero hard transfers.

| Test | Joint transfers / 4 | Preserved direct checks / 8 | Qualification |
|---|---:|---:|---|
| Selected discovery carrier | 4 | 8 | Complete transfer |
| Reader outputs blocked | 0 | 8 | Transfer removed |
| Carrier-active reader outputs injected | 4 | 8 | Transfer restored |
| Reserved case `SSC1-FINAL-0026`, histories 2/7 | 2 | 8 | Fails complete transfer |
| Reserved case `SSC1-FINAL-0054`, histories 1/4 | 1 | 8 | Fails complete transfer |

No individual site passes the complete criterion in the 64-site scan. The ranked
top-1, top-2, top-4 and top-8 groups transfer 0, 1, 2 and 3 joint answers when used
alone. The fixed random controls and leave-out tests are retained in the
[complete source tables](../../reproducibility/completed_followups/carrier/TABLES.md).
The selected carrier therefore remains the unreduced P1 footprint.

A later positional diagnostic transfers all four answers using only positions
426 through 455. It copies 1,920 native vectors and preserves all eight direct
checks. This diagnostic did not replace the already frozen carrier. The other
thirds, 456 through 485 and 486 through 516, transfer no hard answers alone.
The earlier-position condition is empty under the fixed carrier rule and is
recorded as an omission.

Reader interventions act on attention-head outputs before the output projection.
They cover every question-suffix position before an answer token is supplied.
Each selected native key/value group supplies two query heads. For the unreduced
carrier, the broad reader check uses all 16 query heads in layers 16 through 23.
Passive captures and same-state copy controls retain their own results.

All 1,416 non-baseline direct rows pass preservation. The score table also retains
32 baseline/native-reference direct rows, for 1,448 direct rows in total.
There are 179 transitions from a correct recipient joint answer to an incorrect
one. Both the [adverse ledger](../../reproducibility/completed_followups/carrier/ADVERSE_AND_OMISSIONS.json.gz)
and [all 2,172 score rows](../../reproducibility/completed_followups/carrier/ALL_SCORES_AND_EFFECTS.csv.gz)
remain available. Transfer success is therefore distinct from improved accuracy.

## Writer and reader mediation follow-up

This follow-up reuses the discovery pair and the frozen full carrier. Writer
interventions are restricted to the 30-position region 426 through 455. They clamp
the actual attention or MLP residual additions after Gemma's corresponding
post-layer normalization. The scan covers 16 component sites with fixed controls,
for 39 arms.

Copying the small shared-region residual state after block 15 transfers all four
joint answers. The full 16-component donor-output bank transfers two. Selection
chooses the layer-16 attention contribution alone, which transfers none and
preserves all eight direct checks. Its weakest normalized continuous joint effect
is 0.087708. Layer-23-only interventions leave the measured layer-16-through-23
value carrier unchanged, providing the structural negative control.

Reader ranking tests 128 individual query heads. It uses only the literal-format
history-3 recipient with history-5 donor. The ranking score is the smaller of the
normalized blocking loss and injection gain. The top-1, top-2, top-4 and top-8
sets are frozen before coded and reverse-direction checks. The case and its
earlier outcomes were already known.

No set reaches the fixed 0.8 training threshold. The top-8 fallback scores
0.777757 and retains `NO_QUALIFIED_READER_SET`. Its blocking intervention removes
all hard transfers, while injection produces four. Continuous effects remain
partial: blocking removes 71.2% to 95.0% of the carrier-directed margin across
the four cells; injection supplies 75.2% to 89.4%.

The [full result tables](../../reproducibility/completed_followups/mediation/TABLES.md)
retain each direction, answer format and signed effect. The frozen eight-head set
is `(19,15), (18,3), (23,12), (20,11), (21,0), (23,14), (17,0), (21,9)`.

## Loss and restoration controls

| Condition | State or manipulation | Joint donor answers / 4 |
|---|---|---:|
| A | Native recipient | 0 |
| B | Selected writer contribution | 0 |
| C | B with recipient carrier values restored | 0 |
| D | B with the original donor carrier injected | 4 |
| E | Original carrier transfer with the selected reader blocked | 0 |
| F | Carrier-active outputs restored at those same reader sites | 4 |

Every condition preserves all eight direct checks. Restoring the recipient carrier
removes 41.7% to 84.7% of the selected writer's continuous joint effect. Residual
effects remain. Condition D injects the original donor carrier, not a reconstruction
of the carrier produced by the weak writer intervention. Condition F deliberately
undoes a clamp at the same site.

All 948 unique direct rows pass preservation, with no invalid scores. There are
61 new joint-answer errors. The
[adverse ledger](../../reproducibility/completed_followups/mediation/ADVERSE_AND_OMISSIONS.json.gz)
retains new joint errors, negative effects and omitted work. Signed and undefined
fractions remain unaltered. Tensor-direction projections describe alignment;
they are not percentages of a unique mechanism.

Together these results support partial functional mediation of the selected
transplant effect. A unique or minimal natural mechanism, natural-path necessity,
and generalization to other case populations remain unestablished.

## Reproduce the supported checks

```sh
python -m repro.completed_followups --output reproduced/mechanism
```

The command verifies the public manifests and checks exact score-row accounting,
the direct-preservation rule, adverse joint changes and failed reader qualification.
The source report's tensor-verification statements remain labeled historical
reports. Native caches, producer tensors, reader tensors and their execution code
are absent from the supplied compact subset. This route does not reconstruct the
interventions, refit a mechanism or execute a model.
