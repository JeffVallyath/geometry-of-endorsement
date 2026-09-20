# Completed textual breadth

Basic history dependence recurs under both correction sentences and restated
current facts in the three newly measured smaller-model rows. The strongest
logical and cross-format checks are separate outcomes, not synonyms for this
basic recurrence. Failed numerical bridges prevent the planned comparisons
with historical rows.

## Inputs, changes and measurement

The study reuses 64 existing synthetic designs. Each contains eight histories
that receive the same intended final assignments. Model weights, final facts
and later questions stay fixed. A target counts when all current-operand
reports and both references are correct, yet valid joint answers differ across
histories. Each method has 1,152 scheduled target/format opportunities, including
those failing qualification. Whole designs are the resampling units.

## Breadth alongside the separate full-forward study

| Newly measured checkpoint/path | Correction: basic count | Restatement: basic count | Interpretation |
|---|---:|---:|---|
| Gemma-2-9B, cached T4 template | 104/1,152 | 93/1,152 | Completed breadth row |
| Llama-3.1-8B, cached T4 template | 61/1,152 | 68/1,152 | Completed breadth row |
| Qwen3-4B, cached T4 template | 115/1,152 | 110/1,152 | Completed breadth row |
| Qwen3-32B, cached T4 template | Not measured | Not measured | Cached/full DEV mismatch, not zero |
| Qwen3-32B, separate stateless full-forward path | 128/1,152 | 123/1,152 | Strongest two-format result: 0/576 per method |

The final row belongs to the [separate 32B package](../qwen32_full_forward/README.md),
not a repaired cached run. It newly measured the same existing designs, not
newly sampled designs. Its narrower repeated [Yes/No example](../qwen32_full_forward/WORKED_EXAMPLE.md)
fails qualification in the second format. This table is not a scaling curve.

The source retains four historical method rows in their own original uncertainty
family. The six new measured method cells keep the planned eight-cell 99.375%
family despite the missing cached 32B row. All six planned paired contrasts
remain unassessable. There is no supported template-superiority or size-trend
estimate. The DEV gate was amended after DEV and before new FINAL outcomes;
this was not an unchanged preregistration. Original discrepancies remain.

## Offline replay and provenance

```sh
python -m repro.text_scope --output reproduced/text-scope
```

The command checks every compact member, restores original scientific bytes,
then runs the original CPU analyzer without network or model imports. It
reconstructs the new and historical rows, sensitivity outputs, planned
comparisons and missing-row dispositions. Integers and classifications stay
exact; finite derived floats use the existing absolute 1e-12 tolerance.
Original input hashes are not subject to that tolerance.

[text_scope_SOURCE.json](text_scope_SOURCE.json) maps compact storage to every
original member and digest, including the original report, source analyzer,
saved FINAL/repeat/DEV records, inputs, comparisons and technical failures.
Stored gzip is lossless; short storage paths do not change original member names.
The original archive remains external:

`V10_TEXT_SCOPE_RESULTS.zip`, SHA256
`eb5bee9d247bdfbea59ad894053a4b2054a0d605a898e008b889517c614b7441`.

The original 2,578 declared payloads were verified at intake. Compact replay
does not rerun neural inference or independently attest the earlier provider.
Operational privacy derivatives, if listed, have distinct documented hashes
in the [privacy record](../../docs/PRIVACY_BOUNDARY.md).
