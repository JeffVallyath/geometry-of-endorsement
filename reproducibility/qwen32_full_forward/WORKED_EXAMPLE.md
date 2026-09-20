# A repeated history-dependent answer in the Yes/No format

Qwen3-32B reports that Mina and Wren oppose Gallery, yet says that both support
it after one starting history. The other compared history gives the correct
joint answer. Their intended current facts and question are identical.

This example qualifies only in the Yes/No format. The other format fails its
already-correct target reference and gives the same answer under both histories.
It therefore remains outside the strongest two-format result.

## What changes and what stays fixed

One starting history has Mina supporting Gallery; the other has Ivo supporting
it. Wren opposes it in both. Each history receives a textual restatement leading
to the same current facts. Each changes one designated fact and has 834 prefix
tokens. The model, subsequent question and answer interface stay fixed.

The table shows independently asked questions after that update. A qualifying
failure has correct factual and logical controls but different valid target
answers across the matched histories.

| Question | Correct | History 1 | History 2 |
|---|---|---|---|
| Does Wren favor the Gallery proposal? | No | No | No |
| Does Mina favor the Gallery proposal? | No | No | No |
| Do Mina and Wren take the same side on Gallery? | Yes | Yes | Yes |
| Does at least one of Mina and Wren support Gallery? | No | No | No |
| Is the Gallery proposal supported by both Mina and Wren? | No | No | Yes |

All 12 direct checks pass across eight histories and both formats. The eight
other logical questions and both target references pass in Yes/No.

| Target measurement | History 1 | History 2 |
|---|---:|---:|
| Normalized correct-answer score | 0.9983893724 | 0.04365346482 |
| Candidate answer mass | 0.9999828285 | 0.9999966971 |

These normalized interface scores are not calibrated beliefs.

The old target truth is No in both starting histories. The discrepancy therefore
shows dependence on history, rather than literal repetition of the old target truth.

## Exact source binding

The saved design is `SSC1-FINAL-0003`, procedure `LATEST_SAME_WORDING`, target
`SSC1-FINAL-0003-d0-09`, histories 1 and 2. The second-format target is
`SSC1-FINAL-0003-d1-09`.

[WORKED_EXAMPLE.json](WORKED_EXAMPLE.json) contains the reconstructed table,
scores, qualification checks and exact question IDs. The raw source is
[the final score ledger](saved/results/runtime/results/qwen32_t4/final/SSC1-FINAL-0003/RAW_SCORES.jsonl.gz),
with [the full repeat](saved/results/runtime/results/qwen32_t4/repeat_full/SSC1-FINAL-0003/RAW_SCORES.jsonl.gz)
and [the input design](saved/results/runtime/cpu/designs/SSC1-FINAL-0003.json.gz).

All score/token bindings in the 578-read repeat match the main run. Selection
of this example for explanation is post hoc; it creates no new evidence. The
single-format analysis and repeat-selection rule were fixed before these outcomes.
The [CPU replay](README.md#reproduce-every-retained-result-on-cpu) rechecks the
entire example from raw rows without another model call.
