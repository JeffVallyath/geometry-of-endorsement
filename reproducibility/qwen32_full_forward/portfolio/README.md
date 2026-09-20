# Checkpoint portfolio and historical boundaries

Basic textual-update history dependence is supported in five distinct checkpoints
from three families, spanning 4B to 32B. This is a coverage statement, not a
size-scaling comparison. Different templates on the same Gemma checkpoint do
not add another model.
The model rows share the original design set; they are not independent samples
of task designs.

| Checkpoint | Evidence retained here |
|---|---|
| Qwen3-4B | Earlier completed archive; both textual procedures independently reconstructed |
| Qwen3-8B | Existing [original prospective records](../../state_sufficiency/v10/primary_results.json) and their saved-output replay |
| Meta-Llama-3.1-8B-Instruct | Earlier completed archive; both textual procedures independently reconstructed |
| Gemma-2-9B-it | Original prospective records and the earlier completed archive; counted once across prompt forms |
| Qwen3-32B | The new standalone [full-forward package](../README.md) |

[PORTFOLIO_CHECK.json](PORTFOLIO_CHECK.json) records reconstruction of 110,976
earlier FINAL score records. It verifies all six primary counts and their
contributing-design counts for the three newly measured smaller-model rows.
The original [table](MAIN_TABLE.json), [model roster](MODEL_ROSTER_FROZEN.json),
[row dispositions](ROW_DISPOSITIONS.json) and
[planned-comparison status](PLANNED_COMPARISONS.json) retain their original bytes.

In that earlier table, Qwen3-32B remains `NOT_MEASURED`. The new full-forward
measurement is a separate record, not an overwrite of that technical disposition.
Both historical numerical bridges remain failed. No causal repair or
moral-decision evidence is imported from these records.

## Recheck the earlier rows

The earlier `V10_TEXT_SCOPE_RESULTS.zip` stays external to normal Git. It is
134,523,552 bytes, with SHA256
`eb5bee9d247bdfbea59ad894053a4b2054a0d605a898e008b889517c614b7441`.
All 2,578 declared members were checked before using its completed outcomes.

```sh
python reproducibility/qwen32_full_forward/tools/verify_portfolio.py --archive /path/to/V10_TEXT_SCOPE_RESULTS.zip --output reproduced/qwen32-portfolio
```

That command needs only Python's standard library. It rechecks the archive and
independently reconstructs primary qualification and history disagreement from
the saved scores and designs. It does not rerun models or claim that the older
cross-model numerical bridges passed.

For the earlier archive's complete descriptive statistics and intervals, extract
it into a new directory and use its original Python/NumPy replay from `code/`:

```sh
python -m text_scope.analyze --runtime ../runtime --output ../fresh_cpu_replay
```

The new 32B command is standalone without either external archive. Rechecking
the entire earlier portfolio requires the earlier archive, as disclosed here.
