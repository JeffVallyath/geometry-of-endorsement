# Paired-editor training and checkpoint selection

The paired-stage trainer, loss, calibration and checkpoint-selection source are
supplied for inspection. [RECIPE.json](RECIPE.json) records the explicit settings
and the ten retained weight selections. The synthetic [FIT](data/fit.jsonl) and
[CAL](data/cal.jsonl) records are the original bytes used by those runs.

## Source and execution boundary

The `.py.txt` files are inert source snapshots for inspection, not an installed
training package. The original entrypoint is
[`train_condition`](source/rtc7_train.py.txt). Its historical backend imports and
bundle layout have not been integrated into a supported public retraining command.
No source snapshot was used to train a model in preparing this release.

| Source snapshot | Role |
|---|---|
| [Paired trainer](source/rtc7_train.py.txt) | Gradient path, explicit optimizer, warm start, epoch loop and selected checkpoint |
| [Shared definitions](source/rtc7_common.py.txt) | Group order, writer boundary, feasibility and tie-breaking |
| [Calibration](source/rtc7_cal.py.txt) | Original calibration scoring and summary |
| [Loss](source/pair_loss.py.txt) | Full-vocabulary label loss and conditional two-label consistency |
| [Paired inputs](source/terminal_pairs.py.txt) | Exact retained pair/question functions; unrelated corpus generators omitted |

The retained function bodies are unchanged. Only source line endings were
normalized. Names in the snapshots describe their original dependency interfaces;
they do not certify that those interfaces are runnable in this release.

## What the records establish

Every retained condition completed one paired epoch with 384 optimizer steps.
Calibration selected the pre-paired checkpoint for Gemma's unrestricted
consistency procedure at seed zero. The other seven main learned selections use
paired epoch one. Two additional constrained-consistency seed-one selections are
canonical-replay dependencies, not extra main comparison procedures.

Warm starts initialize fresh AdamW moments. Exact continuation is a separate
operation in the original source: it loads an owned optimizer/RNG snapshot.
Those snapshots are neither included nor loaded here. Do not unpickle an untrusted
snapshot when adapting the reference code.

The [study guide](../../../docs/appendix/A05_learned_editors.md) explains the update
and selected weights. The [reproduction matrix](../../../REPRODUCIBILITY.md#reproduction-levels)
is authoritative for remaining fresh-execution limits.
