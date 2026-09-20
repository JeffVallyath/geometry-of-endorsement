# Qwen3-32B: history dependence with correct current facts

Qwen3-32B gave different joint answers after different histories reached the
same intended current facts. This occurred under both textual update procedures.
The strongest test requiring agreement between both answer formats found no
qualifying comparison. A narrower Yes/No test retained three comparisons across
two designs, including an already-repeated high-margin example.

## Method

The inputs are the 64 original synthetic designs, newly measured on this
checkpoint rather than newly sampled. Each design has eight starting histories.
A correction sentence or a restatement of current values updates each history.
The intended final facts, later question and model weights stay fixed.
We measure whether valid answers to the same joint question differ by history.
A primary comparison requires correct current-operand reports under all histories.
Both its current-facts-only reference and its already-correct update reference
must answer the target correctly.

The model is `Qwen/Qwen3-32B`, revision
`9216db5781bf21249d130ec9da846c4624c16137`, with thinking disabled.
Each read processes the complete prompt independently at batch one, with
`use_cache=False` and no supplied or returned key/value cache. The backbone uses
BF16 and the full-vocabulary scoring head uses FP32. This completed path is
separate from the earlier failed cached path.

## Canonical result table

| Measurement | Correction sentence | Restated current values |
|---|---:|---:|
| Saved procedure name | `EXISTING_CORRECTION` | `LATEST_SAME_WORDING` |
| Original designs measured | 64 | 64 |
| Primary target/format comparisons | 128/1,152 | 123/1,152 |
| Primary rate | 11.11% | 10.68% |
| Inherited 99.375% design-bootstrap interval | [7.75%, 14.50%] | [7.47%, 14.32%] |
| Designs with any primary comparison | 43 | 49 |
| Current-operand/reference coverage | 70.23% | 69.01% |
| Individual-fact accuracy | 97.53% | 97.53% |
| Joint-question accuracy | 77.84% | 78.35% |
| Matched full-logical Yes/No comparisons | 1 | 2 |
| Strongest two-format comparisons | 0/576 | 0/576 |
| Fully logically certified semantic opportunities | 80 | 80 |
| Designs contributing those certified opportunities | 9 | 9 |

The primary denominator retains every scheduled target/format opportunity,
including those failing qualification. Histories, questions and formats are
paired within designs. Intervals resample whole designs; repeats are not added
to the population. [CLAIMS.json](CLAIMS.json) preserves the unrounded estimates.
The [full source summary](saved/results/runtime/analysis/SUMMARY.json.gz) retains
all descriptive intervals, control counts and original zero-event bounds.

The strongest test requires all direct facts, both target references and every
other logical question to be correct. It then requires the same matched history
pair to disagree in the same direction under both answer formats. Only the
reported certified opportunities reach this last step. Their zero count is a
result for this test, not evidence of general history independence.

The conditional zero-event upper bound is 28.31% over nine certified designs,
under the original independent/exchangeable-design assumption. The 80 correlated
opportunities are not 80 independent trials.

## Worked example and repeatability

The [Mina/Wren example](WORKED_EXAMPLE.md) passes the narrower Yes/No logical
checks and the high-margin rule. Its second format fails qualification because
its already-correct reference answers the target incorrectly. That failure is
shown beside the positive comparison.

All 4,488 retained repeat comparisons have identical candidate log probabilities
and answer/token bindings. These comprise 2,176 fixed-panel reads and 2,312 reads
from four complete repeated designs. All four complete classifications agree.
The example's 578-read block was already repeated; no new model run is needed.

## Reproduce every retained result on CPU

From the repository root, use the existing CPU environment or install the
repository's pinned replay requirements in a virtual environment:

```sh
python -m pip install -r reproducibility/requirements-replay.lock -e .
python -m repro.qwen32_full_forward --output reproduced/qwen32
python -m pytest tests/test_qwen32_full_forward.py
```

Choose a new output directory. The main command works without the external ZIP,
network, model weights, tokenizer downloads or a provider account. It verifies
the compact members, restores their exact original bytes into the output area,
and reuses the unchanged source analyzer, source replay and supplied independent
checker. The wrapper rejects missing repeats and any scientific discrepancy.

The output contains:

- All 64 recomputed case records, the complete source summary and intervals.
- Both original CSV tables, reproduced byte-for-byte.
- The verified example table and its failed second-format condition.
- Repeat checks for every retained comparison and all full-case classifications.
- Both readable figures, their plotted values and the original figure data.
- `CLAIMS.json` and `VERIFICATION.json`, including numerical tolerances.

Scientific identities and classifications require exact agreement. Only finite
derived floating-point summaries allow absolute error at most `1e-12`, with no
relative tolerance. Original score/token repeat equality remains exact.
The labels in the readable figures are translated into ordinary language; the
source plotting calculations and values are unchanged.

This is saved-score reconstruction, not fresh neural replication. The original
execution environment and amendment are preserved in the source-bound records.
The original archive retains the complete execution code and historical launch
commands; those commands are provenance, not authorization to run another model.

## Integrity and external archives

[SOURCE.json](SOURCE.json) maps every compact payload to its original archive
member, byte count and SHA256. Large JSON/CSV records use lossless gzip;
decompression recovers the original bytes and line endings. Frozen manifests,
raw final/repeat files, prompt inputs, model pins and supplied checkers are unchanged.

The original `QWEN3_32B_FULL_FORWARD_RESULTS.zip` remains external to normal Git.
Its size is 39,352,811 bytes and its SHA256 is:

`5777db17c80cb0145207c447a0303d395f309e045f17f4b2e6d84157a294fca3`

To verify that full archive as well:

```sh
python -m repro.qwen32_full_forward --archive /path/to/QWEN3_32B_FULL_FORWARD_RESULTS.zip
```

This checks all 1,060 declared members. The immutable execution capsule is
`b8b523be57e3058868ea23539c61778024022a903a192d942e58a8bc9a9e70e4`;
all 372 capsule members were checked during intake. The compact package retains
the capsule and the subset needed for scientific replay, not provider operations.

## Claim boundaries and earlier evidence

The [claim map](CLAIM_MAP.md) separates basic recurrence, logical controls,
repeatability and causal repair. The [portfolio record](portfolio/README.md)
documents the earlier smaller-model checks and the five distinct checkpoints.
Gemma appearing under two prompt forms still counts as one checkpoint.

The previous missing Qwen32 row, cached/full-forward failure and failed historical
numerical bridges remain in the source records. The completed standalone row
does not repair those bridges or establish a size-scaling relation. Causal repair
and moral-decision effects require their own evidence; no such result is added here.

[Integration notes](INTEGRATION_NOTES.md) distinguish portability repairs from
scientific results. [Figure captions](FIGURES.md) explain the two visualizations.
