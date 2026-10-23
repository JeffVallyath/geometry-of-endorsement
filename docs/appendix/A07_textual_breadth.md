# Smaller-model textual breadth

Repository topic A07 contains additional smaller-model studies related to
manuscript Section 4.4 and Appendix D.3. For the paper's current stateless 32B
results and Table 4, use the [32B study guide](../04_matched_histories.md#sec:qwen32-breadth).

## Question and method

Basic history dependence appears in all six newly measured smaller-model cells.
These cells combine three checkpoints with two textual update procedures.
The strongest matched, logical and two-format conjunction survives only for Gemma.

The inputs are the primary study's existing 64 synthetic cases. Each case has
eight histories, three fact-update commands and 34 questions. The questions include
nine joint questions under each of two answer mappings. One procedure appends a
correction sentence; the other restates the current values in the same wording.
Weights, questions and the shared case population stay fixed within each cell.

The measurement separates eligibility from disagreement. Correct operand reads
across the eight histories and correct native/current-facts and no-op references
make a joint opportunity eligible. A valid answer disagreement across histories
then makes it a basic witness. All 1,152 scheduled opportunities per cell remain
in the primary denominator, including ineligible opportunities.

The new rows use native chat templates, including instruction-tuned Gemma and
Llama variants. Qwen thinking is disabled. Exact model and tokenizer revisions are
in the [frozen roster](../../reproducibility/text_scope/saved/runtime/MODEL_ROSTER_FROZEN.json).

| Checkpoint | Revision |
|---|---|
| `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |

## Primary results

Intervals resample whole cases using NumPy PCG64, seed 2609141002 and 10,000 draws.
The 99.375% family covers all eight planned new method cells, including the two
unmeasured cached 32B cells. Missing rows do not reduce that family.

| Checkpoint | Update | Eligible / 1,152 | Basic witnesses / 1,152 | Primary rate | 99.375% interval |
|---|---|---:|---:|---:|---:|
| Gemma-2-9B-it | Correction | 971 | 104 | 9.0278% | [5.9896%, 12.5868%] |
| Gemma-2-9B-it | Restatement | 963 | 93 | 8.0729% | [5.4687%, 11.0243%] |
| Llama-3.1-8B-Instruct | Correction | 602 | 61 | 5.2951% | [3.4722%, 7.3570%] |
| Llama-3.1-8B-Instruct | Restatement | 604 | 68 | 5.9028% | [3.9931%, 7.8125%] |
| Qwen3-4B | Correction | 776 | 115 | 9.9826% | [6.4236%, 13.7153%] |
| Qwen3-4B | Restatement | 810 | 110 | 9.5486% | [6.0764%, 13.2598%] |

The complete replay table also includes four historical method rows. Those use
Gemma-2-9B-it with its earlier prompt construction and Qwen3-8B. It preserves both
unmeasured cached Qwen3-32B rows with null rates and explicit missing-row statuses.

## Stronger-control accounting

Matching requires equal changed-fact counts and equal prefix token counts.
The full logical conjunction additionally requires all direct, same-pair and
cross-pair checks. The two-format test requires the same history pair and the
same disagreement direction under both mappings. The single-format denominator
is 1,152; the two-format denominator is 576 distinct joint questions.

| Checkpoint | Update | Matching alone | Full logical eligible | Matched + full logical witnesses | Cases with those witnesses | Full logical two-format eligible | Two-format witnesses |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-2-9B-it | Correction | 67 | 227 | 8 | 6 | 107 | 2 |
| Gemma-2-9B-it | Restatement | 68 | 191 | 9 | 7 | 83 | 2 |
| Llama-3.1-8B-Instruct | Correction | 47 | 45 | 0 | 0 | 18 | 0 |
| Llama-3.1-8B-Instruct | Restatement | 49 | 46 | 1 | 1 | 19 | 0 |
| Qwen3-4B | Correction | 73 | 45 | 1 | 1 | 18 | 0 |
| Qwen3-4B | Restatement | 68 | 64 | 2 | 2 | 27 | 0 |

The high-margin version of the full two-format conjunction is zero in every cell.
The saved per-question outputs retain each answer format and every intermediate
control, including failed gates.

## Historical comparison and development amendment

Neither historical bridge passed its numerical comparison. All six planned paired
contrasts are therefore `NOT_ASSESSABLE`, with missing-row status additionally
blocking the cached 32B comparison. These results support checkpoint breadth;
they do not establish a scaling curve or a successful paired historical contrast.

The development gate was amended after all four development panels were observed
and before any new final output. The earlier implementation had applied historical
replay tolerances to cached/full-forward admission. The amended gate instead
required finite scores, identical discrete decisions, valid cache/token bindings
and batch size one, while retaining the numerical differences. Three checkpoints
passed the amended gate. Cached Qwen3-32B still changed a prediction and remained
unmeasured. The original failed checks and amended checks are both retained.

The [scientific amendment excerpt](../../reproducibility/text_scope/saved/code/text_scope/DEV_GATE_AMENDMENT.md)
records this chronology. This is an amended development-stage qualification,
not an unchanged prospective validation.

For each completed checkpoint, two fresh-process repeat passes covered eight fixed
cases and eight questions per context. The retained repeat report compares 3,264
pairwise reads, with zero score or cache-identity discrepancies. Repeats do not
enter the population estimate. Neutral process labels preserve the distinctness
check without exposing operating-system identifiers.

The later [stateless 32B study](../04_matched_histories.md#sec:qwen32-breadth) is
separate from the unmeasured cached row. It uses the existing designs with a
different execution path and its own technical qualification.

## Reproduce

Run from the repository root into a new output directory:

```sh
python -m repro.text_scope --output reproduced/textual-breadth
```

This reconstructs the completed and historical rows from saved scores, including
the primary table, control decomposition and failed planned comparisons. Its
[public manifest](../../reproducibility/text_scope/MANIFEST.json) binds stored
payloads and shared primary-study inputs. Integers and classifications must match
exactly; finite derived values use the existing absolute tolerance of 1e-12.
Model execution and retraining are outside this verified CPU route.
