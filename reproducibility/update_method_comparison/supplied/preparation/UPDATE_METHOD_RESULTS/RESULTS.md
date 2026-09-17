# Update-method comparison — results

**Status: PENDING. No model measurements have been performed.**

This document is the frozen reporting skeleton. Every cell marked `PENDING` is
filled by running `COMMANDS.md` on a GPU host. Nothing here reports a model
outcome, and no interpretation below has been selected after seeing one.

---

## Question

Does a frozen learned relational editor give better complete-answer correctness,
or a better measured quality/time tradeoff, than strong no-training corrections
and local cache refresh — holding source facts, allowed update commands, later
questions, backbone, and scoring fixed?

Either side is allowed to win. This is an additional comparison, not a
replacement for historical results, and it does not by itself establish
external-domain transfer or a general mechanism.

## Stage 1 — external implementation anchor

| Condition | Decision | Source |
|---|---|---|
| stale (admin, no edit) | PENDING | fresh |
| clean recomputation (full reprefill) | PENDING | fresh |
| erratum | PENDING | fresh |
| field + erratum | PENDING | fresh |

Expected tool decision recovered: **PENDING**

Pinned commit `e9085eafcc6c83e60c548de69060c4bd5b210c96`. Two documented
compatibility changes (notes §4). Released JSON under the external repo's
`results/` is the authors' reported findings and is **not** pooled with anything
here. Stage 1 and the supplied evaluation are different tasks with different
documented decoding setups; their rates are not pooled and this is not a literal
reproduction of the public benchmark.

## Stage 2 — matched relational comparison

Backbones: `Qwen/Qwen3-8B` @ `b968826d…`, `google/gemma-2-9b-it` @ `11c9b309…`.
Reader R0, no-thinking/direct answer, original sites, precision, score head, and
answer mapping retained.

### Primary endpoint

Complete 34-question bundle correct from all four origins; terminal root as the
independent unit. Four planned contrasts, each learned recipe vs
`FIELD_PLUS_LATEST_ERRATUM`, separately per model. Both fixed seeds carried
together within each root, baseline subtracted once. Paired root bootstrap,
10,000 draws, seed 260913902, 98.75% intervals.

| Model | Recipe | Effect vs baseline | 98.75% CI | Seed 0 | Seed 1 |
|---|---|---|---|---|---|
| gemma | INV_PAIR_NLL | PENDING | PENDING | PENDING | PENDING |
| gemma | FREE_PAIR_CONSISTENCY | PENDING | PENDING | PENDING | PENDING |
| qwen | INV_PAIR_NLL | PENDING | PENDING | PENDING | PENDING |
| qwen | FREE_PAIR_CONSISTENCY | PENDING | PENDING | PENDING | PENDING |

### All declared arms

Every arm reports whether or not it performs well. No arm is dropped, and the
weakest simple baseline is not selected for the headline.

| Arm | gemma | qwen |
|---|---|---|
| SOURCE | PENDING | PENDING |
| REBUILD | PENDING | PENDING |
| EXISTING_CORRECTION | PENDING | PENDING |
| LATEST_SAME_WORDING | PENDING | PENDING |
| LATEST_ERRATUM | PENDING | PENDING |
| FIELD_PLUS_LATEST_ERRATUM *(primary baseline)* | PENDING | PENDING |
| INV_PAIR_NLL s0 / s1 | PENDING | PENDING |
| FREE_PAIR_CONSISTENCY s0 / s1 | PENDING | PENDING |
| CANONICAL_REPLAY s0 / s1 | PENDING | PENDING |

Differences among `EXISTING_CORRECTION`, `LATEST_SAME_WORDING`, `LATEST_ERRATUM`
and `FIELD_PLUS_LATEST_ERRATUM` are **explicitly secondary**; no separate
prospective analysis plan has been declared for them.

### Secondary measures

All-origin question accuracy, disagreement, required-change accuracy, invariant
error, direct-operand checks, all-24/main correctness, and the inherited
single/repeat/restore/AB/ABC criteria: **PENDING** (`tables/`).

Whole-bundle and mean-question accuracy are reported separately and not merged.

### Local-refresh fallback

| Model | Refresh attempted | Fallback used | Rate |
|---|---|---|---|
| gemma | PENDING | PENDING | PENDING |
| qwen | PENDING | PENDING | PENDING |

If the rate is high, `FIELD_PLUS_LATEST_ERRATUM` collapses toward
`LATEST_ERRATUM` under the documented fixed fallback, and the primary contrast
must be read as such. See notes §7.

### Timing and memory

Preprocessing, source prefill, update/refresh, and suffix answering measured
separately with synchronized timings and warmups: **PENDING**
(`tables/timing_memory.csv`). The initial source prefill is shared in the
post-update comparison but still reported. Transient model memory and retained
backup/history/cache storage: **PENDING**.

No speed claim is derived from parameter or token counts, and no service-scale or
end-to-end product claim is made.

## Interpretation — decided in advance, not yet applied

Fixed before measurement, from `COMPARISON_SPEC.md`:

- Learned methods show a substantial, uncertainty-supported quality advantage
  without disproportionate time/storage cost → evidence for useful learned
  control **in this setting**; broader tasks still needed before any general
  application claim.
- A published or normalized no-training alternative matches or exceeds the useful
  operation → retain the learned-capability finding, but do **not** claim learning
  is necessary or practically preferable. An interval crossing zero does not prove
  equivalence; report point performance and uncertainty.
- Both classes retain source-dependent errors → the failure is not unique to the
  supplied loss or architecture. Cross-method evidence on this task, not a
  field-wide theorem.
- External anchor fails for unresolved technical reasons → no claim that the
  published method fails; the comparison stays technically incomplete.
- History normalization alone wins → credit the software simplification; do not
  rebrand it as a learned semantic mechanism.

Neither a completed implementation nor an added table automatically raises the
contribution's significance class.

## Verification limits

- Package integrity and generator reproduction: **PASS** (73 files, 0 collisions,
  328 fresh unique sources, 0 model evaluations).
- CPU acceptance tests (`check_cpu.py`): **NOT RUN** during preparation — pinned
  `torch==2.8.0+cpu` was unreachable. Run on Colab first.
- Real tokenizer boundaries, stock SOURCE equivalence, clone immutability,
  precision, both models, external adapter: **PENDING** (`preflight.py --stage model`).
- Adapter cache surgery is ported from the pinned implementation but is
  **unverified against a real backbone**.
- Identified missing cells, if compute runs short:
  `run_comparison.py --report-gaps`. A shortfall is reported as missing cells,
  never as a selected favourable subset.
