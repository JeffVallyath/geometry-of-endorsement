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

**MEASURED.** Qwen3-8B @ `b968826d…`, A100 40GB, reader as released.

| Condition | Decision | Seconds | Source |
|---|---|---|---|
| stale (admin, no edit) | `refund` | 0.278 | fresh |
| clean recomputation (full reprefill) | `escalate` | 0.161 | fresh |
| erratum | `escalate` | 0.177 | fresh |
| field + erratum | `escalate` | 0.238 | fresh |
| in-place only *(released extra)* | `refund` | 0.235 | fresh |
| AUTO *(released extra, excluded from Stage 2)* | `escalate` | 1.513 | fresh |

Expected tool decision recovered: **True**. The stale cache answers `refund`; every
update path except bare in-place recovers `escalate`. Identical decisions on two
independent runs, so the anchor is deterministic across model reloads.

Two observations recorded for the writeup. First, `in_place` alone reverting is
consistent with the released README's non-reasoning figure for this backbone and
must not be read as a failed replication of its reasoning-mode figure; reasoning
mode is out of scope here. Second, `erratum` and `field_plus_erratum` returned the
same decision, so the field refresh added nothing even on the paper's own example.

Timings are single unwarmed calls on one example and are not a speed result.

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

**NOT REPORTABLE — see Verification limits.** The ORIGIN panel executed to
completion (24/24 shards, 0 partial, both models, all twelve arms, 512 roots), but
the raw score rows were destroyed by a Colab runtime-type change before
`analysis.py` was run. No contrast was ever computed, so no effect, interval, or
per-seed figure exists to report. These are identified missing cells.

| Model | Recipe | Effect vs baseline | 98.75% CI | Seed 0 | Seed 1 |
|---|---|---|---|---|---|
| gemma | INV_PAIR_NLL | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| gemma | FREE_PAIR_CONSISTENCY | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| qwen | INV_PAIR_NLL | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| qwen | FREE_PAIR_CONSISTENCY | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

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

Preflight census, 8 DEV scenes per model (`raw/PREFLIGHT_MODEL_*.json`):

| Model | Feasible | Fallback rate | Reason |
|---|---|---|---|
| gemma | 0 / 8 | 1.00 | `PREFIX_TOKEN_LENGTH_CHANGE` |
| qwen | 0 / 8 | 1.00 | `PREFIX_TOKEN_LENGTH_CHANGE` |

Full-run rates per condition: PENDING (`tables/timing_memory.csv`).

**The field refresh never executes on this task.** The clause swap
`in favor of` / `opposed to` is not token-length-preserving, so the published
field-refresh operation is refused on every case and
`FIELD_PLUS_LATEST_ERRATUM` reduces to `LATEST_ERRATUM` under the documented fixed
fallback. The four planned contrasts must therefore be read as
**learned recipe vs erratum-only**. These two arms are the same deterministic
baseline here and are not counted as independent evidence. See notes §7.

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
- **Stage 2 produced no reportable numbers.** ORIGIN completed on an A100
  (24/24 shards, 0 partial) but the raw rows were lost to a Colab runtime-type
  change, which reprovisions the VM and clears local disk, before any table was
  generated. MAIN and PROGRAM were never started. Nothing was computed from the
  lost rows, so no figure here is derived from a partial or selected subset.
- Reproducing Stage 2 requires roughly 9 h of A100 time for ORIGIN alone
  (~22 min/shard x 24), plus ~3 h for MAIN and ~10 h for PROGRAM. Mount Drive and
  point `--raw` at it first; the shard sentinels then survive a VM reset.
- The pipeline itself is validated end to end: 24/24 DEV shards passed with
  1,152 score rows, 0 invalid answers, SOURCE clearly below every update arm, and
  `LATEST_ERRATUM` / `FIELD_PLUS_LATEST_ERRATUM` artifact hashes matching on 4/4
  cases. The implementation is run-ready; only the measurement is outstanding.
