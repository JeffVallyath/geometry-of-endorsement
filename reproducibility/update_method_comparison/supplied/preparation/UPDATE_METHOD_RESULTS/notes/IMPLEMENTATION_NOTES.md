# Implementation notes

Status of this package: **implementation complete, measurements not yet run.**
Every number in `RESULTS.md`, `tables/`, `raw/`, `figure/`, and
`external_reproduction/outputs/` is `PENDING` until the Colab run executes.
Nothing in this package reports a model outcome.

---

## 1. What the supplied input package does and does not provide

`UPDATE_METHOD_INPUTS` ships the fixed cases, the selected frozen editors, the
canonical scientific functions, the source/update wrapper, and independent
scoring. It does **not** ship the pretrained backbones or the external
programmable-KV adapter, and it explicitly refuses two of the declared arms:

```python
# inputs.py, prepare()
if method in ('LATEST_ERRATUM','FIELD_PLUS_LATEST_ERRATUM'):
    raise NotImplementedError('External Stage 1 adapter and stock-equivalence/token
    compatibility checks are required; no replacement implementation supplied')
```

Those two arms are the entire implementation burden of Step 6. They are supplied
by `code/erratum_adapter.py`. `inputs.py`, `metrics.py`, `serve_questions.py`,
`canonical/`, `weights/`, and `cases/` are **unmodified**.

## 2. Verification already completed (no GPU required)

`verify_package.py` passes on the delivered ZIPs:

```
status PASS | files 73 | remaining_collisions 0
fresh_unique_sources 328 | model_evaluations 0
```

This is not only a hash check: it re-runs `generate_cases.build` and asserts the
shipped `main/origin/dev/program` populations are identical to what the declared
generator produces. Populations confirmed: 64 main scenes, 64 origin roots,
8 dev scenes, 32 programs, 1,616 plan rows, 44,288 evaluation rows.

## 3. External code, pinned

```
repo    https://github.com/19PINE-AI/programmable-kv
commit  e9085eafcc6c83e60c548de69060c4bd5b210c96   (2026-06-29)
```

Recorded in `external_reproduction/commit.txt`. The arXiv citation was added at
`210e005646e7c283bd32a15166f0038d2a92d2e9` (2026-06-17); we pin the later head
because it contains that commit. Public results in `results/` are **reported
findings, not measurements rerun for this handoff**, and are labelled as such
wherever quoted.

## 4. Compatibility changes to the external code

Both are minimal, documented, and preserved separately. The released checkout is
not edited in place; `code/stage1_worked_example.py` imports from it.

**PATCH 1 — missing condition.** `editkv/example.py` prints `STALE`, `IN_PLACE`,
`ERRATUM`, `FULL_REPREFILL`, the diagnostic, and `AUTO`. `COMPARISON_SPEC.md`
Stage 1 requires stale, clean recomputation, erratum, **and field-plus-erratum**.
`Mode.FIELD_PLUS_ERRATUM` exists in `editkv/core.py` but the released example never
calls it. We add that one condition and change nothing else. The published
`POLICY` / `CONVO` / `DECISION` strings and the driver's decoding mode are imported
verbatim rather than retyped.

**PATCH 2 — device.** `editkv/example.py` hardcodes `device_map="cuda"`. Kept as-is
on a CUDA host; exposed as `--device` only so a non-CUDA failure is legible.

## 5. The erratum adaptation, stated precisely

The published operation addresses a single named **field** (`account_role`). The
supplied task addresses whole **record clauses** of the canonical form
`<Actor> is in favor of|opposed to the <Project> proposal.`

Trigger template copied verbatim from `editkv/core.py::DEFAULT_TRIGGER`:

```
[STATE UPDATE] {label} has changed to {value}; this overrides any earlier value
AND any earlier conclusion. Apply the current value.
```

Identifier binding (field identifiers and requested values only — no logical
consequences, no recommended answers):

```
label   {actor}'s position on the {project} proposal
value   "in favor" | "opposed"
```

Rendered example, from fixed case `inputs.jsonl[0]`:

```
[STATE UPDATE] Sora's position on the Canal proposal has changed to in favor;
this overrides any earlier value AND any earlier conclusion. Apply the current value.
```

Template fingerprint, **frozen before measurement**:

```
sha256  2d88092b1c5dfa5da26f59ed7262713fb1b28d4f920579aa991ca5d5d0467c68
```

`preflight.py --stage cpu` logs this hash before any evaluation. Wording is part
of what the comparison tests — `LATEST_SAME_WORDING` vs `LATEST_ERRATUM` isolates
exactly the wording change — so altering the binding after seeing any model
outcome would invalidate the arm. If you want a different binding, change it now
and re-freeze; not later.

**Insertion point.** Erratum lines go immediately before the `End of records.`
tail, which is the same position `inputs.rewrite` uses for `EXISTING_CORRECTION`
and `LATEST_SAME_WORDING`. The only difference between `LATEST_SAME_WORDING` and
`LATEST_ERRATUM` is therefore the wording itself, which is what the spec asks that
contrast to isolate.

**No stacked history.** Both erratum arms consume `inputs.normalize(commands)`,
i.e. one erratum for the current value per addressed record, in stable
source-record order. This follows the released README's explicit caution that a
non-monotonic history such as `A→B→A` can let a salient intermediate state
dominate.

## 6. The silent-fallback bug we had to work around

`editkv/core.py::build_cache` swallows `LengthChangeError` for
`FIELD_PLUS_ERRATUM`:

```python
try:
    self._refresh_field_inplace(cache, f, new_value)
except LengthChangeError:
    if mode == Mode.IN_PLACE:
        raise          # <- FIELD_PLUS_ERRATUM falls through silently
```

So stock `FIELD_PLUS_ERRATUM` degrades to erratum-only **without telling you**.
`COMPARISON_SPEC.md` forbids exactly that: *"Do not silently drop these cases or
advertise them as successful local refreshes."* `erratum_adapter.py` therefore
catches the condition explicitly, applies the library-documented erratum-only
fallback as a fixed policy, and marks the row `refresh_fallback=True`. The rate is
reported per condition in `tables/timing_memory.csv`.

Feasibility is checked against the **actual tokenizer and cache interface**, never
string lengths. Unequal-length spans are never padded, aligned, or spliced.

## 7. Open risk that must be read off the preflight

`FIELD_PLUS_LATEST_ERRATUM` is the **primary baseline** for all four planned
contrasts. Its field-refresh half only executes when swapping
`in favor of` ↔ `opposed to` inside a record clause is **token-length-preserving**
under that model's tokenizer.

We could not test this here — it needs the real Qwen and Gemma tokenizers. If the
refresh is infeasible on most cases, then `FIELD_PLUS_LATEST_ERRATUM` collapses
toward `LATEST_ERRATUM` under the documented fallback, and the headline contrast
must state that plainly rather than presenting it as a field-refresh result.

`preflight.py --stage model` runs this census on the 8 DEV scenes and reports
`fallback_rate` plus an explicit interpretation string. **Read it before starting
the full run.** DEV scenes are for adapter checks only, never efficacy selection.

## 8. Multi-address ordering

Fixed ascending source-record offset. Clause *k* is recomputed against a cached
prefix that already contains refreshes `0..k-1`. `refresh_plan()` logs
`cached_prefix_per_clause` for every case, recording exactly which refreshed
clauses each span attends to. No evaluator-created final cache is ever imported as
a donor. All prior and unmodified clauses remain under the method's real
stale-cache semantics.

## 9. Timing and retained state

`inputs.prepare` recompiles the source prefix for the learned and replay arms and
charges that full work; we do not discount it. Preprocessing, source prefill,
update/refresh, and suffix answering are recorded separately, with
`torch.cuda.synchronize` around each measured region via `inputs._sync`.

`methods.retained_state()` emits, per arm, which information and resource
privileges it actually holds (readable source text, full command history vs
normalized last-value only, source backup, stale-cache semantics, learned
parameters). The spec requires reporting these rather than pretending every method
has identical retained state.

Excluded for every Stage 2 arm: answer key, origin label, final-world dictionary,
future-question access, the per-question diagnostic, and `Mode.AUTO`. The
diagnostic and AUTO **are** exercised in Stage 1, because the paper's own example
reports them; they are recorded there and excluded from the matched comparison,
where they would give the editor access the learned updater does not have.

## 10. Analysis fidelity

The primary endpoint and its bootstrap are **not reimplemented**. The supplied
`metrics.paired_primary` already fixes the endpoint (complete 34-question bundle
correct from all four origins), the unit (terminal root), the seed handling (both
seeds carried inside each root, baseline subtracted once), and the bootstrap
(paired root, 10,000 draws, seed 260913902, 98.75% intervals over the four
contrasts). `analysis.py` only builds its `vectors` argument and emits the
secondary tables. `paired_primary` fails closed on missing roots or incomplete
contrasts — that behaviour is deliberately kept, so a shortfall surfaces as
identified missing cells rather than a quiet partial result.

## 11. Known limits of this package

- No arm has been executed against a real backbone. The adapter's cache surgery
  is ported from the pinned `_refresh_field_inplace` but is **unverified**;
  `preflight.py --stage model` is the gate.
- `run_comparison.py` has been syntax-checked only.
- CPU checks (`check_cpu.py`) were **not** run during preparation: the pinned
  `torch==2.8.0+cpu` comes from `download.pytorch.org`, which was unreachable from
  the preparation environment. Run it on Colab before the model stage.
- Neither pretrained tokenizer nor model was loaded during preparation, so no
  claim here certifies either.
