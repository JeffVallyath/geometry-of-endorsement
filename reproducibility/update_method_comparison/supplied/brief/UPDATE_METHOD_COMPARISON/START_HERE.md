# Update-method comparison

## Why this comparison matters

The supplied study uses learned internal updates. A recent paper, *Models Take Notes at Prefill: KV Cache Can Be Editable and Composable*, provides direct correction and cache-refresh methods that do not require training a new editor.

The purpose of this task is to compare those approaches fairly on the same fixed cases. The comparison should answer a simple question:

**Do the learned updates produce better answers or a better quality/time tradeoff than strong no-training alternatives?**

A result favoring either side is useful. Do not tune the comparison to make one method win.

## Files you will receive

You should receive two ZIP files:

1. `UPDATE_METHOD_COMPARISON.zip`
   - `START_HERE.md` — this file
   - `COMPARISON_SPEC.md` — exact experiment rules
   - `REFERENCES.md` — paper and code links
   - `MANIFEST.json` — file hashes

2. `UPDATE_METHOD_INPUTS.zip`
   - fixed evaluation cases
   - selected learned-editor weights
   - the minimal runner and scoring code
   - environment/dependency information
   - a short usage note with the exact local commands

If the second ZIP is missing, stop after reading the references. The comparison is not runnable from the brief alone.

## What to do

### Step 1 — Set up a clean working folder

Create a new folder for this task. Extract both ZIP files there. Do not edit the original ZIP files.

Keep the public comparison code separate from any unrelated research checkout or active experiment.

### Step 2 — Read the fixed instructions

Read, in this order:

1. this file;
2. `REFERENCES.md`;
3. `COMPARISON_SPEC.md`;
4. the usage note inside `UPDATE_METHOD_INPUTS.zip`.

`COMPARISON_SPEC.md` is the authority when a technical detail is unclear.

### Step 3 — Pin the external code

Open the repository linked in `REFERENCES.md` and record the exact Git commit used.

Do not just copy code snippets from the README. Use the released implementation and preserve any small compatibility changes separately.

### Step 4 — Reproduce the paper's small worked example

Before touching the supplied evaluation cases, run the paper's documented account-role example on Qwen3-8B.

Run the conditions requested in `COMPARISON_SPEC.md` and save:

- the exact commands;
- the Git commit;
- stdout/stderr;
- the resulting outputs;
- any compatibility change that was required.

This is only an implementation check. It is not a full reproduction of the paper.

If the worked example cannot be reproduced because of a concrete compatibility problem, record the exact problem and stop that external-method branch. Do not replace it with a newly invented method.

### Step 5 — Verify the supplied comparison package

Run the CPU checks from `UPDATE_METHOD_INPUTS.zip` before any model measurements.

Confirm that:

- the fixed cases load;
- the learned-editor weights load;
- source text and update commands render correctly;
- question text is unchanged;
- scoring produces the documented output schema;
- no method receives the answer key or final-state labels as an input.

Keep the check output.

### Step 6 — Implement the fixed no-training alternatives

Implement exactly the alternatives listed in `COMPARISON_SPEC.md`:

- the existing correction method;
- the same correction wording with only the latest value kept for each record;
- the published correction wording;
- local cache refresh plus the published correction, where the released method supports it.

Do not add extra prompt variants, new training, new readers, or model-specific fixes after seeing results.

For unsupported local-refresh cases, use the fixed fallback described in `COMPARISON_SPEC.md` and record how often it occurs.

### Step 7 — Run a small technical check first

Use only the provided development/technical cases.

Check that all methods run end to end and that timing/memory instrumentation works. This stage is for finding implementation errors, not for choosing which method looks best.

Once the technical checks pass, freeze the implementation and record hashes or a commit before running the fixed evaluation set.

### Step 8 — Run the full fixed comparison

Run every declared method on the supplied fixed cases.

Keep both model backbones and both fixed learned-editor seeds. Do not drop a condition because it performs badly.

For each method, save:

- raw answer scores or predictions;
- correctness;
- errors on answers that should stay unchanged;
- dependence on earlier values where measured;
- update/refresh time;
- answer-generation time;
- source-prefill time;
- retained storage/memory information requested by the spec;
- whether a documented fallback was used.

### Step 9 — Produce the comparison outputs

Return:

1. the exact code used;
2. the pinned external Git commit;
3. exact commands;
4. worked-example outputs;
5. raw comparison outputs;
6. one results table;
7. one comparison figure;
8. timing/memory measurements;
9. a short note listing implementation differences, fallbacks, unavailable cells, or unresolved compatibility issues.

Do not write a long interpretation of which research direction should be pursued. The main goal is a trustworthy comparison.

## Important rules

- Either approach is allowed to win.
- Do not inspect model outputs to decide which cases, seeds, methods, or prompt variants to keep.
- Do not give any method evaluator gold labels, final-world answers, or future-question answers.
- Do not silently drop unsupported cases.
- Do not change the learned-editor checkpoints.
- Do not train new models or editors.
- Do not use the external paper's automatic per-question diagnostic in the matched comparison, because it has access that the learned updater does not.
- Keep compatibility changes minimal and documented.
- Keep credentials, machine-specific paths, account data, and unrelated internal logs out of the returned package.

If something is technically impossible under the fixed specification, report the exact missing capability rather than improvising a different scientific comparison.
