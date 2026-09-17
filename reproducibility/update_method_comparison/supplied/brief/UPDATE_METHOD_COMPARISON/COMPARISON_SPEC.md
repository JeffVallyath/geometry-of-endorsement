# Fixed comparison specification

Status: proposed comparison; no implementation or model measurements have been performed for this handoff. The input export described in the private preparation note is required before running it.

## Question

Does a frozen learned relational editor supply better complete-answer correctness or a measured quality/time tradeoff than strong no-training corrections and local cache refresh, with the same source facts, allowed update commands, later questions, backbone, and scoring?

This is an additional comparison, not a replacement for historical results. It tests necessity/value of the learned construction in this setting. It does not, by itself, prove external-domain transfer or a general mechanism.

## Stage 1: check the external implementation

Use arXiv 2606.17107v1, Section 4 and Appendix B, plus the corresponding released code in `19PINE-AI/programmable-kv`. Record the exact Git commit. Inspect actual code before running commands: the README is guidance, not proof of equivalence between every driver and paper result.

Run the released account-role worked example on Qwen3-8B using stale, clean recomputation, erratum, and field-plus-erratum. Preserve the published text and the driver's decoding mode. Record whether the expected tool decision is recovered and distinguish fresh results from released JSON. A worked-example reproduction is an implementation anchor, not statistical replication of the paper.

If runtime incompatibilities require an adapter, preserve the scientific operation and demonstrate that the unchanged condition still reproduces the stock implementation. Do not spend days building a replacement inference system. A missing dependency or unsupported operation is an integration issue, not an unsuccessful scientific result.

## Stage 2: matched relational comparison

### Fixed writers and reader

Use V7's INV_PAIR_NLL and FREE_PAIR_CONSISTENCY, both original seeds, with their exact selected factor hashes. Models are the original pinned Gemma-2-9B-it and Qwen3-8B revisions supplied in the export. Retain their original sites, precision, score head, answer mapping, question wording R0, and no-thinking/direct-answer procedure. No V8 result is used to select a method or mask. Do not add reasoning to only one arm; reasoning-mode comparisons are outside this first pass.

The public example in Stage 1 and this supplied evaluation are different tasks and may use different documented decoding setups. Do not pool their rates or call this a literal reproduction of the public benchmark.

### Fixed alternatives

- SOURCE: unchanged original prefix.
- REBUILD: construct the final text using the supplied commands and recompute its cache. This is an exact text/cache reference, not perfect truth accuracy.
- EXISTING_CORRECTION: preserve the supplied study's existing text-correction path, including its full sequence of correction lines.
- LATEST_SAME_WORDING: retain only the last command at each addressed record, in stable source-record order, and use the exact existing correction sentence. This isolates history normalization from a wording change.
- LATEST_ERRATUM: use the same last-value commands and order, rendered using the fixed published update pattern from Appendix B, substituting only field identifiers and requested values. Do not supply logical consequences or recommended answers. Log the exact adapter template before any evaluation.
- FIELD_PLUS_LATEST_ERRATUM: additionally refresh the updated clause's K/V states using the published field-refresh operation, then apply the same erratum. Reuse stock cache mechanics where compatible. This is a documented adaptation from a single field to the explicitly addressed clauses, not a claim that the paper evaluated the supplied task.
- The two learned recipes under both seeds.
- Existing canonical replay under its original per-model/seed selections and disclosed backup/history access.

Do not use the external library's AUTO mode or its per-question diagnostic: that would give the editor access to future answers/questions. No condition may use model outputs to decide which requested assignments execute.

Normalize commands using only their address and last supplied value. In particular, do not discard a restoration because evaluator gold says it matches the original state. Do not give any method an answer key, source-origin label, or evaluator final-world dictionary. REBUILD may derive final text by applying actual commands to original input, as an ordinary textual baseline.

All corrected artifacts are made before the later question is supplied. Each question reads an immutable clone. Readable source text, command history, and source backup are real information/resource privileges: report them for each method rather than pretending every method has identical retained state.

### Local-refresh compatibility

Check the actual tokenizer and cache interface, not string lengths. Stock field refresh requires compatible token positions. Never pad, align, or splice unequal-length spans just to obtain a result. For unsupported length-changing cases, use the library-documented erratum-only fallback as a fixed policy, mark the fallback, and report its frequency separately. Do not silently drop these cases or advertise them as successful local refreshes.

For multi-address updates, use a fixed source-record order and explicitly log which cached prefix each refreshed clause receives. Reuse the external documented sequential implementation if it exists and passes the interface checks; otherwise record the multi-address adaptation precisely before measurement. All prior and unmodified clauses remain subject to the method's actual stale-cache semantics. Do not import an evaluator-created final cache as a donor. Compute spent reconstructing any prefix must be counted.

### Populations

The input export must generate one fresh, outcome-independent balanced main population of 64 scenes per model and one fresh population of 64 terminal roots per model with four origins each, using the existing generators and label code. Namespace UPDATE_COMPARISON, seed 260913901. Preserve the existing 24-question/main and 34-question/origin definitions.

Use the lowest-hash balanced 32 main scenes for the inherited full 13-program suite. Use 8 disjoint development/technical scenes for adapter checks, not efficacy selection. Preserve the predefined balance across directions, sizes, and terminal truth cells and deduplicate source texts against prior datasets before any fresh model outcomes. Record the actual generator and manifest hashes.

Do not hand-author replacement cases. They are produced by the supplied generator. They are produced by the existing supplied generator in the preparation step. Both models and both writer seeds remain in the planned measurement. A compute shortfall yields identified missing cells, not a selected favorable subset.

### Measurements

Primary quality endpoint: complete 34-question bundle correct from all four origins, with terminal root as the independent unit. Four planned contrasts: each of the two learned recipes versus FIELD_PLUS_LATEST_ERRATUM, separately for the two models. Carry both fixed seeds together within each root. Do not count the same deterministic untrained baseline twice as independent evidence. Paired root bootstrap, 10,000 draws, seed 260913902, 98.75% intervals across the four contrasts. Show per-seed results as well.

Also report all-origin question accuracy, disagreement, required-change accuracy, invariant error, direct-operand checks, all-24/main correctness, and the original single/repeat/restore/AB/ABC criteria. Retain the distinction between whole-bundle and mean question accuracy. Same-method SOURCE is not allowed to become a conveniently weaker denominator: preserve the supplied study's original SOURCE comparison and show any interface-induced difference separately.

All differences among EXISTING_CORRECTION, LATEST_SAME_WORDING, LATEST_ERRATUM, and FIELD_PLUS_LATEST_ERRATUM are explicitly secondary unless a separate prospective analysis plan is declared before measurement. Do not select the weakest simple baseline for the headline. A strong negative/winning result for any baseline must remain visible.

Record actual preprocessing, source prefill, update/refresh, and suffix-answering times separately, with synchronized timings and warmups. The initial source prefill is shared in the post-update comparison but still reported. Charge replay/rebuild for work actually performed; charge learned methods for their actual implementation, including any recomputation. Do not time correction via full reprefill if a validated incremental cache path is available, while giving the learned method cache reuse. No speed claim from parameter counts or token counts alone.

On a fixed small timing subset, measure the actual supplied question batches rather than inventing a long-horizon speedup. Report transient model memory and retained backup/history/cache storage. A quality/time advantage requires measured comparable operating conditions; no unmeasured service-scale or end-to-end product claim.

## Interpretation decided in advance

- Learned methods have a substantial, uncertainty-supported quality advantage without disproportionate time/storage costs: evidence for useful learned control in this setting. Broader tasks are still needed before a general application claim.
- A published or normalized no-training alternative matches or exceeds the useful operation: retain the learned-capability finding, but do not claim learning is necessary or practically preferable. An interval crossing zero does not prove equivalence; report point performance and uncertainty.
- Both classes retain source-dependent errors: the failure is not unique to the supplied loss or architecture. That is cross-method evidence on this task, not a new field-wide theorem.
- External anchor fails for unresolved technical reasons: no claim that the published method fails. Keep the comparison technically incomplete.
- History normalization alone wins: credit the software simplification. Do not rebrand it as a learned semantic mechanism.

A positive comparison can strengthen the scientific case. A negative result can redirect effort. Neither a completed implementation nor an added table automatically raises the contribution's significance class.

## Return

Return one compact RESULTS.md, exact code/commands/configurations, case IDs and hashes, raw score rows or a separately indexed archive, per-root/per-scene summaries, timing/memory measurements, and a single comparison figure. Include the unmodified external worked-example outputs separately. Clearly label adaptation, fallback, unavailable cells, and verification limits. No private machine paths, account data, secrets, conversations, or administrative logs in the shareable scientific package.
