# Scientific protocol excerpt

These are the preserved scientific sections of the pre-outcome protocol. Former execution authorizations and resource instructions are omitted. They document the completed study, not permission to run it.

## Fixed scope

- Model: google/gemma-2-9b-it, revision 11c9b309abf73637e4b6f9a3fa1e92e615547819.
- Update: the existing INV_PAIR_NLL learned activation update, both stored seeds 0 and 1. Load their saved realized bases verbatim. No fitting.
- New final cases: 64. New technical cases: 8, disjoint from final.
- Keep the original 8 starting histories, 3 updated relations, 34 queries, 2 answer-code draws, and native-final / same-update-already-correct references.
- Retain the original task vocabulary, setting, rendering, demonstration prefixes, update order and answer-scoring conventions.
- Select only these two learned conditions from the original execution schedule. Do not compile unrelated methods.
- Each case needs 16 updated contexts (8 histories x 2 learned seeds) plus one shared native-final context. The two history-0 learned contexts are the no-op references. This gives 578 logical score rows per case, before exact identity reuse.
- Qwen, parameter editing, new causal searches, and alignment training are outside this run.

This is a prospective confirmation in the existing controlled task family. Both the model and update were selected using prior evidence. The claim concerns fresh generated cases, not a new model family or arbitrary tasks.

## Fresh data

Generate 8 technical and 64 final cases with a new local RNG seed 2609182101 and a new namespace. Use the original task-generation procedure, schema and renderer.

Preserve the final balance: four person/proposal size combinations (4/2, 4/3, 6/2, 6/3), eight final truth assignments to the addressed relations, two cases per size/truth cell. Technical cases cover the four sizes and two focal truth directions.

Use new names/layouts/background values sampled by the same procedure. Keep the eight XOR histories of the three addressed relations, including the unchanged final history. All receive the same final SET commands. Verify equality of the complete intended final table, including unaddressed facts.

Check semantic/textual collision hashes with every supplied old training, development, confirmation and audit input. Ignore case IDs when checking collisions. Use deterministic candidate replacement within the same balance cell, based only on collisions or structural invalidity, before inference. Do not reject cases for model answers, likely difficulty, or anticipated disagreement.

Verify every label from the Boolean truth table and every query mapping independently. Across the two draws there must be the inherited 12 direct-read IDs, 18 joint-query IDs and four complementary direct IDs. Preserve the original meaning of each query rather than guessing from its name.

Compute exact tokenized prefix lengths with the frozen tokenizer before model inference. For each case and condition, enumerate pairs with the same number of changed facts and the same prefix length. Record all pairs and their opportunity counts. Do not add a word-identity or clause-position matching claim.

Freeze all cases, rendered prompts, commands, label maps, eligible structural pairs, analysis rules and secondary validation/test partitions before any new model output. Include a static test that original-source rewriting converges to exactly the same final text.

## Single prospective primary result

Use the audit's G2a AND G2b AND M criterion as the one primary outcome. Generalize its rule from semantic query specifications to the new IDs.

For a target question to count:
1. All 12 inherited direct checks are valid and correct under all eight updated histories.
2. The target question is valid and correct in the native-final and same-method no-op references.
3. Under that same answer-code draw, both other Boolean compositions of the same relation pair are valid and correct under all eight histories.
4. All three compositions of each of the other two addressed relation pairs are valid and correct under all eight histories.
5. A pair of histories with equal changed-fact count and equal tokenized prefix length gives distinct valid target answers.

Exactly reproduce the original definitions of valid answers, ties, finite scores, full-vocabulary argmax membership, normalized candidate probability, and reference correctness. All current checks are separate queries on immutable copies of the same pre-question state. Do not concatenate checked answers into the target prompt.

There are 18 scheduled coded targets per case per seed, 1,152 per seed. Ineligible targets remain in this full denominator. Average the two seed rates within each case before averaging the 64 cases.

Report counts, full-denominator rate, number of contributing cases, eligibility coverage and each seed separately. Use a 95% case-bootstrap interval, 10,000 paired case resamples, seed 2609182102. This is uncertainty within the new generator, not deployment prevalence.

Retain the weaker original witness / same-pair-only / cross-pair-only counts as descriptive decompositions. Report the strong >=.95/<=.05 matched-pair criterion and both-code common-pair, same-direction sensitivity separately. Two formats and two learned seeds are repeated measurements of cases.

A completed zero result is a finding for this fixed test. Low eligibility limits its interpretation. Neither outcome permits retuning or sample expansion.

## Secondary experiment: do history checks help with later questions?

Use the same 34-query battery and saved scores. This block requires no extra model inference. Fix its partition before any new scores.

For each case, sort the three addressed relation pairs canonically. With FUNCTIONS = (same, both, either), hold out function (case_index + pair_index) modulo 3 on each pair. These three target questions are never available to either validation rule. The remaining six functions form the validation pool. Both code draws inherit the same semantic split.

Choose one matched history pair by the smallest SHA256 of its case-ID/pair-ID tuple, using ONLY pre-model count/length information. Choose the anchor within that pair by an independent fixed hash bit. Do not select an extreme pair. If no structural pair exists, record missing applicability, not an outcome-selected replacement.

Common preliminary checks, shared and charged equally to both rules: the six direct factual queries under both chosen histories in the chosen code, plus the direct factual native/no-op references. A case/seed that fails these is outside the comparison's common eligible pool but remains in the primary denominator.

Compare these two explicitly defined validation procedures:

A. Six different consequence checks at the anchor history:
   Ask all six validation questions. Accept when all are valid and correct.

B. Three consequence checks across two histories:
   For each relation pair, choose one of its two validation questions by a fixed hash.
   Ask those three questions in both chosen histories. Accept when all six answers
   are valid and correct. The disagreement count is separately reported.

Each procedure receives six additional scored queries. Neither receives the
withheld questions or their responses. This compares query diversity with history
diversity at a fixed scored-query count. Account for the common direct checks and
actual token/prefill costs separately. Do not call either rule the entire field's
standard or claim equal GPU cost from query count alone.

Use plan_checks.py (or an exactly tested equivalent) for the fixed query partition and history-pair selection. Its input is task metadata, never model scores. Freeze and write each procedure's acceptance decisions before unsealing the three withheld target responses. Prefer an isolated decision process that can read only permitted IDs. The full primary analyzer may read the complete battery only afterward.

On the withheld questions, measure correctness and history disagreement separately. For each case/seed/format, the task-error score is the fraction of the six withheld responses that are incorrect or invalid: three withheld questions evaluated at both selected histories. Average this case score equally across accepted or retained cases. Also report the proportion of cases with any withheld error and the proportion of the three targets with a valid hard-answer disagreement. The named equal-retention contrast is rule B's mean task-error score minus rule A's; negative values favor checking two histories. Do not select a different loss after viewing outcomes.

Invalid responses count as task failures; do not turn them into two valid differing answers. Report current-only reference correctness on the withheld questions so generally unanswerable tasks remain visible.

Report:
- the common eligible case count;
- each rule's accept/reject counts;
- withheld errors among accepted cases;
- accepted-and-successful cases;
- cases accepted by A but rejected by B, with their withheld outcomes;
- the reverse comparison too.

To check whether rejection alone explains any apparent benefit, also rank by
the number of wrong/invalid validation answers (0..6), with the same fixed
case-hash tie breaker for both rules. Compare withheld task-error rates while
retaining the same 25%, 50% and 75% of common cases (floor of the fraction times the common case count, identically for both rules). The 50% comparison is the
named utility endpoint; the others are descriptive. Compute rankings and the 50% retained-set error contrast separately for each
learned seed. The named combined summary is the equally weighted mean of the two
seed-specific contrasts. Resample the 64 case IDs together across both seeds;
reapply eligibility and ranking within each bootstrap sample, preserving paired
measurements. Keep the same hash tie-breaker. Report per-seed results too. If a
bootstrap sample has no eligible retained cases, mark that interval unavailable
rather than dropping undefined draws. At least 20 distinct common cases per seed are required
for a quantitative utility headline; otherwise report the limitation.

Literal No/Yes is the named utility interface. Repeat the same analysis under the
original A/B mapping as sensitivity, without choosing a favorable format.

A benefit on withheld questions supports practical value beyond detecting the
same failure used to score the test. A primary robustness success can coexist
with no utility advantage. Report those outcomes separately.

