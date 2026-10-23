# Prospective analysis binding

Historical prospective analysis rules, fixed before the study outcomes.
Use the inherited FHV decoder (full-vocabulary validity, candidate-mass identity)
and `primary` gates unchanged on A only, with 18 targets per case/seed.
Store the A-only W lock before reading intervention outputs in the analysis.

Repair requires the target, all 12 direct questions, the two same-pair functions,
and six other-pair functions in its answer format to remain valid and correct in
all eight histories. Wrong consensus, invalidity or collateral loss gets zero.
The fixed full current-record reference is A/origin-zero for that seed; do not
count its eight identical logical copies as independent observations.
The physically duplicated NATIVE_FINAL rows must agree at the inherited 1e-6
score tolerance and identical discrete/validity/token flags. Retain both physical
copies and use seed-0's native row, fixed in advance, for the shared FHV W input.

Primary case vector: mean of 36 baseline-W-and-repaired indicators for B minus
the corresponding mean for C. Other three family vectors: B-minus-A joint error,
newly introduced direct error divided by all scheduled direct responses, and
newly introduced joint error divided by all scheduled joint responses. Errors
include invalid responses. Family denominators include both seeds, formats and
all eight histories. Also report damage conditional on correct/valid A responses,
every correct-to-wrong and wrong-to-correct transition, whole-34-question
correctness, all-history accuracy and distinct repaired cases.

10,000 case bootstrap replicates, NumPy default_rng(2609184202), 256 cases sampled
with replacement, retaining all seeds/histories/questions/arms. Percentiles
0.00625 and 0.99375 give the prespecified two-sided 98.75% intervals for the four
endpoint family. Seed and format decompositions are descriptive; no row bootstrap.
Strong repair requires >=20 distinct W cases, lower efficacy bound >0, both seed
contrasts >0, >=50% W rows repaired by B, upper net-joint-error bound <0, neither
seed with increased joint error, and both harm upper bounds <=0.01. Do not alter
these conditions for sparsity or a negative result. Reverse corroboration remains
separate: >=8/32 fixed cases with eligible history-dependent errors and both final
semantic-answer directions represented.

Reverse eligibility uses the same factual and logical controls as W, target clean
reference correctness, and inherited count/length matching of donors. Report all
32 cases and failures, plus the explicitly clean-reference-correct subset.
Both B-range and C-range reverse arms are scored on all 34 questions.

Source-content diagnostic: for every joint question and every structurally
matched donor-history pair with different old source truth, orient the pair as
old-true minus old-false. Signed semantic margin is p(true)-p(false) from inherited
two-candidate probabilities, with answer mass reported separately. Report the A
margin contrast and the B/C changes to that contrast, aggregating pairs/questions
within cases before across-case summaries. Stratify by final truth, question type,
answer mapping and seed. Retain individual history margin changes grouped by old
truth and final truth, so an undirected semantic-Yes shift is visible rather than
misdescribed as content transfer. Invalid/nonfinite margins are counted and not
silently dropped from the case inventory.

Exact-absolute-position subset is defined pre-outcome by MASK.json, separately
from end-offset transplantation: report both cases with all eight mappings exact
and the individual history mappings with exact absolute positions. The latter is
a descriptive response subset, never a replacement for the all-history primary.
Empty suffixes remain identity/unmappable cases.
Report gate attrition and strong .95/.05 separation with mass, without replacing
primary denominators. At most one of nine semantic targets can satisfy full W
per seed/answer-format; this is a counting restriction, not a new success measure.

Repeats: the union of the four hash-selected cases from SELECTION.json and the
first four baseline-W cases in frozen order (fewer if insufficient positives),
combined across seeds. Deduplicate that union without filling from later W cases;
overlap can therefore reduce the total below eight. This corrects the earlier
descriptive rule string in generated/SELECTION.json before any model outcome;
its fixed hash-selected IDs are unchanged. REPEAT_SELECTION_RULE.json binds this
literal GOAL.md rule. A/B/C are repeated in a fresh process on each seed, preserving every
discrepancy. Illustration is the first hash-selected case, irrespective of effect.
Show the lexicographically first No/Yes joint query in its frozen gate mapping,
both seeds, all histories and all four arms, with direct and other-logical
correctness counts. Do not substitute a more favorable target if it has no effect.
