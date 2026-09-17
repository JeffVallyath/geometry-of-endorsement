# ICMH1-LP1: prospective source-label precedence amendment

Status: proposed for the user's explicit approval by sending LAUNCHER.txt. CPU
implementation and synthetic checks are provided. No model has been run in this
review, no cloud resources were accessed, and this file alone grants no spending.

## Purpose and scope

This is a computer-science evaluation of a language model's use of updated factual
records. Entity names, attributes and relation types are benchmark strings, not
instructions for laboratory work. Preserve their actual content and labels.

The supplied pre-rental no-go identified a real ambiguity problem in the response
registry. However, an all-gold-label simulation is not a proof that EVERY possible
valid answer fails: the old parser accepts unique aliases for the affected entities.
The review reproduces 8/16 A&C coverage with the standard gold labels AND 16/16 with
valid unique-alias responses under the old parser. Both are synthetic, not model
outcomes. Thus the correct diagnosis is an undesirable canonical-label/alias
collision, not a mathematical upper bound of eight under all model responses.

## One globally applied rule; no gold-dependent disambiguation

After the existing first-line/one-final-punctuation processing, NFKC normalization,
case folding and whitespace collapse, restrict to the SAME source object type as
v2. Use ALL the original source entities, labels and aliases of that type:

1. If the full normalized string matches source labels for exactly one entity ID,
   return that ID, even if other entities use the string as an alias.
2. If it matches source labels for multiple IDs, ABSTAIN. Do not fall through to
   aliases to resolve a label-label tie.
3. If it matches no source label, match aliases. Exactly one ID returns that ID;
   multiple IDs or no match abstain as before.

Do not use the gold answer, current target, question ID, root ID, qualification
membership, registry insertion order, arbitrary ID order, or a desired gate result
as a tie-break. Do not delete, rename, merge or type-change benchmark entities.
Do not change prompts to tell the model which alias would pass the parser.
The existing strict answer-correctness scorer and all numerical gates are unchanged.
The amended extraction rule is used for A and D validity/distinctness, as the single
v2 extraction rule was. Definitions of A, C, D and W otherwise remain unchanged.

This is a NAMING CONVENTION for this finite response interface. It is not a proof of
which entity a model intended when a name is genuinely ambiguous. In particular,
an alias equal to another entity's unique source label now resolves to the labelled
entity even when the alias owner would be the desired answer. Disclose that cost.

## Retained sensitivity and residual ambiguity

Save original text and strict scores. Re-score the same responses under the exact
original v2 union-label/alias registry as a descriptive sensitivity analysis. Report
all A/C/D/W differences and identify any witness that depends on precedence. Never
pick whichever parser makes the result stronger. This needs no further inference.
`checks/score_sensitivity.py` supplies the analysis wrapper.

The all-gold-label structural simulation under LP1 reaches qualification G1 16/16
for each reference, G2 22/22, G4 0/44 unresolved, and G5 16/16 across all six roots.
The final all-gold-label simulation gives A&C 96/98 across 33/34 roots. A place-label
collision remains at popular:627; the ordinary string London must still abstain in
that case's declared type. Preserve the root and denominator. A genuinely unique
alias may resolve it if the model produces that alias; do not instruct it to do so.
These figures are feasibility checks, NOT predictions or actual qualification.

The global rule changes candidate lists for 117 type/string keys in the full
registry. It is not a hand-written patch to just three names. Exact changes are in
review/MODEL_FREE_CHECKS.json. All source members, all 45 root metadata objects,
all five inference plans and all five scoring files remain byte-identical to v2.

## Procedure and authority

This amendment supersedes only the entity-registry union rule in protocol/AMENDMENT.md,
plus the incorrect universal-impossibility wording in the old blocker report. Other
scope, prompts, generation settings, frozen seeds, gate thresholds, denominators,
qualification/final separation and repeat policy are unchanged. Retain the old
no-go unmodified in provenance/ and label its conclusion as corrected by this review.

The logical read IDs retain the v2 study namespace so old/new input equality is
auditable; bind this LP1 amendment, exact parser source and new registry hash in a
separate parser freeze and the later FINAL_FREEZE. Never claim an old receipt
certifies the new parser. Regenerate derived package/capsule hashes normally.

No current model outcomes are present in the no-go archive. Confirm that this remains
true in the local worktree before execution; if the six qualification roots have
since been used for model output or tuning, report exposure rather than silently
calling them untouched. A fresh conversation does not erase scientific exposure.

The source/input/scoring code is present. The archive does NOT contain the watchdog,
remote setup or allocator/phase orchestrator claimed by the later status. Inspect
D:/icmh1-ripple-v2/source and D:/icmh1-ripple-v2/ops for specific reusable local files
without overwriting them or importing unrelated experiment histories. Verify what
is actually there and finish missing lifecycle integration before renting. Preserve
prior provider records privately; no credentials belong in scientific output.
