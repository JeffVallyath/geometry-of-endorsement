# Human review calibration

These synthetic examples do not appear in any study manifest. Complete them before opening a study sheet, then compare your answers with the expected decisions.

## Claim 1 cell examples

### Clear relation

Situation and action

> A clinic asks a patient for permission before sending the patient's medical record to an employer. The clinic sends the record only after the patient agrees.

Named consideration and stored relation

> Privacy `Supports`

Expected answer `CLEAR`

The situation is intelligible, privacy applies directly, and obtaining permission before disclosure has a straightforward privacy-supporting reading.

### Ambiguous relation

Situation and action

> A manager shares an employee's personal information with coworkers when needed for the project.

Named consideration and stored relation

> Privacy `Supports`

Expected answer `AMBIGUOUS`

The phrase `when needed` does not establish what information was shared, who needed it, or whether the disclosure was limited. Privacy is relevant, but both supporting and opposing readings remain live.

Use `COMPETING_RELATION_READINGS` and explain the missing scope.

### Invalid relation

Situation and action

> A person repairs a bicycle tire before riding to work.

Named consideration and stored relation

> Electoral fairness `Opposes`

Expected answer `INVALID`

The named consideration does not apply to the stated action. Use `CONSIDERATION_INAPPLICABLE`.

## Claim 1 sense examples

### Stable sense

The phrase `equal access` concerns access to a public library in one situation and access to a public meeting in another. The protected resource changes, but the ordinary meaning of equal access remains stable.

Expected answer `STABLE`

### Shifted sense

The phrase `security` concerns protection from physical attack in one situation and collateral backing for a loan in another.

Expected answer `SHIFTED`

Use `SHIFTED_MEANING`.

### Uncertain sense

The phrase `respect for authority` concerns a parent in one situation and an unspecified authority in another, where the second text does not identify whether the authority is legal, professional, or familial.

Expected answer `UNCERTAIN`

Use `UNCERTAIN_REFERENCE`.

## Claim 2 examples

### Preserved rewrite

Original text

> The committee published the report after every member approved its release.

Rewrite

> After all committee members approved releasing the report, the committee published it.

Expected answer `PRESERVED`

The actor, action, timing, approval condition, and scope remain fixed. Leave diagnostics blank.

### Changed rewrite

Original text

> The medication may cause temporary dizziness in some patients.

Rewrite

> The medication causes temporary dizziness in all patients.

Expected answer `CHANGED`

The rewrite changes uncertainty and quantifier scope. Use `FACTS_CERTAINTY_SCOPE_SPECIFICITY`.

### Uncertain rewrite

Original text

> Jordan told Casey that Jordan would return the borrowed laptop before Monday.

Rewrite

> Jordan told Casey they would return it soon.

Expected answer `UNCERTAIN`

The pronoun and deadline no longer establish the same actor and timing. Use `ACTOR_ACTION_TARGET_INTENT` and `FACTS_CERTAINTY_SCOPE_SPECIFICITY`.

## Calibration rule

Discuss calibration disagreements before study review begins. Do not discuss study rows after independent work starts. If a reviewer cannot apply these boundaries consistently, revise the training examples or replace the reviewer before distributing a study package.
