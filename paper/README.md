# Paper prose pipeline

Prose for the paper is checked mechanically. The point is narrow: **rewrite for
readability without losing a number, a qualification, or a scope condition.**
Editing by hand is how caveats quietly disappear, so the facts that must survive
an edit are written down separately from the prose that carries them.

```
paper/
  TERMINOLOGY.json          canonical names — the source of truth
  documents.json            which documents are validated, and with what
  contracts/results/        what each Results section must still say
  contracts/captions/       what each caption must still say
  comprehension/results/    blind-reader questions and gold answers
  comprehension/captions/
scripts/prose_check.py        the gates, coverage, and comprehension runner
scripts/resolve_provenance.py archive search for pending claims
tests/test_prose_validation.py
```

Scripts that read the large source archives take their location from
`GOE_ARCHIVE_DIR` (single directory) or `GOE_ARCHIVE_ROOTS` (search paths).
Nothing in the repository hard-codes a machine path.

```
python scripts/prose_check.py                      # all documents
python scripts/prose_check.py --document results
python scripts/prose_check.py --coverage
python scripts/prose_check.py --submission-strict  # paper-freeze gate
python -m pytest tests/test_prose_validation.py
```

## Terminology is the source of truth, not Results

`TERMINOLOGY.json` was seeded from `docs/RESULTS_AND_CLAIMS.md` and then stopped
deferring to it. Results is audited against the terminology file like every other
document. That ordering matters: while one document is privileged, an
inconsistency inside it propagates everywhere by construction, and the documents
that inherit from it will look consistent while all being wrong together.

Each term carries `accepted_variants` (fine to use) and `flag_aliases` (rewrite
these). Canonical names are stripped before aliases are searched, so "strict test
set" is not reported as a use of "test set". Symbols must be introduced next to
their canonical name before appearing alone.

## The gates

**FIDELITY.** Every contracted fact still present; every number declared; every
artifact-bound number matching its frozen artifact; no forbidden claim; required
qualifications intact. Errors.

**TERMINOLOGY.** Canonical names used, flagged aliases absent, symbols
introduced before use. Errors.

**READABILITY.** Warnings only, never blocking: sentences over the unit's limit,
two or more long parentheticals, four or more separate numeric results in one
sentence (a bracketed interval counts as one), ambiguous references such as "the
former", and provenance detail in a unit not marked `allow_provenance`.

Do not tune prose to a readability score. The target is that a competent reader
recovers the intended meaning on one pass.

## Pending claims

A number may be declared **pending**: the claim is written down and tracked, but
no frozen artifact proves it yet.

```jsonc
{"value": "0.390", "status": "pending", "headline": true,
 "claimed_source_family": "causal_intervention_v001",
 "reason": "canonical archive not yet resolved"}
```

Pending is not an excuse. A pending number must still appear in the prose, is
still counted by `--coverage`, and is reported under its own `artifact-pending`
rule. **It never decays into a generic readability warning**, because a document
that reports "0 errors" while half its substantive claims are ungrounded is worse
than one that reports the gap.

Two modes follow from that:

- **writing mode** (default) — pending claims are reported prominently but do not
  block editing;
- **`--submission-strict`** — any pending *headline* claim is an error. This is
  the paper-freeze gate.

Headline status is taken from the document's own emphasis: a value Results bolds,
or puts in a table, is a headline claim. A layer index mentioned in passing is
not. Getting a central causal estimate bound matters far more than binding a
parenthetical count, and the coverage report separates the two.

## Coverage

```
python scripts/prose_check.py --coverage
```

Reports per section how many declared claims are artifact-bound, then two
totals: all empirical claims, and headline claims alone. Watch the headline line.

## Resolving pending archives

```
python scripts/resolve_provenance.py
```

For each experiment family with pending headline claims, this scans candidate
archives and counts how many of that family's **distinctive** published values
appear inside them — three or more decimal places, or five or more significant
digits, so a coincidental match is unlikely. Set `GOE_ARCHIVE_ROOTS` to the
directories holding the archives. The report is written outside the repository
and is not committed.

A match means an archive *contains* a value, not that it produced it. Summary
bundles and handoff packets quote results from runs they did not perform and will
score highly. Confirm a candidate holds the run itself — a config digest, a model
revision, the inputs — before promoting it.

The rule for promoting a candidate to canonical:

1. several distinctive published values must reproduce from it;
2. its run metadata must agree with what the section claims;
3. if two archives match comparably, the family is **AMBIGUOUS** and stays
   pending. Do not guess between them.

Several archives can hold byte-identical parameters under different dispositions,
and only one of them will reproduce the published metrics. The tool produces
evidence; a human promotes.

## Contract format

```jsonc
{
  "unit_id": "results-3",
  "heading_prefix": "3. ",              // binds the contract to a section
  "coverage_label": "§3 ...",
  "numbers": [
    {"path": "figures.models.llama.evaluations.difference_in_means.I_b",
     "decimals": 4, "text": "1.6470", "status": "verified", "headline": true},
    {"value": "0.390", "status": "pending", "headline": true,
     "claimed_source_family": "...", "reason": "..."}
  ],
  "literals": [{"text": "201", "note": "B + 1 for the Gemma runs"}],
  "required_facts": [
    {"id": "p-is-one-sided", "critical": true,
     "description": "The reported p-value is one-sided.",
     "all_of": ["one-sided"],            // every pattern must match
     "any_of": ["..."],                  // at least one must match
     "moved_to": "Methods and provenance"}
  ],
  "forbidden": [{"id": "...", "pattern": "...", "note": "why"}],
  "qualifiers": [{"id": "...", "pattern": "...", "note": "why"}],
  "readability": {"max_sentence_words": 38, "allow_provenance": false}
}
```

Artifact paths are `<source>.<dotted path>`, where source is one of the entries
in `ARTIFACT_SOURCES` — `figures`, `leakage`, `truth`, `m1ref`, `project`. The
artifact a number came from is therefore visible in the contract itself. Facts
are matched against whitespace-collapsed text, so line wrapping cannot defeat a
pattern.

Every numeric token in the prose must be bound, declared pending, or listed as a
literal with a note. An undeclared number is an error, so a silently altered or
invented figure cannot pass.

## Blind comprehension

Fidelity checks that facts are on the page. It cannot check that a reader can
*find* them, which is where Results prose fails most often: formally correct,
cognitively awful.

```
python scripts/prose_check.py --emit-comprehension out/
# hand out/*.prompt.json to a model, collect answers
python scripts/prose_check.py --grade answers.json
```

Each bundle contains instructions, the passage, and the questions. It does not
contain the contract, the gold answers, the accept patterns, or the source
material — a test asserts this. Scoring is recoverable facts over total, and
every `critical` question must be recovered for the unit to pass.

Question archetypes for a Results section: what is the main empirical finding;
which models support it; which model failed; what comparison establishes the
effect; what does this result *not* establish; is this development or
confirmatory evidence. A good section makes those embarrassingly easy.

No model-evaluation API is added, and no network call is made.
`ComprehensionReader` is the protocol a model-backed reader should implement;
`UnavailableReader` is the default and raises rather than inventing answers.

## Working rules

1. Change the contract first only if the science changed. If the science did not
   change, the contract should not either.
2. Rewrite the prose, run the checker, fix errors. Treat warnings as a prompt to
   split a sentence, never to delete content.
3. If you moved a fact rather than dropping it, say where with `moved_to`.
4. **When a check fails, the default assumption is that the prose is wrong, not
   the contract.** Loosening a contract to make a rewrite pass defeats the point.
5. **Never silently reconcile a mismatch between prose and artifact.** Surface it.
   Contracts are derived from the document as it stands; where the document and
   the artifacts disagree, that is a finding for a human, not something to edit
   away. Open items live in a `surfaced_observations` block in the contract, so
   they are tracked in the repository rather than only in a conversation.

## House style recorded as a rule

Do not define a statistic by listing what it is not. "It is not an accuracy, a
logit, or Cohen's d" makes the reader carry three negations, and the scale is
almost always stated positively in the same sentence anyway. State the scale and
stop. The `negation-list` warning enforces this, and only fires when the negated
noun names a statistic type — saying what a *control rules out* is a different
thing and stays legal.

Known false-positive shapes: an alias matching inside a longer canonical name; a
legitimate contrastive mention of a method the text is distinguishing itself
from; and a subscript index read as a reported value. If a check fires on correct
prose, narrow the rule rather than deleting it.
