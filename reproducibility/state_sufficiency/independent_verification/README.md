# Independent reconstruction from saved V10 outputs

This package reconstructs source-history sufficiency witnesses from saved model
scores and generated-case records. It does not start from saved witness flags or
root counts. It uses no original experiment analysis code and runs without a
model, tokenizer, GPU, account or network connection.

The two original source archives were available and checked. Their manifests
bind all 1,383 payloads across the main and raw-companion archives; the raw
companion's outer hash also matches the main archive's binding. The frozen
scientific specification binds 418 source files whose hashes were checked.
These checks establish retained-byte consistency, not independent attestation
of model execution, clock time or runtime receipts.

## Calculate first, compare afterward

Python 3.11 or newer and NumPy are sufficient. From the repository root, choose
a new output directory and run:

```sh
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
```

This command reads only the package's input manifest, protocol and `inputs/`.
It never reads `expected/`, the earlier aggregate headline files or an original
analysis module. It writes the independently reconstructed primary table,
per-condition summaries, root counts, per-query classifications and repeat checks.
The test suite enforces the expected-output access boundary.

Only after examining that calculation, compare it with the reported outputs:

```sh
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
python -m pytest tests/test_independent_state_sufficiency.py
```

If this directory is shared as a standalone archive, run `python reconstruct.py
analyze --output calculation` and then `python reconstruct.py compare --output
calculation` from the extracted directory. No other repository files are required
for those commands. The regression tests live in the full repository.

The comparison checks all eight primary rates and intervals, all 896 root-condition
rows and all 16,128 joint-query classifications, including the SOURCE control.
The six updating conditions account for 13,824 of those classifications. These
are correlated records within roots, not independently sampled cases.

## What is retained

| Location | Content |
|---|---|
| inputs/protocol.json | Scientific model identities and revisions, fixed inventory and bootstrap, witness definitions and the preselected repeat panel |
| inputs/designs/ | Complete starting and intended final relation tables, assignment commands, rendered source texts, fixed questions and answer-code mappings |
| inputs/responses/ | Candidate log probabilities taken from raw journals, token objects, source journal line numbers, context text and cache identities, and logical-to-physical bindings |
| expected/ | Separately stored original primary, root, query and condition results; used only by the comparison command |
| examples/ | Six fully traced positive, eligible non-witness and excluded examples, including the repeat-checked Qwen illustration |
| MANIFEST.json | Every data derivative's source archive/member hashes, output hash, transformation, row counts and omitted fields |

There are 230,656 physical core responses and 248,064 logical responses. The
17,408 aliases are checked against recorded context/cache identities instead of
being counted as additional independent model calls. Both models share 64 case
designs and are analyzed separately. The package is about 27 MB, with the scientific
records compressed as ordinary JSON gzip files. No billing, provider credentials,
host/device information or unrelated private operational logs are included.

## Independent calculation

The script applies the commands to every starting table and checks equality of
the complete intended final table, including unaddressed records. It checks all
eight assignments of the three addressed records, parses the rendered facts,
checks command spans, and reconstructs Boolean answers from the question
semantics. It verifies exact question wording, answer-code order, token hashes,
fixed token suffixes and logical/physical joins. Saved Boolean gold labels are
not needed.

Predictions, candidate mass, normalized probabilities, finiteness and ties are
computed from the saved full-vocabulary candidate log probabilities. The saved
validity flags are retained only as cross-checks. Full-vocabulary logits were
not saved, but they are not needed to classify these records. For every core
response, the larger candidate probability exceeds the total probability mass
outside the two candidates by at least 0.4904544. That proves that the global
argmax is a candidate, conditional on the recorded log probabilities. The check
uses a numerical slack of 0.000001 and fails closed if it cannot establish this
bound. It also applies to every retained repeat response. It does not infer
validity from the recorded `valid` or `argmax_in_labels` flag.

For each joint question, A means that both direct operands are correct and valid
in every history. C means that both native-final and same-method origin-zero
references are correct and valid. D means that at least two valid joint answers
differ across histories. The primary witness is A and C and D. Every root retains
all 18 joint opportunities, including ineligible questions. Learned seeds are
averaged within roots; root prevalence is the union over those seeds.

The strong diagnostic additionally requires all joint responses to be valid,
direct correct-label probabilities at least 0.95, relevant candidate mass at
least 0.90, and joint correct-label probabilities reaching at least 0.95 and at
most 0.05 in different histories. The calculation also reconstructs the
all-measured-direct sensitivity and reports the individual-condition counts.

Intervals use 10,000 root-bootstrap draws, seed 2609141002, and the original
99.375% multiplicity-adjusted level. The separately reported descriptive 95%
intervals and root-prevalence intervals are reconstructed too. Rates are fixed
opportunities for cross-history witnesses, not individual-answer error rates.

## Examples and repeat checks

Read [the illustrated examples](examples/EXAMPLES.md) or inspect
[their complete records](examples/examples.json). They include two positive
witnesses, an eligible non-witness, and exclusions from failed direct operands,
native reference and same-method no-op reference. All eight starting histories,
the common final records, direct reads, joint scores and reference scores are
shown. Each response identifies its input file, original journal line and
physical response ID.

The saved repeat panel contains native-final and four learned conditions, eight
fixed questions, eight selected roots per model and two fresh passes. The script
compares 8,448 fresh score rows and 12,672 core/pass pairwise comparisons, including
token bindings and recorded cache identities. All retained scores match exactly.
There are no independent textual-correction repeats, and not every direct operand
of a repeated joint query was repeated. The examples do not imply otherwise.

Examples can be regenerated from the independent calculation alone:

```sh
python reproducibility/state_sufficiency/independent_verification/make_examples.py --calculation reproduced/independent --output reproduced/independent-examples
```

## Re-extracting and provenance boundaries

`extract_sources.py` uses an explicit scientific-field allowlist. If the original
archives are available, it can reproduce the data derivatives in a fresh folder:

```sh
python reproducibility/state_sufficiency/independent_verification/extract_sources.py --archives PATH_TO_ORIGINAL_ZIPS --output NEW_STAGING_FOLDER
```

The extractor rejects traversal, duplicate members, unmanifested payloads, hash
mismatches and overwrites. It checks the original logical/physical score bindings
against raw journals before selecting fields. The collaborator analysis then
independently reconstructs semantic labels, eligibility, witnesses and inference.
Exact source hashes and omitted fields are recorded per derivative in the manifest.
No experimental evidence is synthesized or regenerated.

No primary classification input is missing at this saved-output level. The
remaining assumptions concern the evidence itself: that the saved scores are
the model outputs they claim to record and that their token IDs came from the
stated tokenizer. Hashes do not authenticate forward passes. Full-vocabulary
logits, original cache tensors, weight tensors and runtime attestation are not
included or recreated. Recorded cache hashes support joins and repeat identity,
not a fresh tensor-level cache check. Prospective chronology and the original
technical execution remain source-supported history, not newly established by
this calculation. A full fresh neural rerun is a different task.
