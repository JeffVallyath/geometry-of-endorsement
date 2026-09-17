# Public-data in-context study

Internal experiment ID: `ICMH1-RIPPLE-v2`; primary parser: `LP1`.

Qwen sometimes returned an obsolete value when answering a composed question,
despite correctly reporting the current facts and answering correctly with only
those facts present. This is a completed in-context study on RippleEdits-derived
questions. Model weights were fixed. Old records remained earlier in the prompt.

## One offline command

After the normal editable installation, from the repository root:

```sh
python -m repro.in_context_updates --output reproduced/qwen
```

The command verifies the imported hashes and journal chains, reconstructs every
phase with the actual frozen scientific analyzer, and compares each scored unit,
benchmark case and question with the separately retained original analysis. It opens
expected outcomes only after classification. No model or network is used.
This reuses the executed analyzer; it is not an independently authored scorer.

A benchmark case groups a shared set of current facts, its starting histories
and questions (`root` in the records). A question qualifies as history-dependent
when its factual and clean-reference checks pass but the downstream answers differ.

Expected: 3/98 qualifying questions across three benchmark cases, 2/98 under the alternative
parser, and 65/98 eligible questions (alternative: 42/98). Every incorrect qualifying
answer is its history's superseded value. Accuracy on unrelated facts (locality) is 170/272 without history
versus 94/272 with the obsolete records from history B in the prompt: eight questions across 34 contexts,
not independent facts. All 98 questions remain in the main denominator.

The two preselected repeats cover `popular:670` (35 generations) and `random:618`
(51). Both match the final recorded text and scoring exactly under both parsers.
Neither benchmark case contains a qualifying question. Positive examples were not independently repeated.

Outputs are `results.json`, `per_root_results.csv` and `public_data_table.md`.
The committed [results](results.json) and [table](public_data_table.md) are generated
from the same command. The table is the single added public-data visual summary.

## What the parsers mean

Both rules take the first response line, trim outer whitespace, and remove at
most one final punctuation mark. Correctness against benchmark answers/aliases
is case-sensitive. Entity identification then normalizes Unicode, case and
whitespace, uses the full string, and restricts candidates to the declared type.

The primary **label-first rule (LP1)** gives an exact source label priority over
aliases for other entities. Multiple matching labels still count as ambiguous;
without a label match, an alias must identify exactly one entity. The original
**union rule** pools labels and aliases before testing uniqueness. An alias
collision can therefore make an answer ambiguous that LP1 resolves. Neither
rule uses the target answer to break a tie. These are interpretation conventions,
not measurements of model intent. Sensitivity re-scores the same responses.

The parser amendment preceded the retained model outputs. The alternative
qualification would fail some gates; it is descriptive, not an after-the-fact
replacement for the frozen primary parser. The byte-preserved
[amendment](supplied/PARSER_AMENDMENT.md) records that distinction.

## Scientific records and provenance

[MANIFEST.json](supplied/MANIFEST.json) binds 63 selected files to the completed
archive and records its outer digest. All 177 declared archive members and the
nested frozen package manifest were checked before import. Exact plans, scoring
maps, benchmark cases, entity registries, both versions of the code, scientific freezes,
development/qualification/final/repeat journals and expected analyses are retained.
Gzip transformations are lossless and have original and compressed hashes.

The model is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The recorded run used BF16,
single-example greedy decoding, at most 16 new tokens, SDPA and TF32 disabled.
Exact software versions and generation settings are in the freezes and journal
start records. Historical paths in those records are provenance, not dependencies.

Cloud controllers, allocation logs, billing, synthetic oracle runs and the much
larger upstream preparation pool are excluded. The exact fixed study inputs are
included; reconstructing that broader preparation pool is not this command's
scope. The builder and inference worker are archival source, not a tested fresh
inference interface. Hash checks do not attest the original run's clock time.
See [data and model notices](../../DATA_NOTICE.md).
