# Fresh matched-history confirmation and validation utility

The generated freeze metadata has a documented local-cache-path privacy
derivative. Its original identity is retained in
[privacy_derivatives.json](../privacy_derivatives.json). All frozen score,
input-file and scientific-analysis hashes remain exact.

The strongest history-matching and logical-correctness controls were fixed before
new outcomes in this Gemma study. The same model scores also tested whether
spreading validation across histories predicts errors on withheld questions.

## Canonical result table

| Measurement | Result | 95% paired case-bootstrap interval |
|---|---:|---:|
| Fresh full-criterion confirmation | 22 coded-target/seed observations across 14/64 cases; 0.9549% of the full denominator | [0.5208%, 1.4323%] |
| Withheld task-error difference at equal 50% retention, B minus A | -1.7447 percentage points | [-3.5346, +0.6504] points |

The fresh confirmation is positive. The practical validation advantage remains
uncertain because the utility interval includes zero. V10 remains the original
primary study; this adds a separate fresh test for Gemma and one learned-update
family, rather than a pooled sample for every model.

The two current figures are [fresh confirmation](figures/PRIMARY_CONFIRMATION.svg)
and [validation utility](figures/VALIDATION_UTILITY.svg). Each has one SVG copy.
The [full repeated Ada/Bryn example](examples/CASE_0017.md) shows the inputs,
questions, controls and original/repeated responses behind a surviving failure.

## What stayed fixed and what was measured

The model is Gemma-2-9B-it at the revision in [the source binding](protocol/SOURCE_BINDINGS.json).
Both stored INV_PAIR_NLL activation-update seeds were loaded without fitting.
Every final case has eight starting histories, two answer formats and the same
34-query battery. Model weights stayed fixed.

The full criterion requires correct direct factual reports across histories,
correct native-current and already-current references, and correct answers to
the other compositions of the target relation pair and the other relation pairs.
A target then counts if valid answers disagree between histories with equal
changed-fact counts and equal tokenized prefix lengths. The
[protocol](protocol/PROTOCOL.md) and [analysis rules](ANALYSIS_RULES.md) were frozen
before the new outcomes.

| Stored seed | Full denominator | Eligible targets | Qualifying observations |
|---|---:|---:|---:|
| 0 | 1,152 | 509 | 12 |
| 1 | 1,152 | 146 | 10 |

Ineligible targets remain in the denominator. Seed rates are averaged within
each case before averaging the 64 cases. The bootstrap resamples cases together
across seeds and formats. Seeds, formats and targets are repeated measurements,
not independent cases. All attrition, weaker gates, seed/format results and
strong-probability sensitivities remain in [PRIMARY.json](expected/PRIMARY.json)
and [the full control table](tables/PRIMARY_GATE_COUNTS.csv).

Rule A checks six different consequence questions at one history. Rule B checks
three consequence questions at each of two histories. Both pay the same common
direct-check cost. Three other questions per format are withheld from both rules.
Decisions and rankings were locked before their outcomes were analyzed.

The named utility endpoint is literal No/Yes at 50% retention. The common pools
have 59 and 57 cases for the two seeds. Acceptance, rejection, accepted errors,
successful cases, native references and both directions of exclusive acceptance
are in [UTILITY.json](expected/UTILITY.json) and the reader tables:

- [Acceptance and errors](tables/UTILITY_ACCEPTANCE.csv).
- [All seeds, formats and 25/50/75% retention levels](tables/UTILITY_EQUAL_RETENTION.csv).
- [Per-seed contrasts and intervals](tables/UTILITY_SEED_CONTRASTS.csv).
- [Cases accepted by only one rule, with withheld outcomes](tables/POLICY_DISAGREEMENT_CASES.csv).

The A/B-format sensitivity is also inconclusive. A negative point estimate
favors history diversity; uncertainty includes no advantage and a disadvantage.
The query budgets match, while shared computation prevents treating those budgets
as separately measured equal GPU costs.

## Post-hoc diagnostics

These checks were added after the outcomes. They leave the original endpoints,
denominators, eligibility rules and saved results unchanged.

Thirteen full-criterion observations have a withheld target, spanning seven cases.
Both validators accept the corresponding case/seed/format for all thirteen.
Nine of those observations also have a target error and valid disagreement at
the validators' selected history pair; the remaining four have their witness at
another matched pair. Correct answers to other questions can therefore miss a
target-specific error. This diagnosis explains a limitation of the tested checks;
the aggregate utility comparison remains the measure of practical benefit.

All 22 observations across 14 cases retain a witness pair with exactly equal
token-ID inventories, including multiplicity. This additional sensitivity
controls the token multiset, while token order and the association between names
and relations remain different. It is a post-hoc sensitivity, not a newly frozen
primary criterion. [All diagnostic rows](diagnostics/DIAGNOSTICS.json) retain the
exact targets, accepted decisions and matched pairs.

## CPU replay and equality

The minimal dependency is NumPy, pinned in [requirements.txt](requirements.txt).
For a fresh environment, run from the checkout root:

```sh
python -m pip install -r reproducibility/fresh_history_validation/requirements.txt
python -m pip install --no-deps -e .
```

Other study commands require the full repository replay environment. With the
repository installed, the fresh-study command runs from any working directory:

```sh
python -m repro.fresh_history --source /path/to/reproducibility/fresh_history_validation --output /fresh/output/directory
python -m pytest tests/test_fresh_history.py tests/test_saved_replay_portability.py
```

The default source is this directory. The entry point verifies every imported
file, then reconstructs all primary rows, paired bootstraps, locked decisions,
rankings, utility summaries and repeats. Its child processes block model/provider
imports and socket access. Private operator tests are outside this scientific
CPU suite.

The same command regenerates the reader tables and both SVG figures. The result
receipt distinguishes byte-identical outputs from finite numerical roundoff.
SVG structure, labels and styles must match exactly; scalar plot coordinates
allow absolute difference at most 1e-12. Computed summary and table floats use
the same numerical bound. The original saved artifacts remain unchanged.

Identities, decisions, ranks and classifications must match exactly. Recomputed
finite floating-point summaries allow absolute error at most 1e-12, with zero
relative tolerance. The original repeat acceptance threshold was 1e-6 for log
probabilities with exact hard-answer and token bindings. The observed result was
stronger: all 2,312 logical records across the four repeated positive cases had
zero log-probability difference and no recorded discrepancy. A tolerance pass
alone would not establish that strict equality.

## Source and compact-copy boundary

[SOURCE.json](SOURCE.json) binds every imported artifact to the immutable source
archive, whose SHA256 is
`0f52f3057797c3d07c9e945c1f920834bab37364fce11f53ed13e404c29431c4`.
[MANIFEST.json](MANIFEST.json) binds the compact copy. Frozen analyzers and input
records retain their original bytes. [PORTABILITY.md](PORTABILITY.md) identifies
the small I/O adaptations and the preserved original replay.

All final and repeated score panels, exact question/token mappings, frozen inputs,
analysis and decisions needed for scientific replay are included. Historical
DONE receipts remain unchanged; the portable loader verifies their score member.
Duplicate raw batch/context-seal copies, private cloud records, credentials,
failed-runtime archives and model-loading dependencies are omitted. The compact
replay reconstructs saved measurements, not original model execution.
