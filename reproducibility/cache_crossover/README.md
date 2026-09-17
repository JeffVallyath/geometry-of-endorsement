# Changing answers by swapping stored activations

Exact study packages: `V10_CACHE_CROSSOVER_RESULTS` and the matched-case
extension `V10_ROBUST_PAIR_CROSSOVER_RESULTS`. Here a swap combines earlier-layer
activations from one starting history with later-layer activations from another.

The saved Gemma results show a clear later-state effect in five of six selected
examples under the main answer format. The alternate format gives one clear
effect, two mixed results and three cases with insufficient original separation.
These selected cases do not estimate prevalence.

[Figure 8](../../figures/fig08_cache_crossover.png) shows one example.
[The six-case table](six_cases.csv) gives both classifications for each selected
case. [The full replay result](results.json) contains all 24 classifications,
including the six matched controls under both formats, direct-answer checks,
checks that recombine a history with itself or copy its full stored state, and the example's unrounded probabilities and raw log scores.

## Recalculate from saved scores

From the repository root, after the normal package installation:

```powershell
python -m repro.cache_crossover --output reproduced/crossover-check
python -m repro figures --output reproduced/crossover-figures
python -m repro verify
python -m repro evidence
```

Choose fresh output directories. No command runs a model. The first command
checks input hashes, independently normalizes the two semantic No/Yes log scores,
derives correct answers from the final fact configurations and applies the fixed classification rules.
It opens the saved summaries only after reconstruction and compares all
classifications, probabilities, effect errors and direct-preservation checks.
The figure renderer uses this same replay, not hand-entered numbers.

`raw.jsonl.gz` is a lossless copy of the entire scientific journal, including
all 576 cells, 1,296 baseline responses, baseline comparison records and controls.
Its decompressed SHA256 is
`2baf17d50b24c674dfe5b920405cdeab25dfd7735eb874efc2de8ee662d82dcd`.
`case_plan.json` preserves the original query, semantic label and main/alternate
bindings. Main does not always mean the `d0` code. Scores are already in semantic
No/Yes order; label-token order must not be applied again. `terminal_worlds.json`
retains the terminal worlds for all twelve benchmark cases. `frozen_gates.py.txt` is the
original classification specification for inspection, not an imported dependency.

The four internal state codes are AA (original A), BB (original B), AB (earlier-layer activations from A
plus later-layer activations from B), and BA (earlier-layer activations from B plus later-layer activations from A). A clear later-state effect requires
original probability separation of at least 0.10, mean deviation from the later
original at most 0.05, mean deviation from the earlier original at least 0.10,
matching hard answers, valid answer mass and direct preservation. The earlier-state
criterion reverses those comparisons. Main-format direct preservation requires
correct answers, correct-label probability at least 0.95, answer mass at least
0.90 and change at most 0.02 relative to the original supplying the later layers.
The alternate format checks correctness where both original direct answers were
correct. Full formulas and all classifications remain visible in the code and JSON.

## Delivery and lineage

[Provenance](provenance.json) binds every input to its source member and SHA256.
The authoritative completed archive is `V10_CACHE_CROSSOVER_RESULTS.zip`,
420,599,025 bytes, SHA256
`fc6f8c3eed7bfca435fc1042453bfb2e44874a3179fda4e950ba95416d18b5e2`.
Browser duplicate suffixes such as `(2)` are not artifact identities.
The archive is not committed and has no public download URL.
[The projected delivery receipt](delivery_receipt.json) retains its status, hash,
size, member count and forward accounting. Its original file hash is in provenance;
the local host path and cost field are omitted from the public projection.

The extractor verified all 70 archive members and all 747 capsule members.
It also checked the saved runtime completion and termination receipts. These are
checks of local delivered bytes, not a live provider query or a new attestation
of the original neural computation. Private operational logs remain outside Git.

[The terminal record](terminal.json) preserves the technical recovery history:
332 forwards were reused and 1,660 were new, for 1,992 total. The earlier
engineering-stop terminal was preserved. The combined journal is byte-for-byte
the original journal followed by the continuation journal. Pair zero's summaries
were repaired deterministically; its original scores were not rerun or rewritten.
The plan's pre-execution status is historical; the terminal record reports
`SCIENTIFIC_EXECUTED`.

To verify and re-extract a separately obtained delivery:

```powershell
python reproducibility/cache_crossover/extract_sources.py --archive PATH_TO_ZIP --receipt PATH_TO_RECEIPT --output reproduced/crossover-inputs
```

The extractor requires the exact completed archive hash, rejects unsafe or
duplicate member paths, checks all source hashes and writes only to a fresh
directory. It omits the archive itself and provider logs. Scientific journal
bytes are retained without projection or rounding; only the terminal worlds
and delivery receipt are projected as described in provenance.

Saved-state hashes and eligibility comparisons are checked as recorded. Offline
replay cannot recreate cache tensors or independently attest their physical
identity. Direct correctness, crossover classifications and sham/full-copy
equality are recalculated from the retained scores. The original-state comparison
to older scores uses the journal's recorded differences; the two new baseline
passes are compared directly. The one-case extension below now combines the
strong behavioral controls and the causal result on the same example.

## One-case extension

One retrospectively selected Gemma history pair now joins the strong behavioral
controls and the causal crossover result. Both histories required two fact changes
and had 517-token prefixes. Under both tested answer formats, the downstream
answer followed the history supplying the later activations. All eight strict
direct-readout checks passed. The [new figure](../../figures/fig09_cache_crossover_extension.png)
shows this case; Figure 8 and the original six-case counts remain unchanged.
The two formats are tests of one example, not independent cases or a prevalence
estimate. Full mechanism identification remains unresolved. The mechanism branch
is closed for the current paper; the planned one-case follow-up is completed.

[Extension tables and checks](../../artifacts/cache_crossover/extension_outcome.json)
retain the supplied outputs without changing values. The
[independent replay](extension/replay.json) checks both classifications, all eight
strict direct checks, 108 exact baseline replays and 24 exact sham/full-copy
comparisons. It derives correct answers from the existing terminal-world input
and independently normalizes the saved semantic No/Yes scores. The coded direct
question about Ada reverses the label mapping used by the other coded questions;
the replay preserves that mapping. Both formats use the same strict thresholds:
correct and valid direct answers, correct-label probability at least 0.95,
candidate mass at least 0.90, and change at most 0.02 from the original state
supplying the later layers. The earlier panel's alternate-format rule is unchanged.

```powershell
python scripts/export_cache_crossover.py --output reproduced/extension-report
python scripts/plot_cache_crossover.py --output reproduced/extension-figure
```

These portable adaptations of the supplied reporting scripts use the current
repo's verified inputs, without a separate historical analysis checkout. The
normal `python -m repro figures` command also renders the extension figure.
All output directories must be fresh. No command loads a model or runs inference.

[Extension provenance](extension/provenance.json) records the overlay and both
source archive hashes, input/member bindings, verification results and merge
decisions. The extension source archive is `V10_ROBUST_PAIR_CROSSOVER_RESULTS.zip`,
SHA256 `307a4f2344e81564300671d616af5cec21ac5bffe28b9553aedd5c4b253ef16a`.
All 73 payload members and its result manifest were checked. The archive records
166 neural forwards for the completed experiment. The supplied integration pass
and this merge performed zero neural forwards. The saved runtime receipts and
capsule hash were verified; the original runtime payload is not re-executed or
newly attested here.

The extension journal is retained losslessly as `extension/raw.jsonl.gz`.
The design and older score inputs already exist in the independent verification package for the prospective study
with identical hashes, so they are reused rather than duplicated. Original-panel
summaries are likewise reused. The [original-panel appendix](../../docs/CACHE_CROSSOVER_APPENDIX.md)
keeps all 24 outcomes separate from the extension. The supplied delivery-check
file records the earlier overlay build, not the current repository's test counts.
