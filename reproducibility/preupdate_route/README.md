# Deleting pre-update messages failed the low-harm screen

The fixed deletion procedure increased answer errors and failed four of five
admission requirements. The study stopped before running any of its 128 fresh
final cases. This rejects that procedure as a low-harm repair under its fixed
gate; other upstream influences remain unresolved.

## What was tested

The input was a fixed historical panel of 16 Gemma cases: 14 earlier positive
cases and two controls. Each case used two selected histories and both stored
learned-update seeds. During context construction, the intervention removed
attention-value messages from three addressed clauses to shared trailing
positions in layers 0 through 15. The learned update, model weights, later
questions and answer formats stayed fixed. Intervention hooks were removed
before answering.

The comparison used the original computation and a matched control that
scrambled the removed message vectors by a signed permutation. We measured
corrections of the preselected target and errors across all scheduled questions.
Admission required corrections across both seeds, an advantage over the control,
fewer downstream errors and limited harm to direct factual answers.

## Results

| Arm | Direct errors / 768 responses | Downstream joint errors / 1,152 responses | Newly introduced joint errors |
|---|---:|---:|---:|
| Original computation (`N`) | 0 | 75 | 0 |
| Deletion (`K`) | 38 | 129 | 67 |
| Matched scrambled control (`SCTRL`) | 0 | 72 | 5 |

Deletion corrected 13 prior joint errors while introducing 67 new ones. All
38 direct errors were new. These response counts include repeated questions,
histories and seeds within the same cases. The historical panel was selected
for earlier failures, so these rates describe that panel.

| Admission requirement | Observed | Result |
|---|---|---|
| Correct at least three distinct positive cases while preserving controls | Three | Pass |
| Correct at least one such case with each seed | None with seed 0; three with seed 1 | Fail |
| Correct at least two more cases than the scrambled control | Three versus two | Fail |
| Reduce mean joint error | 75 to 129 errors / 1,152 responses | Fail |
| Introduce direct errors on at most 2% of scheduled direct responses | 38/768, or 4.948% | Fail |

The saved [screen](supplied/replay/SCREEN.json),
[per-case error table](supplied/SCREEN_DAMAGE.csv) and
[correction decisions](supplied/SCREEN_CORRECTIONS.csv) retain the full accounting.
The [terminal receipt](supplied/replay/CPU_REPLAY.json) records zero final cases.
The [source review packet](supplied/external_review_packet_v1.json) records the
128 frozen, unrun final designs. Final-population intervals and final positive
repeat examples therefore do not exist for this branch.

The preserved figures show [corrections by seed](supplied/figures/SCREEN_CORRECTIONS.svg)
and [answer errors](supplied/figures/SCREEN_DAMAGE.svg).
[Reader captions](../../figures/CAPTION_DETAILS.md#pre-update-deletion-screen)
explain their historical-panel scope.

## Compact check

From an installed checkout, using only the Python standard library:

```sh
python -m repro.preupdate_route
python -m pytest tests/test_preupdate_route.py
```

Expected: `PASS`, with the counts and five gate decisions in
[EXPECTED.json](EXPECTED.json). The check verifies the original bytes of all
15 supplied files, reconstructs totals from the case/seed tables, and recomputes
the admission decision. This is a table-consistency check. The full replay
below reconstructs the decision from saved scores.

## Full saved-score replay with the external archive

The full archive stays outside normal Git, following this repository's compact
evidence convention. [SOURCE_ARCHIVE.json](SOURCE_ARCHIVE.json) binds its expected
filename, `PREUPDATE_ROUTE_RESULTS.zip`, size of 340,740,905 bytes,
and SHA256:

`0d968769fb62335991d365086ed50902bd2a47dfe5bb6881e87652d36d8047a8`

Verify the archive before extraction. Replace the example paths below and choose
new extraction and output directories:

```sh
python -m repro.preupdate_route --archive /path/to/PREUPDATE_ROUTE_RESULTS.zip
python -m zipfile -e /path/to/PREUPDATE_ROUTE_RESULTS.zip /fresh/path/preupdate-bundle
python -m pip install -r /fresh/path/preupdate-bundle/CPU_REQUIREMENTS.txt
python reproducibility/preupdate_route/tools/replay_bundle_portable_v1.py --bundle /fresh/path/preupdate-bundle --work /fresh/path/preupdate-replay
```

The full replay checks all 159 delivered members, the successful immutable
source capsule and all four terminal exports, including failed-attempt exports.
It reconstructs the failed screen from both seeds' saved scores and regenerates
the two SVGs. Expected receipt fields are `status: PASS`,
`result.status: NO_ACTIONABLE_DIRECT_PREUPDATE_ROUTE`, `result.final_cases: 0`,
and `figures_canonical_lf_identical: true`. The reconstructed screen SHA256 is
`bcb88f23f691e63f4e2e91297e82b22868ce8d279aed06b08fd183bdd805fd30`.

The external archive retains the raw scores, safety tensors, both capsules,
technical checks, failed attempts and runtime provenance. The compact copy
retains original manifests and selected small evidence byte-for-byte under
`supplied/`; its historical README describes the full archive, not this subset.
Historical source-relative links in those records refer to that archive.

## Separately versioned SVG portability fix

[replay_bundle_portable_v1.py](tools/replay_bundle_portable_v1.py) differs from
the [original helper](supplied/tools/replay_bundle.py) only in its documented
SVG comparison and receipt field. Regenerated SVG comparisons normalize CRLF to
LF. Changed plotted content still fails. Archive, source, tensor and manifest
integrity checks remain byte-exact; neither frozen figures nor historical
manifests are rewritten.
