# Technical provenance

The authoritative completed delivery is `UPDATE_METHOD_RESULTS_filled.zip`,
SHA-256 `42163f6c18385140ab10797e1bf54de69e9eb019c3744d507332862e2b787cad`.
`extract_completed.py` verifies the archive hash, a closed member inventory, and
all 76 members declared by its embedded manifest. Another 82 members contain
the final additions and are bound by the archive hash and the new import manifest.
The embedded manifest predates those additions. It is retained unchanged as
`supplied/completed/SOURCE_MANIFEST.json`.

`supplied/completed/MANIFEST.json` binds each imported member, its original bytes,
and its transformation. Raw JSONL and answer JSON are losslessly gzip-compressed.
No score or experimental outcome is changed. Development records are retained
for provenance but excluded from the primary reconstruction.

The saved primary data cover ORIGIN, the matched-origin panel. Planned MAIN and
PROGRAM panels are not in this delivery and are not asserted as completed.
The earlier `delivery_inventory.json` describes the three earlier deliveries,
not the present completed archive. Their missing-score reports are superseded
by the saved scores here; their original bytes remain unchanged.

Frozen evaluation cases and learned-factor dependencies are the exact existing
`fresh_inference/supplied/reproduction/STATE_SUFFICIENCY_CONFIRMATION/runtime_inputs/UPDATE_METHOD_INPUTS`
records, checked against their original `ARTIFACT_MANIFEST.json`. The replay
checks every question, label and score identity against those inputs.

Primary root aggregation and the paired bootstrap are independently implemented
in `src/repro/update_methods.py`. The tabulation functions from
`supplied/later_delivery/UPDATE_METHOD_RESULTS/code/analysis.py` are reused only
after independent strict coverage and semantic checks. All four tables and
the complete primary contrast JSON match the completed delivery. This verifies
that analysis against the saved data; the delivery does not identify the exact
executed code revision or provide an execution-environment snapshot. Earlier
implementation code is preserved as supplied, not relabeled as attested code.
No external worked-example outputs were supplied, so that reproduction is not
claimed. Timing and memory are retained telemetry, not new benchmarks.

The saved full-vocabulary argmax-validity flag is used as recorded. The two
candidate log probabilities independently determine prediction, ties, candidate
mass and correctness conditional on that flag; they cannot independently certify
the highest-scoring token outside the candidate set.
