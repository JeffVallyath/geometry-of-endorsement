# Scientific replay portability

The frozen primary/utility analyzer and isolated decision procedure are copied
verbatim as `source/analyze.py` and `source/decision.py`. The original CPU replay
is retained as `source/original_replay.py`. Its source hash is in `SOURCE.json`.

The supported `source/replay.py` changes only its I/O boundary:

- An explicit source argument replaces the original fixed capsule location.
- `source/panels.py` supplies the original score-index and repeat-comparison
  function bodies without importing a neural runner or private worker.
- The panel loader verifies the unchanged DONE receipt's score-file hash.
  Duplicate raw batches and context seals are not needed for the saved-score
  calculations and are outside the compact copy's declared inventory.
- The readout ledger is losslessly gzip-compressed. Its decompressed digest
  remains bound to the original member.
- `source/bindings.py` provides local JSON/hash helpers without private operator
  paths. The public entry point verifies the compact manifest before running it.

No criterion, score, bootstrap, partition, decision or numerical threshold is
changed. `source/diagnostics.py` is new post-hoc analysis and writes separate
outputs. It cannot replace the frozen primary or utility result.

The repository's supplied-source verifier also interprets recorded Windows
separators as relative path separators. It still rejects absolute paths, drives,
traversal, normalized duplicates and escapes. Historical manifests retain their
original source identities. Current compact manifests separately bind the
documented local-cache-path derivative; no old digest is assigned to changed
bytes. Scientific CPU tests do not import or collect private provider,
allocation, deadline-guard or worker-lifecycle tests.

## Coverage-table line endings

The presentation check for `tables/TABLE_COVERAGE.json` permits CRLF versus LF
serialization only. Every other byte, including values and spacing, must agree.
The verification receipt records a newline-only difference as non-byte-identical.
Source/archive/raw hashes, scientific float bounds and other JSON comparisons
are unchanged. An extracted, archive-derived coverage fixture guards both the
permitted newline change and rejected changed counts.
